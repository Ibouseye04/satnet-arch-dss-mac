from __future__ import annotations

from pathlib import Path

import pytest

from satnet.experiments.stage_a_execution.acceptance import evaluate_acceptance, validate_acceptance_report
from satnet.experiments.stage_a_execution.common import atomic_write_json
from satnet.experiments.stage_a_execution.generate import execute_generation
from satnet.experiments.stage_a_execution.ledger import ExclusiveLock, read_ledger
from satnet.experiments.stage_a_execution.plan import build_plan
from satnet.experiments.stage_a_execution.replay import execute_replay

from .conftest import TOOLING_COMMIT, TOOLING_INVENTORY


def _plan(contract, operation: str, generation: Path, replay: Path, acceptance: Path):
    return build_plan(contract, partition="development", operation=operation, tooling_commit=TOOLING_COMMIT, tooling_inventory_hash=TOOLING_INVENTORY, generation_root=generation, replay_root=replay, acceptance_root=acceptance)


def _adapter(plan_run, output_root: Path) -> None:
    atomic_write_json(output_root / "scientific.json", {
        "global_run_id": plan_run["global_run_id"],
        "run_record_hash": plan_run["run_record_hash"],
        "seed": plan_run["satellite_failure_seed"],
    })


def test_generation_ledger_atomic_success_and_resume_skip(synthetic_contract, tmp_path: Path) -> None:
    generation, replay, acceptance = (tmp_path / name for name in ("generation", "replay", "acceptance"))
    plan = _plan(synthetic_contract, "GENERATE", generation, replay, acceptance)
    ledger = execute_generation(contract=synthetic_contract, plan=plan, authorization_hash="a" * 64, campaign_root=generation, adapter=_adapter)
    assert [record["state"] for record in ledger["records"]] == ["SUCCEEDED", "SUCCEEDED"]
    assert all(record["artifacts"] for record in ledger["records"])
    resumed = execute_generation(contract=synthetic_contract, plan=plan, authorization_hash="a" * 64, campaign_root=generation, adapter=lambda *_: pytest.fail("succeeded run must be skipped"), resume=True)
    assert resumed == ledger


def test_crash_is_failed_and_requires_explicit_retry(synthetic_contract, tmp_path: Path) -> None:
    generation, replay, acceptance = (tmp_path / name for name in ("generation", "replay", "acceptance"))
    plan = _plan(synthetic_contract, "GENERATE", generation, replay, acceptance)
    calls = 0
    def crashing(plan_run, output_root: Path) -> None:
        nonlocal calls
        calls += 1
        atomic_write_json(output_root / "partial.json", {"partial": True})
        raise RuntimeError("synthetic crash")
    with pytest.raises(RuntimeError, match="synthetic crash"):
        execute_generation(contract=synthetic_contract, plan=plan, authorization_hash="b" * 64, campaign_root=generation, adapter=crashing)
    ledger = read_ledger(generation / "execution_ledger.json")
    assert ledger["records"][0]["state"] == "FAILED"
    assert ledger["records"][0]["artifacts"] == []
    execute_generation(contract=synthetic_contract, plan=plan, authorization_hash="b" * 64, campaign_root=generation, adapter=_adapter, resume=True, retry_failed=False)
    assert read_ledger(generation / "execution_ledger.json")["records"][0]["state"] == "FAILED"
    with pytest.raises(RuntimeError):
        with ExclusiveLock(tmp_path / "campaign.lock", "one"):
            with ExclusiveLock(tmp_path / "campaign.lock", "two"):
                pass


def test_resume_rejects_changed_seed_plan_and_artifact(synthetic_contract, tmp_path: Path) -> None:
    generation, replay, acceptance = (tmp_path / name for name in ("generation", "replay", "acceptance"))
    plan = _plan(synthetic_contract, "GENERATE", generation, replay, acceptance)
    execute_generation(contract=synthetic_contract, plan=plan, authorization_hash="c" * 64, campaign_root=generation, adapter=_adapter)
    changed = {**plan, "runs": [dict(row) for row in plan["runs"]]}
    changed["runs"][0]["satellite_failure_seed"] += 1
    with pytest.raises(ValueError, match="Plan hash"):
        from satnet.experiments.stage_a_execution.plan import validate_plan
        validate_plan(changed)
    artifact = generation / plan["runs"][0]["expected_output_relative_path"] / "scientific.json"
    artifact.write_text("mutation", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory changed"):
        execute_generation(contract=synthetic_contract, plan=plan, authorization_hash="c" * 64, campaign_root=generation, adapter=_adapter, resume=True)


def test_replay_and_acceptance_synthetic_end_to_end(synthetic_contract, tmp_path: Path) -> None:
    generation, replay, acceptance = (tmp_path / name for name in ("generation", "replay", "acceptance"))
    generation_plan = _plan(synthetic_contract, "GENERATE", generation, replay, acceptance)
    replay_plan = _plan(synthetic_contract, "REPLAY", generation, replay, acceptance)
    acceptance_plan = _plan(synthetic_contract, "ACCEPT", generation, replay, acceptance)
    execute_generation(contract=synthetic_contract, plan=generation_plan, authorization_hash="d" * 64, campaign_root=generation, adapter=_adapter)
    replay_ledger = execute_replay(plan=replay_plan, authorization_hash="e" * 64, generation_root=generation, replay_root=replay, adapter=_adapter)
    assert all(record["state"] == "SUCCEEDED" for record in replay_ledger["records"])
    report = evaluate_acceptance(plan=acceptance_plan, authorization_hash="f" * 64, generation_root=generation, replay_root=replay, acceptance_root=acceptance)
    validate_acceptance_report(report)
    assert report["accepted_run_count"] == 2


def test_replay_mismatch_and_unexpected_artifact_fail_closed(synthetic_contract, tmp_path: Path) -> None:
    generation, replay, acceptance = (tmp_path / name for name in ("generation", "replay", "acceptance"))
    generation_plan = _plan(synthetic_contract, "GENERATE", generation, replay, acceptance)
    replay_plan = _plan(synthetic_contract, "REPLAY", generation, replay, acceptance)
    execute_generation(contract=synthetic_contract, plan=generation_plan, authorization_hash="1" * 64, campaign_root=generation, adapter=_adapter)
    def mismatch(plan_run, output_root: Path) -> None:
        atomic_write_json(output_root / "different.json", {"mismatch": True})
    with pytest.raises(ValueError, match="mismatch"):
        execute_replay(plan=replay_plan, authorization_hash="2" * 64, generation_root=generation, replay_root=replay, adapter=mismatch)
    unexpected = generation / generation_plan["runs"][0]["expected_output_relative_path"] / "unexpected.bin"
    unexpected.write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="inventory changed"):
        execute_generation(contract=synthetic_contract, plan=generation_plan, authorization_hash="1" * 64, campaign_root=generation, adapter=_adapter, resume=True)
