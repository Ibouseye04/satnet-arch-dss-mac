from __future__ import annotations

from pathlib import Path

import pytest

from satnet.experiments.stage_a_execution._synthetic_harness import execute_synthetic_generation, execute_synthetic_replay
from satnet.experiments.stage_a_execution.acceptance import evaluate_acceptance, validate_acceptance_report
from satnet.experiments.stage_a_execution.artifact_contract import write_synthetic_artifacts
from satnet.experiments.stage_a_execution.common import sha256_file
from satnet.experiments.stage_a_execution.ledger import read_ledger
from satnet.experiments.stage_a_execution.locking import ExclusiveLock, campaign_identity
from satnet.experiments.stage_a_execution.plan import build_plan
import satnet.experiments.stage_a_execution.preflight as preflight_module
from satnet.experiments.stage_a_execution.preflight import run_preflight

from .conftest import (
    ARTIFACT_CONTRACT,
    EXECUTABLE_INVENTORY,
    STABLE_EXECUTABLE_COMMIT,
    TOOLING_PROPOSAL,
    make_authorization,
    source_provenance_from_ledger,
)

ROOT = Path(__file__).parents[3]


def _ledger_identity(path: Path) -> tuple[str, int, str]:
    return path.name, path.stat().st_size, sha256_file(path)


def _authorization(contract, operation: str, roots, generation_identity=None, replay_identity=None):
    return make_authorization(
        contract, operation=operation, partition="development", run_ids=[1, 2],
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
        source_generation_ledger=generation_identity, source_replay_ledger=replay_identity,
    )


def _plan(contract, operation: str, generation: Path, replay: Path, acceptance: Path, authorization):
    return build_plan(
        contract, partition="development", operation=operation,
        stable_executable_commit=STABLE_EXECUTABLE_COMMIT,
        executable_inventory_hash=EXECUTABLE_INVENTORY,
        tooling_proposal_hash=TOOLING_PROPOSAL,
        artifact_contract_hash=ARTIFACT_CONTRACT,
        generation_root=generation, replay_root=replay, acceptance_root=acceptance,
        authorization=authorization,
    )


def _adapter(plan_run, output_root: Path):
    return write_synthetic_artifacts(plan_run, output_root)


def _certificate(monkeypatch, contract, plan, roots, authorization, *, resume: bool = False):
    monkeypatch.setattr(preflight_module, "verify_executable_identity", lambda _: {
        "stable_executable_commit": STABLE_EXECUTABLE_COMMIT,
        "executable_inventory_sha256": EXECUTABLE_INVENTORY,
    })
    monkeypatch.setattr(preflight_module, "load_frozen_contract", lambda *_: contract)
    monkeypatch.setattr(preflight_module, "verify_frozen_production_evidence", lambda *_: {
        "combined": {"file_count": 9004, "byte_count": 1337549193, "verified_sha256_count": 9004},
    })
    monkeypatch.setattr(
        preflight_module,
        "load_source_generation_provenance",
        lambda *_args, **_kwargs: source_provenance_from_ledger(
            roots[0] / "execution_ledger.json"
        ),
    )
    certificate = run_preflight(
        repo_root=ROOT, contract=contract, plan=plan, authorization=authorization,
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
        resume=resume, minimum_free_bytes=0,
    )
    return certificate


def test_generation_ledger_atomic_success_and_resume_skip(synthetic_contract, tmp_path: Path, monkeypatch) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    authorization = _authorization(synthetic_contract, "GENERATE", roots)
    plan = _plan(synthetic_contract, "GENERATE", *roots, authorization)
    certificate = _certificate(monkeypatch, synthetic_contract, plan, roots, authorization)
    ledger = execute_synthetic_generation(
        repo_root=ROOT, contract=synthetic_contract, plan=plan,
        authorization_hash=authorization.sha256, preflight=certificate,
        campaign_root=roots[0], adapter=_adapter,
    )
    assert [record["state"] for record in ledger["records"]] == ["SUCCEEDED", "SUCCEEDED"]
    assert all(record["adapter_result"]["validation_status"] == "PASSED" for record in ledger["records"])
    assert all(record["science_completion"]["validation_status"] == "PASSED" for record in ledger["records"])
    certificate = _certificate(monkeypatch, synthetic_contract, plan, roots, authorization, resume=True)
    resumed = execute_synthetic_generation(
        repo_root=ROOT, contract=synthetic_contract, plan=plan,
        authorization_hash=authorization.sha256, preflight=certificate,
        campaign_root=roots[0], adapter=lambda *_: pytest.fail("succeeded run must be skipped"), resume=True,
    )
    assert resumed == ledger


def test_execution_requires_valid_preflight_before_writes(synthetic_contract, tmp_path: Path) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    authorization = _authorization(synthetic_contract, "GENERATE", roots)
    plan = _plan(synthetic_contract, "GENERATE", *roots, authorization)
    with pytest.raises(PermissionError, match="preflight"):
        execute_synthetic_generation(
            repo_root=ROOT, contract=synthetic_contract, plan=plan,
            authorization_hash=authorization.sha256, preflight=None,
            campaign_root=roots[0], adapter=_adapter,
        )
    assert not any(root.exists() for root in roots)
    assert not roots[0].with_name(roots[0].name + ".lock").exists()


def test_crash_is_failed_and_locking_is_exclusive(synthetic_contract, tmp_path: Path, monkeypatch) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    authorization = _authorization(synthetic_contract, "GENERATE", roots)
    plan = _plan(synthetic_contract, "GENERATE", *roots, authorization)
    certificate = _certificate(monkeypatch, synthetic_contract, plan, roots, authorization)
    def crashing(plan_run, output_root: Path):
        (output_root / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("synthetic crash")
    with pytest.raises(RuntimeError, match="synthetic crash"):
        execute_synthetic_generation(
            repo_root=ROOT, contract=synthetic_contract, plan=plan,
            authorization_hash=authorization.sha256, preflight=certificate,
            campaign_root=roots[0], adapter=crashing,
        )
    ledger = read_ledger(roots[0] / "execution_ledger.json")
    assert ledger["records"][0]["state"] == "FAILED"
    assert ledger["records"][0]["artifacts"] == []
    identity = campaign_identity(plan, authorization.sha256)
    with pytest.raises(RuntimeError, match="lock"):
        with ExclusiveLock(tmp_path / "campaign.lock", identity):
            with ExclusiveLock(tmp_path / "campaign.lock", identity):
                pass


def test_resume_rejects_changed_seed_plan_and_artifact(synthetic_contract, tmp_path: Path, monkeypatch) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    authorization = _authorization(synthetic_contract, "GENERATE", roots)
    plan = _plan(synthetic_contract, "GENERATE", *roots, authorization)
    certificate = _certificate(monkeypatch, synthetic_contract, plan, roots, authorization)
    execute_synthetic_generation(
        repo_root=ROOT, contract=synthetic_contract, plan=plan,
        authorization_hash=authorization.sha256, preflight=certificate,
        campaign_root=roots[0], adapter=_adapter,
    )
    changed = {**plan, "runs": [dict(row) for row in plan["runs"]]}
    changed["runs"][0]["satellite_failure_seed"] += 1
    with pytest.raises(ValueError, match="Plan hash"):
        from satnet.experiments.stage_a_execution.plan import validate_plan
        validate_plan(changed)
    artifact = roots[0] / plan["runs"][0]["expected_output_relative_path"] / "scientific.json"
    artifact.write_text("mutation", encoding="utf-8")
    certificate = _certificate(monkeypatch, synthetic_contract, plan, roots, authorization, resume=True)
    with pytest.raises(ValueError, match="inventory changed"):
        execute_synthetic_generation(
            repo_root=ROOT, contract=synthetic_contract, plan=plan,
            authorization_hash=authorization.sha256, preflight=certificate,
            campaign_root=roots[0], adapter=_adapter, resume=True,
        )


def test_replay_and_acceptance_synthetic_end_to_end(synthetic_contract, tmp_path: Path, monkeypatch) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    generation_authorization = _authorization(synthetic_contract, "GENERATE", roots)
    generation_plan = _plan(synthetic_contract, "GENERATE", *roots, generation_authorization)
    generation_certificate = _certificate(monkeypatch, synthetic_contract, generation_plan, roots, generation_authorization)
    execute_synthetic_generation(
        repo_root=ROOT, contract=synthetic_contract, plan=generation_plan,
        authorization_hash=generation_authorization.sha256, preflight=generation_certificate,
        campaign_root=roots[0], adapter=_adapter,
    )
    generation_identity = _ledger_identity(roots[0] / "execution_ledger.json")
    replay_authorization = _authorization(synthetic_contract, "REPLAY", roots, generation_identity)
    replay_plan = _plan(synthetic_contract, "REPLAY", *roots, replay_authorization)
    replay_certificate = _certificate(monkeypatch, synthetic_contract, replay_plan, roots, replay_authorization)
    replay_ledger = execute_synthetic_replay(
        repo_root=ROOT, contract=synthetic_contract, plan=replay_plan,
        authorization_hash=replay_authorization.sha256, preflight=replay_certificate,
        generation_root=roots[0], replay_root=roots[1], adapter=_adapter,
    )
    assert all(record["state"] == "SUCCEEDED" for record in replay_ledger["records"])
    replay_identity = _ledger_identity(roots[1] / "replay_ledger.json")
    acceptance_authorization = _authorization(
        synthetic_contract, "ACCEPT", roots, generation_identity, replay_identity,
    )
    acceptance_plan = _plan(synthetic_contract, "ACCEPT", *roots, acceptance_authorization)
    acceptance_certificate = _certificate(monkeypatch, synthetic_contract, acceptance_plan, roots, acceptance_authorization)
    report = evaluate_acceptance(
        repo_root=ROOT, plan=acceptance_plan, authorization_hash=acceptance_authorization.sha256,
        preflight=acceptance_certificate, generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
    )
    validate_acceptance_report(report)
    assert report["accepted_run_count"] == 2
    assert report["generation_ledger_sha256"] == generation_identity[2]
    assert report["replay_ledger_sha256"] == replay_identity[2]


def test_replay_mismatch_and_unexpected_artifact_fail_closed(synthetic_contract, tmp_path: Path, monkeypatch) -> None:
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    generation_authorization = _authorization(synthetic_contract, "GENERATE", roots)
    generation_plan = _plan(synthetic_contract, "GENERATE", *roots, generation_authorization)
    generation_certificate = _certificate(monkeypatch, synthetic_contract, generation_plan, roots, generation_authorization)
    execute_synthetic_generation(
        repo_root=ROOT, contract=synthetic_contract, plan=generation_plan,
        authorization_hash=generation_authorization.sha256, preflight=generation_certificate,
        campaign_root=roots[0], adapter=_adapter,
    )
    generation_identity = _ledger_identity(roots[0] / "execution_ledger.json")
    replay_authorization = _authorization(synthetic_contract, "REPLAY", roots, generation_identity)
    replay_plan = _plan(synthetic_contract, "REPLAY", *roots, replay_authorization)
    replay_certificate = _certificate(monkeypatch, synthetic_contract, replay_plan, roots, replay_authorization)
    def mismatch(plan_run, output_root: Path):
        result = write_synthetic_artifacts(plan_run, output_root)
        (output_root / "scientific.json").write_text("mutation", encoding="utf-8")
        return result
    with pytest.raises(ValueError, match="manifest|mismatch"):
        execute_synthetic_replay(
            repo_root=ROOT, contract=synthetic_contract, plan=replay_plan,
            authorization_hash=replay_authorization.sha256,
            preflight=replay_certificate, generation_root=roots[0], replay_root=roots[1], adapter=mismatch,
        )
    assert read_ledger(roots[1] / "replay_ledger.json")["records"][0]["state"] == "FAILED"
