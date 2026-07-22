from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from satnet.experiments.stage_a_execution._synthetic_harness import execute_synthetic_generation
from satnet.experiments.stage_a_execution.artifact_contract import (
    PRODUCTION_ARTIFACT_CONTRACT,
    SimulationAdapterResult,
)
from satnet.experiments.stage_a_execution.authorization import validate_authorization
from satnet.experiments.stage_a_execution.contract import FROZEN_CONTRACT_HASH
from satnet.experiments.stage_a_execution.generate import _make_production_science_validator, execute_generation
from satnet.experiments.stage_a_execution.ledger import build_ledger, transition, write_ledger
from satnet.experiments.stage_a_execution.locking import CAMPAIGN_FIELDS, RUN_FIELDS, lock_payload, run_identity
from satnet.experiments.stage_a_execution.plan import build_plan, validate_plan_contract_binding
from satnet.experiments.stage_a_execution.replay import execute_replay

from .conftest import (
    ARTIFACT_CONTRACT,
    EXECUTABLE_INVENTORY,
    STABLE_EXECUTABLE_COMMIT,
    TOOLING_PROPOSAL,
    make_authorization,
)

ROOT = Path(__file__).parents[3]


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    return tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))


def _plan(contract, roots, authorization):
    return build_plan(
        contract, partition="development", operation=authorization.operation,
        stable_executable_commit=STABLE_EXECUTABLE_COMMIT,
        executable_inventory_hash=EXECUTABLE_INVENTORY,
        tooling_proposal_hash=TOOLING_PROPOSAL,
        artifact_contract_hash=ARTIFACT_CONTRACT,
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
        authorization=authorization,
    )


def test_public_production_execution_api_denies_adapter_and_validator_injection() -> None:
    for entry_point in (execute_generation, execute_replay):
        parameters = inspect.signature(entry_point).parameters
        assert "adapter" not in parameters
        assert "science_validator" not in parameters
    assert "execute_synthetic_generation" not in __import__(
        "satnet.experiments.stage_a_execution", fromlist=["__all__"]
    ).__all__


def test_successful_ledger_requires_independent_science_completion(synthetic_contract, tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    authorization = make_authorization(
        synthetic_contract, operation="GENERATE", partition="development", run_ids=[1, 2],
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
    )
    plan = _plan(synthetic_contract, roots, authorization)
    ledger_path = tmp_path / "execution_ledger.json"
    write_ledger(ledger_path, build_ledger(plan, authorization.sha256), overwrite=False)
    transition(ledger_path, global_run_id=1, new_state="STARTING")
    transition(ledger_path, global_run_id=1, new_state="RUNNING")
    with pytest.raises(ValueError, match="science-completion"):
        transition(
            ledger_path, global_run_id=1, new_state="SUCCEEDED",
            artifacts=[{"relative_path": "artifact", "byte_length": 1, "sha256": "a" * 64}],
            artifact_inventory_hash="b" * 64,
            adapter_result={"validation_status": "PASSED", "simulation_return_status": "SUCCEEDED"},
        )


def test_synthetic_harness_rejects_frozen_contract_before_execution(synthetic_contract, tmp_path: Path) -> None:
    frozen = replace(synthetic_contract, contract_identity=FROZEN_CONTRACT_HASH)
    roots = _roots(tmp_path)
    authorization = make_authorization(
        frozen, operation="GENERATE", partition="development", run_ids=[1, 2],
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
    )
    plan = _plan(frozen, roots, authorization)
    with pytest.raises(PermissionError, match="Synthetic execution"):
        execute_synthetic_generation(
            repo_root=ROOT, contract=frozen, plan=plan, authorization_hash=authorization.sha256,
            preflight=None, campaign_root=roots[0],
        )
    assert not any(root.exists() for root in roots)


def test_authoritative_science_completion_requires_exact_satellite_and_g1_g5_stages(
    synthetic_contract, monkeypatch,
) -> None:
    stages = ("satellite", "g1", "g2", "g3", "g4", "g5", "target", "inventory", "result")
    monkeypatch.setattr(
        "satnet.experiments.stage_a_execution.generate._build_production_mapping",
        lambda *_: SimpleNamespace(run_id=7),
    )
    monkeypatch.setattr(
        "satnet.experiments.stage_a_execution.generate.read_json_object",
        lambda *_: {"validated_pipeline_result_hash": "9" * 64},
    )
    authoritative = {
        "stages": [{"stage": stage, "state": "matched"} for stage in stages],
        "result": {"run_result_hash": "9" * 64},
    }
    monkeypatch.setattr(
        "satnet.experiments.stage_a_execution.generate.validate_run_authoritatively",
        lambda **_: authoritative,
    )
    validator = _make_production_science_validator(synthetic_contract, object())
    adapter_result = SimulationAdapterResult(
        run_key="run", design_id="design", global_run_id=1, realization_id="R00",
        design_construction_seed=1, ground_selection_seed=2, satellite_failure_seed=3,
        ground_failure_seed=4, artifact_contract=PRODUCTION_ARTIFACT_CONTRACT,
        artifact_manifest=(), artifact_inventory_hash="7" * 64,
        simulation_return_status="SUCCEEDED", validation_status="PASSED", result_identity="8" * 64,
    )
    result = validator({}, Path("unused"), adapter_result)
    assert result.validation_status == "PASSED"
    assert result.verified_stages == stages
    authoritative["stages"] = authoritative["stages"][:-4]
    with pytest.raises(ValueError, match="G1-G5 completion"):
        validator({}, Path("unused"), adapter_result)


@pytest.mark.parametrize(
    "field",
    ("design_construction_seed", "ground_selection_seed", "satellite_failure_seed", "ground_failure_seed"),
)
def test_all_four_seeds_are_checked_against_frozen_manifest(synthetic_contract, tmp_path: Path, field: str) -> None:
    roots = _roots(tmp_path)
    authorization = make_authorization(
        synthetic_contract, operation="GENERATE", partition="development", run_ids=[1, 2],
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
    )
    plan = _plan(synthetic_contract, roots, authorization)
    plan["runs"][0][field] += 1
    with pytest.raises(ValueError, match="seed identity"):
        validate_plan_contract_binding(plan, synthetic_contract)


def test_replay_and_acceptance_authorizations_require_exact_ledger_identities(synthetic_contract, tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    generation_identity = ("execution_ledger.json", 101, "a" * 64)
    replay_identity = ("replay_ledger.json", 202, "b" * 64)
    replay = make_authorization(
        synthetic_contract, operation="REPLAY", partition="development", run_ids=[1, 2],
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
        source_generation_ledger=generation_identity,
    )
    validate_authorization(
        replay, contract=synthetic_contract, stable_executable_commit=STABLE_EXECUTABLE_COMMIT,
        executable_inventory_hash=EXECUTABLE_INVENTORY, tooling_proposal_hash=TOOLING_PROPOSAL,
        artifact_contract_hash=ARTIFACT_CONTRACT, operation="REPLAY", partition="development",
        run_ids=[1, 2], generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
    )
    acceptance = make_authorization(
        synthetic_contract, operation="ACCEPT", partition="development", run_ids=[1, 2],
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
        source_generation_ledger=generation_identity, source_replay_ledger=replay_identity,
    )
    plan = _plan(synthetic_contract, roots, acceptance)
    assert plan["source_generation_ledger_sha256"] == generation_identity[2]
    assert plan["source_replay_ledger_sha256"] == replay_identity[2]
    missing = make_authorization(
        synthetic_contract, operation="REPLAY", partition="development", run_ids=[1, 2],
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
    )
    with pytest.raises(PermissionError, match="ledger path"):
        validate_authorization(
            missing, contract=synthetic_contract, stable_executable_commit=STABLE_EXECUTABLE_COMMIT,
            executable_inventory_hash=EXECUTABLE_INVENTORY, tooling_proposal_hash=TOOLING_PROPOSAL,
            artifact_contract_hash=ARTIFACT_CONTRACT, operation="REPLAY", partition="development",
            run_ids=[1, 2], generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
        )


def test_lock_payload_contains_complete_campaign_and_run_identity(synthetic_contract, tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    authorization = make_authorization(
        synthetic_contract, operation="GENERATE", partition="development", run_ids=[1, 2],
        generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2],
    )
    plan = _plan(synthetic_contract, roots, authorization)
    identity = run_identity(plan, authorization.sha256, plan["runs"][0])
    payload = lock_payload(identity, "run")
    assert set(CAMPAIGN_FIELDS).issubset(payload)
    assert set(RUN_FIELDS).issubset(payload)
    assert payload["host_identity"]
    assert payload["process_start_identity"]
    assert payload["creation_time"]
    assert len(payload["lock_nonce"]) == 32
    assert len(payload["lock_hash"]) == 64


def test_byte_policy_is_narrow_and_has_no_repository_wide_python_rule() -> None:
    policy = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "*.py text eol=lf" not in policy
    assert "src/satnet/experiments/stage_a_execution/** -text" in policy
    assert "src/satnet/ground/** -text" in policy
    assert "src/satnet/network/** -text" in policy
