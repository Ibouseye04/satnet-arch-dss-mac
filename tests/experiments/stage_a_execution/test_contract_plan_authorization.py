from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from satnet.experiments.stage_a_execution.authorization import validate_authorization
import satnet.experiments.stage_a_execution.contract as contract_module
from satnet.experiments.stage_a_execution.contract import (
    FROZEN_CONTRACT_HASH, FROZEN_DECLARATION_HASH, FROZEN_README_HASH,
    FROZEN_SPECIFICATION_HASH, load_frozen_contract, verify_tag,
)
from satnet.experiments.stage_a_execution.generate import build_scientific_arguments
from satnet.experiments.stage_a_execution.plan import build_plan, validate_plan

from .conftest import ARTIFACT_CONTRACT, EXECUTABLE_INVENTORY, STABLE_EXECUTABLE_COMMIT, TOOLING_PROPOSAL, make_authorization

ROOT = Path(__file__).parents[3]
CONTRACT_ROOT = ROOT / "artifacts/stage_a_discovery_contract_v1"


def test_contract_loader_verifies_exact_frozen_bundle_and_partitions() -> None:
    contract = load_frozen_contract(ROOT)
    assert contract.contract_hash == FROZEN_CONTRACT_HASH
    assert len(contract.designs) == 30
    assert len(contract.runs) == 150
    assert len(contract.seeds) == 150
    assert {name: (value["design_count"], value["run_count"]) for name, value in contract.partitions.items()} == {
        "development": (20, 100), "validation": (5, 25), "sealed_holdout": (5, 25)
    }
    assert contract.declaration["simulation_authorized"] is False
    assert FROZEN_SPECIFICATION_HASH and FROZEN_DECLARATION_HASH and FROZEN_README_HASH


@pytest.mark.parametrize("name", [
    "stage_a_frozen_contract_inventory.json", "stage_a_frozen_contract_specification.json",
    "stage_a_contract_freeze_declaration.json", "STAGE_A_CONTRACT_FREEZE_README.txt",
])
def test_contract_loader_rejects_mutated_top_level_artifact(tmp_path: Path, name: str) -> None:
    copied = tmp_path / "contract"
    shutil.copytree(CONTRACT_ROOT, copied)
    copied.joinpath(name).write_bytes(copied.joinpath(name).read_bytes() + b"mutation")
    with pytest.raises(ValueError):
        load_frozen_contract(ROOT, copied)


def test_tag_verification_rejects_wrong_target(monkeypatch) -> None:
    monkeypatch.setattr(contract_module, "_git", lambda *args: "0" * 40)
    with pytest.raises(ValueError, match="tag target"):
        verify_tag(ROOT)


def test_generation_adapter_arguments_are_exactly_frozen_and_do_not_execute() -> None:
    contract = load_frozen_contract(ROOT)
    run = next(row for row in contract.runs if row["partition"] == "development")
    design = next(row for row in contract.designs if row["design_id"] == run["design_id"])
    seed = next(row for row in contract.seeds if row["run_key"] == run["run_key"])
    arguments = build_scientific_arguments(contract, run["global_run_id"])
    assert arguments["global_run_id"] == run["global_run_id"]
    assert arguments["run_record_hash"] == run["run_record_hash"]
    assert arguments["design_record_hash"] == design["design_record_hash"]
    assert arguments["ground_selection_seed"] == seed["ground_selection_seed"]
    assert arguments["satellite_failure_seed"] == seed["satellite_failure_seed"]
    assert arguments["ground_failure_seed"] == seed["ground_failure_seed"]
    assert arguments["design"]["duration_minutes"] == design["duration_minutes"]
    assert arguments["design"]["step_seconds"] == design["step_seconds"]
    assert arguments["design"]["satellite_failure_model"] == design["satellite_failure_model"]


def test_development_plan_is_exact_ordered_and_deterministic() -> None:
    contract = load_frozen_contract(ROOT)
    roots = [Path("C:/synthetic/generation"), Path("C:/synthetic/replay"), Path("C:/synthetic/acceptance")]
    first = build_plan(contract, partition="development", operation="PLAN", stable_executable_commit=STABLE_EXECUTABLE_COMMIT, executable_inventory_hash=EXECUTABLE_INVENTORY, tooling_proposal_hash=TOOLING_PROPOSAL, artifact_contract_hash=ARTIFACT_CONTRACT, generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])
    second = build_plan(contract, partition="development", operation="PLAN", stable_executable_commit=STABLE_EXECUTABLE_COMMIT, executable_inventory_hash=EXECUTABLE_INVENTORY, tooling_proposal_hash=TOOLING_PROPOSAL, artifact_contract_hash=ARTIFACT_CONTRACT, generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])
    assert first == second
    validate_plan(first)
    assert first["run_count"] == 100
    assert first["design_count"] == 20
    assert [row["global_run_id"] for row in first["runs"]] == sorted(contract.partitions["development"]["global_run_ids"])
    assert all(row["partition"] == "development" and row["sealed"] is False for row in first["runs"])
    assert first["runs"][0]["run_key"] != first["runs"][-1]["run_key"]


def test_plan_hash_binds_partition_roots_authorization_and_run_set(synthetic_contract, tmp_path: Path) -> None:
    roots = [tmp_path / name for name in ("generation", "replay", "acceptance")]
    base = build_plan(synthetic_contract, partition="development", operation="PLAN", stable_executable_commit=STABLE_EXECUTABLE_COMMIT, executable_inventory_hash=EXECUTABLE_INVENTORY, tooling_proposal_hash=TOOLING_PROPOSAL, artifact_contract_hash=ARTIFACT_CONTRACT, generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])
    validation = build_plan(synthetic_contract, partition="validation", operation="PLAN", stable_executable_commit=STABLE_EXECUTABLE_COMMIT, executable_inventory_hash=EXECUTABLE_INVENTORY, tooling_proposal_hash=TOOLING_PROPOSAL, artifact_contract_hash=ARTIFACT_CONTRACT, generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])
    changed_root = build_plan(synthetic_contract, partition="development", operation="PLAN", stable_executable_commit=STABLE_EXECUTABLE_COMMIT, executable_inventory_hash=EXECUTABLE_INVENTORY, tooling_proposal_hash=TOOLING_PROPOSAL, artifact_contract_hash=ARTIFACT_CONTRACT, generation_root=tmp_path / "different", replay_root=roots[1], acceptance_root=roots[2])
    authorization = make_authorization(synthetic_contract, operation="PLAN", partition="development", run_ids=[1, 2], generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])
    authorized = build_plan(synthetic_contract, partition="development", operation="PLAN", stable_executable_commit=STABLE_EXECUTABLE_COMMIT, executable_inventory_hash=EXECUTABLE_INVENTORY, tooling_proposal_hash=TOOLING_PROPOSAL, artifact_contract_hash=ARTIFACT_CONTRACT, generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2], authorization=authorization)
    assert len({base["plan_hash"], validation["plan_hash"], changed_root["plan_hash"], authorized["plan_hash"]}) == 4
    with pytest.raises(PermissionError):
        build_plan(synthetic_contract, partition="development", operation="PLAN", stable_executable_commit=STABLE_EXECUTABLE_COMMIT, executable_inventory_hash=EXECUTABLE_INVENTORY, tooling_proposal_hash=TOOLING_PROPOSAL, artifact_contract_hash=ARTIFACT_CONTRACT, generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2], selected_run_ids=[1])
    with pytest.raises(PermissionError, match="sealed"):
        build_plan(synthetic_contract, partition="sealed_holdout", operation="PLAN", stable_executable_commit=STABLE_EXECUTABLE_COMMIT, executable_inventory_hash=EXECUTABLE_INVENTORY, tooling_proposal_hash=TOOLING_PROPOSAL, artifact_contract_hash=ARTIFACT_CONTRACT, generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])


@pytest.mark.parametrize(("override", "message"), [
    ({"authorization_status": "PENDING"}, "AUTHORIZED"),
    ({"authorized_contract_hash": "0" * 64}, "contract_hash"),
    ({"authorized_executable_inventory_hash": "0" * 64}, "executable_inventory"),
    ({"authorized_operation": "REPLAY"}, "operation"),
    ({"authorized_partition": "validation"}, "partition"),
    ({"authorized_run_count": 1}, "count"),
    ({"authorized_run_ids": [1, 4]}, "run set"),
    ({"authorized_generation_root": "C:/wrong"}, "generation_root"),
    ({"independently_approved": False}, "approved"),
])
def test_authorization_fail_closed(synthetic_contract, tmp_path: Path, override: dict[str, object], message: str) -> None:
    roots = [tmp_path / name for name in ("generation", "replay", "acceptance")]
    authorization = make_authorization(synthetic_contract, operation="GENERATE", partition="development", run_ids=[1, 2], generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2], overrides=override)
    with pytest.raises((PermissionError, ValueError), match=message):
        validate_authorization(authorization, contract=synthetic_contract, stable_executable_commit=STABLE_EXECUTABLE_COMMIT, executable_inventory_hash=EXECUTABLE_INVENTORY, tooling_proposal_hash=TOOLING_PROPOSAL, artifact_contract_hash=ARTIFACT_CONTRACT, operation="GENERATE", partition="development", run_ids=[1, 2], generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])


def test_authorization_operations_are_not_transitive(synthetic_contract, tmp_path: Path) -> None:
    roots = [tmp_path / name for name in ("generation", "replay", "acceptance")]
    authorization = make_authorization(synthetic_contract, operation="GENERATE", partition="development", run_ids=[1, 2], generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])
    validate_authorization(authorization, contract=synthetic_contract, stable_executable_commit=STABLE_EXECUTABLE_COMMIT, executable_inventory_hash=EXECUTABLE_INVENTORY, tooling_proposal_hash=TOOLING_PROPOSAL, artifact_contract_hash=ARTIFACT_CONTRACT, operation="GENERATE", partition="development", run_ids=[1, 2], generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])
    with pytest.raises(PermissionError, match="operation"):
        validate_authorization(authorization, contract=synthetic_contract, stable_executable_commit=STABLE_EXECUTABLE_COMMIT, executable_inventory_hash=EXECUTABLE_INVENTORY, tooling_proposal_hash=TOOLING_PROPOSAL, artifact_contract_hash=ARTIFACT_CONTRACT, operation="REPLAY", partition="development", run_ids=[1, 2], generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])
