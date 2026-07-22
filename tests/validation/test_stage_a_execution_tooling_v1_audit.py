from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

pytestmark = pytest.mark.skip(reason="Immutable historical audit harness applies only to audit/stage-a-execution-tooling-v1")

from satnet.experiments.stage_a_execution.acceptance import evaluate_acceptance
from satnet.experiments.stage_a_execution.authorization import Authorization, authorization_digest, validate_authorization
from satnet.experiments.stage_a_execution.common import atomic_write_json
from satnet.experiments.stage_a_execution.contract import FrozenStageAContract
from satnet.experiments.stage_a_execution.generate import execute_generation
from satnet.experiments.stage_a_execution.ledger import build_ledger, read_ledger, transition, write_ledger
from satnet.experiments.stage_a_execution.paths import validate_output_roots, validate_relative_artifact_path
from satnet.experiments.stage_a_execution.plan import build_plan, validate_plan
from satnet.experiments.stage_a_execution.preflight import tooling_identity
from satnet.experiments.stage_a_execution.replay import execute_replay

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/validation/audit_stage_a_execution_tooling_v1.py"
SPEC = importlib.util.spec_from_file_location("stage_a_execution_tooling_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
TOOLING_COMMIT = "1" * 40
TOOLING_INVENTORY = "2" * 64


def synthetic_contract(tmp_path: Path) -> FrozenStageAContract:
    designs = (
        {"design_id": "SYN-D000", "design_index": 0, "region": "synthetic", "partition": "development", "sealed": False, "design_record_hash": "3" * 64},
        {"design_id": "SYN-D001", "design_index": 1, "region": "synthetic", "partition": "validation", "sealed": False, "design_record_hash": "4" * 64},
        {"design_id": "SYN-D002", "design_index": 2, "region": "synthetic", "partition": "sealed_holdout", "sealed": True, "design_record_hash": "5" * 64},
    )
    runs = (
        {"global_run_id": 1, "run_key": "SYN-D000-R00", "design_id": "SYN-D000", "design_index": 0, "realization_id": "R00", "realization_index": 0, "region": "synthetic", "partition": "development", "sealed": False, "design_record_hash": "3" * 64, "run_record_hash": "6" * 64},
        {"global_run_id": 2, "run_key": "SYN-D000-R01", "design_id": "SYN-D000", "design_index": 0, "realization_id": "R01", "realization_index": 1, "region": "synthetic", "partition": "development", "sealed": False, "design_record_hash": "3" * 64, "run_record_hash": "7" * 64},
        {"global_run_id": 3, "run_key": "SYN-D001-R00", "design_id": "SYN-D001", "design_index": 1, "realization_id": "R00", "realization_index": 0, "region": "synthetic", "partition": "validation", "sealed": False, "design_record_hash": "4" * 64, "run_record_hash": "8" * 64},
        {"global_run_id": 4, "run_key": "SYN-D002-R00", "design_id": "SYN-D002", "design_index": 2, "realization_id": "R00", "realization_index": 0, "region": "synthetic", "partition": "sealed_holdout", "sealed": True, "design_record_hash": "5" * 64, "run_record_hash": "9" * 64},
    )
    seeds = tuple(
        {
            "global_run_id": row["global_run_id"],
            "run_key": row["run_key"],
            "design_id": row["design_id"],
            "realization_id": row["realization_id"],
            "ground_selection_seed": row["global_run_id"] * 10 + 1,
            "satellite_failure_seed": row["global_run_id"] * 10 + 2,
            "ground_failure_seed": row["global_run_id"] * 10 + 3,
        }
        for row in runs
    )
    return FrozenStageAContract(
        repo_root=tmp_path / "repo",
        contract_root=tmp_path / "contract",
        specification={},
        declaration={},
        designs=designs,
        runs=runs,
        seeds=seeds,
        partitions={
            "development": {"global_run_ids": [1, 2], "design_count": 1, "run_count": 2, "sealed": False},
            "validation": {"global_run_ids": [3], "design_count": 1, "run_count": 1, "sealed": False},
            "sealed_holdout": {"global_run_ids": [4], "design_count": 1, "run_count": 1, "sealed": True},
        },
        output_roots={},
        contract_identity="a" * 64,
        frozen_tag="synthetic-contract-v1",
        frozen_commit="b" * 40,
    )


def plan(contract: FrozenStageAContract, operation: str, generation: Path, replay: Path, acceptance: Path) -> dict[str, Any]:
    return build_plan(
        contract,
        partition="development",
        operation=operation,
        tooling_commit=TOOLING_COMMIT,
        tooling_inventory_hash=TOOLING_INVENTORY,
        generation_root=generation,
        replay_root=replay,
        acceptance_root=acceptance,
    )


def adapter(plan_run: dict[str, Any], output_root: Path) -> None:
    atomic_write_json(output_root / "scientific.json", {"global_run_id": plan_run["global_run_id"], "run_record_hash": plan_run["run_record_hash"], "seed": plan_run["satellite_failure_seed"]})


def authorization(contract: FrozenStageAContract, operation: str, roots: tuple[Path, Path, Path], overrides: dict[str, Any] | None = None) -> Authorization:
    value: dict[str, Any] = {
        "schema_identifier": "satnet.stage_a.execution_authorization.v1",
        "authorization_version": "1",
        "authorization_id": f"SYNTHETIC-{operation}",
        "authorization_status": "AUTHORIZED",
        "authorized_contract_hash": contract.contract_hash,
        "authorized_contract_tag": contract.frozen_tag,
        "authorized_frozen_commit": contract.frozen_commit,
        "authorized_execution_tooling_commit": TOOLING_COMMIT,
        "authorized_execution_tooling_inventory_hash": TOOLING_INVENTORY,
        "authorized_partition": "development",
        "authorized_run_ids": [1, 2],
        "authorized_run_count": 2,
        "authorized_operation": operation,
        "authorized_generation_root": str(roots[0].resolve()),
        "authorized_replay_root": str(roots[1].resolve()),
        "authorized_acceptance_root": str(roots[2].resolve()),
        "authorization_date": "2099-01-01",
        "authorizing_decision_reference": "SYNTHETIC-TEST-ONLY",
        "independently_approved": True,
    }
    if overrides:
        value.update(overrides)
    value["authorization_sha256"] = authorization_digest(value)
    return Authorization(document=value, sha256=authorization_digest(value))


def test_exact_input_stable_diff_inventory_schema_contract_and_plan_identities() -> None:
    identity = audit.validate_input_identity(ROOT)
    stable, executable_diff = audit.validate_stable_executable(ROOT)
    inventory = audit.validate_inventory(ROOT)
    schemas = audit.validate_schemas(ROOT)
    contract, development = audit.validate_contract_and_plan(ROOT)
    assert identity["audit_head_at_start"] == audit.TOOLING_HEAD
    assert stable["stable_executable_commit"] == audit.STABLE_EXECUTABLE_COMMIT
    assert stable["execution_source_count"] == 15
    assert stable["allowlist_complete"] is True
    assert stable["current_byte_verification_present"] is False
    assert executable_diff["execution_path_changes"] == []
    assert inventory["inventory_sha256"] == audit.TOOLING_INVENTORY_HASH
    assert inventory["artifact_count_excluding_inventory"] == 28
    assert inventory["git_blob_mismatches"] == [".gitattributes", "tests/experiments/test_final_dataset_isolation.py"]
    assert len(inventory["windows_checkout_byte_mismatches"]) == 19
    assert inventory["windows_checkout_byte_identity"] is False
    assert schemas["hashes"] == audit.SCHEMA_HASHES
    assert contract["design_count"] == 30
    assert contract["run_count"] == 150
    assert contract["seed_count"] == 150
    assert development["development_design_count"] == 20
    assert development["development_run_count"] == 100
    assert development["first_run"] == "SA-D000-R00"
    assert development["last_run"] == "SA-D027-R04"
    assert development["ordered_global_run_ids_sha256"] == audit.ORDERED_RUN_IDS_HASH
    assert development["plan_hash"] == audit.DEVELOPMENT_PLAN_HASH


def test_proposal_reproduces_canonically_and_remains_unauthorized() -> None:
    proposal = audit.validate_proposal(ROOT)
    assert proposal["status"] == "TOOLING_PROPOSAL"
    assert proposal["execution_authorized"] is False
    assert proposal["simulation_authorized"] is False
    assert proposal["production_authorized"] is False
    assert proposal["exact_byte_hashes"] == audit.PROPOSAL_HASHES
    assert proposal["independent_generator_present"] is False
    assert proposal["byte_for_byte_regeneration"] == "UNAVAILABLE"


def test_frozen_contract_and_protected_science_preserved() -> None:
    frozen, protected = audit.validate_preservation(ROOT)
    assert frozen["hashes"]["inventory"] == audit.FROZEN_CONTRACT_HASH
    assert frozen["hashes"]["seed_manifest"] == audit.SEED_MANIFEST_HASH
    assert protected["changed_paths"] == []


def test_reserved_roots_absent_and_no_real_execution_evidence() -> None:
    roots, nonexecution = audit.validate_nonexecution(ROOT)
    assert roots["reserved_root_count"] == 0
    assert nonexecution["active_real_authorization_artifacts"] == []
    assert nonexecution["stage_a_simulations_performed"] == 0


def test_static_audit_records_all_binding_control_gaps() -> None:
    controls = audit.validate_static_controls(ROOT)
    assert controls["authorization"]["pass"] is True
    assert controls["partition"]["sealed_holdout_execution_option"] is False
    assert controls["holdout"]["pass"] is True
    for name in ("roots", "preflight", "success", "locking", "replay", "acceptance", "cli", "security"):
        assert controls[name]["pass"] is False
    assert {finding["finding_id"] for finding in audit.BINDING_FINDINGS} == {f"BINDING-{index:03d}" for index in range(1, 10)}


@pytest.mark.parametrize("status", ["authorized", "Authorized", "TRUE", "1", "PENDING", ""])
def test_authorization_requires_exact_status(status: str, tmp_path: Path) -> None:
    contract = synthetic_contract(tmp_path)
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    value = authorization(contract, "GENERATE", roots, {"authorization_status": status})
    with pytest.raises(PermissionError, match="AUTHORIZED"):
        validate_authorization(value, contract=contract, tooling_commit=TOOLING_COMMIT, tooling_inventory_hash=TOOLING_INVENTORY, operation="GENERATE", partition="development", run_ids=[1, 2], generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])


@pytest.mark.parametrize("operation", ["REPLAY", "ACCEPT", "PLAN", "generate", "UNKNOWN", ""])
def test_generate_authorization_cannot_cross_operation(operation: str, tmp_path: Path) -> None:
    contract = synthetic_contract(tmp_path)
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    value = authorization(contract, "GENERATE", roots)
    with pytest.raises(PermissionError, match="operation"):
        validate_authorization(value, contract=contract, tooling_commit=TOOLING_COMMIT, tooling_inventory_hash=TOOLING_INVENTORY, operation=operation, partition="development", run_ids=[1, 2], generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])


@pytest.mark.parametrize("partition,run_ids", [("validation", [3]), ("sealed_holdout", [4]), ("development", [1]), ("development", [1, 4]), ("development", [1, 1])])
def test_authorization_rejects_partition_and_run_set_changes(partition: str, run_ids: list[int], tmp_path: Path) -> None:
    contract = synthetic_contract(tmp_path)
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    value = authorization(contract, "GENERATE", roots)
    with pytest.raises((PermissionError, ValueError)):
        validate_authorization(value, contract=contract, tooling_commit=TOOLING_COMMIT, tooling_inventory_hash=TOOLING_INVENTORY, operation="GENERATE", partition=partition, run_ids=run_ids, generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])


@pytest.mark.parametrize("value", ["../escape", "a/../escape", "/absolute", "a\\b", ""])
def test_relative_artifact_path_security(value: str) -> None:
    with pytest.raises(ValueError):
        validate_relative_artifact_path(value)


def test_root_overlap_existing_and_repository_paths_rejected(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        validate_output_roots(repo_root=ROOT, generation_root=existing, replay_root=tmp_path / "replay", acceptance_root=tmp_path / "acceptance", require_absent=True)
    with pytest.raises(ValueError, match="overlap"):
        validate_output_roots(repo_root=ROOT, generation_root=tmp_path / "x", replay_root=tmp_path / "x/replay", acceptance_root=tmp_path / "acceptance", require_absent=False)
    with pytest.raises(ValueError, match="protected"):
        validate_output_roots(repo_root=ROOT, generation_root=ROOT / "output", replay_root=tmp_path / "replay", acceptance_root=tmp_path / "acceptance", require_absent=False)


def test_plan_hash_tampering_and_illegal_ledger_transitions_rejected(tmp_path: Path) -> None:
    contract = synthetic_contract(tmp_path)
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    value = plan(contract, "GENERATE", *roots)
    changed = deepcopy(value)
    changed["runs"][0]["satellite_failure_seed"] += 1
    with pytest.raises(ValueError, match="Plan hash"):
        validate_plan(changed)
    ledger_path = tmp_path / "ledger.json"
    write_ledger(ledger_path, build_ledger(value, "a" * 64), overwrite=False)
    with pytest.raises(ValueError, match="Illegal"):
        transition(ledger_path, global_run_id=1, new_state="SUCCEEDED", artifacts=[{"relative_path": "x", "byte_length": 1, "sha256": "0" * 64}], artifact_inventory_hash="0" * 64)


def test_resume_rejects_mutated_success_artifact(tmp_path: Path) -> None:
    contract = synthetic_contract(tmp_path)
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    value = plan(contract, "GENERATE", *roots)
    execute_generation(contract=contract, plan=value, authorization_hash="a" * 64, campaign_root=roots[0], adapter=adapter)
    artifact = roots[0] / value["runs"][0]["expected_output_relative_path"] / "scientific.json"
    artifact.write_text("mutation", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory changed"):
        execute_generation(contract=contract, plan=value, authorization_hash="a" * 64, campaign_root=roots[0], adapter=adapter, resume=True)


def test_required_tooling_identity_rejects_uncommitted_executable_change(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "src/satnet/experiments/stage_a_execution/module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (repo / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
    inventory = repo / "inventory.json"
    inventory.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Synthetic Audit"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "audit@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "synthetic"], cwd=repo, check=True, capture_output=True)
    first = tooling_identity(repo, inventory)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = tooling_identity(repo, inventory)
    assert second != first


def test_drive_like_path_is_contained_but_not_explicitly_rejected(tmp_path: Path) -> None:
    relative = validate_relative_artifact_path("C:/absolute")
    assert not relative.is_absolute()
    assert (tmp_path / relative).is_relative_to(tmp_path)


def test_required_generation_rejects_root_inside_repository(tmp_path: Path) -> None:
    contract = synthetic_contract(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    roots = (repo / "generation", tmp_path / "replay", tmp_path / "acceptance")
    value = plan(contract, "GENERATE", *roots)
    with pytest.raises(ValueError, match="protected"):
        execute_generation(contract=contract, plan=value, authorization_hash="a" * 64, campaign_root=roots[0], adapter=adapter)


def test_required_generation_rejects_malformed_nonempty_output(tmp_path: Path) -> None:
    contract = synthetic_contract(tmp_path)
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    value = plan(contract, "GENERATE", *roots)
    def malformed(_: dict[str, Any], output_root: Path) -> None:
        (output_root / "junk.bin").write_bytes(b"junk")
    with pytest.raises(ValueError, match="metadata|artifact|output"):
        execute_generation(contract=contract, plan=value, authorization_hash="a" * 64, campaign_root=roots[0], adapter=malformed)


def test_required_replay_rejects_generation_tooling_identity_mismatch(tmp_path: Path) -> None:
    contract = synthetic_contract(tmp_path)
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    generation_plan = plan(contract, "GENERATE", *roots)
    replay_plan = plan(contract, "REPLAY", *roots)
    execute_generation(contract=contract, plan=generation_plan, authorization_hash="a" * 64, campaign_root=roots[0], adapter=adapter)
    ledger_path = roots[0] / "execution_ledger.json"
    ledger = read_ledger(ledger_path)
    ledger["tooling_commit"] = "f" * 40
    write_ledger(ledger_path, ledger, overwrite=True)
    with pytest.raises(ValueError, match="tooling|identity"):
        execute_replay(plan=replay_plan, authorization_hash="b" * 64, generation_root=roots[0], replay_root=roots[1], adapter=adapter)


def test_required_acceptance_rejects_generation_seed_mismatch(tmp_path: Path) -> None:
    contract = synthetic_contract(tmp_path)
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    generation_plan = plan(contract, "GENERATE", *roots)
    replay_plan = plan(contract, "REPLAY", *roots)
    acceptance_plan = plan(contract, "ACCEPT", *roots)
    execute_generation(contract=contract, plan=generation_plan, authorization_hash="a" * 64, campaign_root=roots[0], adapter=adapter)
    execute_replay(plan=replay_plan, authorization_hash="b" * 64, generation_root=roots[0], replay_root=roots[1], adapter=adapter)
    ledger_path = roots[0] / "execution_ledger.json"
    ledger = read_ledger(ledger_path)
    ledger["records"][0]["satellite_failure_seed"] += 1
    write_ledger(ledger_path, ledger, overwrite=True)
    with pytest.raises(ValueError, match="seed|identity"):
        evaluate_acceptance(plan=acceptance_plan, authorization_hash="c" * 64, generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])


def test_reproducible_binding_tests_are_present() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert source.count("\n@pytest.mark.xfail(strict=True") == 5
    assert all(f"BINDING-{index:03d}" in source for index in (1, 2, 3, 4, 5))
