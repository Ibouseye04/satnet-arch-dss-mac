from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.validation import reaudit_stage_a_execution_tooling_v1 as reaudit
from satnet.experiments.stage_a_execution.acceptance import evaluate_acceptance
from satnet.experiments.stage_a_execution.artifact_contract import make_adapter_result, validate_run_output, write_synthetic_artifacts
from satnet.experiments.stage_a_execution.authorization import Authorization, authorization_digest
from satnet.experiments.stage_a_execution.common import canonical_json_bytes, payload_hash
from satnet.experiments.stage_a_execution.contract import FrozenStageAContract
from satnet.experiments.stage_a_execution.evidence import validate_frozen_evidence_result
from satnet.experiments.stage_a_execution.generate import execute_generation
from satnet.experiments.stage_a_execution.locking import ExclusiveLock, lock_payload, recover_stale_lock
from satnet.experiments.stage_a_execution.paths import validate_output_roots
from satnet.experiments.stage_a_execution.plan import build_plan
import satnet.experiments.stage_a_execution.preflight as preflight_module
from satnet.experiments.stage_a_execution.preflight import run_preflight
from satnet.experiments.stage_a_execution.replay import execute_replay

ROOT = Path(__file__).parents[2]
WINDOWS_ROOT = Path(r"C:\Users\johns\satnet-stage-a-execution-reaudit-windows-20260722")
REPRODUCTIONS = (
    Path(r"C:\Users\johns\satnet-stage-a-execution-tooling-reaudit-evidence-20260722\proposal_reproduction_1"),
    Path(r"C:\Users\johns\satnet-stage-a-execution-tooling-reaudit-evidence-20260722\proposal_reproduction_2"),
)


def synthetic_contract(tmp_path: Path) -> FrozenStageAContract:
    designs = (
        {"design_id": "SYN-D000", "design_index": 0, "design_record_hash": "3" * 64},
        {"design_id": "SYN-D001", "design_index": 1, "design_record_hash": "4" * 64},
        {"design_id": "SYN-D002", "design_index": 2, "design_record_hash": "5" * 64},
    )
    runs = (
        {"global_run_id": 1, "run_key": "SYN-D000-R00", "design_id": "SYN-D000", "design_index": 0, "realization_id": "R00", "realization_index": 0, "region": "synthetic", "partition": "development", "sealed": False, "design_record_hash": "3" * 64, "run_record_hash": "6" * 64},
        {"global_run_id": 2, "run_key": "SYN-D000-R01", "design_id": "SYN-D000", "design_index": 0, "realization_id": "R01", "realization_index": 1, "region": "synthetic", "partition": "development", "sealed": False, "design_record_hash": "3" * 64, "run_record_hash": "7" * 64},
        {"global_run_id": 3, "run_key": "SYN-D001-R00", "design_id": "SYN-D001", "design_index": 1, "realization_id": "R00", "realization_index": 0, "region": "synthetic", "partition": "validation", "sealed": False, "design_record_hash": "4" * 64, "run_record_hash": "8" * 64},
        {"global_run_id": 4, "run_key": "SYN-D002-R00", "design_id": "SYN-D002", "design_index": 2, "realization_id": "R00", "realization_index": 0, "region": "synthetic", "partition": "sealed_holdout", "sealed": True, "design_record_hash": "5" * 64, "run_record_hash": "9" * 64},
    )
    seeds = tuple(
        {
            "global_run_id": run["global_run_id"],
            "run_key": run["run_key"],
            "design_id": run["design_id"],
            "realization_id": run["realization_id"],
            "design_construction_seed": run["global_run_id"] * 10,
            "ground_selection_seed": run["global_run_id"] * 10 + 1,
            "satellite_failure_seed": run["global_run_id"] * 10 + 2,
            "ground_failure_seed": run["global_run_id"] * 10 + 3,
        }
        for run in runs
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


def make_plan(contract: FrozenStageAContract, operation: str, roots: tuple[Path, Path, Path]) -> dict:
    return build_plan(
        contract,
        partition="development",
        operation=operation,
        stable_executable_commit=reaudit.STABLE_EXECUTABLE_COMMIT,
        executable_inventory_hash=reaudit.EXECUTABLE_INVENTORY_HASH,
        tooling_proposal_hash=reaudit.TOOLING_PROPOSAL_HASH,
        artifact_contract_hash=reaudit.ARTIFACT_CONTRACT_FILE_HASH,
        generation_root=roots[0],
        replay_root=roots[1],
        acceptance_root=roots[2],
    )


def make_authorization(contract: FrozenStageAContract, operation: str, roots: tuple[Path, Path, Path]) -> Authorization:
    value = {
        "schema_identifier": "satnet.stage_a.execution_authorization.v1",
        "authorization_version": "1",
        "authorization_id": f"SYNTHETIC-{operation}",
        "authorization_status": "AUTHORIZED",
        "authorized_contract_hash": contract.contract_hash,
        "authorized_contract_tag": contract.frozen_tag,
        "authorized_frozen_commit": contract.frozen_commit,
        "authorized_stable_executable_commit": reaudit.STABLE_EXECUTABLE_COMMIT,
        "authorized_executable_inventory_hash": reaudit.EXECUTABLE_INVENTORY_HASH,
        "authorized_tooling_proposal_hash": reaudit.TOOLING_PROPOSAL_HASH,
        "authorized_artifact_contract_hash": reaudit.ARTIFACT_CONTRACT_FILE_HASH,
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
    value["authorization_sha256"] = authorization_digest(value)
    return Authorization(document=value, sha256=authorization_digest(value))


def make_certificate(monkeypatch: pytest.MonkeyPatch, contract: FrozenStageAContract, plan: dict, roots: tuple[Path, Path, Path], operation: str):
    authorization = make_authorization(contract, operation, roots)
    monkeypatch.setattr(preflight_module, "verify_executable_identity", lambda _: {
        "stable_executable_commit": reaudit.STABLE_EXECUTABLE_COMMIT,
        "executable_inventory_sha256": reaudit.EXECUTABLE_INVENTORY_HASH,
    })
    monkeypatch.setattr(preflight_module, "load_frozen_contract", lambda *_: contract)
    monkeypatch.setattr(preflight_module, "verify_frozen_production_evidence", lambda *_: {
        "combined": {"file_count": 9004, "byte_count": 1337549193, "verified_sha256_count": 9004},
    })
    certificate = run_preflight(
        repo_root=ROOT,
        contract=contract,
        plan=plan,
        authorization=authorization,
        generation_root=roots[0],
        replay_root=roots[1],
        acceptance_root=roots[2],
        minimum_free_bytes=0,
    )
    return authorization, certificate


def test_exact_input_executable_and_post_stable_identities() -> None:
    identity = reaudit.verify_input_identity(ROOT)
    executable, inventory, post_stable = reaudit.verify_executable(ROOT)
    assert identity["remediation_head_audited"] == reaudit.REMEDIATION_HEAD
    assert executable["stable_executable_commit"] == reaudit.STABLE_EXECUTABLE_COMMIT
    assert inventory["inventory_sha256"] == reaudit.EXECUTABLE_INVENTORY_HASH
    assert inventory["artifact_count"] == 70
    assert inventory["byte_mismatches"] == []
    assert post_stable["changed_executable_paths"] == []


def test_corrected_proposal_reproduces_but_policy_is_not_narrow() -> None:
    generation, inventory = reaudit.verify_proposal(ROOT, REPRODUCTIONS)
    windows = reaudit.verify_windows_checkout(ROOT, WINDOWS_ROOT)
    assert generation["generated_artifact_count"] == 16
    assert generation["byte_mismatches"] == {}
    assert inventory["inventory_sha256"] == reaudit.TOOLING_PROPOSAL_HASH
    assert windows["byte_mismatches"] == []
    assert windows["byte_portability"] == "PASSED"
    assert windows["broad_repository_text_rules"] == ["*.py text eol=lf"]
    assert windows["narrow_policy_requirement"] == "FAILED"


def test_development_plan_and_holdout_are_preserved() -> None:
    development, authorization, holdout = reaudit.verify_development_plan(ROOT)
    assert development["development_design_count"] == 20
    assert development["development_run_count"] == 100
    assert development["first_development_run"] == "SA-D000-R00"
    assert development["last_development_run"] == "SA-D027-R04"
    assert development["ordered_run_id_sha256"] == reaudit.ORDERED_RUN_IDS_HASH
    assert development["partitions"] == {
        "development": {"design_count": 20, "run_count": 100, "sealed": False},
        "validation": {"design_count": 5, "run_count": 25, "sealed": False},
        "sealed_holdout": {"design_count": 5, "run_count": 25, "sealed": True},
    }
    assert authorization["execution_authorized"] is False
    assert holdout["development_validation_run_count"] == 0
    assert holdout["development_sealed_holdout_run_count"] == 0
    assert holdout["sealed_holdout_aggregate"]["identities"] == "REDACTED"


def test_no_real_authorization_reserved_root_or_execution_evidence() -> None:
    result = reaudit.verify_nonexecution(ROOT)
    assert result["real_authorization_artifacts"] == []
    assert result["reserved_root_count"] == 0
    assert result["generation_ledgers"] == []
    assert result["replay_ledgers"] == []
    assert result["acceptance_reports"] == []
    assert result["stage_a_simulations_performed"] == 0


def test_frozen_contract_and_protected_science_are_preserved() -> None:
    frozen, protected = reaudit.verify_preservation(ROOT)
    assert frozen["frozen_contract_hash"] == reaudit.FROZEN_CONTRACT_HASH
    assert frozen["hashes"]["seed_manifest"] == reaudit.SEED_MANIFEST_HASH
    assert protected["changed_paths"] == []


def test_all_public_execution_apis_require_preflight_before_writes(tmp_path: Path) -> None:
    contract = synthetic_contract(tmp_path)
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    adapter_calls = 0
    def adapter(*_):
        nonlocal adapter_calls
        adapter_calls += 1
        raise AssertionError
    calls = (
        lambda: execute_generation(repo_root=ROOT, contract=contract, plan=make_plan(contract, "GENERATE", roots), authorization_hash="f" * 64, preflight=None, campaign_root=roots[0], adapter=adapter),
        lambda: execute_generation(repo_root=ROOT, contract=contract, plan=make_plan(contract, "GENERATE", roots), authorization_hash="f" * 64, preflight=None, campaign_root=roots[0], adapter=adapter, resume=True),
        lambda: execute_replay(repo_root=ROOT, plan=make_plan(contract, "REPLAY", roots), authorization_hash="f" * 64, preflight=None, generation_root=roots[0], replay_root=roots[1], adapter=adapter),
        lambda: evaluate_acceptance(repo_root=ROOT, plan=make_plan(contract, "ACCEPT", roots), authorization_hash="f" * 64, preflight=None, generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2]),
    )
    for call in calls:
        with pytest.raises(PermissionError, match="preflight"):
            call()
    assert adapter_calls == 0
    assert not any(root.exists() for root in roots)
    assert not any(root.with_name(root.name + ".lock").exists() for root in roots)


def test_root_isolation_rejects_existing_overlap_repository_and_frozen_paths(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        validate_output_roots(repo_root=ROOT, generation_root=existing, replay_root=tmp_path / "replay", acceptance_root=tmp_path / "acceptance", require_absent=True)
    with pytest.raises(ValueError, match="overlap"):
        validate_output_roots(repo_root=ROOT, generation_root=tmp_path / "campaign", replay_root=tmp_path / "campaign/replay", acceptance_root=tmp_path / "acceptance", require_absent=False)
    with pytest.raises(ValueError, match="protected"):
        validate_output_roots(repo_root=ROOT, generation_root=ROOT / "forbidden", replay_root=tmp_path / "replay", acceptance_root=tmp_path / "acceptance", require_absent=False)
    with pytest.raises(ValueError, match="protected"):
        validate_output_roots(repo_root=ROOT, generation_root=Path(r"C:\Users\johns\satnet-final-production-20260720\forbidden"), replay_root=tmp_path / "replay", acceptance_root=tmp_path / "acceptance", require_absent=False)


def test_arbitrary_output_and_adapter_filesystem_mismatch_are_rejected(tmp_path: Path) -> None:
    contract = synthetic_contract(tmp_path)
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    plan = make_plan(contract, "GENERATE", roots)
    bound = dict(plan["runs"][0])
    bound.update({
        "plan_hash": plan["plan_hash"],
        "stable_executable_commit": plan["stable_executable_commit"],
        "executable_inventory_hash": plan["executable_inventory_hash"],
        "tooling_proposal_hash": plan["tooling_proposal_hash"],
        "artifact_contract_hash": plan["artifact_contract_hash"],
    })
    output = tmp_path / "output"
    output.mkdir()
    (output / "arbitrary.txt").write_text("arbitrary", encoding="utf-8")
    arbitrary = make_adapter_result(bound, output, "synthetic_stage_a_test_run_v1")
    with pytest.raises(ValueError, match="path set|artifact"):
        validate_run_output(bound, output, arbitrary)
    for path in output.iterdir():
        path.unlink()
    valid = write_synthetic_artifacts(bound, output)
    (output / "scientific.json").write_text("mutation", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        validate_run_output(bound, output, valid)


def test_complete_production_science_validation_can_be_bypassed_by_public_adapter_surface() -> None:
    controls = reaudit.control_outputs(ROOT)
    artifact = controls["reaudit_artifact_contract.json"]
    adapter = controls["reaudit_adapter_result.json"]
    assert artifact["arbitrary_output_rejected"] is True
    assert artifact["public_execution_accepts_injected_adapter"] is True
    assert artifact["production_authoritative_replay_called_by_orchestration_validator"] is False
    assert artifact["complete_scientific_validation_before_succeeded"] is False
    assert adapter["complete_production_science_independently_revalidated"] is False


def test_replay_does_not_bind_exact_source_generation_ledger_bytes() -> None:
    controls = reaudit.control_outputs(ROOT)["reaudit_replay_binding.json"]
    assert controls["generation_ledger_content_validated"] is True
    assert controls["source_generation_ledger_path_persisted"] is False
    assert controls["source_generation_ledger_byte_length_persisted"] is False
    assert controls["source_generation_ledger_sha256_persisted"] is False
    assert controls["exact_source_ledger_bytes_bound"] is False


def test_acceptance_allows_semantic_preserving_generation_ledger_byte_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = synthetic_contract(tmp_path)
    roots = tuple(tmp_path / name for name in ("generation", "replay", "acceptance"))
    generation_plan = make_plan(contract, "GENERATE", roots)
    generation_authorization, generation_certificate = make_certificate(monkeypatch, contract, generation_plan, roots, "GENERATE")
    execute_generation(repo_root=ROOT, contract=contract, plan=generation_plan, authorization_hash=generation_authorization.sha256, preflight=generation_certificate, campaign_root=roots[0], adapter=write_synthetic_artifacts)
    replay_plan = make_plan(contract, "REPLAY", roots)
    replay_authorization, replay_certificate = make_certificate(monkeypatch, contract, replay_plan, roots, "REPLAY")
    execute_replay(repo_root=ROOT, plan=replay_plan, authorization_hash=replay_authorization.sha256, preflight=replay_certificate, generation_root=roots[0], replay_root=roots[1], adapter=write_synthetic_artifacts)
    generation_ledger_path = roots[0] / "execution_ledger.json"
    before = generation_ledger_path.read_bytes()
    value = json.loads(before)
    generation_ledger_path.write_bytes(json.dumps(value, indent=4, sort_keys=False).encode("utf-8") + b"\n")
    assert generation_ledger_path.read_bytes() != before
    acceptance_plan = make_plan(contract, "ACCEPT", roots)
    acceptance_authorization, acceptance_certificate = make_certificate(monkeypatch, contract, acceptance_plan, roots, "ACCEPT")
    report = evaluate_acceptance(repo_root=ROOT, plan=acceptance_plan, authorization_hash=acceptance_authorization.sha256, preflight=acceptance_certificate, generation_root=roots[0], replay_root=roots[1], acceptance_root=roots[2])
    assert report["acceptance_state"] == "PASSED"
    assert report["accepted_run_count"] == 2


def test_complete_frozen_seed_set_is_not_bound() -> None:
    seed = reaudit.control_outputs(ROOT)["reaudit_seed_binding.json"]
    assert seed["frozen_seed_fields"] == ["design_construction_seed", "ground_failure_seed", "ground_selection_seed", "satellite_failure_seed"]
    assert seed["missing_seed_fields"] == ["design_construction_seed"]
    assert seed["complete_frozen_seed_set_bound"] is False


def test_frozen_evidence_result_rejects_identity_count_hash_and_read_only_mutations() -> None:
    good = {
        "production_tooling_sha": "9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab",
        "contract_specification_hash": "482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b",
        "generation_ledger_sha256": "a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1",
        "replay_ledger_sha256": "4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd",
        "freeze_archive_sha256": "375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc",
        "combined": {"file_count": 9004, "byte_count": 1337549193, "verified_sha256_count": 9004},
        "generation": {"all_files_read_only": True},
        "replay": {"all_files_read_only": True},
        "archive_read_only": True,
        "freeze_metadata_all_read_only": True,
        "verification_status": "passed",
    }
    assert validate_frozen_evidence_result(good) == good
    mutations = (
        ("generation_ledger_sha256", "0" * 64),
        ("replay_ledger_sha256", "0" * 64),
        ("freeze_archive_sha256", "0" * 64),
        ("archive_read_only", False),
        ("freeze_metadata_all_read_only", False),
    )
    for field, value in mutations:
        changed = deepcopy(good)
        changed[field] = value
        with pytest.raises(ValueError):
            validate_frozen_evidence_result(changed)
    for field, value in (("file_count", 9003), ("byte_count", 1337549192), ("verified_sha256_count", 9003)):
        changed = deepcopy(good)
        changed["combined"][field] = value
        with pytest.raises(ValueError):
            validate_frozen_evidence_result(changed)


def test_campaign_and_per_run_exclusive_creation_exists_but_payload_is_incomplete(tmp_path: Path) -> None:
    controls = reaudit.control_outputs(ROOT)
    campaign = controls["reaudit_campaign_locking.json"]
    per_run = controls["reaudit_per_run_locking.json"]
    assert campaign["exclusive_creation"] is True
    assert campaign["required_explicit_fields_complete"] is False
    assert per_run["per_run_lock_present"] is True
    assert per_run["run_key_explicit_field"] is False
    assert per_run["global_run_id_explicit_field"] is False
    lock = tmp_path / "exclusive.lock"
    with ExclusiveLock(lock, "campaign"):
        with pytest.raises(RuntimeError, match="already exists"):
            ExclusiveLock(lock, "campaign").acquire()


def test_stale_recovery_accepts_zero_age_lock_without_host_campaign_or_completed_output_proof(tmp_path: Path) -> None:
    controls = reaudit.control_outputs(ROOT)["reaudit_stale_lock_recovery.json"]
    assert controls["host_identity_check"] is False
    assert controls["minimum_lock_age_check"] is False
    assert controls["active_campaign_owner_check"] is False
    assert controls["completed_output_validation"] is False
    assert controls["foreign_host_lock_denial"] is False
    lock = tmp_path / "campaign.lock"
    value = lock_payload("campaign-identity", "campaign")
    value["pid"] = 2_147_483_647
    value["process_start_identity"] = "inactive"
    payload = {field: item for field, item in value.items() if field != "lock_hash"}
    value["lock_hash"] = payload_hash(payload, domain="satnet_stage_a_execution_lock_v2")
    lock.write_bytes(canonical_json_bytes(value))
    record = recover_stale_lock(lock, expected_identity="campaign-identity", recovery_log=tmp_path / "recoveries.json")
    assert record["recovered_lock"]["created_unix_ns"] == value["created_unix_ns"]
    assert not lock.exists()


def test_binding_closure_matrix_has_required_open_findings() -> None:
    findings, closure = reaudit.finding_outputs()
    statuses = closure["independent_statuses"]
    assert statuses == {
        "BINDING-001": "CLOSED",
        "BINDING-002": "CLOSED",
        "BINDING-003": "OPEN",
        "BINDING-004": "OPEN",
        "BINDING-005": "OPEN",
        "BINDING-006": "CLOSED",
        "BINDING-007": "OPEN",
        "BINDING-008": "OPEN",
        "BINDING-009": "CLOSED",
    }
    assert findings["binding_findings_remaining"] == ["BINDING-003", "BINDING-004", "BINDING-005", "BINDING-007", "BINDING-008"]
    assert findings["final_verdict"] == reaudit.VERDICT
