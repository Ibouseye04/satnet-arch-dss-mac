from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import atomic_write_json, payload_hash, read_json_object
from .integrity import verify_artifact_inventory
from .ledger import read_bound_ledger
from .locking import campaign_lock
from .paths import validate_output_roots, validate_relative_artifact_path
from .preflight import PreflightCertificate, require_preflight
from .resume import ledger_provenance_from_mapping, validate_ledger_binding

ACCEPTANCE_SCHEMA = "satnet.stage_a.acceptance_report.v1"
ACCEPTANCE_DOMAIN = "satnet_stage_a_acceptance_report_v1"


def evaluate_acceptance(
    *, repo_root: Path, plan: dict[str, Any], authorization_hash: str,
    preflight: PreflightCertificate, generation_root: Path,
    replay_root: Path, acceptance_root: Path,
) -> dict[str, Any]:
    require_preflight(preflight, plan=plan, authorization_hash=authorization_hash)
    roots = validate_output_roots(
        repo_root=repo_root,
        generation_root=Path(plan["output_roots"]["generation"]),
        replay_root=Path(plan["output_roots"]["replay"]),
        acceptance_root=Path(plan["output_roots"]["acceptance"]),
        require_absent=False,
    )
    supplied = {
        "generation": str(generation_root.resolve(strict=False)),
        "replay": str(replay_root.resolve(strict=False)),
        "acceptance": str(acceptance_root.resolve(strict=False)),
    }
    if roots != plan["output_roots"] or supplied != roots:
        raise PermissionError("Acceptance roots differ from preflight-bound plan")
    generation, generation_identity = read_bound_ledger(
        generation_root,
        relative_path=plan["source_generation_ledger_relative_path"],
        byte_length=plan["source_generation_ledger_byte_length"],
        sha256=plan["source_generation_ledger_sha256"],
    )
    replay, replay_identity = read_bound_ledger(
        replay_root,
        relative_path=plan["source_replay_ledger_relative_path"],
        byte_length=plan["source_replay_ledger_byte_length"],
        sha256=plan["source_replay_ledger_sha256"],
    )
    generation_provenance = ledger_provenance_from_mapping(
        preflight.report["source_provenance"]["generation"]
    )
    validate_ledger_binding(
        generation,
        plan,
        operation="GENERATE",
        provenance=generation_provenance,
    )
    validate_ledger_binding(replay, plan, operation="REPLAY")
    replay_source_identity = {
        "relative_path": replay["source_generation_ledger_relative_path"],
        "byte_length": replay["source_generation_ledger_byte_length"],
        "sha256": replay["source_generation_ledger_sha256"],
    }
    if replay_source_identity != generation_identity:
        raise ValueError("Replay ledger is not bound to the accepted generation ledger bytes")
    expected_ids = [row["global_run_id"] for row in plan["runs"]]
    if [row["global_run_id"] for row in generation["records"]] != expected_ids or [row["global_run_id"] for row in replay["records"]] != expected_ids:
        raise ValueError("Acceptance run-set mismatch")
    comparisons: list[dict[str, Any]] = []
    for plan_run, generated, replayed in zip(plan["runs"], generation["records"], replay["records"], strict=True):
        if generated["state"] != "SUCCEEDED" or replayed["state"] != "SUCCEEDED":
            raise ValueError("Acceptance requires complete successful generation and replay")
        relative = validate_relative_artifact_path(plan_run["expected_output_relative_path"])
        verify_artifact_inventory(generation_root / relative, generated["artifacts"])
        verify_artifact_inventory(replay_root / relative, replayed["artifacts"])
        report = read_json_object(replay_root / relative / "replay_report.json")
        if report.get("generation_replay_equal") is not True or report.get("replay_state") != "SUCCEEDED":
            raise ValueError("Replay mismatch cannot be accepted")
        expected_report = {
            "contract_hash": plan["contract_hash"],
            "generation_authorization_hash": generation["authorization_hash"],
            "generation_plan_hash": generation["plan_hash"],
            "generation_campaign_manifest_hash": generation["campaign_manifest_hash"],
            "source_generation_ledger_relative_path": generation_identity["relative_path"],
            "source_generation_ledger_byte_length": generation_identity["byte_length"],
            "source_generation_ledger_sha256": generation_identity["sha256"],
            "stable_executable_commit": plan["stable_executable_commit"],
            "executable_inventory_hash": plan["executable_inventory_hash"],
            "tooling_proposal_hash": plan["tooling_proposal_hash"],
            "artifact_contract_hash": plan["artifact_contract_hash"],
            "global_run_id": plan_run["global_run_id"],
            "run_key": plan_run["run_key"],
            "run_record_hash": plan_run["run_record_hash"],
            "design_construction_seed": plan_run["design_construction_seed"],
            "ground_selection_seed": plan_run["ground_selection_seed"],
            "satellite_failure_seed": plan_run["satellite_failure_seed"],
            "ground_failure_seed": plan_run["ground_failure_seed"],
        }
        for field, value in expected_report.items():
            if report.get(field) != value:
                raise ValueError(f"Replay report identity or seed mismatch: {field}")
        comparisons.append({
            "global_run_id": plan_run["global_run_id"],
            "run_key": plan_run["run_key"],
            "run_record_hash": plan_run["run_record_hash"],
            "design_construction_seed": plan_run["design_construction_seed"],
            "ground_selection_seed": plan_run["ground_selection_seed"],
            "satellite_failure_seed": plan_run["satellite_failure_seed"],
            "ground_failure_seed": plan_run["ground_failure_seed"],
            "generation_replay_equal": True,
        })
    report: dict[str, Any] = {
        "schema_identifier": ACCEPTANCE_SCHEMA,
        "contract_hash": plan["contract_hash"],
        "plan_hash": plan["plan_hash"],
        "authorization_hash": authorization_hash,
        "stable_executable_commit": plan["stable_executable_commit"],
        "executable_inventory_hash": plan["executable_inventory_hash"],
        "tooling_proposal_hash": plan["tooling_proposal_hash"],
        "artifact_contract_hash": plan["artifact_contract_hash"],
        "generation_plan_hash": generation["plan_hash"],
        "generation_authorization_hash": generation["authorization_hash"],
        "replay_plan_hash": replay["plan_hash"],
        "replay_authorization_hash": replay["authorization_hash"],
        "generation_ledger_relative_path": generation_identity["relative_path"],
        "generation_ledger_byte_length": generation_identity["byte_length"],
        "generation_ledger_sha256": generation_identity["sha256"],
        "replay_ledger_relative_path": replay_identity["relative_path"],
        "replay_ledger_byte_length": replay_identity["byte_length"],
        "replay_ledger_sha256": replay_identity["sha256"],
        "replay_recorded_source_generation_ledger_relative_path": replay_source_identity["relative_path"],
        "replay_recorded_source_generation_ledger_byte_length": replay_source_identity["byte_length"],
        "replay_recorded_source_generation_ledger_sha256": replay_source_identity["sha256"],
        "output_roots": roots,
        "partition": plan["partition"],
        "expected_run_count": plan["run_count"],
        "accepted_run_count": len(comparisons),
        "comparisons": comparisons,
        "acceptance_state": "PASSED",
    }
    report["acceptance_report_hash"] = payload_hash(report, domain=ACCEPTANCE_DOMAIN)
    with campaign_lock(acceptance_root, plan, authorization_hash):
        acceptance_root.mkdir(parents=False, exist_ok=False)
        atomic_write_json(acceptance_root / "acceptance_report.json", report)
    return report


def validate_acceptance_report(report: dict[str, Any]) -> None:
    claimed = report.get("acceptance_report_hash")
    payload = {key: value for key, value in report.items() if key != "acceptance_report_hash"}
    if claimed != payload_hash(payload, domain=ACCEPTANCE_DOMAIN):
        raise ValueError("Acceptance report hash mismatch")
    if report.get("acceptance_state") != "PASSED" or report.get("accepted_run_count") != report.get("expected_run_count"):
        raise ValueError("Acceptance report is incomplete")
