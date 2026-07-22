from __future__ import annotations

from pathlib import Path
from typing import Any

from .integrity import verify_artifact_inventory


def validate_resume_identity(ledger: dict[str, Any], plan: dict[str, Any], authorization_hash: str) -> None:
    expected = {
        "contract_hash": plan["contract_hash"],
        "plan_hash": plan["plan_hash"],
        "campaign_manifest_hash": plan["campaign_manifest_hash"],
        "authorization_hash": authorization_hash,
        "stable_executable_commit": plan["stable_executable_commit"],
        "executable_inventory_hash": plan["executable_inventory_hash"],
        "tooling_proposal_hash": plan["tooling_proposal_hash"],
        "artifact_contract_hash": plan["artifact_contract_hash"],
        "operation": plan["operation"],
        "partition": plan["partition"],
        "expected_run_count": plan["run_count"],
        "output_root_identity": plan["output_roots"],
    }
    for field, value in expected.items():
        if ledger.get(field) != value:
            raise ValueError(f"Resume identity mismatch: {field}")
    expected_runs = [
        {
            "global_run_id": row["global_run_id"],
            "run_key": row["run_key"],
            "run_record_hash": row["run_record_hash"],
            "ground_selection_seed": row["ground_selection_seed"],
            "satellite_failure_seed": row["satellite_failure_seed"],
            "ground_failure_seed": row["ground_failure_seed"],
        }
        for row in plan["runs"]
    ]
    actual_runs = [
        {key: row[key] for key in expected_runs[index]}
        for index, row in enumerate(ledger.get("records", []))
    ]
    if actual_runs != expected_runs:
        raise ValueError("Resume run or seed set mismatch")


def validate_ledger_binding(ledger: dict[str, Any], plan: dict[str, Any], *, operation: str) -> None:
    expected = {
        "contract_hash": plan["contract_hash"],
        "campaign_manifest_hash": plan["campaign_manifest_hash"],
        "stable_executable_commit": plan["stable_executable_commit"],
        "executable_inventory_hash": plan["executable_inventory_hash"],
        "tooling_proposal_hash": plan["tooling_proposal_hash"],
        "artifact_contract_hash": plan["artifact_contract_hash"],
        "operation": operation,
        "partition": plan["partition"],
        "expected_run_count": plan["run_count"],
        "output_root_identity": plan["output_roots"],
    }
    for field, value in expected.items():
        if ledger.get(field) != value:
            raise ValueError(f"Ledger identity mismatch: {field}")
    authorization_hash = ledger.get("authorization_hash")
    if not isinstance(authorization_hash, str) or len(authorization_hash) != 64 or any(character not in "0123456789abcdef" for character in authorization_hash):
        raise ValueError("Ledger authorization identity mismatch")
    expected_records = [
        {
            "global_run_id": row["global_run_id"],
            "run_key": row["run_key"],
            "run_record_hash": row["run_record_hash"],
            "ground_selection_seed": row["ground_selection_seed"],
            "satellite_failure_seed": row["satellite_failure_seed"],
            "ground_failure_seed": row["ground_failure_seed"],
            "output_relative_path": row["expected_output_relative_path"],
        }
        for row in plan["runs"]
    ]
    actual_records = [
        {field: row.get(field) for field in expected_records[index]}
        for index, row in enumerate(ledger.get("records", []))
    ]
    if actual_records != expected_records:
        raise ValueError("Ledger run-record or seed identity mismatch")


def resumable_run_ids(ledger: dict[str, Any], campaign_root: Path, *, retry_failed: bool) -> tuple[int, ...]:
    selected: list[int] = []
    for record in ledger["records"]:
        state = record["state"]
        if state == "SUCCEEDED":
            run_root = campaign_root / record["output_relative_path"]
            verify_artifact_inventory(run_root, record["artifacts"])
            continue
        if state in {"FAILED", "INTERRUPTED"} and not retry_failed:
            continue
        if state not in {"PLANNED", "FAILED", "INTERRUPTED"}:
            raise RuntimeError(f"Run is not safely resumable from state {state}")
        selected.append(record["global_run_id"])
    return tuple(selected)
