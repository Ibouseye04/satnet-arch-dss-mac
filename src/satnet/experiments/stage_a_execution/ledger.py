from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import atomic_write_json, payload_hash, read_json_object

LEDGER_SCHEMA = "satnet.stage_a.execution_ledger.v2"
LEDGER_DOMAIN = "satnet_stage_a_execution_ledger_v2"
STATES = frozenset({"PLANNED", "STARTING", "RUNNING", "SUCCEEDED", "FAILED", "INTERRUPTED"})
TRANSITIONS = {
    "PLANNED": frozenset({"STARTING"}),
    "STARTING": frozenset({"RUNNING", "FAILED", "INTERRUPTED"}),
    "RUNNING": frozenset({"SUCCEEDED", "FAILED", "INTERRUPTED"}),
    "FAILED": frozenset({"STARTING"}),
    "INTERRUPTED": frozenset({"STARTING"}),
    "SUCCEEDED": frozenset(),
}


def ledger_hash(ledger: dict[str, Any]) -> str:
    return payload_hash({key: value for key, value in ledger.items() if key != "ledger_hash"}, domain=LEDGER_DOMAIN)


def validate_ledger(ledger: dict[str, Any]) -> None:
    if ledger.get("schema_identifier") != LEDGER_SCHEMA or ledger.get("ledger_hash") != ledger_hash(ledger):
        raise ValueError("Execution ledger identity or hash mismatch")
    records = ledger.get("records")
    if not isinstance(records, list) or len(records) != ledger.get("expected_run_count"):
        raise ValueError("Execution ledger record count mismatch")
    ids = [record.get("global_run_id") for record in records]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("Execution ledger run ordering or uniqueness mismatch")
    for record in records:
        if record.get("state") not in STATES:
            raise ValueError("Unknown execution ledger state")
        if record.get("state") == "SUCCEEDED":
            if not record.get("artifacts") or not record.get("adapter_result"):
                raise ValueError("Successful run has no verified artifacts or adapter result")
            if record["adapter_result"].get("validation_status") != "PASSED":
                raise ValueError("Successful run adapter result was not validated")


def build_ledger(plan: dict[str, Any], authorization_hash: str) -> dict[str, Any]:
    ledger: dict[str, Any] = {
        "schema_identifier": LEDGER_SCHEMA,
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
        "records": [
            {
                "global_run_id": row["global_run_id"],
                "run_key": row["run_key"],
                "run_record_hash": row["run_record_hash"],
                "ground_selection_seed": row["ground_selection_seed"],
                "satellite_failure_seed": row["satellite_failure_seed"],
                "ground_failure_seed": row["ground_failure_seed"],
                "output_relative_path": row["expected_output_relative_path"],
                "state": "PLANNED",
                "attempt_count": 0,
                "artifacts": [],
                "artifact_inventory_hash": None,
                "adapter_result": None,
                "failure": None,
            }
            for row in plan["runs"]
        ],
    }
    ledger["ledger_hash"] = ledger_hash(ledger)
    return ledger


def read_ledger(path: Path) -> dict[str, Any]:
    ledger = read_json_object(path)
    validate_ledger(ledger)
    return ledger


def write_ledger(path: Path, ledger: dict[str, Any], *, overwrite: bool) -> None:
    ledger["ledger_hash"] = ledger_hash(ledger)
    validate_ledger(ledger)
    atomic_write_json(path, ledger, overwrite=overwrite)


def transition(
    path: Path, *, global_run_id: int, new_state: str,
    artifacts: list[dict[str, Any]] | None = None,
    artifact_inventory_hash: str | None = None, adapter_result: dict[str, Any] | None = None,
    failure: str | None = None,
) -> dict[str, Any]:
    ledger = read_ledger(path)
    record = next((item for item in ledger["records"] if item["global_run_id"] == global_run_id), None)
    if record is None:
        raise ValueError("Unknown run in execution ledger")
    if new_state not in TRANSITIONS[record["state"]]:
        raise ValueError(f"Illegal ledger transition {record['state']} -> {new_state}")
    record["state"] = new_state
    if new_state == "STARTING":
        record["attempt_count"] += 1
        record["failure"] = None
    if new_state == "SUCCEEDED":
        if not artifacts or not artifact_inventory_hash or not adapter_result:
            raise ValueError("Success requires verified artifact identities and adapter result")
        if adapter_result.get("validation_status") != "PASSED" or adapter_result.get("simulation_return_status") != "SUCCEEDED":
            raise ValueError("Success requires a passed structured adapter result")
        record["artifacts"] = artifacts
        record["artifact_inventory_hash"] = artifact_inventory_hash
        record["adapter_result"] = adapter_result
    if new_state in {"FAILED", "INTERRUPTED"}:
        record["failure"] = failure or new_state
    write_ledger(path, ledger, overwrite=True)
    return ledger
