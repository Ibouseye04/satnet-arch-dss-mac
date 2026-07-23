from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .common import ensure_hex
from .integrity import verify_artifact_inventory


@dataclass(frozen=True)
class LedgerProvenance:
    contract_hash: str
    plan_hash: str
    campaign_manifest_hash: str
    authorization_hash: str
    stable_executable_commit: str
    executable_inventory_hash: str
    tooling_proposal_hash: str
    artifact_contract_hash: str
    operation: str
    partition: str
    expected_run_count: int
    output_root_identity: dict[str, str]
    source_generation_ledger_relative_path: str | None
    source_generation_ledger_byte_length: int | None
    source_generation_ledger_sha256: str | None
    source_replay_ledger_relative_path: str | None
    source_replay_ledger_byte_length: int | None
    source_replay_ledger_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def ledger_provenance_from_mapping(value: dict[str, Any]) -> LedgerProvenance:
    expected_fields = set(LedgerProvenance.__dataclass_fields__)
    if set(value) != expected_fields:
        raise ValueError("Ledger provenance field set mismatch")
    provenance = LedgerProvenance(**value)
    ensure_hex(provenance.contract_hash, length=64, field="source_contract_hash")
    ensure_hex(provenance.plan_hash, length=64, field="source_plan_hash")
    ensure_hex(provenance.campaign_manifest_hash, length=64, field="source_campaign_manifest_hash")
    ensure_hex(provenance.authorization_hash, length=64, field="source_authorization_hash")
    ensure_hex(provenance.stable_executable_commit, length=40, field="source_stable_executable_commit")
    ensure_hex(provenance.executable_inventory_hash, length=64, field="source_executable_inventory_hash")
    ensure_hex(provenance.tooling_proposal_hash, length=64, field="source_tooling_proposal_hash")
    ensure_hex(provenance.artifact_contract_hash, length=64, field="source_artifact_contract_hash")
    if provenance.operation not in {"GENERATE", "REPLAY"}:
        raise ValueError("Ledger provenance operation mismatch")
    if provenance.partition not in {"development", "validation"}:
        raise ValueError("Ledger provenance partition mismatch")
    if type(provenance.expected_run_count) is not int or provenance.expected_run_count <= 0:
        raise ValueError("Ledger provenance expected run count is invalid")
    if set(provenance.output_root_identity) != {"generation", "replay", "acceptance"}:
        raise ValueError("Ledger provenance output-root identity mismatch")
    for prefix in ("source_generation_ledger", "source_replay_ledger"):
        relative_path = getattr(provenance, f"{prefix}_relative_path")
        byte_length = getattr(provenance, f"{prefix}_byte_length")
        sha256 = getattr(provenance, f"{prefix}_sha256")
        if relative_path is None:
            if byte_length is not None or sha256 is not None:
                raise ValueError(f"{prefix} provenance is incomplete")
        else:
            if relative_path not in {"execution_ledger.json", "replay_ledger.json"}:
                raise ValueError(f"{prefix} provenance path is not canonical")
            if type(byte_length) is not int or byte_length <= 0:
                raise ValueError(f"{prefix} provenance byte length is invalid")
            ensure_hex(sha256, length=64, field=f"{prefix}_sha256")
    return provenance


def ledger_provenance_from_plan(plan: dict[str, Any], authorization_hash: str) -> LedgerProvenance:
    value = {
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
        "source_generation_ledger_relative_path": plan["source_generation_ledger_relative_path"],
        "source_generation_ledger_byte_length": plan["source_generation_ledger_byte_length"],
        "source_generation_ledger_sha256": plan["source_generation_ledger_sha256"],
        "source_replay_ledger_relative_path": plan["source_replay_ledger_relative_path"],
        "source_replay_ledger_byte_length": plan["source_replay_ledger_byte_length"],
        "source_replay_ledger_sha256": plan["source_replay_ledger_sha256"],
    }
    return ledger_provenance_from_mapping(value)


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
        "source_generation_ledger_relative_path": plan["source_generation_ledger_relative_path"],
        "source_generation_ledger_byte_length": plan["source_generation_ledger_byte_length"],
        "source_generation_ledger_sha256": plan["source_generation_ledger_sha256"],
        "source_replay_ledger_relative_path": plan["source_replay_ledger_relative_path"],
        "source_replay_ledger_byte_length": plan["source_replay_ledger_byte_length"],
        "source_replay_ledger_sha256": plan["source_replay_ledger_sha256"],
    }
    for field, value in expected.items():
        if ledger.get(field) != value:
            raise ValueError(f"Resume identity mismatch: {field}")
    expected_runs = [
        {
            "global_run_id": row["global_run_id"],
            "run_key": row["run_key"],
            "run_record_hash": row["run_record_hash"],
            "design_construction_seed": row["design_construction_seed"],
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


def validate_ledger_binding(
    ledger: dict[str, Any],
    plan: dict[str, Any],
    *,
    operation: str,
    provenance: LedgerProvenance | None = None,
) -> None:
    if provenance is None:
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
            "source_generation_ledger_relative_path": plan["source_generation_ledger_relative_path"] if operation == "REPLAY" else None,
            "source_generation_ledger_byte_length": plan["source_generation_ledger_byte_length"] if operation == "REPLAY" else None,
            "source_generation_ledger_sha256": plan["source_generation_ledger_sha256"] if operation == "REPLAY" else None,
            "source_replay_ledger_relative_path": None,
            "source_replay_ledger_byte_length": None,
            "source_replay_ledger_sha256": None,
        }
    else:
        if provenance.operation != operation:
            raise ValueError("Ledger provenance operation mismatch")
        expected = provenance.as_dict()
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
            "design_construction_seed": row["design_construction_seed"],
            "ground_selection_seed": row["ground_selection_seed"],
            "satellite_failure_seed": row["satellite_failure_seed"],
            "ground_failure_seed": row["ground_failure_seed"],
            "output_relative_path": row["expected_output_relative_path"],
        }
        for row in plan["runs"]
    ]
    records = ledger.get("records", [])
    if not isinstance(records, list) or len(records) != len(expected_records):
        raise ValueError("Ledger run-record or seed identity mismatch")
    actual_records = [
        {field: row.get(field) for field in expected_records[index]}
        for index, row in enumerate(records)
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
