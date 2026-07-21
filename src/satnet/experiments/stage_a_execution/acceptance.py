from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import atomic_write_json, payload_hash, read_json_object
from .integrity import verify_artifact_inventory
from .ledger import read_ledger
from .paths import validate_relative_artifact_path

ACCEPTANCE_SCHEMA = "satnet.stage_a.acceptance_report.v1"
ACCEPTANCE_DOMAIN = "satnet_stage_a_acceptance_report_v1"


def evaluate_acceptance(
    *, plan: dict[str, Any], authorization_hash: str, generation_root: Path,
    replay_root: Path, acceptance_root: Path,
) -> dict[str, Any]:
    generation = read_ledger(generation_root / "execution_ledger.json")
    replay = read_ledger(replay_root / "replay_ledger.json")
    for label, ledger, operation in (
        ("generation", generation, "GENERATE"),
        ("replay", replay, "REPLAY"),
    ):
        if ledger["contract_hash"] != plan["contract_hash"] or ledger["partition"] != plan["partition"]:
            raise ValueError(f"{label.title()} partition or contract mismatch")
        if ledger["operation"] != operation or ledger["tooling_commit"] != plan["tooling_commit"] or ledger["tooling_inventory_hash"] != plan["tooling_inventory_hash"]:
            raise ValueError(f"{label.title()} tooling or operation identity mismatch")
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
        if report.get("run_record_hash") != plan_run["run_record_hash"]:
            raise ValueError("Replay run identity mismatch")
        comparisons.append({
            "global_run_id": plan_run["global_run_id"],
            "run_key": plan_run["run_key"],
            "run_record_hash": plan_run["run_record_hash"],
            "generation_replay_equal": True,
        })
    report: dict[str, Any] = {
        "schema_identifier": ACCEPTANCE_SCHEMA,
        "contract_hash": plan["contract_hash"],
        "plan_hash": plan["plan_hash"],
        "authorization_hash": authorization_hash,
        "partition": plan["partition"],
        "expected_run_count": plan["run_count"],
        "accepted_run_count": len(comparisons),
        "comparisons": comparisons,
        "acceptance_state": "PASSED",
    }
    report["acceptance_report_hash"] = payload_hash(report, domain=ACCEPTANCE_DOMAIN)
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
