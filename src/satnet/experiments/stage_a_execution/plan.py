from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .authorization import Authorization
from .common import payload_hash
from .contract import FrozenStageAContract

PLAN_SCHEMA = "satnet.stage_a.execution_plan.v2"
PLAN_HASH_DOMAIN = "satnet_stage_a_execution_plan_v2"


def build_plan(
    contract: FrozenStageAContract,
    *,
    partition: str,
    operation: str,
    stable_executable_commit: str,
    executable_inventory_hash: str,
    tooling_proposal_hash: str,
    artifact_contract_hash: str,
    generation_root: Path,
    replay_root: Path,
    acceptance_root: Path,
    authorization: Authorization | None = None,
    selected_run_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    if partition not in {"development", "validation"}:
        raise PermissionError("Ordinary plans cannot access sealed holdout")
    frozen_ids = tuple(contract.partitions[partition]["global_run_ids"])
    run_ids = frozen_ids if selected_run_ids is None else tuple(sorted(selected_run_ids))
    if run_ids != frozen_ids:
        raise PermissionError("Plan run set must equal one complete frozen partition")
    designs = {row["design_id"]: row for row in contract.designs}
    seeds = {row["run_key"]: row for row in contract.seeds}
    records: list[dict[str, Any]] = []
    for run in sorted((row for row in contract.runs if row["global_run_id"] in run_ids), key=lambda row: row["global_run_id"]):
        if run["partition"] != partition or run["sealed"] is not False:
            raise PermissionError("Plan crosses partition or includes sealed run")
        seed = seeds[run["run_key"]]
        design = designs[run["design_id"]]
        records.append({
            "contract_hash": contract.contract_hash,
            "design_id": run["design_id"],
            "design_index": run["design_index"],
            "run_key": run["run_key"],
            "global_run_id": run["global_run_id"],
            "realization_id": run["realization_id"],
            "realization_index": run["realization_index"],
            "region": run["region"],
            "partition": partition,
            "sealed": False,
            "design_record_hash": design["design_record_hash"],
            "run_record_hash": run["run_record_hash"],
            "ground_selection_seed": seed["ground_selection_seed"],
            "satellite_failure_seed": seed["satellite_failure_seed"],
            "ground_failure_seed": seed["ground_failure_seed"],
            "expected_output_relative_path": f"{partition}/{run['design_id']}/{run['run_key']}",
        })
    if len(records) != len(run_ids) or len({row["expected_output_relative_path"] for row in records}) != len(records):
        raise ValueError("Plan identity or output path collision")
    plan = {
        "schema_identifier": PLAN_SCHEMA,
        "contract_hash": contract.contract_hash,
        "stable_executable_commit": stable_executable_commit,
        "executable_inventory_hash": executable_inventory_hash,
        "tooling_proposal_hash": tooling_proposal_hash,
        "artifact_contract_hash": artifact_contract_hash,
        "partition": partition,
        "operation": operation,
        "output_roots": {
            "generation": str(generation_root.resolve(strict=False)),
            "replay": str(replay_root.resolve(strict=False)),
            "acceptance": str(acceptance_root.resolve(strict=False)),
        },
        "authorization_hash": None if authorization is None else authorization.sha256,
        "run_count": len(records),
        "design_count": len({row["design_id"] for row in records}),
        "runs": records,
    }
    campaign_manifest = {
        key: value
        for key, value in plan.items()
        if key not in {"schema_identifier", "operation", "authorization_hash"}
    }
    plan["campaign_manifest_hash"] = payload_hash(
        campaign_manifest, domain="satnet_stage_a_campaign_manifest_v2"
    )
    plan["plan_hash"] = payload_hash(plan, domain=PLAN_HASH_DOMAIN)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    claimed = plan.get("plan_hash")
    payload = {key: value for key, value in plan.items() if key != "plan_hash"}
    if claimed != payload_hash(payload, domain=PLAN_HASH_DOMAIN):
        raise ValueError("Plan hash mismatch")
    runs = plan.get("runs")
    if not isinstance(runs, list) or len(runs) != plan.get("run_count"):
        raise ValueError("Plan run count mismatch")
    campaign_manifest = {
        key: value
        for key, value in payload.items()
        if key not in {"schema_identifier", "operation", "authorization_hash", "campaign_manifest_hash"}
    }
    if plan.get("campaign_manifest_hash") != payload_hash(campaign_manifest, domain="satnet_stage_a_campaign_manifest_v2"):
        raise ValueError("Campaign manifest hash mismatch")
    ids = [row["global_run_id"] for row in runs]
    outputs = [row["expected_output_relative_path"] for row in runs]
    if ids != sorted(ids) or len(ids) != len(set(ids)) or len(outputs) != len(set(outputs)):
        raise ValueError("Plan ordering or uniqueness mismatch")
