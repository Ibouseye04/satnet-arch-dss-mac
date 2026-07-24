from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .authorization import Authorization
from .common import payload_hash
from .contract import FrozenStageAContract

PLAN_SCHEMA = "satnet.stage_a.execution_plan.v2"
PLAN_HASH_DOMAIN = "satnet_stage_a_execution_plan_v2"
CAMPAIGN_MANIFEST_HASH_ALGORITHM = "SHA-256"
LEGACY_CAMPAIGN_MANIFEST_VERSION = "1"
LEGACY_CAMPAIGN_MANIFEST_ALGORITHM = "satnet.stage_a.campaign_manifest.legacy.v1"
LEGACY_CAMPAIGN_MANIFEST_DOMAIN = "satnet_stage_a_campaign_manifest_v2"
OPERATION_BOUND_CAMPAIGN_MANIFEST_VERSION = "2"
OPERATION_BOUND_CAMPAIGN_MANIFEST_ALGORITHM = (
    "satnet.stage_a.campaign_manifest.operation_bound.v2"
)
OPERATION_BOUND_CAMPAIGN_MANIFEST_DOMAIN = (
    "satnet_stage_a_campaign_manifest_operation_bound_v2"
)
CAMPAIGN_OPERATIONS = frozenset({"PLAN", "GENERATE", "REPLAY", "ACCEPT"})
_CAMPAIGN_EXCLUDED_FIELDS = frozenset(
    {
        "schema_identifier",
        "operation",
        "authorization_hash",
        "source_generation_ledger_relative_path",
        "source_generation_ledger_byte_length",
        "source_generation_ledger_sha256",
        "source_replay_ledger_relative_path",
        "source_replay_ledger_byte_length",
        "source_replay_ledger_sha256",
        "campaign_manifest_version",
        "campaign_manifest_algorithm",
        "campaign_manifest_operation",
        "campaign_manifest_hash",
        "plan_hash",
    }
)


def campaign_manifest_hash(
    plan: dict[str, Any],
    *,
    version: str,
    algorithm: str,
    operation: str,
) -> str:
    if operation not in CAMPAIGN_OPERATIONS:
        raise ValueError("Campaign manifest operation is invalid")
    payload = {
        key: value
        for key, value in plan.items()
        if key not in _CAMPAIGN_EXCLUDED_FIELDS
    }
    if version == LEGACY_CAMPAIGN_MANIFEST_VERSION:
        if algorithm != LEGACY_CAMPAIGN_MANIFEST_ALGORITHM:
            raise ValueError("Legacy campaign manifest algorithm mismatch")
        domain = LEGACY_CAMPAIGN_MANIFEST_DOMAIN
    elif version == OPERATION_BOUND_CAMPAIGN_MANIFEST_VERSION:
        if algorithm != OPERATION_BOUND_CAMPAIGN_MANIFEST_ALGORITHM:
            raise ValueError("Operation-bound campaign manifest algorithm mismatch")
        payload.update(
            {
                "campaign_manifest_algorithm": algorithm,
                "campaign_manifest_hash_algorithm": CAMPAIGN_MANIFEST_HASH_ALGORITHM,
                "campaign_manifest_operation": operation,
                "campaign_manifest_version": version,
                "operation": operation,
            }
        )
        domain = OPERATION_BOUND_CAMPAIGN_MANIFEST_DOMAIN
    else:
        raise ValueError("Campaign manifest version is unsupported")
    return payload_hash(payload, domain=domain)


def campaign_manifest_hash_for_ledger(
    ledger: dict[str, Any],
    plan: dict[str, Any],
    *,
    version: str,
    algorithm: str,
    operation: str,
) -> str:
    historical = dict(plan)
    replacements = {
        "contract_hash": "contract_hash",
        "stable_executable_commit": "stable_executable_commit",
        "executable_inventory_hash": "executable_inventory_hash",
        "tooling_proposal_hash": "tooling_proposal_hash",
        "artifact_contract_hash": "artifact_contract_hash",
        "partition": "partition",
        "output_roots": "output_root_identity",
        "run_count": "expected_run_count",
        "source_generation_ledger_relative_path": "source_generation_ledger_relative_path",
        "source_generation_ledger_byte_length": "source_generation_ledger_byte_length",
        "source_generation_ledger_sha256": "source_generation_ledger_sha256",
        "source_replay_ledger_relative_path": "source_replay_ledger_relative_path",
        "source_replay_ledger_byte_length": "source_replay_ledger_byte_length",
        "source_replay_ledger_sha256": "source_replay_ledger_sha256",
    }
    for plan_field, ledger_field in replacements.items():
        historical[plan_field] = ledger[ledger_field]
    historical["operation"] = operation
    historical["design_count"] = len({row["design_id"] for row in plan["runs"]})
    return campaign_manifest_hash(
        historical,
        version=version,
        algorithm=algorithm,
        operation=operation,
    )


def validate_legacy_campaign_manifest_identity(
    source: dict[str, Any],
    plan: dict[str, Any],
    *,
    operation: str,
) -> None:
    if source.get("operation") != operation:
        raise ValueError("Historical campaign manifest operation mismatch")
    expected = campaign_manifest_hash_for_ledger(
        source,
        plan,
        version=LEGACY_CAMPAIGN_MANIFEST_VERSION,
        algorithm=LEGACY_CAMPAIGN_MANIFEST_ALGORITHM,
        operation=operation,
    )
    if source.get("campaign_manifest_hash") != expected:
        raise ValueError("Historical legacy campaign manifest hash mismatch")


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
    if operation in {"REPLAY", "ACCEPT"} and authorization is None:
        raise PermissionError(f"{operation} plans require exact source-ledger authorization bindings")
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
            "design_construction_seed": seed["design_construction_seed"],
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
        "campaign_manifest_version": OPERATION_BOUND_CAMPAIGN_MANIFEST_VERSION,
        "campaign_manifest_algorithm": OPERATION_BOUND_CAMPAIGN_MANIFEST_ALGORITHM,
        "campaign_manifest_operation": operation,
        "output_roots": {
            "generation": str(generation_root.resolve(strict=False)),
            "replay": str(replay_root.resolve(strict=False)),
            "acceptance": str(acceptance_root.resolve(strict=False)),
        },
        "authorization_hash": None if authorization is None else authorization.sha256,
        "source_generation_ledger_relative_path": None if authorization is None else authorization.document["source_generation_ledger_relative_path"],
        "source_generation_ledger_byte_length": None if authorization is None else authorization.document["source_generation_ledger_byte_length"],
        "source_generation_ledger_sha256": None if authorization is None else authorization.document["source_generation_ledger_sha256"],
        "source_replay_ledger_relative_path": None if authorization is None else authorization.document["source_replay_ledger_relative_path"],
        "source_replay_ledger_byte_length": None if authorization is None else authorization.document["source_replay_ledger_byte_length"],
        "source_replay_ledger_sha256": None if authorization is None else authorization.document["source_replay_ledger_sha256"],
        "run_count": len(records),
        "design_count": len({row["design_id"] for row in records}),
        "runs": records,
    }
    plan["campaign_manifest_hash"] = campaign_manifest_hash(
        plan,
        version=OPERATION_BOUND_CAMPAIGN_MANIFEST_VERSION,
        algorithm=OPERATION_BOUND_CAMPAIGN_MANIFEST_ALGORITHM,
        operation=operation,
    )
    plan["plan_hash"] = payload_hash(plan, domain=PLAN_HASH_DOMAIN)
    return plan


def validate_plan_contract_binding(plan: dict[str, Any], contract: FrozenStageAContract) -> None:
    frozen_runs = {row["run_key"]: row for row in contract.runs}
    frozen_seeds = {row["run_key"]: row for row in contract.seeds}
    expected_ids = tuple(contract.partitions[plan["partition"]]["global_run_ids"])
    if tuple(row["global_run_id"] for row in plan["runs"]) != expected_ids:
        raise ValueError("Plan run set differs from the frozen partition")
    for row in plan["runs"]:
        frozen_run = frozen_runs.get(row["run_key"])
        frozen_seed = frozen_seeds.get(row["run_key"])
        if frozen_run is None or frozen_seed is None:
            raise ValueError("Plan run is absent from the frozen contract")
        run_fields = (
            "global_run_id", "design_id", "realization_id", "realization_index",
            "design_record_hash", "run_record_hash", "partition", "sealed",
        )
        seed_fields = (
            "design_construction_seed", "ground_selection_seed",
            "satellite_failure_seed", "ground_failure_seed",
        )
        if any(row[field] != frozen_run[field] for field in run_fields):
            raise ValueError("Plan run identity differs from the frozen contract")
        if any(row[field] != frozen_seed[field] for field in seed_fields):
            raise ValueError("Plan seed identity differs from the frozen seed manifest")


def validate_plan(plan: dict[str, Any]) -> None:
    claimed = plan.get("plan_hash")
    payload = {key: value for key, value in plan.items() if key != "plan_hash"}
    if claimed != payload_hash(payload, domain=PLAN_HASH_DOMAIN):
        raise ValueError("Plan hash mismatch")
    runs = plan.get("runs")
    if not isinstance(runs, list) or len(runs) != plan.get("run_count"):
        raise ValueError("Plan run count mismatch")
    if plan.get("campaign_manifest_version") != OPERATION_BOUND_CAMPAIGN_MANIFEST_VERSION:
        raise ValueError("Current campaign manifest version mismatch")
    if plan.get("campaign_manifest_algorithm") != OPERATION_BOUND_CAMPAIGN_MANIFEST_ALGORITHM:
        raise ValueError("Current campaign manifest algorithm mismatch")
    if plan.get("campaign_manifest_operation") != plan.get("operation"):
        raise ValueError("Current campaign manifest operation mismatch")
    expected_campaign_hash = campaign_manifest_hash(
        payload,
        version=plan["campaign_manifest_version"],
        algorithm=plan["campaign_manifest_algorithm"],
        operation=plan["campaign_manifest_operation"],
    )
    if plan.get("campaign_manifest_hash") != expected_campaign_hash:
        raise ValueError("Campaign manifest hash mismatch")
    required_run_fields = {
        "contract_hash", "design_id", "design_index", "run_key", "global_run_id",
        "realization_id", "realization_index", "region", "partition", "sealed",
        "design_record_hash", "run_record_hash", "design_construction_seed",
        "ground_selection_seed", "satellite_failure_seed", "ground_failure_seed",
        "expected_output_relative_path",
    }
    if any(set(row) != required_run_fields for row in runs):
        raise ValueError("Plan run-record field set mismatch")
    ids = [row["global_run_id"] for row in runs]
    outputs = [row["expected_output_relative_path"] for row in runs]
    if ids != sorted(ids) or len(ids) != len(set(ids)) or len(outputs) != len(set(outputs)):
        raise ValueError("Plan ordering or uniqueness mismatch")
