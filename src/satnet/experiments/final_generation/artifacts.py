from __future__ import annotations

from dataclasses import asdict, fields
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from satnet.ground.canonical import canonical_float_string, canonical_hash
from satnet.ground.failure_realization import GROUND_FAILURE_REALIZATION_SCHEMA_VERSION
from satnet.ground.failure_service_aggregation import GROUND_FAILURE_SERVICE_RUN_SCHEMA_VERSION
from satnet.ground.failure_service_metrics import GROUND_FAILURE_SERVICE_STEP_SCHEMA_VERSION
from satnet.ground.integrated_graph import INTEGRATED_GRAPH_SCHEMA_VERSION
from satnet.ground.persistence import GROUND_DESIGN_SCHEMA_VERSION
from satnet.ground.service_aggregation import (
    GROUND_SERVICE_RUN_SCHEMA_VERSION,
    GROUND_SERVICE_STEP_SCHEMA_VERSION,
)
from satnet.ground.visibility_persistence import GROUND_VISIBILITY_SCHEMA_VERSION
from satnet.simulation.tier1_rollout import (
    DATASET_VERSION,
    SCHEMA_VERSION,
    Tier1FailureRealization,
    Tier1RolloutConfig,
    Tier1RolloutStep,
    Tier1RolloutSummary,
)

from .constants import (
    ARTIFACT_ROLES,
    CONTRACT_SPEC_HASH,
    RUN_FILES,
    RUN_RESULT_DOMAIN,
    RUN_RESULT_IDENTITY_VERSION,
    RUN_RESULT_SCHEMA_VERSION,
    SATELLITE_ARTIFACT_DOMAIN,
    SATELLITE_ARTIFACT_IDENTITY_VERSION,
    SATELLITE_ARTIFACT_SCHEMA_VERSION,
    SCIENTIFIC_FILE_KEYS,
    SCIENTIFIC_INVENTORY_DOMAIN,
    SCIENTIFIC_INVENTORY_IDENTITY_VERSION,
    SCIENTIFIC_INVENTORY_SCHEMA_VERSION,
    TARGET_ARTIFACT_DOMAIN,
    TARGET_ARTIFACT_IDENTITY_VERSION,
    TARGET_ARTIFACT_SCHEMA_VERSION,
    TARGET_BOOLEAN_FIELDS,
    TARGET_FIELDS,
    TARGET_FLOAT_FIELDS,
    TARGET_SCHEMA_HASH,
)
from .io import (
    atomic_write_json,
    canonical_relative_path,
    file_identity,
    parse_canonical_float,
    read_canonical_json,
)

_CONFIG_FLOAT_FIELDS = frozenset(
    {"inclination_deg", "altitude_km", "max_isl_distance_km", "gcc_threshold", "node_failure_prob", "edge_failure_prob"}
)
_STEP_FLOAT_FIELDS = frozenset({"gcc_frac", "gcc_frac_original", "gcc_frac_surviving"})
_SUMMARY_FLOAT_FIELDS = frozenset(
    {
        "gcc_frac_min", "gcc_frac_mean", "gcc_frac_min_original", "gcc_frac_mean_original",
        "gcc_frac_min_surviving", "gcc_frac_mean_surviving", "partition_fraction",
        "max_partition_streak_fraction",
    }
)
_ARTIFACT_VERSIONS = {
    "satellite": SATELLITE_ARTIFACT_SCHEMA_VERSION,
    "g1": GROUND_DESIGN_SCHEMA_VERSION,
    "g2": GROUND_VISIBILITY_SCHEMA_VERSION,
    "g3": INTEGRATED_GRAPH_SCHEMA_VERSION,
    "g4_steps": GROUND_SERVICE_STEP_SCHEMA_VERSION,
    "g4_run": GROUND_SERVICE_RUN_SCHEMA_VERSION,
    "g5_realization": GROUND_FAILURE_REALIZATION_SCHEMA_VERSION,
    "g5_steps": GROUND_FAILURE_SERVICE_STEP_SCHEMA_VERSION,
    "g5_run": GROUND_FAILURE_SERVICE_RUN_SCHEMA_VERSION,
    "target": TARGET_ARTIFACT_SCHEMA_VERSION,
}


def _canonical_fields(source: Mapping[str, object], float_fields: frozenset[str]) -> dict[str, object]:
    return {
        name: canonical_float_string(value) if name in float_fields else value
        for name, value in source.items()
    }


def _parse_fields(source: Mapping[str, object], float_fields: frozenset[str]) -> dict[str, object]:
    result = dict(source)
    for name in float_fields:
        result[name] = parse_canonical_float(result[name], name)
    return result


def _reject_bundle(value: object) -> None:
    if isinstance(value, dict):
        if "contract_bundle_hash" in value:
            raise ValueError("contract_bundle_hash is forbidden in scientific payloads")
        for item in value.values():
            _reject_bundle(item)
    elif isinstance(value, list):
        for item in value:
            _reject_bundle(item)


def make_satellite_artifact(
    *,
    design: Mapping[str, Any],
    run: Mapping[str, Any],
    config: Tier1RolloutConfig,
    steps: Sequence[Tier1RolloutStep],
    summary: Tier1RolloutSummary,
    failure_realization: Tier1FailureRealization,
) -> dict[str, Any]:
    if config.config_hash() != run["expected_satellite_config_hash"]:
        raise ValueError("Actual satellite configuration hash differs from frozen expectation")
    ordered_steps = tuple(steps)
    if len(ordered_steps) != config.num_steps or tuple(step.t for step in ordered_steps) != tuple(range(config.num_steps)):
        raise ValueError("Satellite rollout steps are not complete and ordered")
    if summary.config_hash != config.config_hash() or summary.num_steps != len(ordered_steps):
        raise ValueError("Satellite rollout summary mismatch")
    payload: dict[str, Any] = {
        "actual_configuration_hash": config.config_hash(),
        "contract_spec_hash": CONTRACT_SPEC_HASH,
        "design_record_hash": design["design_record_hash"],
        "expected_configuration_hash": run["expected_satellite_config_hash"],
        "failed_edges": [list(edge) for edge in sorted(failure_realization.failed_edges)],
        "failed_nodes": sorted(failure_realization.failed_nodes),
        "failure_model": config.failure_model,
        "final_generation_satellite_artifact_schema_version": SATELLITE_ARTIFACT_SCHEMA_VERSION,
        "identity_domain": SATELLITE_ARTIFACT_DOMAIN,
        "identity_version": SATELLITE_ARTIFACT_IDENTITY_VERSION,
        "run_id": run["run_id"],
        "run_key": run["run_key"],
        "run_record_hash": run["run_record_hash"],
        "satellite_dataset_version": DATASET_VERSION,
        "satellite_rollout_schema_version": SCHEMA_VERSION,
        "summary": _canonical_fields(asdict(summary), _SUMMARY_FLOAT_FIELDS),
        "tier1_rollout_config": _canonical_fields(asdict(config), _CONFIG_FLOAT_FIELDS),
        "steps": [_canonical_fields(asdict(step), _STEP_FLOAT_FIELDS) for step in ordered_steps],
    }
    _reject_bundle(payload)
    result = dict(payload)
    result["satellite_artifact_hash"] = canonical_hash(payload)
    return result


def write_satellite_artifact(path: str | Path, artifact: dict[str, Any]) -> None:
    persisted = dict(artifact)
    artifact_hash = persisted.pop("satellite_artifact_hash", None)
    if artifact_hash != canonical_hash(persisted):
        raise ValueError("Satellite artifact self-hash mismatch")
    atomic_write_json(path, artifact)


def read_satellite_artifact(
    path: str | Path,
) -> tuple[dict[str, Any], Tier1RolloutConfig, tuple[Tier1RolloutStep, ...], Tier1RolloutSummary, Tier1FailureRealization]:
    artifact = read_canonical_json(path)
    payload = dict(artifact)
    artifact_hash = payload.pop("satellite_artifact_hash", None)
    if artifact_hash != canonical_hash(payload):
        raise ValueError("Satellite artifact self-hash mismatch")
    if payload.get("identity_domain") != SATELLITE_ARTIFACT_DOMAIN:
        raise ValueError("Pilot or unsupported satellite identity domain")
    if payload.get("final_generation_satellite_artifact_schema_version") != SATELLITE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported final satellite artifact schema")
    if set(payload["tier1_rollout_config"]) != {field.name for field in fields(Tier1RolloutConfig)}:
        raise ValueError("Persisted Tier1RolloutConfig fields mismatch")
    config = Tier1RolloutConfig(**_parse_fields(payload["tier1_rollout_config"], _CONFIG_FLOAT_FIELDS))
    steps = tuple(Tier1RolloutStep(**_parse_fields(value, _STEP_FLOAT_FIELDS)) for value in payload["steps"])
    summary = Tier1RolloutSummary(**_parse_fields(payload["summary"], _SUMMARY_FLOAT_FIELDS))
    failures = Tier1FailureRealization(
        failed_nodes=set(payload["failed_nodes"]),
        failed_edges={tuple(edge) for edge in payload["failed_edges"]},
    )
    expected = make_satellite_artifact(
        design={"design_record_hash": payload["design_record_hash"]},
        run={
            "expected_satellite_config_hash": payload["expected_configuration_hash"],
            "run_id": payload["run_id"],
            "run_key": payload["run_key"],
            "run_record_hash": payload["run_record_hash"],
        },
        config=config,
        steps=steps,
        summary=summary,
        failure_realization=failures,
    )
    if artifact != expected:
        raise ValueError("Satellite artifact is not canonical")
    return artifact, config, steps, summary, failures


def make_target_artifact(
    *, design: Mapping[str, Any], run: Mapping[str, Any], g5_summary: object
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract_spec_hash": CONTRACT_SPEC_HASH,
        "design_record_hash": design["design_record_hash"],
        "identity_domain": TARGET_ARTIFACT_DOMAIN,
        "identity_version": TARGET_ARTIFACT_IDENTITY_VERSION,
        "run_id": run["run_id"],
        "run_key": run["run_key"],
        "run_record_hash": run["run_record_hash"],
        "target_artifact_schema_version": TARGET_ARTIFACT_SCHEMA_VERSION,
        "target_schema_hash": TARGET_SCHEMA_HASH,
    }
    for name in TARGET_FIELDS:
        value = getattr(g5_summary, name)
        if name in TARGET_BOOLEAN_FIELDS:
            if type(value) is not bool:
                raise TypeError(f"Target {name} must be a Boolean")
            payload[name] = value
        else:
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Target {name} must be a finite unit-interval float")
            payload[name] = canonical_float_string(value)
    _reject_bundle(payload)
    result = dict(payload)
    result["target_artifact_hash"] = canonical_hash(payload)
    return result


def validate_target_artifact(value: dict[str, Any]) -> None:
    expected = {
        "contract_spec_hash", "design_record_hash", "identity_domain", "identity_version",
        "run_id", "run_key", "run_record_hash", "target_artifact_schema_version",
        "target_schema_hash", "target_artifact_hash", *TARGET_FIELDS,
    }
    if set(value) != expected:
        raise ValueError("Target artifact fields mismatch")
    for name in TARGET_BOOLEAN_FIELDS:
        if type(value[name]) is not bool:
            raise TypeError(f"Target {name} must be a Boolean")
    for name in TARGET_FLOAT_FIELDS:
        parsed = parse_canonical_float(value[name], name)
        if not 0.0 <= parsed <= 1.0:
            raise ValueError(f"Target {name} is outside [0, 1]")
    payload = dict(value)
    persisted = payload.pop("target_artifact_hash")
    if persisted != canonical_hash(payload):
        raise ValueError("Target artifact self-hash mismatch")
    _reject_bundle(value)


def make_scientific_inventory(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    records: list[dict[str, Any]] = []
    for key in SCIENTIFIC_FILE_KEYS:
        relative = canonical_relative_path(RUN_FILES[key])
        length, digest = file_identity(root / relative)
        records.append(
            {
                "artifact_role": ARTIFACT_ROLES[key],
                "byte_length": length,
                "path": relative,
                "sha256": digest,
                "version": str(_ARTIFACT_VERSIONS[key]),
            }
        )
    records.sort(key=lambda record: record["path"])
    payload = {
        "artifacts": records,
        "contract_spec_hash": CONTRACT_SPEC_HASH,
        "identity_domain": SCIENTIFIC_INVENTORY_DOMAIN,
        "identity_version": SCIENTIFIC_INVENTORY_IDENTITY_VERSION,
        "scientific_inventory_schema_version": SCIENTIFIC_INVENTORY_SCHEMA_VERSION,
    }
    result = dict(payload)
    result["scientific_inventory_hash"] = canonical_hash(payload)
    return result


def validate_scientific_inventory(run_root: str | Path, inventory: dict[str, Any]) -> None:
    if inventory != make_scientific_inventory(run_root):
        raise ValueError("Scientific artifact inventory mismatch")
    _reject_bundle(inventory)


def make_run_result(
    *, design: Mapping[str, Any], run: Mapping[str, Any], inventory: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "contract_spec_hash": CONTRACT_SPEC_HASH,
        "design_record_hash": design["design_record_hash"],
        "identity_domain": RUN_RESULT_DOMAIN,
        "identity_version": RUN_RESULT_IDENTITY_VERSION,
        "run_id": run["run_id"],
        "run_key": run["run_key"],
        "run_record_hash": run["run_record_hash"],
        "run_result_schema_version": RUN_RESULT_SCHEMA_VERSION,
        "scientific_inventory_hash": inventory["scientific_inventory_hash"],
        "target_artifact_hash": target["target_artifact_hash"],
    }
    result = dict(payload)
    result["run_result_hash"] = canonical_hash(payload)
    return result


def validate_run_result(value: dict[str, Any]) -> None:
    payload = dict(value)
    persisted = payload.pop("run_result_hash", None)
    if persisted != canonical_hash(payload):
        raise ValueError("Final run-result self-hash mismatch")
    _reject_bundle(value)
