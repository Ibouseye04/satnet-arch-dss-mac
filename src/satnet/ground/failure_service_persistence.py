from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from satnet.ground.canonical import (
    canonical_float_string,
    canonical_json,
    canonical_utc_timestamp,
)
from satnet.ground.catalog import GroundStationCatalog
from satnet.ground.failure_policy import GroundFailurePolicy
from satnet.ground.failure_realization import (
    GROUND_FAILURE_REALIZATION_SCHEMA_VERSION,
    GroundFailureRealization,
    GroundFailureRealizationRecord,
    _realization_record_hash,
    create_ground_failure_realization_record,
    validate_ground_failure_realization_record_context,
)
from satnet.ground.failure_service_aggregation import (
    FailureAdjustedGroundServiceRunRecord,
    FailureAdjustedGroundServiceRunSummary,
    _SUMMARY_FLOAT_FIELDS,
    _run_record_hash,
    create_failure_adjusted_run_record,
)
from satnet.ground.failure_service_metrics import (
    FailureAdjustedGroundServiceStepMetrics,
    FailureAdjustedGroundServiceStepRecord,
    _FLOAT_FIELDS,
    _step_record_hash,
    create_failure_adjusted_step_record,
)
from satnet.ground.integrated_persistence import IntegratedGroundGraphRecord
from satnet.ground.persistence import GroundRunDesignRecord
from satnet.ground.service_aggregation import GroundServiceStepRecord
from satnet.ground.service_persistence import (
    GroundServiceRunRecord,
    replay_ground_service_records,
)
from satnet.ground.service_policy import GroundServicePolicy
from satnet.ground.visibility import GroundVisibilityPolicy
from satnet.ground.visibility_persistence import GroundVisibilityRecord
from satnet.simulation.tier1_rollout import Tier1FailureRealization, Tier1RolloutConfig

_REALIZATION_RECORD_FIELDS = frozenset(field.name for field in fields(GroundFailureRealizationRecord))
_REALIZATION_FIELDS = frozenset(field.name for field in fields(GroundFailureRealization))
_STEP_RECORD_FIELDS = frozenset(field.name for field in fields(FailureAdjustedGroundServiceStepRecord))
_STEP_FIELDS = frozenset(field.name for field in fields(FailureAdjustedGroundServiceStepMetrics))
_RUN_RECORD_FIELDS = frozenset(field.name for field in fields(FailureAdjustedGroundServiceRunRecord))
_SUMMARY_FIELDS = frozenset(field.name for field in fields(FailureAdjustedGroundServiceRunSummary))


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key '{key}'")
        result[key] = value
    return result


def _require_fields(
    value: object,
    expected: frozenset[str],
    description: str,
    line_number: int,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"Line {line_number}: {description} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(
            f"Line {line_number}: {description} fields invalid; "
            f"missing={missing}, unknown={unknown}"
        )
    return value


def _parse_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a canonical UTC string")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{field_name} must use canonical UTC format") from exc
    if canonical_utc_timestamp(parsed) != value:
        raise ValueError(f"{field_name} is not canonical UTC")
    return parsed


def _parse_float(value: object, field_name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a canonical float string")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid float string") from exc
    if canonical_float_string(parsed) != value:
        raise ValueError(f"{field_name} is not a canonical float string")
    if parsed == 0.0 and value.startswith("-"):
        raise ValueError(f"{field_name} must use normalized positive zero")
    return parsed


def _read_jsonl(path: str | Path) -> list[object]:
    manifest_path = Path(path)
    if manifest_path.suffix != ".jsonl":
        raise ValueError("Canonical G5 manifests must use .jsonl")
    text = manifest_path.read_text(encoding="utf-8")
    if not text:
        raise ValueError("G5 manifest is empty")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    if not lines or any(line == "" for line in lines):
        raise ValueError("G5 manifest contains an empty line")
    result: list[object] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            result.append(json.loads(line, object_pairs_hook=_pairs_without_duplicates))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Line {line_number}: malformed JSON: {exc}") from exc
    return result


def _realization_manifest(realization: GroundFailureRealization) -> dict[str, object]:
    return {
        field.name: list(value) if isinstance(value, tuple) else value
        for field in fields(GroundFailureRealization)
        for value in (getattr(realization, field.name),)
    }


def _realization_record_manifest(record: GroundFailureRealizationRecord) -> dict[str, object]:
    return {
        "ground_failure_realization_schema_version": record.ground_failure_realization_schema_version,
        "realization": _realization_manifest(record.realization),
        "record_hash": record.record_hash,
        "run_id": record.run_id,
    }


def _step_manifest(metrics: FailureAdjustedGroundServiceStepMetrics) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in fields(FailureAdjustedGroundServiceStepMetrics):
        value = getattr(metrics, field.name)
        if field.name == "timestamp_utc":
            result[field.name] = canonical_utc_timestamp(value)
        elif field.name in _FLOAT_FIELDS:
            result[field.name] = None if value is None else canonical_float_string(value)
        elif isinstance(value, tuple):
            result[field.name] = list(value)
        else:
            result[field.name] = value
    return result


def _step_record_manifest(record: FailureAdjustedGroundServiceStepRecord) -> dict[str, object]:
    return {
        "ground_failure_service_step_schema_version": record.ground_failure_service_step_schema_version,
        "metrics": _step_manifest(record.metrics),
        "record_hash": record.record_hash,
        "run_id": record.run_id,
    }


def _summary_manifest(summary: FailureAdjustedGroundServiceRunSummary) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in fields(FailureAdjustedGroundServiceRunSummary):
        value = getattr(summary, field.name)
        if field.name in _SUMMARY_FLOAT_FIELDS:
            result[field.name] = None if value is None else canonical_float_string(value)
        elif isinstance(value, tuple):
            result[field.name] = list(value)
        else:
            result[field.name] = value
    return result


def _run_record_manifest(record: FailureAdjustedGroundServiceRunRecord) -> dict[str, object]:
    return {
        "ground_failure_service_run_schema_version": record.ground_failure_service_run_schema_version,
        "record_hash": record.record_hash,
        "run_id": record.run_id,
        "summary": _summary_manifest(record.summary),
    }


def _realization_from_object(value: object, line_number: int) -> GroundFailureRealization:
    source = _require_fields(value, _REALIZATION_FIELDS, "realization", line_number)
    converted = dict(source)
    for name in ("selected_station_ids", "failed_station_ids", "operational_station_ids"):
        if not isinstance(source[name], list):
            raise TypeError(f"Line {line_number}: {name} must be an array")
        converted[name] = tuple(source[name])
    return GroundFailureRealization(**converted)


def _realization_record_from_object(
    value: object, line_number: int
) -> GroundFailureRealizationRecord:
    source = _require_fields(
        value, _REALIZATION_RECORD_FIELDS, "realization record", line_number
    )
    if type(source["run_id"]) is not int:
        raise TypeError(f"Line {line_number}: run_id must be an integer")
    return GroundFailureRealizationRecord(
        ground_failure_realization_schema_version=source[
            "ground_failure_realization_schema_version"
        ],
        run_id=source["run_id"],
        realization=_realization_from_object(source["realization"], line_number),
        record_hash=source["record_hash"],
    )


def _step_from_object(
    value: object, line_number: int
) -> FailureAdjustedGroundServiceStepMetrics:
    source = _require_fields(value, _STEP_FIELDS, "adjusted step metrics", line_number)
    converted = dict(source)
    converted["timestamp_utc"] = _parse_timestamp(source["timestamp_utc"], "timestamp_utc")
    for name in _FLOAT_FIELDS:
        converted[name] = _parse_float(
            source[name],
            name,
            optional=name.startswith("failure_adjusted_")
            and name.endswith("_service_fraction")
            and any(class_name in name for class_name in ("civilian", "government", "military")),
        )
    for name in (
        "satellite_gcc_ids",
        "failed_ground_station_ids",
        "operational_ground_station_ids",
        "failure_adjusted_serviced_ground_station_ids",
        "failure_adjusted_unserviced_ground_station_ids",
    ):
        if not isinstance(source[name], list):
            raise TypeError(f"Line {line_number}: {name} must be an array")
        converted[name] = tuple(source[name])
    return FailureAdjustedGroundServiceStepMetrics(**converted)


def _step_record_from_object(
    value: object, line_number: int
) -> FailureAdjustedGroundServiceStepRecord:
    source = _require_fields(value, _STEP_RECORD_FIELDS, "adjusted step record", line_number)
    if type(source["run_id"]) is not int:
        raise TypeError(f"Line {line_number}: run_id must be an integer")
    return FailureAdjustedGroundServiceStepRecord(
        ground_failure_service_step_schema_version=source[
            "ground_failure_service_step_schema_version"
        ],
        run_id=source["run_id"],
        metrics=_step_from_object(source["metrics"], line_number),
        record_hash=source["record_hash"],
    )


def _summary_from_object(
    value: object, line_number: int
) -> FailureAdjustedGroundServiceRunSummary:
    source = _require_fields(value, _SUMMARY_FIELDS, "adjusted run summary", line_number)
    converted = dict(source)
    for name in _SUMMARY_FLOAT_FIELDS:
        converted[name] = _parse_float(
            source[name],
            name,
            optional=name.startswith(
                (
                    "failure_adjusted_civilian_",
                    "failure_adjusted_government_",
                    "failure_adjusted_military_",
                )
            ),
        )
    for name in ("failed_ground_station_ids", "operational_ground_station_ids"):
        if not isinstance(source[name], list):
            raise TypeError(f"Line {line_number}: {name} must be an array")
        converted[name] = tuple(source[name])
    return FailureAdjustedGroundServiceRunSummary(**converted)


def _run_record_from_object(
    value: object, line_number: int
) -> FailureAdjustedGroundServiceRunRecord:
    source = _require_fields(value, _RUN_RECORD_FIELDS, "adjusted run record", line_number)
    if type(source["run_id"]) is not int:
        raise TypeError(f"Line {line_number}: run_id must be an integer")
    return FailureAdjustedGroundServiceRunRecord(
        ground_failure_service_run_schema_version=source[
            "ground_failure_service_run_schema_version"
        ],
        run_id=source["run_id"],
        summary=_summary_from_object(source["summary"], line_number),
        record_hash=source["record_hash"],
    )


def read_ground_failure_realization_manifest(
    path: str | Path,
) -> tuple[GroundFailureRealizationRecord, ...]:
    records = tuple(
        _realization_record_from_object(value, line_number)
        for line_number, value in enumerate(_read_jsonl(path), start=1)
    )
    run_ids = [record.run_id for record in records]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate ground-failure realization run IDs")
    return tuple(sorted(records, key=lambda record: record.run_id))


def read_ground_failure_service_step_manifest(
    path: str | Path,
) -> tuple[FailureAdjustedGroundServiceStepRecord, ...]:
    records = tuple(
        _step_record_from_object(value, line_number)
        for line_number, value in enumerate(_read_jsonl(path), start=1)
    )
    keys = [(record.run_id, record.metrics.timestep_index) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate G5 step run/timestep keys")
    return tuple(sorted(records, key=lambda record: (record.run_id, record.metrics.timestep_index)))


def read_ground_failure_service_run_manifest(
    path: str | Path,
) -> tuple[FailureAdjustedGroundServiceRunRecord, ...]:
    records = tuple(
        _run_record_from_object(value, line_number)
        for line_number, value in enumerate(_read_jsonl(path), start=1)
    )
    run_ids = [record.run_id for record in records]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate G5 run IDs")
    return tuple(sorted(records, key=lambda record: record.run_id))


def _atomic_write_jsonl(
    *, path: str | Path, objects: Sequence[dict[str, object]], overwrite: bool
) -> None:
    manifest_path = Path(path)
    if manifest_path.suffix != ".jsonl":
        raise ValueError("Canonical G5 manifests must use .jsonl")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a Boolean")
    if not objects:
        raise ValueError("G5 manifest requires at least one record")
    serialized = [canonical_json(value) for value in objects]
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"G5 manifest already exists: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(serialized) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"G5 manifest already exists: {manifest_path}")
        os.replace(temporary_path, manifest_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_ground_failure_realization_manifest(
    records: Sequence[GroundFailureRealizationRecord],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not records or any(not isinstance(record, GroundFailureRealizationRecord) for record in records):
        raise ValueError("Realization manifest requires realization records")
    run_ids = [record.run_id for record in records]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate ground-failure realization run IDs")
    ordered = sorted(records, key=lambda record: record.run_id)
    _atomic_write_jsonl(
        path=path,
        objects=[_realization_record_manifest(record) for record in ordered],
        overwrite=overwrite,
    )


def write_ground_failure_service_step_manifest(
    records: Sequence[FailureAdjustedGroundServiceStepRecord],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not records or any(
        not isinstance(record, FailureAdjustedGroundServiceStepRecord)
        for record in records
    ):
        raise ValueError("G5 step manifest requires adjusted step records")
    keys = [(record.run_id, record.metrics.timestep_index) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate G5 step run/timestep keys")
    ordered = sorted(records, key=lambda record: (record.run_id, record.metrics.timestep_index))
    _atomic_write_jsonl(
        path=path,
        objects=[_step_record_manifest(record) for record in ordered],
        overwrite=overwrite,
    )


def write_ground_failure_service_run_manifest(
    records: Sequence[FailureAdjustedGroundServiceRunRecord],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not records or any(
        not isinstance(record, FailureAdjustedGroundServiceRunRecord)
        for record in records
    ):
        raise ValueError("G5 run manifest requires adjusted run records")
    run_ids = [record.run_id for record in records]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate G5 run IDs")
    ordered = sorted(records, key=lambda record: record.run_id)
    _atomic_write_jsonl(
        path=path,
        objects=[_run_record_manifest(record) for record in ordered],
        overwrite=overwrite,
    )


def generate_verified_ground_failure_service_records(
    *,
    satellite_config: Tier1RolloutConfig,
    satellite_failure_realization: Tier1FailureRealization,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    visibility_policy: GroundVisibilityPolicy,
    visibility_records: Sequence[GroundVisibilityRecord],
    integrated_records: Sequence[IntegratedGroundGraphRecord],
    ground_service_policy: GroundServicePolicy,
    g4_step_records: Sequence[GroundServiceStepRecord],
    g4_run_record: GroundServiceRunRecord,
    ground_failure_policy: GroundFailurePolicy,
    ground_failure_seed: int | None = None,
    persisted_realization_record: GroundFailureRealizationRecord | None = None,
) -> tuple[
    GroundFailureRealizationRecord,
    tuple[FailureAdjustedGroundServiceStepRecord, ...],
    FailureAdjustedGroundServiceRunRecord,
]:
    if not isinstance(ground_failure_policy, GroundFailurePolicy):
        raise TypeError("ground_failure_policy must be GroundFailurePolicy")
    generation_mode = ground_failure_seed is not None
    replay_mode = persisted_realization_record is not None
    if generation_mode == replay_mode:
        raise ValueError(
            "Exactly one of ground_failure_seed or persisted_realization_record is required"
        )
    verified_g4_steps, verified_g4_run = replay_ground_service_records(
        satellite_config=satellite_config,
        failure_realization=satellite_failure_realization,
        ground_design=ground_design,
        catalog=catalog,
        visibility_policy=visibility_policy,
        visibility_records=visibility_records,
        integrated_records=integrated_records,
        service_policy=ground_service_policy,
        step_records=g4_step_records,
        run_records=(g4_run_record,),
    )
    if generation_mode:
        realization_record = create_ground_failure_realization_record(
            run_id=ground_design.run_id,
            ground_design=ground_design,
            catalog=catalog,
            policy=ground_failure_policy,
            ground_failure_seed=ground_failure_seed,
        )
    else:
        if not isinstance(persisted_realization_record, GroundFailureRealizationRecord):
            raise TypeError(
                "persisted_realization_record must be GroundFailureRealizationRecord"
            )
        validate_ground_failure_realization_record_context(
            record=persisted_realization_record,
            ground_design=ground_design,
            catalog=catalog,
            policy=ground_failure_policy,
        )
        realization_record = create_ground_failure_realization_record(
            run_id=ground_design.run_id,
            ground_design=ground_design,
            catalog=catalog,
            policy=ground_failure_policy,
            ground_failure_seed=persisted_realization_record.realization.ground_failure_seed,
        )
        if realization_record != persisted_realization_record:
            raise ValueError("Persisted ground-failure realization replay mismatch")
    adjusted_steps = tuple(
        create_failure_adjusted_step_record(
            run_id=ground_design.run_id,
            verified_g4_step_record=record,
            ground_design=ground_design,
            catalog=catalog,
            policy=ground_failure_policy,
            realization_record=realization_record,
            ground_service_policy=ground_service_policy,
        )
        for record in verified_g4_steps
    )
    adjusted_run = create_failure_adjusted_run_record(
        verified_g4_step_records=verified_g4_steps,
        verified_g4_run_record=verified_g4_run,
        g5_step_records=adjusted_steps,
        realization_record=realization_record,
    )
    return realization_record, adjusted_steps, adjusted_run


def persist_verified_ground_failure_service_evidence(
    *,
    satellite_config: Tier1RolloutConfig,
    satellite_failure_realization: Tier1FailureRealization,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    visibility_policy: GroundVisibilityPolicy,
    visibility_records: Sequence[GroundVisibilityRecord],
    integrated_records: Sequence[IntegratedGroundGraphRecord],
    ground_service_policy: GroundServicePolicy,
    g4_step_records: Sequence[GroundServiceStepRecord],
    g4_run_record: GroundServiceRunRecord,
    ground_failure_policy: GroundFailurePolicy,
    ground_failure_seed: int,
    realization_path: str | Path,
    step_path: str | Path,
    run_path: str | Path,
    overwrite: bool = False,
) -> tuple[
    GroundFailureRealizationRecord,
    tuple[FailureAdjustedGroundServiceStepRecord, ...],
    FailureAdjustedGroundServiceRunRecord,
]:
    evidence = generate_verified_ground_failure_service_records(
        satellite_config=satellite_config,
        satellite_failure_realization=satellite_failure_realization,
        ground_design=ground_design,
        catalog=catalog,
        visibility_policy=visibility_policy,
        visibility_records=visibility_records,
        integrated_records=integrated_records,
        ground_service_policy=ground_service_policy,
        g4_step_records=g4_step_records,
        g4_run_record=g4_run_record,
        ground_failure_policy=ground_failure_policy,
        ground_failure_seed=ground_failure_seed,
    )
    realization_record, adjusted_steps, adjusted_run = evidence
    write_ground_failure_realization_manifest(
        (realization_record,), realization_path, overwrite=overwrite
    )
    write_ground_failure_service_step_manifest(
        adjusted_steps, step_path, overwrite=overwrite
    )
    write_ground_failure_service_run_manifest(
        (adjusted_run,), run_path, overwrite=overwrite
    )
    return evidence


def replay_ground_failure_service_records(
    *,
    satellite_config: Tier1RolloutConfig,
    satellite_failure_realization: Tier1FailureRealization,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    visibility_policy: GroundVisibilityPolicy,
    visibility_records: Sequence[GroundVisibilityRecord],
    integrated_records: Sequence[IntegratedGroundGraphRecord],
    ground_service_policy: GroundServicePolicy,
    g4_step_records: Sequence[GroundServiceStepRecord],
    g4_run_record: GroundServiceRunRecord,
    ground_failure_policy: GroundFailurePolicy,
    realization_records: Sequence[GroundFailureRealizationRecord],
    g5_step_records: Sequence[FailureAdjustedGroundServiceStepRecord],
    g5_run_records: Sequence[FailureAdjustedGroundServiceRunRecord],
) -> tuple[
    GroundFailureRealizationRecord,
    tuple[FailureAdjustedGroundServiceStepRecord, ...],
    FailureAdjustedGroundServiceRunRecord,
]:
    if len(realization_records) != 1:
        raise ValueError("G5 replay requires exactly one failure realization record")
    if len(g5_run_records) != 1:
        raise ValueError("G5 replay requires exactly one adjusted run record")
    if not g5_step_records:
        raise ValueError("G5 replay requires adjusted step records")
    keys = [(record.run_id, record.metrics.timestep_index) for record in g5_step_records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate G5 replay step keys")
    if keys != sorted(keys):
        raise ValueError("G5 replay steps must use canonical order")
    expected = generate_verified_ground_failure_service_records(
        satellite_config=satellite_config,
        satellite_failure_realization=satellite_failure_realization,
        ground_design=ground_design,
        catalog=catalog,
        visibility_policy=visibility_policy,
        visibility_records=visibility_records,
        integrated_records=integrated_records,
        ground_service_policy=ground_service_policy,
        g4_step_records=g4_step_records,
        g4_run_record=g4_run_record,
        ground_failure_policy=ground_failure_policy,
        persisted_realization_record=realization_records[0],
    )
    expected_realization, expected_steps, expected_run = expected
    expected_keys = [
        (record.run_id, record.metrics.timestep_index) for record in expected_steps
    ]
    actual_by_key = {
        (record.run_id, record.metrics.timestep_index): record
        for record in g5_step_records
    }
    if set(actual_by_key) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(actual_by_key))
        extra = sorted(set(actual_by_key) - set(expected_keys))
        raise ValueError(f"G5 step keys mismatch; missing={missing}, extra={extra}")
    if realization_records[0] != expected_realization:
        raise ValueError("G5 realization replay mismatch")
    for record in expected_steps:
        key = (record.run_id, record.metrics.timestep_index)
        if actual_by_key[key] != record:
            raise ValueError(f"G5 step replay mismatch at {key}")
    if g5_run_records[0] != expected_run:
        raise ValueError("G5 run replay mismatch")
    return expected
