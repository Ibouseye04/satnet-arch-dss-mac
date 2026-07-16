from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from satnet.ground.canonical import canonical_float_string, canonical_hash, canonical_json, canonical_utc_timestamp
from satnet.ground.catalog import GroundStationCatalog
from satnet.ground.integrated_persistence import (
    IntegratedGroundGraphRecord,
    replay_integrated_graph_records,
)
from satnet.ground.persistence import GroundRunDesignRecord
from satnet.ground.service_aggregation import (
    GROUND_SERVICE_RUN_SCHEMA_VERSION,
    GROUND_SERVICE_STEP_SCHEMA_VERSION,
    GroundServiceRunSummary,
    GroundServiceStepRecord,
    _step_record_hash,
    make_ground_service_step_record,
    summarize_ground_service_run,
    validate_ground_service_run_summary,
)
from satnet.ground.service_metrics import GroundServiceStepMetrics, compute_ground_service_step
from satnet.ground.service_policy import GroundServicePolicy
from satnet.ground.visibility import GroundVisibilityPolicy
from satnet.ground.visibility_persistence import GroundVisibilityRecord
from satnet.simulation.tier1_rollout import Tier1FailureRealization, Tier1RolloutConfig

GROUND_SERVICE_RUN_RECORD_IDENTITY_DOMAIN = "satnet_ground_service_run_record"
GROUND_SERVICE_RUN_RECORD_IDENTITY_VERSION = "1"
STEP_RECORD_FIELDS = frozenset(
    {"ground_service_step_schema_version", "run_id", "metrics", "record_hash"}
)
RUN_RECORD_FIELDS = frozenset(
    {"ground_service_run_schema_version", "run_id", "summary", "record_hash"}
)
STEP_METRICS_FIELDS = frozenset(field.name for field in fields(GroundServiceStepMetrics))
RUN_SUMMARY_FIELDS = frozenset(field.name for field in fields(GroundServiceRunSummary))
STEP_FLOAT_FIELDS = frozenset(
    {
        "civilian_service_fraction",
        "government_service_fraction",
        "military_service_fraction",
        "space_gcc_fraction_original",
        "space_gcc_fraction_surviving",
        "ground_service_fraction",
        "overall_service_fraction",
    }
)
SUMMARY_FLOAT_FIELDS = frozenset(
    field.name
    for field in fields(GroundServiceRunSummary)
    if field.name.endswith("_min") or field.name.endswith("_mean")
)


def _run_record_hash(*, run_id: int, summary: GroundServiceRunSummary) -> str:
    return canonical_hash(
        {
            "ground_service_run_schema_version": GROUND_SERVICE_RUN_SCHEMA_VERSION,
            "identity_domain": GROUND_SERVICE_RUN_RECORD_IDENTITY_DOMAIN,
            "identity_version": GROUND_SERVICE_RUN_RECORD_IDENTITY_VERSION,
            "run_id": run_id,
            "run_summary_hash": summary.run_summary_hash,
        }
    )


@dataclass(frozen=True)
class GroundServiceRunRecord:
    ground_service_run_schema_version: str
    run_id: int
    summary: GroundServiceRunSummary
    record_hash: str

    def __post_init__(self) -> None:
        if self.ground_service_run_schema_version != GROUND_SERVICE_RUN_SCHEMA_VERSION:
            raise ValueError("Unsupported ground_service_run_schema_version")
        if type(self.run_id) is not int or self.run_id < 0:
            raise TypeError("run_id must be a nonnegative integer")
        if not isinstance(self.summary, GroundServiceRunSummary):
            raise TypeError("summary must be GroundServiceRunSummary")
        if self.summary.run_id != self.run_id:
            raise ValueError("Run-record run ID does not match summary run ID")
        expected = _run_record_hash(run_id=self.run_id, summary=self.summary)
        if self.record_hash != expected:
            raise ValueError("record_hash does not match canonical ground-service run record")


def make_ground_service_run_record(
    *,
    summary: GroundServiceRunSummary,
    step_records: Sequence[GroundServiceStepRecord],
) -> GroundServiceRunRecord:
    if not isinstance(summary, GroundServiceRunSummary):
        raise TypeError("summary must be GroundServiceRunSummary")
    validate_ground_service_run_summary(summary=summary, step_records=step_records)
    return GroundServiceRunRecord(
        ground_service_run_schema_version=GROUND_SERVICE_RUN_SCHEMA_VERSION,
        run_id=summary.run_id,
        summary=summary,
        record_hash=_run_record_hash(run_id=summary.run_id, summary=summary),
    )


def _metrics_manifest(metrics: GroundServiceStepMetrics) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in fields(GroundServiceStepMetrics):
        value = getattr(metrics, field.name)
        if field.name == "timestamp_utc":
            result[field.name] = canonical_utc_timestamp(value)
        elif field.name in STEP_FLOAT_FIELDS:
            result[field.name] = None if value is None else canonical_float_string(value)
        elif isinstance(value, tuple):
            result[field.name] = list(value)
        else:
            result[field.name] = value
    return result


def _summary_manifest(summary: GroundServiceRunSummary) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in fields(GroundServiceRunSummary):
        value = getattr(summary, field.name)
        if field.name in SUMMARY_FLOAT_FIELDS:
            result[field.name] = None if value is None else canonical_float_string(value)
        else:
            result[field.name] = value
    return result


def _step_record_manifest(record: GroundServiceStepRecord) -> dict[str, object]:
    return {
        "ground_service_step_schema_version": record.ground_service_step_schema_version,
        "metrics": _metrics_manifest(record.metrics),
        "record_hash": record.record_hash,
        "run_id": record.run_id,
    }


def _run_record_manifest(record: GroundServiceRunRecord) -> dict[str, object]:
    return {
        "ground_service_run_schema_version": record.ground_service_run_schema_version,
        "record_hash": record.record_hash,
        "run_id": record.run_id,
        "summary": _summary_manifest(record.summary),
    }


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key '{key}'")
        result[key] = value
    return result


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
    return parsed


def _require_exact_fields(
    value: object, expected: frozenset[str], description: str, line_number: int
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


def _metrics_from_object(value: object, line_number: int) -> GroundServiceStepMetrics:
    source = _require_exact_fields(value, STEP_METRICS_FIELDS, "step metrics", line_number)
    converted = dict(source)
    converted["timestamp_utc"] = _parse_timestamp(source["timestamp_utc"], "timestamp_utc")
    for field_name in STEP_FLOAT_FIELDS:
        converted[field_name] = _parse_float(
            source[field_name],
            field_name,
            optional=field_name.endswith("_service_fraction")
            and field_name.split("_", 1)[0] in {"civilian", "government", "military"},
        )
    for field_name in (
        "satellite_gcc_ids",
        "serviced_ground_station_ids",
        "unserviced_ground_station_ids",
    ):
        if not isinstance(source[field_name], list):
            raise TypeError(f"{field_name} must be an array")
        converted[field_name] = tuple(source[field_name])
    try:
        return GroundServiceStepMetrics(**converted)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"Line {line_number}: {exc}") from exc


def _summary_from_object(value: object, line_number: int) -> GroundServiceRunSummary:
    source = _require_exact_fields(value, RUN_SUMMARY_FIELDS, "run summary", line_number)
    converted = dict(source)
    for field_name in SUMMARY_FLOAT_FIELDS:
        converted[field_name] = _parse_float(
            source[field_name],
            field_name,
            optional=field_name.startswith(("civilian_", "government_", "military_")),
        )
    try:
        return GroundServiceRunSummary(**converted)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"Line {line_number}: {exc}") from exc


def _step_record_from_object(value: object, line_number: int) -> GroundServiceStepRecord:
    source = _require_exact_fields(value, STEP_RECORD_FIELDS, "step record", line_number)
    if type(source["run_id"]) is not int:
        raise TypeError(f"Line {line_number}: run_id must be an integer")
    try:
        return GroundServiceStepRecord(
            ground_service_step_schema_version=source["ground_service_step_schema_version"],
            run_id=source["run_id"],
            metrics=_metrics_from_object(source["metrics"], line_number),
            record_hash=source["record_hash"],
        )
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"Line {line_number}: {exc}") from exc


def _run_record_from_object(value: object, line_number: int) -> GroundServiceRunRecord:
    source = _require_exact_fields(value, RUN_RECORD_FIELDS, "run record", line_number)
    if type(source["run_id"]) is not int:
        raise TypeError(f"Line {line_number}: run_id must be an integer")
    try:
        return GroundServiceRunRecord(
            ground_service_run_schema_version=source["ground_service_run_schema_version"],
            run_id=source["run_id"],
            summary=_summary_from_object(source["summary"], line_number),
            record_hash=source["record_hash"],
        )
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"Line {line_number}: {exc}") from exc


def _read_jsonl(path: str | Path) -> list[object]:
    manifest_path = Path(path)
    if manifest_path.suffix != ".jsonl":
        raise ValueError("Canonical ground-service manifest must use .jsonl")
    text = manifest_path.read_text(encoding="utf-8")
    if not text:
        raise ValueError("Ground-service manifest is empty")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    if not lines or any(line == "" for line in lines):
        raise ValueError("Ground-service manifest contains an empty line")
    result: list[object] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            result.append(json.loads(line, object_pairs_hook=_pairs_without_duplicates))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Line {line_number}: malformed JSON: {exc}") from exc
    return result


def read_ground_service_step_manifest(
    path: str | Path,
) -> tuple[GroundServiceStepRecord, ...]:
    records = tuple(
        _step_record_from_object(value, line_number)
        for line_number, value in enumerate(_read_jsonl(path), start=1)
    )
    keys = [(record.run_id, record.metrics.timestep_index) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate ground-service step run/timestep keys")
    return tuple(sorted(records, key=lambda record: (record.run_id, record.metrics.timestep_index)))


def read_ground_service_run_manifest(
    path: str | Path,
) -> tuple[GroundServiceRunRecord, ...]:
    records = tuple(
        _run_record_from_object(value, line_number)
        for line_number, value in enumerate(_read_jsonl(path), start=1)
    )
    run_ids = [record.run_id for record in records]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate ground-service run IDs")
    return tuple(sorted(records, key=lambda record: record.run_id))


def _atomic_write_jsonl(
    *, path: str | Path, objects: Sequence[dict[str, object]], overwrite: bool
) -> None:
    manifest_path = Path(path)
    if manifest_path.suffix != ".jsonl":
        raise ValueError("Canonical ground-service manifest must use .jsonl")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a Boolean")
    if not objects:
        raise ValueError("Ground-service manifest requires at least one record")
    serialized = [canonical_json(value) for value in objects]
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Ground-service manifest already exists: {manifest_path}")
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
            raise FileExistsError(f"Ground-service manifest already exists: {manifest_path}")
        os.replace(temporary_path, manifest_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_ground_service_step_manifest(
    records: Sequence[GroundServiceStepRecord],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not records or any(not isinstance(record, GroundServiceStepRecord) for record in records):
        raise ValueError("Step manifest requires GroundServiceStepRecord values")
    keys = [(record.run_id, record.metrics.timestep_index) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate ground-service step run/timestep keys")
    ordered = sorted(records, key=lambda record: (record.run_id, record.metrics.timestep_index))
    _atomic_write_jsonl(
        path=path,
        objects=[_step_record_manifest(record) for record in ordered],
        overwrite=overwrite,
    )


def write_ground_service_run_manifest(
    records: Sequence[GroundServiceRunRecord],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not records or any(not isinstance(record, GroundServiceRunRecord) for record in records):
        raise ValueError("Run manifest requires GroundServiceRunRecord values")
    run_ids = [record.run_id for record in records]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate ground-service run IDs")
    ordered = sorted(records, key=lambda record: record.run_id)
    _atomic_write_jsonl(
        path=path,
        objects=[_run_record_manifest(record) for record in ordered],
        overwrite=overwrite,
    )


def generate_verified_ground_service_records(
    *,
    satellite_config: Tier1RolloutConfig,
    failure_realization: Tier1FailureRealization,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    visibility_policy: GroundVisibilityPolicy,
    visibility_records: Sequence[GroundVisibilityRecord],
    integrated_records: Sequence[IntegratedGroundGraphRecord],
    service_policy: GroundServicePolicy,
) -> tuple[tuple[GroundServiceStepRecord, ...], GroundServiceRunRecord]:
    if not isinstance(service_policy, GroundServicePolicy):
        raise TypeError("service_policy must be a GroundServicePolicy")
    if any(record.run_id != ground_design.run_id for record in visibility_records):
        raise ValueError("G2 record run ID does not match G1 ground-design run ID")
    if any(record.run_id != ground_design.run_id for record in integrated_records):
        raise ValueError("G3 record run ID does not match G1 ground-design run ID")
    verified_snapshots = replay_integrated_graph_records(
        records=integrated_records,
        satellite_config=satellite_config,
        failure_realization=failure_realization,
        ground_design=ground_design,
        catalog=catalog,
        visibility_policy=visibility_policy,
        visibility_records=visibility_records,
    )
    if not verified_snapshots:
        raise ValueError("Verified G3 replay returned no snapshots")
    configured_count = satellite_config.total_satellites
    step_records: list[GroundServiceStepRecord] = []
    for snapshot in verified_snapshots:
        metrics = compute_ground_service_step(
            integrated_snapshot=snapshot,
            ground_design=ground_design,
            catalog=catalog,
            configured_satellite_count=configured_count,
            policy=service_policy,
        )
        step_records.append(
            make_ground_service_step_record(
                run_id=ground_design.run_id,
                metrics=metrics,
                integrated_snapshot=snapshot,
                ground_design=ground_design,
                catalog=catalog,
                configured_satellite_count=configured_count,
                policy=service_policy,
            )
        )
    summary = summarize_ground_service_run(step_records=step_records)
    return tuple(step_records), make_ground_service_run_record(
        summary=summary,
        step_records=step_records,
    )


def persist_verified_ground_service_evidence(
    *,
    satellite_config: Tier1RolloutConfig,
    failure_realization: Tier1FailureRealization,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    visibility_policy: GroundVisibilityPolicy,
    visibility_records: Sequence[GroundVisibilityRecord],
    integrated_records: Sequence[IntegratedGroundGraphRecord],
    service_policy: GroundServicePolicy,
    step_path: str | Path,
    run_path: str | Path,
    overwrite: bool = False,
) -> tuple[tuple[GroundServiceStepRecord, ...], GroundServiceRunRecord]:
    step_records, run_record = generate_verified_ground_service_records(
        satellite_config=satellite_config,
        failure_realization=failure_realization,
        ground_design=ground_design,
        catalog=catalog,
        visibility_policy=visibility_policy,
        visibility_records=visibility_records,
        integrated_records=integrated_records,
        service_policy=service_policy,
    )
    write_ground_service_step_manifest(step_records, step_path, overwrite=overwrite)
    write_ground_service_run_manifest((run_record,), run_path, overwrite=overwrite)
    return step_records, run_record


def replay_ground_service_records(
    *,
    satellite_config: Tier1RolloutConfig,
    failure_realization: Tier1FailureRealization,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    visibility_policy: GroundVisibilityPolicy,
    visibility_records: Sequence[GroundVisibilityRecord],
    integrated_records: Sequence[IntegratedGroundGraphRecord],
    service_policy: GroundServicePolicy,
    step_records: Sequence[GroundServiceStepRecord],
    run_records: Sequence[GroundServiceRunRecord],
) -> tuple[tuple[GroundServiceStepRecord, ...], GroundServiceRunRecord]:
    if not step_records:
        raise ValueError("G4 replay requires nonempty step records")
    step_keys = [(record.run_id, record.metrics.timestep_index) for record in step_records]
    if len(step_keys) != len(set(step_keys)):
        raise ValueError("Duplicate G4 step-record keys")
    if step_keys != sorted(step_keys):
        raise ValueError("G4 step records must use canonical run/timestep order")
    if len(run_records) != 1:
        raise ValueError("G4 replay requires exactly one run summary")
    if any(record.run_id != ground_design.run_id for record in step_records):
        raise ValueError("G4 step-record run ID does not match G1 ground-design run ID")
    if run_records[0].run_id != ground_design.run_id:
        raise ValueError("G4 run-record run ID does not match G1 ground-design run ID")
    expected_steps, expected_run = generate_verified_ground_service_records(
        satellite_config=satellite_config,
        failure_realization=failure_realization,
        ground_design=ground_design,
        catalog=catalog,
        visibility_policy=visibility_policy,
        visibility_records=visibility_records,
        integrated_records=integrated_records,
        service_policy=service_policy,
    )
    expected_keys = [(record.run_id, record.metrics.timestep_index) for record in expected_steps]
    actual_by_key = {
        (record.run_id, record.metrics.timestep_index): record for record in step_records
    }
    if set(actual_by_key) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(actual_by_key))
        extra = sorted(set(actual_by_key) - set(expected_keys))
        raise ValueError(f"G4 step-record keys mismatch; missing={missing}, extra={extra}")
    for expected in expected_steps:
        key = (expected.run_id, expected.metrics.timestep_index)
        if actual_by_key[key] != expected:
            raise ValueError(f"G4 step replay mismatch at run/timestep {key}")
    if run_records[0] != expected_run:
        raise ValueError("G4 run replay mismatch")
    return expected_steps, expected_run
