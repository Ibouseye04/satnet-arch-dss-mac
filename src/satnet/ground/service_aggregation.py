from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping, Sequence

from satnet.ground.canonical import canonical_float_string, canonical_hash, canonical_utc_timestamp
from satnet.ground.catalog import GroundStationCatalog
from satnet.ground.integrated_graph import IntegratedGroundGraphSnapshot
from satnet.ground.persistence import GroundRunDesignRecord
from satnet.ground.service_metrics import (
    GroundServiceStepMetrics,
    validate_ground_service_step_context,
)
from satnet.ground.service_policy import GROUND_SERVICE_MODEL_VERSION, GroundServicePolicy

GROUND_SERVICE_STEP_SCHEMA_VERSION = "1"
GROUND_SERVICE_STEP_RECORD_IDENTITY_DOMAIN = "satnet_ground_service_step_record"
GROUND_SERVICE_STEP_RECORD_IDENTITY_VERSION = "1"
GROUND_SERVICE_STEP_SEQUENCE_IDENTITY_DOMAIN = "satnet_ground_service_step_sequence"
GROUND_SERVICE_STEP_SEQUENCE_IDENTITY_VERSION = "1"
GROUND_SERVICE_RUN_SCHEMA_VERSION = "1"
GROUND_SERVICE_RUN_SUMMARY_IDENTITY_DOMAIN = "satnet_ground_service_run_summary"
GROUND_SERVICE_RUN_SUMMARY_IDENTITY_VERSION = "1"


def _step_record_hash(*, run_id: int, metrics: GroundServiceStepMetrics) -> str:
    return canonical_hash(
        {
            "ground_service_step_schema_version": GROUND_SERVICE_STEP_SCHEMA_VERSION,
            "identity_domain": GROUND_SERVICE_STEP_RECORD_IDENTITY_DOMAIN,
            "identity_version": GROUND_SERVICE_STEP_RECORD_IDENTITY_VERSION,
            "run_id": run_id,
            "step_metrics_hash": metrics.step_metrics_hash,
            "timestep_index": metrics.timestep_index,
            "timestamp_utc": canonical_utc_timestamp(metrics.timestamp_utc),
        }
    )


@dataclass(frozen=True)
class GroundServiceStepRecord:
    ground_service_step_schema_version: str
    run_id: int
    metrics: GroundServiceStepMetrics
    record_hash: str

    def __post_init__(self) -> None:
        if self.ground_service_step_schema_version != GROUND_SERVICE_STEP_SCHEMA_VERSION:
            raise ValueError("Unsupported ground_service_step_schema_version")
        if type(self.run_id) is not int or self.run_id < 0:
            raise TypeError("run_id must be a nonnegative integer")
        if not isinstance(self.metrics, GroundServiceStepMetrics):
            raise TypeError("metrics must be GroundServiceStepMetrics")
        expected = _step_record_hash(run_id=self.run_id, metrics=self.metrics)
        if self.record_hash != expected:
            raise ValueError("record_hash does not match canonical ground-service step record")


def make_ground_service_step_record(
    *,
    run_id: int,
    metrics: GroundServiceStepMetrics,
    integrated_snapshot: IntegratedGroundGraphSnapshot,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    configured_satellite_count: int,
    policy: GroundServicePolicy,
) -> GroundServiceStepRecord:
    if type(run_id) is not int or run_id < 0:
        raise TypeError("run_id must be a nonnegative integer")
    if run_id != ground_design.run_id:
        raise ValueError("G4 step-record run ID does not match G1 ground-design run ID")
    validate_ground_service_step_context(
        metrics=metrics,
        integrated_snapshot=integrated_snapshot,
        ground_design=ground_design,
        catalog=catalog,
        configured_satellite_count=configured_satellite_count,
        policy=policy,
    )
    return GroundServiceStepRecord(
        ground_service_step_schema_version=GROUND_SERVICE_STEP_SCHEMA_VERSION,
        run_id=run_id,
        metrics=metrics,
        record_hash=_step_record_hash(run_id=run_id, metrics=metrics),
    )


def _step_sequence_hash(step_records: Sequence[GroundServiceStepRecord]) -> str:
    return canonical_hash(
        {
            "identity_domain": GROUND_SERVICE_STEP_SEQUENCE_IDENTITY_DOMAIN,
            "identity_version": GROUND_SERVICE_STEP_SEQUENCE_IDENTITY_VERSION,
            "steps": [
                {
                    "step_metrics_hash": record.metrics.step_metrics_hash,
                    "timestep_index": record.metrics.timestep_index,
                    "timestamp_utc": canonical_utc_timestamp(record.metrics.timestamp_utc),
                }
                for record in step_records
            ],
        }
    )


@dataclass(frozen=True)
class GroundServiceRunSummary:
    run_id: int
    satellite_config_hash: str
    ground_design_hash: str
    visibility_policy_hash: str
    ground_service_policy_hash: str
    ground_service_model_version: str
    first_timestep_index: int
    last_timestep_index: int
    timestep_count: int
    configured_satellite_count: int
    total_ground_station_count: int
    total_civilian_count: int
    total_government_count: int
    total_military_count: int
    step_sequence_hash: str
    space_gcc_fraction_original_min: float
    space_gcc_fraction_original_mean: float
    space_gcc_fraction_surviving_min: float
    space_gcc_fraction_surviving_mean: float
    ground_service_fraction_min: float
    ground_service_fraction_mean: float
    overall_service_fraction_min: float
    overall_service_fraction_mean: float
    civilian_service_fraction_min: float | None
    civilian_service_fraction_mean: float | None
    government_service_fraction_min: float | None
    government_service_fraction_mean: float | None
    military_service_fraction_min: float | None
    military_service_fraction_mean: float | None
    space_threshold_breach_any: bool
    ground_threshold_breach_any: bool
    overall_threshold_breach_any: bool
    first_space_threshold_breach_timestep: int | None
    first_ground_threshold_breach_timestep: int | None
    first_overall_threshold_breach_timestep: int | None
    space_threshold_breach_timestep_count: int
    ground_threshold_breach_timestep_count: int
    overall_threshold_breach_timestep_count: int
    run_summary_hash: str

    def __post_init__(self) -> None:
        _validate_run_summary_intrinsic(self)

    def scientific_manifest_object(self) -> dict[str, object]:
        return _run_summary_payload(self)


_SUMMARY_FLOAT_FIELDS = frozenset(
    field.name
    for field in fields(GroundServiceRunSummary)
    if field.name.endswith("_min") or field.name.endswith("_mean")
)
_SUMMARY_FIELDS = tuple(
    field.name
    for field in fields(GroundServiceRunSummary)
    if field.name not in {"run_id", "run_summary_hash"}
)


def _summary_value(source: GroundServiceRunSummary | Mapping[str, object], name: str) -> object:
    if isinstance(source, Mapping):
        return source[name]
    return getattr(source, name)


def _run_summary_payload(
    source: GroundServiceRunSummary | Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "identity_domain": GROUND_SERVICE_RUN_SUMMARY_IDENTITY_DOMAIN,
        "identity_version": GROUND_SERVICE_RUN_SUMMARY_IDENTITY_VERSION,
    }
    for name in _SUMMARY_FIELDS:
        value = _summary_value(source, name)
        if name in _SUMMARY_FLOAT_FIELDS:
            payload[name] = None if value is None else canonical_float_string(value)
        else:
            payload[name] = value
    return payload


def _compute_run_summary_hash(source: Mapping[str, object]) -> str:
    return canonical_hash(_run_summary_payload(source))


def _validate_fraction(value: object, field_name: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{field_name} must be a float")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be finite and within [0.0, 1.0]")


def _validate_optional_timestep(value: object, field_name: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise TypeError(f"{field_name} must be None or a nonnegative integer")


def _validate_run_summary_intrinsic(summary: GroundServiceRunSummary) -> None:
    if type(summary.run_id) is not int or summary.run_id < 0:
        raise TypeError("run_id must be a nonnegative integer")
    if summary.ground_service_model_version != GROUND_SERVICE_MODEL_VERSION:
        raise ValueError("Unsupported ground_service_model_version")
    for field_name in (
        "first_timestep_index",
        "last_timestep_index",
        "timestep_count",
        "configured_satellite_count",
        "total_ground_station_count",
        "total_civilian_count",
        "total_government_count",
        "total_military_count",
        "space_threshold_breach_timestep_count",
        "ground_threshold_breach_timestep_count",
        "overall_threshold_breach_timestep_count",
    ):
        value = getattr(summary, field_name)
        if type(value) is not int or value < 0:
            raise TypeError(f"{field_name} must be a nonnegative integer")
    if summary.timestep_count <= 0:
        raise ValueError("timestep_count must be positive")
    if summary.configured_satellite_count <= 0:
        raise ValueError("configured_satellite_count must be positive")
    if summary.total_ground_station_count <= 0:
        raise ValueError("total_ground_station_count must be positive")
    if summary.last_timestep_index != summary.first_timestep_index + summary.timestep_count - 1:
        raise ValueError("Run timestep range does not match timestep_count")
    if (
        summary.total_civilian_count
        + summary.total_government_count
        + summary.total_military_count
        != summary.total_ground_station_count
    ):
        raise ValueError("Run class totals do not sum to total_ground_station_count")
    for field_name in (
        "satellite_config_hash",
        "ground_design_hash",
        "visibility_policy_hash",
        "ground_service_policy_hash",
        "step_sequence_hash",
        "run_summary_hash",
    ):
        value = getattr(summary, field_name)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{field_name} must be a lowercase SHA-256 value")
    for field_name in _SUMMARY_FLOAT_FIELDS:
        value = getattr(summary, field_name)
        class_name = field_name.split("_", 1)[0]
        if class_name in {"civilian", "government", "military"}:
            class_total = getattr(summary, f"total_{class_name}_count")
            if (value is None) != (class_total == 0):
                raise ValueError(f"{field_name} must be None exactly for an absent class")
            if value is not None:
                _validate_fraction(value, field_name)
        else:
            _validate_fraction(value, field_name)
    for prefix in ("space", "ground", "overall"):
        breach_any = getattr(summary, f"{prefix}_threshold_breach_any")
        breach_count = getattr(summary, f"{prefix}_threshold_breach_timestep_count")
        first_breach = getattr(summary, f"first_{prefix}_threshold_breach_timestep")
        if type(breach_any) is not bool:
            raise TypeError(f"{prefix}_threshold_breach_any must be a Boolean")
        _validate_optional_timestep(first_breach, f"first_{prefix}_threshold_breach_timestep")
        if breach_count > summary.timestep_count:
            raise ValueError(f"{prefix} breach count exceeds timestep_count")
        if breach_any != (breach_count > 0):
            raise ValueError(f"{prefix} breach flag does not match breach count")
        if breach_count == 0 and first_breach is not None:
            raise ValueError(f"{prefix} first breach must be None when no breach occurred")
        if breach_count > 0 and (
            first_breach is None
            or not summary.first_timestep_index <= first_breach <= summary.last_timestep_index
        ):
            raise ValueError(f"{prefix} first breach is outside the run timestep range")
    expected_hash = canonical_hash(_run_summary_payload(summary))
    if summary.run_summary_hash != expected_hash:
        raise ValueError("run_summary_hash does not match canonical run summary")


def _validate_step_record_sequence(
    step_records: Sequence[GroundServiceStepRecord],
) -> tuple[GroundServiceStepRecord, ...]:
    if isinstance(step_records, (str, bytes)) or not isinstance(step_records, Sequence):
        raise TypeError("step_records must be a sequence")
    records = tuple(step_records)
    if not records:
        raise ValueError("At least one ground-service step record is required")
    if any(not isinstance(record, GroundServiceStepRecord) for record in records):
        raise TypeError("step_records must contain GroundServiceStepRecord values")
    run_ids = {record.run_id for record in records}
    if len(run_ids) != 1:
        raise ValueError("Ground-service step records contain mixed run IDs")
    timesteps = [record.metrics.timestep_index for record in records]
    if len(timesteps) != len(set(timesteps)):
        raise ValueError("Ground-service step records contain duplicate timesteps")
    if timesteps != sorted(timesteps):
        raise ValueError("Ground-service step records must be in strictly increasing order")
    expected_timesteps = list(range(timesteps[0], timesteps[0] + len(timesteps)))
    if timesteps != expected_timesteps:
        raise ValueError("Ground-service step records must use contiguous timesteps")
    timestamps = [record.metrics.timestamp_utc for record in records]
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Ground-service step timestamps must be strictly increasing")
    constant_fields = (
        "satellite_config_hash",
        "ground_design_hash",
        "visibility_policy_hash",
        "ground_service_policy_hash",
        "ground_service_model_version",
        "configured_satellite_count",
        "total_ground_station_count",
        "total_civilian_count",
        "total_government_count",
        "total_military_count",
    )
    for field_name in constant_fields:
        if len({getattr(record.metrics, field_name) for record in records}) != 1:
            raise ValueError(f"Ground-service steps do not have constant {field_name}")
    return records


def _mean(values: Sequence[float]) -> float:
    result = math.fsum(values) / len(values)
    if not math.isfinite(result):
        raise ValueError("Run mean must be finite")
    return result


def _optional_aggregate(
    values: Sequence[float | None],
) -> tuple[float | None, float | None]:
    if all(value is None for value in values):
        return None, None
    if any(value is None for value in values):
        raise ValueError("Class-specific fractions must be consistently present or absent")
    numeric = tuple(value for value in values if value is not None)
    return min(numeric), _mean(numeric)


def _breach_values(
    records: Sequence[GroundServiceStepRecord], field_name: str
) -> tuple[bool, int | None, int]:
    timesteps = [
        record.metrics.timestep_index
        for record in records
        if not getattr(record.metrics, field_name)
    ]
    return bool(timesteps), (timesteps[0] if timesteps else None), len(timesteps)


def _build_summary_values(
    records: Sequence[GroundServiceStepRecord],
) -> dict[str, object]:
    metrics = [record.metrics for record in records]
    first = metrics[0]
    original = [step.space_gcc_fraction_original for step in metrics]
    surviving = [step.space_gcc_fraction_surviving for step in metrics]
    ground = [step.ground_service_fraction for step in metrics]
    overall = [step.overall_service_fraction for step in metrics]
    civilian_min, civilian_mean = _optional_aggregate(
        [step.civilian_service_fraction for step in metrics]
    )
    government_min, government_mean = _optional_aggregate(
        [step.government_service_fraction for step in metrics]
    )
    military_min, military_mean = _optional_aggregate(
        [step.military_service_fraction for step in metrics]
    )
    space_breach = _breach_values(records, "space_threshold_met")
    ground_breach = _breach_values(records, "ground_threshold_met")
    overall_breach = _breach_values(records, "overall_threshold_met")
    return {
        "run_id": records[0].run_id,
        "satellite_config_hash": first.satellite_config_hash,
        "ground_design_hash": first.ground_design_hash,
        "visibility_policy_hash": first.visibility_policy_hash,
        "ground_service_policy_hash": first.ground_service_policy_hash,
        "ground_service_model_version": first.ground_service_model_version,
        "first_timestep_index": metrics[0].timestep_index,
        "last_timestep_index": metrics[-1].timestep_index,
        "timestep_count": len(metrics),
        "configured_satellite_count": first.configured_satellite_count,
        "total_ground_station_count": first.total_ground_station_count,
        "total_civilian_count": first.total_civilian_count,
        "total_government_count": first.total_government_count,
        "total_military_count": first.total_military_count,
        "step_sequence_hash": _step_sequence_hash(records),
        "space_gcc_fraction_original_min": min(original),
        "space_gcc_fraction_original_mean": _mean(original),
        "space_gcc_fraction_surviving_min": min(surviving),
        "space_gcc_fraction_surviving_mean": _mean(surviving),
        "ground_service_fraction_min": min(ground),
        "ground_service_fraction_mean": _mean(ground),
        "overall_service_fraction_min": min(overall),
        "overall_service_fraction_mean": _mean(overall),
        "civilian_service_fraction_min": civilian_min,
        "civilian_service_fraction_mean": civilian_mean,
        "government_service_fraction_min": government_min,
        "government_service_fraction_mean": government_mean,
        "military_service_fraction_min": military_min,
        "military_service_fraction_mean": military_mean,
        "space_threshold_breach_any": space_breach[0],
        "ground_threshold_breach_any": ground_breach[0],
        "overall_threshold_breach_any": overall_breach[0],
        "first_space_threshold_breach_timestep": space_breach[1],
        "first_ground_threshold_breach_timestep": ground_breach[1],
        "first_overall_threshold_breach_timestep": overall_breach[1],
        "space_threshold_breach_timestep_count": space_breach[2],
        "ground_threshold_breach_timestep_count": ground_breach[2],
        "overall_threshold_breach_timestep_count": overall_breach[2],
    }


def summarize_ground_service_run(
    *, step_records: Sequence[GroundServiceStepRecord]
) -> GroundServiceRunSummary:
    records = _validate_step_record_sequence(step_records)
    values = _build_summary_values(records)
    values["run_summary_hash"] = _compute_run_summary_hash(values)
    summary = GroundServiceRunSummary(**values)
    validate_ground_service_run_summary(summary=summary, step_records=records)
    return summary


def validate_ground_service_run_summary(
    *,
    summary: GroundServiceRunSummary,
    step_records: Sequence[GroundServiceStepRecord],
) -> None:
    if not isinstance(summary, GroundServiceRunSummary):
        raise TypeError("summary must be GroundServiceRunSummary")
    records = _validate_step_record_sequence(step_records)
    values = _build_summary_values(records)
    values["run_summary_hash"] = _compute_run_summary_hash(values)
    for field in fields(GroundServiceRunSummary):
        if getattr(summary, field.name) != values[field.name]:
            raise ValueError(f"Ground-service run summary mismatch: {field.name}")
