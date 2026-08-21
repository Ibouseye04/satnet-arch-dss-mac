from __future__ import annotations

from dataclasses import dataclass, fields
import math
import re
from typing import Mapping, Sequence

from satnet.ground.canonical import canonical_float_string, canonical_hash, canonical_utc_timestamp
from satnet.ground.failure_realization import GroundFailureRealizationRecord
from satnet.ground.failure_service_metrics import (
    GROUND_FAILURE_SERVICE_MODEL_VERSION,
    FailureAdjustedGroundServiceStepRecord,
)
from satnet.ground.service_aggregation import (
    GroundServiceStepRecord,
    validate_ground_service_run_summary,
)
from satnet.ground.service_persistence import GroundServiceRunRecord

GROUND_FAILURE_SERVICE_STEP_SEQUENCE_IDENTITY_DOMAIN = (
    "satnet_ground_failure_service_step_sequence"
)
GROUND_FAILURE_SERVICE_STEP_SEQUENCE_IDENTITY_VERSION = "1"
GROUND_FAILURE_SERVICE_RUN_SCHEMA_VERSION = "1"
GROUND_FAILURE_SERVICE_RUN_SUMMARY_IDENTITY_DOMAIN = (
    "satnet_ground_failure_service_run_summary"
)
GROUND_FAILURE_SERVICE_RUN_SUMMARY_IDENTITY_VERSION = "1"
GROUND_FAILURE_SERVICE_RUN_RECORD_IDENTITY_DOMAIN = (
    "satnet_ground_failure_service_run_record"
)
GROUND_FAILURE_SERVICE_RUN_RECORD_IDENTITY_VERSION = "1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FailureAdjustedGroundServiceRunSummary:
    run_id: int
    satellite_config_hash: str
    ground_design_hash: str
    visibility_policy_hash: str
    ground_service_policy_hash: str
    ground_failure_policy_hash: str
    ground_failure_realization_hash: str
    ground_failure_service_model_version: str
    baseline_ground_service_run_summary_hash: str
    baseline_ground_service_step_sequence_hash: str
    first_timestep_index: int
    last_timestep_index: int
    timestep_count: int
    configured_satellite_count: int
    total_ground_station_count: int
    failed_ground_station_count: int
    operational_ground_station_count: int
    failed_ground_station_ids: tuple[str, ...]
    operational_ground_station_ids: tuple[str, ...]
    total_civilian_count: int
    failed_civilian_count: int
    operational_civilian_count: int
    total_government_count: int
    failed_government_count: int
    operational_government_count: int
    total_military_count: int
    failed_military_count: int
    operational_military_count: int
    step_sequence_hash: str
    space_gcc_fraction_original_min: float
    space_gcc_fraction_original_mean: float
    space_gcc_fraction_surviving_min: float
    space_gcc_fraction_surviving_mean: float
    baseline_ground_service_fraction_min: float
    baseline_ground_service_fraction_mean: float
    failure_adjusted_ground_service_fraction_min: float
    failure_adjusted_ground_service_fraction_mean: float
    baseline_overall_service_fraction_min: float
    baseline_overall_service_fraction_mean: float
    failure_adjusted_overall_service_fraction_min: float
    failure_adjusted_overall_service_fraction_mean: float
    ground_service_loss_due_to_failures_max: float
    ground_service_loss_due_to_failures_mean: float
    overall_service_loss_due_to_ground_failures_max: float
    overall_service_loss_due_to_ground_failures_mean: float
    failure_adjusted_civilian_service_fraction_min: float | None
    failure_adjusted_civilian_service_fraction_mean: float | None
    failure_adjusted_government_service_fraction_min: float | None
    failure_adjusted_government_service_fraction_mean: float | None
    failure_adjusted_military_service_fraction_min: float | None
    failure_adjusted_military_service_fraction_mean: float | None
    space_threshold_breach_any: bool
    ground_threshold_breach_any: bool
    overall_threshold_breach_any: bool
    first_space_threshold_breach_timestep: int | None
    first_ground_threshold_breach_timestep: int | None
    first_overall_threshold_breach_timestep: int | None
    space_threshold_breach_timestep_count: int
    ground_threshold_breach_timestep_count: int
    overall_threshold_breach_timestep_count: int
    failure_adjusted_run_summary_hash: str

    def __post_init__(self) -> None:
        _validate_summary_intrinsic(self)

    def scientific_manifest_object(self) -> dict[str, object]:
        return _summary_payload(self)


_SUMMARY_FLOAT_FIELDS = frozenset(
    field.name
    for field in fields(FailureAdjustedGroundServiceRunSummary)
    if field.name.endswith(("_min", "_mean", "_max"))
)
_SUMMARY_TUPLE_FIELDS = frozenset(
    {"failed_ground_station_ids", "operational_ground_station_ids"}
)
_SUMMARY_FIELDS = tuple(
    field.name
    for field in fields(FailureAdjustedGroundServiceRunSummary)
    if field.name not in {"run_id", "failure_adjusted_run_summary_hash"}
)


def _summary_value(
    source: FailureAdjustedGroundServiceRunSummary | Mapping[str, object], name: str
) -> object:
    return source[name] if isinstance(source, Mapping) else getattr(source, name)


def _summary_payload(
    source: FailureAdjustedGroundServiceRunSummary | Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "identity_domain": GROUND_FAILURE_SERVICE_RUN_SUMMARY_IDENTITY_DOMAIN,
        "identity_version": GROUND_FAILURE_SERVICE_RUN_SUMMARY_IDENTITY_VERSION,
    }
    for name in _SUMMARY_FIELDS:
        value = _summary_value(source, name)
        if name in _SUMMARY_FLOAT_FIELDS:
            payload[name] = None if value is None else canonical_float_string(value)
        elif name in _SUMMARY_TUPLE_FIELDS:
            payload[name] = list(value)
        else:
            payload[name] = value
    return payload


def _compute_run_summary_hash(source: Mapping[str, object]) -> str:
    return canonical_hash(_summary_payload(source))


def _validate_hash(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be exactly 64 lowercase hexadecimal characters")


def _validate_count(value: object, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise TypeError(f"{field_name} must be a nonnegative integer")


def _validate_fraction(value: object, field_name: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{field_name} must be a float")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be finite and within [0.0, 1.0]")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{field_name} must use normalized positive zero")


def _validate_optional_timestep(value: object, field_name: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise TypeError(f"{field_name} must be None or a nonnegative integer")


def _validate_summary_intrinsic(summary: FailureAdjustedGroundServiceRunSummary) -> None:
    _validate_count(summary.run_id, "run_id")
    if summary.ground_failure_service_model_version != GROUND_FAILURE_SERVICE_MODEL_VERSION:
        raise ValueError("Unsupported ground_failure_service_model_version")
    count_fields = (
        "first_timestep_index",
        "last_timestep_index",
        "timestep_count",
        "configured_satellite_count",
        "total_ground_station_count",
        "failed_ground_station_count",
        "operational_ground_station_count",
        "total_civilian_count",
        "failed_civilian_count",
        "operational_civilian_count",
        "total_government_count",
        "failed_government_count",
        "operational_government_count",
        "total_military_count",
        "failed_military_count",
        "operational_military_count",
        "space_threshold_breach_timestep_count",
        "ground_threshold_breach_timestep_count",
        "overall_threshold_breach_timestep_count",
    )
    for field_name in count_fields:
        _validate_count(getattr(summary, field_name), field_name)
    if summary.timestep_count <= 0:
        raise ValueError("timestep_count must be positive")
    if summary.configured_satellite_count <= 0:
        raise ValueError("configured_satellite_count must be positive")
    if summary.total_ground_station_count <= 0:
        raise ValueError("total_ground_station_count must be positive")
    if summary.last_timestep_index != summary.first_timestep_index + summary.timestep_count - 1:
        raise ValueError("Run timestep range does not match timestep_count")
    if (
        summary.failed_ground_station_count
        + summary.operational_ground_station_count
        != summary.total_ground_station_count
    ):
        raise ValueError("Failure counts do not sum to total ground count")
    for name in ("civilian", "government", "military"):
        total = getattr(summary, f"total_{name}_count")
        failed = getattr(summary, f"failed_{name}_count")
        operational = getattr(summary, f"operational_{name}_count")
        if failed + operational != total:
            raise ValueError(f"{name} failure counts do not sum to class total")
    if sum(
        getattr(summary, f"total_{name}_count")
        for name in ("civilian", "government", "military")
    ) != summary.total_ground_station_count:
        raise ValueError("Class totals do not sum to total ground count")
    if sum(
        getattr(summary, f"failed_{name}_count")
        for name in ("civilian", "government", "military")
    ) != summary.failed_ground_station_count:
        raise ValueError("Class failed counts do not sum to failed ground count")
    if sum(
        getattr(summary, f"operational_{name}_count")
        for name in ("civilian", "government", "military")
    ) != summary.operational_ground_station_count:
        raise ValueError("Class operational counts do not sum to operational ground count")
    for field_name in (
        "satellite_config_hash",
        "ground_design_hash",
        "visibility_policy_hash",
        "ground_service_policy_hash",
        "ground_failure_policy_hash",
        "ground_failure_realization_hash",
        "baseline_ground_service_run_summary_hash",
        "baseline_ground_service_step_sequence_hash",
        "step_sequence_hash",
        "failure_adjusted_run_summary_hash",
    ):
        _validate_hash(getattr(summary, field_name), field_name)
    for field_name in _SUMMARY_TUPLE_FIELDS:
        values = getattr(summary, field_name)
        if not isinstance(values, tuple):
            raise TypeError(f"{field_name} must be a tuple")
        if values != tuple(sorted(values)) or len(values) != len(set(values)):
            raise ValueError(f"{field_name} must use unique ascending IDs")
    if len(summary.failed_ground_station_ids) != summary.failed_ground_station_count:
        raise ValueError("failed_ground_station_ids do not match count")
    if len(summary.operational_ground_station_ids) != summary.operational_ground_station_count:
        raise ValueError("operational_ground_station_ids do not match count")
    if set(summary.failed_ground_station_ids) & set(summary.operational_ground_station_ids):
        raise ValueError("Failed and operational IDs must be disjoint")
    if (
        len(set(summary.failed_ground_station_ids) | set(summary.operational_ground_station_ids))
        != summary.total_ground_station_count
    ):
        raise ValueError("Failed and operational IDs must partition selected stations")
    for field_name in _SUMMARY_FLOAT_FIELDS:
        value = getattr(summary, field_name)
        class_name = next(
            (
                name
                for name in ("civilian", "government", "military")
                if field_name.startswith(f"failure_adjusted_{name}_")
            ),
            None,
        )
        if class_name is not None:
            total = getattr(summary, f"total_{class_name}_count")
            if (value is None) != (total == 0):
                raise ValueError(f"{field_name} must be None exactly for an absent class")
            if value is not None:
                _validate_fraction(value, field_name)
        else:
            _validate_fraction(value, field_name)
    for prefix in ("space", "ground", "overall"):
        breach_any = getattr(summary, f"{prefix}_threshold_breach_any")
        breach_count = getattr(summary, f"{prefix}_threshold_breach_timestep_count")
        first = getattr(summary, f"first_{prefix}_threshold_breach_timestep")
        if type(breach_any) is not bool:
            raise TypeError(f"{prefix}_threshold_breach_any must be a Boolean")
        _validate_optional_timestep(first, f"first_{prefix}_threshold_breach_timestep")
        if breach_count > summary.timestep_count:
            raise ValueError(f"{prefix} breach count exceeds timestep_count")
        if breach_any != (breach_count > 0):
            raise ValueError(f"{prefix} breach flag does not match breach count")
        if breach_count == 0 and first is not None:
            raise ValueError(f"{prefix} first breach must be None without breaches")
        if breach_count > 0 and (
            first is None
            or not summary.first_timestep_index <= first <= summary.last_timestep_index
        ):
            raise ValueError(f"{prefix} first breach is outside run range")
    if summary.failure_adjusted_run_summary_hash != canonical_hash(
        _summary_payload(summary)
    ):
        raise ValueError("failure_adjusted_run_summary_hash does not match canonical summary")


def _validate_adjusted_sequence(
    records: Sequence[FailureAdjustedGroundServiceStepRecord],
) -> tuple[FailureAdjustedGroundServiceStepRecord, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("step_records must be a sequence")
    normalized = tuple(records)
    if not normalized:
        raise ValueError("At least one G5 step record is required")
    if any(not isinstance(record, FailureAdjustedGroundServiceStepRecord) for record in normalized):
        raise TypeError("step_records must contain G5 step records")
    keys = [(record.run_id, record.metrics.timestep_index) for record in normalized]
    if len(keys) != len(set(keys)):
        raise ValueError("G5 step records contain duplicate keys")
    if keys != sorted(keys):
        raise ValueError("G5 step records must use increasing run/timestep order")
    if len({record.run_id for record in normalized}) != 1:
        raise ValueError("G5 step records contain mixed run IDs")
    timesteps = [record.metrics.timestep_index for record in normalized]
    if timesteps != list(range(timesteps[0], timesteps[0] + len(timesteps))):
        raise ValueError("G5 step records must use contiguous timesteps")
    timestamps = [record.metrics.timestamp_utc for record in normalized]
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("G5 step timestamps must be strictly increasing")
    constant_fields = (
        "satellite_config_hash",
        "ground_design_hash",
        "visibility_policy_hash",
        "ground_service_policy_hash",
        "ground_failure_policy_hash",
        "ground_failure_realization_hash",
        "ground_failure_service_model_version",
        "configured_satellite_count",
        "total_ground_station_count",
        "failed_ground_station_count",
        "operational_ground_station_count",
        "failed_ground_station_ids",
        "operational_ground_station_ids",
        "total_civilian_count",
        "failed_civilian_count",
        "operational_civilian_count",
        "total_government_count",
        "failed_government_count",
        "operational_government_count",
        "total_military_count",
        "failed_military_count",
        "operational_military_count",
    )
    for field_name in constant_fields:
        if len({getattr(record.metrics, field_name) for record in normalized}) != 1:
            raise ValueError(f"G5 steps do not have constant {field_name}")
    return normalized


def _validate_g4_binding(
    *,
    g5_records: Sequence[FailureAdjustedGroundServiceStepRecord],
    realization_record: GroundFailureRealizationRecord,
    g4_records: Sequence[GroundServiceStepRecord],
    g4_run_record: GroundServiceRunRecord,
) -> tuple[
    tuple[FailureAdjustedGroundServiceStepRecord, ...],
    tuple[GroundServiceStepRecord, ...],
]:
    adjusted = _validate_adjusted_sequence(g5_records)
    baseline = tuple(g4_records)
    if not baseline or any(not isinstance(record, GroundServiceStepRecord) for record in baseline):
        raise ValueError("verified_g4_step_records must contain G4 step records")
    if not isinstance(g4_run_record, GroundServiceRunRecord):
        raise TypeError("verified_g4_run_record must be GroundServiceRunRecord")
    if not isinstance(realization_record, GroundFailureRealizationRecord):
        raise TypeError("failure_realization_record must be GroundFailureRealizationRecord")
    validate_ground_service_run_summary(
        summary=g4_run_record.summary,
        step_records=baseline,
    )
    g4_keys = [(record.run_id, record.metrics.timestep_index) for record in baseline]
    g5_keys = [(record.run_id, record.metrics.timestep_index) for record in adjusted]
    if len(g4_keys) != len(set(g4_keys)):
        raise ValueError("Verified G4 steps contain duplicate keys")
    if set(g4_keys) != set(g5_keys):
        raise ValueError("G4 and G5 step keys do not match exactly")
    if g4_keys != sorted(g4_keys):
        raise ValueError("Verified G4 steps must use canonical order")
    run_ids = {
        realization_record.run_id,
        g4_run_record.run_id,
        *(record.run_id for record in baseline),
        *(record.run_id for record in adjusted),
    }
    if len(run_ids) != 1:
        raise ValueError("G4, G5, and realization records contain mixed run IDs")
    g4_by_key = {
        (record.run_id, record.metrics.timestep_index): record for record in baseline
    }
    for record in adjusted:
        key = (record.run_id, record.metrics.timestep_index)
        if (
            record.metrics.baseline_ground_service_step_hash
            != g4_by_key[key].metrics.step_metrics_hash
        ):
            raise ValueError(f"G5 baseline step hash mismatch at {key}")
    return adjusted, baseline


def _mean(values: Sequence[float]) -> float:
    result = math.fsum(values) / len(values)
    if not math.isfinite(result):
        raise ValueError("Run mean must be finite")
    return 0.0 if result == 0.0 else result


def _optional_aggregate(values: Sequence[float | None]) -> tuple[float | None, float | None]:
    if all(value is None for value in values):
        return None, None
    if any(value is None for value in values):
        raise ValueError("Class fractions must be consistently present or absent")
    numeric = tuple(value for value in values if value is not None)
    return min(numeric), _mean(numeric)


def _breaches(
    records: Sequence[FailureAdjustedGroundServiceStepRecord], field_name: str
) -> tuple[bool, int | None, int]:
    timesteps = [
        record.metrics.timestep_index
        for record in records
        if not getattr(record.metrics, field_name)
    ]
    return bool(timesteps), timesteps[0] if timesteps else None, len(timesteps)


def _step_sequence_hash(
    records: Sequence[FailureAdjustedGroundServiceStepRecord],
) -> str:
    return canonical_hash(
        {
            "identity_domain": GROUND_FAILURE_SERVICE_STEP_SEQUENCE_IDENTITY_DOMAIN,
            "identity_version": GROUND_FAILURE_SERVICE_STEP_SEQUENCE_IDENTITY_VERSION,
            "steps": [
                {
                    "failure_adjusted_step_hash": record.metrics.failure_adjusted_step_hash,
                    "timestep_index": record.metrics.timestep_index,
                    "timestamp_utc": canonical_utc_timestamp(record.metrics.timestamp_utc),
                }
                for record in records
            ],
        }
    )


def _build_summary_values(
    *,
    step_records: Sequence[FailureAdjustedGroundServiceStepRecord],
    failure_realization_record: GroundFailureRealizationRecord,
    verified_g4_step_records: Sequence[GroundServiceStepRecord],
    verified_g4_run_record: GroundServiceRunRecord,
) -> dict[str, object]:
    adjusted, baseline = _validate_g4_binding(
        g5_records=step_records,
        realization_record=failure_realization_record,
        g4_records=verified_g4_step_records,
        g4_run_record=verified_g4_run_record,
    )
    metrics = [record.metrics for record in adjusted]
    first = metrics[0]
    g4_summary = verified_g4_run_record.summary
    original = [value.space_gcc_fraction_original for value in metrics]
    surviving = [value.space_gcc_fraction_surviving for value in metrics]
    baseline_ground = [value.baseline_ground_service_fraction for value in metrics]
    adjusted_ground = [value.failure_adjusted_ground_service_fraction for value in metrics]
    baseline_overall = [value.baseline_overall_service_fraction for value in metrics]
    adjusted_overall = [value.failure_adjusted_overall_service_fraction for value in metrics]
    ground_loss = [value.ground_service_loss_due_to_failures for value in metrics]
    overall_loss = [value.overall_service_loss_due_to_ground_failures for value in metrics]
    class_aggregates = {
        name: _optional_aggregate(
            [
                getattr(value, f"failure_adjusted_{name}_service_fraction")
                for value in metrics
            ]
        )
        for name in ("civilian", "government", "military")
    }
    space_breach = _breaches(adjusted, "space_threshold_met")
    ground_breach = _breaches(adjusted, "ground_threshold_met")
    overall_breach = _breaches(adjusted, "overall_threshold_met")
    values: dict[str, object] = {
        "run_id": adjusted[0].run_id,
        "satellite_config_hash": first.satellite_config_hash,
        "ground_design_hash": first.ground_design_hash,
        "visibility_policy_hash": first.visibility_policy_hash,
        "ground_service_policy_hash": first.ground_service_policy_hash,
        "ground_failure_policy_hash": first.ground_failure_policy_hash,
        "ground_failure_realization_hash": first.ground_failure_realization_hash,
        "ground_failure_service_model_version": first.ground_failure_service_model_version,
        "baseline_ground_service_run_summary_hash": g4_summary.run_summary_hash,
        "baseline_ground_service_step_sequence_hash": g4_summary.step_sequence_hash,
        "first_timestep_index": metrics[0].timestep_index,
        "last_timestep_index": metrics[-1].timestep_index,
        "timestep_count": len(metrics),
        "configured_satellite_count": first.configured_satellite_count,
        "total_ground_station_count": first.total_ground_station_count,
        "failed_ground_station_count": first.failed_ground_station_count,
        "operational_ground_station_count": first.operational_ground_station_count,
        "failed_ground_station_ids": first.failed_ground_station_ids,
        "operational_ground_station_ids": first.operational_ground_station_ids,
        "total_civilian_count": first.total_civilian_count,
        "failed_civilian_count": first.failed_civilian_count,
        "operational_civilian_count": first.operational_civilian_count,
        "total_government_count": first.total_government_count,
        "failed_government_count": first.failed_government_count,
        "operational_government_count": first.operational_government_count,
        "total_military_count": first.total_military_count,
        "failed_military_count": first.failed_military_count,
        "operational_military_count": first.operational_military_count,
        "step_sequence_hash": _step_sequence_hash(adjusted),
        "space_gcc_fraction_original_min": min(original),
        "space_gcc_fraction_original_mean": _mean(original),
        "space_gcc_fraction_surviving_min": min(surviving),
        "space_gcc_fraction_surviving_mean": _mean(surviving),
        "baseline_ground_service_fraction_min": min(baseline_ground),
        "baseline_ground_service_fraction_mean": _mean(baseline_ground),
        "failure_adjusted_ground_service_fraction_min": min(adjusted_ground),
        "failure_adjusted_ground_service_fraction_mean": _mean(adjusted_ground),
        "baseline_overall_service_fraction_min": min(baseline_overall),
        "baseline_overall_service_fraction_mean": _mean(baseline_overall),
        "failure_adjusted_overall_service_fraction_min": min(adjusted_overall),
        "failure_adjusted_overall_service_fraction_mean": _mean(adjusted_overall),
        "ground_service_loss_due_to_failures_max": max(ground_loss),
        "ground_service_loss_due_to_failures_mean": _mean(ground_loss),
        "overall_service_loss_due_to_ground_failures_max": max(overall_loss),
        "overall_service_loss_due_to_ground_failures_mean": _mean(overall_loss),
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
    for name, aggregate in class_aggregates.items():
        values[f"failure_adjusted_{name}_service_fraction_min"] = aggregate[0]
        values[f"failure_adjusted_{name}_service_fraction_mean"] = aggregate[1]
    expected_baseline = {
        "space_gcc_fraction_original_min": g4_summary.space_gcc_fraction_original_min,
        "space_gcc_fraction_original_mean": g4_summary.space_gcc_fraction_original_mean,
        "space_gcc_fraction_surviving_min": g4_summary.space_gcc_fraction_surviving_min,
        "space_gcc_fraction_surviving_mean": g4_summary.space_gcc_fraction_surviving_mean,
        "baseline_ground_service_fraction_min": g4_summary.ground_service_fraction_min,
        "baseline_ground_service_fraction_mean": g4_summary.ground_service_fraction_mean,
        "baseline_overall_service_fraction_min": g4_summary.overall_service_fraction_min,
        "baseline_overall_service_fraction_mean": g4_summary.overall_service_fraction_mean,
        "space_threshold_breach_any": g4_summary.space_threshold_breach_any,
        "first_space_threshold_breach_timestep": g4_summary.first_space_threshold_breach_timestep,
        "space_threshold_breach_timestep_count": g4_summary.space_threshold_breach_timestep_count,
    }
    for field_name, expected in expected_baseline.items():
        if values[field_name] != expected:
            raise ValueError(f"G5 baseline aggregate does not match verified G4: {field_name}")
    return values


def summarize_failure_adjusted_ground_service_run(
    *,
    step_records: Sequence[FailureAdjustedGroundServiceStepRecord],
    failure_realization_record: GroundFailureRealizationRecord,
    verified_g4_step_records: Sequence[GroundServiceStepRecord],
    verified_g4_run_record: GroundServiceRunRecord,
) -> FailureAdjustedGroundServiceRunSummary:
    values = _build_summary_values(
        step_records=step_records,
        failure_realization_record=failure_realization_record,
        verified_g4_step_records=verified_g4_step_records,
        verified_g4_run_record=verified_g4_run_record,
    )
    values["failure_adjusted_run_summary_hash"] = _compute_run_summary_hash(values)
    summary = FailureAdjustedGroundServiceRunSummary(**values)
    validate_failure_adjusted_ground_service_run_summary(
        summary=summary,
        step_records=step_records,
        failure_realization_record=failure_realization_record,
        verified_g4_step_records=verified_g4_step_records,
        verified_g4_run_record=verified_g4_run_record,
    )
    return summary


def validate_failure_adjusted_ground_service_run_summary(
    *,
    summary: FailureAdjustedGroundServiceRunSummary,
    step_records: Sequence[FailureAdjustedGroundServiceStepRecord],
    failure_realization_record: GroundFailureRealizationRecord,
    verified_g4_step_records: Sequence[GroundServiceStepRecord],
    verified_g4_run_record: GroundServiceRunRecord,
) -> None:
    if not isinstance(summary, FailureAdjustedGroundServiceRunSummary):
        raise TypeError("summary must be FailureAdjustedGroundServiceRunSummary")
    values = _build_summary_values(
        step_records=step_records,
        failure_realization_record=failure_realization_record,
        verified_g4_step_records=verified_g4_step_records,
        verified_g4_run_record=verified_g4_run_record,
    )
    values["failure_adjusted_run_summary_hash"] = _compute_run_summary_hash(values)
    for field in fields(FailureAdjustedGroundServiceRunSummary):
        if getattr(summary, field.name) != values[field.name]:
            raise ValueError(f"Failure-adjusted run summary mismatch: {field.name}")


def _run_record_hash(
    *, run_id: int, summary: FailureAdjustedGroundServiceRunSummary
) -> str:
    return canonical_hash(
        {
            "failure_adjusted_run_summary_hash": summary.failure_adjusted_run_summary_hash,
            "ground_failure_service_run_schema_version": GROUND_FAILURE_SERVICE_RUN_SCHEMA_VERSION,
            "identity_domain": GROUND_FAILURE_SERVICE_RUN_RECORD_IDENTITY_DOMAIN,
            "identity_version": GROUND_FAILURE_SERVICE_RUN_RECORD_IDENTITY_VERSION,
            "run_id": run_id,
        }
    )


@dataclass(frozen=True)
class FailureAdjustedGroundServiceRunRecord:
    ground_failure_service_run_schema_version: str
    run_id: int
    summary: FailureAdjustedGroundServiceRunSummary
    record_hash: str

    def __post_init__(self) -> None:
        if (
            self.ground_failure_service_run_schema_version
            != GROUND_FAILURE_SERVICE_RUN_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported ground_failure_service_run_schema_version")
        if type(self.run_id) is not int or self.run_id < 0:
            raise TypeError("run_id must be a nonnegative integer")
        if not isinstance(self.summary, FailureAdjustedGroundServiceRunSummary):
            raise TypeError("summary must be FailureAdjustedGroundServiceRunSummary")
        if self.summary.run_id != self.run_id:
            raise ValueError("Run-record run ID does not match summary run ID")
        if self.record_hash != _run_record_hash(run_id=self.run_id, summary=self.summary):
            raise ValueError("record_hash does not match canonical adjusted run record")


def create_failure_adjusted_run_record(
    *,
    verified_g4_step_records: Sequence[GroundServiceStepRecord],
    verified_g4_run_record: GroundServiceRunRecord,
    g5_step_records: Sequence[FailureAdjustedGroundServiceStepRecord],
    realization_record: GroundFailureRealizationRecord,
) -> FailureAdjustedGroundServiceRunRecord:
    summary = summarize_failure_adjusted_ground_service_run(
        step_records=g5_step_records,
        failure_realization_record=realization_record,
        verified_g4_step_records=verified_g4_step_records,
        verified_g4_run_record=verified_g4_run_record,
    )
    validate_failure_adjusted_ground_service_run_summary(
        summary=summary,
        step_records=g5_step_records,
        failure_realization_record=realization_record,
        verified_g4_step_records=verified_g4_step_records,
        verified_g4_run_record=verified_g4_run_record,
    )
    return FailureAdjustedGroundServiceRunRecord(
        ground_failure_service_run_schema_version=GROUND_FAILURE_SERVICE_RUN_SCHEMA_VERSION,
        run_id=summary.run_id,
        summary=summary,
        record_hash=_run_record_hash(run_id=summary.run_id, summary=summary),
    )
