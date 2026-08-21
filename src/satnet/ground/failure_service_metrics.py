from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
import math
import re
from typing import Mapping

from satnet.ground.canonical import (
    canonical_float_string,
    canonical_hash,
    canonical_utc_timestamp,
)
from satnet.ground.catalog import GroundStationCatalog, STATION_ID_PATTERN
from satnet.ground.failure_policy import GroundFailurePolicy
from satnet.ground.failure_realization import (
    GroundFailureRealization,
    GroundFailureRealizationRecord,
    validate_ground_failure_realization_record_context,
)
from satnet.ground.persistence import GroundRunDesignRecord, reconstruct_ground_selection
from satnet.ground.service_aggregation import GroundServiceStepRecord
from satnet.ground.service_metrics import GroundServiceStepMetrics
from satnet.ground.service_policy import GroundServicePolicy

GROUND_FAILURE_SERVICE_MODEL_VERSION = "1"
GROUND_FAILURE_SERVICE_STEP_IDENTITY_DOMAIN = "satnet_ground_failure_service_step"
GROUND_FAILURE_SERVICE_STEP_IDENTITY_VERSION = "1"
GROUND_FAILURE_SERVICE_STEP_SCHEMA_VERSION = "1"
GROUND_FAILURE_SERVICE_STEP_RECORD_IDENTITY_DOMAIN = (
    "satnet_ground_failure_service_step_record"
)
GROUND_FAILURE_SERVICE_STEP_RECORD_IDENTITY_VERSION = "1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FailureAdjustedGroundServiceStepMetrics:
    timestep_index: int
    timestamp_utc: datetime
    satellite_config_hash: str
    ground_design_hash: str
    visibility_policy_hash: str
    integrated_graph_hash: str
    ground_service_policy_hash: str
    baseline_ground_service_step_hash: str
    ground_failure_policy_hash: str
    ground_failure_realization_hash: str
    ground_failure_service_model_version: str
    configured_satellite_count: int
    operational_satellite_count: int
    satellite_component_count: int
    satellite_gcc_size: int
    satellite_gcc_ids: tuple[int, ...]
    total_ground_station_count: int
    failed_ground_station_count: int
    operational_ground_station_count: int
    failed_ground_station_ids: tuple[str, ...]
    operational_ground_station_ids: tuple[str, ...]
    baseline_serviced_ground_station_count: int
    baseline_ground_service_fraction: float
    baseline_overall_service_fraction: float
    failure_adjusted_serviced_ground_station_count: int
    failure_adjusted_unserviced_ground_station_count: int
    failure_adjusted_serviced_ground_station_ids: tuple[str, ...]
    failure_adjusted_unserviced_ground_station_ids: tuple[str, ...]
    total_civilian_count: int
    failed_civilian_count: int
    operational_civilian_count: int
    failure_adjusted_serviced_civilian_count: int
    failure_adjusted_civilian_service_fraction: float | None
    total_government_count: int
    failed_government_count: int
    operational_government_count: int
    failure_adjusted_serviced_government_count: int
    failure_adjusted_government_service_fraction: float | None
    total_military_count: int
    failed_military_count: int
    operational_military_count: int
    failure_adjusted_serviced_military_count: int
    failure_adjusted_military_service_fraction: float | None
    space_gcc_fraction_original: float
    space_gcc_fraction_surviving: float
    failure_adjusted_ground_service_fraction: float
    failure_adjusted_overall_service_fraction: float
    ground_service_loss_due_to_failures: float
    overall_service_loss_due_to_ground_failures: float
    space_threshold_met: bool
    ground_threshold_met: bool
    overall_threshold_met: bool
    failure_adjusted_step_hash: str

    def __post_init__(self) -> None:
        _validate_step_intrinsic(self)

    def scientific_manifest_object(self) -> dict[str, object]:
        return _step_payload(self)


_FLOAT_FIELDS = frozenset(
    {
        "baseline_ground_service_fraction",
        "baseline_overall_service_fraction",
        "failure_adjusted_civilian_service_fraction",
        "failure_adjusted_government_service_fraction",
        "failure_adjusted_military_service_fraction",
        "space_gcc_fraction_original",
        "space_gcc_fraction_surviving",
        "failure_adjusted_ground_service_fraction",
        "failure_adjusted_overall_service_fraction",
        "ground_service_loss_due_to_failures",
        "overall_service_loss_due_to_ground_failures",
    }
)
_TUPLE_FIELDS = frozenset(
    {
        "satellite_gcc_ids",
        "failed_ground_station_ids",
        "operational_ground_station_ids",
        "failure_adjusted_serviced_ground_station_ids",
        "failure_adjusted_unserviced_ground_station_ids",
    }
)
_STEP_FIELDS = tuple(
    field.name
    for field in fields(FailureAdjustedGroundServiceStepMetrics)
    if field.name != "failure_adjusted_step_hash"
)


def _source_value(
    source: FailureAdjustedGroundServiceStepMetrics | Mapping[str, object], name: str
) -> object:
    return source[name] if isinstance(source, Mapping) else getattr(source, name)


def _step_payload(
    source: FailureAdjustedGroundServiceStepMetrics | Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "identity_domain": GROUND_FAILURE_SERVICE_STEP_IDENTITY_DOMAIN,
        "identity_version": GROUND_FAILURE_SERVICE_STEP_IDENTITY_VERSION,
    }
    for name in _STEP_FIELDS:
        value = _source_value(source, name)
        if name == "timestamp_utc":
            payload[name] = canonical_utc_timestamp(value)
        elif name in _FLOAT_FIELDS:
            payload[name] = None if value is None else canonical_float_string(value)
        elif name in _TUPLE_FIELDS:
            payload[name] = list(value)
        else:
            payload[name] = value
    return payload


def _compute_failure_adjusted_step_hash(source: Mapping[str, object]) -> str:
    return canonical_hash(_step_payload(source))


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


def _validate_station_ids(values: object, field_name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if any(
        not isinstance(value, str) or not STATION_ID_PATTERN.fullmatch(value)
        for value in values
    ):
        raise ValueError(f"{field_name} contains an invalid station ID")
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{field_name} must use unique ascending station IDs")


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("G5 service denominators must be positive")
    return numerator / denominator


def _positive_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _validate_class(
    *,
    class_name: str,
    total: int,
    failed: int,
    operational: int,
    serviced: int,
    fraction: float | None,
) -> None:
    for field_name, value in (
        (f"total_{class_name}_count", total),
        (f"failed_{class_name}_count", failed),
        (f"operational_{class_name}_count", operational),
        (f"failure_adjusted_serviced_{class_name}_count", serviced),
    ):
        _validate_count(value, field_name)
    if failed + operational != total:
        raise ValueError(f"{class_name} failure counts do not sum to class total")
    if not 0 <= serviced <= operational <= total:
        raise ValueError(f"{class_name} adjusted service counts are invalid")
    if total == 0:
        if fraction is not None:
            raise ValueError(
                f"failure_adjusted_{class_name}_service_fraction must be None for an absent class"
            )
    else:
        _validate_fraction(
            fraction, f"failure_adjusted_{class_name}_service_fraction"
        )
        if fraction != _ratio(serviced, total):
            raise ValueError(f"{class_name} adjusted fraction does not match counts")


def _validate_step_intrinsic(metrics: FailureAdjustedGroundServiceStepMetrics) -> None:
    _validate_count(metrics.timestep_index, "timestep_index")
    canonical_utc_timestamp(metrics.timestamp_utc)
    for field_name in (
        "satellite_config_hash",
        "ground_design_hash",
        "visibility_policy_hash",
        "integrated_graph_hash",
        "ground_service_policy_hash",
        "baseline_ground_service_step_hash",
        "ground_failure_policy_hash",
        "ground_failure_realization_hash",
        "failure_adjusted_step_hash",
    ):
        _validate_hash(getattr(metrics, field_name), field_name)
    if (
        metrics.ground_failure_service_model_version
        != GROUND_FAILURE_SERVICE_MODEL_VERSION
    ):
        raise ValueError("Unsupported ground_failure_service_model_version")
    if type(metrics.configured_satellite_count) is not int:
        raise TypeError("configured_satellite_count must be a positive integer")
    if metrics.configured_satellite_count <= 0:
        raise ValueError("configured_satellite_count must be positive")
    count_fields = (
        "operational_satellite_count",
        "satellite_component_count",
        "satellite_gcc_size",
        "total_ground_station_count",
        "failed_ground_station_count",
        "operational_ground_station_count",
        "baseline_serviced_ground_station_count",
        "failure_adjusted_serviced_ground_station_count",
        "failure_adjusted_unserviced_ground_station_count",
    )
    for field_name in count_fields:
        _validate_count(getattr(metrics, field_name), field_name)
    if metrics.total_ground_station_count <= 0:
        raise ValueError("total_ground_station_count must be positive")
    if metrics.operational_satellite_count > metrics.configured_satellite_count:
        raise ValueError("operational_satellite_count exceeds configured count")
    if metrics.satellite_gcc_size > metrics.operational_satellite_count:
        raise ValueError("satellite_gcc_size exceeds operational satellite count")
    if not isinstance(metrics.satellite_gcc_ids, tuple) or any(
        type(value) is not int or value < 0 for value in metrics.satellite_gcc_ids
    ):
        raise TypeError("satellite_gcc_ids must contain nonnegative integers")
    if metrics.satellite_gcc_ids != tuple(sorted(metrics.satellite_gcc_ids)) or len(
        metrics.satellite_gcc_ids
    ) != len(set(metrics.satellite_gcc_ids)):
        raise ValueError("satellite_gcc_ids must use unique ascending IDs")
    if len(metrics.satellite_gcc_ids) != metrics.satellite_gcc_size:
        raise ValueError("satellite_gcc_size does not match satellite_gcc_ids")
    if any(value >= metrics.configured_satellite_count for value in metrics.satellite_gcc_ids):
        raise ValueError("satellite_gcc_ids exceed configured range")
    if metrics.operational_satellite_count == 0:
        if (
            metrics.satellite_component_count != 0
            or metrics.satellite_gcc_size != 0
            or metrics.satellite_gcc_ids
        ):
            raise ValueError("Zero operational satellites require an empty component state")
    elif not 1 <= metrics.satellite_component_count <= metrics.operational_satellite_count:
        raise ValueError("satellite_component_count is invalid")
    for field_name in (
        "failed_ground_station_ids",
        "operational_ground_station_ids",
        "failure_adjusted_serviced_ground_station_ids",
        "failure_adjusted_unserviced_ground_station_ids",
    ):
        _validate_station_ids(getattr(metrics, field_name), field_name)
    failed = set(metrics.failed_ground_station_ids)
    operational = set(metrics.operational_ground_station_ids)
    serviced = set(metrics.failure_adjusted_serviced_ground_station_ids)
    unserviced = set(metrics.failure_adjusted_unserviced_ground_station_ids)
    selected = failed | operational
    if failed & operational or len(selected) != metrics.total_ground_station_count:
        raise ValueError("Failed and operational IDs must partition selected stations")
    if len(failed) != metrics.failed_ground_station_count:
        raise ValueError("failed_ground_station_count does not match IDs")
    if len(operational) != metrics.operational_ground_station_count:
        raise ValueError("operational_ground_station_count does not match IDs")
    if (
        metrics.failed_ground_station_count
        + metrics.operational_ground_station_count
        != metrics.total_ground_station_count
    ):
        raise ValueError("Failure counts do not sum to total ground count")
    if serviced & unserviced or serviced | unserviced != selected:
        raise ValueError("Adjusted serviced and unserviced IDs must partition selection")
    if len(serviced) != metrics.failure_adjusted_serviced_ground_station_count:
        raise ValueError("Adjusted serviced count does not match IDs")
    if len(unserviced) != metrics.failure_adjusted_unserviced_ground_station_count:
        raise ValueError("Adjusted unserviced count does not match IDs")
    if not serviced <= operational:
        raise ValueError("Failed stations cannot receive adjusted service")
    if (
        metrics.failure_adjusted_serviced_ground_station_count
        > metrics.baseline_serviced_ground_station_count
    ):
        raise ValueError("Adjusted serviced count exceeds baseline serviced count")
    for class_name in ("civilian", "government", "military"):
        _validate_class(
            class_name=class_name,
            total=getattr(metrics, f"total_{class_name}_count"),
            failed=getattr(metrics, f"failed_{class_name}_count"),
            operational=getattr(metrics, f"operational_{class_name}_count"),
            serviced=getattr(
                metrics, f"failure_adjusted_serviced_{class_name}_count"
            ),
            fraction=getattr(
                metrics, f"failure_adjusted_{class_name}_service_fraction"
            ),
        )
    if sum(
        getattr(metrics, f"total_{name}_count")
        for name in ("civilian", "government", "military")
    ) != metrics.total_ground_station_count:
        raise ValueError("Class totals do not sum to total ground count")
    if sum(
        getattr(metrics, f"failed_{name}_count")
        for name in ("civilian", "government", "military")
    ) != metrics.failed_ground_station_count:
        raise ValueError("Class failed counts do not sum to failed ground count")
    if sum(
        getattr(metrics, f"operational_{name}_count")
        for name in ("civilian", "government", "military")
    ) != metrics.operational_ground_station_count:
        raise ValueError("Class operational counts do not sum to operational ground count")
    if sum(
        getattr(metrics, f"failure_adjusted_serviced_{name}_count")
        for name in ("civilian", "government", "military")
    ) != metrics.failure_adjusted_serviced_ground_station_count:
        raise ValueError("Class adjusted service counts do not sum to adjusted total")
    for field_name in _FLOAT_FIELDS:
        value = getattr(metrics, field_name)
        if value is not None:
            _validate_fraction(value, field_name)
    if metrics.space_gcc_fraction_original != (
        metrics.satellite_gcc_size / metrics.configured_satellite_count
    ):
        raise ValueError("Original space fraction does not match satellite counts")
    expected_surviving = (
        0.0
        if metrics.operational_satellite_count == 0
        else metrics.satellite_gcc_size / metrics.operational_satellite_count
    )
    if metrics.space_gcc_fraction_surviving != expected_surviving:
        raise ValueError("Surviving space fraction does not match satellite counts")
    if metrics.baseline_ground_service_fraction != _ratio(
        metrics.baseline_serviced_ground_station_count,
        metrics.total_ground_station_count,
    ):
        raise ValueError("Baseline ground fraction does not match counts")
    if metrics.baseline_overall_service_fraction != min(
        metrics.space_gcc_fraction_original,
        metrics.baseline_ground_service_fraction,
    ):
        raise ValueError("Baseline overall fraction does not match bottleneck")
    if metrics.failure_adjusted_ground_service_fraction != _ratio(
        metrics.failure_adjusted_serviced_ground_station_count,
        metrics.total_ground_station_count,
    ):
        raise ValueError("Adjusted ground fraction does not match counts")
    if metrics.failure_adjusted_overall_service_fraction != min(
        metrics.space_gcc_fraction_original,
        metrics.failure_adjusted_ground_service_fraction,
    ):
        raise ValueError("Adjusted overall fraction does not match bottleneck")
    if metrics.ground_service_loss_due_to_failures != _positive_zero(
        metrics.baseline_ground_service_fraction
        - metrics.failure_adjusted_ground_service_fraction
    ):
        raise ValueError("Ground service loss does not match fractions")
    if metrics.overall_service_loss_due_to_ground_failures != _positive_zero(
        metrics.baseline_overall_service_fraction
        - metrics.failure_adjusted_overall_service_fraction
    ):
        raise ValueError("Overall service loss does not match fractions")
    if metrics.failure_adjusted_ground_service_fraction > metrics.baseline_ground_service_fraction:
        raise ValueError("Ground failures cannot improve ground service")
    if metrics.failure_adjusted_overall_service_fraction > metrics.baseline_overall_service_fraction:
        raise ValueError("Ground failures cannot improve overall service")
    for field_name in ("space_threshold_met", "ground_threshold_met", "overall_threshold_met"):
        if type(getattr(metrics, field_name)) is not bool:
            raise TypeError(f"{field_name} must be a Boolean")
    if metrics.overall_threshold_met != (
        metrics.space_threshold_met and metrics.ground_threshold_met
    ):
        raise ValueError("overall_threshold_met must be the conjunction of component states")
    if metrics.failure_adjusted_step_hash != canonical_hash(_step_payload(metrics)):
        raise ValueError("failure_adjusted_step_hash does not match canonical step")


def _build_step_values(
    *,
    baseline_metrics: GroundServiceStepMetrics,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    failure_realization: GroundFailureRealization,
    ground_service_policy: GroundServicePolicy,
) -> dict[str, object]:
    if not isinstance(baseline_metrics, GroundServiceStepMetrics):
        raise TypeError("baseline_metrics must be GroundServiceStepMetrics")
    if not isinstance(ground_design, GroundRunDesignRecord):
        raise TypeError("ground_design must be a GroundRunDesignRecord")
    if not isinstance(catalog, GroundStationCatalog):
        raise TypeError("catalog must be a GroundStationCatalog")
    if not isinstance(failure_realization, GroundFailureRealization):
        raise TypeError("failure_realization must be GroundFailureRealization")
    if not isinstance(ground_service_policy, GroundServicePolicy):
        raise TypeError("ground_service_policy must be GroundServicePolicy")
    if not ground_design.ground_segment_enabled:
        raise ValueError("G5 requires an enabled ground design")
    selection = reconstruct_ground_selection(ground_design, catalog)
    if selection is None or not selection.selected_station_ids:
        raise ValueError("G5 requires a nonempty ground selection")
    selected = set(selection.selected_station_ids)
    realization_selected = set(failure_realization.selected_station_ids)
    if len(selected) != len(realization_selected) or selected != realization_selected:
        raise ValueError("Failure realization does not match exact G1 population")
    if failure_realization.ground_design_hash != ground_design.ground_design_hash:
        raise ValueError("Failure realization does not match G1 ground design")
    if baseline_metrics.ground_design_hash != ground_design.ground_design_hash:
        raise ValueError("Baseline G4 step does not match G1 ground design")
    if (
        baseline_metrics.ground_service_policy_hash
        != ground_service_policy.ground_service_policy_hash
    ):
        raise ValueError("Baseline G4 step does not match ground-service policy")
    if baseline_metrics.total_ground_station_count != len(selected):
        raise ValueError("Baseline G4 ground count does not match G1 selection")
    baseline_serviced = set(baseline_metrics.serviced_ground_station_ids)
    baseline_unserviced = set(baseline_metrics.unserviced_ground_station_ids)
    if baseline_serviced & baseline_unserviced or baseline_serviced | baseline_unserviced != selected:
        raise ValueError("Baseline G4 station sets do not partition G1 selection")
    failed = set(failure_realization.failed_station_ids)
    operational = set(failure_realization.operational_station_ids)
    adjusted_serviced = baseline_serviced - failed
    adjusted_unserviced = selected - adjusted_serviced
    class_ids = {
        "civilian": set(selection.civilian_station_ids),
        "government": set(selection.government_station_ids),
        "military": set(selection.military_station_ids),
    }
    class_total = {name: len(ids) for name, ids in class_ids.items()}
    class_failed = {name: len(ids & failed) for name, ids in class_ids.items()}
    class_operational = {name: len(ids & operational) for name, ids in class_ids.items()}
    class_serviced = {name: len(ids & adjusted_serviced) for name, ids in class_ids.items()}
    class_fraction = {
        name: None
        if class_total[name] == 0
        else class_serviced[name] / class_total[name]
        for name in class_ids
    }
    adjusted_ground = len(adjusted_serviced) / len(selected)
    adjusted_overall = min(
        baseline_metrics.space_gcc_fraction_original, adjusted_ground
    )
    ground_met = adjusted_ground >= ground_service_policy.ground_service_threshold
    values: dict[str, object] = {
        "timestep_index": baseline_metrics.timestep_index,
        "timestamp_utc": baseline_metrics.timestamp_utc,
        "satellite_config_hash": baseline_metrics.satellite_config_hash,
        "ground_design_hash": baseline_metrics.ground_design_hash,
        "visibility_policy_hash": baseline_metrics.visibility_policy_hash,
        "integrated_graph_hash": baseline_metrics.integrated_graph_hash,
        "ground_service_policy_hash": baseline_metrics.ground_service_policy_hash,
        "baseline_ground_service_step_hash": baseline_metrics.step_metrics_hash,
        "ground_failure_policy_hash": failure_realization.ground_failure_policy_hash,
        "ground_failure_realization_hash": failure_realization.ground_failure_realization_hash,
        "ground_failure_service_model_version": GROUND_FAILURE_SERVICE_MODEL_VERSION,
        "configured_satellite_count": baseline_metrics.configured_satellite_count,
        "operational_satellite_count": baseline_metrics.operational_satellite_count,
        "satellite_component_count": baseline_metrics.satellite_component_count,
        "satellite_gcc_size": baseline_metrics.satellite_gcc_size,
        "satellite_gcc_ids": baseline_metrics.satellite_gcc_ids,
        "total_ground_station_count": len(selected),
        "failed_ground_station_count": len(failed),
        "operational_ground_station_count": len(operational),
        "failed_ground_station_ids": tuple(sorted(failed)),
        "operational_ground_station_ids": tuple(sorted(operational)),
        "baseline_serviced_ground_station_count": len(baseline_serviced),
        "baseline_ground_service_fraction": baseline_metrics.ground_service_fraction,
        "baseline_overall_service_fraction": baseline_metrics.overall_service_fraction,
        "failure_adjusted_serviced_ground_station_count": len(adjusted_serviced),
        "failure_adjusted_unserviced_ground_station_count": len(adjusted_unserviced),
        "failure_adjusted_serviced_ground_station_ids": tuple(sorted(adjusted_serviced)),
        "failure_adjusted_unserviced_ground_station_ids": tuple(sorted(adjusted_unserviced)),
        "space_gcc_fraction_original": baseline_metrics.space_gcc_fraction_original,
        "space_gcc_fraction_surviving": baseline_metrics.space_gcc_fraction_surviving,
        "failure_adjusted_ground_service_fraction": adjusted_ground,
        "failure_adjusted_overall_service_fraction": adjusted_overall,
        "ground_service_loss_due_to_failures": _positive_zero(
            baseline_metrics.ground_service_fraction - adjusted_ground
        ),
        "overall_service_loss_due_to_ground_failures": _positive_zero(
            baseline_metrics.overall_service_fraction - adjusted_overall
        ),
        "space_threshold_met": baseline_metrics.space_threshold_met,
        "ground_threshold_met": ground_met,
        "overall_threshold_met": baseline_metrics.space_threshold_met and ground_met,
    }
    for name in ("civilian", "government", "military"):
        values[f"total_{name}_count"] = class_total[name]
        values[f"failed_{name}_count"] = class_failed[name]
        values[f"operational_{name}_count"] = class_operational[name]
        values[f"failure_adjusted_serviced_{name}_count"] = class_serviced[name]
        values[f"failure_adjusted_{name}_service_fraction"] = class_fraction[name]
    return values


def compute_failure_adjusted_ground_service_step(
    *,
    baseline_metrics: GroundServiceStepMetrics,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    failure_realization: GroundFailureRealization,
    ground_service_policy: GroundServicePolicy,
) -> FailureAdjustedGroundServiceStepMetrics:
    values = _build_step_values(
        baseline_metrics=baseline_metrics,
        ground_design=ground_design,
        catalog=catalog,
        failure_realization=failure_realization,
        ground_service_policy=ground_service_policy,
    )
    values["failure_adjusted_step_hash"] = _compute_failure_adjusted_step_hash(values)
    metrics = FailureAdjustedGroundServiceStepMetrics(**values)
    validate_failure_adjusted_ground_service_step_context(
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        ground_design=ground_design,
        catalog=catalog,
        failure_realization=failure_realization,
        ground_service_policy=ground_service_policy,
    )
    return metrics


def validate_failure_adjusted_ground_service_step_context(
    *,
    metrics: FailureAdjustedGroundServiceStepMetrics,
    baseline_metrics: GroundServiceStepMetrics,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    failure_realization: GroundFailureRealization,
    ground_service_policy: GroundServicePolicy,
) -> None:
    if not isinstance(metrics, FailureAdjustedGroundServiceStepMetrics):
        raise TypeError("metrics must be FailureAdjustedGroundServiceStepMetrics")
    values = _build_step_values(
        baseline_metrics=baseline_metrics,
        ground_design=ground_design,
        catalog=catalog,
        failure_realization=failure_realization,
        ground_service_policy=ground_service_policy,
    )
    values["failure_adjusted_step_hash"] = _compute_failure_adjusted_step_hash(values)
    for field in fields(FailureAdjustedGroundServiceStepMetrics):
        if getattr(metrics, field.name) != values[field.name]:
            raise ValueError(f"Failure-adjusted step contextual mismatch: {field.name}")
    baseline_class_counts = {
        "civilian": baseline_metrics.serviced_civilian_count,
        "government": baseline_metrics.serviced_government_count,
        "military": baseline_metrics.serviced_military_count,
    }
    for name, baseline_count in baseline_class_counts.items():
        if getattr(metrics, f"failure_adjusted_serviced_{name}_count") > baseline_count:
            raise ValueError(f"Adjusted {name} service exceeds baseline G4 service")
    if metrics.ground_threshold_met and not baseline_metrics.ground_threshold_met:
        raise ValueError("Ground threshold compliance cannot improve after failures")
    if metrics.overall_threshold_met and not baseline_metrics.overall_threshold_met:
        raise ValueError("Overall threshold compliance cannot improve after failures")


def _step_record_hash(
    *, run_id: int, metrics: FailureAdjustedGroundServiceStepMetrics
) -> str:
    return canonical_hash(
        {
            "failure_adjusted_step_hash": metrics.failure_adjusted_step_hash,
            "ground_failure_service_step_schema_version": GROUND_FAILURE_SERVICE_STEP_SCHEMA_VERSION,
            "identity_domain": GROUND_FAILURE_SERVICE_STEP_RECORD_IDENTITY_DOMAIN,
            "identity_version": GROUND_FAILURE_SERVICE_STEP_RECORD_IDENTITY_VERSION,
            "run_id": run_id,
            "timestep_index": metrics.timestep_index,
            "timestamp_utc": canonical_utc_timestamp(metrics.timestamp_utc),
        }
    )


@dataclass(frozen=True)
class FailureAdjustedGroundServiceStepRecord:
    ground_failure_service_step_schema_version: str
    run_id: int
    metrics: FailureAdjustedGroundServiceStepMetrics
    record_hash: str

    def __post_init__(self) -> None:
        if (
            self.ground_failure_service_step_schema_version
            != GROUND_FAILURE_SERVICE_STEP_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported ground_failure_service_step_schema_version")
        if type(self.run_id) is not int or self.run_id < 0:
            raise TypeError("run_id must be a nonnegative integer")
        if not isinstance(self.metrics, FailureAdjustedGroundServiceStepMetrics):
            raise TypeError("metrics must be FailureAdjustedGroundServiceStepMetrics")
        if self.record_hash != _step_record_hash(run_id=self.run_id, metrics=self.metrics):
            raise ValueError("record_hash does not match canonical adjusted step record")


def create_failure_adjusted_step_record(
    *,
    run_id: int,
    verified_g4_step_record: GroundServiceStepRecord,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    policy: GroundFailurePolicy,
    realization_record: GroundFailureRealizationRecord,
    ground_service_policy: GroundServicePolicy,
) -> FailureAdjustedGroundServiceStepRecord:
    if type(run_id) is not int or run_id < 0:
        raise TypeError("run_id must be a nonnegative integer")
    if not isinstance(verified_g4_step_record, GroundServiceStepRecord):
        raise TypeError("verified_g4_step_record must be GroundServiceStepRecord")
    if not isinstance(realization_record, GroundFailureRealizationRecord):
        raise TypeError("realization_record must be GroundFailureRealizationRecord")
    if not (
        run_id
        == verified_g4_step_record.run_id
        == realization_record.run_id
        == ground_design.run_id
    ):
        raise ValueError("G5 step sources contain mismatched run IDs")
    validate_ground_failure_realization_record_context(
        record=realization_record,
        ground_design=ground_design,
        catalog=catalog,
        policy=policy,
    )
    metrics = compute_failure_adjusted_ground_service_step(
        baseline_metrics=verified_g4_step_record.metrics,
        ground_design=ground_design,
        catalog=catalog,
        failure_realization=realization_record.realization,
        ground_service_policy=ground_service_policy,
    )
    validate_failure_adjusted_ground_service_step_context(
        metrics=metrics,
        baseline_metrics=verified_g4_step_record.metrics,
        ground_design=ground_design,
        catalog=catalog,
        failure_realization=realization_record.realization,
        ground_service_policy=ground_service_policy,
    )
    return FailureAdjustedGroundServiceStepRecord(
        ground_failure_service_step_schema_version=GROUND_FAILURE_SERVICE_STEP_SCHEMA_VERSION,
        run_id=run_id,
        metrics=metrics,
        record_hash=_step_record_hash(run_id=run_id, metrics=metrics),
    )
