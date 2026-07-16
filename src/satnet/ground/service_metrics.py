from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
import math
import re
from typing import Mapping

import networkx as nx

from satnet.ground.canonical import (
    canonical_float_string,
    canonical_hash,
    canonical_utc_timestamp,
)
from satnet.ground.catalog import GroundStationCatalog, STATION_ID_PATTERN
from satnet.ground.integrated_graph import (
    IntegratedEdgeKind,
    IntegratedGroundGraphSnapshot,
    IntegratedNodeKind,
    project_satellite_subgraph,
)
from satnet.ground.persistence import GroundRunDesignRecord, reconstruct_ground_selection
from satnet.ground.service_policy import GROUND_SERVICE_MODEL_VERSION, GroundServicePolicy

GROUND_SERVICE_STEP_IDENTITY_DOMAIN = "satnet_ground_service_step"
GROUND_SERVICE_STEP_IDENTITY_VERSION = "1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GroundServiceStepMetrics:
    timestep_index: int
    timestamp_utc: datetime
    satellite_config_hash: str
    ground_design_hash: str
    visibility_policy_hash: str
    integrated_graph_hash: str
    ground_service_policy_hash: str
    ground_service_model_version: str
    configured_satellite_count: int
    operational_satellite_count: int
    satellite_component_count: int
    satellite_gcc_size: int
    satellite_gcc_ids: tuple[int, ...]
    total_ground_station_count: int
    serviced_ground_station_count: int
    unserviced_ground_station_count: int
    serviced_ground_station_ids: tuple[str, ...]
    unserviced_ground_station_ids: tuple[str, ...]
    total_civilian_count: int
    serviced_civilian_count: int
    civilian_service_fraction: float | None
    total_government_count: int
    serviced_government_count: int
    government_service_fraction: float | None
    total_military_count: int
    serviced_military_count: int
    military_service_fraction: float | None
    space_gcc_fraction_original: float
    space_gcc_fraction_surviving: float
    ground_service_fraction: float
    overall_service_fraction: float
    space_threshold_met: bool
    ground_threshold_met: bool
    overall_threshold_met: bool
    step_metrics_hash: str

    def __post_init__(self) -> None:
        _validate_step_metrics_intrinsic(self)

    def scientific_manifest_object(self) -> dict[str, object]:
        return _step_metrics_payload(self)


_FLOAT_FIELDS = frozenset(
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
_TUPLE_FIELDS = frozenset(
    {
        "satellite_gcc_ids",
        "serviced_ground_station_ids",
        "unserviced_ground_station_ids",
    }
)
_STEP_FIELDS = tuple(
    field.name for field in fields(GroundServiceStepMetrics) if field.name != "step_metrics_hash"
)


def _source_value(source: GroundServiceStepMetrics | Mapping[str, object], name: str) -> object:
    if isinstance(source, Mapping):
        return source[name]
    return getattr(source, name)


def _step_metrics_payload(
    source: GroundServiceStepMetrics | Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "identity_domain": GROUND_SERVICE_STEP_IDENTITY_DOMAIN,
        "identity_version": GROUND_SERVICE_STEP_IDENTITY_VERSION,
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


def _compute_step_metrics_hash(source: Mapping[str, object]) -> str:
    return canonical_hash(_step_metrics_payload(source))


def _validate_hash(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be exactly 64 lowercase hexadecimal characters")


def _validate_exact_nonnegative_integer(value: object, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise TypeError(f"{field_name} must be a nonnegative integer")


def _validate_fraction(value: object, field_name: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{field_name} must be a float")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be finite and within [0.0, 1.0]")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{field_name} must use normalized positive zero")


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _validate_class_values(
    *, total: int, serviced: int, fraction: float | None, class_name: str
) -> None:
    _validate_exact_nonnegative_integer(total, f"total_{class_name}_count")
    _validate_exact_nonnegative_integer(serviced, f"serviced_{class_name}_count")
    if serviced > total:
        raise ValueError(f"serviced_{class_name}_count exceeds class total")
    if total == 0:
        if fraction is not None:
            raise ValueError(f"{class_name}_service_fraction must be None for an absent class")
    else:
        _validate_fraction(fraction, f"{class_name}_service_fraction")
        if fraction != _ratio(serviced, total):
            raise ValueError(f"{class_name}_service_fraction does not match class counts")


def _validate_station_ids(values: object, field_name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if any(not isinstance(value, str) or not STATION_ID_PATTERN.fullmatch(value) for value in values):
        raise ValueError(f"{field_name} contains an invalid station ID")
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{field_name} must use unique ascending station IDs")


def _validate_step_metrics_intrinsic(metrics: GroundServiceStepMetrics) -> None:
    _validate_exact_nonnegative_integer(metrics.timestep_index, "timestep_index")
    canonical_utc_timestamp(metrics.timestamp_utc)
    for field_name in (
        "satellite_config_hash",
        "ground_design_hash",
        "visibility_policy_hash",
        "integrated_graph_hash",
        "ground_service_policy_hash",
        "step_metrics_hash",
    ):
        _validate_hash(getattr(metrics, field_name), field_name)
    if metrics.ground_service_model_version != GROUND_SERVICE_MODEL_VERSION:
        raise ValueError("Unsupported ground_service_model_version")
    if type(metrics.configured_satellite_count) is not int:
        raise TypeError("configured_satellite_count must be a positive integer")
    if metrics.configured_satellite_count <= 0:
        raise ValueError("configured_satellite_count must be positive")
    for field_name in (
        "operational_satellite_count",
        "satellite_component_count",
        "satellite_gcc_size",
        "total_ground_station_count",
        "serviced_ground_station_count",
        "unserviced_ground_station_count",
    ):
        _validate_exact_nonnegative_integer(getattr(metrics, field_name), field_name)
    if metrics.operational_satellite_count > metrics.configured_satellite_count:
        raise ValueError("operational_satellite_count exceeds configured_satellite_count")
    if metrics.satellite_gcc_size > metrics.operational_satellite_count:
        raise ValueError("satellite_gcc_size exceeds operational_satellite_count")
    if not isinstance(metrics.satellite_gcc_ids, tuple) or any(
        type(value) is not int or value < 0 for value in metrics.satellite_gcc_ids
    ):
        raise TypeError("satellite_gcc_ids must contain nonnegative integers")
    if metrics.satellite_gcc_ids != tuple(sorted(metrics.satellite_gcc_ids)) or len(
        metrics.satellite_gcc_ids
    ) != len(set(metrics.satellite_gcc_ids)):
        raise ValueError("satellite_gcc_ids must use unique ascending integer IDs")
    if len(metrics.satellite_gcc_ids) != metrics.satellite_gcc_size:
        raise ValueError("satellite_gcc_size does not match satellite_gcc_ids")
    if any(value >= metrics.configured_satellite_count for value in metrics.satellite_gcc_ids):
        raise ValueError("satellite_gcc_ids exceed the configured satellite range")
    if metrics.operational_satellite_count == 0:
        if (
            metrics.satellite_component_count != 0
            or metrics.satellite_gcc_size != 0
            or metrics.satellite_gcc_ids
        ):
            raise ValueError("Zero operational satellites require an empty component state")
    elif not 1 <= metrics.satellite_component_count <= metrics.operational_satellite_count:
        raise ValueError("satellite_component_count is invalid for operational satellites")
    if metrics.total_ground_station_count <= 0:
        raise ValueError("total_ground_station_count must be positive")
    _validate_station_ids(metrics.serviced_ground_station_ids, "serviced_ground_station_ids")
    _validate_station_ids(metrics.unserviced_ground_station_ids, "unserviced_ground_station_ids")
    if set(metrics.serviced_ground_station_ids) & set(metrics.unserviced_ground_station_ids):
        raise ValueError("Serviced and unserviced station IDs must be disjoint")
    if len(metrics.serviced_ground_station_ids) != metrics.serviced_ground_station_count:
        raise ValueError("serviced_ground_station_count does not match station IDs")
    if len(metrics.unserviced_ground_station_ids) != metrics.unserviced_ground_station_count:
        raise ValueError("unserviced_ground_station_count does not match station IDs")
    if (
        metrics.serviced_ground_station_count + metrics.unserviced_ground_station_count
        != metrics.total_ground_station_count
    ):
        raise ValueError("Ground service counts do not sum to the selected total")
    _validate_class_values(
        total=metrics.total_civilian_count,
        serviced=metrics.serviced_civilian_count,
        fraction=metrics.civilian_service_fraction,
        class_name="civilian",
    )
    _validate_class_values(
        total=metrics.total_government_count,
        serviced=metrics.serviced_government_count,
        fraction=metrics.government_service_fraction,
        class_name="government",
    )
    _validate_class_values(
        total=metrics.total_military_count,
        serviced=metrics.serviced_military_count,
        fraction=metrics.military_service_fraction,
        class_name="military",
    )
    if (
        metrics.total_civilian_count
        + metrics.total_government_count
        + metrics.total_military_count
        != metrics.total_ground_station_count
    ):
        raise ValueError("Class totals do not sum to total_ground_station_count")
    if (
        metrics.serviced_civilian_count
        + metrics.serviced_government_count
        + metrics.serviced_military_count
        != metrics.serviced_ground_station_count
    ):
        raise ValueError("Serviced class counts do not sum to serviced_ground_station_count")
    for field_name in (
        "space_gcc_fraction_original",
        "space_gcc_fraction_surviving",
        "ground_service_fraction",
        "overall_service_fraction",
    ):
        _validate_fraction(getattr(metrics, field_name), field_name)
    if metrics.space_gcc_fraction_original != _ratio(
        metrics.satellite_gcc_size, metrics.configured_satellite_count
    ):
        raise ValueError("space_gcc_fraction_original does not match satellite counts")
    if metrics.space_gcc_fraction_surviving != _ratio(
        metrics.satellite_gcc_size, metrics.operational_satellite_count
    ):
        raise ValueError("space_gcc_fraction_surviving does not match satellite counts")
    if metrics.ground_service_fraction != _ratio(
        metrics.serviced_ground_station_count, metrics.total_ground_station_count
    ):
        raise ValueError("ground_service_fraction does not match ground counts")
    if metrics.overall_service_fraction != min(
        metrics.space_gcc_fraction_original, metrics.ground_service_fraction
    ):
        raise ValueError("overall_service_fraction does not match bottleneck definition")
    for field_name in ("space_threshold_met", "ground_threshold_met", "overall_threshold_met"):
        if type(getattr(metrics, field_name)) is not bool:
            raise TypeError(f"{field_name} must be a Boolean")
    if metrics.overall_threshold_met != (
        metrics.space_threshold_met and metrics.ground_threshold_met
    ):
        raise ValueError("overall_threshold_met must be the conjunction of component states")
    expected_hash = canonical_hash(_step_metrics_payload(metrics))
    if metrics.step_metrics_hash != expected_hash:
        raise ValueError("step_metrics_hash does not match canonical step metrics")


def _select_authoritative_gcc(graph: nx.Graph) -> tuple[tuple[int, ...], ...]:
    components = tuple(
        sorted(tuple(sorted(component)) for component in nx.connected_components(graph))
    )
    return components


def _build_step_values(
    *,
    integrated_snapshot: IntegratedGroundGraphSnapshot,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    configured_satellite_count: int,
    policy: GroundServicePolicy,
) -> dict[str, object]:
    if not isinstance(integrated_snapshot, IntegratedGroundGraphSnapshot):
        raise TypeError("integrated_snapshot must be an IntegratedGroundGraphSnapshot")
    if not isinstance(ground_design, GroundRunDesignRecord):
        raise TypeError("ground_design must be a GroundRunDesignRecord")
    if not isinstance(catalog, GroundStationCatalog):
        raise TypeError("catalog must be a GroundStationCatalog")
    if not isinstance(policy, GroundServicePolicy):
        raise TypeError("policy must be a GroundServicePolicy")
    if type(configured_satellite_count) is not int:
        raise TypeError("configured_satellite_count must be a positive integer")
    if configured_satellite_count <= 0:
        raise ValueError("configured_satellite_count must be positive")
    if integrated_snapshot.satellite_config_hash != ground_design.satellite_config_hash:
        raise ValueError("Integrated snapshot satellite configuration does not match G1 design")
    if integrated_snapshot.ground_design_hash != ground_design.ground_design_hash:
        raise ValueError("Integrated snapshot ground design does not match G1 design")
    selection = reconstruct_ground_selection(ground_design, catalog)
    if selection is None or selection.total_ground_station_count <= 0:
        raise ValueError("G4 requires an enabled nonempty ground selection")
    selected_ids = set(selection.selected_station_ids)
    graph_ground_ids = {
        node.node_ref.ground_station_id
        for node in integrated_snapshot.canonical_nodes
        if node.node_ref.kind is IntegratedNodeKind.GROUND_STATION
    }
    if graph_ground_ids != selected_ids:
        raise ValueError("Integrated ground nodes do not match the exact G1 selection")
    satellite_graph = project_satellite_subgraph(integrated_snapshot)
    operational_ids = set(satellite_graph.nodes)
    if any(
        type(satellite_id) is not int
        or satellite_id < 0
        or satellite_id >= configured_satellite_count
        for satellite_id in operational_ids
    ):
        raise ValueError("Projected satellite ID is outside the configured satellite range")
    components = _select_authoritative_gcc(satellite_graph)
    gcc_ids = min(components, key=lambda component: (-len(component), component)) if components else ()
    attached_to_gcc: set[str] = set()
    gcc_set = set(gcc_ids)
    for edge in integrated_snapshot.canonical_edges:
        if edge.edge_kind is IntegratedEdgeKind.SATELLITE_GROUND:
            satellite_id = edge.endpoint_a.satellite_id
            station_id = edge.endpoint_b.ground_station_id
            if station_id not in selected_ids:
                raise ValueError("Satellite-ground edge references a station outside G1 selection")
            if satellite_id in gcc_set:
                attached_to_gcc.add(station_id)
    serviced_ids = tuple(sorted(attached_to_gcc))
    unserviced_ids = tuple(sorted(selected_ids - attached_to_gcc))
    class_ids = {
        "civilian": set(selection.civilian_station_ids),
        "government": set(selection.government_station_ids),
        "military": set(selection.military_station_ids),
    }
    class_totals = {name: len(ids) for name, ids in class_ids.items()}
    class_serviced = {name: len(ids & attached_to_gcc) for name, ids in class_ids.items()}
    class_fractions = {
        name: None if class_totals[name] == 0 else _ratio(class_serviced[name], class_totals[name])
        for name in class_ids
    }
    satellite_gcc_size = len(gcc_ids)
    operational_count = len(operational_ids)
    ground_total = len(selected_ids)
    ground_serviced = len(serviced_ids)
    space_original = _ratio(satellite_gcc_size, configured_satellite_count)
    space_surviving = _ratio(satellite_gcc_size, operational_count)
    ground_fraction = _ratio(ground_serviced, ground_total)
    space_met = space_original >= policy.space_gcc_threshold
    ground_met = ground_fraction >= policy.ground_service_threshold
    return {
        "timestep_index": integrated_snapshot.timestep_index,
        "timestamp_utc": integrated_snapshot.timestamp_utc,
        "satellite_config_hash": integrated_snapshot.satellite_config_hash,
        "ground_design_hash": integrated_snapshot.ground_design_hash,
        "visibility_policy_hash": integrated_snapshot.visibility_policy_hash,
        "integrated_graph_hash": integrated_snapshot.graph_hash,
        "ground_service_policy_hash": policy.ground_service_policy_hash,
        "ground_service_model_version": GROUND_SERVICE_MODEL_VERSION,
        "configured_satellite_count": configured_satellite_count,
        "operational_satellite_count": operational_count,
        "satellite_component_count": len(components),
        "satellite_gcc_size": satellite_gcc_size,
        "satellite_gcc_ids": gcc_ids,
        "total_ground_station_count": ground_total,
        "serviced_ground_station_count": ground_serviced,
        "unserviced_ground_station_count": len(unserviced_ids),
        "serviced_ground_station_ids": serviced_ids,
        "unserviced_ground_station_ids": unserviced_ids,
        "total_civilian_count": class_totals["civilian"],
        "serviced_civilian_count": class_serviced["civilian"],
        "civilian_service_fraction": class_fractions["civilian"],
        "total_government_count": class_totals["government"],
        "serviced_government_count": class_serviced["government"],
        "government_service_fraction": class_fractions["government"],
        "total_military_count": class_totals["military"],
        "serviced_military_count": class_serviced["military"],
        "military_service_fraction": class_fractions["military"],
        "space_gcc_fraction_original": space_original,
        "space_gcc_fraction_surviving": space_surviving,
        "ground_service_fraction": ground_fraction,
        "overall_service_fraction": min(space_original, ground_fraction),
        "space_threshold_met": space_met,
        "ground_threshold_met": ground_met,
        "overall_threshold_met": space_met and ground_met,
    }


def compute_ground_service_step(
    *,
    integrated_snapshot: IntegratedGroundGraphSnapshot,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    configured_satellite_count: int,
    policy: GroundServicePolicy,
) -> GroundServiceStepMetrics:
    values = _build_step_values(
        integrated_snapshot=integrated_snapshot,
        ground_design=ground_design,
        catalog=catalog,
        configured_satellite_count=configured_satellite_count,
        policy=policy,
    )
    values["step_metrics_hash"] = _compute_step_metrics_hash(values)
    metrics = GroundServiceStepMetrics(**values)
    validate_ground_service_step_context(
        metrics=metrics,
        integrated_snapshot=integrated_snapshot,
        ground_design=ground_design,
        catalog=catalog,
        configured_satellite_count=configured_satellite_count,
        policy=policy,
    )
    return metrics


def validate_ground_service_step_context(
    *,
    metrics: GroundServiceStepMetrics,
    integrated_snapshot: IntegratedGroundGraphSnapshot,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    configured_satellite_count: int,
    policy: GroundServicePolicy,
) -> None:
    if not isinstance(metrics, GroundServiceStepMetrics):
        raise TypeError("metrics must be GroundServiceStepMetrics")
    expected_values = _build_step_values(
        integrated_snapshot=integrated_snapshot,
        ground_design=ground_design,
        catalog=catalog,
        configured_satellite_count=configured_satellite_count,
        policy=policy,
    )
    expected_values["step_metrics_hash"] = _compute_step_metrics_hash(expected_values)
    for field in fields(GroundServiceStepMetrics):
        if getattr(metrics, field.name) != expected_values[field.name]:
            raise ValueError(f"Ground-service step contextual mismatch: {field.name}")
