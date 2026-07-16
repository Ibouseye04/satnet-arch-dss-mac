from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import math

import pytest

from satnet.ground.catalog import GroundStation, GroundStationCatalog, GroundStationClass
from satnet.ground.integrated_graph import (
    CanonicalIntegratedEdge,
    CanonicalIntegratedNode,
    IntegratedEdgeKind,
    IntegratedNodeRef,
    create_integrated_graph_snapshot,
)
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.ground.service_metrics import (
    GroundServiceStepMetrics,
    _compute_step_metrics_hash,
    compute_ground_service_step,
    validate_ground_service_step_context,
)
from satnet.ground.service_policy import GroundServicePolicy

TIMESTAMP = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
SATELLITE_HASH = "a" * 64
VISIBILITY_HASH = "b" * 64
VISIBILITY_SNAPSHOT_HASH = "c" * 64


def station(station_id: str, station_class: GroundStationClass, offset: int) -> GroundStation:
    return GroundStation(
        station_id=station_id,
        name=f"G4 Station {station_id}",
        station_class=station_class,
        latitude_deg=float(offset),
        longitude_deg=float(offset),
        altitude_m=0.0,
        region="region_test",
        country_code="ZZ",
    )


def make_context(
    *,
    class_counts: tuple[int, int, int] = (2, 1, 1),
    satellite_ids: tuple[int, ...] = (0, 1, 2, 3),
    isl_edges: tuple[tuple[int, int], ...] = ((0, 1), (2, 3)),
    attachments: tuple[tuple[int, str], ...] = (),
    configured_satellite_count: int = 4,
    policy: GroundServicePolicy | None = None,
):
    all_stations = (
        station("CIV_G4_001", GroundStationClass.CIVILIAN, 1),
        station("CIV_G4_002", GroundStationClass.CIVILIAN, 2),
        station("GOV_G4_001", GroundStationClass.GOVERNMENT, 3),
        station("MIL_G4_001", GroundStationClass.MILITARY, 4),
    )
    catalog = GroundStationCatalog(all_stations)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(*class_counts, station_selection_seed=7),
    )
    design = make_enabled_ground_design_record(
        run_id=11,
        satellite_config_hash=SATELLITE_HASH,
        selection=selection,
    )
    nodes = tuple(
        CanonicalIntegratedNode(IntegratedNodeRef.for_satellite(satellite_id), ())
        for satellite_id in satellite_ids
    ) + tuple(
        CanonicalIntegratedNode(IntegratedNodeRef.for_ground_station(station_id), ())
        for station_id in selection.selected_station_ids
    )
    edges = tuple(
        CanonicalIntegratedEdge(
            IntegratedEdgeKind.INTER_SATELLITE,
            IntegratedNodeRef.for_satellite(left),
            IntegratedNodeRef.for_satellite(right),
            (),
        )
        for left, right in isl_edges
    ) + tuple(
        CanonicalIntegratedEdge(
            IntegratedEdgeKind.SATELLITE_GROUND,
            IntegratedNodeRef.for_satellite(satellite_id),
            IntegratedNodeRef.for_ground_station(station_id),
            (),
        )
        for satellite_id, station_id in attachments
    )
    snapshot = create_integrated_graph_snapshot(
        timestep_index=0,
        timestamp_utc=TIMESTAMP,
        satellite_config_hash=SATELLITE_HASH,
        ground_design_hash=design.ground_design_hash,
        visibility_policy_hash=VISIBILITY_HASH,
        visibility_snapshot_hash=VISIBILITY_SNAPSHOT_HASH,
        graph_attributes=(),
        nodes=nodes,
        edges=edges,
    )
    active_policy = policy or GroundServicePolicy(0.5, 0.5)
    metrics = compute_ground_service_step(
        integrated_snapshot=snapshot,
        ground_design=design,
        catalog=catalog,
        configured_satellite_count=configured_satellite_count,
        policy=active_policy,
    )
    return catalog, design, selection, snapshot, active_policy, metrics


def rebuilt_metrics(metrics: GroundServiceStepMetrics, **changes: object) -> GroundServiceStepMetrics:
    values = asdict(metrics)
    values.update(changes)
    values["step_metrics_hash"] = _compute_step_metrics_hash(values)
    return GroundServiceStepMetrics(**values)


@pytest.mark.parametrize("field_name", ["space_gcc_threshold", "ground_service_threshold"])
@pytest.mark.parametrize("value", [True, "0.5", float("nan"), float("inf"), -0.1, 1.1])
def test_policy_rejects_invalid_thresholds(field_name: str, value: object) -> None:
    values = {"space_gcc_threshold": 0.5, "ground_service_threshold": 0.5}
    values[field_name] = value
    with pytest.raises((TypeError, ValueError)):
        GroundServicePolicy(**values)


def test_policy_normalizes_numbers_and_negative_zero_deterministically() -> None:
    first = GroundServicePolicy(-0.0, 1)
    second = GroundServicePolicy(0, 1.0)
    assert first.space_gcc_threshold == 0.0
    assert math.copysign(1.0, first.space_gcc_threshold) == 1.0
    assert first == second
    assert first.ground_service_policy_hash == second.ground_service_policy_hash


def test_equal_size_gcc_tie_break_controls_ground_and_class_service() -> None:
    catalog, design, selection, snapshot, policy, metrics = make_context()
    civ_a, civ_b = selection.civilian_station_ids
    gov = selection.government_station_ids[0]
    mil = selection.military_station_ids[0]
    _, _, _, _, _, metrics = make_context(
        attachments=((0, civ_a), (1, gov), (2, civ_b), (3, mil))
    )
    assert metrics.satellite_component_count == 2
    assert metrics.satellite_gcc_ids == (0, 1)
    assert metrics.serviced_ground_station_ids == tuple(sorted((civ_a, gov)))
    assert metrics.unserviced_ground_station_ids == tuple(sorted((civ_b, mil)))
    assert metrics.civilian_service_fraction == 0.5
    assert metrics.government_service_fraction == 1.0
    assert metrics.military_service_fraction == 0.0
    assert metrics.ground_service_fraction == 0.5
    assert metrics.space_gcc_fraction_original == 0.5
    assert metrics.overall_service_fraction == 0.5
    assert metrics.space_threshold_met
    assert metrics.ground_threshold_met
    assert metrics.overall_threshold_met
    validate_ground_service_step_context(
        metrics=metrics,
        integrated_snapshot=make_context(
            attachments=((0, civ_a), (1, gov), (2, civ_b), (3, mil))
        )[3],
        ground_design=design,
        catalog=catalog,
        configured_satellite_count=4,
        policy=policy,
    )


def test_fully_serviced_and_multiple_gcc_neighbors_count_once() -> None:
    _, _, selection, _, _, metrics = make_context(
        satellite_ids=(0, 1, 2, 3),
        isl_edges=((0, 1), (1, 2), (2, 3)),
        attachments=(),
    )
    attachments = tuple((0, station_id) for station_id in selection.selected_station_ids) + (
        (1, selection.selected_station_ids[0]),
    )
    _, _, _, _, _, metrics = make_context(
        satellite_ids=(0, 1, 2, 3),
        isl_edges=((0, 1), (1, 2), (2, 3)),
        attachments=attachments,
    )
    assert metrics.space_gcc_fraction_original == 1.0
    assert metrics.space_gcc_fraction_surviving == 1.0
    assert metrics.serviced_ground_station_count == 4
    assert metrics.ground_service_fraction == 1.0
    assert metrics.overall_service_fraction == 1.0


def test_catastrophic_attrition_preserves_original_denominator() -> None:
    _, _, selection, _, _, _ = make_context()
    _, _, _, _, _, metrics = make_context(
        satellite_ids=(7,),
        isl_edges=(),
        attachments=((7, selection.selected_station_ids[0]),),
        configured_satellite_count=100,
        policy=GroundServicePolicy(0.01, 0.25),
    )
    assert metrics.operational_satellite_count == 1
    assert metrics.satellite_gcc_size == 1
    assert metrics.space_gcc_fraction_surviving == 1.0
    assert metrics.space_gcc_fraction_original == 0.01
    assert metrics.ground_service_fraction == 0.25
    assert metrics.overall_service_fraction == 0.01
    assert metrics.overall_threshold_met


def test_zero_operational_satellites_and_zero_threshold_semantics() -> None:
    _, _, _, _, _, metrics = make_context(
        satellite_ids=(),
        isl_edges=(),
        attachments=(),
        policy=GroundServicePolicy(0.0, 0.0),
    )
    assert metrics.satellite_component_count == 0
    assert metrics.satellite_gcc_ids == ()
    assert metrics.space_gcc_fraction_original == 0.0
    assert metrics.space_gcc_fraction_surviving == 0.0
    assert metrics.ground_service_fraction == 0.0
    assert metrics.overall_service_fraction == 0.0
    assert metrics.unserviced_ground_station_count == 4
    assert metrics.space_threshold_met
    assert metrics.ground_threshold_met
    assert metrics.overall_threshold_met


def test_absent_classes_use_none() -> None:
    _, _, selection, _, _, _ = make_context(class_counts=(2, 0, 0))
    _, _, _, _, _, metrics = make_context(
        class_counts=(2, 0, 0),
        attachments=((0, selection.civilian_station_ids[0]),),
    )
    assert metrics.total_civilian_count == 2
    assert metrics.civilian_service_fraction == 0.5
    assert metrics.total_government_count == 0
    assert metrics.government_service_fraction is None
    assert metrics.total_military_count == 0
    assert metrics.military_service_fraction is None


@pytest.mark.parametrize(
    ("policy", "space_met", "ground_met", "overall_met"),
    [
        (GroundServicePolicy(0.5, 0.5), True, True, True),
        (GroundServicePolicy(0.5, 0.51), True, False, False),
        (GroundServicePolicy(0.51, 0.5), False, True, False),
        (GroundServicePolicy(0.51, 0.51), False, False, False),
    ],
)
def test_exact_threshold_boundaries_and_combinations(
    policy: GroundServicePolicy, space_met: bool, ground_met: bool, overall_met: bool
) -> None:
    _, _, selection, _, _, _ = make_context()
    _, _, _, _, _, metrics = make_context(
        attachments=((0, selection.civilian_station_ids[0]), (1, selection.government_station_ids[0])),
        policy=policy,
    )
    assert metrics.space_threshold_met is space_met
    assert metrics.ground_threshold_met is ground_met
    assert metrics.overall_threshold_met is overall_met


def test_station_outside_gcc_and_station_without_edges_remain_in_denominator() -> None:
    _, _, selection, _, _, _ = make_context()
    station_outside = selection.selected_station_ids[0]
    _, _, _, _, _, metrics = make_context(attachments=((2, station_outside),))
    assert metrics.serviced_ground_station_count == 0
    assert metrics.total_ground_station_count == 4
    assert station_outside in metrics.unserviced_ground_station_ids


def test_configured_count_and_projected_satellite_bounds_are_exact() -> None:
    with pytest.raises(TypeError, match="configured_satellite_count"):
        make_context(configured_satellite_count=True)
    with pytest.raises(ValueError, match="outside"):
        make_context(satellite_ids=(4,), isl_edges=(), configured_satellite_count=4)


def test_ground_node_set_must_equal_reconstructed_g1_selection() -> None:
    catalog, design, selection, snapshot, policy, _ = make_context()
    reduced = create_integrated_graph_snapshot(
        timestep_index=snapshot.timestep_index,
        timestamp_utc=snapshot.timestamp_utc,
        satellite_config_hash=snapshot.satellite_config_hash,
        ground_design_hash=snapshot.ground_design_hash,
        visibility_policy_hash=snapshot.visibility_policy_hash,
        visibility_snapshot_hash=snapshot.visibility_snapshot_hash,
        graph_attributes=snapshot.canonical_graph_attributes,
        nodes=tuple(
            node
            for node in snapshot.canonical_nodes
            if node.node_ref.ground_station_id != selection.selected_station_ids[0]
        ),
        edges=snapshot.canonical_edges,
    )
    with pytest.raises(ValueError, match="ground nodes"):
        compute_ground_service_step(
            integrated_snapshot=reduced,
            ground_design=design,
            catalog=catalog,
            configured_satellite_count=4,
            policy=policy,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("operational_satellite_count", 3),
        ("satellite_gcc_size", 1),
        ("satellite_gcc_ids", (0,)),
        ("serviced_ground_station_count", 1),
        ("ground_service_fraction", 0.25),
        ("space_gcc_fraction_original", 0.25),
        ("space_gcc_fraction_surviving", 0.25),
        ("overall_service_fraction", 0.25),
        ("serviced_civilian_count", 0),
    ],
)
def test_relational_corruption_fails_even_with_replacement_hash(
    field_name: str, value: object
) -> None:
    _, _, selection, _, _, _ = make_context()
    _, _, _, _, _, metrics = make_context(
        attachments=((0, selection.civilian_station_ids[0]), (1, selection.government_station_ids[0]))
    )
    with pytest.raises(ValueError):
        rebuilt_metrics(metrics, **{field_name: value})


def test_threshold_corruption_with_replacement_hash_fails_contextual_validation() -> None:
    catalog, design, selection, snapshot, policy, _ = make_context()
    catalog, design, selection, snapshot, policy, metrics = make_context(
        attachments=((0, selection.civilian_station_ids[0]), (1, selection.government_station_ids[0]))
    )
    corrupted = rebuilt_metrics(
        metrics,
        space_threshold_met=False,
        overall_threshold_met=False,
    )
    with pytest.raises(ValueError, match="contextual mismatch"):
        validate_ground_service_step_context(
            metrics=corrupted,
            integrated_snapshot=snapshot,
            ground_design=design,
            catalog=catalog,
            configured_satellite_count=4,
            policy=policy,
        )


def test_inputs_and_upstream_hashes_are_unchanged() -> None:
    catalog, design, _, snapshot, policy, metrics = make_context()
    before = (
        catalog.catalog_hash,
        design.selection_hash,
        design.ground_design_hash,
        snapshot.graph_hash,
        snapshot.canonical_nodes,
        snapshot.canonical_edges,
        policy.ground_service_policy_hash,
    )
    compute_ground_service_step(
        integrated_snapshot=snapshot,
        ground_design=design,
        catalog=catalog,
        configured_satellite_count=4,
        policy=policy,
    )
    after = (
        catalog.catalog_hash,
        design.selection_hash,
        design.ground_design_hash,
        snapshot.graph_hash,
        snapshot.canonical_nodes,
        snapshot.canonical_edges,
        policy.ground_service_policy_hash,
    )
    assert before == after
    assert metrics.integrated_graph_hash == snapshot.graph_hash
