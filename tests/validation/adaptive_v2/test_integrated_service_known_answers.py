"""Permanent Adaptive-v2 integrated space-ground service known answers."""

from __future__ import annotations

from datetime import datetime, timezone

import networkx as nx

from satnet.ground.catalog import (
    GroundStation,
    GroundStationCatalog,
    GroundStationClass,
)
from satnet.ground.coordinates import (
    CoordinateFrame,
    FramedSatellitePosition,
    OperationalSatellitePositionSnapshot,
    WGS84_SEMI_MAJOR_AXIS_KM,
)
from satnet.ground.integrated_builder import (
    build_integrated_ground_graph,
    validate_visibility_edge_equality,
)
from satnet.ground.integrated_graph import (
    IntegratedNodeRef,
    operational_snapshot_from_networkx,
    validate_satellite_projection,
)
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.selection import (
    GroundSegmentEnabledConfig,
    select_ground_stations,
)
from satnet.ground.visibility import (
    GroundVisibilityPolicy,
    evaluate_ground_visibility,
)


TIMESTAMP = datetime(2026, 7, 19, 0, 0, 0, tzinfo=timezone.utc)
SATELLITE_CONFIG_HASH = "a" * 64


def _controlled_ground_catalog() -> GroundStationCatalog:
    """Three colocated stations, one from each final ground-station class."""
    return GroundStationCatalog(
        (
            GroundStation(
                station_id="CIV_TEST_001",
                name="Civilian Test",
                station_class=GroundStationClass.CIVILIAN,
                latitude_deg=0.0,
                longitude_deg=0.0,
                altitude_m=0.0,
                region="test_region",
                country_code="US",
            ),
            GroundStation(
                station_id="GOV_TEST_001",
                name="Government Test",
                station_class=GroundStationClass.GOVERNMENT,
                latitude_deg=0.0,
                longitude_deg=0.0,
                altitude_m=0.0,
                region="test_region",
                country_code="US",
            ),
            GroundStation(
                station_id="MIL_TEST_001",
                name="Military Test",
                station_class=GroundStationClass.MILITARY,
                latitude_deg=0.0,
                longitude_deg=0.0,
                altitude_m=0.0,
                region="test_region",
                country_code="US",
            ),
        )
    )


def _controlled_integrated_context():
    catalog = _controlled_ground_catalog()

    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(
            civilian_count=1,
            government_count=1,
            military_count=1,
            station_selection_seed=7,
        ),
    )

    design = make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash=SATELLITE_CONFIG_HASH,
        selection=selection,
    )

    # Authoritative controlled satellite graph:
    #
    #     0 -- 1 -- 2
    #
    satellite_graph = nx.Graph()
    satellite_graph.add_nodes_from((0, 1, 2))
    satellite_graph.add_edges_from(((0, 1), (1, 2)))

    satellite_snapshot = operational_snapshot_from_networkx(
        timestep_index=0,
        timestamp_utc=TIMESTAMP,
        satellite_config_hash=SATELLITE_CONFIG_HASH,
        graph=satellite_graph,
    )

    # All three ground stations are at lat=0, lon=0.
    #
    # Satellites 0 and 2 lie directly above them (+ECEF X) and therefore
    # have +90-degree elevation.
    #
    # Satellite 1 lies on the opposite side of Earth (-ECEF X) and therefore
    # has -90-degree elevation.
    satellite_positions = OperationalSatellitePositionSnapshot(
        timestep_index=0,
        timestamp_utc=TIMESTAMP,
        satellite_config_hash=SATELLITE_CONFIG_HASH,
        positions=(
            FramedSatellitePosition(
                satellite_id=0,
                timestamp_utc=TIMESTAMP,
                x_km=WGS84_SEMI_MAJOR_AXIS_KM + 500.0,
                y_km=0.0,
                z_km=0.0,
                frame=CoordinateFrame.ECEF,
            ),
            FramedSatellitePosition(
                satellite_id=1,
                timestamp_utc=TIMESTAMP,
                x_km=-(WGS84_SEMI_MAJOR_AXIS_KM + 500.0),
                y_km=0.0,
                z_km=0.0,
                frame=CoordinateFrame.ECEF,
            ),
            FramedSatellitePosition(
                satellite_id=2,
                timestamp_utc=TIMESTAMP,
                x_km=WGS84_SEMI_MAJOR_AXIS_KM + 1000.0,
                y_km=0.0,
                z_km=0.0,
                frame=CoordinateFrame.ECEF,
            ),
        ),
    )

    visibility = evaluate_ground_visibility(
        selected_stations=catalog.stations,
        satellite_snapshot=satellite_positions,
        policy=GroundVisibilityPolicy(minimum_elevation_deg=10.0),
        ground_design_hash=design.ground_design_hash,
    )

    integrated = build_integrated_ground_graph(
        ground_design=design,
        catalog=catalog,
        satellite_graph_snapshot=satellite_snapshot,
        verified_visibility_snapshot=visibility,
    )

    return (
        catalog,
        selection,
        design,
        satellite_snapshot,
        visibility,
        integrated,
    )


def test_g3_controlled_visibility_pairs_known_answer() -> None:
    """Every station sees satellites 0 and 2, but not satellite 1."""
    (
        _catalog,
        selection,
        _design,
        _satellite_snapshot,
        visibility,
        _integrated,
    ) = _controlled_integrated_context()

    expected_pairs = {
        (station_id, satellite_id)
        for station_id in selection.selected_station_ids
        for satellite_id in (0, 2)
    }

    observed_pairs = {
        (observation.station_id, observation.satellite_id)
        for observation in visibility.visible_links
    }

    assert observed_pairs == expected_pairs
    assert len(observed_pairs) == 6


def test_g3_integrated_graph_counts_known_answer() -> None:
    """Freeze the controlled 3-satellite + 3-ground integrated graph counts."""
    *_, integrated = _controlled_integrated_context()

    assert integrated.satellite_node_count == 3
    assert integrated.ground_node_count == 3
    assert integrated.isl_edge_count == 2
    assert integrated.satellite_ground_edge_count == 6

    graph = integrated.to_networkx()

    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 8

    for satellite_id in (0, 1, 2):
        assert IntegratedNodeRef.for_satellite(satellite_id) in graph

    for station_id in (
        "CIV_TEST_001",
        "GOV_TEST_001",
        "MIL_TEST_001",
    ):
        assert IntegratedNodeRef.for_ground_station(station_id) in graph


def test_g3_satellite_projection_is_exact() -> None:
    """Adding the ground layer must not alter the satellite-only graph."""
    (
        _catalog,
        _selection,
        _design,
        satellite_snapshot,
        visibility,
        integrated,
    ) = _controlled_integrated_context()

    validate_satellite_projection(
        source_snapshot=satellite_snapshot,
        integrated_snapshot=integrated,
    )

    validate_visibility_edge_equality(
        integrated_snapshot=integrated,
        visibility_snapshot=visibility,
    )

    projected = integrated.to_networkx().subgraph(
        [
            IntegratedNodeRef.for_satellite(0),
            IntegratedNodeRef.for_satellite(1),
            IntegratedNodeRef.for_satellite(2),
        ]
    )

    expected_edges = {
        frozenset(
            (
                IntegratedNodeRef.for_satellite(0),
                IntegratedNodeRef.for_satellite(1),
            )
        ),
        frozenset(
            (
                IntegratedNodeRef.for_satellite(1),
                IntegratedNodeRef.for_satellite(2),
            )
        ),
    }

    observed_edges = {
        frozenset((left, right))
        for left, right in projected.edges()
    }

    assert observed_edges == expected_edges


# ---------------------------------------------------------------------------
# G4 baseline ground-service known answers
# ---------------------------------------------------------------------------

from satnet.ground.service_metrics import (
    compute_ground_service_step,
    validate_ground_service_step_context,
)
from satnet.ground.service_policy import GroundServicePolicy


def test_g4_baseline_service_known_answer() -> None:
    """All satellites and all three ground stations are fully serviced."""
    (
        catalog,
        selection,
        design,
        _satellite_snapshot,
        _visibility,
        integrated,
    ) = _controlled_integrated_context()

    policy = GroundServicePolicy(
        space_gcc_threshold=0.80,
        ground_service_threshold=0.80,
    )

    metrics = compute_ground_service_step(
        integrated_snapshot=integrated,
        ground_design=design,
        catalog=catalog,
        configured_satellite_count=3,
        policy=policy,
    )

    # Space graph: 0 -- 1 -- 2
    assert metrics.configured_satellite_count == 3
    assert metrics.operational_satellite_count == 3
    assert metrics.satellite_component_count == 1
    assert metrics.satellite_gcc_size == 3
    assert metrics.satellite_gcc_ids == (0, 1, 2)

    assert metrics.space_gcc_fraction_original == 1.0
    assert metrics.space_gcc_fraction_surviving == 1.0

    # Every selected station has at least one attachment into the GCC.
    assert metrics.total_ground_station_count == 3
    assert metrics.serviced_ground_station_count == 3
    assert metrics.unserviced_ground_station_count == 0

    assert metrics.serviced_ground_station_ids == tuple(
        sorted(selection.selected_station_ids)
    )
    assert metrics.unserviced_ground_station_ids == ()

    # One selected and serviced station from each class.
    assert metrics.total_civilian_count == 1
    assert metrics.serviced_civilian_count == 1
    assert metrics.civilian_service_fraction == 1.0

    assert metrics.total_government_count == 1
    assert metrics.serviced_government_count == 1
    assert metrics.government_service_fraction == 1.0

    assert metrics.total_military_count == 1
    assert metrics.serviced_military_count == 1
    assert metrics.military_service_fraction == 1.0

    assert metrics.ground_service_fraction == 1.0

    # Authoritative integrated-service bottleneck:
    # min(space=1.0, ground=1.0) = 1.0.
    assert metrics.overall_service_fraction == 1.0

    assert metrics.space_threshold_met is True
    assert metrics.ground_threshold_met is True
    assert metrics.overall_threshold_met is True

    # Production contextual replay/validation must reproduce every field.
    validate_ground_service_step_context(
        metrics=metrics,
        integrated_snapshot=integrated,
        ground_design=design,
        catalog=catalog,
        configured_satellite_count=3,
        policy=policy,
    )


# ---------------------------------------------------------------------------
# G5 failure-adjusted ground-service known answers
# ---------------------------------------------------------------------------

from satnet.ground.failure_policy import GroundFailurePolicy
from satnet.ground.failure_realization import (
    create_ground_failure_realization_record,
)
from satnet.ground.failure_service_metrics import (
    create_failure_adjusted_step_record,
)
from satnet.ground.service_aggregation import (
    make_ground_service_step_record,
)


def _controlled_g4_record():
    """Build the verified G4 record used as the authoritative G5 baseline."""
    (
        catalog,
        selection,
        design,
        _satellite_snapshot,
        _visibility,
        integrated,
    ) = _controlled_integrated_context()

    service_policy = GroundServicePolicy(
        space_gcc_threshold=0.80,
        ground_service_threshold=0.80,
    )

    baseline = compute_ground_service_step(
        integrated_snapshot=integrated,
        ground_design=design,
        catalog=catalog,
        configured_satellite_count=3,
        policy=service_policy,
    )

    baseline_record = make_ground_service_step_record(
        run_id=design.run_id,
        metrics=baseline,
        integrated_snapshot=integrated,
        ground_design=design,
        catalog=catalog,
        configured_satellite_count=3,
        policy=service_policy,
    )

    return (
        catalog,
        selection,
        design,
        service_policy,
        baseline,
        baseline_record,
    )


def test_g5_zero_ground_failures_exactly_preserve_g4() -> None:
    """p=0 must leave ground, overall, and satellite service unchanged."""
    (
        catalog,
        selection,
        design,
        service_policy,
        baseline,
        baseline_record,
    ) = _controlled_g4_record()

    failure_policy = GroundFailurePolicy(
        ground_station_failure_probability=0.0
    )

    realization_record = create_ground_failure_realization_record(
        run_id=design.run_id,
        ground_design=design,
        catalog=catalog,
        policy=failure_policy,
        ground_failure_seed=123456789,
    )

    adjusted_record = create_failure_adjusted_step_record(
        run_id=design.run_id,
        verified_g4_step_record=baseline_record,
        ground_design=design,
        catalog=catalog,
        policy=failure_policy,
        realization_record=realization_record,
        ground_service_policy=service_policy,
    )

    realization = realization_record.realization
    metrics = adjusted_record.metrics

    assert realization.failed_station_ids == ()
    assert realization.operational_station_ids == tuple(
        sorted(selection.selected_station_ids)
    )

    assert metrics.failed_ground_station_count == 0
    assert metrics.operational_ground_station_count == 3

    assert metrics.failure_adjusted_serviced_ground_station_ids == (
        baseline.serviced_ground_station_ids
    )
    assert metrics.failure_adjusted_serviced_ground_station_count == 3

    assert metrics.failure_adjusted_ground_service_fraction == 1.0
    assert metrics.failure_adjusted_overall_service_fraction == 1.0

    assert metrics.ground_service_loss_due_to_failures == 0.0
    assert metrics.overall_service_loss_due_to_ground_failures == 0.0

    # Ground-only failure processing must not alter satellite evidence.
    assert (
        metrics.configured_satellite_count
        == baseline.configured_satellite_count
    )
    assert (
        metrics.operational_satellite_count
        == baseline.operational_satellite_count
    )
    assert metrics.satellite_component_count == baseline.satellite_component_count
    assert metrics.satellite_gcc_size == baseline.satellite_gcc_size
    assert metrics.satellite_gcc_ids == baseline.satellite_gcc_ids

    assert (
        metrics.space_gcc_fraction_original
        == baseline.space_gcc_fraction_original
        == 1.0
    )
    assert (
        metrics.space_gcc_fraction_surviving
        == baseline.space_gcc_fraction_surviving
        == 1.0
    )

    assert metrics.space_threshold_met is True
    assert metrics.ground_threshold_met is True
    assert metrics.overall_threshold_met is True


def test_g5_all_ground_failures_zero_ground_and_overall_service_only() -> None:
    """p=1 destroys ground service while preserving the complete space metric."""
    (
        catalog,
        selection,
        design,
        service_policy,
        baseline,
        baseline_record,
    ) = _controlled_g4_record()

    failure_policy = GroundFailurePolicy(
        ground_station_failure_probability=1.0
    )

    realization_record = create_ground_failure_realization_record(
        run_id=design.run_id,
        ground_design=design,
        catalog=catalog,
        policy=failure_policy,
        ground_failure_seed=123456789,
    )

    adjusted_record = create_failure_adjusted_step_record(
        run_id=design.run_id,
        verified_g4_step_record=baseline_record,
        ground_design=design,
        catalog=catalog,
        policy=failure_policy,
        realization_record=realization_record,
        ground_service_policy=service_policy,
    )

    realization = realization_record.realization
    metrics = adjusted_record.metrics

    expected_station_ids = tuple(sorted(selection.selected_station_ids))

    assert realization.failed_station_ids == expected_station_ids
    assert realization.operational_station_ids == ()

    assert metrics.failed_ground_station_count == 3
    assert metrics.operational_ground_station_count == 0

    assert metrics.failure_adjusted_serviced_ground_station_ids == ()
    assert (
        metrics.failure_adjusted_unserviced_ground_station_ids
        == expected_station_ids
    )

    assert metrics.failure_adjusted_serviced_ground_station_count == 0
    assert metrics.failure_adjusted_unserviced_ground_station_count == 3

    assert metrics.failure_adjusted_ground_service_fraction == 0.0

    # Bottleneck definition:
    # min(space=1.0, ground=0.0) = 0.0.
    assert metrics.failure_adjusted_overall_service_fraction == 0.0

    assert metrics.ground_service_loss_due_to_failures == 1.0
    assert metrics.overall_service_loss_due_to_ground_failures == 1.0

    # Ground-only failure must leave the space-side evidence unchanged.
    for field_name in (
        "configured_satellite_count",
        "operational_satellite_count",
        "satellite_component_count",
        "satellite_gcc_size",
        "satellite_gcc_ids",
        "space_gcc_fraction_original",
        "space_gcc_fraction_surviving",
        "space_threshold_met",
    ):
        assert getattr(metrics, field_name) == getattr(
            baseline,
            field_name,
        )

    assert metrics.space_gcc_fraction_original == 1.0
    assert metrics.space_gcc_fraction_surviving == 1.0

    # Each class had one selected station, and each one failed.
    assert metrics.failed_civilian_count == 1
    assert metrics.failed_government_count == 1
    assert metrics.failed_military_count == 1

    assert metrics.failure_adjusted_civilian_service_fraction == 0.0
    assert metrics.failure_adjusted_government_service_fraction == 0.0
    assert metrics.failure_adjusted_military_service_fraction == 0.0

    assert metrics.space_threshold_met is True
    assert metrics.ground_threshold_met is False
    assert metrics.overall_threshold_met is False
