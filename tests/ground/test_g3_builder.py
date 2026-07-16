from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import pytest

from satnet.ground.catalog import (
    GroundStation,
    GroundStationCatalog,
    GroundStationClass,
    load_ground_station_catalog,
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
    IntegratedEdgeKind,
    IntegratedNodeKind,
    IntegratedNodeRef,
    operational_snapshot_from_networkx,
    validate_satellite_projection,
)
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.position_adapter import reconstruct_operational_satellite_position_sequence
from satnet.ground.satellite_graph_adapter import reconstruct_operational_satellite_graph_sequence
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.ground.visibility import (
    GroundVisibilityPolicy,
    evaluate_ground_design_visibility,
    evaluate_ground_visibility,
)
from satnet.simulation.tier1_rollout import Tier1FailureRealization, Tier1RolloutConfig

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "ground_segment"
    / "synthetic_ground_station_catalog.csv"
)
TIMESTAMP = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
SATELLITE_HASH = "a" * 64


def config() -> Tier1RolloutConfig:
    return Tier1RolloutConfig(
        num_planes=2,
        sats_per_plane=2,
        duration_minutes=1,
        step_seconds=60,
        node_failure_prob=0.0,
        edge_failure_prob=0.0,
        seed=303,
    )


def edge_config() -> Tier1RolloutConfig:
    return Tier1RolloutConfig(
        num_planes=2,
        sats_per_plane=10,
        duration_minutes=1,
        step_seconds=60,
        node_failure_prob=0.0,
        edge_failure_prob=0.0,
        seed=304,
    )


def empty_failures() -> Tier1FailureRealization:
    return Tier1FailureRealization(failed_nodes=set(), failed_edges=set())


def ground_context(count: int = 2):
    catalog = load_ground_station_catalog(FIXTURE)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(count, 0, 0, 42),
    )
    design = make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash=config().config_hash(),
        selection=selection,
    )
    return catalog, selection, design


def test_operational_graph_reconstruction_covers_all_timesteps_and_failures() -> None:
    satellite_config = edge_config()
    baseline = reconstruct_operational_satellite_graph_sequence(
        satellite_config=satellite_config,
        failure_realization=empty_failures(),
    )
    assert len(baseline) == satellite_config.num_steps == 2
    assert tuple(snapshot.timestep_index for snapshot in baseline) == (0, 1)
    assert all(snapshot.satellite_config_hash == satellite_config.config_hash() for snapshot in baseline)
    expected_nodes = set(range(satellite_config.total_satellites))
    assert all(
        {node.satellite_id for node in snapshot.canonical_nodes} == expected_nodes
        for snapshot in baseline
    )
    assert baseline[0].canonical_edges
    first_edge = baseline[0].canonical_edges[0]
    failures = Tier1FailureRealization(
        failed_nodes={3},
        failed_edges={(first_edge.satellite_id_a, first_edge.satellite_id_b)},
    )
    failed = reconstruct_operational_satellite_graph_sequence(
        satellite_config=satellite_config,
        failure_realization=failures,
    )
    assert all(3 not in {node.satellite_id for node in snapshot.canonical_nodes} for snapshot in failed)
    assert all(
        (first_edge.satellite_id_a, first_edge.satellite_id_b)
        not in {(edge.satellite_id_a, edge.satellite_id_b) for edge in snapshot.canonical_edges}
        for snapshot in failed
    )


def test_all_satellites_failed_produces_valid_empty_graphs() -> None:
    snapshots = reconstruct_operational_satellite_graph_sequence(
        satellite_config=config(),
        failure_realization=Tier1FailureRealization(
            failed_nodes={0, 1, 2, 3}, failed_edges=set()
        ),
    )
    assert len(snapshots) == 2
    assert all(snapshot.canonical_nodes == () for snapshot in snapshots)
    assert all(snapshot.canonical_edges == () for snapshot in snapshots)
    assert all(len(snapshot.graph_hash) == 64 for snapshot in snapshots)


def test_basic_integrated_graph_preserves_projection_and_visible_edges() -> None:
    satellite_config = config()
    failures = empty_failures()
    graph_source = reconstruct_operational_satellite_graph_sequence(
        satellite_config=satellite_config,
        failure_realization=failures,
    )[0]
    position_source = reconstruct_operational_satellite_position_sequence(
        satellite_config=satellite_config,
        failure_realization=failures,
    )[0]
    catalog, selection, design = ground_context(2)
    visibility = evaluate_ground_design_visibility(
        ground_design=design,
        catalog=catalog,
        satellite_snapshot=position_source,
        policy=GroundVisibilityPolicy(0.0),
    )
    integrated = build_integrated_ground_graph(
        ground_design=design,
        catalog=catalog,
        satellite_graph_snapshot=graph_source,
        verified_visibility_snapshot=visibility,
    )
    assert integrated.satellite_node_count == 4
    assert integrated.ground_node_count == 2
    assert integrated.isl_edge_count == len(graph_source.canonical_edges)
    assert integrated.satellite_ground_edge_count == len(visibility.visible_links)
    validate_satellite_projection(
        source_snapshot=graph_source,
        integrated_snapshot=integrated,
    )
    validate_visibility_edge_equality(
        integrated_snapshot=integrated,
        visibility_snapshot=visibility,
    )
    projected = integrated.to_networkx()
    assert all(
        "integrated_node_kind" not in {attribute.name for attribute in node.attributes}
        for node in integrated.canonical_nodes
        if node.node_ref.kind is IntegratedNodeKind.SATELLITE
    )
    for station_id in selection.selected_station_ids:
        assert IntegratedNodeRef.for_ground_station(station_id) in projected


def synthetic_station(station_id: str) -> GroundStation:
    return GroundStation(
        station_id=station_id,
        name=f"Synthetic {station_id}",
        station_class=GroundStationClass.CIVILIAN,
        latitude_deg=0.0,
        longitude_deg=0.0,
        altitude_m=0.0,
        region="region_alpha",
        country_code="ZZ",
    )


def synthetic_source_graph(node_ids=(2, 10), with_edge=True):
    graph = nx.Graph()
    graph.graph["source"] = "synthetic"
    for node_id in node_ids:
        graph.add_node(node_id, type="satellite", label=f"SAT-{node_id:05d}")
    if with_edge and len(node_ids) >= 2:
        graph.add_edge(node_ids[0], node_ids[1], distance_km=500.0, link_type="test")
    return operational_snapshot_from_networkx(
        timestep_index=0,
        timestamp_utc=TIMESTAMP,
        satellite_config_hash=SATELLITE_HASH,
        graph=graph,
    )


def synthetic_position_source(node_ids=(2, 10)):
    return OperationalSatellitePositionSnapshot(
        timestep_index=0,
        timestamp_utc=TIMESTAMP,
        satellite_config_hash=SATELLITE_HASH,
        positions=tuple(
            FramedSatellitePosition(
                satellite_id=node_id,
                timestamp_utc=TIMESTAMP,
                x_km=WGS84_SEMI_MAJOR_AXIS_KM + 500.0,
                y_km=0.0,
                z_km=0.0,
                frame=CoordinateFrame.ECEF,
            )
            for node_id in node_ids
        ),
    )


def synthetic_ground_design():
    stations = (synthetic_station("CIV_G3_TEST_001"), synthetic_station("CIV_G3_TEST_002"))
    catalog = GroundStationCatalog(stations)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(2, 0, 0, 42),
    )
    design = make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash=SATELLITE_HASH,
        selection=selection,
    )
    return catalog, selection, design


def test_cartesian_product_rejects_union_equal_but_station_incomplete() -> None:
    catalog, selection, design = synthetic_ground_design()
    source_graph = synthetic_source_graph()
    one_station = catalog.by_id()[selection.selected_station_ids[0]]
    incomplete_visibility = evaluate_ground_visibility(
        selected_stations=(one_station,),
        satellite_snapshot=synthetic_position_source(),
        policy=GroundVisibilityPolicy(0.0),
        ground_design_hash=design.ground_design_hash,
    )
    observed_union = {item.satellite_id for item in incomplete_visibility.link_observations}
    assert observed_union == {2, 10}
    with pytest.raises(ValueError, match="Cartesian product"):
        build_integrated_ground_graph(
            ground_design=design,
            catalog=catalog,
            satellite_graph_snapshot=source_graph,
            verified_visibility_snapshot=incomplete_visibility,
        )


def test_all_failed_integrated_graph_keeps_ground_nodes() -> None:
    catalog, selection, design = synthetic_ground_design()
    graph_source = synthetic_source_graph(node_ids=(), with_edge=False)
    position_source = synthetic_position_source(node_ids=())
    visibility = evaluate_ground_design_visibility(
        ground_design=design,
        catalog=catalog,
        satellite_snapshot=position_source,
        policy=GroundVisibilityPolicy(10.0),
    )
    integrated = build_integrated_ground_graph(
        ground_design=design,
        catalog=catalog,
        satellite_graph_snapshot=graph_source,
        verified_visibility_snapshot=visibility,
    )
    assert integrated.satellite_node_count == 0
    assert integrated.ground_node_count == 2
    assert integrated.isl_edge_count == 0
    assert integrated.satellite_ground_edge_count == 0
    assert set(node.node_ref.ground_station_id for node in integrated.canonical_nodes) == set(
        selection.selected_station_ids
    )


def test_nonvisible_change_changes_identity_but_not_structure() -> None:
    catalog, _, design = synthetic_ground_design()
    graph_source = synthetic_source_graph()
    position_source = OperationalSatellitePositionSnapshot(
        timestep_index=0,
        timestamp_utc=TIMESTAMP,
        satellite_config_hash=SATELLITE_HASH,
        positions=tuple(
            FramedSatellitePosition(
                satellite_id=node_id,
                timestamp_utc=TIMESTAMP,
                x_km=WGS84_SEMI_MAJOR_AXIS_KM - 100.0,
                y_km=500.0,
                z_km=0.0,
                frame=CoordinateFrame.ECEF,
            )
            for node_id in (2, 10)
        ),
    )
    hidden_policy = GroundVisibilityPolicy(0.0)
    first_visibility = evaluate_ground_design_visibility(
        ground_design=design,
        catalog=catalog,
        satellite_snapshot=position_source,
        policy=hidden_policy,
    )
    shifted_positions = tuple(
        FramedSatellitePosition(
            satellite_id=position.satellite_id,
            timestamp_utc=position.timestamp_utc,
            x_km=position.x_km - 100.0,
            y_km=position.y_km + 100.0,
            z_km=position.z_km,
            frame=position.frame,
        )
        for position in position_source.positions
    )
    second_visibility = evaluate_ground_design_visibility(
        ground_design=design,
        catalog=catalog,
        satellite_snapshot=OperationalSatellitePositionSnapshot(
            timestep_index=0,
            timestamp_utc=TIMESTAMP,
            satellite_config_hash=SATELLITE_HASH,
            positions=shifted_positions,
        ),
        policy=hidden_policy,
    )
    first = build_integrated_ground_graph(
        ground_design=design,
        catalog=catalog,
        satellite_graph_snapshot=graph_source,
        verified_visibility_snapshot=first_visibility,
    )
    second = build_integrated_ground_graph(
        ground_design=design,
        catalog=catalog,
        satellite_graph_snapshot=graph_source,
        verified_visibility_snapshot=second_visibility,
    )
    assert first.satellite_ground_edge_count == second.satellite_ground_edge_count == 0
    assert first.canonical_nodes == second.canonical_nodes
    assert first.canonical_edges == second.canonical_edges
    assert first.graph_hash != second.graph_hash
