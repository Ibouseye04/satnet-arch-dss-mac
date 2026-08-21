from __future__ import annotations

from datetime import datetime, timezone
import json

import networkx as nx

from satnet.ground.catalog import GroundStation, GroundStationCatalog, GroundStationClass
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
    IntegratedGroundGraphSnapshot,
    operational_snapshot_from_networkx,
    validate_satellite_projection,
)
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.ground.visibility import GroundVisibilityPolicy, evaluate_ground_design_visibility

TIMESTAMP = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
SATELLITE_HASH = "1" * 64


def _station(station_id: str, longitude: float) -> GroundStation:
    return GroundStation(
        station_id=station_id,
        name=f"Synthetic Diagnostic {station_id}",
        station_class=GroundStationClass.CIVILIAN,
        latitude_deg=0.0,
        longitude_deg=longitude,
        altitude_m=0.0,
        region="diagnostic_region",
        country_code="ZZ",
    )


def _case(
    name: str,
    *,
    station_count: int,
    satellite_ids: tuple[int, ...],
    visible_satellite_ids: set[int],
    include_isl: bool,
) -> tuple[str, IntegratedGroundGraphSnapshot, object, object, object]:
    stations = tuple(
        _station(f"CIV_G3_DIAG_{index + 1:03d}", float(index * 20))
        for index in range(station_count)
    )
    catalog = GroundStationCatalog(stations)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(station_count, 0, 0, 42),
    )
    design = make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash=SATELLITE_HASH,
        selection=selection,
    )
    graph = nx.Graph()
    graph.graph["diagnostic_case"] = name
    for satellite_id in satellite_ids:
        graph.add_node(
            satellite_id,
            type="satellite",
            label=f"SAT-{satellite_id:05d}",
        )
    if include_isl and len(satellite_ids) >= 2:
        graph.add_edge(
            satellite_ids[0],
            satellite_ids[1],
            distance_km=500.0,
            link_type="diagnostic",
        )
    graph_source = operational_snapshot_from_networkx(
        timestep_index=0,
        timestamp_utc=TIMESTAMP,
        satellite_config_hash=SATELLITE_HASH,
        graph=graph,
    )
    positions = tuple(
        FramedSatellitePosition(
            satellite_id=satellite_id,
            timestamp_utc=TIMESTAMP,
            x_km=WGS84_SEMI_MAJOR_AXIS_KM
            + (500.0 if satellite_id in visible_satellite_ids else -100.0),
            y_km=0.0 if satellite_id in visible_satellite_ids else 500.0,
            z_km=0.0,
            frame=CoordinateFrame.ECEF,
        )
        for satellite_id in satellite_ids
    )
    position_source = OperationalSatellitePositionSnapshot(
        timestep_index=0,
        timestamp_utc=TIMESTAMP,
        satellite_config_hash=SATELLITE_HASH,
        positions=positions,
    )
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
    validate_satellite_projection(
        source_snapshot=graph_source,
        integrated_snapshot=integrated,
    )
    validate_visibility_edge_equality(
        integrated_snapshot=integrated,
        visibility_snapshot=visibility,
    )
    return name, integrated, graph_source, visibility, design


def build_diagnostic_snapshots():
    return (
        _case(
            "one_station_one_satellite_visible",
            station_count=1,
            satellite_ids=(2,),
            visible_satellite_ids={2},
            include_isl=False,
        ),
        _case(
            "one_station_one_satellite_nonvisible",
            station_count=1,
            satellite_ids=(2,),
            visible_satellite_ids=set(),
            include_isl=False,
        ),
        _case(
            "multiple_stations_multiple_satellites",
            station_count=2,
            satellite_ids=(2, 10),
            visible_satellite_ids={2, 10},
            include_isl=True,
        ),
        _case(
            "station_with_no_access",
            station_count=2,
            satellite_ids=(2,),
            visible_satellite_ids=set(),
            include_isl=False,
        ),
        _case(
            "all_satellites_failed",
            station_count=2,
            satellite_ids=(),
            visible_satellite_ids=set(),
            include_isl=False,
        ),
        _case(
            "complete_satellite_graph_zero_ground_links",
            station_count=1,
            satellite_ids=(2, 10),
            visible_satellite_ids=set(),
            include_isl=True,
        ),
    )


def build_diagnostics() -> list[dict[str, object]]:
    results = []
    for name, integrated, _, visibility, _ in build_diagnostic_snapshots():
        results.append(
            {
                "case": name,
                "timestep_index": integrated.timestep_index,
                "satellite_nodes": integrated.satellite_node_count,
                "ground_nodes": integrated.ground_node_count,
                "isl_edges": integrated.isl_edge_count,
                "satellite_ground_edges": integrated.satellite_ground_edge_count,
                "visible_station_mappings": [
                    [station_id, list(satellite_ids)]
                    for station_id, satellite_ids in visibility.visible_satellite_ids_by_station
                ],
                "satellite_projection_valid": True,
                "visible_edge_validation_valid": True,
                "graph_hash": integrated.graph_hash,
            }
        )
    return results


def main() -> None:
    print(json.dumps(build_diagnostics(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
