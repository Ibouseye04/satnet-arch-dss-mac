from __future__ import annotations

from satnet.ground.catalog import GroundStationCatalog
from satnet.ground.graph_attributes import attributes_to_dict, canonicalize_attributes
from satnet.ground.integrated_graph import (
    CanonicalIntegratedEdge,
    CanonicalIntegratedNode,
    IntegratedEdgeKind,
    IntegratedGroundGraphSnapshot,
    IntegratedNodeKind,
    IntegratedNodeRef,
    OperationalSatelliteGraphSnapshot,
    create_integrated_graph_snapshot,
    validate_satellite_projection,
)
from satnet.ground.persistence import GroundRunDesignRecord, reconstruct_ground_selection
from satnet.ground.visibility import GroundVisibilitySnapshot


def _validate_visibility_cartesian_product(
    *,
    selected_station_ids: set[str],
    operational_satellite_ids: set[int],
    visibility_snapshot: GroundVisibilitySnapshot,
) -> None:
    expected_pairs = {
        (station_id, satellite_id)
        for station_id in selected_station_ids
        for satellite_id in operational_satellite_ids
    }
    actual_pairs = {
        (observation.station_id, observation.satellite_id)
        for observation in visibility_snapshot.link_observations
    }
    if len(visibility_snapshot.link_observations) != len(actual_pairs):
        raise ValueError("Visibility observations contain duplicate station-satellite pairs")
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        raise ValueError(
            f"Visibility observations do not match complete Cartesian product; "
            f"missing={missing}, extra={extra}"
        )
    if len(visibility_snapshot.link_observations) != len(selected_station_ids) * len(
        operational_satellite_ids
    ):
        raise ValueError("Visibility observation count does not match Cartesian product")
    mapping_station_ids = {
        station_id
        for station_id, _ in visibility_snapshot.visible_satellite_ids_by_station
    }
    if mapping_station_ids != selected_station_ids:
        raise ValueError("Visibility station mappings do not match selected ground stations")
    for station_id, satellite_ids in visibility_snapshot.visible_satellite_ids_by_station:
        if any(satellite_id not in operational_satellite_ids for satellite_id in satellite_ids):
            raise ValueError(f"Station mapping contains nonoperational satellite: {station_id}")
    if any(
        observation.timestamp_utc != visibility_snapshot.timestamp_utc
        for observation in visibility_snapshot.link_observations
    ):
        raise ValueError("Visibility observation timestamp mismatch")


def _ground_node_attributes(station, ground_design_hash: str):
    return canonicalize_attributes(
        {
            "altitude_m": station.altitude_m,
            "country_code": station.country_code,
            "enabled": station.enabled,
            "ground_design_hash": ground_design_hash,
            "ground_station_id": station.station_id,
            "integrated_node_kind": IntegratedNodeKind.GROUND_STATION.value,
            "latitude_deg": station.latitude_deg,
            "longitude_deg": station.longitude_deg,
            "region": station.region,
            "station_class": station.station_class.value,
        }
    )


def _satellite_ground_edge_attributes(observation, visibility_snapshot):
    return canonicalize_attributes(
        {
            "elevation_deg": observation.elevation_deg,
            "frame_contract_version": visibility_snapshot.frame_contract_version,
            "integrated_edge_kind": IntegratedEdgeKind.SATELLITE_GROUND.value,
            "satellite_id": observation.satellite_id,
            "slant_range_km": observation.slant_range_km,
            "station_id": observation.station_id,
            "visibility_model_version": visibility_snapshot.visibility_model_version,
            "visibility_policy_hash": visibility_snapshot.visibility_policy_hash,
            "visibility_snapshot_hash": visibility_snapshot.snapshot_hash,
            "wgs84_model_version": visibility_snapshot.wgs84_model_version,
        }
    )


def build_integrated_ground_graph(
    *,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    satellite_graph_snapshot: OperationalSatelliteGraphSnapshot,
    verified_visibility_snapshot: GroundVisibilitySnapshot,
) -> IntegratedGroundGraphSnapshot:
    if not isinstance(ground_design, GroundRunDesignRecord):
        raise TypeError("ground_design must be a GroundRunDesignRecord")
    if not isinstance(catalog, GroundStationCatalog):
        raise TypeError("catalog must be a GroundStationCatalog")
    if not isinstance(satellite_graph_snapshot, OperationalSatelliteGraphSnapshot):
        raise TypeError("satellite_graph_snapshot must be an OperationalSatelliteGraphSnapshot")
    if not isinstance(verified_visibility_snapshot, GroundVisibilitySnapshot):
        raise TypeError("verified_visibility_snapshot must be a GroundVisibilitySnapshot")
    if ground_design.satellite_config_hash != satellite_graph_snapshot.satellite_config_hash:
        raise ValueError("Ground design satellite hash does not match source graph")
    if verified_visibility_snapshot.satellite_config_hash != satellite_graph_snapshot.satellite_config_hash:
        raise ValueError("Visibility satellite hash does not match source graph")
    if verified_visibility_snapshot.ground_design_hash != ground_design.ground_design_hash:
        raise ValueError("Visibility ground-design hash mismatch")
    if verified_visibility_snapshot.timestep_index != satellite_graph_snapshot.timestep_index:
        raise ValueError("Visibility and source graph timestep mismatch")
    if verified_visibility_snapshot.timestamp_utc != satellite_graph_snapshot.timestamp_utc:
        raise ValueError("Visibility and source graph timestamp mismatch")
    selection = reconstruct_ground_selection(ground_design, catalog)
    if selection is None:
        raise ValueError("Integrated graph requires an enabled ground design")
    selected_station_ids = set(selection.selected_station_ids)
    operational_satellite_ids = {
        node.satellite_id for node in satellite_graph_snapshot.canonical_nodes
    }
    _validate_visibility_cartesian_product(
        selected_station_ids=selected_station_ids,
        operational_satellite_ids=operational_satellite_ids,
        visibility_snapshot=verified_visibility_snapshot,
    )
    stations_by_id = catalog.by_id()
    satellite_nodes = tuple(
        CanonicalIntegratedNode(
            node_ref=IntegratedNodeRef.for_satellite(node.satellite_id),
            attributes=node.attributes,
        )
        for node in satellite_graph_snapshot.canonical_nodes
    )
    ground_nodes = tuple(
        CanonicalIntegratedNode(
            node_ref=IntegratedNodeRef.for_ground_station(station_id),
            attributes=_ground_node_attributes(
                stations_by_id[station_id], ground_design.ground_design_hash
            ),
        )
        for station_id in selection.selected_station_ids
    )
    isl_edges = tuple(
        CanonicalIntegratedEdge(
            edge_kind=IntegratedEdgeKind.INTER_SATELLITE,
            endpoint_a=IntegratedNodeRef.for_satellite(edge.satellite_id_a),
            endpoint_b=IntegratedNodeRef.for_satellite(edge.satellite_id_b),
            attributes=edge.attributes,
        )
        for edge in satellite_graph_snapshot.canonical_edges
    )
    satellite_ground_edges = tuple(
        CanonicalIntegratedEdge(
            edge_kind=IntegratedEdgeKind.SATELLITE_GROUND,
            endpoint_a=IntegratedNodeRef.for_satellite(observation.satellite_id),
            endpoint_b=IntegratedNodeRef.for_ground_station(observation.station_id),
            attributes=_satellite_ground_edge_attributes(
                observation, verified_visibility_snapshot
            ),
        )
        for observation in verified_visibility_snapshot.visible_links
    )
    snapshot = create_integrated_graph_snapshot(
        timestep_index=satellite_graph_snapshot.timestep_index,
        timestamp_utc=satellite_graph_snapshot.timestamp_utc,
        satellite_config_hash=satellite_graph_snapshot.satellite_config_hash,
        ground_design_hash=ground_design.ground_design_hash,
        visibility_policy_hash=verified_visibility_snapshot.visibility_policy_hash,
        visibility_snapshot_hash=verified_visibility_snapshot.snapshot_hash,
        graph_attributes=satellite_graph_snapshot.canonical_graph_attributes,
        nodes=satellite_nodes + ground_nodes,
        edges=isl_edges + satellite_ground_edges,
    )
    validate_satellite_projection(
        source_snapshot=satellite_graph_snapshot,
        integrated_snapshot=snapshot,
    )
    validate_visibility_edge_equality(
        integrated_snapshot=snapshot,
        visibility_snapshot=verified_visibility_snapshot,
    )
    return snapshot


def validate_visibility_edge_equality(
    *,
    integrated_snapshot: IntegratedGroundGraphSnapshot,
    visibility_snapshot: GroundVisibilitySnapshot,
) -> None:
    expected = {
        (observation.satellite_id, observation.station_id): observation
        for observation in visibility_snapshot.visible_links
    }
    actual_edges = [
        edge
        for edge in integrated_snapshot.canonical_edges
        if edge.edge_kind is IntegratedEdgeKind.SATELLITE_GROUND
    ]
    actual = {
        (edge.endpoint_a.satellite_id, edge.endpoint_b.ground_station_id): edge
        for edge in actual_edges
    }
    if len(actual) != len(actual_edges):
        raise ValueError("Integrated graph contains duplicate satellite-ground edges")
    if set(actual) != set(expected):
        raise ValueError("Integrated satellite-ground edge set does not match visible observations")
    for key, observation in expected.items():
        attributes = attributes_to_dict(actual[key].attributes)
        required = {
            "elevation_deg": observation.elevation_deg,
            "frame_contract_version": visibility_snapshot.frame_contract_version,
            "integrated_edge_kind": IntegratedEdgeKind.SATELLITE_GROUND.value,
            "satellite_id": observation.satellite_id,
            "slant_range_km": observation.slant_range_km,
            "station_id": observation.station_id,
            "visibility_model_version": visibility_snapshot.visibility_model_version,
            "visibility_policy_hash": visibility_snapshot.visibility_policy_hash,
            "visibility_snapshot_hash": visibility_snapshot.snapshot_hash,
            "wgs84_model_version": visibility_snapshot.wgs84_model_version,
        }
        if attributes != required:
            raise ValueError(f"Satellite-ground edge attributes do not match G2: {key}")
