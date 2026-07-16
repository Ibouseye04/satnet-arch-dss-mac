from __future__ import annotations

from datetime import datetime, timezone
import math

import networkx as nx
import pytest

from satnet.ground.graph_attributes import (
    CanonicalAttribute,
    CanonicalAttributeType,
    attributes_to_dict,
    canonicalize_attributes,
    validate_canonical_attributes,
)
from satnet.ground.integrated_graph import (
    CanonicalIntegratedEdge,
    CanonicalIntegratedNode,
    IntegratedEdgeKind,
    IntegratedNodeKind,
    IntegratedNodeRef,
    create_integrated_graph_snapshot,
    operational_snapshot_from_networkx,
    project_satellite_subgraph,
    validate_satellite_projection,
)

TIMESTAMP = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
SATELLITE_HASH = "a" * 64
GROUND_HASH = "b" * 64
POLICY_HASH = "c" * 64
VISIBILITY_HASH = "d" * 64


def source_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.graph.update({"name": "source", "revision": 1, "active": True, "note": None})
    graph.add_node(2, type="satellite", plane=0, gain=1.25, code="1.25")
    graph.add_node(10, type="satellite", plane=1, gain=2.5, code="2.5")
    graph.add_edge(2, 10, distance_km=1000.0, link_type="inter_plane", enabled=True)
    return graph


def source_snapshot():
    return operational_snapshot_from_networkx(
        timestep_index=0,
        timestamp_utc=TIMESTAMP,
        satellite_config_hash=SATELLITE_HASH,
        graph=source_graph(),
    )


def integrated_snapshot():
    source = source_snapshot()
    satellite_nodes = tuple(
        CanonicalIntegratedNode(
            node_ref=IntegratedNodeRef.for_satellite(node.satellite_id),
            attributes=node.attributes,
        )
        for node in source.canonical_nodes
    )
    ground = CanonicalIntegratedNode(
        node_ref=IntegratedNodeRef.for_ground_station("CIV_G3_TEST_001"),
        attributes=canonicalize_attributes(
            {
                "station_class": "civilian",
                "latitude_deg": 0.0,
                "enabled": True,
            }
        ),
    )
    isl_edges = tuple(
        CanonicalIntegratedEdge(
            edge_kind=IntegratedEdgeKind.INTER_SATELLITE,
            endpoint_a=IntegratedNodeRef.for_satellite(edge.satellite_id_a),
            endpoint_b=IntegratedNodeRef.for_satellite(edge.satellite_id_b),
            attributes=edge.attributes,
        )
        for edge in source.canonical_edges
    )
    ground_edge = CanonicalIntegratedEdge(
        edge_kind=IntegratedEdgeKind.SATELLITE_GROUND,
        endpoint_a=IntegratedNodeRef.for_satellite(2),
        endpoint_b=IntegratedNodeRef.for_ground_station("CIV_G3_TEST_001"),
        attributes=canonicalize_attributes(
            {"elevation_deg": 45.0, "slant_range_km": 500.0}
        ),
    )
    return create_integrated_graph_snapshot(
        timestep_index=0,
        timestamp_utc=TIMESTAMP,
        satellite_config_hash=SATELLITE_HASH,
        ground_design_hash=GROUND_HASH,
        visibility_policy_hash=POLICY_HASH,
        visibility_snapshot_hash=VISIBILITY_HASH,
        graph_attributes=source.canonical_graph_attributes,
        nodes=satellite_nodes + (ground,),
        edges=isl_edges + (ground_edge,),
    )


@pytest.mark.parametrize(
    ("value", "expected_type", "expected_value"),
    [
        (None, CanonicalAttributeType.NONE, None),
        (True, CanonicalAttributeType.BOOLEAN, True),
        (7, CanonicalAttributeType.INTEGER, 7),
        (1.25, CanonicalAttributeType.FLOAT, "1.25"),
        ("1.25", CanonicalAttributeType.STRING, "1.25"),
    ],
)
def test_canonical_attribute_types_are_explicit(
    value: object, expected_type: CanonicalAttributeType, expected_value: object
) -> None:
    attribute = CanonicalAttribute.from_value("value", value)
    assert attribute.value_type is expected_type
    assert attribute.value == expected_value
    restored = attribute.to_value()
    if expected_type is CanonicalAttributeType.FLOAT:
        assert type(restored) is float
        assert restored == 1.25
    else:
        assert type(restored) is type(value) or restored is None


def test_boolean_integer_and_float_string_remain_distinct() -> None:
    attributes = canonicalize_attributes(
        {"boolean": True, "integer": 1, "float": 1.0, "string": "1"}
    )
    restored = attributes_to_dict(attributes)
    assert type(restored["boolean"]) is bool
    assert type(restored["integer"]) is int
    assert type(restored["float"]) is float
    assert type(restored["string"]) is str


@pytest.mark.parametrize(
    "value",
    [[1], (1,), {1}, {"a": 1}, b"x", object(), math.nan, math.inf, -math.inf],
)
def test_unsupported_and_nonfinite_attributes_fail(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        CanonicalAttribute.from_value("value", value)


def test_canonical_attributes_are_sorted_and_duplicates_fail() -> None:
    attributes = canonicalize_attributes({"z": 1, "a": 2})
    assert tuple(attribute.name for attribute in attributes) == ("a", "z")
    duplicate = (
        CanonicalAttribute.from_value("a", 1),
        CanonicalAttribute.from_value("a", 2),
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_canonical_attributes(duplicate)


@pytest.mark.parametrize("satellite_id", [True, -1, "1", 1.0])
def test_satellite_node_ref_requires_exact_integer(satellite_id: object) -> None:
    with pytest.raises(TypeError, match="satellite_id"):
        IntegratedNodeRef.for_satellite(satellite_id)


def test_integrated_node_union_invariants() -> None:
    satellite = IntegratedNodeRef.for_satellite(2)
    ground = IntegratedNodeRef.for_ground_station("CIV_G3_TEST_001")
    assert satellite.kind is IntegratedNodeKind.SATELLITE
    assert ground.kind is IntegratedNodeKind.GROUND_STATION
    with pytest.raises(ValueError):
        IntegratedNodeRef(
            IntegratedNodeKind.SATELLITE,
            satellite_id=2,
            ground_station_id="CIV_G3_TEST_001",
        )
    with pytest.raises(ValueError):
        IntegratedNodeRef(IntegratedNodeKind.GROUND_STATION, satellite_id=2)


def test_operational_snapshot_round_trips_exact_types_and_metadata() -> None:
    snapshot = source_snapshot()
    first = snapshot.to_networkx()
    second = snapshot.to_networkx()
    assert first is not second
    assert first.graph == source_graph().graph
    assert dict(first.nodes(data=True)) == dict(source_graph().nodes(data=True))
    assert {
        (min(u, v), max(u, v)): attributes for u, v, attributes in first.edges(data=True)
    } == {
        (min(u, v), max(u, v)): attributes
        for u, v, attributes in source_graph().edges(data=True)
    }
    assert list(node.satellite_id for node in snapshot.canonical_nodes) == [2, 10]


def test_source_snapshot_rejects_directed_multigraph_and_boolean_nodes() -> None:
    for graph in (nx.DiGraph(), nx.MultiGraph(), nx.MultiDiGraph()):
        with pytest.raises(TypeError, match="exactly"):
            operational_snapshot_from_networkx(
                timestep_index=0,
                timestamp_utc=TIMESTAMP,
                satellite_config_hash=SATELLITE_HASH,
                graph=graph,
            )
    graph = nx.Graph()
    graph.add_node(True)
    with pytest.raises(TypeError, match="node IDs"):
        operational_snapshot_from_networkx(
            timestep_index=0,
            timestamp_utc=TIMESTAMP,
            satellite_config_hash=SATELLITE_HASH,
            graph=graph,
        )


def test_integrated_snapshot_uses_typed_order_and_exact_counts() -> None:
    snapshot = integrated_snapshot()
    refs = [node.node_ref for node in snapshot.canonical_nodes]
    assert refs == [
        IntegratedNodeRef.for_satellite(2),
        IntegratedNodeRef.for_satellite(10),
        IntegratedNodeRef.for_ground_station("CIV_G3_TEST_001"),
    ]
    assert snapshot.satellite_node_count == 2
    assert snapshot.ground_node_count == 1
    assert snapshot.isl_edge_count == 1
    assert snapshot.satellite_ground_edge_count == 1
    assert len(snapshot.graph_hash) == 64


def test_mutating_networkx_view_does_not_mutate_snapshot() -> None:
    snapshot = integrated_snapshot()
    original_hash = snapshot.graph_hash
    first = snapshot.to_networkx()
    first.clear()
    first.graph["tampered"] = True
    second = snapshot.to_networkx()
    assert first is not second
    assert len(second.nodes) == 3
    assert len(second.edges) == 2
    assert "tampered" not in second.graph
    assert snapshot.graph_hash == original_hash


def test_satellite_projection_exactly_restores_source_graph() -> None:
    source = source_snapshot()
    integrated = integrated_snapshot()
    validate_satellite_projection(
        source_snapshot=source,
        integrated_snapshot=integrated,
    )
    projected = project_satellite_subgraph(integrated)
    assert set(projected.nodes) == {2, 10}
    assert all(type(node_id) is int for node_id in projected.nodes)
    assert projected.graph == source.to_networkx().graph
    assert dict(projected.nodes(data=True)) == dict(source.to_networkx().nodes(data=True))
    assert all("integrated_node_kind" not in attrs for _, attrs in projected.nodes(data=True))


def test_edge_union_invariants() -> None:
    sat2 = IntegratedNodeRef.for_satellite(2)
    sat10 = IntegratedNodeRef.for_satellite(10)
    ground = IntegratedNodeRef.for_ground_station("CIV_G3_TEST_001")
    attributes = canonicalize_attributes({})
    CanonicalIntegratedEdge(IntegratedEdgeKind.INTER_SATELLITE, sat2, sat10, attributes)
    CanonicalIntegratedEdge(IntegratedEdgeKind.SATELLITE_GROUND, sat2, ground, attributes)
    with pytest.raises(ValueError):
        CanonicalIntegratedEdge(IntegratedEdgeKind.INTER_SATELLITE, sat10, sat2, attributes)
    with pytest.raises(ValueError):
        CanonicalIntegratedEdge(IntegratedEdgeKind.SATELLITE_GROUND, ground, sat2, attributes)
