from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import re

import networkx as nx

from satnet.ground.canonical import canonical_hash, canonical_utc_timestamp
from satnet.ground.catalog import STATION_ID_PATTERN
from satnet.ground.graph_attributes import (
    CanonicalAttribute,
    attributes_to_dict,
    canonicalize_attributes,
    validate_canonical_attributes,
)

INTEGRATED_GRAPH_MODEL_VERSION = "1"
INTEGRATED_GRAPH_SCHEMA_VERSION = "1"
INTEGRATED_GRAPH_IDENTITY_DOMAIN = "satnet_integrated_ground_graph"
INTEGRATED_GRAPH_IDENTITY_VERSION = "1"
OPERATIONAL_GRAPH_IDENTITY_DOMAIN = "satnet_operational_satellite_graph"
OPERATIONAL_GRAPH_IDENTITY_VERSION = "1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class IntegratedNodeKind(str, Enum):
    SATELLITE = "satellite"
    GROUND_STATION = "ground_station"


@dataclass(frozen=True)
class IntegratedNodeRef:
    kind: IntegratedNodeKind
    satellite_id: int | None = None
    ground_station_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, IntegratedNodeKind):
            raise TypeError("kind must be an IntegratedNodeKind")
        if self.kind is IntegratedNodeKind.SATELLITE:
            if type(self.satellite_id) is not int or self.satellite_id < 0:
                raise TypeError("Satellite node requires a nonnegative integer satellite_id")
            if self.ground_station_id is not None:
                raise ValueError("Satellite node must not define ground_station_id")
        else:
            if self.satellite_id is not None:
                raise ValueError("Ground node must not define satellite_id")
            if not isinstance(
                self.ground_station_id, str
            ) or not STATION_ID_PATTERN.fullmatch(self.ground_station_id):
                raise ValueError("Ground node requires a valid ground_station_id")

    @classmethod
    def for_satellite(cls, satellite_id: int) -> "IntegratedNodeRef":
        return cls(IntegratedNodeKind.SATELLITE, satellite_id=satellite_id)

    @classmethod
    def for_ground_station(cls, station_id: str) -> "IntegratedNodeRef":
        return cls(IntegratedNodeKind.GROUND_STATION, ground_station_id=station_id)

    def canonical_record(self) -> dict[str, object]:
        return {
            "ground_station_id": self.ground_station_id,
            "kind": self.kind.value,
            "satellite_id": self.satellite_id,
        }


class IntegratedEdgeKind(str, Enum):
    INTER_SATELLITE = "inter_satellite"
    SATELLITE_GROUND = "satellite_ground"


@dataclass(frozen=True)
class CanonicalOperationalSatelliteNode:
    satellite_id: int
    attributes: tuple[CanonicalAttribute, ...]

    def __post_init__(self) -> None:
        if type(self.satellite_id) is not int or self.satellite_id < 0:
            raise TypeError("satellite_id must be a nonnegative integer")
        validate_canonical_attributes(self.attributes)

    def canonical_record(self) -> dict[str, object]:
        return {
            "attributes": [attribute.canonical_record() for attribute in self.attributes],
            "satellite_id": self.satellite_id,
        }


@dataclass(frozen=True)
class CanonicalOperationalSatelliteEdge:
    satellite_id_a: int
    satellite_id_b: int
    attributes: tuple[CanonicalAttribute, ...]

    def __post_init__(self) -> None:
        if type(self.satellite_id_a) is not int or type(self.satellite_id_b) is not int:
            raise TypeError("Operational edge endpoints must be integers")
        if self.satellite_id_a < 0 or self.satellite_id_a >= self.satellite_id_b:
            raise ValueError("Operational edge endpoints must be distinct, nonnegative, and ascending")
        validate_canonical_attributes(self.attributes)

    def canonical_record(self) -> dict[str, object]:
        return {
            "attributes": [attribute.canonical_record() for attribute in self.attributes],
            "satellite_id_a": self.satellite_id_a,
            "satellite_id_b": self.satellite_id_b,
        }


@dataclass(frozen=True)
class CanonicalIntegratedNode:
    node_ref: IntegratedNodeRef
    attributes: tuple[CanonicalAttribute, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.node_ref, IntegratedNodeRef):
            raise TypeError("node_ref must be an IntegratedNodeRef")
        validate_canonical_attributes(self.attributes)

    def canonical_record(self) -> dict[str, object]:
        return {
            "attributes": [attribute.canonical_record() for attribute in self.attributes],
            "node_ref": self.node_ref.canonical_record(),
        }


@dataclass(frozen=True)
class CanonicalIntegratedEdge:
    edge_kind: IntegratedEdgeKind
    endpoint_a: IntegratedNodeRef
    endpoint_b: IntegratedNodeRef
    attributes: tuple[CanonicalAttribute, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.edge_kind, IntegratedEdgeKind):
            raise TypeError("edge_kind must be an IntegratedEdgeKind")
        if not isinstance(self.endpoint_a, IntegratedNodeRef) or not isinstance(
            self.endpoint_b, IntegratedNodeRef
        ):
            raise TypeError("Integrated edge endpoints must be IntegratedNodeRef values")
        if self.edge_kind is IntegratedEdgeKind.INTER_SATELLITE:
            if (
                self.endpoint_a.kind is not IntegratedNodeKind.SATELLITE
                or self.endpoint_b.kind is not IntegratedNodeKind.SATELLITE
                or self.endpoint_a.satellite_id >= self.endpoint_b.satellite_id
            ):
                raise ValueError("ISL endpoints must be distinct ascending satellite references")
        else:
            if (
                self.endpoint_a.kind is not IntegratedNodeKind.SATELLITE
                or self.endpoint_b.kind is not IntegratedNodeKind.GROUND_STATION
            ):
                raise ValueError("Satellite-ground edge must use satellite then ground endpoint")
        validate_canonical_attributes(self.attributes)

    def canonical_record(self) -> dict[str, object]:
        return {
            "attributes": [attribute.canonical_record() for attribute in self.attributes],
            "edge_kind": self.edge_kind.value,
            "endpoint_a": self.endpoint_a.canonical_record(),
            "endpoint_b": self.endpoint_b.canonical_record(),
        }


def _validate_hash(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be exactly 64 lowercase hexadecimal characters")


def _node_sort_key(node: CanonicalIntegratedNode) -> tuple[int, int | str]:
    if node.node_ref.kind is IntegratedNodeKind.SATELLITE:
        return (0, node.node_ref.satellite_id)
    return (1, node.node_ref.ground_station_id)


def _ref_sort_key(node_ref: IntegratedNodeRef) -> tuple[int, int | str]:
    if node_ref.kind is IntegratedNodeKind.SATELLITE:
        return (0, node_ref.satellite_id)
    return (1, node_ref.ground_station_id)


def _edge_sort_key(
    edge: CanonicalIntegratedEdge,
) -> tuple[str, tuple[int, int | str], tuple[int, int | str]]:
    return (
        edge.edge_kind.value,
        _ref_sort_key(edge.endpoint_a),
        _ref_sort_key(edge.endpoint_b),
    )


def _operational_graph_hash(
    *,
    timestep_index: int,
    timestamp_utc: datetime,
    satellite_config_hash: str,
    graph_attributes: tuple[CanonicalAttribute, ...],
    nodes: tuple[CanonicalOperationalSatelliteNode, ...],
    edges: tuple[CanonicalOperationalSatelliteEdge, ...],
) -> str:
    return canonical_hash(
        {
            "canonical_edges": [edge.canonical_record() for edge in edges],
            "canonical_graph_attributes": [
                attribute.canonical_record() for attribute in graph_attributes
            ],
            "canonical_nodes": [node.canonical_record() for node in nodes],
            "identity_domain": OPERATIONAL_GRAPH_IDENTITY_DOMAIN,
            "identity_version": OPERATIONAL_GRAPH_IDENTITY_VERSION,
            "satellite_config_hash": satellite_config_hash,
            "timestep_index": timestep_index,
            "timestamp_utc": canonical_utc_timestamp(timestamp_utc),
        }
    )


@dataclass(frozen=True)
class OperationalSatelliteGraphSnapshot:
    timestep_index: int
    timestamp_utc: datetime
    satellite_config_hash: str
    canonical_graph_attributes: tuple[CanonicalAttribute, ...]
    canonical_nodes: tuple[CanonicalOperationalSatelliteNode, ...]
    canonical_edges: tuple[CanonicalOperationalSatelliteEdge, ...]
    graph_hash: str

    def __post_init__(self) -> None:
        if type(self.timestep_index) is not int or self.timestep_index < 0:
            raise TypeError("timestep_index must be a nonnegative integer")
        canonical_utc_timestamp(self.timestamp_utc)
        _validate_hash(self.satellite_config_hash, "satellite_config_hash")
        validate_canonical_attributes(self.canonical_graph_attributes)
        if not isinstance(self.canonical_nodes, tuple) or any(
            not isinstance(node, CanonicalOperationalSatelliteNode)
            for node in self.canonical_nodes
        ):
            raise TypeError("canonical_nodes must contain operational satellite nodes")
        node_ids = [node.satellite_id for node in self.canonical_nodes]
        if node_ids != sorted(node_ids) or len(node_ids) != len(set(node_ids)):
            raise ValueError("Operational nodes must use unique numeric order")
        if not isinstance(self.canonical_edges, tuple) or any(
            not isinstance(edge, CanonicalOperationalSatelliteEdge)
            for edge in self.canonical_edges
        ):
            raise TypeError("canonical_edges must contain operational satellite edges")
        edge_keys = [
            (edge.satellite_id_a, edge.satellite_id_b) for edge in self.canonical_edges
        ]
        if edge_keys != sorted(edge_keys) or len(edge_keys) != len(set(edge_keys)):
            raise ValueError("Operational edges must use unique canonical order")
        if any(endpoint not in set(node_ids) for key in edge_keys for endpoint in key):
            raise ValueError("Operational edge endpoint does not exist")
        _validate_hash(self.graph_hash, "graph_hash")
        expected = _operational_graph_hash(
            timestep_index=self.timestep_index,
            timestamp_utc=self.timestamp_utc,
            satellite_config_hash=self.satellite_config_hash,
            graph_attributes=self.canonical_graph_attributes,
            nodes=self.canonical_nodes,
            edges=self.canonical_edges,
        )
        if self.graph_hash != expected:
            raise ValueError("graph_hash does not match operational graph")

    def to_networkx(self) -> nx.Graph:
        graph = nx.Graph()
        graph.graph.update(attributes_to_dict(self.canonical_graph_attributes))
        for node in self.canonical_nodes:
            graph.add_node(node.satellite_id, **attributes_to_dict(node.attributes))
        for edge in self.canonical_edges:
            graph.add_edge(
                edge.satellite_id_a,
                edge.satellite_id_b,
                **attributes_to_dict(edge.attributes),
            )
        return graph


def operational_snapshot_from_networkx(
    *,
    timestep_index: int,
    timestamp_utc: datetime,
    satellite_config_hash: str,
    graph: nx.Graph,
) -> OperationalSatelliteGraphSnapshot:
    if type(graph) is not nx.Graph or graph.is_directed() or graph.is_multigraph():
        raise TypeError("G3 model version 1 requires exactly an undirected nx.Graph")
    for node_id in graph.nodes:
        if type(node_id) is not int or node_id < 0:
            raise TypeError("Source satellite node IDs must be nonnegative integers")
    graph_attributes = canonicalize_attributes(graph.graph)
    nodes = tuple(
        CanonicalOperationalSatelliteNode(
            satellite_id=node_id,
            attributes=canonicalize_attributes(graph.nodes[node_id]),
        )
        for node_id in sorted(graph.nodes)
    )
    edges = tuple(
        CanonicalOperationalSatelliteEdge(
            satellite_id_a=min(u, v),
            satellite_id_b=max(u, v),
            attributes=canonicalize_attributes(attributes),
        )
        for u, v, attributes in sorted(
            graph.edges(data=True), key=lambda item: (min(item[0], item[1]), max(item[0], item[1]))
        )
    )
    graph_hash = _operational_graph_hash(
        timestep_index=timestep_index,
        timestamp_utc=timestamp_utc,
        satellite_config_hash=satellite_config_hash,
        graph_attributes=graph_attributes,
        nodes=nodes,
        edges=edges,
    )
    return OperationalSatelliteGraphSnapshot(
        timestep_index=timestep_index,
        timestamp_utc=timestamp_utc,
        satellite_config_hash=satellite_config_hash,
        canonical_graph_attributes=graph_attributes,
        canonical_nodes=nodes,
        canonical_edges=edges,
        graph_hash=graph_hash,
    )


def _integrated_graph_hash(
    *,
    timestep_index: int,
    timestamp_utc: datetime,
    satellite_config_hash: str,
    ground_design_hash: str,
    visibility_policy_hash: str,
    visibility_snapshot_hash: str,
    graph_attributes: tuple[CanonicalAttribute, ...],
    nodes: tuple[CanonicalIntegratedNode, ...],
    edges: tuple[CanonicalIntegratedEdge, ...],
) -> str:
    return canonical_hash(
        {
            "canonical_edges": [edge.canonical_record() for edge in edges],
            "canonical_graph_attributes": [
                attribute.canonical_record() for attribute in graph_attributes
            ],
            "canonical_nodes": [node.canonical_record() for node in nodes],
            "ground_design_hash": ground_design_hash,
            "identity_domain": INTEGRATED_GRAPH_IDENTITY_DOMAIN,
            "identity_version": INTEGRATED_GRAPH_IDENTITY_VERSION,
            "integrated_graph_model_version": INTEGRATED_GRAPH_MODEL_VERSION,
            "satellite_config_hash": satellite_config_hash,
            "timestep_index": timestep_index,
            "timestamp_utc": canonical_utc_timestamp(timestamp_utc),
            "visibility_policy_hash": visibility_policy_hash,
            "visibility_snapshot_hash": visibility_snapshot_hash,
        }
    )


@dataclass(frozen=True)
class IntegratedGroundGraphSnapshot:
    timestep_index: int
    timestamp_utc: datetime
    satellite_config_hash: str
    ground_design_hash: str
    visibility_policy_hash: str
    visibility_snapshot_hash: str
    integrated_graph_model_version: str
    canonical_graph_attributes: tuple[CanonicalAttribute, ...]
    canonical_nodes: tuple[CanonicalIntegratedNode, ...]
    canonical_edges: tuple[CanonicalIntegratedEdge, ...]
    satellite_node_count: int
    ground_node_count: int
    isl_edge_count: int
    satellite_ground_edge_count: int
    graph_hash: str

    def __post_init__(self) -> None:
        if type(self.timestep_index) is not int or self.timestep_index < 0:
            raise TypeError("timestep_index must be a nonnegative integer")
        canonical_utc_timestamp(self.timestamp_utc)
        for field_name in (
            "satellite_config_hash",
            "ground_design_hash",
            "visibility_policy_hash",
            "visibility_snapshot_hash",
        ):
            _validate_hash(getattr(self, field_name), field_name)
        if self.integrated_graph_model_version != INTEGRATED_GRAPH_MODEL_VERSION:
            raise ValueError("Unsupported integrated_graph_model_version")
        validate_canonical_attributes(self.canonical_graph_attributes)
        if not isinstance(self.canonical_nodes, tuple) or any(
            not isinstance(node, CanonicalIntegratedNode) for node in self.canonical_nodes
        ):
            raise TypeError("canonical_nodes must contain integrated nodes")
        if list(self.canonical_nodes) != sorted(self.canonical_nodes, key=_node_sort_key):
            raise ValueError("Integrated nodes must use canonical order")
        node_refs = [node.node_ref for node in self.canonical_nodes]
        if len(node_refs) != len(set(node_refs)):
            raise ValueError("Integrated nodes contain duplicate identities")
        if not isinstance(self.canonical_edges, tuple) or any(
            not isinstance(edge, CanonicalIntegratedEdge) for edge in self.canonical_edges
        ):
            raise TypeError("canonical_edges must contain integrated edges")
        if list(self.canonical_edges) != sorted(self.canonical_edges, key=_edge_sort_key):
            raise ValueError("Integrated edges must use canonical order")
        edge_keys = [
            (edge.edge_kind, edge.endpoint_a, edge.endpoint_b)
            for edge in self.canonical_edges
        ]
        if len(edge_keys) != len(set(edge_keys)):
            raise ValueError("Integrated edges contain duplicate identities")
        if any(
            endpoint not in set(node_refs)
            for edge in self.canonical_edges
            for endpoint in (edge.endpoint_a, edge.endpoint_b)
        ):
            raise ValueError("Integrated edge endpoint does not exist")
        expected_counts = (
            sum(node.node_ref.kind is IntegratedNodeKind.SATELLITE for node in self.canonical_nodes),
            sum(node.node_ref.kind is IntegratedNodeKind.GROUND_STATION for node in self.canonical_nodes),
            sum(edge.edge_kind is IntegratedEdgeKind.INTER_SATELLITE for edge in self.canonical_edges),
            sum(edge.edge_kind is IntegratedEdgeKind.SATELLITE_GROUND for edge in self.canonical_edges),
        )
        actual_counts = (
            self.satellite_node_count,
            self.ground_node_count,
            self.isl_edge_count,
            self.satellite_ground_edge_count,
        )
        if any(type(count) is not int or count < 0 for count in actual_counts):
            raise TypeError("Integrated graph counts must be nonnegative integers")
        if actual_counts != expected_counts:
            raise ValueError("Integrated graph counts do not match canonical records")
        _validate_hash(self.graph_hash, "graph_hash")
        expected_hash = _integrated_graph_hash(
            timestep_index=self.timestep_index,
            timestamp_utc=self.timestamp_utc,
            satellite_config_hash=self.satellite_config_hash,
            ground_design_hash=self.ground_design_hash,
            visibility_policy_hash=self.visibility_policy_hash,
            visibility_snapshot_hash=self.visibility_snapshot_hash,
            graph_attributes=self.canonical_graph_attributes,
            nodes=self.canonical_nodes,
            edges=self.canonical_edges,
        )
        if self.graph_hash != expected_hash:
            raise ValueError("graph_hash does not match integrated graph")

    def to_networkx(self) -> nx.Graph:
        graph = nx.Graph()
        graph.graph.update(attributes_to_dict(self.canonical_graph_attributes))
        for node in self.canonical_nodes:
            graph.add_node(node.node_ref, **attributes_to_dict(node.attributes))
        for edge in self.canonical_edges:
            graph.add_edge(
                edge.endpoint_a,
                edge.endpoint_b,
                **attributes_to_dict(edge.attributes),
            )
        return graph


def create_integrated_graph_snapshot(
    *,
    timestep_index: int,
    timestamp_utc: datetime,
    satellite_config_hash: str,
    ground_design_hash: str,
    visibility_policy_hash: str,
    visibility_snapshot_hash: str,
    graph_attributes: tuple[CanonicalAttribute, ...],
    nodes: tuple[CanonicalIntegratedNode, ...],
    edges: tuple[CanonicalIntegratedEdge, ...],
) -> IntegratedGroundGraphSnapshot:
    ordered_nodes = tuple(sorted(nodes, key=_node_sort_key))
    ordered_edges = tuple(sorted(edges, key=_edge_sort_key))
    graph_hash = _integrated_graph_hash(
        timestep_index=timestep_index,
        timestamp_utc=timestamp_utc,
        satellite_config_hash=satellite_config_hash,
        ground_design_hash=ground_design_hash,
        visibility_policy_hash=visibility_policy_hash,
        visibility_snapshot_hash=visibility_snapshot_hash,
        graph_attributes=graph_attributes,
        nodes=ordered_nodes,
        edges=ordered_edges,
    )
    return IntegratedGroundGraphSnapshot(
        timestep_index=timestep_index,
        timestamp_utc=timestamp_utc,
        satellite_config_hash=satellite_config_hash,
        ground_design_hash=ground_design_hash,
        visibility_policy_hash=visibility_policy_hash,
        visibility_snapshot_hash=visibility_snapshot_hash,
        integrated_graph_model_version=INTEGRATED_GRAPH_MODEL_VERSION,
        canonical_graph_attributes=graph_attributes,
        canonical_nodes=ordered_nodes,
        canonical_edges=ordered_edges,
        satellite_node_count=sum(
            node.node_ref.kind is IntegratedNodeKind.SATELLITE for node in ordered_nodes
        ),
        ground_node_count=sum(
            node.node_ref.kind is IntegratedNodeKind.GROUND_STATION for node in ordered_nodes
        ),
        isl_edge_count=sum(
            edge.edge_kind is IntegratedEdgeKind.INTER_SATELLITE for edge in ordered_edges
        ),
        satellite_ground_edge_count=sum(
            edge.edge_kind is IntegratedEdgeKind.SATELLITE_GROUND for edge in ordered_edges
        ),
        graph_hash=graph_hash,
    )


def project_satellite_subgraph(
    integrated_snapshot: IntegratedGroundGraphSnapshot,
) -> nx.Graph:
    if not isinstance(integrated_snapshot, IntegratedGroundGraphSnapshot):
        raise TypeError("integrated_snapshot must be an IntegratedGroundGraphSnapshot")
    graph = nx.Graph()
    graph.graph.update(attributes_to_dict(integrated_snapshot.canonical_graph_attributes))
    for node in integrated_snapshot.canonical_nodes:
        if node.node_ref.kind is IntegratedNodeKind.SATELLITE:
            graph.add_node(
                node.node_ref.satellite_id,
                **attributes_to_dict(node.attributes),
            )
    for edge in integrated_snapshot.canonical_edges:
        if edge.edge_kind is IntegratedEdgeKind.INTER_SATELLITE:
            graph.add_edge(
                edge.endpoint_a.satellite_id,
                edge.endpoint_b.satellite_id,
                **attributes_to_dict(edge.attributes),
            )
    return graph


def validate_satellite_projection(
    *,
    source_snapshot: OperationalSatelliteGraphSnapshot,
    integrated_snapshot: IntegratedGroundGraphSnapshot,
) -> None:
    source = source_snapshot.to_networkx()
    projected = project_satellite_subgraph(integrated_snapshot)
    if type(source) is not type(projected):
        raise ValueError("Satellite projection graph type mismatch")
    if source.is_directed() != projected.is_directed():
        raise ValueError("Satellite projection directedness mismatch")
    if source.is_multigraph() != projected.is_multigraph():
        raise ValueError("Satellite projection multigraph mismatch")
    if source.graph != projected.graph:
        raise ValueError("Satellite projection graph metadata mismatch")
    if set(source.nodes) != set(projected.nodes):
        raise ValueError("Satellite projection node identity mismatch")
    for node_id in source.nodes:
        if source.nodes[node_id] != projected.nodes[node_id]:
            raise ValueError(f"Satellite projection node attribute mismatch: {node_id}")
    source_edges = {
        (min(u, v), max(u, v)): attributes for u, v, attributes in source.edges(data=True)
    }
    projected_edges = {
        (min(u, v), max(u, v)): attributes
        for u, v, attributes in projected.edges(data=True)
    }
    if source_edges != projected_edges:
        raise ValueError("Satellite projection edge or attribute mismatch")
