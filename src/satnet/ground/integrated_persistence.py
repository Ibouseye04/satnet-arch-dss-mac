from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from satnet.ground.canonical import canonical_hash, canonical_json, canonical_utc_timestamp
from satnet.ground.catalog import GroundStationCatalog
from satnet.ground.graph_attributes import CanonicalAttribute, CanonicalAttributeType
from satnet.ground.integrated_builder import build_integrated_ground_graph
from satnet.ground.integrated_graph import (
    INTEGRATED_GRAPH_MODEL_VERSION,
    INTEGRATED_GRAPH_SCHEMA_VERSION,
    CanonicalIntegratedEdge,
    CanonicalIntegratedNode,
    IntegratedEdgeKind,
    IntegratedGroundGraphSnapshot,
    IntegratedNodeKind,
    IntegratedNodeRef,
)
from satnet.ground.persistence import GroundRunDesignRecord
from satnet.ground.position_adapter import reconstruct_operational_satellite_position_sequence
from satnet.ground.satellite_graph_adapter import reconstruct_operational_satellite_graph_sequence
from satnet.ground.visibility import GroundVisibilityPolicy
from satnet.ground.visibility_persistence import (
    GroundVisibilityRecord,
    replay_ground_visibility_records,
)
from satnet.simulation.tier1_rollout import Tier1FailureRealization, Tier1RolloutConfig

INTEGRATED_GRAPH_RECORD_IDENTITY_DOMAIN = "satnet_integrated_ground_graph_record"
INTEGRATED_GRAPH_RECORD_IDENTITY_VERSION = "1"
RECORD_FIELDS = frozenset(
    {
        "integrated_graph_schema_version",
        "run_id",
        "timestep_index",
        "timestamp_utc",
        "satellite_config_hash",
        "ground_design_hash",
        "visibility_policy_hash",
        "visibility_snapshot_hash",
        "integrated_graph_model_version",
        "satellite_node_count",
        "ground_node_count",
        "isl_edge_count",
        "satellite_ground_edge_count",
        "canonical_graph_attributes",
        "canonical_nodes",
        "canonical_edges",
        "graph_hash",
        "record_hash",
    }
)


def _parse_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a canonical UTC string")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{field_name} must use canonical UTC format") from exc
    if canonical_utc_timestamp(parsed) != value:
        raise ValueError(f"{field_name} is not canonical UTC")
    return parsed


def _record_hash(
    *, run_id: int, timestep_index: int, timestamp_utc: datetime, graph_hash: str
) -> str:
    return canonical_hash(
        {
            "graph_hash": graph_hash,
            "identity_domain": INTEGRATED_GRAPH_RECORD_IDENTITY_DOMAIN,
            "identity_version": INTEGRATED_GRAPH_RECORD_IDENTITY_VERSION,
            "integrated_graph_schema_version": INTEGRATED_GRAPH_SCHEMA_VERSION,
            "run_id": run_id,
            "timestep_index": timestep_index,
            "timestamp_utc": canonical_utc_timestamp(timestamp_utc),
        }
    )


@dataclass(frozen=True)
class IntegratedGroundGraphRecord:
    integrated_graph_schema_version: str
    run_id: int
    timestep_index: int
    timestamp_utc: datetime
    satellite_config_hash: str
    ground_design_hash: str
    visibility_policy_hash: str
    visibility_snapshot_hash: str
    integrated_graph_model_version: str
    satellite_node_count: int
    ground_node_count: int
    isl_edge_count: int
    satellite_ground_edge_count: int
    canonical_graph_attributes: tuple[CanonicalAttribute, ...]
    canonical_nodes: tuple[CanonicalIntegratedNode, ...]
    canonical_edges: tuple[CanonicalIntegratedEdge, ...]
    graph_hash: str
    record_hash: str

    def __post_init__(self) -> None:
        if self.integrated_graph_schema_version != INTEGRATED_GRAPH_SCHEMA_VERSION:
            raise ValueError("Unsupported integrated_graph_schema_version")
        if type(self.run_id) is not int or self.run_id < 0:
            raise TypeError("run_id must be a nonnegative integer")
        snapshot = IntegratedGroundGraphSnapshot(
            timestep_index=self.timestep_index,
            timestamp_utc=self.timestamp_utc,
            satellite_config_hash=self.satellite_config_hash,
            ground_design_hash=self.ground_design_hash,
            visibility_policy_hash=self.visibility_policy_hash,
            visibility_snapshot_hash=self.visibility_snapshot_hash,
            integrated_graph_model_version=self.integrated_graph_model_version,
            canonical_graph_attributes=self.canonical_graph_attributes,
            canonical_nodes=self.canonical_nodes,
            canonical_edges=self.canonical_edges,
            satellite_node_count=self.satellite_node_count,
            ground_node_count=self.ground_node_count,
            isl_edge_count=self.isl_edge_count,
            satellite_ground_edge_count=self.satellite_ground_edge_count,
            graph_hash=self.graph_hash,
        )
        expected = _record_hash(
            run_id=self.run_id,
            timestep_index=snapshot.timestep_index,
            timestamp_utc=snapshot.timestamp_utc,
            graph_hash=snapshot.graph_hash,
        )
        if self.record_hash != expected:
            raise ValueError("record_hash does not match canonical integrated graph record")

    def to_snapshot(self) -> IntegratedGroundGraphSnapshot:
        return IntegratedGroundGraphSnapshot(
            timestep_index=self.timestep_index,
            timestamp_utc=self.timestamp_utc,
            satellite_config_hash=self.satellite_config_hash,
            ground_design_hash=self.ground_design_hash,
            visibility_policy_hash=self.visibility_policy_hash,
            visibility_snapshot_hash=self.visibility_snapshot_hash,
            integrated_graph_model_version=self.integrated_graph_model_version,
            canonical_graph_attributes=self.canonical_graph_attributes,
            canonical_nodes=self.canonical_nodes,
            canonical_edges=self.canonical_edges,
            satellite_node_count=self.satellite_node_count,
            ground_node_count=self.ground_node_count,
            isl_edge_count=self.isl_edge_count,
            satellite_ground_edge_count=self.satellite_ground_edge_count,
            graph_hash=self.graph_hash,
        )

    def to_manifest_object(self) -> dict[str, object]:
        return {
            "canonical_edges": [edge.canonical_record() for edge in self.canonical_edges],
            "canonical_graph_attributes": [
                attribute.canonical_record() for attribute in self.canonical_graph_attributes
            ],
            "canonical_nodes": [node.canonical_record() for node in self.canonical_nodes],
            "graph_hash": self.graph_hash,
            "ground_design_hash": self.ground_design_hash,
            "ground_node_count": self.ground_node_count,
            "integrated_graph_model_version": self.integrated_graph_model_version,
            "integrated_graph_schema_version": self.integrated_graph_schema_version,
            "isl_edge_count": self.isl_edge_count,
            "record_hash": self.record_hash,
            "run_id": self.run_id,
            "satellite_config_hash": self.satellite_config_hash,
            "satellite_ground_edge_count": self.satellite_ground_edge_count,
            "satellite_node_count": self.satellite_node_count,
            "timestep_index": self.timestep_index,
            "timestamp_utc": canonical_utc_timestamp(self.timestamp_utc),
            "visibility_policy_hash": self.visibility_policy_hash,
            "visibility_snapshot_hash": self.visibility_snapshot_hash,
        }


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key '{key}'")
        result[key] = value
    return result


def _attribute_from_object(value: object) -> CanonicalAttribute:
    if not isinstance(value, dict) or set(value) != {"name", "value", "value_type"}:
        raise ValueError("Invalid canonical attribute object")
    try:
        value_type = CanonicalAttributeType(value["value_type"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid canonical attribute type") from exc
    return CanonicalAttribute(value["name"], value_type, value["value"])


def _attributes_from_objects(value: object) -> tuple[CanonicalAttribute, ...]:
    if not isinstance(value, list):
        raise TypeError("Canonical attributes must be an array")
    return tuple(_attribute_from_object(attribute) for attribute in value)


def _ref_from_object(value: object) -> IntegratedNodeRef:
    if not isinstance(value, dict) or set(value) != {
        "kind",
        "satellite_id",
        "ground_station_id",
    }:
        raise ValueError("Invalid integrated node reference")
    try:
        kind = IntegratedNodeKind(value["kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid integrated node kind") from exc
    return IntegratedNodeRef(
        kind=kind,
        satellite_id=value["satellite_id"],
        ground_station_id=value["ground_station_id"],
    )


def _node_from_object(value: object) -> CanonicalIntegratedNode:
    if not isinstance(value, dict) or set(value) != {"node_ref", "attributes"}:
        raise ValueError("Invalid canonical integrated node")
    return CanonicalIntegratedNode(
        node_ref=_ref_from_object(value["node_ref"]),
        attributes=_attributes_from_objects(value["attributes"]),
    )


def _edge_from_object(value: object) -> CanonicalIntegratedEdge:
    if not isinstance(value, dict) or set(value) != {
        "edge_kind",
        "endpoint_a",
        "endpoint_b",
        "attributes",
    }:
        raise ValueError("Invalid canonical integrated edge")
    try:
        edge_kind = IntegratedEdgeKind(value["edge_kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid integrated edge kind") from exc
    return CanonicalIntegratedEdge(
        edge_kind=edge_kind,
        endpoint_a=_ref_from_object(value["endpoint_a"]),
        endpoint_b=_ref_from_object(value["endpoint_b"]),
        attributes=_attributes_from_objects(value["attributes"]),
    )


def _record_from_object(value: object, line_number: int) -> IntegratedGroundGraphRecord:
    if not isinstance(value, dict):
        raise ValueError(f"Line {line_number}: integrated graph record must be an object")
    missing = sorted(RECORD_FIELDS - set(value))
    unknown = sorted(set(value) - RECORD_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"Line {line_number}: integrated graph fields invalid; "
            f"missing={missing}, unknown={unknown}"
        )
    for field_name in (
        "run_id",
        "timestep_index",
        "satellite_node_count",
        "ground_node_count",
        "isl_edge_count",
        "satellite_ground_edge_count",
    ):
        if type(value[field_name]) is not int:
            raise TypeError(f"Line {line_number}: {field_name} must be an integer")
    try:
        return IntegratedGroundGraphRecord(
            integrated_graph_schema_version=value["integrated_graph_schema_version"],
            run_id=value["run_id"],
            timestep_index=value["timestep_index"],
            timestamp_utc=_parse_timestamp(value["timestamp_utc"], "timestamp_utc"),
            satellite_config_hash=value["satellite_config_hash"],
            ground_design_hash=value["ground_design_hash"],
            visibility_policy_hash=value["visibility_policy_hash"],
            visibility_snapshot_hash=value["visibility_snapshot_hash"],
            integrated_graph_model_version=value["integrated_graph_model_version"],
            satellite_node_count=value["satellite_node_count"],
            ground_node_count=value["ground_node_count"],
            isl_edge_count=value["isl_edge_count"],
            satellite_ground_edge_count=value["satellite_ground_edge_count"],
            canonical_graph_attributes=_attributes_from_objects(
                value["canonical_graph_attributes"]
            ),
            canonical_nodes=tuple(_node_from_object(node) for node in value["canonical_nodes"]),
            canonical_edges=tuple(_edge_from_object(edge) for edge in value["canonical_edges"]),
            graph_hash=value["graph_hash"],
            record_hash=value["record_hash"],
        )
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"Line {line_number}: {exc}") from exc


def make_integrated_graph_record(
    *,
    ground_design: GroundRunDesignRecord,
    snapshot: IntegratedGroundGraphSnapshot,
) -> IntegratedGroundGraphRecord:
    if not isinstance(ground_design, GroundRunDesignRecord):
        raise TypeError("ground_design must be a GroundRunDesignRecord")
    if snapshot.ground_design_hash != ground_design.ground_design_hash:
        raise ValueError("Snapshot ground-design hash mismatch")
    if snapshot.satellite_config_hash != ground_design.satellite_config_hash:
        raise ValueError("Snapshot satellite hash mismatch")
    record_hash = _record_hash(
        run_id=ground_design.run_id,
        timestep_index=snapshot.timestep_index,
        timestamp_utc=snapshot.timestamp_utc,
        graph_hash=snapshot.graph_hash,
    )
    return IntegratedGroundGraphRecord(
        integrated_graph_schema_version=INTEGRATED_GRAPH_SCHEMA_VERSION,
        run_id=ground_design.run_id,
        timestep_index=snapshot.timestep_index,
        timestamp_utc=snapshot.timestamp_utc,
        satellite_config_hash=snapshot.satellite_config_hash,
        ground_design_hash=snapshot.ground_design_hash,
        visibility_policy_hash=snapshot.visibility_policy_hash,
        visibility_snapshot_hash=snapshot.visibility_snapshot_hash,
        integrated_graph_model_version=snapshot.integrated_graph_model_version,
        satellite_node_count=snapshot.satellite_node_count,
        ground_node_count=snapshot.ground_node_count,
        isl_edge_count=snapshot.isl_edge_count,
        satellite_ground_edge_count=snapshot.satellite_ground_edge_count,
        canonical_graph_attributes=snapshot.canonical_graph_attributes,
        canonical_nodes=snapshot.canonical_nodes,
        canonical_edges=snapshot.canonical_edges,
        graph_hash=snapshot.graph_hash,
        record_hash=record_hash,
    )


def _validate_unique_keys(records: Sequence[IntegratedGroundGraphRecord]) -> None:
    keys = [(record.run_id, record.timestep_index) for record in records]
    duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
    if duplicates:
        raise ValueError(f"Duplicate integrated graph run/timestep keys: {duplicates}")


def read_integrated_graph_manifest(
    path: str | Path,
) -> tuple[IntegratedGroundGraphRecord, ...]:
    manifest_path = Path(path)
    if manifest_path.suffix != ".jsonl":
        raise ValueError("Canonical integrated graph manifest must use .jsonl")
    text = manifest_path.read_text(encoding="utf-8")
    if not text:
        raise ValueError("Integrated graph manifest is empty")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    if not lines or any(line == "" for line in lines):
        raise ValueError("Integrated graph manifest contains an empty line")
    records: list[IntegratedGroundGraphRecord] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line, object_pairs_hook=_pairs_without_duplicates)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Line {line_number}: malformed JSON: {exc}") from exc
        records.append(_record_from_object(value, line_number))
    _validate_unique_keys(records)
    return tuple(sorted(records, key=lambda record: (record.run_id, record.timestep_index)))


def write_integrated_graph_manifest(
    records: Sequence[IntegratedGroundGraphRecord],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    manifest_path = Path(path)
    if manifest_path.suffix != ".jsonl":
        raise ValueError("Canonical integrated graph manifest must use .jsonl")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a Boolean")
    if not records or any(not isinstance(record, IntegratedGroundGraphRecord) for record in records):
        raise ValueError("Integrated graph manifest requires valid records")
    _validate_unique_keys(records)
    ordered = sorted(records, key=lambda record: (record.run_id, record.timestep_index))
    serialized = [canonical_json(record.to_manifest_object()) for record in ordered]
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Integrated graph manifest already exists: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(serialized) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"Integrated graph manifest already exists: {manifest_path}")
        os.replace(temporary_path, manifest_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def validate_integrated_records_against_sources(
    *,
    records: Sequence[IntegratedGroundGraphRecord],
    source_snapshots,
    ground_design: GroundRunDesignRecord,
    visibility_records: Sequence[GroundVisibilityRecord],
) -> None:
    if not records or not source_snapshots or not visibility_records:
        raise ValueError("Integrated completeness validation requires nonempty inputs")
    if any(record.run_id != ground_design.run_id for record in records):
        raise ValueError("G3 record run ID does not match G1 ground-design run ID")
    if any(record.run_id != ground_design.run_id for record in visibility_records):
        raise ValueError("G2 record run ID does not match G1 ground-design run ID")
    _validate_unique_keys(records)
    expected = {
        (ground_design.run_id, source.timestep_index) for source in source_snapshots
    }
    actual = {(record.run_id, record.timestep_index) for record in records}
    if actual != expected:
        raise ValueError(
            f"Integrated graph record keys mismatch; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    source_by_timestep = {source.timestep_index: source for source in source_snapshots}
    visibility_by_timestep = {
        record.timestep_index: record for record in visibility_records
    }
    if set(visibility_by_timestep) != set(source_by_timestep):
        raise ValueError("G2 visibility record keys do not match source timesteps")
    for record in records:
        source = source_by_timestep[record.timestep_index]
        visibility = visibility_by_timestep[record.timestep_index]
        expected_values = (
            source.timestamp_utc,
            source.satellite_config_hash,
            ground_design.ground_design_hash,
            visibility.visibility_policy_hash,
            visibility.snapshot_hash,
        )
        actual_values = (
            record.timestamp_utc,
            record.satellite_config_hash,
            record.ground_design_hash,
            record.visibility_policy_hash,
            record.visibility_snapshot_hash,
        )
        if actual_values != expected_values:
            raise ValueError(f"Integrated record identity mismatch at timestep {record.timestep_index}")


def replay_integrated_graph_records(
    *,
    records: Sequence[IntegratedGroundGraphRecord],
    satellite_config: Tier1RolloutConfig,
    failure_realization: Tier1FailureRealization,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    visibility_policy: GroundVisibilityPolicy,
    visibility_records: Sequence[GroundVisibilityRecord],
) -> tuple[IntegratedGroundGraphSnapshot, ...]:
    if ground_design.satellite_config_hash != satellite_config.config_hash():
        raise ValueError("Ground design does not match satellite configuration")
    source_graphs = reconstruct_operational_satellite_graph_sequence(
        satellite_config=satellite_config,
        failure_realization=failure_realization,
    )
    source_positions = reconstruct_operational_satellite_position_sequence(
        satellite_config=satellite_config,
        failure_realization=failure_realization,
    )
    verified_visibility = replay_ground_visibility_records(
        records=visibility_records,
        run_id=ground_design.run_id,
        operational_satellite_snapshots=source_positions,
        ground_design=ground_design,
        catalog=catalog,
        policy=visibility_policy,
    )
    validate_integrated_records_against_sources(
        records=records,
        source_snapshots=source_graphs,
        ground_design=ground_design,
        visibility_records=visibility_records,
    )
    record_by_timestep = {record.timestep_index: record for record in records}
    replayed: list[IntegratedGroundGraphSnapshot] = []
    for source, visibility in zip(source_graphs, verified_visibility, strict=True):
        expected = build_integrated_ground_graph(
            ground_design=ground_design,
            catalog=catalog,
            satellite_graph_snapshot=source,
            verified_visibility_snapshot=visibility,
        )
        record = record_by_timestep[source.timestep_index]
        persisted = record.to_snapshot()
        if (
            persisted.canonical_graph_attributes != expected.canonical_graph_attributes
            or persisted.canonical_nodes != expected.canonical_nodes
            or persisted.canonical_edges != expected.canonical_edges
            or persisted.graph_hash != expected.graph_hash
        ):
            raise ValueError(
                f"Integrated graph replay mismatch at timestep {source.timestep_index}"
            )
        replayed.append(expected)
    return tuple(replayed)
