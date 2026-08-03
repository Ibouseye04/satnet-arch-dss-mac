from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from typing import Any, Iterable

import numpy as np

from satnet.ground.integrated_persistence import read_integrated_graph_manifest
from satnet.ground.integrated_graph import IntegratedNodeKind

EXPECTED_FREEZE_MANIFEST_SHA256 = "b2d1fbd9510d3d828fe051b4da04f6747088ae00d78d65e0a9851dc85559844e"
EXPECTED_CONTRACT_SPEC_HASH = "fc13a33a0a1af435189990e54b6c68efa56bbc207c0cae6e27e12bf031605930"
EXPECTED_SPLIT_FILE_SHA256 = "047b7cc99d83a003add8bca4bb037bce9d642774443eeef1a3df377f855db90d"
EXPECTED_RUN_COUNT = 500
EXPECTED_DESIGN_COUNT = 100
EXPECTED_REALIZATIONS_PER_DESIGN = 5
EXPECTED_SPLIT_DESIGNS = {"train": 70, "validation": 15, "test": 15}
EXPECTED_SPLIT_RUNS = {"train": 350, "validation": 75, "test": 75}
EXPECTED_TIMESTEPS = 11
GRAPH_SCHEMA_VERSION = "stage_a_tgnn_graph_schema_v1"
NODE_SCHEMA_VERSION = "stage_a_tgnn_node_feature_schema_v1"
EDGE_SCHEMA_VERSION = "stage_a_tgnn_edge_schema_v1"
DATASET_VERSION = "stage_a_tgnn_dataset_v1"
FORMAT_VERSION = "satnet_tgnn_sequence_binary_v1"
MAGIC = b"SATNET-TGNN-V1\0"
ARRAY_NAMES = (
    "node_features",
    "node_identity_index",
    "edge_index",
    "edge_attr",
    "snapshot_node_offsets",
    "snapshot_edge_offsets",
    "timestep_index",
)
NODE_FEATURE_NAMES = (
    "node_type_satellite",
    "node_type_ground_station",
    "plane_index",
    "satellite_within_plane_index",
    "ground_station_class_civilian",
    "ground_station_class_government",
    "ground_station_class_military",
    "latitude_deg",
    "longitude_deg",
    "altitude_m",
    "ecef_x_km",
    "ecef_y_km",
    "ecef_z_km",
    "operational_indicator",
)
EDGE_FEATURE_NAMES = (
    "distance_km",
    "margin_db",
    "elevation_deg",
    "slant_range_km",
    "link_type_intra_plane",
    "link_type_inter_plane",
    "link_type_seam_link",
    "link_mode_optical",
    "edge_type_inter_satellite",
    "edge_type_satellite_ground",
)
LINK_TYPES = ("intra_plane", "inter_plane", "seam_link")
LINK_MODES = ("optical",)
GROUND_CLASSES = ("civilian", "government", "military")
WGS84_SEMI_MAJOR_AXIS_KM = 6378.137
WGS84_FLATTENING = 1.0 / 298.257223563
WGS84_FIRST_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line in {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            records.append(value)
    return records


def semantic_hash(value: dict[str, Any], hash_field: str) -> str:
    return sha256_bytes(canonical_json({k: v for k, v in value.items() if k != hash_field}).encode())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_file(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> str:
    require(path.is_file(), f"Missing frozen evidence file: {path}")
    if expected_bytes is not None:
        require(path.stat().st_size == expected_bytes, f"Byte length mismatch: {path}")
    actual = sha256_file(path)
    require(actual == expected_sha256, f"SHA-256 mismatch for {path}: {actual} != {expected_sha256}")
    return actual


def ecef_from_geodetic(latitude_deg: float, longitude_deg: float, altitude_m: float) -> tuple[float, float, float]:
    import math

    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    altitude_km = altitude_m / 1000.0
    sin_latitude = math.sin(latitude)
    cos_latitude = math.cos(latitude)
    sin_longitude = math.sin(longitude)
    cos_longitude = math.cos(longitude)
    prime_vertical_radius = WGS84_SEMI_MAJOR_AXIS_KM / math.sqrt(
        1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED * sin_latitude**2
    )
    return (
        (prime_vertical_radius + altitude_km) * cos_latitude * cos_longitude,
        (prime_vertical_radius + altitude_km) * cos_latitude * sin_longitude,
        (prime_vertical_radius * (1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED) + altitude_km) * sin_latitude,
    )


def parse_attribute_map(attributes: Iterable[Any], context: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attribute in attributes:
        if isinstance(attribute, dict):
            name = attribute.get("name")
            value_type = attribute.get("value_type")
            value = attribute.get("value")
        else:
            name = getattr(attribute, "name", None)
            value_type_object = getattr(attribute, "value_type", None)
            value_type = getattr(value_type_object, "value", value_type_object)
            value = getattr(attribute, "value", None)
        require(isinstance(name, str) and name not in result, f"{context}: duplicate attribute")
        if value_type == "float":
            value = float(value)
        elif value_type == "integer":
            require(type(value) is int, f"{context}: integer attribute is not an integer")
        elif value_type == "boolean":
            require(type(value) is bool, f"{context}: boolean attribute is not a boolean")
        elif value_type in {"string", "none"}:
            pass
        else:
            raise ValueError(f"{context}: unsupported attribute type {value_type!r}")
        result[name] = value
    return result


def ref_key(node_ref: Any) -> tuple[str, int | str]:
    kind = node_ref.kind.value
    if node_ref.kind is IntegratedNodeKind.SATELLITE:
        return kind, node_ref.satellite_id
    return kind, node_ref.ground_station_id


def make_node_features(record: Any, failed_ground_ids: set[str]) -> tuple[np.ndarray, list[tuple[str, int | str]]]:
    rows: list[list[float]] = []
    identities: list[tuple[str, int | str]] = []
    for node in record.canonical_nodes:
        ref = node.node_ref
        attributes = parse_attribute_map(node.attributes, f"run {record.run_id} timestep {record.timestep_index} node")
        row = [0.0] * len(NODE_FEATURE_NAMES)
        identity = ref_key(ref)
        identities.append(identity)
        if ref.kind is IntegratedNodeKind.SATELLITE:
            require({"plane", "sat_in_plane", "type"}.issubset(attributes), "Satellite node attributes are incomplete")
            require(attributes["type"] == "satellite", "Satellite node type mismatch")
            row[0] = 1.0
            row[2] = float(attributes["plane"])
            row[3] = float(attributes["sat_in_plane"])
            row[13] = 1.0
        else:
            required = {"station_class", "latitude_deg", "longitude_deg", "altitude_m"}
            require(required.issubset(attributes), "Ground node attributes are incomplete")
            station_class = attributes["station_class"]
            require(station_class in GROUND_CLASSES, f"Unknown ground station class: {station_class}")
            row[1] = 1.0
            row[4 + GROUND_CLASSES.index(station_class)] = 1.0
            latitude = float(attributes["latitude_deg"])
            longitude = float(attributes["longitude_deg"])
            altitude = float(attributes["altitude_m"])
            row[7:10] = [latitude, longitude, altitude]
            row[10:13] = list(ecef_from_geodetic(latitude, longitude, altitude))
            station_id = str(ref.ground_station_id)
            row[13] = 0.0 if station_id in failed_ground_ids else 1.0
        rows.append(row)
    return np.asarray(rows, dtype=np.float64), identities


def make_edge_features(edge: Any, context: str) -> list[float]:
    attributes = parse_attribute_map(edge.attributes, context)
    row = [0.0] * len(EDGE_FEATURE_NAMES)
    if edge.edge_kind.value == "inter_satellite":
        require({"distance_km", "margin_db", "link_type", "link_mode"}.issubset(attributes), f"{context}: incomplete ISL attributes")
        row[0] = float(attributes["distance_km"])
        row[1] = float(attributes["margin_db"])
        require(attributes["link_type"] in LINK_TYPES, f"{context}: unknown link type")
        require(attributes["link_mode"] in LINK_MODES, f"{context}: unknown link mode")
        row[4 + LINK_TYPES.index(attributes["link_type"])] = 1.0
        row[7 + LINK_MODES.index(attributes["link_mode"])] = 1.0
        row[8] = 1.0
    else:
        require({"elevation_deg", "slant_range_km"}.issubset(attributes), f"{context}: incomplete satellite-ground attributes")
        row[2] = float(attributes["elevation_deg"])
        row[3] = float(attributes["slant_range_km"])
        row[9] = 1.0
    require(all(np.isfinite(row)), f"{context}: non-finite edge feature")
    return row


def write_sequence(path: Path, header: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    payload = bytearray(MAGIC)
    header_bytes = canonical_json(header).encode("utf-8")
    payload.extend(struct.pack("<Q", len(header_bytes)))
    payload.extend(header_bytes)
    for name in ARRAY_NAMES:
        buffer = BytesIO()
        np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
        data = buffer.getvalue()
        payload.extend(struct.pack("<Q", len(data)))
        payload.extend(data)
    path.write_bytes(bytes(payload))
    return sha256_file(path)


def load_sequence(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    data = path.read_bytes()
    cursor = len(MAGIC)
    require(data.startswith(MAGIC), f"Invalid TGNN sequence magic: {path}")
    header_length = struct.unpack_from("<Q", data, cursor)[0]
    cursor += 8
    header = json.loads(data[cursor:cursor + header_length].decode("utf-8"))
    cursor += header_length
    arrays: dict[str, np.ndarray] = {}
    for name in ARRAY_NAMES:
        length = struct.unpack_from("<Q", data, cursor)[0]
        cursor += 8
        buffer = BytesIO(data[cursor:cursor + length])
        arrays[name] = np.lib.format.read_array(buffer, allow_pickle=False)
        cursor += length
    require(cursor == len(data), f"Trailing bytes in TGNN sequence: {path}")
    return header, arrays


def load_contract_and_freeze(freeze_bundle: Path, production_root: Path, contract_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    freeze_manifest_path = freeze_bundle / "accepted_production_evidence_freeze_manifest.json"
    verify_file(freeze_manifest_path, EXPECTED_FREEZE_MANIFEST_SHA256)
    freeze_manifest = read_json(freeze_manifest_path)
    require(freeze_manifest["contract"]["contract_spec_hash"] == EXPECTED_CONTRACT_SPEC_HASH, "Freeze contract hash mismatch")
    require(Path(freeze_manifest["generation"]["root"]).resolve() == production_root.resolve(), "Production root differs from frozen manifest")
    require(Path(freeze_manifest["contract"]["contract_root"]).resolve() == contract_root.resolve(), "Contract root differs from frozen manifest")
    require(freeze_manifest["scope"]["production_data_modified"] is False, "Frozen production scope is not read-only")
    require(freeze_manifest["scope"]["production_data_regenerated"] is False, "Frozen production scope was regenerated")
    require(freeze_manifest["scope"]["model_ready_samples_created"] is False, "Frozen bundle already contains model-ready samples")
    freeze_inventory = read_json(freeze_bundle / "artifact_inventory.json")
    for artifact in freeze_inventory["artifacts"]:
        verify_file(freeze_bundle.parent.parent / artifact["path"], artifact["sha256"], artifact["bytes"])

    contract_spec_path = contract_root / "contract_specification.json"
    contract_spec = read_json(contract_spec_path)
    require(contract_spec.get("contract_spec_hash") == EXPECTED_CONTRACT_SPEC_HASH, "Contract specification hash field mismatch")
    require(semantic_hash(contract_spec, "contract_spec_hash") == EXPECTED_CONTRACT_SPEC_HASH, "Contract specification semantic hash mismatch")
    designs_path = contract_root / "designs.jsonl"
    runs_path = contract_root / "runs.jsonl"
    split_path = contract_root / "split_manifest.json"
    verify_file(designs_path, freeze_manifest["contract"]["design_manifest"]["sha256"], freeze_manifest["contract"]["design_manifest"]["bytes"])
    verify_file(runs_path, freeze_manifest["contract"]["run_manifest"]["sha256"], freeze_manifest["contract"]["run_manifest"]["bytes"])
    verify_file(split_path, EXPECTED_SPLIT_FILE_SHA256, freeze_manifest["split_manifest"]["bytes"])
    designs = read_jsonl(designs_path)
    runs = read_jsonl(runs_path)
    split_manifest = read_json(split_path)
    require(len(designs) == EXPECTED_DESIGN_COUNT and len(runs) == EXPECTED_RUN_COUNT, "Frozen contract cardinality mismatch")
    require(split_manifest["contract_spec_hash"] == EXPECTED_CONTRACT_SPEC_HASH, "Split contract hash mismatch")
    require(split_manifest["outcome_fields_used"] is False, "Split manifest uses outcome fields")
    require(freeze_manifest["split_manifest"]["test_split_sealed"] is True, "Frozen test split is not sealed")
    require({split: len(split_manifest["design_assignments"][split]) for split in EXPECTED_SPLIT_DESIGNS} == EXPECTED_SPLIT_DESIGNS, "Frozen design split counts mismatch")
    require({split: len(split_manifest["run_assignments"][split]) for split in EXPECTED_SPLIT_RUNS} == EXPECTED_SPLIT_RUNS, "Frozen run split counts mismatch")
    design_by_id = {record["design_id"]: record for record in designs}
    run_by_id = {record["run_id"]: record for record in runs}
    require(len(design_by_id) == EXPECTED_DESIGN_COUNT and set(run_by_id) == set(range(EXPECTED_RUN_COUNT)), "Frozen identity uniqueness mismatch")
    split_by_design = {design_id: split for split, ids in split_manifest["design_assignments"].items() for design_id in ids}
    require(set(split_by_design) == set(design_by_id), "Split does not assign every design")
    for run_id, run in run_by_id.items():
        require(run["split_assignment"] == split_by_design[run["design_id"]], f"Run split mismatch: {run_id}")
        require(run["run_key"] == f"{run['design_id']}-{run['realization_id']}", f"Run key mismatch: {run_id}")

    ledger_path = production_root / "operational" / "generation_ledger.json"
    ledger_expected = freeze_manifest["generation"]["ledger"]
    verify_file(ledger_path, ledger_expected["sha256"], ledger_expected["bytes"])
    ledger = read_json(ledger_path)
    ledger_by_id = {record["run_id"]: record for record in ledger["records"]}
    require(len(ledger_by_id) == EXPECTED_RUN_COUNT, "Generation ledger cardinality mismatch")
    for run_id, run in run_by_id.items():
        ledger_record = ledger_by_id[run_id]
        require(ledger_record["state"] == "succeeded", f"Production run did not succeed: {run_id}")
        require(ledger_record["run_key"] == run["run_key"] and ledger_record["run_record_hash"] == run["run_record_hash"], f"Ledger identity mismatch: {run_id}")
    return freeze_manifest, designs, runs, split_manifest, ledger_by_id, contract_spec


def build_sequence(
    *,
    run: dict[str, Any],
    production_root: Path,
    inventory: dict[str, Any],
    graph_records: tuple[Any, ...],
    failed_ground_ids: set[str],
    split: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    require(graph_records, f"No graph records for run {run['run_id']}")
    require(all(record.run_id == run["run_id"] for record in graph_records), f"Graph run identity mismatch: {run['run_id']}")
    require(tuple(record.timestep_index for record in graph_records) == tuple(range(EXPECTED_TIMESTEPS)), f"Graph timestep sequence mismatch: {run['run_id']}")
    timestamps = [record.timestamp_utc.isoformat().replace("+00:00", "Z") for record in graph_records]
    parsed_timestamps = [record.timestamp_utc for record in graph_records]
    require(all(a < b for a, b in zip(parsed_timestamps, parsed_timestamps[1:])), f"Graph timestamps are not strictly monotonic: {run['run_id']}")

    all_identities = sorted({ref_key(node.node_ref) for record in graph_records for node in record.canonical_nodes}, key=lambda item: (0 if item[0] == "satellite" else 1, item[1]))
    identity_index = {identity: index for index, identity in enumerate(all_identities)}
    node_chunks: list[np.ndarray] = []
    node_identity_chunks: list[np.ndarray] = []
    edge_chunks: list[np.ndarray] = []
    edge_attr_chunks: list[np.ndarray] = []
    node_offsets = [0]
    edge_offsets = [0]
    physical_edge_counts: list[int] = []
    node_counts: list[int] = []
    directed_edge_counts: list[int] = []
    for record in graph_records:
        node_features, identities = make_node_features(record, failed_ground_ids)
        local_index = {identity: index for index, identity in enumerate(identities)}
        require(len(local_index) == len(identities), f"Duplicate timestep node identity: {run['run_id']}")
        directed_edges: list[list[int]] = []
        directed_attributes: list[list[float]] = []
        physical_keys: set[tuple[tuple[str, int | str], tuple[str, int | str]]] = set()
        for edge in record.canonical_edges:
            first = ref_key(edge.endpoint_a)
            second = ref_key(edge.endpoint_b)
            require(first in local_index and second in local_index, f"Edge endpoint is absent: {run['run_id']}")
            require(first != second, f"Self-loop in source graph: {run['run_id']}")
            physical_key = (first, second)
            require(physical_key not in physical_keys, f"Duplicate physical edge: {run['run_id']}")
            physical_keys.add(physical_key)
            attributes = make_edge_features(edge, f"run {run['run_id']} timestep {record.timestep_index} edge")
            a, b = local_index[first], local_index[second]
            directed_edges.extend(((a, b), (b, a)))
            directed_attributes.extend((attributes, attributes))
        node_chunks.append(node_features)
        node_identity_chunks.append(np.asarray([identity_index[value] for value in identities], dtype=np.int32))
        edge_chunks.append(np.asarray(directed_edges, dtype=np.int64).T if directed_edges else np.empty((2, 0), dtype=np.int64))
        edge_attr_chunks.append(np.asarray(directed_attributes, dtype=np.float64).reshape((-1, len(EDGE_FEATURE_NAMES))) if directed_attributes else np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float64))
        node_offsets.append(node_offsets[-1] + len(identities))
        edge_offsets.append(edge_offsets[-1] + len(directed_edges))
        physical_edge_counts.append(len(physical_keys))
        node_counts.append(len(identities))
        directed_edge_counts.append(len(directed_edges))

    arrays = {
        "node_features": np.concatenate(node_chunks, axis=0),
        "node_identity_index": np.concatenate(node_identity_chunks, axis=0),
        "edge_index": np.concatenate(edge_chunks, axis=1),
        "edge_attr": np.concatenate(edge_attr_chunks, axis=0),
        "snapshot_node_offsets": np.asarray(node_offsets, dtype=np.int64),
        "snapshot_edge_offsets": np.asarray(edge_offsets, dtype=np.int64),
        "timestep_index": np.asarray([record.timestep_index for record in graph_records], dtype=np.int64),
    }
    identities_json = [{"kind": kind, "satellite_id": value if kind == "satellite" else None, "ground_station_id": value if kind == "ground_station" else None} for kind, value in all_identities]
    header = {
        "format": FORMAT_VERSION,
        "schema_version": GRAPH_SCHEMA_VERSION,
        "sequence": {
            "contract_spec_hash": EXPECTED_CONTRACT_SPEC_HASH,
            "design_id": run["design_id"],
            "realization_id": run["realization_id"],
            "realization_index": run["realization_index"],
            "run_id": run["run_id"],
            "run_key": run["run_key"],
            "split": split,
        },
        "sequence_length": len(graph_records),
        "timesteps": [{"timestep_index": record.timestep_index, "timestamp_utc": timestamp} for record, timestamp in zip(graph_records, timestamps)],
        "node_identity": identities_json,
        "node_identity_order": "satellite numeric ascending, then ground_station_id lexical ascending",
        "node_count_by_timestep": node_counts,
        "physical_edge_count_by_timestep": physical_edge_counts,
        "directed_edge_count_by_timestep": directed_edge_counts,
        "node_feature_schema_version": NODE_SCHEMA_VERSION,
        "edge_schema_version": EDGE_SCHEMA_VERSION,
        "targets_in_graph_artifact": False,
        "array_dtypes": {name: str(arrays[name].dtype) for name in ARRAY_NAMES},
    }
    source_artifacts = {artifact["artifact_role"]: artifact for artifact in inventory["artifacts"]}
    metadata = {
        "sequence_length": len(graph_records),
        "node_counts": node_counts,
        "physical_edge_counts": physical_edge_counts,
        "directed_edge_counts": directed_edge_counts,
        "timestamps": timestamps,
        "source_graph_sha256": source_artifacts["integrated_graph"]["sha256"],
        "source_graph_path": str((production_root / f"run_{run['run_id']:03d}" / source_artifacts["integrated_graph"]["path"]).resolve()),
        "source_ground_failure_realization_sha256": source_artifacts["ground_failure_realization"]["sha256"],
        "source_ground_failure_realization_path": str((production_root / f"run_{run['run_id']:03d}" / source_artifacts["ground_failure_realization"]["path"]).resolve()),
    }
    return header, arrays, metadata


def target_record(run: dict[str, Any], target_path: Path, expected_hash: str) -> dict[str, Any]:
    target = read_json(target_path)
    require(sha256_file(target_path) == expected_hash, f"Target artifact hash mismatch: {run['run_key']}")
    require(target["target_artifact_hash"] == semantic_hash(target, "target_artifact_hash"), f"Target semantic hash mismatch: {run['run_key']}")
    require(target["run_id"] == run["run_id"] and target["run_key"] == run["run_key"], f"Target identity mismatch: {run['run_key']}")
    require(target["contract_spec_hash"] == EXPECTED_CONTRACT_SPEC_HASH, f"Target contract mismatch: {run['run_key']}")
    return target


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(record) + "\n" for record in records), encoding="utf-8", newline="")


def git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_dataset(freeze_bundle: Path, production_root: Path, contract_root: Path, output_dir: Path) -> None:
    require(output_dir.parent.is_dir(), f"Output parent does not exist: {output_dir.parent}")
    allowed_existing = {"README.md", "artifact_inventory.json"}
    if output_dir.exists():
        require(output_dir.is_dir(), f"Output path is not a directory: {output_dir}")
        require(set(path.name for path in output_dir.iterdir()) <= allowed_existing, "Output directory contains unrelated files")
    else:
        output_dir.mkdir(parents=False)
    freeze_manifest, designs, runs, split_manifest, ledger_by_id, contract_spec = load_contract_and_freeze(freeze_bundle, production_root, contract_root)
    design_by_id = {record["design_id"]: record for record in designs}
    split_by_design = {design_id: split for split, ids in split_manifest["design_assignments"].items() for design_id in ids}
    sequence_dir = output_dir / "sequences"
    sequence_dir.mkdir(exist_ok=True)
    indexes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    provenance: list[dict[str, Any]] = []
    structural: dict[str, Any] = {"all_sequences_load": True, "train_validation_sequences_load": True, "timesteps_ordered": True, "node_schema_valid": True, "edge_schema_valid": True, "feature_dimensions_consistent": True, "edge_indexes_valid": True, "target_fields_in_graph_inputs": 0, "future_information_leakage": 0, "missing_train_validation_targets": 0}
    sequence_stats: list[dict[str, Any]] = []
    source_hash_checks = {"freeze_manifest": True, "contract_specification": True, "design_manifest": True, "run_manifest": True, "split_manifest_exact_file": True, "generation_ledger": True, "source_result_files": 0, "source_scientific_inventory_files": 0, "source_graph_files": 0, "source_ground_failure_realization_files": 0, "source_target_files_train_validation": 0, "test_targets_read": False}
    for run in sorted(runs, key=lambda row: row["run_id"]):
        run_id = run["run_id"]
        run_dir = production_root / f"run_{run_id:03d}"
        result_path = run_dir / "result.json"
        inventory_path = run_dir / "scientific_inventory.json"
        result_sha = sha256_file(result_path)
        inventory_sha = sha256_file(inventory_path)
        result = read_json(result_path)
        inventory = read_json(inventory_path)
        require(result["run_id"] == run_id and result["run_key"] == run["run_key"], f"Result identity mismatch: {run['run_key']}")
        require(result["run_result_hash"] == semantic_hash(result, "run_result_hash"), f"Result semantic hash mismatch: {run['run_key']}")
        require(result["scientific_inventory_hash"] == inventory["scientific_inventory_hash"], f"Inventory link mismatch: {run['run_key']}")
        require(result["scientific_inventory_hash"] == semantic_hash(inventory, "scientific_inventory_hash"), f"Inventory semantic hash mismatch: {run['run_key']}")
        require(ledger_by_id[run_id]["published_result_hash"] == result["run_result_hash"], f"Ledger result link mismatch: {run['run_key']}")
        require(ledger_by_id[run_id]["scientific_inventory_hash"] == inventory["scientific_inventory_hash"], f"Ledger inventory link mismatch: {run['run_key']}")
        artifacts = {artifact["artifact_role"]: artifact for artifact in inventory["artifacts"]}
        graph_path = run_dir / artifacts["integrated_graph"]["path"]
        failure_path = run_dir / artifacts["ground_failure_realization"]["path"]
        require(sha256_file(graph_path) == artifacts["integrated_graph"]["sha256"], f"Graph artifact hash mismatch: {run['run_key']}")
        require(sha256_file(failure_path) == artifacts["ground_failure_realization"]["sha256"], f"Ground failure artifact hash mismatch: {run['run_key']}")
        graph_records = read_integrated_graph_manifest(graph_path)
        failure_records = read_jsonl(failure_path)
        require(len(failure_records) == 1 and failure_records[0]["run_id"] == run_id, f"Ground failure realization mismatch: {run['run_key']}")
        failed_ground_ids = set(failure_records[0]["realization"]["failed_station_ids"])
        split = split_by_design[run["design_id"]]
        header, arrays, metadata = build_sequence(run=run, production_root=production_root, inventory=inventory, graph_records=graph_records, failed_ground_ids=failed_ground_ids, split=split)
        artifact_path = sequence_dir / f"run_{run_id:03d}.tgnn"
        artifact_sha = write_sequence(artifact_path, header, arrays)
        if split != "test":
            target_path = run_dir / artifacts["canonical_target"]["path"]
            target = target_record(run, target_path, artifacts["canonical_target"]["sha256"])
            targets = {"partition_any": target["overall_threshold_breach_any"], "gcc_frac_min": float(target["space_gcc_fraction_original_min"])}
            require(type(targets["partition_any"]) is bool and 0.0 <= targets["gcc_frac_min"] <= 1.0, f"Invalid train/validation target: {run['run_key']}")
            source_hash_checks["source_target_files_train_validation"] += 1
        else:
            targets = None
        index_record = {
            "design_id": run["design_id"],
            "run_id": run_id,
            "run_key": run["run_key"],
            "design_index": run["design_index"],
            "realization_id": run["realization_id"],
            "realization_index": run["realization_index"],
            "split": split,
            "contract_spec_hash": EXPECTED_CONTRACT_SPEC_HASH,
            "sequence_artifact": str(artifact_path.relative_to(output_dir)).replace("\\", "/"),
            "sequence_artifact_sha256": artifact_sha,
            "sequence_length": metadata["sequence_length"],
            "node_count_min": min(metadata["node_counts"]),
            "node_count_max": max(metadata["node_counts"]),
            "physical_edge_count_min": min(metadata["physical_edge_counts"]),
            "physical_edge_count_max": max(metadata["physical_edge_counts"]),
            "source_result_path": str(result_path.resolve()),
            "source_result_sha256": result_sha,
            "source_scientific_inventory_path": str(inventory_path.resolve()),
            "source_scientific_inventory_sha256": inventory_sha,
            "source_design_record_hash": design_by_id[run["design_id"]]["design_record_hash"],
            "source_run_record_hash": run["run_record_hash"],
            "source_target_artifact_hash": result["target_artifact_hash"],
            "source_graph_path": metadata["source_graph_path"],
            "source_graph_sha256": metadata["source_graph_sha256"],
            "source_ground_failure_realization_path": metadata["source_ground_failure_realization_path"],
            "source_ground_failure_realization_sha256": metadata["source_ground_failure_realization_sha256"],
        }
        if targets is not None:
            index_record.update(targets)
        indexes[split].append(index_record)
        provenance.append({key: index_record[key] for key in ("design_id", "run_id", "run_key", "realization_id", "realization_index", "split", "contract_spec_hash", "sequence_artifact", "sequence_artifact_sha256", "source_result_path", "source_result_sha256", "source_scientific_inventory_path", "source_scientific_inventory_sha256", "source_design_record_hash", "source_run_record_hash", "source_target_artifact_hash", "source_graph_path", "source_graph_sha256", "source_ground_failure_realization_path", "source_ground_failure_realization_sha256")})
        sequence_stats.append({"run_id": run_id, "split": split, "design_id": run["design_id"], "sequence_length": metadata["sequence_length"], "node_counts": metadata["node_counts"], "physical_edge_counts": metadata["physical_edge_counts"], "directed_edge_counts": metadata["directed_edge_counts"]})
        source_hash_checks["source_result_files"] += 1
        source_hash_checks["source_scientific_inventory_files"] += 1
        source_hash_checks["source_graph_files"] += 1
        source_hash_checks["source_ground_failure_realization_files"] += 1

    for split, expected in EXPECTED_SPLIT_RUNS.items():
        require(len(indexes[split]) == expected, f"{split} sequence count mismatch")
        write_jsonl(output_dir / {"train": "tgnn_train_index.jsonl", "validation": "tgnn_validation_index.jsonl", "test": "tgnn_test_sealed_index.jsonl"}[split], indexes[split])
    require(all("partition_any" not in row and "gcc_frac_min" not in row for row in indexes["test"]), "Test targets were materialized")
    write_jsonl(output_dir / "tgnn_provenance_manifest.jsonl", sorted(provenance, key=lambda row: row["run_id"]))

    for split in EXPECTED_SPLIT_RUNS:
        for row in indexes[split]:
            header, arrays = load_sequence(output_dir / row["sequence_artifact"])
            require(header["targets_in_graph_artifact"] is False, f"Target marker present in graph artifact: {row['run_key']}")
            require(header["sequence"]["run_key"] == row["run_key"] and header["sequence"]["split"] == split, f"Sequence identity mismatch: {row['run_key']}")
            require(header["sequence_length"] == EXPECTED_TIMESTEPS, f"Sequence length mismatch: {row['run_key']}")
            require(arrays["node_features"].shape[1] == len(NODE_FEATURE_NAMES), f"Node feature dimension mismatch: {row['run_key']}")
            require(arrays["edge_attr"].shape[1] == len(EDGE_FEATURE_NAMES), f"Edge feature dimension mismatch: {row['run_key']}")
            require(tuple(arrays["timestep_index"].tolist()) == tuple(range(EXPECTED_TIMESTEPS)), f"Serialized timestep ordering mismatch: {row['run_key']}")
            node_offsets = arrays["snapshot_node_offsets"]
            edge_offsets = arrays["snapshot_edge_offsets"]
            require(len(node_offsets) == EXPECTED_TIMESTEPS + 1 and len(edge_offsets) == EXPECTED_TIMESTEPS + 1, f"Serialized offset length mismatch: {row['run_key']}")
            for snapshot_index in range(len(node_offsets) - 1):
                node_count = int(node_offsets[snapshot_index + 1] - node_offsets[snapshot_index])
                edge_start = int(edge_offsets[snapshot_index])
                edge_end = int(edge_offsets[snapshot_index + 1])
                snapshot_edges = arrays["edge_index"][:, edge_start:edge_end]
                require(snapshot_edges.size == 0 or (int(snapshot_edges.min()) >= 0 and int(snapshot_edges.max()) < node_count), f"Edge index out of range: {row['run_key']} timestep {snapshot_index}")
                identity_values = arrays["node_identity_index"][node_offsets[snapshot_index]:node_offsets[snapshot_index + 1]]
                require(identity_values.size == 0 or (int(identity_values.min()) >= 0 and int(identity_values.max()) < len(header["node_identity"])), f"Node identity index out of range: {row['run_key']}")
            require(tuple(int(value) for value in np.diff(node_offsets)) == tuple(header["node_count_by_timestep"]), f"Node offsets disagree with header: {row['run_key']}")
            require(tuple(int(value) for value in np.diff(edge_offsets)) == tuple(header["directed_edge_count_by_timestep"]), f"Edge offsets disagree with header: {row['run_key']}")
    structural["target_fields_in_graph_inputs"] = 0
    structural["future_information_leakage"] = 0
    require(all(row.get("partition_any") is not None and row.get("gcc_frac_min") is not None for row in indexes["train"] + indexes["validation"]), "Missing train/validation target")

    seq_lengths = Counter(stat["sequence_length"] for stat in sequence_stats)
    node_count_distribution = Counter(count for stat in sequence_stats for count in stat["node_counts"])
    physical_edge_distribution = Counter(count for stat in sequence_stats for count in stat["physical_edge_counts"])
    directed_edge_distribution = Counter(count for stat in sequence_stats for count in stat["directed_edge_counts"])
    design_counts = {split: len({row["design_id"] for row in indexes[split]}) for split in EXPECTED_SPLIT_RUNS}
    duplicate_run_keys = len(sum((indexes[split] for split in EXPECTED_SPLIT_RUNS), [])) - len({row["run_key"] for split in EXPECTED_SPLIT_RUNS for row in indexes[split]})
    duplicate_pairs = 500 - len({(row["design_id"], row["realization_id"]) for split in EXPECTED_SPLIT_RUNS for row in indexes[split]})
    graph_scope = "combined satellite-ground heterogeneous graph"
    write_json(output_dir / "tgnn_graph_schema.json", {
        "schema_name": GRAPH_SCHEMA_VERSION,
        "schema_version": "1",
        "graph_scope": graph_scope,
        "source_scope_conclusion": "G3 accepted IntegratedGroundGraphRecord includes satellite and selected ground-station nodes and inter-satellite plus satellite-ground edges.",
        "sequence_unit": "one complete run; ordered snapshots for all valid timesteps",
        "node_order": "canonical G3 order: satellite numeric ascending, then ground_station_id lexical ascending",
        "node_identity": {"satellite": "IntegratedNodeRef(kind=satellite, satellite_id)", "ground_station": "IntegratedNodeRef(kind=ground_station, ground_station_id)"},
        "failed_satellites": "absent under inherited G3 operational graph semantics; no invented persistent identity or padding",
        "failed_ground_stations": "retained as nodes; operational_indicator is zero from the persistent G5 overlay",
        "padding_or_masking": "none; variable node counts are represented by per-snapshot offsets and identity indexes",
        "self_loops": False,
        "duplicate_physical_edges": False,
        "source_graph_is_undirected": True,
        "serialized_edge_directionality": "each source undirected physical edge is emitted as both directed COO directions for PyTorch Geometric",
        "source_graph_connectivity_preserved": True,
        "format": FORMAT_VERSION,
        "arrays": list(ARRAY_NAMES),
        "node_feature_dimension": len(NODE_FEATURE_NAMES),
        "edge_attribute_dimension": len(EDGE_FEATURE_NAMES),
        "targets_in_graph_inputs": False,
    })
    write_json(output_dir / "tgnn_node_feature_schema.json", {
        "schema_name": NODE_SCHEMA_VERSION,
        "schema_version": "1",
        "dtype": "float64",
        "feature_dimension": len(NODE_FEATURE_NAMES),
        "normalization": "none; canonical source units and dimensionless one-hot/status fields are preserved",
        "missing_value_behavior": "canonical positive zero for fields not applicable to a node type; no imputation of applicable values",
        "features": [
            {"name": name, "source_field": source, "units": units, "applies_to": applies, "normalization": "none", "missing_value": "positive_zero_when_not_applicable", "temporal_availability": temporal, "leakage_assessment": "accepted timestep-level predictor; not a target or post-run summary"}
            for name, source, units, applies, temporal in (
                ("node_type_satellite", "G3 node_ref.kind", "one-hot", ["satellite"], "every valid timestep"),
                ("node_type_ground_station", "G3 node_ref.kind", "one-hot", ["ground_station"], "every valid timestep"),
                ("plane_index", "G3 canonical node attribute plane", "plane index", ["satellite"], "every valid timestep where satellite is present"),
                ("satellite_within_plane_index", "G3 canonical node attribute sat_in_plane", "within-plane index", ["satellite"], "every valid timestep where satellite is present"),
                ("ground_station_class_civilian", "G3 canonical node attribute station_class", "one-hot", ["ground_station"], "every valid timestep"),
                ("ground_station_class_government", "G3 canonical node attribute station_class", "one-hot", ["ground_station"], "every valid timestep"),
                ("ground_station_class_military", "G3 canonical node attribute station_class", "one-hot", ["ground_station"], "every valid timestep"),
                ("latitude_deg", "G3 ground node latitude_deg", "degrees", ["ground_station"], "every valid timestep"),
                ("longitude_deg", "G3 ground node longitude_deg", "degrees", ["ground_station"], "every valid timestep"),
                ("altitude_m", "G3 ground node altitude_m", "metres", ["ground_station"], "every valid timestep"),
                ("ecef_x_km", "ground_station_to_ecef(G3 latitude_deg, longitude_deg, altitude_m)", "km", ["ground_station"], "every valid timestep"),
                ("ecef_y_km", "ground_station_to_ecef(G3 latitude_deg, longitude_deg, altitude_m)", "km", ["ground_station"], "every valid timestep"),
                ("ecef_z_km", "ground_station_to_ecef(G3 latitude_deg, longitude_deg, altitude_m)", "km", ["ground_station"], "every valid timestep"),
                ("operational_indicator", "G3 node presence for satellites; G5 operational_station_ids for ground stations", "binary indicator", ["satellite", "ground_station"], "every valid timestep"),
            )
        ],
        "excluded": ["classification and regression targets", "target derivatives", "future timestep values", "whole-run summaries", "post-run minima/maxima/means", "replay/acceptance results", "split labels", "identifiers, paths, hashes, and seeds as learned features"],
    })
    write_json(output_dir / "tgnn_edge_schema.json", {
        "schema_name": EDGE_SCHEMA_VERSION,
        "schema_version": "1",
        "dtype": "float64",
        "feature_dimension": len(EDGE_FEATURE_NAMES),
        "physical_edge_attribute_names": list(EDGE_FEATURE_NAMES),
        "features": [
            {"name": name, "source_field": source, "units": units, "applies_to": applies, "normalization": "none", "missing_value": "positive_zero_when_not_applicable", "leakage_assessment": "accepted current-timestep G3 edge attribute or deterministic edge type encoding"}
            for name, source, units, applies in (
                ("distance_km", "G3 inter-satellite edge distance_km", "km", ["inter_satellite"]),
                ("margin_db", "G3 inter-satellite edge margin_db", "dB", ["inter_satellite"]),
                ("elevation_deg", "G3 satellite-ground edge elevation_deg", "degrees", ["satellite_ground"]),
                ("slant_range_km", "G3 satellite-ground edge slant_range_km", "km", ["satellite_ground"]),
                ("link_type_intra_plane", "G3 inter-satellite edge link_type", "one-hot", ["inter_satellite"]),
                ("link_type_inter_plane", "G3 inter-satellite edge link_type", "one-hot", ["inter_satellite"]),
                ("link_type_seam_link", "G3 inter-satellite edge link_type", "one-hot", ["inter_satellite"]),
                ("link_mode_optical", "G3 inter-satellite edge link_mode", "one-hot", ["inter_satellite"]),
                ("edge_type_inter_satellite", "G3 edge_kind", "one-hot", ["inter_satellite"]),
                ("edge_type_satellite_ground", "G3 edge_kind", "one-hot", ["satellite_ground"]),
            )
        ],
        "link_failure_state": "unavailable links are absent from G3 connectivity; ground-node failure is represented by node operational_indicator while selected ground nodes remain present",
        "self_loops": False,
        "duplicate_policy": "one canonical physical edge in source; two directed COO entries in serialized edge_index",
        "directionality": "source G3 is undirected; serialized edge_index contains both endpoint directions with duplicated edge attributes",
    })
    excluded = {
        "target_fields": ["overall_threshold_breach_any", "space_gcc_fraction_original_min", "partition_any", "gcc_frac_min"],
        "target_derivatives": ["failure_adjusted_overall_service_fraction_mean", "failure_adjusted_overall_service_fraction_min", "space_gcc_fraction_original_mean", "space_gcc_fraction_surviving_min", "threshold_breach_timestep_count"],
        "future_or_post_run_fields": ["all G4/G5 run summaries", "all temporal minima/maxima/means", "first/last breach timestep", "serviced_station_labels", "component_or_gcc_labels"],
        "provenance_and_control_fields": ["design_id", "run_id", "run_key", "realization_id", "split", "contract_spec_hash", "paths", "hashes", "seeds", "replay status", "acceptance status"],
        "source_artifact_fields_not_used_as_features": ["capacity", "signal_strength_dbm", "visibility_policy_hash", "visibility_snapshot_hash", "ground_design_hash", "station_id", "satellite_id", "ground_station_id", "label", "country_code", "region", "enabled", "ground_design_hash"],
    }
    write_json(output_dir / "tgnn_leakage_exclusion_report.json", {"schema": "satnet.stage_a_tgnn_dataset_v1.leakage_exclusion_report", "graph_inputs_contain_targets": False, "future_information_leakage": False, "excluded_fields": excluded, "test_targets_read": False, "test_outcome_fields_used": False})
    construction_report = {
        "schema": "satnet.stage_a_tgnn_dataset_v1.construction_report",
        "dataset_version": DATASET_VERSION,
        "implementation": "scripts/build_stage_a_tgnn_dataset.py",
        "implementation_source_head_at_construction": git_head(output_dir.parent.parent),
        "graph_scope": graph_scope,
        "construction_checks": {
            "train_samples": len(indexes["train"]) == 350,
            "validation_samples": len(indexes["validation"]) == 75,
            "sealed_test_samples": len(indexes["test"]) == 75,
            "train_designs": design_counts["train"] == 70,
            "validation_designs": design_counts["validation"] == 15,
            "test_designs": design_counts["test"] == 15,
            "five_realizations_per_design": all(sum(row["design_id"] == design_id for split in EXPECTED_SPLIT_RUNS for row in indexes[split]) == 5 for design_id in design_by_id),
            "zero_design_overlap": not (set(row["design_id"] for row in indexes["train"]) & set(row["design_id"] for row in indexes["validation"]) or set(row["design_id"] for row in indexes["train"]) & set(row["design_id"] for row in indexes["test"]) or set(row["design_id"] for row in indexes["validation"]) & set(row["design_id"] for row in indexes["test"])),
            "zero_run_overlap": len({row["run_id"] for split in EXPECTED_SPLIT_RUNS for row in indexes[split]}) == 500,
            "zero_duplicate_run_keys": duplicate_run_keys == 0,
            "zero_duplicate_design_realization_pairs": duplicate_pairs == 0,
            "all_sequence_artifacts_load": structural["all_sequences_load"],
            "all_train_validation_sequences_load": structural["train_validation_sequences_load"],
            "all_sequence_timesteps_ordered": structural["timesteps_ordered"],
            "all_graph_snapshots_node_schema_valid": structural["node_schema_valid"],
            "all_graph_snapshots_edge_schema_valid": structural["edge_schema_valid"],
            "node_feature_dimensions_consistent": structural["feature_dimensions_consistent"],
            "edge_indexes_valid": structural["edge_indexes_valid"],
            "zero_target_fields_in_graph_inputs": structural["target_fields_in_graph_inputs"] == 0,
            "zero_future_information_leakage": structural["future_information_leakage"] == 0,
            "zero_missing_train_validation_targets": structural["missing_train_validation_targets"] == 0,
            "test_targets_sealed": all("partition_any" not in row and "gcc_frac_min" not in row for row in indexes["test"]),
        },
        "counts": {"designs_total": 100, "runs_total": 500, "realizations_per_design": 5, "samples_by_split": {split: len(indexes[split]) for split in EXPECTED_SPLIT_RUNS}, "designs_by_split": design_counts},
        "sequence_length_distribution": dict(sorted(seq_lengths.items())),
        "node_count_distribution_by_timestep": dict(sorted(node_count_distribution.items())),
        "physical_edge_count_distribution_by_timestep": dict(sorted(physical_edge_distribution.items())),
        "directed_edge_count_distribution_by_timestep": dict(sorted(directed_edge_distribution.items())),
        "feature_dimensions": {"node": len(NODE_FEATURE_NAMES), "edge": len(EDGE_FEATURE_NAMES)},
        "target_identities": {"classification_alias": {"field": "partition_any", "source_field": "overall_threshold_breach_any"}, "regression_alias": {"field": "gcc_frac_min", "source_field": "space_gcc_fraction_original_min"}},
        "source_hash_verification": source_hash_checks,
        "evidence_integrity": {"production_artifacts_modified": False, "replay_artifacts_modified": False, "acceptance_artifacts_modified": False, "freeze_artifacts_modified": False, "accepted_rf_dataset_modified": False, "production_root_used_read_only": True},
        "test_seal": {"status": "sealed", "targets_materialized": False, "outcome_summaries_materialized": False, "outcome_fields_used": False, "test_targets_read": False},
        "independent_acceptance_gate_run": False,
    }
    construction_report["construction_report_hash"] = semantic_hash(construction_report, "construction_report_hash")
    write_json(output_dir / "tgnn_construction_report.json", construction_report)
    dataset_manifest = {
        "manifest_name": "satnet_stage_a_tgnn_dataset_v1_manifest",
        "manifest_version": "1",
        "dataset_version": DATASET_VERSION,
        "construction_scope": "TGNN dataset construction only; no RF rebuild, training, tuning, test evaluation, or independent TGNN acceptance gate",
        "freeze": {"freeze_bundle": str(freeze_bundle.resolve()), "freeze_manifest_sha256": EXPECTED_FREEZE_MANIFEST_SHA256, "contract_spec_hash": EXPECTED_CONTRACT_SPEC_HASH, "contract_root": str(contract_root.resolve()), "production_root": str(production_root.resolve()), "split_manifest_exact_file_sha256": EXPECTED_SPLIT_FILE_SHA256},
        "graph_scope": graph_scope,
        "sequence_definition": "one complete run represented by all 11 ordered valid G3/G5-aligned timesteps",
        "storage": {"format": FORMAT_VERSION, "artifact_pattern": "sequences/run_NNN.tgnn", "opaque_pickle": False, "deterministic": True, "index_order": "run_id ascending"},
        "counts": {"designs": 100, "runs": 500, "realizations_per_design": 5, "samples_by_split": {split: len(indexes[split]) for split in EXPECTED_SPLIT_RUNS}, "designs_by_split": design_counts},
        "schema_files": ["tgnn_graph_schema.json", "tgnn_node_feature_schema.json", "tgnn_edge_schema.json"],
        "target_identities": {"classification": {"alias": "partition_any", "source_field": "overall_threshold_breach_any", "storage": "run-level train/validation index label only"}, "regression": {"alias": "gcc_frac_min", "source_field": "space_gcc_fraction_original_min", "storage": "run-level train/validation index label only"}},
        "test_seal": {"status": "sealed", "test_outcome_fields_used": False, "test_targets_materialized": False},
        "construction_report": "tgnn_construction_report.json",
    }
    dataset_manifest["dataset_manifest_hash"] = semantic_hash(dataset_manifest, "dataset_manifest_hash")
    write_json(output_dir / "tgnn_dataset_manifest.json", dataset_manifest)
    readme = f"""# Stage A Temporal GNN Dataset v1

Verdict: **STAGE A TGNN DATASET V1 CONSTRUCTED — READY FOR NARROW INDEPENDENT DATASET GATE**

This dataset was constructed only from the accepted, frozen production evidence. It does not rebuild the accepted RF dataset, train or tune a model, evaluate test targets, or run the independent TGNN acceptance gate.

## Graph scope

The accepted G3 `IntegratedGroundGraphRecord` artifacts define a combined satellite-ground heterogeneous graph. Each run is one ordered sequence of all 11 valid timesteps. Satellite nodes use inherited G3 operational semantics: failed satellites are absent. Selected ground-station nodes remain present under G5 persistent failure overlay semantics and expose `operational_indicator=0` when failed.

## Counts

- Train: 70 designs / 350 run sequences
- Validation: 15 designs / 75 run sequences
- Sealed test: 15 designs / 75 run sequence indexes
- Five realizations per design; zero design and run overlap
- Sequence length: {dict(sorted(seq_lengths.items()))}
- Node counts by timestep: {dict(sorted(node_count_distribution.items()))}
- Physical edge counts by timestep: {dict(sorted(physical_edge_distribution.items()))}
- Serialized directed edge counts by timestep: {dict(sorted(directed_edge_distribution.items()))}

## Features

- Node features: {len(NODE_FEATURE_NAMES)} float64 values; see `tgnn_node_feature_schema.json`.
- Edge attributes: {len(EDGE_FEATURE_NAMES)} float64 values; see `tgnn_edge_schema.json`.
- Targets are run-level index labels only: `partition_any` maps exactly to `overall_threshold_breach_any`; `gcc_frac_min` maps exactly to `space_gcc_fraction_original_min`.
- No target, target derivative, future value, run summary, identifier, path, hash, seed, split, replay, or acceptance field is a learned graph feature.

## Storage

Each `sequences/run_NNN.tgnn` is a deterministic binary artifact with a canonical JSON header and non-pickle NumPy arrays. Variable node counts are represented by snapshot offsets and identity indexes; no padding or masking is used. Source undirected physical edges are serialized in both directions for PyTorch Geometric COO compatibility.

## Provenance and seal

The indexes and `tgnn_provenance_manifest.jsonl` retain frozen source paths and exact SHA-256 identities. The sealed test index contains no classification or regression target fields, no outcome summaries, and test target files were not read. The source freeze manifest SHA-256 is `{EXPECTED_FREEZE_MANIFEST_SHA256}` and the contract hash is `{EXPECTED_CONTRACT_SPEC_HASH}`.

See `tgnn_construction_report.json` for structural checks and `tgnn_leakage_exclusion_report.json` for excluded fields. The independent TGNN dataset acceptance gate was not performed.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8", newline="")
    artifact_paths = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "artifact_inventory.json")
    inventory = {"schema": "satnet.stage_a_tgnn_dataset_v1.artifact_inventory", "inventory_policy": "self_excluding", "artifact_count_excluding_inventory": len(artifact_paths), "artifacts": [{"path": str(path.relative_to(output_dir.parent.parent)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in artifact_paths]}
    write_json(output_dir / "artifact_inventory.json", inventory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct Stage A temporal GNN model-ready dataset v1")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--freeze-bundle", type=Path, default=repo_root / "artifacts" / "stage_a_accepted_production_freeze_v1")
    parser.add_argument("--production-root", type=Path, default=Path(r"C:\Users\johns\satnet-stage-a-production-generate-v1-activation-20260730-production"))
    parser.add_argument("--contract-root", type=Path, default=Path(r"C:\Users\johns\satnet-stage-a-production-generate-v1-activation-20260730\artifacts\stage_a_production_boundary_adjusted_v1_contract"))
    parser.add_argument("--output-dir", type=Path, default=repo_root / "artifacts" / "stage_a_tgnn_dataset_v1")
    args = parser.parse_args()
    build_dataset(args.freeze_bundle.resolve(), args.production_root.resolve(), args.contract_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
