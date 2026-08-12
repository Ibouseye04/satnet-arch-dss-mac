from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .contracts import DATASET_ROOT, TaskContract, get_task, load_json, sha256_file, verify_dataset_bundle
from .splits import FrozenSplits, validate_frozen_split_frame, validate_manifest_identities
from .test_access import MetadataView, TargetGate

MAGIC = b"SATNET-TGNN-V1\x00"
EXPECTED_SEQUENCE_LENGTH = 11
EXPECTED_NODE_FEATURE_DIMENSION = 3
EXPECTED_EDGE_FEATURE_DIMENSION = 4
EXPECTED_NODE_FEATURES = ("plane_idx_normalized", "sat_in_plane_normalized", "node_exists_constant")
EXPECTED_EDGE_FEATURES = ("distance_km_scaled_10000", "margin_db_scaled_100", "link_type_code_scaled_2", "link_mode_binary")
ARRAY_ORDER = ("node_features", "node_identity_index", "edge_index", "edge_attr", "snapshot_node_offsets", "snapshot_edge_offsets", "timestep_index")


@dataclass(frozen=True)
class SequenceArtifact:
    path: Path
    header: dict[str, Any]
    node_features: np.ndarray
    node_identity_index: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    snapshot_node_offsets: np.ndarray
    snapshot_edge_offsets: np.ndarray
    timestep_index: np.ndarray

    @property
    def sequence_length(self) -> int:
        return len(self.timestep_index)

    def data_list(self) -> list[Data]:
        result: list[Data] = []
        for timestep in range(self.sequence_length):
            n0, n1 = map(int, self.snapshot_node_offsets[timestep:timestep + 2])
            e0, e1 = map(int, self.snapshot_edge_offsets[timestep:timestep + 2])
            x = torch.as_tensor(self.node_features[n0:n1], dtype=torch.float32)
            edge_index = torch.as_tensor(self.edge_index[:, e0:e1], dtype=torch.long)
            edge_attr = torch.as_tensor(self.edge_attr[e0:e1], dtype=torch.float32)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data.node_identity_index = torch.as_tensor(self.node_identity_index[n0:n1], dtype=torch.int32)
            # The frozen scientific model consumes the existing scalar edge_weight API;
            # preserve all four frozen attributes on edge_attr without changing that model.
            data.edge_weight = edge_attr[:, 0]
            data.timestep_index = torch.tensor([int(self.timestep_index[timestep])], dtype=torch.long)
            result.append(data)
        return result


@dataclass(frozen=True)
class TGNNView:
    split: str
    records: tuple[dict[str, Any], ...]
    gate: TargetGate

    @property
    def targets(self) -> tuple[Any, ...]:
        return self.gate.targets

    def authorized_targets(self, authorization):
        return self.gate.authorized_targets(authorization)


@dataclass
class TGNNDataBundle:
    task: TaskContract
    dataset_root: Path
    graph_records: tuple[dict[str, Any], ...]
    target_records: tuple[dict[str, Any], ...]
    splits: FrozenSplits
    _targets_by_run: dict[int, Any]
    _records_by_run: dict[int, dict[str, Any]]

    def view(self, split: str) -> TGNNView:
        run_ids = self.splits.run_ids[split]
        records = tuple(self._records_by_run[run] for run in run_ids)
        metadata = MetadataView(tuple({k: record[k] for k in ("run_id", "run_key", "design_id", "realization_id", "split")} for record in records))
        gate = TargetGate(split=split, metadata=metadata, target_values=tuple(self._targets_by_run[run] for run in run_ids))
        return TGNNView(split, records, gate)

    def load_run(self, run_id: int, *, authorization=None) -> tuple[SequenceArtifact, Any | None]:
        record = self._records_by_run[int(run_id)]
        artifact = read_sequence(self.dataset_root / record["sequence_artifact"])
        if sha256_file(artifact.path) != record["sequence_artifact_sha256"]:
            raise ValueError(f"Sequence hash mismatch for run {run_id}")
        identity = artifact.header.get("sequence", {})
        for field in ("run_id", "run_key", "design_id", "realization_id", "split"):
            if identity.get(field) != record.get(field):
                raise ValueError(f"TGNN sequence identity mismatch for run {run_id}: {field}")
        if record["split"] == "test":
            if authorization is None:
                return artifact, None
            target = self.view("test").gate.authorized_targets(authorization)[self.splits.run_ids["test"].index(int(run_id))]
            return artifact, target
        return artifact, self._targets_by_run[int(run_id)]

    def structural_sample(self, split: str) -> SequenceArtifact:
        run_id = self.splits.run_ids[split][0]
        return self.load_run(run_id)[0]


def _load_numpy_array(blob: bytes) -> np.ndarray:
    value = np.load(io.BytesIO(blob), allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise ValueError("Sequence array is not a NumPy ndarray")
    return value


def read_sequence(path: Path) -> SequenceArtifact:
    raw = path.read_bytes()
    if not raw.startswith(MAGIC):
        raise ValueError(f"Malformed TGNN sequence magic: {path}")
    if len(raw) < len(MAGIC) + 8:
        raise ValueError(f"Truncated TGNN sequence: {path}")
    header_length = struct.unpack_from("<Q", raw, len(MAGIC))[0]
    header_start = len(MAGIC) + 8
    header_end = header_start + header_length
    if header_end > len(raw):
        raise ValueError(f"Truncated TGNN sequence header: {path}")
    header = json.loads(raw[header_start:header_end].decode("utf-8"))
    if header.get("format") != "satnet_tgnn_sequence_binary_v1" or header.get("sequence_length") != EXPECTED_SEQUENCE_LENGTH:
        raise ValueError("Unsupported or malformed frozen TGNN sequence format")
    if header.get("node_feature_dimension") != EXPECTED_NODE_FEATURE_DIMENSION or tuple(header.get("node_feature_names", ())) != EXPECTED_NODE_FEATURES:
        raise ValueError("TGNN node feature contract mismatch")
    if header.get("edge_feature_dimension") != EXPECTED_EDGE_FEATURE_DIMENSION or tuple(header.get("edge_feature_names", ())) != EXPECTED_EDGE_FEATURES:
        raise ValueError("TGNN edge feature contract mismatch")
    arrays: dict[str, np.ndarray] = {}
    offset = header_end
    for name in ARRAY_ORDER:
        if offset + 8 > len(raw):
            raise ValueError(f"Truncated TGNN array length for {name}")
        length = struct.unpack_from("<Q", raw, offset)[0]
        offset += 8
        if offset + length > len(raw):
            raise ValueError(f"Truncated TGNN array payload for {name}")
        arrays[name] = _load_numpy_array(raw[offset:offset + length])
        offset += length
    if offset != len(raw):
        raise ValueError("Unexpected trailing bytes in TGNN sequence")
    expected_dtypes = {"node_features": np.float64, "node_identity_index": np.int32, "edge_index": np.int64, "edge_attr": np.float64, "snapshot_node_offsets": np.int64, "snapshot_edge_offsets": np.int64, "timestep_index": np.int64}
    for name, dtype in expected_dtypes.items():
        if arrays[name].dtype != dtype:
            raise ValueError(f"TGNN array {name} has dtype {arrays[name].dtype}, expected {dtype}")
    if arrays["node_features"].ndim != 2 or arrays["node_features"].shape[1] != 3 or arrays["edge_attr"].ndim != 2 or arrays["edge_attr"].shape[1] != 4 or arrays["edge_index"].shape[0] != 2:
        raise ValueError("TGNN array dimensions do not match frozen contract")
    if len(arrays["snapshot_node_offsets"]) != 12 or len(arrays["snapshot_edge_offsets"]) != 12 or len(arrays["timestep_index"]) != 11:
        raise ValueError("TGNN offsets/timestep dimensions do not match 11 snapshots")
    if not np.array_equal(arrays["timestep_index"], np.arange(11)):
        raise ValueError("TGNN timestep order must be exactly ascending 0..10")
    if not np.all(np.diff(arrays["snapshot_node_offsets"]) >= 0) or not np.all(np.diff(arrays["snapshot_edge_offsets"]) >= 0):
        raise ValueError("TGNN snapshot offsets must be monotonic")
    if arrays["snapshot_node_offsets"][-1] != len(arrays["node_features"]) or arrays["snapshot_edge_offsets"][-1] != len(arrays["edge_attr"]):
        raise ValueError("TGNN offsets do not cover the serialized arrays")
    if len(arrays["node_identity_index"]) != len(arrays["node_features"]):
        raise ValueError("TGNN node identity array length mismatch")
    return SequenceArtifact(path, header, *(arrays[name] for name in ARRAY_ORDER))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Manifest row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _parse_target(value: object, task_type: str) -> Any:
    if task_type == "classification":
        text = str(value).strip().lower()
        if text in {"true", "1"}: return 1
        if text in {"false", "0"}: return 0
        raise ValueError(f"Invalid TGNN classification target: {value!r}")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError("TGNN regression target is non-finite")
    return parsed


def load_tgnn_task(task_id: str, *, dataset_root: Path = DATASET_ROOT, verify_bundle: bool = True) -> TGNNDataBundle:
    task = get_task(task_id)
    if task.family != "TGNN":
        raise ValueError(f"{task_id} is not a TGNN task")
    if verify_bundle:
        verify_dataset_bundle(dataset_root, full=True)
    graph_manifest_path = dataset_root / "tgnn_space_classification" / "tgnn_space_graph_manifest.jsonl"
    target_manifest_path = dataset_root / str(task.target_relative_path)
    graph_records = _read_jsonl(graph_manifest_path)
    target_records = _read_jsonl(target_manifest_path)
    if len(graph_records) != 10000 or len(target_records) != 10000:
        raise ValueError("TGNN manifests must contain exactly 10000 records")
    frame = pd.DataFrame(graph_records)
    splits = validate_frozen_split_frame(frame)
    validate_manifest_identities(graph_records, splits)
    validate_manifest_identities(target_records, splits)
    by_graph = {int(row["run_id"]): row for row in graph_records}
    by_target = {int(row["run_id"]): row for row in target_records}
    if set(by_graph) != set(by_target):
        raise ValueError("TGNN graph and target manifests have different run identities")
    targets: dict[int, Any] = {}
    for run_id, target in by_target.items():
        if target.get("target_field") != task.target:
            raise ValueError(f"TGNN target field mismatch for run {run_id}")
        if target.get("design_id") != by_graph[run_id].get("design_id") or target.get("realization_id") != by_graph[run_id].get("realization_id") or target.get("run_key") != by_graph[run_id].get("run_key"):
            raise ValueError(f"TGNN target identity mismatch for run {run_id}")
        targets[run_id] = _parse_target(target.get("target"), task.task_type)
    records_by_run: dict[int, dict[str, Any]] = {}
    for run_id, record in by_graph.items():
        if record.get("sequence_length") != 11 or len(record.get("node_counts", [])) != 11 or len(record.get("directed_edge_counts", [])) != 11:
            raise ValueError(f"Malformed TGNN graph manifest row for run {run_id}")
        sequence_path = dataset_root / str(record["sequence_artifact"])
        if not sequence_path.is_file():
            raise FileNotFoundError(sequence_path)
        records_by_run[run_id] = record
    return TGNNDataBundle(task, dataset_root, tuple(graph_records), tuple(target_records), splits, targets, records_by_run)
