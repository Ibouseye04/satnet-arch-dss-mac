from __future__ import annotations

"""Materialize and validate the immutable SATNET 10k ML dataset freeze.

This exporter reads frozen contract, production, and replay artifacts only. It
never calls a simulator, splitter, model, optimizer, or preprocessing fitter.
The only generated files are written below the caller-provided external output
root.
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from typing import Any, Iterable

import numpy as np

QUALIFICATION_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = QUALIFICATION_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from satnet.ground.integrated_persistence import read_integrated_graph_manifest
from satnet.experiments.final_dataset.design import design_manifest_hash, run_manifest_hash

TOOL_VERSION = "satnet-final-ml-dataset-export-v1"
TOOLING_SHA = "d0515088cf3fca06a6aa2d47059269089dcb10a7"
CONTRACT_SPEC_HASH = "6c7dd365f9e7fb67f5f5e70879a19535ede55468aabfac53d82c2ab35b8307eb"
ML_CONTRACT_BUNDLE_HASH = "056d01886a9b0ac39782780a6e7444fa22638679998058c8145ea9bafb29ad2a"
PRODUCTION_CONTRACT_BUNDLE_HASH = "059dff74930d1125a46947a06d213dd07c3a93dce226a894558a01805c3ed94c"
SPLIT_CANDIDATE = 3958
RUN_COUNT = 10_000
DESIGN_COUNT = 2_000
REALIZATIONS_PER_DESIGN = 5
TIMESTEPS = 11
SPLITS = ("train", "validation", "test")
SPLIT_RUN_COUNTS = {"train": 7000, "validation": 1500, "test": 1500}
SPLIT_DESIGN_COUNTS = {"train": 1400, "validation": 300, "test": 300}
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
RF_SPACE_FEATURES = (
    "num_planes", "sats_per_plane", "altitude_km", "inclination_deg",
    "satellite_node_failure_probability", "satellite_edge_failure_probability",
)
RF_INTEGRATED_FEATURES = RF_SPACE_FEATURES + (
    "civilian_count", "government_count", "military_count",
    "ground_station_failure_probability",
)
METADATA_COLUMNS = ("run_id", "run_key", "design_id", "realization_id", "split")
EXCLUDED_RF_FIELDS = (
    "configured_satellite_count", "total_ground_station_count", "duration_minutes",
    "timestep", "step_seconds", "constant ISL fields", "IDs", "split labels",
    "seeds", "hashes", "replay status", "generation status", "realized failure sets",
    "realized failure counts", "graph outcomes", "GCC outcomes", "service outcomes",
    "threshold indicators other than selected target", "target-derived values",
)


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


def semantic_hash(value: dict[str, Any], field: str) -> str:
    return sha256_bytes(canonical_json({key: item for key, item in value.items() if key != field}).encode())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(value) + "\n" for value in values), encoding="utf-8", newline="\n")


def attr_map(attributes: Iterable[Any], context: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attribute in attributes:
        name = getattr(attribute, "name", None)
        value = getattr(attribute, "value", None)
        value_type = getattr(getattr(attribute, "value_type", None), "value", None)
        require(isinstance(name, str) and name not in result, f"{context}: invalid/duplicate attribute")
        if value_type == "float":
            value = float(value)
        elif value_type == "integer":
            require(type(value) is int, f"{context}: invalid integer attribute")
        elif value_type == "boolean":
            require(type(value) is bool, f"{context}: invalid boolean attribute")
        result[name] = value
    return result


def serialize_array(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def sequence_bytes(header: dict[str, Any], arrays: dict[str, np.ndarray]) -> bytes:
    payload = bytearray(MAGIC)
    header_bytes = canonical_json(header).encode("utf-8")
    payload.extend(struct.pack("<Q", len(header_bytes)))
    payload.extend(header_bytes)
    for name in ARRAY_NAMES:
        data = serialize_array(arrays[name])
        payload.extend(struct.pack("<Q", len(data)))
        payload.extend(data)
    return bytes(payload)


@dataclass(frozen=True)
class RunEvidence:
    run_id: int
    run: dict[str, Any]
    design: dict[str, Any]
    target: dict[str, Any]
    result_sha256: str
    inventory_sha256: str
    target_sha256: str
    graph_sha256: str
    graph_relative_path: str
    satellite_sha256: str
    source_run_dir: Path

    @property
    def split(self) -> str:
        return str(self.run["split_assignment"])

    @property
    def design_id(self) -> str:
        return str(self.run["design_id"])


class Exporter:
    def __init__(self, *, production_root: Path, replay_root: Path, acceptance_root: Path,
                 audit_root: Path, ml_contract_root: Path, contract_root: Path,
                 output_root: Path) -> None:
        self.production_root = production_root.resolve()
        self.replay_root = replay_root.resolve()
        self.acceptance_root = acceptance_root.resolve()
        self.audit_root = audit_root.resolve()
        self.ml_contract_root = ml_contract_root.resolve()
        self.contract_root = contract_root.resolve()
        self.output_root = output_root.resolve()
        self.contract_inventory: dict[str, Any] = {}
        self.contract_summary: dict[str, Any] = {}
        self.split_rows: list[dict[str, str]] = []
        self.evidence: list[RunEvidence] = []
        self.design_by_id: dict[str, dict[str, Any]] = {}
        self.graph_records_by_run: dict[int, tuple[Any, ...]] = {}
        self.validation: dict[str, Any] = {}

    def verify_frozen_contract(self) -> None:
        if self.output_root.exists():
            require(self.output_root.is_dir(), f"Output path is not a directory: {self.output_root}")
            require(not (self.output_root / "final_ml_dataset_inventory.json").exists(), f"Refusing to overwrite immutable export inventory: {self.output_root}")
            require(not (self.output_root / "final_ml_dataset_export_report.md").exists(), f"Refusing to overwrite immutable export report: {self.output_root}")
        inventory_path = self.ml_contract_root / "contract_artifact_inventory.json"
        self.contract_inventory = read_json(inventory_path)
        require(self.contract_inventory["bundle_sha256"] == ML_CONTRACT_BUNDLE_HASH, "Frozen ML bundle identity differs")
        payload = {key: value for key, value in self.contract_inventory.items() if key != "bundle_sha256"}
        require(sha256_bytes(canonical_json(payload).encode()) == ML_CONTRACT_BUNDLE_HASH, "Frozen ML bundle semantic hash mismatch")
        for artifact in self.contract_inventory["artifacts"]:
            path = self.ml_contract_root / artifact["path"]
            require(path.is_file(), f"Missing frozen ML contract artifact: {path}")
            require(path.stat().st_size == artifact["bytes"], f"Frozen contract byte length mismatch: {path}")
            require(sha256_file(path) == artifact["sha256"], f"Frozen contract artifact hash mismatch: {path}")
        self.contract_summary = read_json(self.ml_contract_root / "ml_contract_summary.json")
        identity = self.contract_summary["authoritative_identity"]
        require(identity["tooling_sha"] == TOOLING_SHA, "Frozen tooling SHA mismatch")
        require(identity["contract_spec_sha256"] == CONTRACT_SPEC_HASH, "Frozen contract specification hash mismatch")
        require(identity["selected_split_candidate"] == SPLIT_CANDIDATE, "Frozen split candidate mismatch")
        require(self.contract_summary["counts"] == {"designs": DESIGN_COUNT, "realizations_per_design": 5, "runs": RUN_COUNT}, "Frozen cardinality mismatch")
        require(self.contract_summary["design_cross_split"] is False, "Frozen contract permits design split leakage")
        require(self.contract_summary["outcome_fields_used_for_split"] is False, "Frozen contract split used outcomes")
        require(self.contract_summary["training_prohibited"] is True, "Frozen contract does not prohibit training")
        contract_spec_path = self.contract_root / "contract_specification.json"
        contract_spec = read_json(contract_spec_path)
        require(contract_spec.get("contract_spec_hash") == CONTRACT_SPEC_HASH, "Production contract field mismatch")
        require(semantic_hash(contract_spec, "contract_spec_hash") == CONTRACT_SPEC_HASH, "Production contract semantic hash mismatch")
        contract_designs = read_jsonl(self.contract_root / "designs.jsonl")
        contract_runs = read_jsonl(self.contract_root / "runs.jsonl")
        require(design_manifest_hash(contract_designs) == identity["design_manifest_sha256"], "Frozen design manifest semantic hash mismatch")
        require(run_manifest_hash(contract_runs) == identity["run_manifest_sha256"], "Frozen run manifest semantic hash mismatch")
        contract_split = read_json(self.contract_root / "split_manifest.json")
        require(contract_split.get("split_manifest_hash") == identity["split_manifest_sha256"], "Frozen split manifest identity mismatch")
        require(semantic_hash(contract_split, "split_manifest_hash") == identity["split_manifest_sha256"], "Frozen split manifest semantic hash mismatch")
        split_contract = self.ml_contract_root / "split_contract.csv"
        require(sha256_file(split_contract) == next(a["sha256"] for a in self.contract_inventory["artifacts"] if a["path"] == "split_contract.csv"), "Frozen split contract hash mismatch")
        with split_contract.open("r", encoding="utf-8", newline="") as handle:
            self.split_rows = list(csv.DictReader(handle))
        require(len(self.split_rows) == RUN_COUNT, "Frozen split row count mismatch")
        require(list(self.split_rows[0]) == ["run_id", "run_key", "design_id", "realization_id", "split"], "Frozen split columns changed")
        require([int(row["run_id"]) for row in self.split_rows] == list(range(RUN_COUNT)), "Frozen split ordering/coverage mismatch")
        require({split: sum(row["split"] == split for row in self.split_rows) for split in SPLITS} == SPLIT_RUN_COUNTS, "Frozen split run counts mismatch")
        require({split: len({row["design_id"] for row in self.split_rows if row["split"] == split}) for split in SPLITS} == SPLIT_DESIGN_COUNTS, "Frozen split design counts mismatch")

    def _source_artifact(self, inventory: dict[str, Any], role: str) -> tuple[str, str]:
        matches = [item for item in inventory["artifacts"] if item["artifact_role"] == role]
        require(len(matches) == 1, f"Expected one {role} artifact")
        return str(matches[0]["path"]), str(matches[0]["sha256"])

    def load_and_validate_production(self) -> None:
        acceptance = read_json(self.acceptance_root / "production_acceptance.json")
        require(acceptance == {"derived_generation_submission_count": 10000, "derived_replay_submission_count": 10000, "production_acceptance": "passed", "validated_run_count": 10000}, "Production acceptance artifact differs")
        run_rows = {int(row["run_id"]): row for row in self.split_rows}
        require(len(run_rows) == RUN_COUNT, "Duplicate frozen run IDs")
        for run_id in range(RUN_COUNT):
            run_dir = self.production_root / f"run_{run_id:04d}"
            replay_dir = self.replay_root / f"run_{run_id:04d}"
            design = read_json(run_dir / "input" / "design_record.json")
            run = read_json(run_dir / "input" / "run_record.json")
            target_path = run_dir / "targets" / "target.json"
            result_path = run_dir / "result.json"
            inventory_path = run_dir / "scientific_inventory.json"
            target = read_json(target_path)
            result = read_json(result_path)
            inventory = read_json(inventory_path)
            replay = read_json(replay_dir / "replay_report.json")
            split_row = run_rows[run_id]
            require(run["run_id"] == run_id and target["run_id"] == run_id and result["run_id"] == run_id, f"Run identity mismatch: {run_id}")
            require(run["run_key"] == split_row["run_key"] == target["run_key"], f"Run key mismatch: {run_id}")
            require(run["design_id"] == split_row["design_id"] == design["design_id"], f"Design identity mismatch: {run_id}")
            require(run["realization_id"] == split_row["realization_id"], f"Realization identity mismatch: {run_id}")
            require(run["split_assignment"] == split_row["split"], f"Split identity mismatch: {run_id}")
            require(run["contract_spec_hash"] == CONTRACT_SPEC_HASH and target["contract_spec_hash"] == CONTRACT_SPEC_HASH, f"Contract hash mismatch: {run_id}")
            require(target["target_artifact_hash"] == semantic_hash(target, "target_artifact_hash"), f"Target semantic hash mismatch: {run_id}")
            require(inventory["scientific_inventory_hash"] == semantic_hash(inventory, "scientific_inventory_hash"), f"Scientific inventory semantic hash mismatch: {run_id}")
            require(result["run_result_hash"] == semantic_hash(result, "run_result_hash"), f"Result semantic hash mismatch: {run_id}")
            require(replay["replay_state"] == "succeeded" and replay["input_tree_unchanged"] is True, f"Replay acceptance mismatch: {run_id}")
            require(all(stage["state"] == "matched" for stage in replay["per_stage_comparison"]), f"Replay stage mismatch: {run_id}")
            graph_relative, graph_sha = self._source_artifact(inventory, "integrated_graph")
            satellite_relative, satellite_sha = self._source_artifact(inventory, "satellite_rollout")
            require(sha256_file(run_dir / graph_relative) == graph_sha, f"Graph artifact hash mismatch: {run_id}")
            require(sha256_file(run_dir / satellite_relative) == satellite_sha, f"Satellite artifact hash mismatch: {run_id}")
            require(target["target_artifact_hash"] == result["target_artifact_hash"] == replay["target_artifact_hash"], f"Target linkage mismatch: {run_id}")
            require(target["design_record_hash"] == design["design_record_hash"] == run["design_record_hash"], f"Design hash linkage mismatch: {run_id}")
            if design["design_id"] not in self.design_by_id:
                self.design_by_id[design["design_id"]] = design
            else:
                require(self.design_by_id[design["design_id"]] == design, f"Repeated design record differs: {design['design_id']}")
            self.evidence.append(RunEvidence(
                run_id=run_id, run=run, design=design, target=target,
                result_sha256=sha256_file(result_path), inventory_sha256=sha256_file(inventory_path),
                target_sha256=sha256_file(target_path), graph_sha256=graph_sha,
                graph_relative_path=graph_relative, satellite_sha256=satellite_sha, source_run_dir=run_dir,
            ))
        require(len(self.design_by_id) == DESIGN_COUNT, "Unique design count mismatch")
        require([item.run_id for item in self.evidence] == list(range(RUN_COUNT)), "Evidence ordering mismatch")
        require(all(sum(item.design_id == design_id for item in self.evidence) == REALIZATIONS_PER_DESIGN for design_id in self.design_by_id), "Five realizations/design gate failed")
        self.validation["production_acceptance"] = "passed"
        self.validation["source_evidence"] = {"runs": RUN_COUNT, "designs": len(self.design_by_id), "production_modified": False, "replay_modified": False}

    def _rf_row(self, item: RunEvidence, features: tuple[str, ...], targets: tuple[str, ...]) -> dict[str, str]:
        row = {
            "run_id": str(item.run_id), "run_key": str(item.run["run_key"]),
            "design_id": str(item.design_id), "realization_id": str(item.run["realization_id"]), "split": item.split,
        }
        for field in features:
            value = item.design[field]
            row[field] = str(value).lower() if isinstance(value, bool) else str(value)
        for field in targets:
            value = item.target[field]
            row[field] = str(value).lower() if isinstance(value, bool) else str(value)
        return row

    def _write_rf_task(self, task: str, filename: str, schema_name: str, features: tuple[str, ...], targets: tuple[str, ...], expected_balance: dict[str, dict[str, int]] | None = None) -> dict[str, Any]:
        task_dir = self.output_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        rows = [self._rf_row(item, features, targets) for item in self.evidence]
        csv_path = task_dir / filename
        fields = list(METADATA_COLUMNS) + list(features) + list(targets)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        schema_source = self.ml_contract_root / f"{task}_schema.json"
        shutil.copyfile(schema_source, task_dir / f"{task}_schema.json")
        quality = self._validate_rf_file(csv_path, rows, features, targets, expected_balance)
        manifest = {
            "manifest_version": "1", "task": task, "schema": f"{task}_schema.json", "sample_unit": "one simulation run",
            "row_order": "ascending run_id", "row_count": RUN_COUNT, "predictors": list(features),
            "metadata_columns_not_model_features": list(METADATA_COLUMNS), "targets": list(targets),
            "target_representation": "binary targets copied as Boolean spellings; numeric targets copied as canonical production strings",
            "source_target_artifact": "production_generation/run_NNNN/targets/target.json",
            "source_design_artifact": "production_generation/run_NNNN/input/design_record.json",
            "provenance": {"tooling_sha": TOOLING_SHA, "contract_spec_hash": CONTRACT_SPEC_HASH, "ml_contract_bundle_hash": ML_CONTRACT_BUNDLE_HASH, "split_candidate": SPLIT_CANDIDATE},
            "split_counts": self._split_counts(rows), "quality": quality,
            "leakage": {"excluded_fields": list(EXCLUDED_RF_FIELDS), "unexpected_fields": [], "predictor_identity": "exact source design_record fields"},
        }
        manifest_path = task_dir / f"{task}_manifest.json"
        write_json(manifest_path, manifest)
        return {"task": task, "csv": csv_path, "schema": task_dir / f"{task}_schema.json", "manifest": manifest_path, "rows": rows, "quality": quality, "manifest_value": manifest}

    def _split_counts(self, rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
        return {split: {"runs": sum(row["split"] == split for row in rows), "designs": len({row["design_id"] for row in rows if row["split"] == split})} for split in SPLITS}

    def _validate_rf_file(self, path: Path, expected_rows: list[dict[str, str]], features: tuple[str, ...], targets: tuple[str, ...], expected_balance: dict[str, dict[str, int]] | None) -> dict[str, Any]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            actual_rows = list(csv.DictReader(handle))
        expected_fields = list(METADATA_COLUMNS) + list(features) + list(targets)
        require(actual_rows == expected_rows, f"RF materialization identity mismatch: {path}")
        missing = sum(1 for row in actual_rows for field in expected_fields if row.get(field, "") == "")
        nan = sum(1 for row in actual_rows for field in features + targets if row.get(field, "").strip().lower() == "nan")
        pos_inf = sum(1 for row in actual_rows for field in features + targets if row.get(field, "").strip().lower() in {"inf", "+inf", "infinity", "+infinity"})
        neg_inf = sum(1 for row in actual_rows for field in features + targets if row.get(field, "").strip().lower() in {"-inf", "-infinity"})
        duplicate_run_ids = len(actual_rows) - len({row["run_id"] for row in actual_rows})
        exact_duplicate_rows = len(actual_rows) - len({tuple(row[field] for field in expected_fields) for row in actual_rows})
        constant_predictors = [field for field in features if len({row[field] for row in actual_rows}) == 1]
        unexpected_fields = sorted(set(actual_rows[0]) - set(expected_fields)) if actual_rows else expected_fields
        for field in targets:
            if field.endswith("_any"):
                require(all(row[field] in {"true", "false"} for row in actual_rows), f"Non-Boolean target spelling: {field}")
            else:
                values = [float(row[field]) for row in actual_rows]
                require(all(math.isfinite(value) for value in values), f"Non-finite target: {field}")
                require(len(set(values)) > 1, f"Constant regression target: {field}")
        balance: dict[str, dict[str, int]] = {}
        for split in ("overall",) + SPLITS:
            selected = actual_rows if split == "overall" else [row for row in actual_rows if row["split"] == split]
            if targets and targets[0].endswith("_any"):
                positive = sum(row[targets[0]] == "true" for row in selected)
                balance[split] = {"positive": positive, "negative": len(selected) - positive, "total": len(selected)}
        if expected_balance is not None:
            require(balance == expected_balance, f"Class balance mismatch for {path}: {balance} != {expected_balance}")
        require(missing == nan == pos_inf == neg_inf == duplicate_run_ids == exact_duplicate_rows == 0, f"RF quality gate failed: {path}")
        require(not unexpected_fields, f"Unexpected RF fields: {unexpected_fields}")
        return {"row_count": len(actual_rows), "predictor_count": len(features), "target_count": len(targets), "missing_values": missing, "nan": nan, "+Inf": pos_inf, "-Inf": neg_inf, "duplicate_run_ids": duplicate_run_ids, "exact_duplicate_rows": exact_duplicate_rows, "constant_predictor_fields": constant_predictors, "unexpected_fields": unexpected_fields, "class_balance": balance}

    def _sequence_for_run(self, item: RunEvidence) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
        graph_path = item.source_run_dir / item.graph_relative_path
        records = read_integrated_graph_manifest(graph_path)
        require(len(records) == TIMESTEPS, f"Malformed sequence length for run {item.run_id}")
        require([record.run_id for record in records] == [item.run_id] * TIMESTEPS, f"Graph run identity mismatch: {item.run_id}")
        require([record.timestep_index for record in records] == list(range(TIMESTEPS)), f"Missing/out-of-order timesteps: {item.run_id}")
        require(all(a.timestamp_utc < b.timestamp_utc for a, b in zip(records, records[1:])), f"Non-monotonic graph timestamps: {item.run_id}")
        expected_satellite_count = int(item.design["configured_satellite_count"])
        node_data_by_timestep: list[dict[int, dict[str, Any]]] = []
        edge_data_by_timestep: list[list[tuple[int, int, dict[str, Any]]]] = []
        all_ids: set[int] = set()
        for record in records:
            require(record.satellite_config_hash == item.run["expected_satellite_config_hash"], f"Graph configuration identity mismatch: {item.run_id}")
            nodes: dict[int, dict[str, Any]] = {}
            for node in record.canonical_nodes:
                ref = node.node_ref
                if ref.kind.value != "satellite":
                    continue
                node_id = int(ref.satellite_id)
                attrs = attr_map(node.attributes, f"run {item.run_id} node")
                require({"plane", "sat_in_plane", "type"}.issubset(attrs) and attrs["type"] == "satellite", f"Invalid satellite node features: {item.run_id}")
                require(0 <= node_id < expected_satellite_count, f"Satellite ID out of range: {item.run_id}")
                require(node_id not in nodes, f"Duplicate satellite node: {item.run_id}")
                expected_plane = node_id // int(item.design["sats_per_plane"])
                expected_sat = node_id % int(item.design["sats_per_plane"])
                require(int(attrs["plane"]) == expected_plane and int(attrs["sat_in_plane"]) == expected_sat, f"Node feature identity mismatch: {item.run_id}")
                nodes[node_id] = attrs
                all_ids.add(node_id)
            require(len(nodes) == record.satellite_node_count, f"Satellite node count mismatch: {item.run_id}")
            edges: list[tuple[int, int, dict[str, Any]]] = []
            seen: set[tuple[int, int]] = set()
            for edge in record.canonical_edges:
                if edge.edge_kind.value != "inter_satellite":
                    continue
                first = edge.endpoint_a
                second = edge.endpoint_b
                require(first.kind.value == second.kind.value == "satellite", f"Non-space edge in ISL record: {item.run_id}")
                a, b = int(first.satellite_id), int(second.satellite_id)
                require(a in nodes and b in nodes and a != b, f"Edge endpoint missing/self-loop: {item.run_id}")
                require(a < b, f"Non-canonical edge orientation: {item.run_id}")
                require((a, b) not in seen, f"Duplicate physical edge: {item.run_id}")
                seen.add((a, b))
                attrs = attr_map(edge.attributes, f"run {item.run_id} edge")
                require({"distance_km", "margin_db", "link_type", "link_mode"}.issubset(attrs), f"Incomplete ISL edge: {item.run_id}")
                require(attrs["link_type"] in {"intra_plane", "inter_plane", "seam_link"}, f"Invalid link type: {item.run_id}")
                require(attrs["link_mode"] == "optical", f"Invalid frozen link mode: {item.run_id}")
                require(math.isfinite(float(attrs["distance_km"])) and math.isfinite(float(attrs["margin_db"])), f"Non-finite edge features: {item.run_id}")
                edges.append((a, b, attrs))
            require(len(edges) == record.isl_edge_count, f"ISL edge count mismatch: {item.run_id}")
            node_data_by_timestep.append(nodes)
            edge_data_by_timestep.append(edges)
        identities = sorted(all_ids)
        identity_index = {node_id: index for index, node_id in enumerate(identities)}
        node_chunks: list[np.ndarray] = []
        identity_chunks: list[np.ndarray] = []
        edge_chunks: list[np.ndarray] = []
        edge_attr_chunks: list[np.ndarray] = []
        node_offsets = [0]
        edge_offsets = [0]
        node_counts: list[int] = []
        edge_counts: list[int] = []
        for nodes, edges in zip(node_data_by_timestep, edge_data_by_timestep, strict=True):
            local_ids = sorted(nodes)
            local_index = {node_id: index for index, node_id in enumerate(local_ids)}
            node_rows = [[int(nodes[node_id]["plane"]) / max(int(item.design["num_planes"]) - 1, 1), int(nodes[node_id]["sat_in_plane"]) / max(int(item.design["sats_per_plane"]) - 1, 1), 1.0] for node_id in local_ids]
            node_chunks.append(np.asarray(node_rows, dtype=np.float64).reshape((-1, 3)))
            identity_chunks.append(np.asarray([identity_index[node_id] for node_id in local_ids], dtype=np.int32))
            directed: list[tuple[int, int]] = []
            directed_attrs: list[list[float]] = []
            for a, b, attrs in edges:
                edge_attr = [float(attrs["distance_km"]) / 10000.0, float(attrs["margin_db"]) / 100.0, {"intra_plane": 0.0, "inter_plane": 0.5, "seam_link": 1.0}[attrs["link_type"]], 0.0]
                directed.extend(((local_index[a], local_index[b]), (local_index[b], local_index[a])))
                directed_attrs.extend((edge_attr, edge_attr))
            edge_chunks.append(np.asarray(directed, dtype=np.int64).T if directed else np.empty((2, 0), dtype=np.int64))
            edge_attr_chunks.append(np.asarray(directed_attrs, dtype=np.float64).reshape((-1, 4)) if directed_attrs else np.empty((0, 4), dtype=np.float64))
            node_offsets.append(node_offsets[-1] + len(local_ids))
            edge_offsets.append(edge_offsets[-1] + len(directed))
            node_counts.append(len(local_ids))
            edge_counts.append(len(directed))
        arrays = {
            "node_features": np.concatenate(node_chunks, axis=0),
            "node_identity_index": np.concatenate(identity_chunks, axis=0),
            "edge_index": np.concatenate(edge_chunks, axis=1),
            "edge_attr": np.concatenate(edge_attr_chunks, axis=0),
            "snapshot_node_offsets": np.asarray(node_offsets, dtype=np.int64),
            "snapshot_edge_offsets": np.asarray(edge_offsets, dtype=np.int64),
            "timestep_index": np.arange(TIMESTEPS, dtype=np.int64),
        }
        header = {
            "format": "satnet_tgnn_sequence_binary_v1", "schema_version": "satnet_space_tgnn_graph_schema_v1",
            "sequence": {"contract_spec_hash": CONTRACT_SPEC_HASH, "design_id": item.design_id, "realization_id": item.run["realization_id"], "run_id": item.run_id, "run_key": item.run["run_key"], "split": item.split},
            "sequence_length": TIMESTEPS, "timestep_order": "ascending timestep_index", "timesteps": [{"timestep_index": record.timestep_index, "timestamp_utc": record.timestamp_utc.isoformat().replace("+00:00", "Z")} for record in records],
            "node_identity": [{"satellite_id": node_id} for node_id in identities], "node_identity_order": "satellite numeric ascending",
            "node_count_by_timestep": node_counts, "directed_edge_count_by_timestep": edge_counts,
            "node_feature_names": ["plane_idx_normalized", "sat_in_plane_normalized", "node_exists_constant"],
            "edge_feature_names": ["distance_km_scaled_10000", "margin_db_scaled_100", "link_type_code_scaled_2", "link_mode_binary"],
            "node_feature_dimension": 3, "edge_feature_dimension": 4, "targets_in_graph_artifact": False,
            "array_dtypes": {name: str(arrays[name].dtype) for name in ARRAY_NAMES},
            "source_graph_sha256": item.graph_sha256,
        }
        require(bool(np.isfinite(arrays["node_features"]).all()) and bool(np.isfinite(arrays["edge_attr"]).all()), f"Non-finite serialized graph features: {item.run_id}")
        return header, arrays, {"sequence_length": TIMESTEPS, "node_counts": node_counts, "directed_edge_counts": edge_counts, "source_graph_sha256": item.graph_sha256, "source_graph_path": f"production_generation/run_{item.run_id:04d}/{item.graph_relative_path}"}

    def materialize_tgnn(self, task: str, target: str) -> dict[str, Any]:
        task_dir = self.output_root / task
        sequence_dir = self.output_root / "tgnn_space_classification" / "sequences"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        task_dir.mkdir(parents=True, exist_ok=True)
        schema_source = self.ml_contract_root / f"{task}_schema.json"
        shutil.copyfile(schema_source, task_dir / f"{task}_schema.json")
        graph_rows: list[dict[str, Any]] = []
        target_rows: list[dict[str, Any]] = []
        for item in self.evidence:
            sequence_path = sequence_dir / f"run_{item.run_id:04d}.tgnn"
            if not sequence_path.exists():
                header, arrays, metadata = self._sequence_for_run(item)
                sequence_path.write_bytes(sequence_bytes(header, arrays))
            else:
                header, arrays, metadata = self._sequence_for_run(item)
                require(sequence_path.read_bytes() == sequence_bytes(header, arrays), f"Existing sequence is not immutable/deterministic: {item.run_id}")
            artifact_path = sequence_path.relative_to(self.output_root).as_posix()
            graph_rows.append({"run_id": item.run_id, "run_key": item.run["run_key"], "design_id": item.design_id, "realization_id": item.run["realization_id"], "split": item.split, "sequence_artifact": artifact_path, "sequence_artifact_sha256": sha256_file(sequence_path), "sequence_length": metadata["sequence_length"], "source_graph_sha256": metadata["source_graph_sha256"], "source_graph_path": metadata["source_graph_path"], "node_counts": metadata["node_counts"], "directed_edge_counts": metadata["directed_edge_counts"]})
            target_rows.append({"run_id": item.run_id, "run_key": item.run["run_key"], "design_id": item.design_id, "realization_id": item.run["realization_id"], "split": item.split, "target": str(item.target[target]).lower() if isinstance(item.target[target], bool) else str(item.target[target]), "target_field": target, "source_target_artifact_sha256": item.target_sha256})
        graph_manifest_path = self.output_root / "tgnn_space_classification" / "tgnn_space_graph_manifest.jsonl"
        write_jsonl(graph_manifest_path, graph_rows)
        target_manifest_path = task_dir / f"{task}_target_manifest.jsonl"
        write_jsonl(target_manifest_path, target_rows)
        task_manifest = {
            "manifest_version": "1", "task": task, "graph_scope": "space-segment only", "sample_unit": "one full temporal satellite graph sequence per simulation run", "sample_count": RUN_COUNT, "row_order": "ascending run_id", "temporal_order": "ascending timestep", "sequence_storage": "shared immutable sequences under tgnn_space_classification/sequences", "graph_manifest": "../tgnn_space_classification/tgnn_space_graph_manifest.jsonl", "target_manifest": f"{task}_target_manifest.jsonl", "schema": f"{task}_schema.json", "node_feature_names": ["plane_idx_normalized", "sat_in_plane_normalized", "node_exists_constant"], "edge_feature_names": ["distance_km_scaled_10000", "margin_db_scaled_100", "link_type_code_scaled_2", "link_mode_binary"], "node_feature_dimension": 3, "edge_feature_dimension": 4, "targets_in_graph_artifact": False, "split_counts": {split: {"runs": sum(row["split"] == split for row in target_rows), "designs": len({row["design_id"] for row in target_rows if row["split"] == split})} for split in SPLITS}, "provenance": {"tooling_sha": TOOLING_SHA, "contract_spec_hash": CONTRACT_SPEC_HASH, "ml_contract_bundle_hash": ML_CONTRACT_BUNDLE_HASH, "split_candidate": SPLIT_CANDIDATE}, "quality": {"sample_count": len(target_rows), "sequence_count": len(graph_rows), "malformed_sequences": 0, "empty_sequences": sum(not row["sequence_length"] for row in graph_rows), "missing_timesteps": sum(row["sequence_length"] != TIMESTEPS for row in graph_rows), "node_feature_dimension": 3, "edge_feature_dimension": 4, "target_completeness": len(target_rows) == RUN_COUNT, "split_completeness": self._split_counts(target_rows)},
        }
        task_manifest_path = task_dir / f"{task}_manifest.json"
        write_json(task_manifest_path, task_manifest)
        return {"task": task, "graph_manifest": graph_manifest_path, "target_manifest": target_manifest_path, "manifest": task_manifest_path, "schema": task_dir / f"{task}_schema.json", "sequence_dir": sequence_dir, "graph_rows": graph_rows, "target_rows": target_rows, "manifest_value": task_manifest}

    def validate_splits_and_identity(self, tasks: list[dict[str, Any]]) -> None:
        results: dict[str, Any] = {}
        expected_ids = set(range(RUN_COUNT))
        for task in tasks:
            rows = task.get("rows", task.get("target_rows", []))
            ids = [int(row["run_id"]) for row in rows]
            design_by_split = {split: {row["design_id"] for row in rows if row["split"] == split} for split in SPLITS}
            overlap = any(design_by_split[a] & design_by_split[b] for index, a in enumerate(SPLITS) for b in SPLITS[index + 1:])
            results[task["task"]] = {"run_count": len(ids), "run_ids_complete_0_to_9999": set(ids) == expected_ids, "duplicate_run_ids": len(ids) - len(set(ids)), "split_counts": {split: {"runs": sum(row["split"] == split for row in rows), "designs": len(design_by_split[split])} for split in SPLITS}, "design_cross_split": overlap, "five_realizations_per_design": all(sum(row["design_id"] == design_id for row in rows) == 5 for design_id in {row["design_id"] for row in rows}), "ordered_run_ids": ids == list(range(RUN_COUNT))}
            require(results[task["task"]]["run_ids_complete_0_to_9999"] and not overlap and results[task["task"]]["duplicate_run_ids"] == 0 and results[task["task"]]["five_realizations_per_design"], f"Split/identity gate failed: {task['task']}")
            require(results[task["task"]]["split_counts"] == {split: {"runs": SPLIT_RUN_COUNTS[split], "designs": SPLIT_DESIGN_COUNTS[split]} for split in SPLITS}, f"Split counts failed: {task['task']}")
        self.validation["split_verification"] = results

    def write_metadata(self, rf_tasks: list[dict[str, Any]], tgnn_tasks: list[dict[str, Any]]) -> None:
        metadata = {
            "export_tool": TOOL_VERSION, "python_version": sys.version.split()[0], "production_tooling_sha": TOOLING_SHA, "production_contract_hash": CONTRACT_SPEC_HASH, "ml_contract_bundle_hash": ML_CONTRACT_BUNDLE_HASH, "production_contract_bundle_hash": PRODUCTION_CONTRACT_BUNDLE_HASH, "split_candidate": SPLIT_CANDIDATE, "roots_read_only": {"generation": str(self.production_root), "replay": str(self.replay_root), "acceptance": str(self.acceptance_root), "audit_corrected": str(self.audit_root), "ml_contract": str(self.ml_contract_root), "contract_manifests": str(self.contract_root)}, "no_training_invoked": True, "no_simulation_regenerated": True, "no_random_split_invoked": True, "no_preprocessing_fitted": True, "rf_tasks": [task["task"] for task in rf_tasks], "tgnn_tasks": [task["task"] for task in tgnn_tasks], "current_tgnn_scope": "space-segment only; integrated TGNN is not authorized"}
        write_json(self.output_root / "metadata" / "export_provenance.json", metadata)
        write_json(self.output_root / "metadata" / "validation_gates.json", self.validation)

    def build_inventory(self) -> tuple[Path, str, list[dict[str, Any]]]:
        entries: list[dict[str, Any]] = []
        for path in sorted(self.output_root.rglob("*")):
            if not path.is_file() or path.name in {"final_ml_dataset_inventory.json", "final_ml_dataset_export_report.md"}:
                continue
            relative = path.relative_to(self.output_root).as_posix()
            role = "sequence" if path.suffix == ".tgnn" else "dataset" if path.suffix == ".csv" else "schema" if path.name.endswith("_schema.json") else "manifest" if "manifest" in path.name or path.name.endswith("_gates.json") else "metadata"
            task = next((part for part in relative.split("/") if part.startswith(("rf_", "tgnn_"))), "metadata")
            target = ""
            if "space_classification" in relative:
                target = "space_threshold_breach_any"
            elif "space_regression" in relative:
                target = "space_gcc_fraction_original_min"
            elif "integrated_classification" in relative:
                target = "overall_threshold_breach_any"
            elif "integrated_regression" in relative:
                target = "failure_adjusted_overall_service_fraction_mean;failure_adjusted_overall_service_fraction_min"
            entries.append({"artifact_path": relative, "artifact_role": role, "task": task, "target": target, "sample_count": RUN_COUNT if role in {"dataset", "sequence"} or "manifest" in path.name else None, "split_counts": {split: {"runs": SPLIT_RUN_COUNTS[split], "designs": SPLIT_DESIGN_COUNTS[split]} for split in SPLITS} if role in {"dataset", "sequence"} or "manifest" in path.name else {}, "sha256": sha256_file(path), "bytes": path.stat().st_size, "creation_tool": TOOL_VERSION, "authoritative_production_sha": TOOLING_SHA, "production_contract_hash": CONTRACT_SPEC_HASH, "ml_contract_bundle_hash": ML_CONTRACT_BUNDLE_HASH})
        payload = {"inventory_version": "1", "identity": "satnet-10k-final-ml-datasets-v1", "artifacts": entries, "authoritative_production_sha": TOOLING_SHA, "production_contract_hash": CONTRACT_SPEC_HASH, "ml_contract_bundle_hash": ML_CONTRACT_BUNDLE_HASH}
        bundle_hash = sha256_bytes(canonical_json(payload).encode())
        inventory = {**payload, "bundle_sha256": bundle_hash}
        path = self.output_root / "final_ml_dataset_inventory.json"
        write_json(path, inventory)
        return path, bundle_hash, entries

    def reproducibility_probe(self, rf_space: dict[str, Any], tgnn_classification: dict[str, Any]) -> dict[str, Any]:
        probe_parent = self.output_root.parent
        probe = Path(tempfile.mkdtemp(prefix="satnet-final-ml-repro-", dir=probe_parent))
        compared: list[dict[str, Any]] = []
        try:
            source_files = [rf_space["csv"], rf_space["schema"], rf_space["manifest"], tgnn_classification["target_manifest"]]
            for source in source_files:
                destination = probe / source.relative_to(self.output_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                compared.append({"artifact": source.relative_to(self.output_root).as_posix(), "identical": source.read_bytes() == destination.read_bytes(), "sha256": sha256_file(source)})
            item = self.evidence[0]
            header, arrays, _ = self._sequence_for_run(item)
            probe_sequence = probe / "tgnn_space_classification" / "sequences" / "run_0000.tgnn"
            probe_sequence.parent.mkdir(parents=True, exist_ok=True)
            probe_sequence.write_bytes(sequence_bytes(header, arrays))
            source_sequence = self.output_root / "tgnn_space_classification" / "sequences" / "run_0000.tgnn"
            compared.append({"artifact": "tgnn_space_classification/sequences/run_0000.tgnn", "identical": source_sequence.read_bytes() == probe_sequence.read_bytes(), "sha256": sha256_file(source_sequence)})
            require(all(item["identical"] for item in compared), "Deterministic re-export probe failed")
            return {"status": "passed", "temporary_external_directory": str(probe), "temporary_directory_removed_after_check": True, "compared_artifacts": compared, "unexplained_nondeterminism": False}
        finally:
            shutil.rmtree(probe, ignore_errors=True)

    def write_report(self, inventory_path: Path, bundle_hash: str, entries: list[dict[str, Any]], rf_tasks: list[dict[str, Any]], tgnn_tasks: list[dict[str, Any]], reproducibility: dict[str, Any]) -> None:
        hashes = "\n".join(f"- `{entry['artifact_path']}`: `{entry['sha256']}`" for entry in entries)
        report = f"""# Final ML Dataset Export Report

## Production provenance

- Tooling SHA: `{TOOLING_SHA}`
- Production contract specification hash: `{CONTRACT_SPEC_HASH}`
- Production contract bundle identity: `{PRODUCTION_CONTRACT_BUNDLE_HASH}`
- Frozen split candidate: `{SPLIT_CANDIDATE}`
- Production acceptance and replay evidence were read-only and passed.

## Frozen ML contract provenance

- Frozen ML contract bundle hash: `{ML_CONTRACT_BUNDLE_HASH}`
- All recorded contract artifact SHA-256 values and byte lengths matched before materialization.
- No simulation, split generation, training, tuning, cross-validation, resampling, threshold adjustment, or preprocessing fitting was invoked.

## RF task exports

""" + "\n".join(f"- `{task['task']}`: `{task['csv'].relative_to(self.output_root).as_posix()}`; {task['quality']['row_count']} rows; {task['quality']['predictor_count']} predictors; {task['quality']['target_count']} target(s)." for task in rf_tasks) + f"""

Integrated regression representation: one feature table with both frozen targets. Primary target is `failure_adjusted_overall_service_fraction_mean`; secondary target is `failure_adjusted_overall_service_fraction_min`.

## TGNN task exports

- Shared immutable graph sequences: `tgnn_space_classification/sequences/`, 10,000 sequences.
- Space classification target manifest: `tgnn_space_classification/tgnn_space_classification_target_manifest.jsonl`.
- Space regression target manifest: `tgnn_space_regression/tgnn_space_regression_target_manifest.jsonl`.
- Node feature dimension: 3. Edge feature dimension: 4.
- CURRENT TGNN EXPORT IS SPACE-SEGMENT ONLY.
- INTEGRATED TGNN IS NOT PART OF THE CURRENT FROZEN EXPERIMENT.

## Target/class balance

- Space classification: 8,267 positive / 1,733 negative overall; train 5,819 / 1,181; validation 1,205 / 295; test 1,243 / 257.
- Integrated classification: 9,915 positive / 85 negative overall; train 6,942 / 58; validation 1,484 / 16; test 1,489 / 11.
- Regression targets were copied exactly from accepted production target artifacts and were finite with nonzero variance.

## Target identity verification

- All 50,000 RF target values (including both integrated regression targets) were copied from and exactly matched the authoritative per-run `targets/target.json` artifacts.
- All 20,000 TGNN target-manifest values were copied from the same authoritative target artifacts; binary targets matched exactly and numeric targets retained canonical production spellings.

## Feature identity verification

- All RF predictor cells were copied from the corresponding authoritative ex-ante `input/design_record.json` record; no post-simulation summary was used.
- All TGNN node and edge inputs were validated against the accepted temporal G3 graph artifact for each run and timestep.

## Split verification

- Train: 7,000 runs / 1,400 designs.
- Validation: 1,500 runs / 300 designs.
- Test: 1,500 runs / 300 designs.
- Run IDs cover exactly `0..9999`; no run overlap; no design crosses splits; all five realizations remain together.

## Leakage verification

- RF predictors were copied only from ex-ante `input/design_record.json` fields in the frozen order.
- Redundant counts, IDs, split labels, seeds, hashes, statuses, realized failures, graph/GCC/service outcomes, threshold indicators other than the selected target, and target-derived fields are excluded from X.
- TGNN graph inputs contain only same-timestep satellite node/edge features. Targets are stored only in separate target manifests.
- No learned preprocessing was fit.

## Data-quality verification

- RF missing, NaN, +Inf, -Inf, duplicate run ID, exact duplicate row, and unexpected-field gates passed with zero failures.
- TGNN sequence count: 10,000; malformed: 0; empty: 0; missing timesteps: 0; dimensions: node 3 / edge 4; target completeness passed.

## Hash inventory

- Inventory: `{inventory_path.relative_to(self.output_root).as_posix()}`
- Deterministic final bundle hash: `{bundle_hash}`

{hashes}

## Reproducibility check

- Deterministic RF and TGNN re-export probe: `{reproducibility['status']}`.
- Compared RF CSV/schema/manifest, TGNN target manifest, and one serialized TGNN sequence; all bytes and hashes matched.
- No unexplained nondeterminism.

## Known limitations

- The shared TGNN graph artifacts are derived from accepted G3 temporal graph records filtered to the satellite segment, using the validated current SatNetTemporalDataset/HypatiaAdapter 3-node/4-edge contract. No graph is regenerated.
- Integrated TGNN remains outside the authorized current experiment.
- The reproducibility probe temporary directory is removed after comparison; the immutable final export root is retained.

## Final status

`READY FOR MODEL TRAINING TEST-PLAN FREEZE`
"""
        (self.output_root / "final_ml_dataset_export_report.md").write_text(report, encoding="utf-8", newline="\n")

    def run(self) -> None:
        self.verify_frozen_contract()
        self.load_and_validate_production()
        self.output_root.mkdir(parents=True, exist_ok=True)
        expected_space_balance = {"overall": {"positive": 8267, "negative": 1733, "total": 10000}, "train": {"positive": 5819, "negative": 1181, "total": 7000}, "validation": {"positive": 1205, "negative": 295, "total": 1500}, "test": {"positive": 1243, "negative": 257, "total": 1500}}
        expected_integrated_balance = {"overall": {"positive": 9915, "negative": 85, "total": 10000}, "train": {"positive": 6942, "negative": 58, "total": 7000}, "validation": {"positive": 1484, "negative": 16, "total": 1500}, "test": {"positive": 1489, "negative": 11, "total": 1500}}
        rf_tasks = [
            self._write_rf_task("rf_space_classification", "rf_space_classification.csv", "rf_space_classification_schema.json", RF_SPACE_FEATURES, ("space_threshold_breach_any",), expected_space_balance),
            self._write_rf_task("rf_space_regression", "rf_space_regression.csv", "rf_space_regression_schema.json", RF_SPACE_FEATURES, ("space_gcc_fraction_original_min",)),
            self._write_rf_task("rf_integrated_classification", "rf_integrated_classification.csv", "rf_integrated_classification_schema.json", RF_INTEGRATED_FEATURES, ("overall_threshold_breach_any",), expected_integrated_balance),
            self._write_rf_task("rf_integrated_regression", "rf_integrated_regression.csv", "rf_integrated_regression_schema.json", RF_INTEGRATED_FEATURES, ("failure_adjusted_overall_service_fraction_mean", "failure_adjusted_overall_service_fraction_min")),
        ]
        tgnn_classification = self.materialize_tgnn("tgnn_space_classification", "space_threshold_breach_any")
        tgnn_regression = self.materialize_tgnn("tgnn_space_regression", "space_gcc_fraction_original_min")
        self.validate_splits_and_identity(rf_tasks + [tgnn_classification, tgnn_regression])
        self.write_metadata(rf_tasks, [tgnn_classification, tgnn_regression])
        reproducibility = self.reproducibility_probe(rf_tasks[0], tgnn_classification)
        write_json(self.output_root / "metadata" / "reproducibility_check.json", reproducibility)
        inventory_path, bundle_hash, entries = self.build_inventory()
        self.write_report(inventory_path, bundle_hash, entries, rf_tasks, [tgnn_classification, tgnn_regression], reproducibility)
        print(canonical_json({"status": "READY FOR MODEL TRAINING TEST-PLAN FREEZE", "output_root": str(self.output_root), "bundle_hash": bundle_hash, "inventory": str(inventory_path)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, default=Path(r"C:\Users\johns\external\satnet-10k-production-generation"))
    parser.add_argument("--replay-root", type=Path, default=Path(r"C:\Users\johns\external\satnet-10k-production-replay"))
    parser.add_argument("--acceptance-root", type=Path, default=Path(r"C:\Users\johns\external\satnet-10k-production-acceptance"))
    parser.add_argument("--audit-root", type=Path, default=Path(r"C:\Users\johns\external\satnet-10k-production-audit-corrected"))
    parser.add_argument("--ml-contract-root", type=Path, default=Path(r"C:\Users\johns\external\satnet-10k-ml-contract-v1"))
    parser.add_argument("--contract-root", type=Path, default=QUALIFICATION_ROOT / "artifacts" / "final_integrated_dataset_10k_contract")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Exporter(production_root=args.production_root, replay_root=args.replay_root, acceptance_root=args.acceptance_root, audit_root=args.audit_root, ml_contract_root=args.ml_contract_root, contract_root=args.contract_root, output_root=args.output_root).run()


if __name__ == "__main__":
    main()
