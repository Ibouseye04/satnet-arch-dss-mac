"""SATNET Phase 4B frozen real-data external model inference.

This module performs inference only against the immutable Phase 4A package and
frozen synthetic-trained model artifacts. It intentionally contains no training,
fit, optimizer, threshold-tuning, resampling, or model-selection operations.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

EXTERNAL_ROOT = Path(r"C:\Users\johns\external\satnet-real-external-validation-v1")
INFERENCE_ROOT = Path(r"C:\Users\johns\external\satnet-real-external-inference-v1")
MODEL_ROOT = Path(r"C:\Users\johns\external\satnet-10k-model-training-v1")
FINAL_ROBUSTNESS_ROOT = MODEL_ROOT / "final_robustness"
HELDOUT_ROOT = MODEL_ROOT / "heldout_test_v1"
SYNTHETIC_DATASET_ROOT = Path(r"C:\Users\johns\external\satnet-10k-final-ml-datasets-v1-final")
TRAINING_WORKTREE = Path(r"C:\Users\johns\external\satnet-10k-training-worktree-v1")
REPO_ROOT = Path(r"C:\Users\johns\Developer\satnet-arch-dss-mac")
PHASE4A_PATH = REPO_ROOT / "src/satnet/experiments/external_validation/phase4a.py"
HYPATIA_ADAPTER_PATH = REPO_ROOT / "src/satnet/network/hypatia_adapter.py"

EXPECTED_EXTERNAL_BUNDLE_SHA = "9abede2f83adbfc96e8267805e3253d6dbcd6f5240d675729d5677aa053b7e28"
EXPECTED_ADAPTER_SHA = "83ebc1e4d31b6aa8ff9b1574e3a39b4b9716d1b182ebe2fd5e7e961ed4ef4e23"
EXPECTED_TLE_2025_SHA = "9e9339cdbfc536cb91a1b5a26c898f193fdb683202f1cb1ceb7cee8bda81f64e"
EXPECTED_SYNTHETIC_HELDOUT_BUNDLE_SHA = "4359650814b960283b711e49e29301edd98b4966933bad0fb2aff532e2384551"
SYNTHETIC_DATASET_SHA = "38dacd66432bfa660410ff9cab7f181a53151120e8102dc40df57016f78bbfc3"
TRAINING_PLAN_SHA = "e14ec5e5b2221aa2c9aa187a2baacc517e676f9dc748612c206eb1dc43788a0a"
IMPLEMENTATION_SHA = "e55dba59e83864d2dd11fa47482bd4ec2bdd797a"
SEEDS = (42, 123, 456, 789, 2026)
PRIMARY_SEED = 42
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_REPLICATES = 2000
EPISODE_COUNT = 300
TIMESTEPS = 11
RF_FEATURES = (
    "num_planes",
    "sats_per_plane",
    "altitude_km",
    "inclination_deg",
    "satellite_node_failure_probability",
    "satellite_edge_failure_probability",
)
NODE_FEATURES = ("plane_idx_normalized", "sat_in_plane_normalized", "node_exists_constant")
EDGE_FEATURES = (
    "distance_km_scaled_10000",
    "margin_db_scaled_100",
    "link_type_code_scaled_2",
    "link_mode_binary",
)
TASKS = (
    "rf_space_classification",
    "tgnn_space_classification",
    "rf_space_regression",
    "tgnn_space_regression",
)
TASK_TYPE = {task: ("classification" if "classification" in task else "regression") for task in TASKS}
MODEL_FAMILY = {task: ("RF" if task.startswith("rf_") else "TGNN") for task in TASKS}
TARGET = {
    "rf_space_classification": "space_threshold_breach_any",
    "tgnn_space_classification": "space_threshold_breach_any",
    "rf_space_regression": "space_gcc_fraction_original_min",
    "tgnn_space_regression": "space_gcc_fraction_original_min",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_hash_excluding(root: Path, excluded_names: set[str], trailing_separator: bool = True) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in excluded_names):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if trailing_separator:
            digest.update(b"\0")
    return digest.hexdigest()


def bundle_hash(root: Path, excluded_name: str) -> str:
    return bundle_hash_excluding(root, {excluded_name})


def adapter_sha256() -> str:
    digest = hashlib.sha256()
    for path in (PHASE4A_PATH, HYPATIA_ADAPTER_PATH):
        digest.update(path.as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def finite(value: Any, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite value at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite(item, f"{location}[{index}]")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def verify_json_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{label} hash mismatch: {observed} != {expected}")


def verify_source_manifest() -> dict[str, Any]:
    manifest_path = EXTERNAL_ROOT / "contracts/external_source_manifest.json"
    manifest = load_json(manifest_path)
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("external source manifest has no sources")
    for entry in sources:
        source_file = EXTERNAL_ROOT / "raw" / entry["source_repository"].split("/")[-1] / entry["file"]
        if not source_file.is_file():
            raise RuntimeError(f"missing frozen raw source: {source_file}")
        if len(str(entry.get("sha256", ""))) != 64 or any(character not in "0123456789abcdef" for character in entry["sha256"]):
            raise RuntimeError(f"source manifest SHA is not a full lowercase SHA-256: {entry}")
        if source_file.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"source byte count mismatch: {source_file}")
        if sha256_file(source_file) != entry["sha256"]:
            raise RuntimeError(f"source hash mismatch: {source_file}")
    tle = next(entry for entry in sources if entry["file"] == "data/tle_2025.parquet")
    if tle["sha256"] != EXPECTED_TLE_2025_SHA:
        raise RuntimeError("manifest tle_2025.parquet SHA differs from authorization")
    return {"manifest_sha256": sha256_file(manifest_path), "sources": len(sources), "tle_2025_sha256": tle["sha256"]}


def verify_external_dataset() -> dict[str, Any]:
    rf_path = EXTERNAL_ROOT / "episodes/external_rf_dataset.csv"
    fields, rows = read_csv(rf_path)
    expected_fields = [
        "episode_id", "episode_timestamp", *RF_FEATURES,
        "space_gcc_fraction_original_min", "space_threshold_breach_any",
        "tgnn_sequence", "tle_age_max_seconds", "node_failure_numerator",
        "node_failure_denominator", "edge_failure_numerator", "edge_failure_denominator", "source_norad_ids",
    ]
    if fields != expected_fields:
        raise RuntimeError(f"external RF contract order mismatch: {fields}")
    if len(rows) != EPISODE_COUNT or [int(row["episode_id"]) for row in rows] != list(range(EPISODE_COUNT)):
        raise RuntimeError("external episode IDs/count are not exactly 0..299")
    classification = [int(row["space_threshold_breach_any"]) for row in rows]
    if classification.count(1) != 300 or classification.count(0) != 0:
        raise RuntimeError("external classification balance differs from frozen Phase 4A")
    for row in rows:
        finite({key: float(row[key]) for key in (*RF_FEATURES, "space_gcc_fraction_original_min")}, row["episode_id"])
        if not 0.0 <= float(row["space_gcc_fraction_original_min"]) <= 1.0:
            raise RuntimeError(f"external target outside [0,1]: {row['episode_id']}")
        sequence = EXTERNAL_ROOT / row["tgnn_sequence"]
        if not sequence.is_file():
            raise RuntimeError(f"missing TGNN sequence: {sequence}")
        payload = load_json(sequence)
        if payload.get("episode_id") != int(row["episode_id"]):
            raise RuntimeError(f"TGNN episode identity mismatch: {sequence}")
        if tuple(payload.get("node_feature_order", ())) != NODE_FEATURES or tuple(payload.get("edge_feature_order", ())) != EDGE_FEATURES:
            raise RuntimeError(f"TGNN feature contract mismatch: {sequence}")
        snapshots = payload.get("snapshots")
        if not isinstance(snapshots, list) or len(snapshots) != TIMESTEPS:
            raise RuntimeError(f"TGNN snapshot count mismatch: {sequence}")
        for snapshot in snapshots:
            for node in snapshot.get("nodes", []):
                if len(node.get("features", [])) != 3:
                    raise RuntimeError(f"TGNN node dimension mismatch: {sequence}")
                finite(node["features"], f"{sequence}:node")
            for edge in snapshot.get("edges", []):
                values = [edge[name] for name in EDGE_FEATURES]
                if len(values) != 4:
                    raise RuntimeError(f"TGNN edge dimension mismatch: {sequence}")
                finite(values, f"{sequence}:edge")
    for path, target in ((EXTERNAL_ROOT / "episodes/rf_space_classification.csv", "space_threshold_breach_any"), (EXTERNAL_ROOT / "episodes/rf_space_regression.csv", "space_gcc_fraction_original_min")):
        fields, rows_for_task = read_csv(path)
        if len(rows_for_task) != EPISODE_COUNT or fields[:2] != ["episode_id", "episode_timestamp"] or fields[2:8] != list(RF_FEATURES) or fields[-1] != target:
            raise RuntimeError(f"RF dataset contract mismatch: {path}")
    target_manifest = EXTERNAL_ROOT / "episodes/tgnn_space_target_manifest.jsonl"
    target_rows = [json.loads(line) for line in target_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(target_rows) != EPISODE_COUNT or [int(row["episode_id"]) for row in target_rows] != list(range(EPISODE_COUNT)):
        raise RuntimeError("TGNN target manifest episode IDs/count mismatch")
    return {"episode_count": EPISODE_COUNT, "classification_positive": 300, "classification_negative": 0, "timesteps_per_episode": TIMESTEPS, "rf_feature_order": list(RF_FEATURES), "tgnn_node_feature_order": list(NODE_FEATURES), "tgnn_edge_feature_order": list(EDGE_FEATURES)}


def model_artifact(task: str, seed: int) -> tuple[Path, dict[str, Any]]:
    manifest_path = FINAL_ROBUSTNESS_ROOT / task / f"seed_{seed}" / "final_manifest.json"
    manifest = load_json(manifest_path)
    if task.startswith("rf_"):
        path = FINAL_ROBUSTNESS_ROOT / task / f"seed_{seed}" / "final_model.joblib"
        expected = manifest.get("model_sha256")
    else:
        path = FINAL_ROBUSTNESS_ROOT / task / f"seed_{seed}" / "best_validation_checkpoint.pt"
        expected = manifest.get("checkpoint_sha256")
    if manifest.get("status") != "completed" or manifest.get("task") != task or manifest.get("seed") != seed or manifest.get("test_accessed") is not False:
        raise RuntimeError(f"invalid frozen model manifest: {task}/seed_{seed}")
    if manifest.get("dataset_bundle_hash") != SYNTHETIC_DATASET_SHA or manifest.get("training_plan_hash") != TRAINING_PLAN_SHA or manifest.get("code_sha") != IMPLEMENTATION_SHA:
        raise RuntimeError(f"frozen provenance mismatch: {task}/seed_{seed}")
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"frozen model/checkpoint SHA mismatch: {task}/seed_{seed}")
    return path, {"task": task, "seed": seed, "path": str(path), "sha256": expected, "kind": path.name}


def verify_models() -> list[dict[str, Any]]:
    artifacts = [model_artifact(task, seed)[1] for task in TASKS for seed in SEEDS]
    if len(artifacts) != 20:
        raise RuntimeError("expected exactly 20 authorized model artifacts")
    inventory = load_json(FINAL_ROBUSTNESS_ROOT / "final_robustness_inventory.json")
    entries = {(item["path"], item["sha256"]) for item in inventory.get("artifacts", [])}
    for artifact in artifacts:
        relative = Path(artifact["path"]).relative_to(FINAL_ROBUSTNESS_ROOT).as_posix()
        if (relative, artifact["sha256"]) not in entries:
            raise RuntimeError(f"model absent from frozen robustness inventory: {relative}")
    return artifacts


def verify_synthetic_heldout_bundle() -> str:
    observed = bundle_hash_excluding(HELDOUT_ROOT, {"heldout_test_inventory.json"}, trailing_separator=False)
    inventory = load_json(HELDOUT_ROOT / "heldout_test_inventory.json")
    if observed != EXPECTED_SYNTHETIC_HELDOUT_BUNDLE_SHA or inventory.get("bundle_sha256") != EXPECTED_SYNTHETIC_HELDOUT_BUNDLE_SHA:
        raise RuntimeError(f"synthetic held-out bundle mismatch: {observed}")
    return observed


def preflight() -> dict[str, Any]:
    observed_bundle = bundle_hash(EXTERNAL_ROOT, "external_validation_inventory.json")
    if observed_bundle != EXPECTED_EXTERNAL_BUNDLE_SHA:
        raise RuntimeError(f"external bundle mismatch: {observed_bundle}")
    observed_adapter = adapter_sha256()
    if observed_adapter != EXPECTED_ADAPTER_SHA:
        raise RuntimeError(f"adapter mismatch: {observed_adapter}")
    tle_info = verify_source_manifest()
    dataset_info = verify_external_dataset()
    model_info = verify_models()
    heldout_sha = verify_synthetic_heldout_bundle()
    inventory = load_json(EXTERNAL_ROOT / "external_validation_inventory.json")
    specification = load_json(EXTERNAL_ROOT / "contracts/external_adapter_specification.json")
    if inventory.get("adapter_sha256") != EXPECTED_ADAPTER_SHA or specification.get("implementation_sha256") != EXPECTED_ADAPTER_SHA:
        raise RuntimeError("frozen Phase 4A adapter provenance mismatch")
    if inventory.get("external_pre_inference_bundle_sha256") != EXPECTED_EXTERNAL_BUNDLE_SHA or inventory.get("model_inference_performed") is not False:
        raise RuntimeError("frozen Phase 4A inventory state mismatch")
    return {"status": "PASS", "external_bundle_sha256": observed_bundle, "adapter_sha256": observed_adapter, "tle_2025_sha256": tle_info["tle_2025_sha256"], "source_manifest_sha256": tle_info["manifest_sha256"], "dataset": dataset_info, "model_artifacts": model_info, "synthetic_heldout_bundle_sha256": heldout_sha, "training_performed": False, "adapter_modified": False, "external_data_modified": False}


def write_authorization(preflight_result: dict[str, Any]) -> tuple[dict[str, Any], str]:
    path = INFERENCE_ROOT / "authorization/external_inference_authorization.json"
    if path.is_file():
        authorization = load_json(path)
        authorization_sha = sha256_file(path)
        if authorization.get("expected_adapter_sha256") != EXPECTED_ADAPTER_SHA or authorization.get("external_bundle_sha256") != EXPECTED_EXTERNAL_BUNDLE_SHA:
            raise RuntimeError("existing authorization does not match corrected frozen contract")
        return authorization, authorization_sha
    authorization = {"schema_version": "satnet.phase4b.external_inference_authorization.v1", "status": "FROZEN_BEFORE_FIRST_MODEL_PREDICTION", "expected_adapter_sha256": EXPECTED_ADAPTER_SHA, "external_bundle_sha256": EXPECTED_EXTERNAL_BUNDLE_SHA, "tle_2025_sha256": EXPECTED_TLE_2025_SHA, "synthetic_heldout_bundle_sha256": EXPECTED_SYNTHETIC_HELDOUT_BUNDLE_SHA, "synthetic_dataset_sha256": SYNTHETIC_DATASET_SHA, "training_plan_sha256": TRAINING_PLAN_SHA, "implementation_sha": IMPLEMENTATION_SHA, "tasks": list(TASKS), "seeds": list(SEEDS), "primary_reporting_seed": PRIMARY_SEED, "episode_count": EPISODE_COUNT, "timesteps_per_episode": TIMESTEPS, "classification_target": TARGET["rf_space_classification"], "regression_target": TARGET["rf_space_regression"], "classification_single_class": {"positive": 300, "negative": 0, "descriptive_only": True}, "model_artifacts": preflight_result["model_artifacts"], "bootstrap": {"unit": "episode_id", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "confidence_interval": "95% percentile", "comparison": "rf_space_regression seed 42 - tgnn_space_regression seed 42"}, "training_performed": False, "retraining_performed": False, "fine_tuning_performed": False, "threshold_tuning_performed": False, "model_selection_performed": False, "episode_resampling_or_redesign": False}
    atomic_json(path, authorization)
    authorization_sha = sha256_file(path)
    (path.parent / "external_inference_authorization.sha256").write_text(f"{authorization_sha}  {path.name}\n", encoding="ascii")
    return authorization, authorization_sha


def load_external_records() -> list[dict[str, Any]]:
    _, rows = read_csv(EXTERNAL_ROOT / "episodes/rf_space_regression.csv")
    return [{"episode_id": int(row["episode_id"]), "timestamp": row["episode_timestamp"], "num_planes": int(row["num_planes"]), "sats_per_plane": int(row["sats_per_plane"]), "features": [float(row[name]) for name in RF_FEATURES], "regression_target": float(row["space_gcc_fraction_original_min"]), "classification_target": 1} for row in rows]


def metrics_classification(target: np.ndarray, prediction: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    positive = int(np.sum(target == 1)); negative = int(np.sum(target == 0)); tp = int(np.sum((target == 1) & (prediction == 1))); fn = int(np.sum((target == 1) & (prediction == 0)))
    score = np.asarray(score, dtype=float)
    return {"positive_class_recall_sensitivity": float(tp / positive) if positive else None, "false_negative_count": fn, "false_negative_rate": float(fn / positive) if positive else None, "true_positive_count": tp, "predicted_positive_count": int(np.sum(prediction == 1)), "predicted_negative_count": int(np.sum(prediction == 0)), "accuracy": float(np.mean(prediction == target)), "positive_class_score_probability_mean": float(np.mean(score)), "positive_class_score_probability_median": float(np.median(score)), "positive_class_score_probability_standard_deviation": float(np.std(score)), "positive_class_score_probability_min": float(np.min(score)), "positive_class_score_probability_max": float(np.max(score)), "observed_class_confusion_accounting": [[int(np.sum((target == 1) & (prediction == 1)))]], "specificity": None, "balanced_accuracy_two_class": None, "roc_auc": None, "negative_class_precision_recall_f1": None, "two_class_confusion_matrix_interpretation": "NOT MEANINGFUL: frozen external classification data contain one observed class only", "descriptive_only": True}


def metrics_regression(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - target; absolute = np.abs(error); centered = target - np.mean(target); denominator = float(np.sum(centered * centered))
    return {"mae": float(np.mean(absolute)), "rmse": float(np.sqrt(np.mean(error * error))), "r2": float(1.0 - np.sum(error * error) / denominator) if denominator else None, "median_absolute_error": float(np.median(absolute)), "maximum_absolute_error": float(np.max(absolute)), "target_mean": float(np.mean(target)), "target_median": float(np.median(target)), "target_standard_deviation": float(np.std(target)), "prediction_mean": float(np.mean(prediction)), "prediction_median": float(np.median(prediction)), "prediction_standard_deviation": float(np.std(prediction)), "prediction_min": float(np.min(prediction)), "prediction_max": float(np.max(prediction)), "predictions_below_zero": int(np.sum(prediction < 0)), "predictions_above_one": int(np.sum(prediction > 1))}


def evaluation_paths(task: str, seed: int) -> tuple[Path, Path, Path]:
    root = INFERENCE_ROOT / task / f"seed_{seed}"
    return root / "predictions.csv", root / "metrics.json", root / "evaluation_manifest.json"


def completed_evaluation(task: str, seed: int, model_sha: str, authorization_sha: str) -> bool:
    prediction_path, metrics_path, manifest_path = evaluation_paths(task, seed)
    if not all(path.is_file() for path in (prediction_path, metrics_path, manifest_path)):
        return False
    manifest = load_json(manifest_path)
    if not all((manifest.get("status") == "completed", manifest.get("task") == task, manifest.get("seed") == seed, manifest.get("model_sha256") == model_sha, manifest.get("external_bundle_sha256") == EXPECTED_EXTERNAL_BUNDLE_SHA, manifest.get("authorization_sha256") == authorization_sha, manifest.get("predictions_sha256") == sha256_file(prediction_path), manifest.get("metrics_sha256") == sha256_file(metrics_path))):
        return False
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in handle) - 1 == EPISODE_COUNT


def mark_running(task: str, seed: int, model_sha: str, authorization_sha: str) -> None:
    _, _, manifest_path = evaluation_paths(task, seed)
    atomic_json(manifest_path, {"schema_version": "satnet.phase4b.evaluation_manifest.v1", "status": "running", "task": task, "model_family": MODEL_FAMILY[task], "seed": seed, "model_sha256": model_sha, "external_bundle_sha256": EXPECTED_EXTERNAL_BUNDLE_SHA, "authorization_sha256": authorization_sha, "training_performed": False, "started_at": datetime.now(timezone.utc).isoformat()})


def persist_evaluation(task: str, seed: int, model_sha: str, authorization_sha: str, records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    prediction_path, metrics_path, manifest_path = evaluation_paths(task, seed)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    temporary = prediction_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    os.replace(temporary, prediction_path)
    atomic_json(metrics_path, {"task": task, "model_family": MODEL_FAMILY[task], "seed": seed, **metrics})
    manifest = {"schema_version": "satnet.phase4b.evaluation_manifest.v1", "status": "completed", "task": task, "model_family": MODEL_FAMILY[task], "seed": seed, "model_sha256": model_sha, "external_bundle_sha256": EXPECTED_EXTERNAL_BUNDLE_SHA, "authorization_sha256": authorization_sha, "row_count": len(records), "predictions_sha256": sha256_file(prediction_path), "metrics_sha256": sha256_file(metrics_path), "training_performed": False, "checkpoint_modified": False, "completed_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(manifest_path, manifest)


def json_sequence_to_data(sequence_path: Path) -> list[Any]:
    import torch
    from torch_geometric.data import Data
    payload = load_json(sequence_path)
    result = []
    for snapshot in payload["snapshots"]:
        node_by_id = {int(node["node_id"]): node for node in snapshot["nodes"]}
        nodes = sorted(node_by_id)
        index = {node_id: position for position, node_id in enumerate(nodes)}
        edge_index = [[index[int(edge["source"])] for edge in snapshot["edges"]], [index[int(edge["target"])] for edge in snapshot["edges"]]]
        edge_attr = [[float(edge[name]) for name in EDGE_FEATURES] for edge in snapshot["edges"]]
        data = Data(x=torch.tensor([node_by_id[node_id]["features"] for node_id in nodes], dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long).reshape(2, -1), edge_attr=torch.tensor(edge_attr, dtype=torch.float32).reshape(-1, 4))
        data.edge_weight = data.edge_attr[:, 0]
        result.append(data)
    return result


def run_rf(task: str, seed: int, model_path: Path, model_sha: str, authorization_sha: str, rows: list[dict[str, Any]]) -> None:
    import joblib
    estimator = joblib.load(model_path)
    x = np.asarray([row["features"] for row in rows], dtype=float)
    if TASK_TYPE[task] == "classification":
        prediction = np.asarray(estimator.predict(x), dtype=int)
        probabilities = np.asarray(estimator.predict_proba(x), dtype=float)
        positive_index = list(estimator.classes_).index(1)
        score = probabilities[:, positive_index]
        target = np.ones(EPISODE_COUNT, dtype=int)
        output = [{"episode_id": row["episode_id"], "timestamp": row["timestamp"], "target": 1, "prediction": int(prediction[index]), "positive_class_score_probability": float(score[index]), "seed": seed, "model_family": "RF"} for index, row in enumerate(rows)]
        metrics = metrics_classification(target, prediction, score)
    else:
        prediction = np.asarray(estimator.predict(x), dtype=float)
        target = np.asarray([row["regression_target"] for row in rows], dtype=float)
        output = [{"episode_id": row["episode_id"], "timestamp": row["timestamp"], "num_planes": row["num_planes"], "sats_per_plane": row["sats_per_plane"], "target": float(target[index]), "prediction": float(prediction[index]), "error": float(prediction[index] - target[index]), "absolute_error": float(abs(prediction[index] - target[index])), "seed": seed, "model_family": "RF"} for index, row in enumerate(rows)]
        metrics = metrics_regression(target, prediction)
    finite(metrics); persist_evaluation(task, seed, model_sha, authorization_sha, output, metrics)


def run_tgnn(task: str, seed: int, model_path: Path, model_manifest: dict[str, Any], model_sha: str, authorization_sha: str, rows: list[dict[str, Any]]) -> None:
    import torch
    sys.path.insert(0, str(TRAINING_WORKTREE / "src"))
    from satnet.models.gnn_model import SatelliteGNN
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = model_manifest["configuration"]
    task_type = TASK_TYPE[task]
    model = SatelliteGNN(node_features=3, hidden_channels=int(config["hidden_dim"]), out_channels=2 if task_type == "classification" else 1, task_type=task_type, cheb_k=int(config["cheb_k"]))
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval(); torch.set_num_threads(8)
    predictions: list[float] = []; scores: list[float] = []
    with torch.no_grad():
        for row in rows:
            sequence_path = EXTERNAL_ROOT / "episodes" / "tgnn_sequences" / f"episode_{row['episode_id']:04d}.json"
            data_list = json_sequence_to_data(sequence_path)
            output = model(data_list)
            if task_type == "classification":
                probability = torch.softmax(output, dim=1).cpu().numpy()[0]
                scores.append(float(probability[1])); predictions.append(float(np.argmax(probability)))
            else:
                predictions.append(float(output.squeeze().item()))
    prediction = np.asarray(predictions, dtype=float)
    if task_type == "classification":
        target = np.ones(EPISODE_COUNT, dtype=int); predicted_int = prediction.astype(int)
        output = [{"episode_id": row["episode_id"], "timestamp": row["timestamp"], "target": 1, "prediction": int(predicted_int[index]), "positive_class_score_probability": float(scores[index]), "seed": seed, "model_family": "TGNN"} for index, row in enumerate(rows)]
        metrics = metrics_classification(target, predicted_int, np.asarray(scores))
    else:
        target = np.asarray([row["regression_target"] for row in rows], dtype=float)
        output = [{"episode_id": row["episode_id"], "timestamp": row["timestamp"], "num_planes": row["num_planes"], "sats_per_plane": row["sats_per_plane"], "target": float(target[index]), "prediction": float(prediction[index]), "error": float(prediction[index] - target[index]), "absolute_error": float(abs(prediction[index] - target[index])), "seed": seed, "model_family": "TGNN"} for index, row in enumerate(rows)]
        metrics = metrics_regression(target, prediction)
    finite(metrics); persist_evaluation(task, seed, model_sha, authorization_sha, output, metrics)


def load_metric(task: str, seed: int) -> dict[str, Any]:
    return load_json(INFERENCE_ROOT / task / f"seed_{seed}/metrics.json")


def baselines(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _, train_rows = read_csv(SYNTHETIC_DATASET_ROOT / "rf_space_regression/rf_space_regression.csv")
    train = [float(row["space_gcc_fraction_original_min"]) for row in train_rows if row["split"].lower() == "train"]
    if len(train) != 7000:
        raise RuntimeError("frozen synthetic TRAIN row count is not 7000")
    target = np.asarray([row["regression_target"] for row in rows], dtype=float)
    result = {}
    for name, value in (("synthetic_train_mean", float(np.mean(train))), ("synthetic_train_median", float(np.median(train)))):
        result[name] = {"predictor_value": value, **metrics_regression(target, np.full(EPISODE_COUNT, value))}
    atomic_json(INFERENCE_ROOT / "baselines/external_baselines.json", result)
    return result


def generalization_gaps() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in ("rf_space_regression", "tgnn_space_regression"):
        family = MODEL_FAMILY[task]
        values = []
        for seed in SEEDS:
            synthetic_metrics = load_json(HELDOUT_ROOT / task / f"seed_{seed}/metrics.json")
            external_metrics = load_metric(task, seed)
            synthetic_mae = float(synthetic_metrics["mae"])
            external_mae = float(external_metrics["mae"])
            values.append({"seed": seed, "synthetic_heldout_test_mae": synthetic_mae, "external_real_data_driven_mae": external_mae, "absolute_mae_change": external_mae - synthetic_mae, "relative_mae_change": (external_mae - synthetic_mae) / synthetic_mae})
        result[family] = values
    atomic_json(INFERENCE_ROOT / "comparisons/external_generalization_gap.json", result)
    return result


def seed_summary() -> dict[str, Any]:
    classification: dict[str, Any] = {}
    for task in ("rf_space_classification", "tgnn_space_classification"):
        metrics = [load_metric(task, seed) for seed in SEEDS]
        sensitivity = [float(item["positive_class_recall_sensitivity"]) for item in metrics]
        classification[MODEL_FAMILY[task]] = {"sensitivity_per_seed": dict(zip(SEEDS, sensitivity)), "false_negative_rate_per_seed": dict(zip(SEEDS, [float(item["false_negative_rate"]) for item in metrics])), "mean_sensitivity": float(np.mean(sensitivity)), "standard_deviation_sensitivity": float(np.std(sensitivity)), "min_sensitivity": float(np.min(sensitivity)), "max_sensitivity": float(np.max(sensitivity))}
    regression: dict[str, Any] = {}
    for task in ("rf_space_regression", "tgnn_space_regression"):
        maes = [float(load_metric(task, seed)["mae"]) for seed in SEEDS]
        regression[MODEL_FAMILY[task]] = {"mae_per_seed": dict(zip(SEEDS, maes)), "mean_mae": float(np.mean(maes)), "population_standard_deviation_mae": float(np.std(maes)), "min_mae": float(np.min(maes)), "max_mae": float(np.max(maes)), "seed_42_mae": float(load_metric(task, PRIMARY_SEED)["mae"])}
    result = {"classification_descriptive": classification, "regression": regression}
    atomic_json(INFERENCE_ROOT / "external_seed_summary.json", result)
    return result


def paired_bootstrap() -> dict[str, Any]:
    rf = np.asarray([float(row["absolute_error"]) for row in read_prediction_rows("rf_space_regression", PRIMARY_SEED)], dtype=float)
    tgnn = np.asarray([float(row["absolute_error"]) for row in read_prediction_rows("tgnn_space_regression", PRIMARY_SEED)], dtype=float)
    if len(rf) != EPISODE_COUNT or len(tgnn) != EPISODE_COUNT:
        raise RuntimeError("seed-42 regression prediction row count mismatch")
    observed_rf = float(np.mean(rf)); observed_tgnn = float(np.mean(tgnn)); observed_delta = observed_rf - observed_tgnn
    rng = np.random.default_rng(BOOTSTRAP_SEED); deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, EPISODE_COUNT, size=EPISODE_COUNT)
        deltas[index] = float(np.mean(rf[sample]) - np.mean(tgnn[sample]))
    result = {"observed_rf_mae": observed_rf, "observed_tgnn_mae": observed_tgnn, "observed_delta_mae_rf_minus_tgnn": observed_delta, "bootstrap_mean_delta": float(np.mean(deltas)), "bootstrap_sd": float(np.std(deltas)), "percentile_2_5": float(np.percentile(deltas, 2.5)), "percentile_97_5": float(np.percentile(deltas, 97.5)), "valid_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "sampling_unit": "episode_id", "paired": True, "confidence_interval": "95% percentile"}
    atomic_json(INFERENCE_ROOT / "external_regression_bootstrap.json", result)
    return result


def read_prediction_rows(task: str, seed: int) -> list[dict[str, str]]:
    _, rows = read_csv(evaluation_paths(task, seed)[0])
    return rows


def aggregate_results() -> tuple[dict[str, Any], dict[str, Any]]:
    classification = {MODEL_FAMILY[task]: {str(seed): load_metric(task, seed) for seed in SEEDS} for task in ("rf_space_classification", "tgnn_space_classification")}
    regression = {MODEL_FAMILY[task]: {str(seed): load_metric(task, seed) for seed in SEEDS} for task in ("rf_space_regression", "tgnn_space_regression")}
    atomic_json(INFERENCE_ROOT / "external_classification_results.json", classification); atomic_json(INFERENCE_ROOT / "external_regression_results.json", regression)
    return classification, regression


def write_report(preflight_result: dict[str, Any], authorization_sha: str, generalization: dict[str, Any], summary: dict[str, Any], bootstrap: dict[str, Any], started: float) -> None:
    rf42 = summary["regression"]["RF"]["seed_42_mae"]; tgnn42 = summary["regression"]["TGNN"]["seed_42_mae"]
    conclusion = "preserved" if rf42 > tgnn42 else "not preserved"
    prediction_hashes = []
    for task in TASKS:
        for seed in SEEDS:
            manifest = load_json(evaluation_paths(task, seed)[2])
            prediction_hashes.append({"task": task, "seed": seed, "predictions_sha256": manifest["predictions_sha256"], "metrics_sha256": manifest["metrics_sha256"]})
    lines = ["# SATNET Phase 4B Frozen Real-Data External Model Inference", "", "External evaluation on real-world Starlink orbital observations transformed through the frozen SATNET feature and network-construction methodology.", "", "## Status", "", "REAL-DATA EXTERNAL MODEL INFERENCE COMPLETE — EXTERNAL VALIDATION RESULTS FROZEN", "", f"Preflight: PASS; evaluations: 20/20; failures: 0; runtime_seconds: {time.monotonic() - started:.6f}", f"External bundle SHA-256: `{EXPECTED_EXTERNAL_BUNDLE_SHA}`", f"Adapter SHA-256: `{EXPECTED_ADAPTER_SHA}`", f"tle_2025.parquet SHA-256: `{EXPECTED_TLE_2025_SHA}`", f"Authorization SHA-256: `{authorization_sha}`", "", "## Frozen model/checkpoint hashes", "", "```json", json.dumps(preflight_result["model_artifacts"], indent=2, sort_keys=True), "```", "", "## Prediction and metric hashes", "", "```json", json.dumps(prediction_hashes, indent=2, sort_keys=True), "```", "", "## Classification", "", "The external classification set contains 300 positives and 0 negatives. Results are descriptive only and do not estimate two-class discrimination. Specificity, two-class balanced accuracy, ROC-AUC, and negative-class precision/recall/F1 are not estimable or meaningful.", json.dumps(summary["classification_descriptive"], indent=2, sort_keys=True), "", "## Regression", "", json.dumps(summary["regression"], indent=2, sort_keys=True), "", "## Synthetic baselines and generalization", "", json.dumps(load_json(INFERENCE_ROOT / "baselines/external_baselines.json"), indent=2, sort_keys=True), json.dumps(generalization, indent=2, sort_keys=True), "", "## Seed-42 paired RF versus TGNN", "", json.dumps(bootstrap, indent=2, sort_keys=True), f"The synthetic model-family conclusion is **{conclusion}** for seed 42 (RF MAE `{rf42}`, TGNN MAE `{tgnn42}`).", "", "## Distribution shift carried forward", "", "External edge failure probability mean: 0.7536; synthetic TRAIN mean: 0.1162.", "External altitude mean: 519.602 km; synthetic TRAIN mean: 739.108 km.", "External inclination mean: 53.169 degrees; synthetic TRAIN mean: 62.967 degrees.", "This is interpreted as out-of-distribution generalization. The experiment does not independently validate proprietary Starlink ISL routing or service telemetry.", "", "## Controls", "", "No training, retraining, fine-tuning, hyperparameter search, threshold tuning, model selection, episode resampling/redesign, adapter modification, or external-data modification occurred."]
    (INFERENCE_ROOT / "external_inference_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_inventory(preflight_result: dict[str, Any], authorization_sha: str, started: float, expected_bundle_sha: str) -> str:
    inventory_path = INFERENCE_ROOT / "external_inference_inventory.json"
    excluded = {inventory_path.name, "external_inference_summary.json"}
    observed = bundle_hash_excluding(INFERENCE_ROOT, excluded)
    if observed != expected_bundle_sha:
        raise RuntimeError("final bundle changed while writing inventory")
    entries = []
    for path in sorted(item for item in INFERENCE_ROOT.rglob("*") if item.is_file() and item.name not in excluded):
        entries.append({"path": path.relative_to(INFERENCE_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    inventory = {"schema_version": "satnet.phase4b.external_inference_inventory.v1", "status": "REAL-DATA EXTERNAL MODEL INFERENCE COMPLETE — EXTERNAL VALIDATION RESULTS FROZEN", "bundle_sha256": observed, "bundle_excluded_files": sorted(excluded), "artifact_count": len(entries), "artifacts": entries, "preflight": preflight_result, "authorization_sha256": authorization_sha, "driver_sha256": sha256_file(Path(__file__)), "runtime_seconds": time.monotonic() - started, "training_performed": False, "external_data_modified": False, "adapter_modified": False}
    atomic_json(inventory_path, inventory)
    return observed


def run() -> dict[str, Any]:
    started = time.monotonic()
    preflight_result = preflight()
    reconciliation = load_json(INFERENCE_ROOT / "authorization/external_adapter_sha_reconciliation.json")
    if reconciliation.get("authoritative_frozen_adapter_sha") != EXPECTED_ADAPTER_SHA:
        raise RuntimeError("adapter SHA reconciliation artifact does not authorize corrected SHA")
    authorization, authorization_sha = write_authorization(preflight_result)
    rows = load_external_records()
    progress_path = INFERENCE_ROOT / "external_inference_progress.json"
    progress = {"schema_version": "satnet.phase4b.external_inference_progress.v1", "status": "running", "evaluations": {}}
    atomic_json(progress_path, progress)
    for task in TASKS:
        for seed in SEEDS:
            model_path, model_manifest = model_artifact(task, seed)
            model_sha = model_manifest["sha256"]
            if completed_evaluation(task, seed, model_sha, authorization_sha):
                progress["evaluations"][f"{task}/seed_{seed}"] = "completed_skipped_verified"
                atomic_json(progress_path, progress)
                continue
            mark_running(task, seed, model_sha, authorization_sha)
            if MODEL_FAMILY[task] == "RF":
                run_rf(task, seed, model_path, model_sha, authorization_sha, rows)
            else:
                run_tgnn(task, seed, model_path, load_json(FINAL_ROBUSTNESS_ROOT / task / f"seed_{seed}/final_manifest.json"), model_sha, authorization_sha, rows)
            progress["evaluations"][f"{task}/seed_{seed}"] = "completed"
            atomic_json(progress_path, progress)
    progress["status"] = "completed"; atomic_json(progress_path, progress)
    baselines(rows); aggregate_results(); generalization = generalization_gaps(); summary = seed_summary(); bootstrap = paired_bootstrap()
    write_report(preflight_result, authorization_sha, generalization, summary, bootstrap, started)
    final_bundle_sha = bundle_hash_excluding(INFERENCE_ROOT, {"external_inference_inventory.json", "external_inference_summary.json"})
    result = {"preflight": preflight_result, "authorization_sha256": authorization_sha, "final_external_inference_bundle_sha256": final_bundle_sha, "runtime_seconds": time.monotonic() - started}
    atomic_json(INFERENCE_ROOT / "external_inference_summary.json", result)
    write_inventory(preflight_result, authorization_sha, started, final_bundle_sha)
    return result


if __name__ == "__main__":
    run()
    print("REAL-DATA EXTERNAL MODEL INFERENCE COMPLETE — EXTERNAL VALIDATION RESULTS FROZEN")
