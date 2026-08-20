from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(r"C:\Users\johns\external\satnet-10k-model-training-v1")
REPO = Path(r"C:\Users\johns\external\satnet-10k-training-worktree-v1")
SRC = REPO / "src"
DATASET_ROOT = Path(r"C:\Users\johns\external\satnet-10k-final-ml-datasets-v1-final")
PLAN_ROOT = Path(r"C:\Users\johns\external\satnet-10k-training-test-plan-v1")
FINAL_ROOT = ROOT / "final_robustness"
RECON_ROOT = ROOT / "robustness_reconciliation"
OUT_ROOT = ROOT / "heldout_test_v1"
AUTH_ROOT = OUT_ROOT / "authorization"
QUALIFIED_PYTHON = Path(r"C:\Users\johns\venvs\satnet-10k-qualification\Scripts\python.exe")

IMPLEMENTATION_SHA = "e55dba59e83864d2dd11fa47482bd4ec2bdd797a"
PRODUCTION_SHA = "d0515088cf3fca06a6aa2d47059269089dcb10a7"
DATASET_SHA = "38dacd66432bfa660410ff9cab7f181a53151120e8102dc40df57016f78bbfc3"
PLAN_SHA = "e14ec5e5b2221aa2c9aa187a2baacc517e676f9dc748612c206eb1dc43788a0a"
RF_SELECTION_SHA = "a1b7076197f123ecf62b0b79ddd092bdb66f0d7d8b94d6c113dc201744163911"
TGNN_SELECTION_SHA = "c85dfdcd3b6a62d741675ac522fda76bd3de7dd4c311a82bc1b12e079dab2bfb"
ROBUSTNESS_BUNDLE_SHA = "12d8db5f7b96f3b2e920911c0c877525153e2a23fab3b3d21321af5b79f3ab33"
RECON_BUNDLE_SHA = "cfecdbe07a5088f2f621b12a05ece9bef61bc4e24876eb0d85f01db9825ea75e"
EXPECTED_HEAD = IMPLEMENTATION_SHA
SEEDS = (42, 123, 456, 789, 2026)
PRIMARY_SEED = 42
BOOTSTRAP_SEED = 20260812
BOOTSTRAP_REPLICATES = 2000
TEST_RUNS = 1500
TEST_DESIGNS = 300
REALIZATIONS_PER_DESIGN = 5

RF_TASKS = (
    "rf_space_classification",
    "rf_space_regression",
    "rf_integrated_regression_mean",
    "rf_integrated_regression_min",
    "rf_integrated_classification",
)
TGNN_TASKS = ("tgnn_space_classification", "tgnn_space_regression")
TASKS = RF_TASKS + TGNN_TASKS
TASK_TYPE = {
    "rf_space_classification": "classification",
    "rf_space_regression": "regression",
    "rf_integrated_regression_mean": "regression",
    "rf_integrated_regression_min": "regression",
    "rf_integrated_classification": "classification",
    "tgnn_space_classification": "classification",
    "tgnn_space_regression": "regression",
}
TARGETS = {
    "rf_space_classification": "space_threshold_breach_any",
    "rf_space_regression": "space_gcc_fraction_original_min",
    "rf_integrated_regression_mean": "failure_adjusted_overall_service_fraction_mean",
    "rf_integrated_regression_min": "failure_adjusted_overall_service_fraction_min",
    "rf_integrated_classification": "overall_threshold_breach_any",
    "tgnn_space_classification": "space_threshold_breach_any",
    "tgnn_space_regression": "space_gcc_fraction_original_min",
}
RF_DATASETS = {
    "rf_space_classification": ("rf_space_classification/rf_space_classification.csv", ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability")),
    "rf_space_regression": ("rf_space_regression/rf_space_regression.csv", ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability")),
    "rf_integrated_regression_mean": ("rf_integrated_regression/rf_integrated_regression.csv", ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability", "civilian_count", "government_count", "military_count", "ground_station_failure_probability")),
    "rf_integrated_regression_min": ("rf_integrated_regression/rf_integrated_regression.csv", ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability", "civilian_count", "government_count", "military_count", "ground_station_failure_probability")),
    "rf_integrated_classification": ("rf_integrated_classification/rf_integrated_classification.csv", ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability", "civilian_count", "government_count", "military_count", "ground_station_failure_probability")),
}
METADATA = ("run_id", "run_key", "design_id", "realization_id", "split")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def self_excluding_bundle_hash(root: Path, exclude_name: str) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != exclude_name):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        digest.update(relative.encode("utf-8")); digest.update(b"\0"); digest.update(data)
        entries.append({"path": relative, "bytes": len(data), "sha256": sha256_bytes(data)})
    return digest.hexdigest(), entries


def git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True, check=True)
    return result.stdout.strip()


def verify_reconciliation() -> dict[str, Any]:
    path = RECON_ROOT / "robustness_provenance_reconciliation.json"
    inventory_path = RECON_ROOT / "robustness_reconciliation_inventory.json"
    result = load_json(path)
    inventory = load_json(inventory_path)
    bundle_sha, entries = self_excluding_bundle_hash(RECON_ROOT, inventory_path.name)
    required = (
        result.get("rf", {}).get("models_checked") == 25,
        result.get("rf", {}).get("random_state_matches") == 25,
        result.get("rf", {}).get("mismatches") == 0,
        result.get("tgnn", {}).get("runs_checked") == 10,
        result.get("tgnn", {}).get("checkpoint_selected_epoch_matches") == 10,
        result.get("tgnn", {}).get("progress_logs_valid") == 10,
        result.get("tgnn", {}).get("mismatches") == 0,
        result.get("scientific_training_invalidated") is False,
        result.get("retraining_required") is False,
        result.get("robustness_bundle_modified") is False,
        result.get("test_accessed") is False,
        bundle_sha == RECON_BUNDLE_SHA,
        inventory.get("bundle_sha256") == RECON_BUNDLE_SHA,
        inventory.get("artifacts") == entries,
    )
    if not all(required):
        raise RuntimeError("Phase A reconciliation evidence is not a complete PASS")
    return {"path": str(path), "bundle_sha256": bundle_sha, "driver_sha256": result["reconciliation_driver_sha256"]}


def verify_robustness_bundle() -> None:
    inventory_path = FINAL_ROOT / "final_robustness_inventory.json"
    inventory = load_json(inventory_path)
    observed, entries = self_excluding_bundle_hash(FINAL_ROOT, inventory_path.name)
    if observed != ROBUSTNESS_BUNDLE_SHA or inventory.get("bundle_sha256") != ROBUSTNESS_BUNDLE_SHA or inventory.get("artifacts") != entries:
        raise RuntimeError("Frozen robustness bundle changed or is not byte-identical")


def verify_model_artifacts() -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for task in TASKS:
        family = "RF" if task.startswith("rf_") else "TGNN"
        for seed in SEEDS:
            run_root = FINAL_ROOT / task / f"seed_{seed}"
            manifest_path = run_root / "final_manifest.json"
            manifest = load_json(manifest_path)
            if manifest.get("status") != "completed" or manifest.get("task") != task or manifest.get("seed") != seed or manifest.get("test_accessed") is not False:
                raise RuntimeError(f"Incomplete or test-accessed frozen artifact: {task}/seed_{seed}")
            if family == "RF":
                artifact_path = run_root / "final_model.joblib"
                observed = sha256_file(artifact_path)
                if observed != manifest.get("model_sha256"):
                    raise RuntimeError(f"RF model hash mismatch: {task}/seed_{seed}")
                import joblib
                estimator = joblib.load(artifact_path)
                if getattr(estimator, "random_state", None) != seed:
                    raise RuntimeError(f"RF persisted random_state mismatch: {task}/seed_{seed}")
                artifact_kind = "final_model.joblib"
                artifact_sha = manifest["model_sha256"]
            else:
                artifact_path = run_root / "best_validation_checkpoint.pt"
                observed = sha256_file(artifact_path)
                if observed != manifest.get("checkpoint_sha256"):
                    raise RuntimeError(f"TGNN checkpoint hash mismatch: {task}/seed_{seed}")
                artifact_kind = "best_validation_checkpoint.pt"
                artifact_sha = manifest["checkpoint_sha256"]
            artifacts.append({"task": task, "model_family": family, "seed": seed, "path": str(artifact_path), "kind": artifact_kind, "sha256": artifact_sha})
    if len(artifacts) != 35:
        raise RuntimeError("Expected exactly 35 frozen model/checkpoint artifacts")
    return artifacts


def metadata_frame_from_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(METADATA))


def metadata_frame_from_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)[list(METADATA)]


def validate_metadata_frame(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    if len(frame) != 10000:
        raise RuntimeError(f"{label} has {len(frame)} rows, expected 10000")
    frame = frame.copy()
    frame["run_id"] = frame["run_id"].astype(int)
    frame["split"] = frame["split"].astype(str).str.lower()
    if set(frame["run_id"]) != set(range(10000)) or frame["run_id"].duplicated().any():
        raise RuntimeError(f"{label} run identities are not exactly 0..9999")
    counts = frame.groupby("split").size().to_dict()
    if counts != {"train": 7000, "test": 1500, "validation": 1500}:
        raise RuntimeError(f"{label} split counts mismatch: {counts}")
    design_counts = frame[frame["split"] == "test"].groupby("design_id")["realization_id"].nunique()
    if len(design_counts) != TEST_DESIGNS or set(design_counts) != {REALIZATIONS_PER_DESIGN}:
        raise RuntimeError(f"{label} test design/realization counts mismatch")
    if set(frame[frame["split"] == "test"]["realization_id"]) != {"R00", "R01", "R02", "R03", "R04"}:
        raise RuntimeError(f"{label} test realizations are not R00..R04")
    design_split = frame.groupby("design_id")["split"].nunique()
    if (design_split > 1).any():
        raise RuntimeError(f"{label} has design leakage across splits")
    return {"rows": len(frame), "split_counts": counts, "test_designs": len(design_counts), "design_leakage": False}


def preaccess_split_audit() -> dict[str, Any]:
    mappings: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for task, (relative, _) in RF_DATASETS.items():
        frame = metadata_frame_from_csv(DATASET_ROOT / relative)
        audits[task] = validate_metadata_frame(frame, task)
        mappings.append(frame[["run_id", "design_id", "realization_id", "split"]].sort_values("run_id").reset_index(drop=True))
    graph_manifest = DATASET_ROOT / "tgnn_space_classification" / "tgnn_space_graph_manifest.jsonl"
    graph_frame = metadata_frame_from_jsonl(graph_manifest)
    audits["tgnn_graph_manifest"] = validate_metadata_frame(graph_frame, "tgnn_graph_manifest")
    mappings.append(graph_frame[["run_id", "design_id", "realization_id", "split"]].sort_values("run_id").reset_index(drop=True))
    reference = mappings[0]
    for other in mappings[1:]:
        if not reference.equals(other):
            raise RuntimeError("Authoritative metadata mapping differs between task artifacts")
    return {"authoritative_mapping_consistent": True, "audits": audits, "test_runs": TEST_RUNS, "test_designs": TEST_DESIGNS, "design_leakage": False}


def write_authorization(reconciliation: dict[str, Any], model_artifacts: list[dict[str, Any]], split_audit: dict[str, Any], dataset_verification: dict[str, Any], plan_verification: dict[str, Any]) -> tuple[dict[str, Any], str]:
    driver_sha = sha256_file(Path(__file__))
    authorization = {
        "schema_version": "satnet.heldout_test_authorization.v1",
        "authorized": True,
        "test_accessed_at_freeze": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identities": {
            "implementation_sha": IMPLEMENTATION_SHA,
            "production_sha": PRODUCTION_SHA,
            "dataset_bundle_sha": DATASET_SHA,
            "training_plan_bundle_sha": PLAN_SHA,
            "rf_selection_sha": RF_SELECTION_SHA,
            "tgnn_selection_sha": TGNN_SELECTION_SHA,
            "robustness_bundle_sha": ROBUSTNESS_BUNDLE_SHA,
            "reconciliation_bundle_sha": reconciliation["bundle_sha256"],
        },
        "execution": {"worktree": str(REPO), "qualified_python": str(QUALIFIED_PYTHON), "git_head": git_head(), "driver_sha256": driver_sha},
        "seeds": {"all": list(SEEDS), "primary_reporting_seed": PRIMARY_SEED, "bootstrap_seed": BOOTSTRAP_SEED},
        "test_split": {"runs": TEST_RUNS, "designs": TEST_DESIGNS, "realizations_per_design": REALIZATIONS_PER_DESIGN, "split_column_authoritative": True, "designs_clustered": True, "audit": split_audit},
        "metrics": {
            "classification": ["balanced_accuracy", "accuracy", "precision_by_class", "recall_by_class", "f1_by_class", "macro_f1", "weighted_f1", "specificity", "sensitivity", "confusion_matrix", "roc_auc", "pr_auc"],
            "regression": ["mae", "rmse", "r2", "median_absolute_error", "maximum_absolute_error", "target_mean", "target_median", "target_standard_deviation", "prediction_mean", "prediction_median", "prediction_standard_deviation", "predictions_below_zero", "predictions_above_one"],
            "clipping": False,
        },
        "classification_behavior": {"rf": "estimator.predict() and estimator.predict_proba()", "tgnn": "frozen SatelliteGNN output, argmax and positive-class softmax probability", "threshold_tuning": False},
        "baselines": {"classification": ["TRAIN-majority classifier", "TRAIN-prevalence stratified random classifier, seed 42"], "regression": ["TRAIN mean predictor", "TRAIN median predictor"], "construction_split": "TRAIN only"},
        "bootstrap": {"cluster_unit": "design_id", "paired": True, "designs": TEST_DESIGNS, "realizations_per_design": REALIZATIONS_PER_DESIGN, "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "confidence_interval": "95% percentile", "comparisons": ["rf_space_classification vs tgnn_space_classification", "rf_space_regression vs tgnn_space_regression"]},
        "preaccess_verification": {"dataset": dataset_verification, "training_plan": plan_verification, "reconciliation": reconciliation, "robustness_artifacts": model_artifacts, "git_head_unchanged_expected": EXPECTED_HEAD, "test_targets_read": False},
    }
    path = AUTH_ROOT / "heldout_test_authorization.json"
    atomic_json(path, authorization)
    authorization_sha = sha256_file(path)
    (AUTH_ROOT / "heldout_test_authorization_sha256.txt").write_text(authorization_sha + "\n", encoding="ascii")
    return authorization, authorization_sha


def verify_existing_authorization() -> tuple[dict[str, Any], str]:
    path = AUTH_ROOT / "heldout_test_authorization.json"
    sha_path = AUTH_ROOT / "heldout_test_authorization_sha256.txt"
    authorization = load_json(path)
    actual_sha = sha256_file(path)
    expected_sha = sha_path.read_text(encoding="ascii").strip()
    if actual_sha != expected_sha or authorization.get("authorized") is not True or authorization.get("test_accessed_at_freeze") is not False:
        raise RuntimeError("Held-out authorization artifact is missing, changed, or invalid")
    if authorization.get("execution", {}).get("driver_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("Held-out test driver changed after authorization freeze")
    if authorization.get("execution", {}).get("git_head") != git_head():
        raise RuntimeError("Git HEAD changed after authorization freeze")
    return authorization, actual_sha


def set_test_accessed(authorization_sha: str) -> None:
    progress_path = OUT_ROOT / "heldout_test_progress.json"
    if progress_path.is_file():
        progress = load_json(progress_path)
        if progress.get("test_accessed") is True:
            return
    atomic_json(progress_path, {"schema_version": "satnet.heldout_test_progress.v1", "test_accessed": True, "authorization_sha256": authorization_sha, "started_at": datetime.now(timezone.utc).isoformat(), "completed": 0, "failed": 0, "evaluations": {}})


def normalize_class_target(values: Iterable[Any]) -> np.ndarray:
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            result.append(int(value))
        elif str(value).strip().lower() in {"true", "1"}:
            result.append(1)
        elif str(value).strip().lower() in {"false", "0"}:
            result.append(0)
        else:
            raise ValueError(f"Invalid classification target: {value!r}")
    return np.asarray(result, dtype=np.int64)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    y_true = normalize_class_target(y_true); y_pred = normalize_class_target(y_pred); score = np.asarray(score, dtype=float)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    result: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)), "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_by_class": {"0": float(precision[0]), "1": float(precision[1])}, "recall_by_class": {"0": float(recall[0]), "1": float(recall[1])}, "f1_by_class": {"0": float(f1[0]), "1": float(f1[1])},
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)), "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "specificity": float(recall[0]), "sensitivity": float(recall[1]), "confusion_matrix": cm.tolist(), "positive_class_count": int(np.sum(y_true == 1)), "negative_class_count": int(np.sum(y_true == 0)),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, score)) if len(np.unique(score)) > 1 else None
        result["pr_auc"] = float(average_precision_score(y_true, score))
    else:
        result["roc_auc"] = None; result["pr_auc"] = None
    return result


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float); prediction = np.asarray(prediction, dtype=float); error = prediction - y_true
    return {
        "mae": float(mean_absolute_error(y_true, prediction)), "rmse": float(math.sqrt(mean_squared_error(y_true, prediction))), "r2": float(r2_score(y_true, prediction)),
        "median_absolute_error": float(np.median(np.abs(error))), "maximum_absolute_error": float(np.max(np.abs(error))), "target_mean": float(np.mean(y_true)), "target_median": float(np.median(y_true)), "target_standard_deviation": float(np.std(y_true)),
        "prediction_mean": float(np.mean(prediction)), "prediction_median": float(np.median(prediction)), "prediction_standard_deviation": float(np.std(prediction)), "predictions_below_zero": int(np.sum(prediction < 0)), "predictions_above_one": int(np.sum(prediction > 1)),
    }


def read_rf_task(task: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    relative, features = RF_DATASETS[task]
    frame = pd.read_csv(DATASET_ROOT / relative)
    target = TARGETS[task]
    required = set(METADATA) | set(features) | {target}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{task} missing columns: {sorted(missing)}")
    frame["run_id"] = frame["run_id"].astype(int); frame["split"] = frame["split"].astype(str).str.lower()
    train = frame[frame["split"] == "train"].copy(); test = frame[frame["split"] == "test"].copy()
    if len(train) != 7000 or len(test) != TEST_RUNS or test["design_id"].nunique() != TEST_DESIGNS:
        raise RuntimeError(f"{task} frozen split count mismatch")
    x_train = train[list(features)].to_numpy(dtype=float); x_test = test[list(features)].to_numpy(dtype=float)
    if TASK_TYPE[task] == "classification":
        y_train = normalize_class_target(train[target].to_numpy()); y_test = normalize_class_target(test[target].to_numpy())
    else:
        y_train = train[target].to_numpy(dtype=float); y_test = test[target].to_numpy(dtype=float)
    return test, y_train, y_test, x_train, x_test


def model_artifact_for(task: str, seed: int) -> tuple[Path, dict[str, Any]]:
    manifest = load_json(FINAL_ROOT / task / f"seed_{seed}" / "final_manifest.json")
    if task.startswith("rf_"):
        return Path(manifest["model_path"]), manifest
    return Path(manifest["checkpoint_path"]), manifest


def evaluation_is_complete(task: str, seed: int, authorization: dict[str, Any]) -> bool:
    directory = OUT_ROOT / task / f"seed_{seed}"
    manifest_path = directory / "evaluation_manifest.json"; prediction_path = directory / "predictions.csv"; metrics_path = directory / "metrics.json"
    if not (manifest_path.is_file() and prediction_path.is_file() and metrics_path.is_file()):
        return False
    manifest = load_json(manifest_path)
    expected_artifact = next(item for item in authorization["preaccess_verification"]["robustness_artifacts"] if item["task"] == task and item["seed"] == seed)
    return all((manifest.get("status") == "completed", manifest.get("task") == task, manifest.get("seed") == seed, manifest.get("dataset_bundle_sha") == DATASET_SHA, manifest.get("model_sha256") == expected_artifact["sha256"], manifest.get("row_count") == TEST_RUNS, manifest.get("predictions_sha256") == sha256_file(prediction_path), manifest.get("metrics_sha256") == sha256_file(metrics_path), len(pd.read_csv(prediction_path)) == TEST_RUNS))


def persist_evaluation(task: str, seed: int, predictions: pd.DataFrame, metrics: dict[str, Any], model_sha: str, authorization_sha: str) -> dict[str, Any]:
    directory = OUT_ROOT / task / f"seed_{seed}"; directory.mkdir(parents=True, exist_ok=True)
    prediction_path = directory / "predictions.csv"; metrics_path = directory / "metrics.json"; manifest_path = directory / "evaluation_manifest.json"
    prediction_temp = prediction_path.with_suffix(".csv.tmp"); predictions.to_csv(prediction_temp, index=False); os.replace(prediction_temp, prediction_path)
    atomic_json(metrics_path, metrics)
    manifest = {"schema_version": "satnet.heldout_test_evaluation_manifest.v1", "status": "completed", "task": task, "model_family": "RF" if task.startswith("rf_") else "TGNN", "seed": seed, "dataset_bundle_sha": DATASET_SHA, "model_sha256": model_sha, "authorization_sha256": authorization_sha, "split": "test", "row_count": len(predictions), "predictions_sha256": sha256_file(prediction_path), "metrics_sha256": sha256_file(metrics_path), "training_performed": False, "checkpoint_modified": False, "created_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(manifest_path, manifest)
    return manifest


def evaluate_rf(task: str, seed: int, authorization_sha: str) -> dict[str, Any]:
    test, y_train, y_test, x_train, x_test = read_rf_task(task)
    model_path, model_manifest = model_artifact_for(task, seed)
    import joblib
    estimator = joblib.load(model_path)
    if TASK_TYPE[task] == "classification":
        predicted = normalize_class_target(estimator.predict(x_test))
        raw_probability = np.asarray(estimator.predict_proba(x_test), dtype=float)
        classes = list(estimator.classes_)
        positive_index = next(index for index, value in enumerate(classes) if normalize_class_target([value])[0] == 1)
        score = raw_probability[:, positive_index]
        metrics = classification_metrics(y_test, predicted, score)
        output = test[list(METADATA)].copy(); output["target"] = y_test; output["prediction"] = predicted; output["positive_class_score"] = score
    else:
        predicted = np.asarray(estimator.predict(x_test), dtype=float); metrics = regression_metrics(y_test, predicted)
        output = test[list(METADATA)].copy(); output["target"] = y_test; output["prediction"] = predicted; output["error"] = predicted - y_test; output["absolute_error"] = np.abs(predicted - y_test)
    metrics["task"] = task; metrics["seed"] = seed; metrics["model_sha256"] = model_manifest["model_sha256"]
    return persist_evaluation(task, seed, output, metrics, model_manifest["model_sha256"], authorization_sha)


def load_tgnn_test_records(task: str) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    graph_path = DATASET_ROOT / "tgnn_space_classification" / "tgnn_space_graph_manifest.jsonl"
    target_path = DATASET_ROOT / ("tgnn_space_classification/tgnn_space_classification_target_manifest.jsonl" if task == "tgnn_space_classification" else "tgnn_space_regression/tgnn_space_regression_target_manifest.jsonl")
    graphs = [json.loads(line) for line in graph_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    targets = [json.loads(line) for line in target_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_target = {int(row["run_id"]): row for row in targets}
    test_graphs = [row for row in graphs if str(row["split"]).lower() == "test"]
    if len(test_graphs) != TEST_RUNS or len({row["design_id"] for row in test_graphs}) != TEST_DESIGNS:
        raise RuntimeError(f"{task} TGNN test identity count mismatch")
    target_values: dict[int, Any] = {}
    for row in test_graphs:
        run_id = int(row["run_id"]); target_row = by_target[run_id]
        if target_row.get("design_id") != row.get("design_id") or target_row.get("realization_id") != row.get("realization_id") or target_row.get("run_key") != row.get("run_key"):
            raise RuntimeError(f"{task} target identity mismatch for run {run_id}")
        target_values[run_id] = normalize_class_target([target_row["target"]])[0] if TASK_TYPE[task] == "classification" else float(target_row["target"])
    return sorted(test_graphs, key=lambda row: int(row["run_id"])), target_values


def evaluate_tgnn(task: str, seed: int, authorization_sha: str) -> dict[str, Any]:
    import torch
    sys.path.insert(0, str(SRC))
    from satnet.experiments.final_training.tgnn_loader import read_sequence
    from satnet.models.gnn_model import SatelliteGNN
    graph_rows, targets = load_tgnn_test_records(task)
    model_path, model_manifest = model_artifact_for(task, seed)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = model_manifest["configuration"]
    task_type = TASK_TYPE[task]
    model = SatelliteGNN(node_features=3, hidden_channels=int(config["hidden_dim"]), out_channels=2 if task_type == "classification" else 1, task_type=task_type, cheb_k=int(config["cheb_k"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval(); torch.set_num_threads(8)
    predictions: list[float | int] = []; scores: list[float] = []
    for row in graph_rows:
        sequence = read_sequence(DATASET_ROOT / str(row["sequence_artifact"]))
        data_list = sequence.data_list()
        with torch.no_grad():
            if task_type == "classification":
                probability = model.predict_proba(data_list).cpu().numpy()[0]
                predictions.append(int(np.argmax(probability))); scores.append(float(probability[1]))
            else:
                predictions.append(float(model.predict(data_list)))
    y_test = np.asarray([targets[int(row["run_id"])] for row in graph_rows], dtype=float if task_type == "regression" else np.int64)
    if task_type == "classification":
        predicted = np.asarray(predictions, dtype=np.int64); metrics = classification_metrics(y_test, predicted, np.asarray(scores))
        output = pd.DataFrame([{**{key: row[key] for key in METADATA}, "target": int(y_test[index]), "prediction": int(predicted[index]), "positive_class_score": scores[index]} for index, row in enumerate(graph_rows)])
    else:
        predicted = np.asarray(predictions, dtype=float); metrics = regression_metrics(y_test, predicted)
        output = pd.DataFrame([{**{key: row[key] for key in METADATA}, "target": float(y_test[index]), "prediction": predicted[index], "error": predicted[index] - y_test[index], "absolute_error": abs(predicted[index] - y_test[index])} for index, row in enumerate(graph_rows)])
    metrics["task"] = task; metrics["seed"] = seed; metrics["model_sha256"] = model_manifest["checkpoint_sha256"]
    return persist_evaluation(task, seed, output, metrics, model_manifest["checkpoint_sha256"], authorization_sha)


def load_predictions(task: str, seed: int) -> pd.DataFrame:
    path = OUT_ROOT / task / f"seed_{seed}" / "predictions.csv"
    frame = pd.read_csv(path)
    if len(frame) != TEST_RUNS or set(frame["split"]) != {"test"} or frame["design_id"].nunique() != TEST_DESIGNS:
        raise RuntimeError(f"Prediction identity contract failed for {task}/seed_{seed}")
    return frame


def update_progress(authorization_sha: str, completed: int, failed: int, evaluations: dict[str, Any]) -> None:
    atomic_json(OUT_ROOT / "heldout_test_progress.json", {"schema_version": "satnet.heldout_test_progress.v1", "test_accessed": True, "authorization_sha256": authorization_sha, "completed": completed, "expected": 35, "failed": failed, "evaluations": evaluations})


def seed_summary(evaluation_frames: dict[tuple[str, int], pd.DataFrame]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in TASKS:
        metric = "balanced_accuracy" if TASK_TYPE[task] == "classification" else "mae"
        values: dict[str, float] = {}
        for seed in SEEDS:
            metrics = load_json(OUT_ROOT / task / f"seed_{seed}" / "metrics.json")
            values[str(seed)] = float(metrics[metric])
        array = np.asarray(list(values.values()), dtype=float)
        result[task] = {"primary_metric": metric, "seed_42": values["42"], "seed_values": values, "mean": float(np.mean(array)), "population_standard_deviation": float(np.std(array)), "min": float(np.min(array)), "max": float(np.max(array))}
    return result


def train_baselines(task: str) -> dict[str, Any]:
    test, y_train, y_test, _, _ = read_rf_task(task)
    if TASK_TYPE[task] == "classification":
        counts = np.bincount(y_train.astype(int), minlength=2); majority = int(np.argmax(counts)); prevalence = float(np.mean(y_train == 1)); rng = np.random.default_rng(42); random_prediction = (rng.random(len(y_test)) < prevalence).astype(int); majority_prediction = np.full(len(y_test), majority, dtype=int)
        output = {"majority_class_classifier": classification_metrics(y_test, majority_prediction, majority_prediction.astype(float)), "stratified_random_classifier": classification_metrics(y_test, random_prediction, random_prediction.astype(float))}
        output["majority_class_classifier"].update({"train_class_counts": {"0": int(counts[0]), "1": int(counts[1])}, "train_positive_prevalence": prevalence})
        output["stratified_random_classifier"].update({"train_class_counts": {"0": int(counts[0]), "1": int(counts[1])}, "train_positive_prevalence": prevalence, "seed": 42})
    else:
        mean_value = float(np.mean(y_train)); median_value = float(np.median(y_train)); output = {"training_mean_predictor": regression_metrics(y_test, np.full(len(y_test), mean_value)), "training_median_predictor": regression_metrics(y_test, np.full(len(y_test), median_value))}
        output["training_mean_predictor"]["train_mean"] = mean_value; output["training_median_predictor"]["train_median"] = median_value
    return {"task": task, "target": TARGETS[task], "train_rows": 7000, "test_rows": TEST_RUNS, "metrics": output}


def paired_bootstrap(classification: bool, rf_task: str, tgnn_task: str) -> dict[str, Any]:
    rf = load_predictions(rf_task, PRIMARY_SEED); tgnn = load_predictions(tgnn_task, PRIMARY_SEED)
    rf = rf.sort_values("run_id").reset_index(drop=True); tgnn = tgnn.sort_values("run_id").reset_index(drop=True)
    if not rf[["run_id", "design_id", "realization_id"]].equals(tgnn[["run_id", "design_id", "realization_id"]]):
        raise RuntimeError("Primary RF/TGNN prediction identities are not paired")
    designs = np.asarray(sorted(rf["design_id"].unique()), dtype=object); rng = np.random.default_rng(BOOTSTRAP_SEED); deltas: list[float] = []; rejected = 0
    rf_groups = {design: group.index.to_numpy() for design, group in rf.groupby("design_id", sort=False)}
    tgnn_groups = {design: group.index.to_numpy() for design, group in tgnn.groupby("design_id", sort=False)}
    while len(deltas) < BOOTSTRAP_REPLICATES:
        sampled = rng.choice(designs, size=TEST_DESIGNS, replace=True); indexes = np.concatenate([rf_groups[design] for design in sampled]); t_indexes = np.concatenate([tgnn_groups[design] for design in sampled])
        if classification:
            y_rf = normalize_class_target(rf.iloc[indexes]["target"]); y_tgnn = normalize_class_target(tgnn.iloc[t_indexes]["target"])
            if len(np.unique(y_rf)) < 2 or len(np.unique(y_tgnn)) < 2:
                rejected += 1; continue
            rf_metric = float(balanced_accuracy_score(y_rf, normalize_class_target(rf.iloc[indexes]["prediction"]))); tgnn_metric = float(balanced_accuracy_score(y_tgnn, normalize_class_target(tgnn.iloc[t_indexes]["prediction"])))
        else:
            rf_metric = float(mean_absolute_error(rf.iloc[indexes]["target"].to_numpy(float), rf.iloc[indexes]["prediction"].to_numpy(float))); tgnn_metric = float(mean_absolute_error(tgnn.iloc[t_indexes]["target"].to_numpy(float), tgnn.iloc[t_indexes]["prediction"].to_numpy(float)))
        deltas.append(rf_metric - tgnn_metric)
    array = np.asarray(deltas)
    rf_observed = load_json(OUT_ROOT / rf_task / "seed_42" / "metrics.json")["balanced_accuracy" if classification else "mae"]; tgnn_observed = load_json(OUT_ROOT / tgnn_task / "seed_42" / "metrics.json")["balanced_accuracy" if classification else "mae"]
    return {"comparison": f"{rf_task}/seed_42 vs {tgnn_task}/seed_42", "statistic": "delta_balanced_accuracy = RF - TGNN" if classification else "delta_MAE = RF - TGNN", "observed_rf_metric": rf_observed, "observed_tgnn_metric": tgnn_observed, "observed_delta": float(rf_observed - tgnn_observed), "bootstrap_mean_delta": float(np.mean(array)), "bootstrap_standard_deviation": float(np.std(array)), "percentile_2_5": float(np.percentile(array, 2.5)), "percentile_97_5": float(np.percentile(array, 97.5)), "bootstrap_seed": BOOTSTRAP_SEED, "valid_replicates": len(deltas), "rejected_replicates": rejected, "cluster_unit": "design_id", "paired": True, "designs": TEST_DESIGNS, "realizations_per_design": REALIZATIONS_PER_DESIGN}


def main() -> None:
    started = time.perf_counter()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if AUTH_ROOT.joinpath("heldout_test_authorization.json").is_file():
        authorization, authorization_sha = verify_existing_authorization()
    else:
        verify_robustness_bundle()
        reconciliation = verify_reconciliation()
        model_artifacts = verify_model_artifacts()
        if git_head() != EXPECTED_HEAD:
            raise RuntimeError("Git HEAD does not match the frozen implementation SHA")
        sys.path.insert(0, str(SRC))
        from satnet.experiments.final_training.contracts import verify_dataset_bundle, verify_training_plan
        dataset_verification = verify_dataset_bundle(DATASET_ROOT, full=True)
        plan_verification = verify_training_plan(PLAN_ROOT)
        split_audit = preaccess_split_audit()
        if git_head() != EXPECTED_HEAD:
            raise RuntimeError("Git HEAD changed during pre-access verification")
        authorization, authorization_sha = write_authorization(reconciliation, model_artifacts, split_audit, dataset_verification, plan_verification)
    set_test_accessed(authorization_sha)
    evaluations: dict[str, Any] = load_json(OUT_ROOT / "heldout_test_progress.json").get("evaluations", {}) if (OUT_ROOT / "heldout_test_progress.json").is_file() else {}
    completed = sum(1 for task in TASKS for seed in SEEDS if evaluation_is_complete(task, seed, authorization))
    failed = 0
    update_progress(authorization_sha, completed, failed, evaluations)
    for task in TASKS:
        for seed in SEEDS:
            key = f"{task}/seed_{seed}"
            if evaluation_is_complete(task, seed, authorization):
                evaluations[key] = {"status": "completed", "skipped_verified": True}
                continue
            try:
                manifest = evaluate_rf(task, seed, authorization_sha) if task.startswith("rf_") else evaluate_tgnn(task, seed, authorization_sha)
                evaluations[key] = {"status": "completed", "manifest_sha256": sha256_file(OUT_ROOT / task / f"seed_{seed}" / "evaluation_manifest.json"), "model_sha256": manifest["model_sha256"]}
                completed += 1
            except Exception as exc:
                failed += 1; evaluations[key] = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}; update_progress(authorization_sha, completed, failed, evaluations); raise
            update_progress(authorization_sha, completed, failed, evaluations)
    frames = {(task, seed): load_predictions(task, seed) for task in TASKS for seed in SEEDS}
    summaries = seed_summary(frames)
    baseline_results = {task: train_baselines(task) for task in ("rf_space_classification", "rf_space_regression", "rf_integrated_regression_mean", "rf_integrated_regression_min", "rf_integrated_classification")}
    bootstrap_results = {"space_classification": paired_bootstrap(True, "rf_space_classification", "tgnn_space_classification"), "space_regression": paired_bootstrap(False, "rf_space_regression", "tgnn_space_regression")}
    test_class_counts = {task: {"negative": int(load_json(OUT_ROOT / task / "seed_42" / "metrics.json")["negative_class_count"]), "positive": int(load_json(OUT_ROOT / task / "seed_42" / "metrics.json")["positive_class_count"])} for task in TASKS if TASK_TYPE[task] == "classification"}
    primary_results = {task: load_json(OUT_ROOT / task / "seed_42" / "metrics.json") for task in TASKS}
    atomic_json(OUT_ROOT / "heldout_test_seed_summary.json", summaries)
    atomic_json(OUT_ROOT / "heldout_test_primary_results.json", primary_results)
    atomic_json(OUT_ROOT / "heldout_test_baselines.json", baseline_results)
    atomic_json(OUT_ROOT / "paired_design_cluster_bootstrap.json", bootstrap_results)
    final_summary = {"schema_version": "satnet.heldout_test_summary.v1", "status": "complete", "test_accessed": True, "evaluations_completed": completed, "evaluations_expected": 35, "failed": failed, "test_rows_per_model": TEST_RUNS, "test_designs": TEST_DESIGNS, "test_class_counts_seed_42": test_class_counts, "seed_summary": summaries, "primary_seed": PRIMARY_SEED, "primary_results": primary_results, "baselines": baseline_results, "bootstrap": bootstrap_results, "no_training_performed": True, "no_test_based_selection_or_tuning": True, "authorization_sha256": authorization_sha, "heldout_test_driver_sha256": sha256_file(Path(__file__)), "runtime_seconds": time.perf_counter() - started, "model_artifacts": authorization["preaccess_verification"]["robustness_artifacts"]}
    atomic_json(OUT_ROOT / "heldout_test_summary.json", final_summary)
    report = "# Held-Out Synthetic Test v1\n\n" + json.dumps(final_summary, indent=2, sort_keys=True) + "\n\nNo training, retraining, checkpoint update, model selection, threshold tuning, or TEST-derived refit occurred.\n"
    (OUT_ROOT / "heldout_test_report.md").write_text(report, encoding="utf-8")
    verify_robustness_bundle(); sys.path.insert(0, str(SRC))
    from satnet.experiments.final_training.contracts import verify_dataset_bundle, verify_training_plan
    verify_dataset_bundle(DATASET_ROOT, full=True); verify_training_plan(PLAN_ROOT)
    if git_head() != EXPECTED_HEAD:
        raise RuntimeError("Git HEAD changed during held-out evaluation")
    if completed != 35 or failed != 0:
        raise RuntimeError(f"Held-out acceptance failed: {completed}/35, failures={failed}")
    update_progress(authorization_sha, completed, failed, evaluations)
    inventory_path = OUT_ROOT / "heldout_test_inventory.json"; bundle_sha, entries = self_excluding_bundle_hash(OUT_ROOT, inventory_path.name)
    atomic_json(inventory_path, {"schema_version": "satnet.heldout_test_inventory.v1", "inventory_self_excluding": True, "hash_algorithm": "sha256 over sorted relative UTF-8 path + NUL byte + raw file bytes", "bundle_sha256": bundle_sha, "artifacts": entries, "driver_path": str(Path(__file__)), "driver_sha256": sha256_file(Path(__file__)), "authorization_sha256": authorization_sha, "test_accessed": True})
    print("HELD-OUT SYNTHETIC TEST COMPLETE — FINAL SYNTHETIC RESULTS FROZEN")


if __name__ == "__main__":
    main()
