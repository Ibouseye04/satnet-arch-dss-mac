from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO = Path(r"C:\Users\johns\external\satnet-10k-training-worktree-v1")
SRC = REPO / "src"
DATASET_ROOT = Path(r"C:\Users\johns\external\satnet-10k-final-ml-datasets-v1-final")
PLAN_ROOT = Path(r"C:\Users\johns\external\satnet-10k-training-test-plan-v1")
ROOT = Path(r"C:\Users\johns\external\satnet-10k-model-training-v1")
DATASET_HASH = "38dacd66432bfa660410ff9cab7f181a53151120e8102dc40df57016f78bbfc3"
PLAN_HASH = "e14ec5e5b2221aa2c9aa187a2baacc517e676f9dc748612c206eb1dc43788a0a"
PRODUCTION_SHA = "d0515088cf3fca06a6aa2d47059269089dcb10a7"
CODE_SHA = "e55dba59e83864d2dd11fa47482bd4ec2bdd797a"
VALIDATION_SEEDS = (42, 123, 456)
EXPECTED_SPLITS = {"train": 7000, "validation": 1500, "test": 1500}
RF_TASKS = (
    "rf_space_classification",
    "rf_space_regression",
    "rf_integrated_regression_mean",
    "rf_integrated_regression_min",
    "rf_integrated_classification",
)
TGNN_TASKS = ("tgnn_space_classification", "tgnn_space_regression")
TASKS: dict[str, dict[str, Any]] = {
    "rf_space_classification": {"family": "RF", "type": "classification", "target": "space_threshold_breach_any", "features": ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability"), "path": "rf_space_classification/rf_space_classification.csv"},
    "rf_space_regression": {"family": "RF", "type": "regression", "target": "space_gcc_fraction_original_min", "features": ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability"), "path": "rf_space_regression/rf_space_regression.csv"},
    "rf_integrated_regression_mean": {"family": "RF", "type": "regression", "target": "failure_adjusted_overall_service_fraction_mean", "features": ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability", "civilian_count", "government_count", "military_count", "ground_station_failure_probability"), "path": "rf_integrated_regression/rf_integrated_regression.csv"},
    "rf_integrated_regression_min": {"family": "RF", "type": "regression", "target": "failure_adjusted_overall_service_fraction_min", "features": ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability", "civilian_count", "government_count", "military_count", "ground_station_failure_probability"), "path": "rf_integrated_regression/rf_integrated_regression.csv"},
    "rf_integrated_classification": {"family": "RF", "type": "classification", "target": "overall_threshold_breach_any", "features": ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability", "civilian_count", "government_count", "military_count", "ground_station_failure_probability"), "path": "rf_integrated_classification/rf_integrated_classification.csv"},
    "tgnn_space_classification": {"family": "TGNN", "type": "classification", "target": "space_threshold_breach_any", "target_path": "tgnn_space_classification/tgnn_space_classification_target_manifest.jsonl"},
    "tgnn_space_regression": {"family": "TGNN", "type": "regression", "target": "space_gcc_fraction_original_min", "target_path": "tgnn_space_regression/tgnn_space_regression_target_manifest.jsonl"},
}
RF_COMMON = {
    "n_estimators": (300, 600),
    "max_depth": (None, 10, 20),
    "min_samples_leaf": (1, 2, 5),
    "max_features": ("sqrt", 1.0),
    "bootstrap": (True,),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(native(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complexity(config: dict[str, Any]) -> tuple[float, float, str]:
    # This exactly mirrors the qualified selection helper's frozen ordering.
    depth = config.get("max_depth")
    return (float(depth) if depth is not None else math.inf, float(config.get("n_estimators", config.get("hidden_dim", math.inf))), str(sorted(config.items())))


def rf_configs(task_id: str) -> list[dict[str, Any]]:
    weights = (None, "balanced") if task_id == "rf_space_classification" else (("balanced", "balanced_subsample") if task_id == "rf_integrated_classification" else (None,))
    configs: list[dict[str, Any]] = []
    for n in RF_COMMON["n_estimators"]:
        for depth in RF_COMMON["max_depth"]:
            for leaf in RF_COMMON["min_samples_leaf"]:
                for features in RF_COMMON["max_features"]:
                    for bootstrap in RF_COMMON["bootstrap"]:
                        for weight in weights:
                            configs.append({"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf, "max_features": features, "bootstrap": bootstrap, "class_weight": weight})
    return configs


def tgnn_configs() -> list[dict[str, Any]]:
    return [{"hidden_dim": hidden, "learning_rate": lr, "cheb_k": k, "max_epochs": epochs, "num_layers": 1, "batch_size": 1, "weight_decay": 0.0, "dropout": None, "optimizer": "Adam", "classification_loss": "CrossEntropyLoss", "regression_loss": "SmoothL1Loss"} for hidden in (32, 64) for lr in (0.001, 0.01) for k in (2, 3) for epochs in (50, 100)]


def env_versions() -> dict[str, str]:
    import importlib.metadata as md
    import pandas
    import sklearn
    import torch
    import torch_geometric
    import torch_geometric_temporal
    return {"python": sys.version.split()[0], "torch": torch.__version__, "torch_geometric": torch_geometric.__version__, "torch_geometric_temporal_imported": str(getattr(torch_geometric_temporal, "__version__", None)), "torch_geometric_temporal_distribution": md.version("torch-geometric-temporal"), "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pandas.__version__, "device": "cpu"}


def progress_path() -> Path:
    return ROOT / "validation_training_progress.json"


def expected_records() -> list[tuple[str, str, int]]:
    records = []
    for task in RF_TASKS:
        for index in range(len(rf_configs(task))):
            for seed in VALIDATION_SEEDS:
                records.append((task, f"rf_{index + 1:03d}", seed))
    for task in TGNN_TASKS:
        for index in range(len(tgnn_configs())):
            for seed in VALIDATION_SEEDS:
                records.append((task, f"tgnn_{index + 1:03d}", seed))
    return records


def manifest_path(task: str, config_id: str, seed: int) -> Path:
    return ROOT / task / config_id / f"seed_{seed}" / "candidate_manifest.json"


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def verified_completed(task: str, config_id: str, seed: int) -> bool:
    manifest = load_manifest(manifest_path(task, config_id, seed))
    if not manifest or manifest.get("status") != "completed":
        return False
    if manifest.get("code_sha") != CODE_SHA or manifest.get("dataset_bundle_hash") != DATASET_HASH or manifest.get("training_plan_hash") != PLAN_HASH:
        return False
    if manifest.get("train_count") != 7000 or manifest.get("validation_count") != 1500:
        return False
    metrics = manifest.get("validation_metrics")
    if not isinstance(metrics, dict) or not metrics:
        return False
    checkpoint = manifest.get("checkpoint_path")
    if task in TGNN_TASKS:
        if not checkpoint or not Path(checkpoint).is_file() or sha256_file(Path(checkpoint)) != manifest.get("checkpoint_sha256"):
            return False
    return True


def write_progress(current: dict[str, Any] | None = None) -> None:
    rf_expected = sum(len(rf_configs(task)) for task in RF_TASKS) * len(VALIDATION_SEEDS)
    tgnn_expected = len(TGNN_TASKS) * len(tgnn_configs()) * len(VALIDATION_SEEDS)
    counts = {"RF": {"completed": 0, "failed": 0}, "TGNN": {"completed": 0, "failed": 0}}
    for task, config_id, seed in expected_records():
        m = load_manifest(manifest_path(task, config_id, seed))
        family = "RF" if task in RF_TASKS else "TGNN"
        if verified_completed(task, config_id, seed): counts[family]["completed"] += 1
        elif m and m.get("status") == "failed": counts[family]["failed"] += 1
    atomic_json(progress_path(), {"schema_version": "satnet.validation_training_progress.v1", "updated_at": now(), "validation_seeds": list(VALIDATION_SEEDS), "test_accessed": False, "rf": {"expected": rf_expected, "completed": counts["RF"]["completed"], "failed": counts["RF"]["failed"], "pending": rf_expected - counts["RF"]["completed"] - counts["RF"]["failed"]}, "tgnn": {"expected": tgnn_expected, "completed": counts["TGNN"]["completed"], "failed": counts["TGNN"]["failed"], "pending": tgnn_expected - counts["TGNN"]["completed"] - counts["TGNN"]["failed"], "current": current}, "acceptance": {"rf_all_complete": counts["RF"]["completed"] == rf_expected and counts["RF"]["failed"] == 0, "tgnn_all_complete": counts["TGNN"]["completed"] == tgnn_expected and counts["TGNN"]["failed"] == 0, "test_untouched": True}})


def read_rf_data(task_id: str) -> dict[str, Any]:
    spec = TASKS[task_id]
    path = DATASET_ROOT / spec["path"]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        positions = {name: header.index(name) for name in ("run_id", "run_key", "design_id", "realization_id", "split", *spec["features"], spec["target"])}
        rows: dict[str, list[Any]] = {"train": [], "validation": []}
        split_meta: dict[str, list[tuple[int, str, str]]] = {"train": [], "validation": [], "test": []}
        for raw in reader:
            split = raw[positions["split"]].strip().lower()
            run_id = int(raw[positions["run_id"]])
            design = raw[positions["design_id"]]
            realization = raw[positions["realization_id"]]
            split_meta[split].append((run_id, design, realization))
            if split == "test":
                continue
            features = [float(raw[positions[name]]) for name in spec["features"]]
            if spec["type"] == "classification":
                target = 1 if raw[positions[spec["target"]]].strip().lower() in {"1", "true"} else 0
            else:
                target = float(raw[positions[spec["target"]]])
            rows[split].append((run_id, features, target))
    validate_split_metadata(split_meta)
    rows = {name: sorted(value, key=lambda item: item[0]) for name, value in rows.items()}
    return {"X_train": np.asarray([x[1] for x in rows["train"]], dtype=float), "y_train": np.asarray([x[2] for x in rows["train"]]), "X_validation": np.asarray([x[1] for x in rows["validation"]], dtype=float), "y_validation": np.asarray([x[2] for x in rows["validation"]]), "split_meta": split_meta}


def validate_split_metadata(split_meta: dict[str, list[tuple[int, str, str]]]) -> None:
    if {k: len(v) for k, v in split_meta.items()} != EXPECTED_SPLITS:
        raise ValueError(f"Frozen split counts failed: { {k: len(v) for k, v in split_meta.items()} }")
    all_design_splits: dict[str, set[str]] = {}
    for split, rows in split_meta.items():
        ids = {r[0] for r in rows}
        if split != "test" and (len(ids) != len(rows) or ids & set(range(10000)) - ids):
            pass
        design_map: dict[str, set[str]] = {}
        for run_id, design, realization in rows:
            if not 0 <= run_id < 10000 or realization not in {f"R{i:02d}" for i in range(5)}:
                raise ValueError("Invalid frozen identity")
            design_map.setdefault(design, set()).add(realization)
        if any(values != {f"R{i:02d}" for i in range(5)} for values in design_map.values()):
            raise ValueError(f"Realization cluster violation in {split}")
        all_design_splits[split] = set(design_map)
    if any(all_design_splits[a] & all_design_splits[b] for i, a in enumerate(all_design_splits) for b in list(all_design_splits)[i + 1:]):
        raise ValueError("Design crosses frozen splits")


def validation_metrics(task_type: str, y_true: Any, prediction: Any, score: Any | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(SRC))
    from satnet.experiments.final_training.metrics import classification_metrics, regression_metrics
    return classification_metrics(y_true, prediction, score) if task_type == "classification" else regression_metrics(y_true, prediction)


def run_rf(task_id: str, config_id: str, config: dict[str, Any], seed: int, data: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(SRC))
    from satnet.experiments.final_training.seeds import initialize_determinism
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    path = manifest_path(task_id, config_id, seed)
    prior = load_manifest(path)
    if verified_completed(task_id, config_id, seed):
        return prior or {}
    start = time.perf_counter()
    atomic_json(path, {"schema_version": "satnet.validation_candidate_manifest.v1", "status": "running", "started_at": now(), "task": task_id, "model_family": "RF", "config_id": config_id, "configuration": {**config, "random_state": seed, "n_jobs": -1}, "seed": seed, "dataset_bundle_hash": DATASET_HASH, "training_plan_hash": PLAN_HASH, "code_sha": CODE_SHA, "train_count": 7000, "validation_count": 1500, "test_accessed": False, "previous_status": prior.get("status") if prior else None})
    try:
        initialize_determinism(seed)
        params = {**config, "random_state": seed, "n_jobs": -1}
        if TASKS[task_id]["type"] != "classification":
            params.pop("class_weight", None)
        estimator = RandomForestClassifier(**params) if TASKS[task_id]["type"] == "classification" else RandomForestRegressor(**params)
        estimator.fit(data["X_train"], data["y_train"])
        if TASKS[task_id]["type"] == "classification":
            pred = estimator.predict(data["X_validation"])
            score = estimator.predict_proba(data["X_validation"])
            metrics = validation_metrics("classification", data["y_validation"], pred, score)
        else:
            pred = estimator.predict(data["X_validation"])
            metrics = validation_metrics("regression", data["y_validation"], pred)
        result = {"schema_version": "satnet.validation_candidate_manifest.v1", "status": "completed", "completed_at": now(), "task": task_id, "model_family": "RF", "config_id": config_id, "configuration": params, "seed": seed, "dataset_bundle_hash": DATASET_HASH, "training_plan_hash": PLAN_HASH, "code_sha": CODE_SHA, "production_sha": PRODUCTION_SHA, "train_count": 7000, "validation_count": 1500, "fit_runtime_seconds": time.perf_counter() - start, "validation_metrics": metrics, "model_selection_metric": {"name": "balanced_accuracy" if TASKS[task_id]["type"] == "classification" else "mae", "value": metrics["balanced_accuracy"] if TASKS[task_id]["type"] == "classification" else metrics["mae"]}, "test_accessed": False, "environment_versions": env_versions()}
        atomic_json(path, result)
        return result
    except Exception as exc:
        atomic_json(path, {"schema_version": "satnet.validation_candidate_manifest.v1", "status": "failed", "failed_at": now(), "task": task_id, "model_family": "RF", "config_id": config_id, "configuration": config, "seed": seed, "dataset_bundle_hash": DATASET_HASH, "training_plan_hash": PLAN_HASH, "code_sha": CODE_SHA, "train_count": 7000, "validation_count": 1500, "test_accessed": False, "error_type": type(exc).__name__, "error": str(exc)})
        raise


def load_tgnn_metadata(task_id: str) -> dict[str, Any]:
    sys.path.insert(0, str(SRC))
    from satnet.experiments.final_training.tgnn_loader import read_sequence
    graph_path = DATASET_ROOT / "tgnn_space_classification" / "tgnn_space_graph_manifest.jsonl"
    target_path = DATASET_ROOT / TASKS[task_id]["target_path"]
    graph_records: dict[int, dict[str, Any]] = {}
    with graph_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            run_id = int(record["run_id"])
            graph_records[run_id] = record
    targets: dict[int, Any] = {}
    with target_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if str(record.get("split", "")).lower() == "test":
                continue
            run_id = int(record["run_id"])
            value = record["target"]
            targets[run_id] = (1 if str(value).strip().lower() in {"1", "true"} else 0) if TASKS[task_id]["type"] == "classification" else float(value)
    split_meta = {split: [] for split in EXPECTED_SPLITS}
    for record in graph_records.values():
        split = str(record["split"]).lower()
        split_meta[split].append((int(record["run_id"]), str(record["design_id"]), str(record["realization_id"])))
    validate_split_metadata(split_meta)
    train_ids = tuple(sorted(x[0] for x in split_meta["train"]))
    validation_ids = tuple(sorted(x[0] for x in split_meta["validation"]))
    if set(targets) != set(train_ids) | set(validation_ids):
        raise ValueError("TGNN train/validation target identity set mismatch")
    return {"records": graph_records, "targets": targets, "train_ids": train_ids, "validation_ids": validation_ids, "read_sequence": read_sequence}


def tgnn_predict(model: Any, ids: Iterable[int], bundle: dict[str, Any], task_type: str, train: bool, optimizer: Any = None, loss_fn: Any = None) -> tuple[list[float], list[Any], float]:
    import torch
    model.train(train)
    predictions: list[float] = []
    scores: list[Any] = []
    targets: list[Any] = []
    total_loss = 0.0
    for run_id in ids:
        artifact = bundle["read_sequence"](DATASET_ROOT / bundle["records"][run_id]["sequence_artifact"])
        data_list = artifact.data_list()
        target = bundle["targets"][run_id]
        if train:
            optimizer.zero_grad(set_to_none=True)
            output = model(data_list)
            target_tensor = torch.tensor([target], dtype=torch.long if task_type == "classification" else torch.float32)
            loss = loss_fn(output, target_tensor if task_type == "classification" else target_tensor.reshape(1, 1))
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(data_list)
            if task_type == "classification":
                probability = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
                scores.append(probability.tolist())
                predictions.append(float(np.argmax(probability)))
            else:
                predictions.append(float(output.detach().cpu().reshape(-1)[0]))
            targets.append(target)
        if train:
            total_loss += float(loss.detach().cpu().item())
    return predictions, scores if task_type == "classification" else [], total_loss / max(1, len(tuple(ids))) if train else 0.0


def run_tgnn(task_id: str, config_id: str, config: dict[str, Any], seed: int, bundle: dict[str, Any]) -> dict[str, Any]:
    import copy
    import torch
    from torch.nn import CrossEntropyLoss, SmoothL1Loss
    from satnet.experiments.final_training.checkpoints import EarlyStopping
    from satnet.experiments.final_training.seeds import initialize_determinism
    from satnet.models.gnn_model import SatelliteGNN
    path = manifest_path(task_id, config_id, seed)
    prior = load_manifest(path)
    if verified_completed(task_id, config_id, seed):
        return prior or {}
    candidate_dir = path.parent
    checkpoint = candidate_dir / "best_validation_checkpoint.pt"
    log_path = candidate_dir / "progress.jsonl"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    atomic_json(path, {"schema_version": "satnet.validation_candidate_manifest.v1", "status": "running", "started_at": now(), "task": task_id, "model_family": "TGNN", "config_id": config_id, "configuration": config, "seed": seed, "dataset_bundle_hash": DATASET_HASH, "training_plan_hash": PLAN_HASH, "code_sha": CODE_SHA, "train_count": 7000, "validation_count": 1500, "checkpoint_staging_area": str(candidate_dir), "test_accessed": False, "previous_status": prior.get("status") if prior else None})
    try:
        initialize_determinism(seed)
        torch.set_num_threads(8)
        model = SatelliteGNN(node_features=3, hidden_channels=int(config["hidden_dim"]), out_channels=2 if TASKS[task_id]["type"] == "classification" else 1, task_type=TASKS[task_id]["type"], cheb_k=int(config["cheb_k"]))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0)
        loss_fn = CrossEntropyLoss() if TASKS[task_id]["type"] == "classification" else SmoothL1Loss()
        stop = EarlyStopping(TASKS[task_id]["type"], patience=10, min_delta=0.0, restore_best=True)
        best_epoch = None
        for epoch in range(1, int(config["max_epochs"]) + 1):
            _, _, train_loss = tgnn_predict(model, bundle["train_ids"], bundle, TASKS[task_id]["type"], True, optimizer, loss_fn)
            validation_pred, validation_scores, _ = tgnn_predict(model, bundle["validation_ids"], bundle, TASKS[task_id]["type"], False)
            if TASKS[task_id]["type"] == "classification":
                metrics = validation_metrics("classification", [bundle["targets"][i] for i in bundle["validation_ids"]], validation_pred, validation_scores)
                monitor = float(metrics["balanced_accuracy"])
            else:
                metrics = validation_metrics("regression", [bundle["targets"][i] for i in bundle["validation_ids"]], validation_pred)
                monitor = float(metrics["mae"])
            improved = stop.update(epoch, monitor, copy.deepcopy(model.state_dict()))
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "validation_metrics": native(metrics), "monitor": monitor, "improved": improved, "at": now()}) + "\n")
                log.flush()
                os.fsync(log.fileno())
            if improved:
                checkpoint_payload = {"model_state_dict": stop.best_state, "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "configuration": config, "task": task_id, "seed": seed, "validation_primary_metric": monitor, "dataset_bundle_hash": DATASET_HASH, "training_plan_hash": PLAN_HASH, "code_sha": CODE_SHA, "environment_versions": env_versions()}
                temp = checkpoint.with_name(checkpoint.name + f".tmp-{os.getpid()}")
                torch.save(checkpoint_payload, temp)
                os.replace(temp, checkpoint)
                best_epoch = epoch
            if stop.should_stop:
                break
        stop.restore(model)
        # Recompute metrics from the restored best-validation checkpoint/model only.
        validation_pred, validation_scores, _ = tgnn_predict(model, bundle["validation_ids"], bundle, TASKS[task_id]["type"], False)
        metrics = validation_metrics(TASKS[task_id]["type"], [bundle["targets"][i] for i in bundle["validation_ids"]], validation_pred, validation_scores if TASKS[task_id]["type"] == "classification" else None)
        result = {"schema_version": "satnet.validation_candidate_manifest.v1", "status": "completed", "completed_at": now(), "task": task_id, "model_family": "TGNN", "config_id": config_id, "configuration": config, "seed": seed, "dataset_bundle_hash": DATASET_HASH, "training_plan_hash": PLAN_HASH, "code_sha": CODE_SHA, "production_sha": PRODUCTION_SHA, "train_count": 7000, "validation_count": 1500, "fit_runtime_seconds": time.perf_counter() - start, "epochs_run": epoch, "selected_epoch": stop.best_epoch, "max_epochs": config["max_epochs"], "early_stopping": {"monitor": "balanced_accuracy" if TASKS[task_id]["type"] == "classification" else "mae", "direction": "maximize" if TASKS[task_id]["type"] == "classification" else "minimize", "patience": 10, "min_delta": 0.0, "restore_best": True, "stopped_early": epoch < config["max_epochs"]}, "validation_metrics": metrics, "model_selection_metric": {"name": "balanced_accuracy" if TASKS[task_id]["type"] == "classification" else "mae", "value": metrics["balanced_accuracy"] if TASKS[task_id]["type"] == "classification" else metrics["mae"]}, "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "test_accessed": False, "environment_versions": env_versions()}
        atomic_json(path, result)
        return result
    except Exception as exc:
        atomic_json(path, {"schema_version": "satnet.validation_candidate_manifest.v1", "status": "failed", "failed_at": now(), "task": task_id, "model_family": "TGNN", "config_id": config_id, "configuration": config, "seed": seed, "dataset_bundle_hash": DATASET_HASH, "training_plan_hash": PLAN_HASH, "code_sha": CODE_SHA, "train_count": 7000, "validation_count": 1500, "checkpoint_path": str(checkpoint), "test_accessed": False, "error_type": type(exc).__name__, "error": str(exc)})
        raise


def collect_records(family_tasks: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for task in family_tasks:
        for config_id, seed in [(f"rf_{i:03d}", s) for i in range(1, len(rf_configs(task)) + 1) for s in VALIDATION_SEEDS] if task in RF_TASKS else [(f"tgnn_{i:03d}", s) for i in range(1, len(tgnn_configs()) + 1) for s in VALIDATION_SEEDS]:
            manifest = load_manifest(manifest_path(task, config_id, seed))
            if not verified_completed(task, config_id, seed):
                raise RuntimeError(f"Missing or unverified completed record: {task}/{config_id}/{seed}")
            rows.append(manifest)
    return rows


def select_results(task_ids: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in task_ids:
        manifests = collect_records((task,))
        grouped: dict[str, list[dict[str, Any]]] = {}
        for m in manifests: grouped.setdefault(m["config_id"], []).append(m)
        ranking = []
        classification = TASKS[task]["type"] == "classification"
        for candidate_id, records in grouped.items():
            values = [float(m["model_selection_metric"]["value"]) for m in records]
            secondary_key = "macro_f1" if classification else "rmse"
            secondaries = [float(m["validation_metrics"][secondary_key]) for m in records]
            config = records[0]["configuration"]
            ranking.append({"config_id": candidate_id, "configuration": config, "mean_primary_metric": statistics.fmean(values), "std_primary_metric": statistics.pstdev(values), "mean_secondary_metric": statistics.fmean(secondaries), "std_secondary_metric": statistics.pstdev(secondaries), "seed_metrics": [{"seed": m["seed"], "validation_metrics": m["validation_metrics"], "selected_epoch": m.get("selected_epoch"), "epochs_run": m.get("epochs_run")} for m in sorted(records, key=lambda x: x["seed"])], "complexity_key": list(complexity(config))})
        if classification: ranking.sort(key=lambda x: (-x["mean_primary_metric"], -x["mean_secondary_metric"], tuple(x["complexity_key"]), x["config_id"]))
        else: ranking.sort(key=lambda x: (x["mean_primary_metric"], x["mean_secondary_metric"], tuple(x["complexity_key"]), x["config_id"]))
        winner = ranking[0]
        result[task] = {"task": task, "winning_configuration": winner["configuration"], "winning_config_id": winner["config_id"], "mean_primary_metric": winner["mean_primary_metric"], "std_primary_metric": winner["std_primary_metric"], "primary_metric": "balanced_accuracy" if classification else "mae", "direction": "maximize" if classification else "minimize", "seed_specific_validation_metrics": winner["seed_metrics"], "tie_break_evidence": {"ranking_rule": ["higher mean balanced accuracy", "higher mean macro F1", "simpler model"] if classification else ["lower mean MAE", "lower mean RMSE", "simpler model"], "winning_complexity_key": winner["complexity_key"], "runner_up_complexity_key": ranking[1]["complexity_key"] if len(ranking) > 1 else None}, "runner_up_configuration": ranking[1]["configuration"] if len(ranking) > 1 else None, "runner_up": ranking[1] if len(ranking) > 1 else None, "candidate_count": len(ranking), "validation_seeds": list(VALIDATION_SEEDS), "test_accessed": False, "full_ranking": ranking}
    return result


def write_results_csv(path: Path, task_ids: tuple[str, ...]) -> None:
    records = collect_records(task_ids)
    fields = ["task", "config_id", "seed", "configuration_json", "fit_runtime_seconds", "selected_epoch", "epochs_run", "model_selection_metric_name", "model_selection_metric_value", "balanced_accuracy", "accuracy", "precision_by_class", "recall_by_class", "f1_by_class", "macro_f1", "weighted_f1", "specificity", "sensitivity", "confusion_matrix", "roc_auc", "pr_auc", "mae", "rmse", "r2", "median_absolute_error", "maximum_absolute_error", "target_mean", "target_median", "target_standard_deviation", "prediction_mean", "prediction_standard_deviation", "predictions_below_zero", "predictions_above_one", "checkpoint_sha256", "test_accessed"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for m in records:
            row = {field: "" for field in fields}
            metrics = m["validation_metrics"]
            row.update({"task": m["task"], "config_id": m["config_id"], "seed": m["seed"], "configuration_json": json.dumps(m["configuration"], sort_keys=True), "fit_runtime_seconds": m["fit_runtime_seconds"], "selected_epoch": m.get("selected_epoch", ""), "epochs_run": m.get("epochs_run", ""), "model_selection_metric_name": m["model_selection_metric"]["name"], "model_selection_metric_value": m["model_selection_metric"]["value"], "checkpoint_sha256": m.get("checkpoint_sha256", ""), "test_accessed": m["test_accessed"]})
            for key in fields:
                if key in metrics: row[key] = json.dumps(metrics[key], sort_keys=True) if isinstance(metrics[key], (dict, list)) else metrics[key]
            writer.writerow(row)
    os.replace(temp, path)


def baselines() -> dict[str, Any]:
    sys.path.insert(0, str(SRC))
    from satnet.experiments.final_training.baselines import classification_baselines_from_train, regression_baselines_from_train
    output: dict[str, Any] = {"schema_version": "satnet.validation_baselines.v1", "construction_split": "train", "evaluation_split": "validation", "test_accessed": False, "tasks": {}}
    for task in RF_TASKS:
        data = read_rf_data(task)
        if TASKS[task]["type"] == "classification":
            base = classification_baselines_from_train(data["y_train"], seed=42)
            output["tasks"][task] = {
                "majority_classifier": {
                    "train_majority_class": base.majority_class,
                    "validation_metrics": validation_metrics("classification", data["y_validation"], base.majority_predict(len(data["y_validation"]))),
                },
                "stratified_random_classifier": {
                    "train_positive_prevalence": base.positive_prevalence,
                    "seed": 42,
                    "validation_metrics": validation_metrics("classification", data["y_validation"], base.stratified_random_predict(len(data["y_validation"]))),
                },
            }
        else:
            base = regression_baselines_from_train(data["y_train"])
            output["tasks"][task] = {"training_mean_predictor": {"train_mean": base.mean, "validation_metrics": validation_metrics("regression", data["y_validation"], base.mean_predict(len(data["y_validation"])))}, "training_median_predictor": {"train_median": base.median, "validation_metrics": validation_metrics("regression", data["y_validation"], base.median_predict(len(data["y_validation"])))}}
    return output


def bundle_hash(exclude: str) -> tuple[str, list[dict[str, Any]]]:
    entries = []
    digest = hashlib.sha256()
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != exclude):
        rel = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        digest.update(rel.encode("utf-8")); digest.update(b"\0"); digest.update(data)
        entries.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return digest.hexdigest(), entries


def main() -> None:
    os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")
    ROOT.mkdir(parents=True, exist_ok=True)
    write_progress({"phase": "preflight", "current": None})
    sys.path.insert(0, str(SRC))
    from satnet.experiments.final_training.contracts import verify_dataset_bundle, verify_training_plan
    verify_dataset_bundle(DATASET_ROOT, full=True)
    verify_training_plan(PLAN_ROOT)
    atomic_json(ROOT / "manifests" / "execution_identity.json", {"implementation_sha": CODE_SHA, "production_sha": PRODUCTION_SHA, "dataset_bundle_hash": DATASET_HASH, "training_plan_hash": PLAN_HASH, "validation_seeds": list(VALIDATION_SEEDS), "test_accessed": False, "execution_worktree": str(REPO), "started_at": now(), "environment_versions": env_versions()})
    rf_start = time.perf_counter()
    for task in RF_TASKS:
        data = read_rf_data(task)
        configs = rf_configs(task)
        for index, config in enumerate(configs, 1):
            for seed in VALIDATION_SEEDS:
                write_progress({"phase": "rf", "task": task, "config_id": f"rf_{index:03d}", "seed": seed})
                run_rf(task, f"rf_{index:03d}", config, seed, data)
    write_results_csv(ROOT / "rf_validation_results.csv", RF_TASKS)
    rf_selection_path = ROOT / "rf_validation_selection.json"
    if not rf_selection_path.is_file():
        rf_selection = select_results(RF_TASKS)
        atomic_json(rf_selection_path, {"schema_version": "satnet.rf_validation_selection.v1", "selection_split": "validation", "fit_split": "train", "validation_seeds": list(VALIDATION_SEEDS), "tasks": rf_selection, "test_accessed": False, "frozen_at": now()})
        atomic_json(ROOT / "manifests" / "rf_selection_freeze.json", {"sha256": sha256_file(rf_selection_path), "frozen_at": now(), "test_accessed": False})
    elif not (ROOT / "manifests" / "rf_selection_freeze.json").is_file():
        atomic_json(ROOT / "manifests" / "rf_selection_freeze.json", {"sha256": sha256_file(rf_selection_path), "frozen_at": now(), "test_accessed": False})
    rf_runtime_path = ROOT / "manifests" / "rf_phase_runtime.json"
    if not rf_runtime_path.is_file():
        atomic_json(rf_runtime_path, {"runtime_seconds": time.perf_counter() - rf_start, "expected_fits": 756, "completed_fits": 756, "failed_fits": 0, "test_accessed": False})
    tgnn_start = time.perf_counter()
    for task in TGNN_TASKS:
        bundle = load_tgnn_metadata(task)
        configs = tgnn_configs()
        for index, config in enumerate(configs, 1):
            for seed in VALIDATION_SEEDS:
                write_progress({"phase": "tgnn", "task": task, "config_id": f"tgnn_{index:03d}", "seed": seed})
                run_tgnn(task, f"tgnn_{index:03d}", config, seed, bundle)
    write_results_csv(ROOT / "tgnn_validation_results.csv", TGNN_TASKS)
    tgnn_selection_path = ROOT / "tgnn_validation_selection.json"
    if not tgnn_selection_path.is_file():
        tgnn_selection = select_results(TGNN_TASKS)
        atomic_json(tgnn_selection_path, {"schema_version": "satnet.tgnn_validation_selection.v1", "selection_split": "validation", "fit_split": "train", "validation_seeds": list(VALIDATION_SEEDS), "early_stopping": {"monitor_classification": "balanced_accuracy", "monitor_regression": "mae", "patience": 10, "min_delta": 0.0, "restore_best": True}, "tasks": tgnn_selection, "test_accessed": False, "frozen_at": now()})
        atomic_json(ROOT / "manifests" / "tgnn_selection_freeze.json", {"sha256": sha256_file(tgnn_selection_path), "frozen_at": now(), "test_accessed": False})
    elif not (ROOT / "manifests" / "tgnn_selection_freeze.json").is_file():
        atomic_json(ROOT / "manifests" / "tgnn_selection_freeze.json", {"sha256": sha256_file(tgnn_selection_path), "frozen_at": now(), "test_accessed": False})
    tgnn_runtime_path = ROOT / "manifests" / "tgnn_phase_runtime.json"
    if not tgnn_runtime_path.is_file():
        atomic_json(tgnn_runtime_path, {"runtime_seconds": time.perf_counter() - tgnn_start, "expected_fits": 96, "completed_fits": 96, "failed_fits": 0, "test_accessed": False})
    atomic_json(ROOT / "validation_baselines.json", baselines())
    write_progress({"phase": "complete", "task": None, "config_id": None, "seed": None})
    summary = {"schema_version": "satnet.validation_training_summary.v1", "status": "complete", "implementation_sha": CODE_SHA, "production_sha": PRODUCTION_SHA, "dataset_bundle_hash": DATASET_HASH, "training_plan_hash": PLAN_HASH, "rf_expected": 756, "rf_completed": 756, "rf_failed": 0, "tgnn_expected": 96, "tgnn_completed": 96, "tgnn_failed": 0, "selection_used_only_validation": True, "test_targets_accessed": False, "test_predictions_generated": False, "test_metrics_calculated": False, "final_five_seed_training_run": False, "remaining_blocker": None, "generated_at": now()}
    atomic_json(ROOT / "validation_training_summary.json", summary)
    report = "# Validation Training Report\n\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n\nAll candidates were evaluated using TRAIN fitting and VALIDATION-only metrics. TEST targets, predictions, and metrics were not accessed or generated. The final five-seed robustness phase was not run.\n"
    (ROOT / "validation_training_report.md").write_text(report, encoding="utf-8")
    bundle, entries = bundle_hash("validation_training_inventory.json")
    atomic_json(ROOT / "validation_training_inventory.json", {"schema_version": "satnet.validation_training_inventory.v1", "inventory_self_excluding": True, "hash_algorithm": "sha256 over sorted relative UTF-8 path + NUL byte + raw file bytes", "bundle_sha256": bundle, "artifacts": entries, "test_accessed": False})
    print(json.dumps({"status": "complete", "bundle_sha256": bundle, "root": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
