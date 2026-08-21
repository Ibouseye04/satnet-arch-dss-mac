from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np

# Authoritative identities and paths (frozen/qualified)
REPO = Path(r"C:\Users\johns\external\satnet-10k-training-worktree-v1")
SRC = REPO / "src"
DATASET_ROOT = Path(r"C:\Users\johns\external\satnet-10k-final-ml-datasets-v1-final")
PLAN_ROOT = Path(r"C:\Users\johns\external\satnet-10k-training-test-plan-v1")
ROOT = Path(r"C:\Users\johns\external\satnet-10k-model-training-v1")
FINAL_ROOT = ROOT / "final_robustness"

DATASET_HASH = "38dacd66432bfa660410ff9cab7f181a53151120e8102dc40df57016f78bbfc3"
PLAN_HASH = "e14ec5e5b2221aa2c9aa187a2baacc517e676f9dc748612c206eb1dc43788a0a"
PRODUCTION_SHA = "d0515088cf3fca06a6aa2d47059269089dcb10a7"
CODE_SHA = "e55dba59e83864d2dd11fa47482bd4ec2bdd797a"

# Validation driver file, used for preflight identity check
VALIDATION_DRIVER = ROOT / "run_validation_training.py"
KNOWN_VALIDATION_DRIVER_SHA256 = "F9178B2C7BF5D09279F363B966CFDD565F1C4DD81CA418D0696706D590248A56"

# Frozen validation selections (must match exactly)
RF_SELECTION_PATH = ROOT / "rf_validation_selection.json"
TGNN_SELECTION_PATH = ROOT / "tgnn_validation_selection.json"
FROZEN_RF_SELECTION_SHA256 = "a1b7076197f123ecf62b0b79ddd092bdb66f0d7d8b94d6c113dc201744163911"
FROZEN_TGNN_SELECTION_SHA256 = "c85dfdcd3b6a62d741675ac522fda76bd3de7dd4c311a82bc1b12e079dab2bfb"

# Robustness seeds (exactly these five)
ROBUSTNESS_SEEDS = (42, 123, 456, 789, 2026)

# Expected frozen split sizes
EXPECTED_SPLITS = {"train": 7000, "validation": 1500, "test": 1500}

# Task registry (authoritative)
RF_TASKS = (
    "rf_space_classification",
    "rf_space_regression",
    "rf_integrated_regression_mean",
    "rf_integrated_regression_min",
    "rf_integrated_classification",
)
TGNN_TASKS = ("tgnn_space_classification", "tgnn_space_regression")

# Winning configuration expectations for sanity checks (will not override frozen selections)
EXPECTED_WINNERS = {
    "rf_space_classification": {"winning_config_id": "rf_022", "configuration": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5, "max_features": "sqrt", "bootstrap": True, "class_weight": "balanced"}},
    "rf_space_regression": {"winning_config_id": "rf_036", "configuration": {"n_estimators": 600, "max_depth": 20, "min_samples_leaf": 5, "max_features": 1.0, "bootstrap": True}},
    "rf_integrated_regression_mean": {"winning_config_id": "rf_006", "configuration": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5, "max_features": 1.0, "bootstrap": True}},
    "rf_integrated_regression_min": {"winning_config_id": "rf_018", "configuration": {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 5, "max_features": 1.0, "bootstrap": True}},
    "rf_integrated_classification": {"winning_config_id": "rf_023", "configuration": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5, "max_features": 1.0, "bootstrap": True, "class_weight": "balanced"}},
    "tgnn_space_classification": {"winning_config_id": "tgnn_002", "configuration": {"hidden_dim": 32, "learning_rate": 0.001, "cheb_k": 2, "max_epochs": 100, "num_layers": 1, "batch_size": 1, "weight_decay": 0.0, "dropout": None, "optimizer": "Adam", "classification_loss": "CrossEntropyLoss"}},
    "tgnn_space_regression": {"winning_config_id": "tgnn_010", "configuration": {"hidden_dim": 64, "learning_rate": 0.001, "cheb_k": 2, "max_epochs": 100, "num_layers": 1, "batch_size": 1, "weight_decay": 0.0, "dropout": None, "optimizer": "Adam", "regression_loss": "SmoothL1Loss"}},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for block in iter(lambda: h.read(1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def env_versions() -> dict[str, str]:
    import importlib.metadata as md
    import pandas
    import sklearn
    import torch
    import torch_geometric
    import torch_geometric_temporal

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "torch_geometric_temporal_imported": str(getattr(torch_geometric_temporal, "__version__", None)),
        "torch_geometric_temporal_distribution": md.version("torch-geometric-temporal"),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "device": "cpu",
    }


def bundle_hash(root: Path, exclude: str) -> tuple[str, list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    d = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != exclude):
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        d.update(rel.encode("utf-8")); d.update(b"\0"); d.update(data)
        entries.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return d.hexdigest(), entries


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_git_head(expected_sha: str) -> None:
    # Best-effort Git HEAD verification; read-only.
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True, check=True)
        head = result.stdout.strip()
        if head != expected_sha:
            raise RuntimeError(f"Git HEAD mismatch: {head} != {expected_sha}")
    except FileNotFoundError:
        # Git not present — fail closed because HEAD identity is required.
        raise RuntimeError("Git not available to verify implementation identity")


def preflight() -> dict[str, Any]:
    os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)

    # Verify dataset and training-plan bundles using qualified code
    sys.path.insert(0, str(SRC))
    from satnet.experiments.final_training.contracts import verify_dataset_bundle, verify_training_plan

    verify_dataset_bundle(DATASET_ROOT, full=True)
    verify_training_plan(PLAN_ROOT)

    # Verify validation summary
    summary_path = ROOT / "validation_training_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Missing validation_training_summary.json")
    summary = read_json(summary_path)
    conditions = [
        summary.get("status") == "complete",
        summary.get("rf_expected") == 756,
        summary.get("rf_completed") == 756,
        summary.get("rf_failed") == 0,
        summary.get("tgnn_expected") == 96,
        summary.get("tgnn_completed") == 96,
        summary.get("tgnn_failed") == 0,
        summary.get("selection_used_only_validation") is True,
        summary.get("test_targets_accessed") is False,
        summary.get("test_predictions_generated") is False,
        summary.get("test_metrics_calculated") is False,
        summary.get("final_five_seed_training_run") is False,
    ]
    if not all(conditions):
        raise RuntimeError("Validation summary preconditions failed")

    # Verify selection SHAs (and that files exist)
    if not RF_SELECTION_PATH.is_file() or not TGNN_SELECTION_PATH.is_file():
        raise RuntimeError("Missing frozen selection files")
    rf_sel_sha = sha256_file(RF_SELECTION_PATH)
    tgnn_sel_sha = sha256_file(TGNN_SELECTION_PATH)
    if rf_sel_sha.lower() != FROZEN_RF_SELECTION_SHA256.lower() or tgnn_sel_sha.lower() != FROZEN_TGNN_SELECTION_SHA256.lower():
        raise RuntimeError("Frozen selection SHA mismatch")

    # Verify validation driver identity
    if not VALIDATION_DRIVER.is_file():
        raise RuntimeError("Missing validation execution driver")
    driver_sha = sha256_file(VALIDATION_DRIVER).upper()
    if driver_sha != KNOWN_VALIDATION_DRIVER_SHA256:
        raise RuntimeError(f"Validation driver SHA mismatch: {driver_sha}")

    # Verify implementation identity via Git
    verify_git_head(CODE_SHA)

    # Load winners (will also be used later)
    rf_selection = read_json(RF_SELECTION_PATH)
    tgnn_selection = read_json(TGNN_SELECTION_PATH)

    rf_tasks = rf_selection.get("tasks", {})
    tgnn_tasks = tgnn_selection.get("tasks", {})
    if len(rf_tasks) != 5 or len(tgnn_tasks) != 2:
        raise RuntimeError("Winner count mismatch")

    # Optional sanity check against expected winners
    for task, spec in EXPECTED_WINNERS.items():
        if task in rf_tasks or task in tgnn_tasks:
            record = (rf_tasks.get(task) or tgnn_tasks.get(task))
            if record.get("winning_config_id") != spec["winning_config_id"]:
                raise RuntimeError(f"Winning config ID mismatch for {task}")

    # Print the exact precheck message
    print("PRECHECK PASS")
    print("RF winner count: 5")
    print("TGNN winner count: 2")
    print("robustness seeds: [42,123,456,789,2026]")
    print("expected RF fits: 25")
    print("expected TGNN fits: 10")
    print("expected total fits: 35")
    print("TEST TARGET ACCESS: FALSE")

    # Write reconciliation artifact
    atomic_json(
        FINAL_ROOT / "final_robustness_fit_count_reconciliation.json",
        {
            "schema_version": "satnet.final_robustness_fit_count_reconciliation.v1",
            "rf_tasks": 5,
            "tgnn_tasks": 2,
            "seeds_per_task": 5,
            "rf_expected": 25,
            "tgnn_expected": 10,
            "total_expected": 35,
            "note": "Arithmetic/bookkeeping reconciliation only; no change to any task, model, feature, target, seed, split, configuration, metric, or scientific procedure.",
        },
    )

    # Identity manifest for this phase
    driver_sha256 = sha256_file(Path(__file__))
    atomic_json(
        FINAL_ROOT / "manifests" / "final_robustness_execution_identity.json",
        {
            "schema_version": "satnet.final_robustness_execution_identity.v1",
            "implementation_sha": CODE_SHA,
            "production_sha": PRODUCTION_SHA,
            "dataset_bundle_hash": DATASET_HASH,
            "training_plan_hash": PLAN_HASH,
            "rf_selection_sha256": rf_sel_sha,
            "tgnn_selection_sha256": tgnn_sel_sha,
            "known_validation_driver_sha256": KNOWN_VALIDATION_DRIVER_SHA256,
            "observed_validation_driver_sha256": driver_sha,
            "final_robustness_driver_sha256": driver_sha256,
            "execution_worktree": str(REPO),
            "qualified_python": sys.executable,
            "environment_versions": env_versions(),
            "started_at": now(),
            "test_accessed": False,
        },
    )

    return {"rf": rf_tasks, "tgnn": tgnn_tasks, "rf_sel_sha": rf_sel_sha, "tgnn_sel_sha": tgnn_sel_sha}


def progress_path() -> Path:
    return FINAL_ROOT / "final_robustness_progress.json"


def write_progress(current: dict[str, Any] | None, rf_completed: int, rf_failed: int, tgnn_completed: int, tgnn_failed: int) -> None:
    atomic_json(
        progress_path(),
        {
            "schema_version": "satnet.final_robustness_progress.v1",
            "updated_at": now(),
            "validation_only": True,
            "rf": {"expected": 25, "completed": rf_completed, "failed": rf_failed, "pending": 25 - rf_completed - rf_failed},
            "tgnn": {"expected": 10, "completed": tgnn_completed, "failed": tgnn_failed, "pending": 10 - tgnn_completed - tgnn_failed, "current": current},
            "overall": {"expected": 35, "completed": rf_completed + tgnn_completed, "failed": rf_failed + tgnn_failed, "pending": 35 - (rf_completed + tgnn_completed) - (rf_failed + tgnn_failed)},
            "test_accessed": False,
        },
    )


def read_rf_data_from_validation(task_id: str) -> dict[str, Any]:
    # Reuse qualified helper from validation driver for exact semantics
    sys.path.insert(0, str(ROOT))
    import run_validation_training as rvt

    return rvt.read_rf_data(task_id)


def validation_metrics_from_validation(task_type: str, y_true: Any, prediction: Any, score: Any | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    import run_validation_training as rvt

    return rvt.validation_metrics(task_type, y_true, prediction, score)


def load_tgnn_metadata_from_validation(task_id: str) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    import run_validation_training as rvt

    return rvt.load_tgnn_metadata(task_id)


def tgnn_predict_from_validation(model: Any, ids: Iterable[int], bundle: dict[str, Any], task_type: str, train: bool, optimizer: Any = None, loss_fn: Any = None) -> Tuple[list[float], list[Any], float]:
    sys.path.insert(0, str(ROOT))
    import run_validation_training as rvt

    return rvt.tgnn_predict(model, ids, bundle, task_type, train, optimizer, loss_fn)


def rf_final_manifest_path(task: str, seed: int) -> Path:
    return FINAL_ROOT / task / f"seed_{seed}" / "final_manifest.json"


def tgnn_final_manifest_path(task: str, seed: int) -> Path:
    return FINAL_ROOT / task / f"seed_{seed}" / "final_manifest.json"


def verified_completed_final(task: str, seed: int, family: str, winning_config_id: str, selection_sha: str) -> bool:
    path = (rf_final_manifest_path if family == "RF" else tgnn_final_manifest_path)(task, seed)
    if not path.is_file():
        return False
    try:
        m = read_json(path)
    except Exception:
        return False
    if m.get("status") != "completed":
        return False
    if m.get("task") != task or m.get("seed") != seed or m.get("model_family") != family:
        return False
    if m.get("winning_config_id") != winning_config_id:
        return False
    if m.get("code_sha") != CODE_SHA or m.get("dataset_bundle_hash") != DATASET_HASH or m.get("training_plan_hash") != PLAN_HASH:
        return False
    if str(m.get("selection_sha256", "")).lower() != selection_sha.lower():
        return False
    if m.get("test_accessed") is not False:
        return False
    # Verify saved artifact integrity
    if family == "RF":
        model_path = Path(m.get("model_path", ""))
        if not model_path.is_file() or sha256_file(model_path) != m.get("model_sha256"):
            return False
    else:
        checkpoint_path = Path(m.get("checkpoint_path", ""))
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != m.get("checkpoint_sha256"):
            return False
    return True


def run_rf_final(task: str, seed: int, winning: dict[str, Any], selection_sha: str) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    sys.path.insert(0, str(SRC))
    from satnet.experiments.final_training.seeds import initialize_determinism

    manifest_path = rf_final_manifest_path(task, seed)

    # Skip verified completed
    if verified_completed_final(task, seed, "RF", winning["winning_config_id"], selection_sha):
        return read_json(manifest_path)

    spec = winning["winning_configuration"].copy()
    # Set the robustness seed (override any serialized random_state)
    params = {**spec, "random_state": seed, "n_jobs": -1}
    if task.endswith("classification"):
        estimator = RandomForestClassifier(**params)
    else:
        # Remove class_weight if present for regression
        params.pop("class_weight", None)
        estimator = RandomForestRegressor(**params)

    data = read_rf_data_from_validation(task)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    atomic_json(
        manifest_path,
        {
            "schema_version": "satnet.final_robustness_manifest.v1",
            "status": "running",
            "started_at": now(),
            "task": task,
            "model_family": "RF",
            "winning_config_id": winning["winning_config_id"],
            "configuration": spec,
            "seed": seed,
            "dataset_bundle_hash": DATASET_HASH,
            "training_plan_hash": PLAN_HASH,
            "selection_sha256": selection_sha,
            "code_sha": CODE_SHA,
            "production_sha": PRODUCTION_SHA,
            "train_count": EXPECTED_SPLITS["train"],
            "validation_count": EXPECTED_SPLITS["validation"],
            "test_accessed": False,
        },
    )

    try:
        initialize_determinism(seed)
        estimator.fit(data["X_train"], data["y_train"])  # TRAIN only
        if task.endswith("classification"):
            pred = estimator.predict(data["X_validation"])  # VALIDATION only
            score = estimator.predict_proba(data["X_validation"])  # VALIDATION only
            metrics = validation_metrics_from_validation("classification", data["y_validation"], pred, score)
        else:
            pred = estimator.predict(data["X_validation"])  # VALIDATION only
            metrics = validation_metrics_from_validation("regression", data["y_validation"], pred)

        # Persist model for future TEST inference
        model_path = manifest_path.parent / "final_model.joblib"
        try:
            import joblib
            joblib.dump(estimator, model_path)
        except Exception:
            # Fall back to pickle if joblib unavailable
            import pickle
            with model_path.open("wb") as fh:
                pickle.dump(estimator, fh)
        model_sha = sha256_file(model_path)

        result = {
            "schema_version": "satnet.final_robustness_manifest.v1",
            "status": "completed",
            "completed_at": now(),
            "task": task,
            "model_family": "RF",
            "winning_config_id": winning["winning_config_id"],
            "configuration": spec,
            "seed": seed,
            "dataset_bundle_hash": DATASET_HASH,
            "training_plan_hash": PLAN_HASH,
            "selection_sha256": selection_sha,
            "code_sha": CODE_SHA,
            "production_sha": PRODUCTION_SHA,
            "train_count": EXPECTED_SPLITS["train"],
            "validation_count": EXPECTED_SPLITS["validation"],
            "fit_runtime_seconds": time.perf_counter() - start,
            "validation_metrics": metrics,
            "model_selection_metric": {"name": "balanced_accuracy" if task.endswith("classification") else "mae", "value": metrics["balanced_accuracy"] if task.endswith("classification") else metrics["mae"]},
            "model_path": str(model_path),
            "model_sha256": model_sha,
            "test_accessed": False,
            "environment_versions": env_versions(),
        }
        atomic_json(manifest_path, result)
        return result
    except Exception as exc:
        atomic_json(
            manifest_path,
            {
                "schema_version": "satnet.final_robustness_manifest.v1",
                "status": "failed",
                "failed_at": now(),
                "task": task,
                "model_family": "RF",
                "winning_config_id": winning["winning_config_id"],
                "configuration": spec,
                "seed": seed,
                "dataset_bundle_hash": DATASET_HASH,
                "training_plan_hash": PLAN_HASH,
                "selection_sha256": selection_sha,
                "code_sha": CODE_SHA,
                "train_count": EXPECTED_SPLITS["train"],
                "validation_count": EXPECTED_SPLITS["validation"],
                "test_accessed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def run_tgnn_final(task: str, seed: int, winning: dict[str, Any], selection_sha: str) -> dict[str, Any]:
    import copy
    import torch
    from torch.nn import CrossEntropyLoss, SmoothL1Loss
    sys.path.insert(0, str(SRC))
    from satnet.experiments.final_training.checkpoints import EarlyStopping
    from satnet.experiments.final_training.seeds import initialize_determinism
    from satnet.models.gnn_model import SatelliteGNN

    manifest_path = tgnn_final_manifest_path(task, seed)

    # Skip verified completed
    if verified_completed_final(task, seed, "TGNN", winning["winning_config_id"], selection_sha):
        return read_json(manifest_path)

    bundle = load_tgnn_metadata_from_validation(task)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    config = winning["winning_configuration"].copy()

    atomic_json(
        manifest_path,
        {
            "schema_version": "satnet.final_robustness_manifest.v1",
            "status": "running",
            "started_at": now(),
            "task": task,
            "model_family": "TGNN",
            "winning_config_id": winning["winning_config_id"],
            "configuration": config,
            "seed": seed,
            "dataset_bundle_hash": DATASET_HASH,
            "training_plan_hash": PLAN_HASH,
            "selection_sha256": selection_sha,
            "code_sha": CODE_SHA,
            "production_sha": PRODUCTION_SHA,
            "train_count": EXPECTED_SPLITS["train"],
            "validation_count": EXPECTED_SPLITS["validation"],
            "test_accessed": False,
        },
    )

    try:
        initialize_determinism(seed)
        torch.set_num_threads(8)
        model = SatelliteGNN(
            node_features=3,
            hidden_channels=int(config["hidden_dim"]),
            out_channels=2 if task.endswith("classification") else 1,
            task_type="classification" if task.endswith("classification") else "regression",
            cheb_k=int(config["cheb_k"]),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0)
        loss_fn = CrossEntropyLoss() if task.endswith("classification") else SmoothL1Loss()
        stop = EarlyStopping("classification" if task.endswith("classification") else "regression", patience=10, min_delta=0.0, restore_best=True)

        candidate_dir = manifest_path.parent
        checkpoint = candidate_dir / "best_validation_checkpoint.pt"
        best_epoch = None
        for epoch in range(1, int(config["max_epochs"]) + 1):
            _, _, train_loss = tgnn_predict_from_validation(model, bundle["train_ids"], bundle, "classification" if task.endswith("classification") else "regression", True, optimizer, loss_fn)
            validation_pred, validation_scores, _ = tgnn_predict_from_validation(model, bundle["validation_ids"], bundle, "classification" if task.endswith("classification") else "regression", False)
            if task.endswith("classification"):
                metrics = validation_metrics_from_validation("classification", [bundle["targets"][i] for i in bundle["validation_ids"]], validation_pred, validation_scores)
                monitor = float(metrics["balanced_accuracy"])
            else:
                metrics = validation_metrics_from_validation("regression", [bundle["targets"][i] for i in bundle["validation_ids"]], validation_pred)
                monitor = float(metrics["mae"])
            improved = stop.update(epoch, monitor, copy.deepcopy(model.state_dict()))
            # Minimal per-epoch log
            with (candidate_dir / "progress.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "validation_metrics": metrics, "monitor": monitor, "improved": improved, "at": now()}, sort_keys=True) + "\n")
                log.flush(); os.fsync(log.fileno())
            if improved:
                payload = {"model_state_dict": stop.best_state, "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "configuration": config, "task": task, "seed": seed, "validation_primary_metric": monitor, "dataset_bundle_hash": DATASET_HASH, "training_plan_hash": PLAN_HASH, "code_sha": CODE_SHA, "environment_versions": env_versions()}
                temp = checkpoint.with_name(checkpoint.name + f".tmp-{os.getpid()}")
                torch.save(payload, temp)
                os.replace(temp, checkpoint)
                best_epoch = epoch
            if stop.should_stop:
                break
        stop.restore(model)
        # Recompute metrics from best checkpoint/model only
        validation_pred, validation_scores, _ = tgnn_predict_from_validation(model, bundle["validation_ids"], bundle, "classification" if task.endswith("classification") else "regression", False)
        metrics = validation_metrics_from_validation("classification" if task.endswith("classification") else "regression", [bundle["targets"][i] for i in bundle["validation_ids"]], validation_pred, validation_scores if task.endswith("classification") else None)

        result = {
            "schema_version": "satnet.final_robustness_manifest.v1",
            "status": "completed",
            "completed_at": now(),
            "task": task,
            "model_family": "TGNN",
            "winning_config_id": winning["winning_config_id"],
            "configuration": config,
            "seed": seed,
            "dataset_bundle_hash": DATASET_HASH,
            "training_plan_hash": PLAN_HASH,
            "selection_sha256": selection_sha,
            "code_sha": CODE_SHA,
            "production_sha": PRODUCTION_SHA,
            "train_count": EXPECTED_SPLITS["train"],
            "validation_count": EXPECTED_SPLITS["validation"],
            "fit_runtime_seconds": time.perf_counter() - start,
            "epochs_run": int(best_epoch if best_epoch is not None else config["max_epochs"]),
            "selected_epoch": int(best_epoch if best_epoch is not None else config["max_epochs"]),
            "max_epochs": int(config["max_epochs"]),
            "early_stopping": {
                "monitor": "balanced_accuracy" if task.endswith("classification") else "mae",
                "direction": "maximize" if task.endswith("classification") else "minimize",
                "patience": 10,
                "min_delta": 0.0,
                "restore_best": True,
                "stopped_early": (best_epoch is not None) and (best_epoch < int(config["max_epochs"]))
            },
            "validation_metrics": metrics,
            "model_selection_metric": {"name": "balanced_accuracy" if task.endswith("classification") else "mae", "value": metrics["balanced_accuracy"] if task.endswith("classification") else metrics["mae"]},
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "test_accessed": False,
            "environment_versions": env_versions(),
        }
        atomic_json(manifest_path, result)
        return result
    except Exception as exc:
        atomic_json(
            manifest_path,
            {
                "schema_version": "satnet.final_robustness_manifest.v1",
                "status": "failed",
                "failed_at": now(),
                "task": task,
                "model_family": "TGNN",
                "winning_config_id": winning["winning_config_id"],
                "configuration": config,
                "seed": seed,
                "dataset_bundle_hash": DATASET_HASH,
                "training_plan_hash": PLAN_HASH,
                "selection_sha256": selection_sha,
                "code_sha": CODE_SHA,
                "train_count": EXPECTED_SPLITS["train"],
                "validation_count": EXPECTED_SPLITS["validation"],
                "test_accessed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def summarize_validation_metrics(manifests: list[dict[str, Any]], primary: str) -> dict[str, Any]:
    values = [float(m["validation_metrics"][primary]) for m in manifests]
    seeds = [int(m["seed"]) for m in manifests]
    per_seed = {int(m["seed"]): float(m["validation_metrics"][primary]) for m in manifests}
    return {
        "primary_metric": primary,
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "seed_42": per_seed.get(42),
        "seed_values": {str(k): v for k, v in sorted(per_seed.items())},
    }


def summarize_phase(rf_winners: dict[str, Any], tgnn_winners: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"schema_version": "satnet.final_robustness_summary.v1", "generated_at": now(), "test_accessed": False, "tasks": {}}
    for task, winning in rf_winners.items():
        manifests = [read_json(rf_final_manifest_path(task, s)) for s in ROBUSTNESS_SEEDS]
        primary = "balanced_accuracy" if task.endswith("classification") else "mae"
        summary["tasks"][task] = summarize_validation_metrics(manifests, primary)
        summary["tasks"][task]["winning_config_id"] = winning["winning_config_id"]
    for task, winning in tgnn_winners.items():
        manifests = [read_json(tgnn_final_manifest_path(task, s)) for s in ROBUSTNESS_SEEDS]
        primary = "balanced_accuracy" if task.endswith("classification") else "mae"
        summary["tasks"][task] = summarize_validation_metrics(manifests, primary)
        summary["tasks"][task]["winning_config_id"] = winning["winning_config_id"]
    return summary


def write_inventory() -> None:
    bundle, entries = bundle_hash(FINAL_ROOT, "final_robustness_inventory.json")
    atomic_json(
        FINAL_ROOT / "final_robustness_inventory.json",
        {
            "schema_version": "satnet.final_robustness_inventory.v1",
            "inventory_self_excluding": True,
            "hash_algorithm": "sha256 over sorted relative UTF-8 path + NUL byte + raw file bytes",
            "bundle_sha256": bundle,
            "artifacts": entries,
            "test_accessed": False,
        },
    )


def main() -> None:
    # Ensure import path for qualified modules
    os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")

    # Preflight (raises on any violation)
    winners = preflight()

    rf_winners: dict[str, Any] = winners["rf"]
    tgnn_winners: dict[str, Any] = winners["tgnn"]

    rf_sel_sha = winners["rf_sel_sha"]
    tgnn_sel_sha = winners["tgnn_sel_sha"]

    rf_completed = 0
    rf_failed = 0
    tgnn_completed = 0
    tgnn_failed = 0

    write_progress(None, rf_completed, rf_failed, tgnn_completed, tgnn_failed)

    # RF final fits (25 expected)
    for task in RF_TASKS:
        winning = rf_winners[task]
        for seed in ROBUSTNESS_SEEDS:
            current = {"task": task, "model_family": "RF", "seed": seed, "winning_config_id": winning["winning_config_id"]}
            write_progress(current, rf_completed, rf_failed, tgnn_completed, tgnn_failed)
            try:
                run_rf_final(task, seed, winning, rf_sel_sha)
                rf_completed += 1
            except Exception:
                rf_failed += 1
                raise
            finally:
                write_progress(None, rf_completed, rf_failed, tgnn_completed, tgnn_failed)

    # TGNN final fits (10 expected), strictly sequential
    for task in TGNN_TASKS:
        winning = tgnn_winners[task]
        for seed in ROBUSTNESS_SEEDS:
            current = {"task": task, "model_family": "TGNN", "seed": seed, "winning_config_id": winning["winning_config_id"]}
            write_progress(current, rf_completed, rf_failed, tgnn_completed, tgnn_failed)
            try:
                run_tgnn_final(task, seed, winning, tgnn_sel_sha)
                tgnn_completed += 1
            except Exception:
                tgnn_failed += 1
                raise
            finally:
                write_progress(None, rf_completed, rf_failed, tgnn_completed, tgnn_failed)

    # Summaries (validation metrics only)
    summary = summarize_phase(rf_winners, tgnn_winners)
    atomic_json(FINAL_ROOT / "final_robustness_summary.json", summary)

    # Human-readable report
    report = "# Final Five-Seed Robustness Training Summary\n\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n\nValidation metrics only. TEST remains sealed.\n"
    (FINAL_ROOT / "final_robustness_report.md").write_text(report, encoding="utf-8")

    # Final acceptance checks (progress-style)
    write_progress(None, rf_completed, rf_failed, tgnn_completed, tgnn_failed)

    # Inventory (self-excluding)
    write_inventory()

    # Completion line (exact string)
    if rf_completed == 25 and rf_failed == 0 and tgnn_completed == 10 and tgnn_failed == 0:
        print("FINAL FIVE-SEED ROBUSTNESS TRAINING COMPLETE — READY FOR HELD-OUT TEST AUTHORIZATION")


if __name__ == "__main__":
    main()
