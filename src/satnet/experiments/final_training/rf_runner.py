from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import AUTHORIZED_TASKS, DATASET_ROOT, verify_dataset_bundle
from .rf_loader import RFDataBundle, load_rf_task
from .search import candidate_counts, make_rf_estimator, rf_configs


class TrainingDisabledError(RuntimeError):
    """Raised by the compatibility-only control plane instead of fitting a model."""


def dry_run_rf_tasks(*, dataset_root: Path = DATASET_ROOT) -> dict[str, Any]:
    verify_dataset_bundle(dataset_root, full=True)
    result: dict[str, Any] = {"fit_called": False, "tasks": {}}
    for task_id, task in AUTHORIZED_TASKS.items():
        if task.family != "RF":
            continue
        bundle = load_rf_task(task_id, dataset_root=dataset_root, verify_bundle=False)
        # Construct references for every split. Test targets are intentionally never requested.
        train = bundle.view("train")
        validation = bundle.view("validation")
        test_features = bundle._frame.iloc[list(bundle.splits.indices_for("test"))].loc[:, list(task.features)]
        for config in rf_configs(task_id):
            _ = make_rf_estimator(task_id, config, seed=42)
        result["tasks"][task_id] = {"rows": len(bundle._frame), "feature_order": list(task.features), "split_counts": {name: len(bundle.splits.indices_for(name)) for name in ("train", "validation", "test")}, "candidate_count": len(rf_configs(task_id)), "train_features": train.features.shape, "validation_features": validation.features.shape, "test_features": test_features.shape, "fit_called": False}
    result["candidate_counts"] = candidate_counts()
    return result


def train_validation(*args: Any, **kwargs: Any) -> None:
    raise TrainingDisabledError("Model training is prohibited during compatibility qualification")


def train_final(*args: Any, **kwargs: Any) -> None:
    raise TrainingDisabledError("Model training is prohibited during compatibility qualification")
