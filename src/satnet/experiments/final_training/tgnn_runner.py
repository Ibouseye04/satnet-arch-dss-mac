from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import AUTHORIZED_TASKS, DATASET_ROOT, verify_dataset_bundle
from .search import TGNNConfig, tgnn_configs
from .tgnn_loader import load_tgnn_task


class TrainingDisabledError(RuntimeError):
    """Raised instead of entering an epoch, backward pass, or optimizer step."""


def instantiate_untrained_tgnn(task_id: str, config: TGNNConfig):
    from satnet.models.gnn_model import SatelliteGNN
    task = AUTHORIZED_TASKS[task_id]
    return SatelliteGNN(node_features=3, hidden_channels=config.hidden_dim, out_channels=2 if task.task_type == "classification" else 1, task_type=task.task_type, cheb_k=config.cheb_k)


def dry_run_tgnn_tasks(*, dataset_root: Path = DATASET_ROOT) -> dict[str, Any]:
    verify_dataset_bundle(dataset_root, full=True)
    result: dict[str, Any] = {"fit_called": False, "backward_called": False, "optimizer_step_called": False, "tasks": {}}
    for task_id, task in AUTHORIZED_TASKS.items():
        if task.family != "TGNN":
            continue
        bundle = load_tgnn_task(task_id, dataset_root=dataset_root, verify_bundle=False)
        samples = {}
        for split in ("train", "validation", "test"):
            artifact = bundle.structural_sample(split)
            samples[split] = {"sequence_length": artifact.sequence_length, "node_feature_dimension": artifact.node_features.shape[1], "edge_feature_dimension": artifact.edge_attr.shape[1], "run_id": artifact.header["sequence"]["run_id"], "split": artifact.header["sequence"]["split"]}
        for config in tgnn_configs():
            _ = instantiate_untrained_tgnn(task_id, config)
        result["tasks"][task_id] = {"candidate_count": len(tgnn_configs()), "split_counts": {name: len(bundle.splits.indices_for(name)) for name in ("train", "validation", "test")}, "representative_samples": samples, "fit_called": False}
    return result


def train_validation(*args: Any, **kwargs: Any) -> None:
    raise TrainingDisabledError("TGNN training is prohibited during compatibility qualification")


def train_final(*args: Any, **kwargs: Any) -> None:
    raise TrainingDisabledError("TGNN training is prohibited during compatibility qualification")
