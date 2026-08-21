from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import DATASET_BUNDLE_HASH, PRODUCTION_TOOLING_SHA, TRAINING_PLAN_BUNDLE_HASH


@dataclass
class EarlyStopping:
    task_type: str
    patience: int = 10
    min_delta: float = 0.0
    restore_best: bool = True
    best_metric: float | None = None
    best_epoch: int | None = None
    best_state: Any = None
    bad_epochs: int = 0

    @property
    def direction(self) -> str:
        if self.task_type == "classification": return "maximize"
        if self.task_type == "regression": return "minimize"
        raise ValueError("task_type must be classification or regression")

    def update(self, epoch: int, metric: float, state: Any = None) -> bool:
        improved = self.best_metric is None or (metric > self.best_metric + self.min_delta if self.direction == "maximize" else metric < self.best_metric - self.min_delta)
        if improved:
            self.best_metric = float(metric)
            self.best_epoch = int(epoch)
            self.best_state = copy.deepcopy(state)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def restore(self, model: Any) -> Any:
        if self.restore_best and self.best_state is not None:
            if hasattr(model, "load_state_dict"):
                model.load_state_dict(self.best_state)
            else:
                raise TypeError("Model cannot restore a checkpoint state")
        return model


def monitored_metric(task_type: str) -> tuple[str, str]:
    if task_type == "classification": return "balanced_accuracy", "maximize"
    if task_type == "regression": return "mae", "minimize"
    raise ValueError("Unsupported TGNN task type")


@dataclass(frozen=True)
class CheckpointManifest:
    task: str
    model_family: str
    configuration: dict[str, Any]
    seed: int
    dataset_bundle_hash: str
    training_plan_bundle_hash: str
    train_count: int
    validation_count: int
    selected_epoch: int | None
    validation_metrics: dict[str, Any]
    checkpoint_sha256: str
    environment_versions: dict[str, str]
    code_tooling_sha: str = PRODUCTION_TOOLING_SHA

    def validate(self) -> None:
        if self.dataset_bundle_hash != DATASET_BUNDLE_HASH or self.training_plan_bundle_hash != TRAINING_PLAN_BUNDLE_HASH:
            raise ValueError("Checkpoint manifest is not bound to frozen bundles")
        if self.train_count != 7000 or self.validation_count != 1500:
            raise ValueError("Checkpoint manifest split counts are not frozen")
        if not self.checkpoint_sha256 or len(self.checkpoint_sha256) != 64:
            raise ValueError("Checkpoint SHA-256 is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def sha256_checkpoint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path, manifest: CheckpointManifest) -> None:
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
