from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .contracts import get_task

RF_COMMON = {
    "n_estimators": (300, 600),
    "max_depth": (None, 10, 20),
    "min_samples_leaf": (1, 2, 5),
    "max_features": ("sqrt", 1.0),
    "bootstrap": (True,),
}

@dataclass(frozen=True)
class RFConfig:
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    max_features: str | float
    bootstrap: bool
    class_weight: str | None = None

    def as_dict(self, seed: int) -> dict[str, Any]:
        return {**self.__dict__, "random_state": seed, "n_jobs": -1}


def rf_configs(task_id: str) -> tuple[RFConfig, ...]:
    task = get_task(task_id)
    if task.family != "RF":
        raise ValueError("RF configuration requested for non-RF task")
    if task.task_type == "regression":
        weights: tuple[str | None, ...] = (None,)
    elif task_id == "rf_space_classification":
        weights = (None, "balanced")
    else:
        weights = ("balanced", "balanced_subsample")
    values = product(*(RF_COMMON[key] for key in ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "bootstrap")), weights)
    return tuple(RFConfig(*row) for row in values)


def make_rf_estimator(task_id: str, config: RFConfig, *, seed: int):
    params = config.as_dict(seed)
    params.pop("class_weight", None) if task_id not in {"rf_space_classification", "rf_integrated_classification"} else None
    task = get_task(task_id)
    if task.task_type == "classification":
        return RandomForestClassifier(**params)
    return RandomForestRegressor(**params)


def validate_rf_configs_without_fit(task_id: str, *, seed: int = 42) -> int:
    configs = rf_configs(task_id)
    for config in configs:
        estimator = make_rf_estimator(task_id, config, seed=seed)
        if estimator.get_params(deep=False)["n_estimators"] != config.n_estimators:
            raise ValueError("Installed scikit-learn rejected a frozen RF parameter")
    return len(configs)


@dataclass(frozen=True)
class TGNNConfig:
    hidden_dim: int
    learning_rate: float
    cheb_k: int
    max_epochs: int
    num_layers: int = 1
    weight_decay: float = 0.0
    batch_size: int = 1
    dropout: None = None


def tgnn_configs() -> tuple[TGNNConfig, ...]:
    return tuple(TGNNConfig(hidden, lr, k, epochs) for hidden, lr, k, epochs in product((32, 64), (0.001, 0.01), (2, 3), (50, 100)))


def candidate_counts() -> dict[str, int]:
    return {task: len(rf_configs(task)) for task in ("rf_space_classification", "rf_space_regression", "rf_integrated_regression_mean", "rf_integrated_regression_min", "rf_integrated_classification")}
