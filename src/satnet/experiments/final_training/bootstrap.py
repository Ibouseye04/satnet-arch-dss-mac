from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error

from .contracts import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, REALIZATIONS_PER_DESIGN


@dataclass(frozen=True)
class BootstrapResult:
    difference: float
    confidence_interval: tuple[float, float]
    replicates: int
    seed: int
    metric: str


def paired_design_cluster_bootstrap(rows: Iterable[Mapping[str, Any]], *, task_type: str, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> BootstrapResult:
    records = list(rows)
    required = {"design_id", "realization_id", "y_true", "rf_prediction", "tgnn_prediction"}
    if not records or not required.issubset(records[0]):
        raise ValueError(f"Bootstrap rows must contain {sorted(required)}")
    by_design: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        by_design.setdefault(str(row["design_id"]), []).append(row)
    if any(len(group) != REALIZATIONS_PER_DESIGN for group in by_design.values()):
        raise ValueError("Each bootstrap design cluster must contain exactly five realizations")
    for group in by_design.values():
        if len({str(row["realization_id"]) for row in group}) != REALIZATIONS_PER_DESIGN:
            raise ValueError("Realization IDs must be unique within a design cluster")
    designs = sorted(by_design)
    def score(sample: list[str]) -> float:
        sampled = [row for design in sample for row in by_design[design]]
        true = np.asarray([row["y_true"] for row in sampled])
        rf = np.asarray([row["rf_prediction"] for row in sampled])
        tgnn = np.asarray([row["tgnn_prediction"] for row in sampled])
        if task_type == "classification":
            return float(balanced_accuracy_score(true, rf) - balanced_accuracy_score(true, tgnn))
        if task_type == "regression":
            return float(mean_absolute_error(true, rf) - mean_absolute_error(true, tgnn))
        raise ValueError("task_type must be classification or regression")
    observed = score(designs)
    rng = np.random.default_rng(seed)
    values = np.asarray([score(rng.choice(designs, size=len(designs), replace=True).tolist()) for _ in range(replicates)])
    low, high = np.quantile(values, [0.025, 0.975])
    return BootstrapResult(observed, (float(low), float(high)), replicates, seed, "balanced_accuracy" if task_type == "classification" else "mae")
