from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClassificationBaselines:
    majority_class: int
    positive_prevalence: float
    seed: int

    def majority_predict(self, size: int) -> np.ndarray:
        return np.full(size, self.majority_class, dtype=np.int64)

    def stratified_random_predict(self, size: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return (rng.random(size) < self.positive_prevalence).astype(np.int64)


@dataclass(frozen=True)
class RegressionBaselines:
    mean: float
    median: float

    def mean_predict(self, size: int) -> np.ndarray:
        return np.full(size, self.mean, dtype=float)

    def median_predict(self, size: int) -> np.ndarray:
        return np.full(size, self.median, dtype=float)


def classification_baselines_from_train(y_train: Any, *, seed: int) -> ClassificationBaselines:
    values = np.asarray(y_train).astype(int)
    if values.size == 0 or not set(values).issubset({0, 1}):
        raise ValueError("TRAIN classification targets must be non-empty binary values")
    counts = np.bincount(values, minlength=2)
    return ClassificationBaselines(int(np.argmax(counts)), float(np.mean(values)), seed)


def regression_baselines_from_train(y_train: Any) -> RegressionBaselines:
    values = np.asarray(y_train, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("TRAIN regression targets must be non-empty and finite")
    return RegressionBaselines(float(np.mean(values)), float(np.median(values)))
