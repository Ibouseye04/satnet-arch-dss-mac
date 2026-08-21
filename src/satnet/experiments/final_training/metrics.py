from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, average_precision_score, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, median_absolute_error, precision_score,
    r2_score, recall_score, roc_auc_score,
)


def classification_metrics(y_true: Any, y_pred: Any, y_score: Any | None = None) -> dict[str, Any]:
    true = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    if true.shape != pred.shape:
        raise ValueError("Classification targets and predictions must have equal shape")
    labels = [0, 1]
    matrix = confusion_matrix(true, pred, labels=labels)
    tn, fp, fn, tp = (int(x) for x in matrix.ravel())
    score = None if y_score is None else np.asarray(y_score, dtype=float)
    if score is not None and score.ndim == 2:
        if score.shape[1] != 2:
            raise ValueError("Classification scores must be one-dimensional or have two class columns")
        score = score[:, 1]
    unique_true = np.unique(true)
    roc_auc = None
    pr_auc = None
    if score is not None and len(unique_true) == 2:
        roc_auc = float(roc_auc_score(true, score))
        pr_auc = float(average_precision_score(true, score))
    majority = max(int(np.sum(true == 0)), int(np.sum(true == 1))) / len(true) if len(true) else float("nan")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)) if len(true) else float("nan"),
        "accuracy": float(accuracy_score(true, pred)) if len(true) else float("nan"),
        "precision_by_class": {str(label): float(x) for label, x in zip(labels, precision_score(true, pred, labels=labels, average=None, zero_division=0))},
        "recall_by_class": {str(label): float(x) for label, x in zip(labels, recall_score(true, pred, labels=labels, average=None, zero_division=0))},
        "f1_by_class": {str(label): float(x) for label, x in zip(labels, f1_score(true, pred, labels=labels, average=None, zero_division=0))},
        "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, labels=labels, average="weighted", zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "confusion_matrix": matrix.tolist(),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "positive_class_count": int(np.sum(true == 1)),
        "negative_class_count": int(np.sum(true == 0)),
        "majority_class_baseline_accuracy": float(majority),
    }


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("Regression targets and predictions must have equal shape")
    absolute = np.abs(true - pred)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "r2": float(r2_score(true, pred)) if len(true) > 1 else float("nan"),
        "median_absolute_error": float(median_absolute_error(true, pred)),
        "maximum_absolute_error": float(np.max(absolute)) if len(absolute) else float("nan"),
        "target_mean": float(np.mean(true)),
        "target_median": float(np.median(true)),
        "target_standard_deviation": float(np.std(true)),
        "prediction_mean": float(np.mean(pred)),
        "prediction_standard_deviation": float(np.std(pred)),
        "predictions_below_zero": int(np.sum(pred < 0)),
        "predictions_above_one": int(np.sum(pred > 1)),
    }
