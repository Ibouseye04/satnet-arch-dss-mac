from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def impurity_feature_importance(fitted_estimator: Any, feature_names: Sequence[str]) -> dict[str, float]:
    """Return RF impurity importance only for an explicitly fitted selected model."""
    values = getattr(fitted_estimator, "feature_importances_", None)
    if values is None:
        raise ValueError("Estimator must be fitted before impurity importance is requested")
    if len(values) != len(feature_names):
        raise ValueError("Feature importance length does not match frozen feature order")
    return {str(name): float(value) for name, value in zip(feature_names, np.asarray(values))}


def permutation_feature_importance(fitted_estimator: Any, X: Any, y: Any, feature_names: Sequence[str], *, scoring: str) -> dict[str, Any]:
    """Compute future RF permutation importance; this function is never called by dry-run paths."""
    from sklearn.inspection import permutation_importance
    result = permutation_importance(fitted_estimator, X, y, scoring=scoring, random_state=42, n_jobs=-1)
    if result.importances_mean.shape[0] != len(feature_names):
        raise ValueError("Permutation importance length does not match frozen feature order")
    return {"feature_names": list(feature_names), "mean": result.importances_mean.tolist(), "std": result.importances_std.tolist()}
