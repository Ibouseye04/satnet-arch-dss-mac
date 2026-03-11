from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple, Union

import joblib
import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


@dataclass
class RiskModelConfig:
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int | None = None


# ---------------------------------------------------------------------------
# Tier 1 Temporal Dataset Feature/Label Definitions (v1 schema)
# ---------------------------------------------------------------------------

# Design-time features for v1 temporal dataset (inputs only, no leakage)
TIER1_V1_FEATURE_COLUMNS: List[str] = [
    "num_planes",
    "sats_per_plane",
    "total_satellites",
    "inclination_deg",
    "altitude_km",
    "node_failure_prob",
    "edge_failure_prob",
    "duration_minutes",
    "step_seconds",
]

# Pure design features (no failure params) for design-time risk prediction
TIER1_V1_DESIGN_FEATURE_COLUMNS: List[str] = [
    "num_planes",
    "sats_per_plane",
    "inclination_deg",
    "altitude_km",
]

# Label column for v1 temporal dataset
TIER1_V1_LABEL_COLUMN = "partition_any"

# Alternative label: partition_fraction > threshold
TIER1_V1_LABEL_COLUMN_ALT = "partition_fraction"

# ---------------------------------------------------------------------------
# Legacy Feature Definitions (for backward compatibility)
# ---------------------------------------------------------------------------

# Features we’ll use from failure_dataset.csv
FEATURE_COLUMNS: List[str] = [
    "num_nodes_0",
    "num_edges_0",
    "avg_degree_0",
    "num_satellites",
    "num_ground_stations",
    "node_failure_prob",
    "edge_failure_prob",
    "failed_nodes",
    "failed_edges",
    "largest_component_ratio",
]

LABEL_COLUMN = "partitioned"

DESIGN_FEATURE_COLUMNS: List[str] = [
    "num_nodes_0",
    "num_edges_0",
    "avg_degree_0",
    "num_satellites",
    "num_ground_stations",
    "isl_degree",
    "node_failure_prob",
    "edge_failure_prob",
]

TIER1_FEATURE_COLUMNS: List[str] = [
    "num_planes",
    "sats_per_plane",
    "inclination_deg",
]


def load_design_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    X = df[DESIGN_FEATURE_COLUMNS].copy()
    y = df[LABEL_COLUMN].astype(int)
    return X, y


def train_design_risk_model(
    csv_path: Path,
    cfg: RiskModelConfig | None = None,
) -> Tuple[RandomForestClassifier, dict]:
    if cfg is None:
        cfg = RiskModelConfig()

    X, y = load_design_dataset(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics: dict = {}
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    except ValueError as e:
        logger.warning("ROC AUC computation failed: %s", e)
        metrics["roc_auc"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, output_dict=True
    )

    metrics["feature_importances"] = dict(
        zip(DESIGN_FEATURE_COLUMNS, clf.feature_importances_.tolist())
    )

    return clf, metrics


def load_tier1_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    X = df[TIER1_FEATURE_COLUMNS].copy()
    # Derive partitioned label: 1 if components_after_failure > 1, else 0
    y = (df["components_after_failure"] > 1).astype(int)
    return X, y


def train_tier1_risk_model(
    csv_path: Path,
    cfg: RiskModelConfig | None = None,
) -> Tuple[RandomForestClassifier, dict]:
    if cfg is None:
        cfg = RiskModelConfig()

    X, y = load_tier1_dataset(csv_path)

    # Check if we have both classes
    unique_classes = y.unique()
    if len(unique_classes) < 2:
        raise ValueError(
            f"Dataset has only one class (all labels = {unique_classes[0]}). "
            "Need both partitioned (1) and non-partitioned (0) samples to train. "
            "Generate more diverse failure scenarios."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    # Handle case where only one class is present in predictions
    if proba.shape[1] == 2:
        y_proba = proba[:, 1]
    else:
        y_proba = proba[:, 0] if clf.classes_[0] == 1 else 1 - proba[:, 0]

    metrics: dict = {}
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    except ValueError as e:
        logger.warning("ROC AUC computation failed: %s", e)
        metrics["roc_auc"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, output_dict=True
    )

    metrics["feature_importances"] = dict(
        zip(TIER1_FEATURE_COLUMNS, clf.feature_importances_.tolist())
    )

    return clf, metrics


def load_failure_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS].copy()
    y = df[LABEL_COLUMN].astype(int)
    return X, y


def train_risk_model(
    csv_path: Path,
    cfg: RiskModelConfig | None = None,
) -> Tuple[RandomForestClassifier, dict]:
    if cfg is None:
        cfg = RiskModelConfig()

    X, y = load_failure_dataset(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics: dict = {}
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    except ValueError as e:
        logger.warning("ROC AUC computation failed: %s", e)
        metrics["roc_auc"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, output_dict=True
    )

    # feature importances
    metrics["feature_importances"] = dict(
        zip(FEATURE_COLUMNS, clf.feature_importances_.tolist())
    )

    return clf, metrics


def save_model(model: RandomForestClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> RandomForestClassifier:
    return joblib.load(path)


def predict_partition_probabilities(
    model: RandomForestClassifier,
    X: pd.DataFrame,
) -> List[float]:
    proba = model.predict_proba(X)[:, 1]
    return proba.tolist()


# ---------------------------------------------------------------------------
# Tier 1 v1 Temporal Dataset Training Functions
# ---------------------------------------------------------------------------


def load_tier1_v1_dataset(
    csv_path: Path,
    label_column: str = TIER1_V1_LABEL_COLUMN,
    partition_threshold: float = 0.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load Tier 1 v1 temporal dataset for training.

    Args:
        csv_path: Path to the runs CSV file.
        label_column: Column to use as label. Default is 'partition_any'.
                      Can also use 'partition_fraction' with threshold.
        partition_threshold: If using partition_fraction, threshold for
                             binary classification (fraction > threshold → 1).

    Returns:
        Tuple of (X, y) where X is feature DataFrame and y is label Series.
    """
    df = pd.read_csv(csv_path)

    # Select features that exist in the dataset
    available_features = [c for c in TIER1_V1_FEATURE_COLUMNS if c in df.columns]
    X = df[available_features].copy()

    # Handle label
    if label_column == TIER1_V1_LABEL_COLUMN:
        y = df[label_column].astype(int)
    elif label_column == TIER1_V1_LABEL_COLUMN_ALT:
        # Convert fraction to binary using threshold
        y = (df[label_column] > partition_threshold).astype(int)
    else:
        y = df[label_column].astype(int)

    return X, y


def train_tier1_v1_risk_model(
    csv_path: Path,
    cfg: RiskModelConfig | None = None,
    label_column: str = TIER1_V1_LABEL_COLUMN,
) -> Tuple[RandomForestClassifier, dict]:
    """Train a risk model on Tier 1 v1 temporal dataset.

    Uses design-time features only (no post-failure leakage).
    Labels are temporal aggregate metrics (partition_any or partition_fraction).

    Args:
        csv_path: Path to the runs CSV file.
        cfg: Model configuration.
        label_column: Column to use as label.

    Returns:
        Tuple of (trained model, metrics dict).
    """
    if cfg is None:
        cfg = RiskModelConfig()

    X, y = load_tier1_v1_dataset(csv_path, label_column=label_column)

    # Check if we have both classes
    unique_classes = y.unique()
    if len(unique_classes) < 2:
        raise ValueError(
            f"Dataset has only one class (all labels = {unique_classes[0]}). "
            "Need both partitioned (1) and non-partitioned (0) samples to train. "
            "Try increasing failure probabilities or using smaller constellations."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    if proba.shape[1] == 2:
        y_proba = proba[:, 1]
    else:
        y_proba = proba[:, 0] if clf.classes_[0] == 1 else 1 - proba[:, 0]

    metrics: dict = {}
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    except ValueError as e:
        logger.warning("ROC AUC computation failed: %s", e)
        metrics["roc_auc"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, output_dict=True
    )

    # Feature importances
    feature_names = list(X.columns)
    metrics["feature_importances"] = dict(
        zip(feature_names, clf.feature_importances_.tolist())
    )

    metrics["label_column"] = label_column
    metrics["num_samples"] = len(y)
    metrics["positive_rate"] = float(y.mean())

    return clf, metrics


def train_tier1_v1_design_model(
    csv_path: Path,
    cfg: RiskModelConfig | None = None,
) -> Tuple[RandomForestClassifier, dict]:
    """Train a design-time risk model on Tier 1 v1 temporal dataset.

    Uses only pure design features (num_planes, sats_per_plane, inclination_deg,
    altitude_km) to predict partition risk. No failure parameters are used.

    Args:
        csv_path: Path to the runs CSV file (e.g., tier1_design_runs.csv).
        cfg: Model configuration.

    Returns:
        Tuple of (trained model, metrics dict).
    """
    if cfg is None:
        cfg = RiskModelConfig()

    df = pd.read_csv(csv_path)

    # Use only design features (no failure params)
    X = df[TIER1_V1_DESIGN_FEATURE_COLUMNS].copy()
    y = df[TIER1_V1_LABEL_COLUMN].astype(int)

    # Check if we have both classes
    unique_classes = y.unique()
    if len(unique_classes) < 2:
        raise ValueError(
            f"Dataset has only one class (all labels = {unique_classes[0]}). "
            "Need both partitioned (1) and non-partitioned (0) samples to train. "
            "Try using smaller constellations or longer durations."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    if proba.shape[1] == 2:
        y_proba = proba[:, 1]
    else:
        y_proba = proba[:, 0] if clf.classes_[0] == 1 else 1 - proba[:, 0]

    metrics: dict = {}
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    except ValueError as e:
        logger.warning("ROC AUC computation failed: %s", e)
        metrics["roc_auc"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, output_dict=True
    )

    # Feature importances
    metrics["feature_importances"] = dict(
        zip(TIER1_V1_DESIGN_FEATURE_COLUMNS, clf.feature_importances_.tolist())
    )

    metrics["label_column"] = TIER1_V1_LABEL_COLUMN
    metrics["num_samples"] = len(y)
    metrics["positive_rate"] = float(y.mean())

    return clf, metrics


# ---------------------------------------------------------------------------
# Unified RF Training with Target Selection (Phase 5/6)
# ---------------------------------------------------------------------------


def train_rf_model(
    csv_path: Path,
    target_name: str = "partition_any",
    feature_columns: List[str] | None = None,
    cfg: RiskModelConfig | None = None,
) -> Tuple[Union[RandomForestClassifier, RandomForestRegressor], dict, pd.DataFrame]:
    """Train a RandomForest model on any supported resilience target.

    Automatically selects classification vs regression based on *target_name*.
    Returns the model, metrics dict, and a predictions DataFrame.

    Args:
        csv_path: Path to the runs CSV file.
        target_name: Column name for the target variable.
        feature_columns: Feature columns to use. Defaults to TIER1_V1_FEATURE_COLUMNS.
        cfg: Model configuration.

    Returns:
        Tuple of (model, metrics, predictions_df).
        predictions_df has columns: sample_idx, true, predicted, split, seed.
    """
    from satnet.metrics.resilience_targets import infer_task_type

    if cfg is None:
        cfg = RiskModelConfig()

    task_type = infer_task_type(target_name)

    df = pd.read_csv(csv_path)
    if feature_columns is None:
        feature_columns = [c for c in TIER1_V1_FEATURE_COLUMNS if c in df.columns]

    X = df[feature_columns].copy()
    y = df[target_name].astype(float)

    stratify = y.round().astype(int) if task_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )

    if task_type == "classification":
        model: Any = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        model = RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics: dict = {
        "task_type": task_type,
        "target_name": target_name,
        "num_samples": len(y),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "seed": cfg.random_state,
    }

    if task_type == "classification":
        metrics.update(_classification_metrics(y_test, y_pred_test, model))
    else:
        metrics.update(_regression_metrics(y_test, y_pred_test, prefix="test"))
        metrics.update(_regression_metrics(y_train, y_pred_train, prefix="train"))

    metrics["feature_importances"] = dict(
        zip(feature_columns, model.feature_importances_.tolist())
    )

    # Build predictions DataFrame
    preds_rows: list[dict] = []
    for i, (idx, true, pred) in enumerate(zip(X_train.index, y_train, y_pred_train)):
        preds_rows.append({
            "sample_idx": int(idx), "true": float(true), "predicted": float(pred),
            "split": "train", "seed": cfg.random_state,
        })
    for i, (idx, true, pred) in enumerate(zip(X_test.index, y_test, y_pred_test)):
        preds_rows.append({
            "sample_idx": int(idx), "true": float(true), "predicted": float(pred),
            "split": "test", "seed": cfg.random_state,
        })
    predictions_df = pd.DataFrame(preds_rows)

    return model, metrics, predictions_df


def _classification_metrics(
    y_true: Any, y_pred: Any, model: RandomForestClassifier,
) -> dict:
    metrics: dict = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    try:
        proba = model.predict_proba(y_true.values.reshape(-1, 1) if hasattr(y_true, 'values') else y_true)
        # We can't call predict_proba on y_true; use the stored predictions
    except Exception:
        pass
    metrics["confusion_matrix"] = confusion_matrix(
        y_true.astype(int), np.round(y_pred).astype(int)
    ).tolist()
    metrics["classification_report"] = classification_report(
        y_true.astype(int), np.round(y_pred).astype(int), output_dict=True,
    )
    return metrics


def _regression_metrics(y_true: Any, y_pred: Any, prefix: str = "") -> dict:
    from scipy.stats import kendalltau, spearmanr

    pre = f"{prefix}_" if prefix else ""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    sp_corr, sp_p = spearmanr(y_true, y_pred)
    kt_corr, kt_p = kendalltau(y_true, y_pred)

    return {
        f"{pre}mae": mae,
        f"{pre}rmse": rmse,
        f"{pre}r2": r2,
        f"{pre}spearman_rho": float(sp_corr),
        f"{pre}spearman_p": float(sp_p),
        f"{pre}kendall_tau": float(kt_corr),
        f"{pre}kendall_p": float(kt_p),
    }