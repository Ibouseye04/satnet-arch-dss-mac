from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
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
    except ValueError:
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
    except ValueError:
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
    except ValueError:
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
    except ValueError:
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
    except ValueError:
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