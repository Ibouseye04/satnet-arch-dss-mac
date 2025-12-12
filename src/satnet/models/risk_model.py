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