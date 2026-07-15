from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple, Union

from satnet.metrics.resilience_targets import ALL_TARGETS
from satnet.utils.split_manifest import validate_manifest_for_dataset

import joblib
import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


@dataclass
class RiskModelConfig:
    test_size: float = 0.2
    val_size: float = 0.1
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

RF_FEATURE_SET_FULL = "full"
RF_FEATURE_SET_ARCHITECTURE_ONLY = "architecture_only"
RF_FEATURE_SET_NO_GEOMETRY = "no_geometry"
RF_FEATURE_SET_REGISTRY: dict[str, list[str]] = {
    RF_FEATURE_SET_FULL: list(TIER1_V1_FEATURE_COLUMNS),
    RF_FEATURE_SET_ARCHITECTURE_ONLY: [
        column
        for column in TIER1_V1_FEATURE_COLUMNS
        if column not in {"node_failure_prob", "edge_failure_prob"}
    ],
    RF_FEATURE_SET_NO_GEOMETRY: [
        column
        for column in TIER1_V1_FEATURE_COLUMNS
        if column not in {"altitude_km", "inclination_deg"}
    ],
}
RF_OUTCOME_FIELD_PREFIXES = (
    "gcc_",
    "partition",
    "component",
)
RF_OUTCOME_FIELD_NAMES = frozenset(
    {
        "failed_nodes",
        "failed_edges",
        "failed_nodes_json",
        "failed_edges_json",
        "num_failed_nodes",
        "num_failed_edges",
        "largest_component_ratio",
        "num_components",
    }
) | frozenset(ALL_TARGETS)


def get_rf_feature_columns(feature_set: str = RF_FEATURE_SET_FULL) -> list[str]:
    if feature_set not in RF_FEATURE_SET_REGISTRY:
        raise ValueError(
            f"Unknown RF feature set '{feature_set}'. "
            f"Allowed values: {sorted(RF_FEATURE_SET_REGISTRY)}"
        )
    columns = list(RF_FEATURE_SET_REGISTRY[feature_set])
    validate_rf_feature_columns(columns)
    return columns


def validate_rf_feature_columns(columns: list[str]) -> None:
    if not columns:
        raise ValueError("RF feature set cannot be empty")
    leakage = [
        column
        for column in columns
        if column in RF_OUTCOME_FIELD_NAMES
        or any(column.startswith(prefix) for prefix in RF_OUTCOME_FIELD_PREFIXES)
    ]
    if leakage:
        raise ValueError(f"RF feature set contains outcome/leakage fields: {leakage}")
    unknown = [column for column in columns if column not in TIER1_V1_FEATURE_COLUMNS]
    if unknown:
        raise ValueError(
            f"RF feature set contains unknown/non-approved feature columns: {unknown}"
        )

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


def _validated_prediction_config_hashes(
    df: pd.DataFrame,
    *,
    source_name: Path,
) -> pd.Series:
    if "config_hash" not in df.columns:
        raise ValueError(
            "Prediction export requires a `config_hash` column for stable ranking "
            f"joins. Source dataset '{source_name}' is missing `config_hash`."
        )
    normalized = df["config_hash"].astype("string").str.strip()
    invalid_mask = df["config_hash"].isna() | normalized.fillna("").eq("")
    if invalid_mask.any():
        invalid_rows = [int(idx) for idx in df.index[invalid_mask][:5].tolist()]
        raise ValueError(
            "Prediction export requires non-null, non-empty `config_hash` values "
            "for stable ranking joins. "
            f"Found {int(invalid_mask.sum())} invalid row(s) in '{source_name}' "
            f"(example row indices: {invalid_rows})."
        )
    return normalized


def _safe_classification_stratify(labels: pd.Series) -> pd.Series | None:
    """Return labels for stratification only when the split can support it."""
    counts = labels.value_counts(dropna=False)
    if len(counts) < 2 or int(counts.min()) < 2:
        return None
    return labels


def _run_level_split_indices(
    df: pd.DataFrame,
    y: pd.Series,
    *,
    task_type: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> dict[str, list[int]]:
    """Create seeded train/validation/test splits without crossing run_ids.

    The Tier 1 runs table has one row per simulation run, but this helper is
    stricter than row splitting so future exports with repeated run rows cannot
    leak a run across splits.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in the open interval (0, 1)")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("val_size must be in the interval [0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1")

    if "run_id" in df.columns:
        group_series = df["run_id"]
        group_name = "run_id"
    else:
        group_series = pd.Series(df.index, index=df.index, name="_row_index")
        group_name = "_row_index"

    split_frame = pd.DataFrame(
        {
            "_row_index": df.index,
            "_group": group_series.values,
            "_target": y.values,
        },
        index=df.index,
    )

    target_counts_per_group = split_frame.groupby("_group")["_target"].nunique(dropna=False)
    inconsistent_groups = target_counts_per_group[target_counts_per_group > 1]
    if len(inconsistent_groups) > 0:
        examples = list(inconsistent_groups.index[:5])
        raise ValueError(
            f"Cannot split by {group_name}: target varies within group(s) {examples}. "
            "Use a run-level dataset with one target per run."
        )

    group_frame = (
        split_frame.sort_index()
        .drop_duplicates("_group")
        .loc[:, ["_group", "_target"]]
        .reset_index(drop=True)
    )

    if len(group_frame) < 3:
        raise ValueError(
            "Need at least 3 unique runs/groups to create train/validation/test splits."
        )

    group_ids = group_frame["_group"]
    if task_type == "classification":
        group_labels = group_frame["_target"].round().astype(int)
        if group_labels.nunique() < 2:
            raise ValueError(
                "Classification target has only one class. Generate a dataset with "
                "both positive and negative runs or use a regression target."
            )
        stratify = _safe_classification_stratify(group_labels)
    else:
        stratify = None

    train_val_groups, test_groups = train_test_split(
        group_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    val_groups: pd.Series
    if val_size > 0.0:
        relative_val_size = val_size / (1.0 - test_size)
        train_val_targets = group_frame.loc[group_frame["_group"].isin(train_val_groups)]
        if task_type == "classification":
            train_val_labels = train_val_targets["_target"].round().astype(int)
            train_val_stratify = _safe_classification_stratify(train_val_labels)
        else:
            train_val_stratify = None

        train_groups, val_groups = train_test_split(
            train_val_targets["_group"],
            test_size=relative_val_size,
            random_state=random_state,
            stratify=train_val_stratify,
        )
    else:
        train_groups = train_val_groups
        val_groups = pd.Series([], dtype=group_ids.dtype)

    group_sets = {
        "train": set(train_groups.tolist()),
        "val": set(val_groups.tolist()),
        "test": set(test_groups.tolist()),
    }

    return {
        split: split_frame.index[split_frame["_group"].isin(groups)].tolist()
        for split, groups in group_sets.items()
    }


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
    feature_set_name: str = RF_FEATURE_SET_FULL,
    split_manifest: dict[str, Any] | None = None,
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
        predictions_df has stable identifiers and prediction columns:
        config_hash, target_name, task_type, seed, split, sample_idx, y_true, y_pred.
    """
    from satnet.metrics.resilience_targets import infer_task_type

    if cfg is None:
        cfg = RiskModelConfig()

    task_type = infer_task_type(target_name)

    df = pd.read_csv(csv_path)
    config_hashes = _validated_prediction_config_hashes(df, source_name=csv_path)
    if target_name not in df.columns:
        raise ValueError(
            f"Missing target column '{target_name}' in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )
    if feature_columns is None:
        feature_columns = [c for c in get_rf_feature_columns(feature_set_name) if c in df.columns]
    validate_rf_feature_columns(list(feature_columns))
    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing requested RF feature columns: {missing_features}")

    X = df[feature_columns].copy()
    if task_type == "classification":
        y = df[target_name].astype(int)
    else:
        y = df[target_name].astype(float)

    if split_manifest is not None:
        split_indices = validate_manifest_for_dataset(
            split_manifest,
            csv_path=csv_path,
            target_name=target_name,
        )
        split_strategy = split_manifest["split_strategy"]
    else:
        split_indices = _run_level_split_indices(
            df,
            y,
            task_type=task_type,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state,
        )
        split_strategy = "run_id_grouped" if "run_id" in df.columns else "row_index"
    active_sample_count = len(split_indices["train"]) + len(split_indices["val"]) + len(split_indices["test"])
    X_train = X.loc[split_indices["train"]]
    y_train = y.loc[split_indices["train"]]
    X_val = X.loc[split_indices["val"]]
    y_val = y.loc[split_indices["val"]]
    X_test = X.loc[split_indices["test"]]
    y_test = y.loc[split_indices["test"]]

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
    model.satnet_metadata_ = {
        "model_type": "RandomForest",
        "target_name": target_name,
        "task_type": task_type,
        "feature_set": feature_set_name,
        "feature_columns": list(feature_columns),
        "feature_count": len(feature_columns),
        "seed": cfg.random_state,
        "n_estimators": cfg.n_estimators,
        "max_depth": cfg.max_depth,
    }

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val) if len(X_val) else np.asarray([])
    y_pred_test = model.predict(X_test)

    metrics: dict = {
        "task_type": task_type,
        "target_name": target_name,
        "num_samples": active_sample_count,
        "dataset_num_samples": len(y),
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "seed": cfg.random_state,
        "split_strategy": split_strategy,
        "feature_set": feature_set_name,
        "feature_columns": list(feature_columns),
        "feature_count": len(feature_columns),
        "rf_n_estimators": cfg.n_estimators,
        "rf_max_depth": cfg.max_depth,
        "split_indices": {
            "train": [int(i) for i in split_indices["train"]],
            "val": [int(i) for i in split_indices["val"]],
            "test": [int(i) for i in split_indices["test"]],
        },
    }

    if task_type == "classification":
        y_score_train = _positive_class_scores(model, X_train)
        y_score_val = _positive_class_scores(model, X_val) if len(X_val) else None
        y_score_test = _positive_class_scores(model, X_test)

        train_metrics = _classification_metrics(
            y_train, y_pred_train, y_score_train, prefix="train",
        )
        test_metrics = _classification_metrics(
            y_test, y_pred_test, y_score_test, prefix="test",
        )
        metrics.update(train_metrics)
        if len(X_val):
            metrics.update(
                _classification_metrics(y_val, y_pred_val, y_score_val, prefix="val")
            )
        metrics.update(test_metrics)

        # Preserve the original top-level keys as aliases for test metrics.
        metrics["accuracy"] = test_metrics["test_accuracy"]
        metrics["precision"] = test_metrics["test_precision"]
        metrics["recall"] = test_metrics["test_recall"]
        metrics["f1"] = test_metrics["test_f1"]
        if "test_roc_auc" in test_metrics:
            metrics["roc_auc"] = test_metrics["test_roc_auc"]
        metrics["confusion_matrix"] = test_metrics["test_confusion_matrix"]
        metrics["classification_report"] = test_metrics["test_classification_report"]
    else:
        metrics.update(_regression_metrics(y_test, y_pred_test, prefix="test"))
        if len(X_val):
            metrics.update(_regression_metrics(y_val, y_pred_val, prefix="val"))
        metrics.update(_regression_metrics(y_train, y_pred_train, prefix="train"))

    metrics["feature_importances"] = dict(
        zip(feature_columns, model.feature_importances_.tolist())
    )

    # Build predictions DataFrame
    preds_rows: list[dict] = []

    def _build_row(split: str, idx: int, true: float, pred: float) -> dict:
        row: dict[str, Any] = {
            "config_hash": str(config_hashes.iloc[int(idx)]),
            "target_name": target_name,
            "task_type": task_type,
            "seed": cfg.random_state,
            "split": split,
            "sample_idx": int(idx),
            "y_true": float(true),
            "y_pred": float(pred),
            "model_type": "RandomForest",
            "data_path": str(csv_path),
            "feature_set": feature_set_name,
        }
        source_row = df.iloc[int(idx)]
        if "run_id" in df.columns and not pd.isna(source_row.get("run_id")):
            row["run_id"] = int(source_row["run_id"])
        return row

    for idx, true, pred in zip(X_train.index, y_train, y_pred_train):
        preds_rows.append(_build_row("train", int(idx), float(true), float(pred)))
    for idx, true, pred in zip(X_val.index, y_val, y_pred_val):
        preds_rows.append(_build_row("val", int(idx), float(true), float(pred)))
    for idx, true, pred in zip(X_test.index, y_test, y_pred_test):
        preds_rows.append(_build_row("test", int(idx), float(true), float(pred)))

    predictions_df = pd.DataFrame(preds_rows)

    return model, metrics, predictions_df


def _positive_class_scores(
    model: RandomForestClassifier,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return P(class=1) when available for binary classification metrics."""
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        only_class = int(model.classes_[0])
        return np.ones(len(X)) if only_class == 1 else np.zeros(len(X))
    class_to_col = {int(cls): i for i, cls in enumerate(model.classes_)}
    return proba[:, class_to_col.get(1, proba.shape[1] - 1)]


def _classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any | None = None,
    prefix: str = "",
) -> dict:
    pre = f"{prefix}_" if prefix else ""
    y_true_int = pd.Series(y_true).astype(int)
    y_pred_int = pd.Series(y_pred).round().astype(int)
    labels = [0, 1] if set(y_true_int.unique()) | set(y_pred_int.unique()) <= {0, 1} else None

    metrics: dict = {
        f"{pre}accuracy": accuracy_score(y_true_int, y_pred_int),
        f"{pre}precision": precision_score(
            y_true_int, y_pred_int, zero_division=0,
        ),
        f"{pre}recall": recall_score(y_true_int, y_pred_int, zero_division=0),
        f"{pre}f1": f1_score(y_true_int, y_pred_int, zero_division=0),
        f"{pre}confusion_matrix": confusion_matrix(
            y_true_int, y_pred_int, labels=labels,
        ).tolist(),
        f"{pre}classification_report": classification_report(
            y_true_int,
            y_pred_int,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    if y_score is not None and y_true_int.nunique() > 1:
        try:
            metrics[f"{pre}roc_auc"] = roc_auc_score(y_true_int, y_score)
        except ValueError as e:
            logger.warning("ROC AUC computation failed: %s", e)
            metrics[f"{pre}roc_auc"] = float("nan")

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
