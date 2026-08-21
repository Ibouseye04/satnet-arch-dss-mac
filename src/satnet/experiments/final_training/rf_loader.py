from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import DATASET_ROOT, METADATA_COLUMNS, TaskContract, get_task, load_json, verify_dataset_bundle
from .splits import FrozenSplits, validate_frozen_split_frame
from .test_access import MetadataView, TargetGate


@dataclass(frozen=True)
class RFView:
    split: str
    indices: tuple[int, ...]
    features: pd.DataFrame
    gate: TargetGate

    @property
    def targets(self) -> tuple[Any, ...]:
        return self.gate.targets

    def authorized_targets(self, authorization):
        return self.gate.authorized_targets(authorization)


@dataclass
class RFDataBundle:
    task: TaskContract
    source_csv: Path
    metadata: pd.DataFrame
    splits: FrozenSplits
    _frame: pd.DataFrame

    def view(self, split: str) -> RFView:
        indices = self.splits.indices_for(split)
        rows = self._frame.iloc[list(indices)]
        target_values = tuple(rows[self.task.target].tolist())
        metadata = MetadataView(tuple(rows[list(METADATA_COLUMNS)].to_dict(orient="records")))
        gate = TargetGate(split=split, metadata=metadata, target_values=target_values)
        return RFView(split, indices, rows.loc[:, list(self.task.features)].copy(), gate)

    def model_matrix(self, split: str) -> tuple[pd.DataFrame, tuple[Any, ...]]:
        view = self.view(split)
        return view.features, view.targets

    @property
    def feature_order(self) -> tuple[str, ...]:
        return self.task.features


def _normalize_target(series: pd.Series, task_type: str) -> pd.Series:
    if task_type == "classification":
        if series.dtype == bool:
            return series.astype(np.int64)
        values = series.astype("string").str.strip().str.lower()
        mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
        unknown = sorted(set(values.dropna()) - set(mapping))
        if unknown:
            raise ValueError(f"Unexpected classification target values: {unknown}")
        return values.map(mapping).astype("int64")
    result = pd.to_numeric(series, errors="raise")
    if result.isna().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Regression target contains missing or non-finite values")
    return result.astype(float)


def _validate_schema_and_manifest(task: TaskContract, csv_path: Path) -> tuple[str, ...]:
    directory = csv_path.parent
    schema = load_json(directory / f"{directory.name}_schema.json")
    manifest = load_json(directory / f"{directory.name}_manifest.json")
    schema_features = tuple(item["field"] for item in schema.get("predictors", []))
    manifest_features = tuple(manifest.get("predictors", []))
    if schema_features != task.features or manifest_features != task.features:
        raise ValueError(f"Frozen feature order mismatch for {task.task_id}")
    schema_target = schema.get("target", {}).get("field")
    manifest_targets = tuple(manifest.get("targets", ()))
    if task.target not in manifest_targets or (len(manifest_targets) == 1 and schema_target != task.target):
        raise ValueError(f"Frozen target mismatch for {task.task_id}")
    if manifest.get("row_count") != 10000 or schema.get("row_count_before_training_preprocessing") != 10000:
        raise ValueError("Frozen RF export must contain 10000 rows")
    if tuple(manifest.get("metadata_columns_not_model_features", [])) != METADATA_COLUMNS:
        raise ValueError("RF manifest metadata exclusion contract mismatch")
    return tuple(manifest.get("targets", (task.target,)))


def load_rf_task(task_id: str, *, dataset_root: Path = DATASET_ROOT, verify_bundle: bool = True) -> RFDataBundle:
    task = get_task(task_id)
    if task.family != "RF":
        raise ValueError(f"{task_id} is not an RF task")
    if verify_bundle:
        verify_dataset_bundle(dataset_root, full=True)
    csv_path = dataset_root / task.dataset_relative_path
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    target_columns = _validate_schema_and_manifest(task, csv_path)
    frame = pd.read_csv(csv_path)
    expected = list(METADATA_COLUMNS) + list(task.features) + list(target_columns)
    if list(frame.columns) != expected or task.target not in frame.columns:
        raise ValueError(f"Unexpected RF schema. Expected {expected}, got {frame.columns.tolist()}")
    if len(frame) != 10000:
        raise ValueError("RF dataset must contain exactly 10000 rows")
    frame = frame.copy()
    frame[task.target] = _normalize_target(frame[task.target], task.task_type)
    if frame.loc[:, list(task.features)].isna().any().any():
        raise ValueError("RF predictors contain missing values")
    splits = validate_frozen_split_frame(frame)
    metadata = frame.loc[:, list(METADATA_COLUMNS)].copy()
    return RFDataBundle(task, csv_path, metadata, splits, frame)
