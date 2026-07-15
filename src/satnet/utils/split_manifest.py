from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SPLIT_MANIFEST_SCHEMA_VERSION = 1
SPLIT_STRATEGY_RUN_LEVEL_SEEDED = "shared_run_level_seeded"


@dataclass(frozen=True)
class DatasetIdentity:
    content_hash: str
    row_count: int
    columns: list[str]
    source_path: str


def compute_dataset_identity(csv_path: Path) -> DatasetIdentity:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")
    content = path.read_bytes()
    df = pd.read_csv(path)
    return DatasetIdentity(
        content_hash=hashlib.sha256(content).hexdigest(),
        row_count=int(len(df)),
        columns=[str(c) for c in df.columns.tolist()],
        source_path=str(path),
    )


def _safe_classification_stratify(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts(dropna=False)
    if len(counts) < 2 or int(counts.min()) < 2:
        return None
    return labels


def make_shared_run_level_splits(
    df: pd.DataFrame,
    *,
    target_name: str,
    task_type: str,
    test_size: float,
    val_size: float,
    seed: int,
    subset: int | None = None,
) -> dict[str, list[int]]:
    if target_name not in df.columns:
        raise ValueError(f"Missing target column '{target_name}'")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in the open interval (0, 1)")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("val_size must be in the interval [0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1")

    active_indices = df.index.tolist()
    if subset is not None:
        if subset < 3:
            raise ValueError("subset must be at least 3 when provided")
        active_indices = _select_subset_indices(
            df,
            target_name=target_name,
            task_type=task_type,
            subset=subset,
            seed=seed,
        )

    active_df = df.loc[active_indices].copy()
    if len(active_df) < 3:
        raise ValueError("Need at least 3 active samples for train/validation/test splits")

    group_name = "run_id" if "run_id" in active_df.columns else "_row_index"
    group_series = active_df["run_id"] if group_name == "run_id" else pd.Series(active_df.index, index=active_df.index)
    split_frame = pd.DataFrame(
        {
            "_row_index": active_df.index,
            "_group": group_series.values,
            "_target": active_df[target_name].values,
        },
        index=active_df.index,
    )
    inconsistent_groups = split_frame.groupby("_group")["_target"].nunique(dropna=False)
    inconsistent_groups = inconsistent_groups[inconsistent_groups > 1]
    if len(inconsistent_groups) > 0:
        examples = list(inconsistent_groups.index[:5])
        raise ValueError(
            f"Cannot split by {group_name}: target varies within group(s) {examples}."
        )

    group_frame = (
        split_frame.sort_index()
        .drop_duplicates("_group")
        .loc[:, ["_group", "_target"]]
        .reset_index(drop=True)
    )
    if len(group_frame) < 3:
        raise ValueError("Need at least 3 unique runs/groups to create splits")

    group_ids = group_frame["_group"]
    if task_type == "classification":
        group_labels = group_frame["_target"].round().astype(int)
        if group_labels.nunique() < 2:
            raise ValueError("Classification target has only one class")
        stratify = _safe_classification_stratify(group_labels)
    elif task_type == "regression":
        stratify = None
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")

    train_val_groups, test_groups = train_test_split(
        group_ids,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

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
            random_state=seed,
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
    splits = {
        split: split_frame.index[split_frame["_group"].isin(groups)].astype(int).tolist()
        for split, groups in group_sets.items()
    }
    validate_split_indices(splits, active_indices=active_indices)
    return splits


def _select_subset_indices(
    df: pd.DataFrame,
    *,
    target_name: str,
    task_type: str,
    subset: int,
    seed: int,
) -> list[int]:
    subset_count = min(subset, len(df))
    if task_type != "classification":
        rng = np.random.default_rng(seed)
        return [int(idx) for idx in rng.permutation(df.index.tolist()).tolist()[:subset_count]]

    labels = df[target_name].round().astype(int)
    classes = sorted(labels.dropna().unique().tolist())
    if len(classes) < 2:
        rng = np.random.default_rng(seed)
        return [int(idx) for idx in rng.permutation(df.index.tolist()).tolist()[:subset_count]]

    per_class_minimum = min(2, max(1, subset_count // len(classes)))
    selected: list[int] = []
    rng = np.random.default_rng(seed)
    for class_value in classes:
        class_indices = df.index[labels == class_value].tolist()
        if len(class_indices) < per_class_minimum:
            rng = np.random.default_rng(seed)
            return [int(idx) for idx in rng.permutation(df.index.tolist()).tolist()[:subset_count]]
        selected.extend(int(idx) for idx in rng.permutation(class_indices).tolist()[:per_class_minimum])

    remaining = [int(idx) for idx in df.index.tolist() if idx not in set(selected)]
    selected.extend(int(idx) for idx in rng.permutation(remaining).tolist()[: max(0, subset_count - len(selected))])
    return [int(idx) for idx in selected[:subset_count]]


def validate_split_indices(
    splits: dict[str, list[int]],
    *,
    active_indices: list[int] | None = None,
) -> None:
    required = {"train", "val", "test"}
    if set(splits) != required:
        raise ValueError(f"Split manifest must contain exactly {sorted(required)}")
    all_indices = splits["train"] + splits["val"] + splits["test"]
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Split indices contain duplicates")
    for split in required:
        if any(int(i) < 0 for i in splits[split]):
            raise ValueError("Split indices must be non-negative")
    if active_indices is not None and set(all_indices) != set(active_indices):
        raise ValueError("Split union does not equal intended active sample set")
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("Train and validation splits overlap")
    if set(splits["train"]) & set(splits["test"]):
        raise ValueError("Train and test splits overlap")
    if set(splits["val"]) & set(splits["test"]):
        raise ValueError("Validation and test splits overlap")


def build_split_manifest(
    *,
    csv_path: Path,
    target_name: str,
    task_type: str,
    seed: int,
    test_size: float,
    val_size: float,
    subset: int | None = None,
) -> dict[str, Any]:
    identity = compute_dataset_identity(csv_path)
    df = pd.read_csv(csv_path)
    splits = make_shared_run_level_splits(
        df,
        target_name=target_name,
        task_type=task_type,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        subset=subset,
    )
    active_indices = sorted(splits["train"] + splits["val"] + splits["test"])
    return {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "dataset_identity": identity.__dict__,
        "target_name": target_name,
        "task_type": task_type,
        "seed": int(seed),
        "split_seed": int(seed),
        "test_size_fraction": float(test_size),
        "val_size_fraction": float(val_size),
        "subset": subset,
        "split_strategy": SPLIT_STRATEGY_RUN_LEVEL_SEEDED,
        "row_count": identity.row_count,
        "active_indices": active_indices,
        "train_indices": splits["train"],
        "val_indices": splits["val"],
        "test_indices": splits["test"],
    }


def write_split_manifest(manifest: dict[str, Any], path: Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
    return path


def read_split_manifest(path: Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def validate_manifest_for_dataset(
    manifest: dict[str, Any],
    *,
    csv_path: Path,
    target_name: str,
) -> dict[str, list[int]]:
    if int(manifest.get("schema_version", -1)) != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported split manifest schema_version")
    if manifest.get("target_name") != target_name:
        raise ValueError(
            f"Split manifest target '{manifest.get('target_name')}' does not match '{target_name}'"
        )
    identity = compute_dataset_identity(csv_path)
    manifest_identity = manifest.get("dataset_identity", {})
    if manifest_identity.get("content_hash") != identity.content_hash:
        raise ValueError("Split manifest dataset hash does not match current dataset")
    if int(manifest.get("row_count", -1)) != identity.row_count:
        raise ValueError("Split manifest row_count does not match current dataset")
    splits = {
        "train": [int(i) for i in manifest["train_indices"]],
        "val": [int(i) for i in manifest["val_indices"]],
        "test": [int(i) for i in manifest["test_indices"]],
    }
    validate_split_indices(splits, active_indices=[int(i) for i in manifest["active_indices"]])
    return splits
