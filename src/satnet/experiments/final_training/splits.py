from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from .contracts import REALIZATIONS_PER_DESIGN, SPLIT_DESIGN_COUNTS, SPLIT_RUN_COUNTS, SPLITS


@dataclass(frozen=True)
class FrozenSplits:
    indices: Mapping[str, tuple[int, ...]]
    run_ids: Mapping[str, tuple[int, ...]]
    design_ids: Mapping[str, tuple[str, ...]]

    def indices_for(self, split: str) -> tuple[int, ...]:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        return self.indices[split]

    def metadata_for(self, split: str) -> pd.DataFrame:
        raise RuntimeError("FrozenSplits contains identities only; use the loader's metadata view")


def _as_int(value: object, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer: {value!r}") from exc
    if str(value).strip() not in {str(parsed), f"{parsed}.0"} and not isinstance(value, int):
        raise ValueError(f"{name} is not an exact integer: {value!r}")
    return parsed


def validate_frozen_split_frame(frame: pd.DataFrame) -> FrozenSplits:
    """Validate the already-materialized authoritative split column.

    This function intentionally has no randomization, ordering inference, or fallback.
    """
    required = {"run_id", "design_id", "realization_id", "split"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Frozen split metadata missing required columns: {missing}")
    if len(frame) != 10000:
        raise ValueError(f"Frozen split requires 10000 rows, found {len(frame)}")
    normalized = frame["split"].astype("string").str.lower()
    if normalized.isna().any() or set(normalized.unique()) != set(SPLITS):
        raise ValueError("Split column must contain exactly train, validation, and test")
    run_ids = frame["run_id"].map(lambda v: _as_int(v, "run_id"))
    if run_ids.duplicated().any() or set(run_ids) != set(range(10000)):
        raise ValueError("run IDs must uniquely cover exactly 0..9999")
    if frame["design_id"].isna().any() or frame["realization_id"].isna().any():
        raise ValueError("Design and realization identities must be non-null")
    work = frame.copy()
    work["_split"] = normalized
    work["_run_id"] = run_ids
    work["_design_id"] = work["design_id"].astype(str)
    work["_realization_id"] = work["realization_id"].astype(str)
    split_indices: dict[str, tuple[int, ...]] = {}
    split_runs: dict[str, tuple[int, ...]] = {}
    split_designs: dict[str, tuple[str, ...]] = {}
    for split in SPLITS:
        subset = work[work["_split"] == split]
        if len(subset) != SPLIT_RUN_COUNTS[split]:
            raise ValueError(f"{split} must contain {SPLIT_RUN_COUNTS[split]} runs, found {len(subset)}")
        designs = sorted(subset["_design_id"].unique())
        if len(designs) != SPLIT_DESIGN_COUNTS[split]:
            raise ValueError(f"{split} must contain {SPLIT_DESIGN_COUNTS[split]} designs, found {len(designs)}")
        counts = subset.groupby("_design_id")["_realization_id"].nunique()
        if set(counts) != {REALIZATIONS_PER_DESIGN}:
            raise ValueError(f"Every {split} design must have exactly five realizations")
        for design, group in subset.groupby("_design_id"):
            realizations = set(group["_realization_id"])
            if realizations != {f"R{i:02d}" for i in range(REALIZATIONS_PER_DESIGN)}:
                raise ValueError(f"Design {design} does not contain R00..R04")
        split_indices[split] = tuple(int(i) for i in subset.index.tolist())
        split_runs[split] = tuple(sorted(int(i) for i in subset["_run_id"]))
        split_designs[split] = tuple(designs)
    design_split = work.groupby("_design_id")["_split"].nunique()
    if (design_split > 1).any():
        raise ValueError("A design appears in more than one frozen split")
    if set().union(*(set(v) for v in split_runs.values())) != set(range(10000)):
        raise ValueError("Frozen split run union is incomplete")
    return FrozenSplits(split_indices, split_runs, split_designs)


def validate_manifest_identities(records: Iterable[Mapping[str, object]], splits: FrozenSplits) -> None:
    """Validate TGNN manifest identities against the authoritative split mapping."""
    seen: set[int] = set()
    expected_by_run = {run: split for split, runs in splits.run_ids.items() for run in runs}
    for record in records:
        run_id = _as_int(record.get("run_id"), "manifest run_id")
        if run_id in seen:
            raise ValueError(f"Duplicate manifest run_id {run_id}")
        seen.add(run_id)
        split = str(record.get("split", "")).lower()
        if expected_by_run.get(run_id) != split:
            raise ValueError(f"Manifest split mismatch for run {run_id}")
    if seen != set(expected_by_run):
        raise ValueError("Manifest run IDs do not cover the frozen 0..9999 run set")
