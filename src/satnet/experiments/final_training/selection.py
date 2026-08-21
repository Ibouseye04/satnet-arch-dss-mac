from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import inf
from typing import Any, Iterable

from .contracts import VALIDATION_SEEDS


@dataclass(frozen=True)
class ValidationRecord:
    candidate_id: str
    seed: int
    metrics: dict[str, float]
    configuration: dict[str, Any]


def _mean(records: list[ValidationRecord], key: str) -> float:
    values = [float(record.metrics[key]) for record in records]
    if not values:
        raise ValueError(f"No metric records for {key}")
    return sum(values) / len(values)


def _complexity(configuration: dict[str, Any]) -> tuple[float, float, str]:
    depth = configuration.get("max_depth")
    return (float(depth) if depth is not None else inf, float(configuration.get("n_estimators", configuration.get("hidden_dim", inf))), str(sorted(configuration.items())))


def select_model(records: Iterable[ValidationRecord], *, task_type: str, required_seeds: tuple[int, ...] = VALIDATION_SEEDS) -> dict[str, Any]:
    grouped: dict[str, list[ValidationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.candidate_id].append(record)
    if not grouped:
        raise ValueError("Cannot select from empty validation records")
    summaries: list[dict[str, Any]] = []
    for candidate_id, candidate_records in grouped.items():
        seeds = {record.seed for record in candidate_records}
        if seeds != set(required_seeds):
            raise ValueError(f"Candidate {candidate_id} does not have exactly the frozen validation seeds")
        primary_key = "balanced_accuracy" if task_type == "classification" else "mae"
        secondary_key = "macro_f1" if task_type == "classification" else "rmse"
        primary = _mean(candidate_records, primary_key)
        secondary = _mean(candidate_records, secondary_key)
        summaries.append({"candidate_id": candidate_id, "configuration": candidate_records[0].configuration, "primary": primary, "secondary": secondary})
    if task_type == "classification":
        summaries.sort(key=lambda x: (-x["primary"], -x["secondary"], _complexity(x["configuration"]), x["candidate_id"]))
    elif task_type == "regression":
        summaries.sort(key=lambda x: (x["primary"], x["secondary"], _complexity(x["configuration"]), x["candidate_id"]))
    else:
        raise ValueError("task_type must be classification or regression")
    winner = summaries[0]
    return {"selected_candidate_id": winner["candidate_id"], "configuration": winner["configuration"], "mean_primary_metric": winner["primary"], "mean_secondary_metric": winner["secondary"], "ranking": summaries}
