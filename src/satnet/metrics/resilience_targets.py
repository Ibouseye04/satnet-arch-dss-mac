"""Canonical resilience target computation from per-step metrics.

This module provides a single entry point for computing all resilience
targets from a sequence of per-step metric dictionaries. It reuses the
existing pure functions in ``satnet.metrics.labels`` and adds the
aggregation logic that was previously only available inside the rollout
runner.

The canonical targets are:

- ``partition_any``       – 1 if any step is partitioned, else 0
- ``partition_fraction``  – fraction of steps where the graph is partitioned
- ``gcc_frac_min``        – minimum GCC fraction across steps
- ``gcc_frac_mean``       – mean GCC fraction across steps
- ``max_partition_streak``– longest consecutive run of partitioned steps
"""

from __future__ import annotations

from typing import Literal

from satnet.metrics.labels import aggregate_partition_streaks


# ── target taxonomy ─────────────────────────────────────────────────

BINARY_TARGETS = frozenset({"partition_any"})
CONTINUOUS_TARGETS = frozenset({
    "partition_fraction",
    "gcc_frac_min",
    "gcc_frac_mean",
    "max_partition_streak",
})
ALL_TARGETS = BINARY_TARGETS | CONTINUOUS_TARGETS


def infer_task_type(target_name: str) -> Literal["classification", "regression"]:
    """Return the task type implied by *target_name*.

    Raises ``ValueError`` for unknown target names.
    """
    if target_name in BINARY_TARGETS:
        return "classification"
    if target_name in CONTINUOUS_TARGETS:
        return "regression"
    raise ValueError(
        f"Unknown target '{target_name}'. Must be one of {sorted(ALL_TARGETS)}"
    )


# ── core computation ────────────────────────────────────────────────

def compute_resilience_targets(
    step_metrics: list[dict],
    gcc_threshold: float = 0.8,
) -> dict:
    """Compute all canonical resilience targets from per-step metric dicts.

    Each element of *step_metrics* must contain at least:
    - ``gcc_frac``  (float): GCC fraction at that time step
    - ``num_components`` (int): number of connected components

    ``partitioned`` may be pre-computed in the dict; if absent it is
    derived from *gcc_frac* < *gcc_threshold*.

    Returns a dict with keys matching ``ALL_TARGETS``.
    """
    if not step_metrics:
        return {
            "partition_any": 0,
            "partition_fraction": 0.0,
            "gcc_frac_min": 0.0,
            "gcc_frac_mean": 0.0,
            "max_partition_streak": 0,
        }

    gcc_fracs: list[float] = []
    partitioned_flags: list[int] = []

    for step in step_metrics:
        gcc_frac = float(step["gcc_frac"])
        gcc_fracs.append(gcc_frac)

        if "partitioned" in step:
            partitioned_flags.append(int(step["partitioned"]))
        else:
            partitioned_flags.append(1 if gcc_frac < gcc_threshold else 0)

    n = len(step_metrics)
    partition_count = sum(partitioned_flags)

    return {
        "partition_any": 1 if partition_count > 0 else 0,
        "partition_fraction": partition_count / n,
        "gcc_frac_min": min(gcc_fracs),
        "gcc_frac_mean": sum(gcc_fracs) / n,
        "max_partition_streak": aggregate_partition_streaks(partitioned_flags),
    }
