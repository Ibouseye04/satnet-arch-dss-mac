"""Canonical resilience target computation from per-step metrics.

This module provides a single entry point for computing all resilience
targets from a sequence of per-step metric dictionaries. It reuses the
existing pure functions in ``satnet.metrics.labels`` and adds the
aggregation logic that was previously only available inside the rollout
runner.

The canonical targets are:

- ``partition_any``       – 1 if any sampled state has a threshold-based partition, else 0
- ``partition_fraction``  – fraction of sampled states with a resilience threshold breach
- ``gcc_frac_min``        – minimum original-denominator GCC fraction across steps
- ``gcc_frac_mean``       – mean original-denominator GCC fraction across steps
- ``gcc_frac_min_original`` – explicit alias for ``gcc_frac_min``
- ``gcc_frac_mean_original`` – explicit alias for ``gcc_frac_mean``
- ``gcc_frac_min_surviving`` – minimum current-graph GCC fraction across steps
- ``gcc_frac_mean_surviving`` – mean current-graph GCC fraction across steps
- ``max_partition_streak`` – longest consecutive run of threshold-breach sampled states
- ``max_partition_streak_seconds`` – sampled-state streak converted to physical-time-equivalent seconds
- ``max_partition_streak_fraction`` – longest threshold-breach sampled-state run divided by total sampled states
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
    "gcc_frac_min_original",
    "gcc_frac_mean_original",
    "gcc_frac_min_surviving",
    "gcc_frac_mean_surviving",
    "max_partition_streak",
    "max_partition_streak_seconds",
    "max_partition_streak_fraction",
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
    step_seconds: int = 60,
) -> dict:
    """Compute all canonical resilience targets from per-step metric dicts.

    Each element of *step_metrics* must contain at least:
    - ``gcc_frac``  (float): primary original-denominator GCC fraction
    - ``num_components`` (int): number of connected components

    ``gcc_frac_original`` and ``gcc_frac_surviving`` are used when present.
    ``gcc_frac`` remains the primary original-denominator alias. ``partitioned``
    may be pre-computed in the dict; if absent it is derived from the
    original-denominator GCC fraction being below *gcc_threshold*. This is a
    threshold-based partition/resilience breach, not strict graph-theoretic
    disconnectedness and not ``num_components > 1``.

    ``max_partition_streak_seconds`` is derived from sampled state spacing as
    ``max_partition_streak * step_seconds``. SATNET samples inclusive states at
    ``t = 0, step_seconds, ..., duration_seconds``; this value is not recovered
    from exact continuous-time failure onset or recovery boundaries.

    Returns a dict with keys matching ``ALL_TARGETS``.
    """
    if not step_metrics:
        return {
            "partition_any": 0,
            "partition_fraction": 0.0,
            "gcc_frac_min": 0.0,
            "gcc_frac_mean": 0.0,
            "gcc_frac_min_original": 0.0,
            "gcc_frac_mean_original": 0.0,
            "gcc_frac_min_surviving": 0.0,
            "gcc_frac_mean_surviving": 0.0,
            "max_partition_streak": 0,
            "max_partition_streak_seconds": 0,
            "max_partition_streak_fraction": 0.0,
        }

    gcc_fracs_original: list[float] = []
    gcc_fracs_surviving: list[float] = []
    partitioned_flags: list[int] = []

    for step in step_metrics:
        gcc_frac_original = float(step.get("gcc_frac_original", step["gcc_frac"]))
        gcc_frac_surviving = float(step.get("gcc_frac_surviving", step["gcc_frac"]))
        gcc_fracs_original.append(gcc_frac_original)
        gcc_fracs_surviving.append(gcc_frac_surviving)

        if "partitioned" in step:
            partitioned_flags.append(int(step["partitioned"]))
        else:
            partitioned_flags.append(1 if gcc_frac_original < gcc_threshold else 0)

    n = len(step_metrics)
    partition_count = sum(partitioned_flags)
    max_partition_streak = aggregate_partition_streaks(partitioned_flags)
    gcc_frac_min_original = min(gcc_fracs_original)
    gcc_frac_mean_original = sum(gcc_fracs_original) / n
    gcc_frac_min_surviving = min(gcc_fracs_surviving)
    gcc_frac_mean_surviving = sum(gcc_fracs_surviving) / n

    return {
        "partition_any": 1 if partition_count > 0 else 0,
        "partition_fraction": partition_count / n,
        "gcc_frac_min": gcc_frac_min_original,
        "gcc_frac_mean": gcc_frac_mean_original,
        "gcc_frac_min_original": gcc_frac_min_original,
        "gcc_frac_mean_original": gcc_frac_mean_original,
        "gcc_frac_min_surviving": gcc_frac_min_surviving,
        "gcc_frac_mean_surviving": gcc_frac_mean_surviving,
        "max_partition_streak": max_partition_streak,
        "max_partition_streak_seconds": max_partition_streak * step_seconds,
        "max_partition_streak_fraction": max_partition_streak / n,
    }
