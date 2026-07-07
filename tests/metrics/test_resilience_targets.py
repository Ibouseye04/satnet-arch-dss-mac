"""Unit tests for satnet.metrics.resilience_targets."""

from __future__ import annotations

import pytest

from satnet.metrics.resilience_targets import (
    ALL_TARGETS,
    compute_resilience_targets,
    infer_task_type,
)


# ── infer_task_type ─────────────────────────────────────────────────


class TestInferTaskType:
    def test_binary_target(self) -> None:
        assert infer_task_type("partition_any") == "classification"

    def test_continuous_targets(self) -> None:
        for name in (
            "gcc_frac_min",
            "gcc_frac_mean",
            "gcc_frac_min_original",
            "gcc_frac_mean_original",
            "gcc_frac_min_surviving",
            "gcc_frac_mean_surviving",
            "partition_fraction",
            "max_partition_streak",
            "max_partition_streak_seconds",
            "max_partition_streak_fraction",
        ):
            assert infer_task_type(name) == "regression"

    def test_unknown_target(self) -> None:
        with pytest.raises(ValueError, match="Unknown target"):
            infer_task_type("bogus_metric")


# ── compute_resilience_targets ──────────────────────────────────────


def _make_step(gcc_frac: float, num_components: int = 1) -> dict:
    return {"gcc_frac": gcc_frac, "num_components": num_components}


class TestComputeResilienceTargets:
    def test_empty_steps(self) -> None:
        result = compute_resilience_targets([])
        assert result["partition_any"] == 0
        assert result["gcc_frac_min"] == 0.0
        assert result["gcc_frac_mean"] == 0.0
        assert result["gcc_frac_min_original"] == 0.0
        assert result["gcc_frac_mean_original"] == 0.0
        assert result["gcc_frac_min_surviving"] == 0.0
        assert result["gcc_frac_mean_surviving"] == 0.0
        assert result["partition_fraction"] == 0.0
        assert result["max_partition_streak"] == 0
        assert result["max_partition_streak_seconds"] == 0
        assert result["max_partition_streak_fraction"] == 0.0

    def test_never_partitioned(self) -> None:
        steps = [_make_step(1.0), _make_step(0.95), _make_step(0.9)]
        result = compute_resilience_targets(steps, gcc_threshold=0.8)
        assert result["partition_any"] == 0
        assert result["partition_fraction"] == 0.0
        assert result["max_partition_streak"] == 0
        assert result["gcc_frac_min"] == pytest.approx(0.9)
        assert result["gcc_frac_mean"] == pytest.approx((1.0 + 0.95 + 0.9) / 3)
        assert result["gcc_frac_min_original"] == pytest.approx(0.9)
        assert result["gcc_frac_mean_original"] == pytest.approx((1.0 + 0.95 + 0.9) / 3)

    def test_always_partitioned(self) -> None:
        steps = [_make_step(0.5, 3), _make_step(0.4, 4), _make_step(0.3, 5)]
        result = compute_resilience_targets(steps, gcc_threshold=0.8)
        assert result["partition_any"] == 1
        assert result["partition_fraction"] == pytest.approx(1.0)
        assert result["max_partition_streak"] == 3
        assert result["max_partition_streak_seconds"] == 180
        assert result["max_partition_streak_fraction"] == pytest.approx(1.0)
        assert result["gcc_frac_min"] == pytest.approx(0.3)

    def test_intermittent_partitioning(self) -> None:
        steps = [
            _make_step(1.0),   # ok
            _make_step(0.5),   # partitioned
            _make_step(0.5),   # partitioned
            _make_step(1.0),   # ok
            _make_step(0.6),   # partitioned
            _make_step(1.0),   # ok
        ]
        result = compute_resilience_targets(steps, gcc_threshold=0.8)
        assert result["partition_any"] == 1
        assert result["partition_fraction"] == pytest.approx(3 / 6)
        assert result["max_partition_streak"] == 2
        assert result["max_partition_streak_seconds"] == 120
        assert result["max_partition_streak_fraction"] == pytest.approx(2 / 6)
        assert result["gcc_frac_min"] == pytest.approx(0.5)

    def test_partition_streak_physical_time_changes_with_step_seconds(self) -> None:
        flags = [0, 1, 1, 0, 1, 1, 1, 0, 0, 0]
        steps = [
            {"gcc_frac": 1.0, "num_components": 1, "partitioned": flag}
            for flag in flags
        ]

        result_60s = compute_resilience_targets(steps, step_seconds=60)
        result_30s = compute_resilience_targets(steps, step_seconds=30)

        assert result_60s["max_partition_streak"] == 3
        assert result_60s["max_partition_streak_seconds"] == 180
        assert result_60s["max_partition_streak_fraction"] == pytest.approx(3 / 10)
        assert result_30s["max_partition_streak"] == 3
        assert result_30s["max_partition_streak_seconds"] == 90
        assert result_30s["max_partition_streak_fraction"] == pytest.approx(3 / 10)

    def test_single_timestep_partitioned(self) -> None:
        steps = [_make_step(0.3, 4)]
        result = compute_resilience_targets(steps, gcc_threshold=0.8)
        assert result["partition_any"] == 1
        assert result["partition_fraction"] == 1.0
        assert result["max_partition_streak"] == 1
        assert result["gcc_frac_min"] == pytest.approx(0.3)
        assert result["gcc_frac_mean"] == pytest.approx(0.3)

    def test_single_timestep_healthy(self) -> None:
        steps = [_make_step(0.95)]
        result = compute_resilience_targets(steps, gcc_threshold=0.8)
        assert result["partition_any"] == 0

    def test_precomputed_partitioned_flag(self) -> None:
        steps = [
            {"gcc_frac": 0.9, "num_components": 1, "partitioned": 1},
            {"gcc_frac": 0.9, "num_components": 1, "partitioned": 0},
        ]
        result = compute_resilience_targets(steps)
        assert result["partition_any"] == 1
        assert result["partition_fraction"] == pytest.approx(0.5)

    def test_explicit_original_and_surviving_gcc_targets(self) -> None:
        steps = [
            {
                "gcc_frac": 0.6,
                "gcc_frac_original": 0.6,
                "gcc_frac_surviving": 1.0,
                "num_components": 1,
            },
        ]
        result = compute_resilience_targets(steps, gcc_threshold=0.8)
        assert result["partition_any"] == 1
        assert result["gcc_frac_min"] == pytest.approx(0.6)
        assert result["gcc_frac_mean"] == pytest.approx(0.6)
        assert result["gcc_frac_min_original"] == pytest.approx(0.6)
        assert result["gcc_frac_mean_original"] == pytest.approx(0.6)
        assert result["gcc_frac_min_surviving"] == pytest.approx(1.0)
        assert result["gcc_frac_mean_surviving"] == pytest.approx(1.0)

    def test_all_target_keys_present(self) -> None:
        result = compute_resilience_targets([_make_step(1.0)])
        assert set(result.keys()) == ALL_TARGETS
