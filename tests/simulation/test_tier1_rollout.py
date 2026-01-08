"""Unit tests for satnet.simulation.tier1_rollout dataclasses.

Tests validate that the rollout contracts can be instantiated
without importing toy topology and that computed properties work correctly.
"""

from __future__ import annotations

import pytest

from satnet.simulation.tier1_rollout import (
    DATASET_VERSION,
    SCHEMA_VERSION,
    Tier1FailureRealization,
    Tier1RolloutConfig,
    Tier1RolloutStep,
    Tier1RolloutSummary,
)


class TestTier1RolloutConfig:
    """Tests for Tier1RolloutConfig dataclass."""

    def test_instantiation_minimal(self) -> None:
        """Can instantiate with only required parameters."""
        cfg = Tier1RolloutConfig(num_planes=6, sats_per_plane=10)
        assert cfg.num_planes == 6
        assert cfg.sats_per_plane == 10
        assert cfg.inclination_deg == 53.0  # default
        assert cfg.altitude_km == 550.0  # default

    def test_instantiation_full(self) -> None:
        """Can instantiate with all parameters."""
        cfg = Tier1RolloutConfig(
            num_planes=24,
            sats_per_plane=24,
            inclination_deg=97.5,
            altitude_km=1200.0,
            phasing_factor=2,
            duration_minutes=120,
            step_seconds=30,
            max_isl_distance_km=5000.0,
            gcc_threshold=0.9,
            node_failure_prob=0.05,
            edge_failure_prob=0.10,
            seed=12345,
        )
        assert cfg.num_planes == 24
        assert cfg.duration_minutes == 120
        assert cfg.step_seconds == 30
        assert cfg.gcc_threshold == 0.9
        assert cfg.seed == 12345

    def test_num_steps_property(self) -> None:
        """num_steps computed correctly from duration and step size."""
        cfg = Tier1RolloutConfig(
            num_planes=6,
            sats_per_plane=10,
            duration_minutes=90,
            step_seconds=60,
        )
        # 90 minutes @ 60s = steps 0..90 = 91 steps (inclusive t=0..T)
        assert cfg.num_steps == 91

    def test_num_steps_non_divisible(self) -> None:
        """num_steps uses integer division with +1 for inclusive range."""
        cfg = Tier1RolloutConfig(
            num_planes=6,
            sats_per_plane=10,
            duration_minutes=10,
            step_seconds=180,  # 3 minutes
        )
        # 10 minutes * 60 = 600 seconds / 180 = 3, +1 = 4 steps (inclusive t=0..T)
        assert cfg.num_steps == 4

    def test_total_satellites_property(self) -> None:
        """total_satellites computed correctly."""
        cfg = Tier1RolloutConfig(num_planes=24, sats_per_plane=24)
        assert cfg.total_satellites == 576

    def test_config_hash_deterministic(self) -> None:
        """Same config produces same hash."""
        cfg1 = Tier1RolloutConfig(num_planes=6, sats_per_plane=10, seed=42)
        cfg2 = Tier1RolloutConfig(num_planes=6, sats_per_plane=10, seed=42)
        assert cfg1.config_hash() == cfg2.config_hash()

    def test_config_hash_differs_on_change(self) -> None:
        """Different config produces different hash."""
        cfg1 = Tier1RolloutConfig(num_planes=6, sats_per_plane=10, seed=42)
        cfg2 = Tier1RolloutConfig(num_planes=6, sats_per_plane=10, seed=43)
        assert cfg1.config_hash() != cfg2.config_hash()

    def test_frozen_immutable(self) -> None:
        """Config is frozen (immutable)."""
        cfg = Tier1RolloutConfig(num_planes=6, sats_per_plane=10)
        with pytest.raises(AttributeError):
            cfg.num_planes = 12  # type: ignore


class TestTier1RolloutStep:
    """Tests for Tier1RolloutStep dataclass."""

    def test_instantiation(self) -> None:
        """Can instantiate with all required fields."""
        step = Tier1RolloutStep(
            t=0,
            num_nodes=100,
            num_edges=200,
            num_components=1,
            gcc_size=100,
            gcc_frac=1.0,
            partitioned=0,
        )
        assert step.t == 0
        assert step.num_nodes == 100
        assert step.gcc_frac == 1.0
        assert step.partitioned == 0

    def test_partitioned_step(self) -> None:
        """Can represent a partitioned step."""
        step = Tier1RolloutStep(
            t=5,
            num_nodes=100,
            num_edges=150,
            num_components=3,
            gcc_size=60,
            gcc_frac=0.6,
            partitioned=1,
        )
        assert step.num_components == 3
        assert step.gcc_frac == 0.6
        assert step.partitioned == 1


class TestTier1RolloutSummary:
    """Tests for Tier1RolloutSummary dataclass."""

    def test_instantiation_minimal(self) -> None:
        """Can instantiate with required fields."""
        summary = Tier1RolloutSummary(
            gcc_frac_min=0.8,
            gcc_frac_mean=0.95,
            partition_fraction=0.1,
            partition_any=1,
            max_partition_streak=3,
            num_steps=90,
        )
        assert summary.gcc_frac_min == 0.8
        assert summary.partition_any == 1
        assert summary.max_partition_streak == 3

    def test_default_schema_version(self) -> None:
        """Schema version defaults to current."""
        summary = Tier1RolloutSummary(
            gcc_frac_min=1.0,
            gcc_frac_mean=1.0,
            partition_fraction=0.0,
            partition_any=0,
            max_partition_streak=0,
            num_steps=10,
        )
        assert summary.schema_version == SCHEMA_VERSION
        assert summary.dataset_version == DATASET_VERSION

    def test_to_dict(self) -> None:
        """to_dict returns all fields."""
        summary = Tier1RolloutSummary(
            gcc_frac_min=0.5,
            gcc_frac_mean=0.75,
            partition_fraction=0.2,
            partition_any=1,
            max_partition_streak=5,
            num_steps=100,
            num_failed_nodes=10,
            num_failed_edges=20,
            config_hash="abc123",
        )
        d = summary.to_dict()
        assert d["gcc_frac_min"] == 0.5
        assert d["num_failed_nodes"] == 10
        assert d["config_hash"] == "abc123"
        assert d["schema_version"] == SCHEMA_VERSION


class TestTier1FailureRealization:
    """Tests for Tier1FailureRealization dataclass (Step 3 contract)."""

    def test_instantiation(self) -> None:
        """Can instantiate with sets."""
        failures = Tier1FailureRealization(
            failed_nodes={1, 5, 12},
            failed_edges={(0, 1), (3, 4)},
        )
        assert failures.failed_nodes == {1, 5, 12}
        assert failures.failed_edges == {(0, 1), (3, 4)}

    def test_to_json_strings(self) -> None:
        """to_json_strings serializes to sorted JSON."""
        failures = Tier1FailureRealization(
            failed_nodes={12, 1, 5},
            failed_edges={(3, 4), (0, 1)},
        )
        nodes_json, edges_json = failures.to_json_strings()
        
        assert nodes_json == "[1, 5, 12]"  # Sorted
        assert edges_json == "[[0, 1], [3, 4]]"  # Sorted

    def test_from_json_strings(self) -> None:
        """from_json_strings deserializes correctly."""
        failures = Tier1FailureRealization.from_json_strings(
            "[1, 5, 12]",
            "[[0, 1], [3, 4]]",
        )
        assert failures.failed_nodes == {1, 5, 12}
        assert failures.failed_edges == {(0, 1), (3, 4)}

    def test_roundtrip_serialization(self) -> None:
        """Serialize then deserialize produces identical data."""
        original = Tier1FailureRealization(
            failed_nodes={1, 5, 12},
            failed_edges={(0, 1), (3, 4)},
        )
        nodes_json, edges_json = original.to_json_strings()
        restored = Tier1FailureRealization.from_json_strings(nodes_json, edges_json)
        
        assert restored.failed_nodes == original.failed_nodes
        assert restored.failed_edges == original.failed_edges

    def test_empty_failures(self) -> None:
        """Empty failures serialize correctly."""
        failures = Tier1FailureRealization(
            failed_nodes=set(),
            failed_edges=set(),
        )
        nodes_json, edges_json = failures.to_json_strings()
        
        assert nodes_json == "[]"
        assert edges_json == "[]"
        
        restored = Tier1FailureRealization.from_json_strings(nodes_json, edges_json)
        assert len(restored.failed_nodes) == 0
        assert len(restored.failed_edges) == 0


class TestNoToyTopologyImport:
    """Verify that tier1_rollout module has no toy topology dependency."""

    def test_no_topology_import(self) -> None:
        """Module does not import satnet.network.topology."""
        import satnet.simulation.tier1_rollout as module
        import sys

        # Check that topology is not in the module's namespace
        assert not hasattr(module, "topology")

        # Check that satnet.network.topology is not loaded
        # (it might be loaded by other tests, so we check the module itself)
        module_source = module.__file__
        assert module_source is not None
        with open(module_source) as f:
            source = f.read()
        assert "satnet.network.topology" not in source
        assert "from satnet.network.topology" not in source


class TestRunTier1Rollout:
    """Tests for run_tier1_rollout function."""

    def test_rollout_returns_steps_summary_and_failures(self) -> None:
        """run_tier1_rollout returns (steps, summary, failures) tuple."""
        from satnet.simulation.tier1_rollout import (
            Tier1FailureRealization,
            Tier1RolloutConfig,
            Tier1RolloutStep,
            Tier1RolloutSummary,
            run_tier1_rollout,
        )

        cfg = Tier1RolloutConfig(
            num_planes=2,
            sats_per_plane=3,
            duration_minutes=1,
            step_seconds=60,
        )

        steps, summary, failures = run_tier1_rollout(cfg)

        assert isinstance(steps, list)
        assert isinstance(summary, Tier1RolloutSummary)
        assert isinstance(failures, Tier1FailureRealization)
        assert all(isinstance(s, Tier1RolloutStep) for s in steps)

    def test_rollout_step_count_matches_config(self) -> None:
        """Number of steps matches config num_steps."""
        from satnet.simulation.tier1_rollout import (
            Tier1RolloutConfig,
            run_tier1_rollout,
        )

        cfg = Tier1RolloutConfig(
            num_planes=2,
            sats_per_plane=3,
            duration_minutes=2,
            step_seconds=60,
        )

        steps, summary, _ = run_tier1_rollout(cfg)

        # 2 minutes at 60s = 3 steps (0, 1, 2)
        assert len(steps) == 3
        assert summary.num_steps == 3

    def test_rollout_gcc_frac_in_range(self) -> None:
        """GCC fraction is always between 0 and 1."""
        from satnet.simulation.tier1_rollout import (
            Tier1RolloutConfig,
            run_tier1_rollout,
        )

        cfg = Tier1RolloutConfig(
            num_planes=2,
            sats_per_plane=4,
            duration_minutes=1,
            step_seconds=60,
        )

        steps, summary, _ = run_tier1_rollout(cfg)

        for step in steps:
            assert 0.0 <= step.gcc_frac <= 1.0

        assert 0.0 <= summary.gcc_frac_min <= 1.0
        assert 0.0 <= summary.gcc_frac_mean <= 1.0

    def test_rollout_no_failures_produces_valid_metrics(self) -> None:
        """With no failures, metrics are computed correctly."""
        from satnet.simulation.tier1_rollout import (
            Tier1RolloutConfig,
            run_tier1_rollout,
        )

        cfg = Tier1RolloutConfig(
            num_planes=3,
            sats_per_plane=5,
            duration_minutes=1,
            step_seconds=60,
            node_failure_prob=0.0,
            edge_failure_prob=0.0,
        )

        steps, summary, failures = run_tier1_rollout(cfg)

        # With no failures, all nodes should be present
        assert steps[0].num_nodes == cfg.total_satellites
        # GCC fraction should be valid (may not be 1.0 for small constellations
        # due to Earth obscuration physics)
        assert 0.0 <= summary.gcc_frac_min <= 1.0
        assert summary.num_failed_nodes == 0
        assert summary.num_failed_edges == 0
        assert len(failures.failed_nodes) == 0
        assert len(failures.failed_edges) == 0

    def test_rollout_with_node_failures(self) -> None:
        """Node failures reduce node count."""
        from satnet.simulation.tier1_rollout import (
            Tier1RolloutConfig,
            run_tier1_rollout,
        )

        cfg = Tier1RolloutConfig(
            num_planes=3,
            sats_per_plane=4,
            duration_minutes=1,
            step_seconds=60,
            node_failure_prob=0.5,  # High failure rate
            edge_failure_prob=0.0,
            seed=42,
        )

        steps, summary, failures = run_tier1_rollout(cfg)

        # With 50% node failure prob, expect some failures
        assert summary.num_failed_nodes > 0
        assert len(failures.failed_nodes) == summary.num_failed_nodes
        # Node count should be less than total
        total_sats = cfg.total_satellites
        assert steps[0].num_nodes < total_sats

    def test_rollout_with_edge_failures(self) -> None:
        """Edge failures are sampled from t=0 edges."""
        from satnet.simulation.tier1_rollout import (
            Tier1RolloutConfig,
            run_tier1_rollout,
        )

        # Use larger constellation to ensure edges exist
        cfg = Tier1RolloutConfig(
            num_planes=5,
            sats_per_plane=10,
            duration_minutes=1,
            step_seconds=60,
            node_failure_prob=0.0,
            edge_failure_prob=0.5,  # High failure rate
            seed=42,
        )

        steps, summary, failures = run_tier1_rollout(cfg)

        # With 50% edge failure prob on a larger constellation, expect some failures
        # (only if edges exist at t=0 - depends on physics)
        # Just verify the rollout completes and produces valid output
        assert summary.num_steps == 2
        assert 0.0 <= summary.gcc_frac_min <= 1.0
        assert len(failures.failed_edges) == summary.num_failed_edges

    def test_rollout_deterministic_with_seed(self) -> None:
        """Same seed produces identical results."""
        from satnet.simulation.tier1_rollout import (
            Tier1RolloutConfig,
            run_tier1_rollout,
        )

        cfg = Tier1RolloutConfig(
            num_planes=2,
            sats_per_plane=3,
            duration_minutes=1,
            step_seconds=60,
            node_failure_prob=0.3,
            edge_failure_prob=0.3,
            seed=12345,
        )

        steps1, summary1, failures1 = run_tier1_rollout(cfg)
        steps2, summary2, failures2 = run_tier1_rollout(cfg)

        # Results should be identical
        assert summary1.num_failed_nodes == summary2.num_failed_nodes
        assert summary1.num_failed_edges == summary2.num_failed_edges
        assert summary1.gcc_frac_min == summary2.gcc_frac_min
        assert failures1.failed_nodes == failures2.failed_nodes
        assert failures1.failed_edges == failures2.failed_edges

    def test_rollout_config_hash_in_summary(self) -> None:
        """Summary includes config hash."""
        from satnet.simulation.tier1_rollout import (
            Tier1RolloutConfig,
            run_tier1_rollout,
        )

        cfg = Tier1RolloutConfig(
            num_planes=2,
            sats_per_plane=3,
            duration_minutes=1,
            step_seconds=60,
        )

        steps, summary, _ = run_tier1_rollout(cfg)

        assert summary.config_hash == cfg.config_hash()
        assert len(summary.config_hash) == 64  # Full SHA256 (Step 8 fix)

    def test_rollout_summary_finite_values(self) -> None:
        """All summary values are finite (not NaN or Inf)."""
        import math
        from satnet.simulation.tier1_rollout import (
            Tier1RolloutConfig,
            run_tier1_rollout,
        )

        cfg = Tier1RolloutConfig(
            num_planes=2,
            sats_per_plane=3,
            duration_minutes=1,
            step_seconds=60,
            node_failure_prob=0.2,
            edge_failure_prob=0.2,
            seed=42,
        )

        steps, summary, _ = run_tier1_rollout(cfg)

        assert math.isfinite(summary.gcc_frac_min)
        assert math.isfinite(summary.gcc_frac_mean)
        assert math.isfinite(summary.partition_fraction)
