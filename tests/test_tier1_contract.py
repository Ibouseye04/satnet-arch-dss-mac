"""Regression tests for the Tier 1-only contract.

These tests ensure that:
1. No src/ module imports satnet.network.topology (toy topology)
2. Temporal rollout produces correct output shapes
3. The refactor maintains the Tier 1-only invariant
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


class TestNoToyImports:
    """Verify no src/ module imports the toy topology generator."""

    @pytest.fixture
    def src_python_files(self) -> list[Path]:
        """Get all Python files in src/."""
        src_dir = Path(__file__).parent.parent / "src"
        return list(src_dir.rglob("*.py"))

    def test_no_topology_import_in_src(self, src_python_files) -> None:
        """No src/ module imports satnet.network.topology."""
        violations = []

        for py_file in src_python_files:
            content = py_file.read_text()

            # Check for direct string matches
            if "satnet.network.topology" in content:
                violations.append(str(py_file))
            if "from satnet.network.topology" in content:
                violations.append(str(py_file))

        assert not violations, (
            f"Found toy topology imports in src/:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_topology_file_deleted(self) -> None:
        """The toy topology file should not exist in src/."""
        topology_path = (
            Path(__file__).parent.parent
            / "src"
            / "satnet"
            / "network"
            / "topology.py"
        )
        assert not topology_path.exists(), (
            f"Toy topology file still exists: {topology_path}"
        )


class TestRolloutShapes:
    """Verify temporal rollout produces correct output shapes."""

    def test_rollout_step_count(self) -> None:
        """Rollout produces expected number of steps."""
        from satnet.simulation.tier1_rollout import (
            Tier1RolloutConfig,
            run_tier1_rollout,
        )

        cfg = Tier1RolloutConfig(
            num_planes=2,
            sats_per_plane=3,
            duration_minutes=3,
            step_seconds=60,
        )

        steps, summary, _ = run_tier1_rollout(cfg)

        # 3 minutes at 60s = 4 steps (0, 1, 2, 3)
        expected_steps = 4
        assert len(steps) == expected_steps
        assert summary.num_steps == expected_steps

    def test_rollout_step_indices_sequential(self) -> None:
        """Step indices are sequential starting from 0."""
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

        steps, _, _ = run_tier1_rollout(cfg)

        step_indices = [s.t for s in steps]
        expected = list(range(len(steps)))
        assert step_indices == expected

    def test_monte_carlo_output_shapes(self) -> None:
        """Monte Carlo produces matching runs and steps counts."""
        from satnet.simulation.monte_carlo import (
            Tier1MonteCarloConfig,
            generate_tier1_temporal_dataset,
        )

        cfg = Tier1MonteCarloConfig(
            num_runs=3,
            num_planes_range=(2, 2),
            sats_per_plane_range=(3, 3),
            duration_minutes=1,
            step_seconds=60,
        )

        runs, steps = generate_tier1_temporal_dataset(cfg)

        # Should have exactly 3 runs
        assert len(runs) == 3

        # Each run has 2 steps (1 min at 60s = 2 steps: 0, 1)
        steps_per_run = 2
        assert len(steps) == 3 * steps_per_run

        # Verify run_id linkage
        run_ids_in_steps = {s.run_id for s in steps}
        run_ids_in_runs = {r.run_id for r in runs}
        assert run_ids_in_steps == run_ids_in_runs


class TestSchemaCompliance:
    """Verify generated data complies with v1 schema."""

    def test_runs_have_required_columns(self) -> None:
        """Run rows have all required v1 schema columns."""
        from satnet.simulation.monte_carlo import (
            RUNS_REQUIRED_COLUMNS,
            Tier1MonteCarloConfig,
            generate_tier1_temporal_dataset,
            runs_to_dicts,
        )

        cfg = Tier1MonteCarloConfig(
            num_runs=1,
            num_planes_range=(2, 2),
            sats_per_plane_range=(3, 3),
            duration_minutes=1,
            step_seconds=60,
        )

        runs, _ = generate_tier1_temporal_dataset(cfg)
        runs_dicts = runs_to_dicts(runs)

        present_columns = set(runs_dicts[0].keys())
        missing = RUNS_REQUIRED_COLUMNS - present_columns

        assert not missing, f"Missing required columns: {missing}"

    def test_steps_have_required_columns(self) -> None:
        """Step rows have all required v1 schema columns."""
        from satnet.simulation.monte_carlo import (
            STEPS_REQUIRED_COLUMNS,
            Tier1MonteCarloConfig,
            generate_tier1_temporal_dataset,
            steps_to_dicts,
        )

        cfg = Tier1MonteCarloConfig(
            num_runs=1,
            num_planes_range=(2, 2),
            sats_per_plane_range=(3, 3),
            duration_minutes=1,
            step_seconds=60,
        )

        _, steps = generate_tier1_temporal_dataset(cfg)
        steps_dicts = steps_to_dicts(steps)

        present_columns = set(steps_dicts[0].keys())
        missing = STEPS_REQUIRED_COLUMNS - present_columns

        assert not missing, f"Missing required columns: {missing}"


class TestDeterminism:
    """Verify reproducibility with same seed."""

    def test_same_seed_same_results(self) -> None:
        """Same config + seed produces identical results."""
        from satnet.simulation.tier1_rollout import (
            Tier1RolloutConfig,
            run_tier1_rollout,
        )

        cfg = Tier1RolloutConfig(
            num_planes=2,
            sats_per_plane=4,
            duration_minutes=1,
            step_seconds=60,
            node_failure_prob=0.2,
            edge_failure_prob=0.2,
            seed=99999,
        )

        steps1, summary1, _ = run_tier1_rollout(cfg)
        steps2, summary2, _ = run_tier1_rollout(cfg)

        # Summaries should be identical
        assert summary1.gcc_frac_min == summary2.gcc_frac_min
        assert summary1.gcc_frac_mean == summary2.gcc_frac_mean
        assert summary1.num_failed_nodes == summary2.num_failed_nodes
        assert summary1.num_failed_edges == summary2.num_failed_edges
        assert summary1.config_hash == summary2.config_hash

        # Step-by-step should be identical
        for s1, s2 in zip(steps1, steps2):
            assert s1.gcc_frac == s2.gcc_frac
            assert s1.partitioned == s2.partitioned
