"""Unit tests for satnet.simulation.monte_carlo module.

Tests validate the Tier 1 temporal Monte Carlo dataset generator.
"""

from __future__ import annotations

import pytest


class TestTier1MonteCarloConfig:
    """Tests for Tier1MonteCarloConfig dataclass."""

    def test_instantiation_defaults(self) -> None:
        """Can instantiate with defaults."""
        from satnet.simulation.monte_carlo import Tier1MonteCarloConfig

        cfg = Tier1MonteCarloConfig()
        assert cfg.num_runs == 100
        assert cfg.seed == 42

    def test_instantiation_custom(self) -> None:
        """Can instantiate with custom values."""
        from satnet.simulation.monte_carlo import Tier1MonteCarloConfig

        cfg = Tier1MonteCarloConfig(
            num_runs=10,
            num_planes_range=(4, 8),
            sats_per_plane_range=(6, 12),
            seed=999,
        )
        assert cfg.num_runs == 10
        assert cfg.num_planes_range == (4, 8)
        assert cfg.seed == 999


class TestGenerateTier1TemporalDataset:
    """Tests for generate_tier1_temporal_dataset function."""

    def test_returns_runs_and_steps(self) -> None:
        """Function returns (runs_rows, steps_rows) tuple."""
        from satnet.simulation.monte_carlo import (
            Tier1MonteCarloConfig,
            Tier1RunRow,
            Tier1StepRow,
            generate_tier1_temporal_dataset,
        )

        cfg = Tier1MonteCarloConfig(
            num_runs=2,
            num_planes_range=(2, 3),
            sats_per_plane_range=(3, 4),
            duration_minutes=1,
            step_seconds=60,
        )

        runs, steps = generate_tier1_temporal_dataset(cfg)

        assert isinstance(runs, list)
        assert isinstance(steps, list)
        assert len(runs) == 2
        assert all(isinstance(r, Tier1RunRow) for r in runs)
        assert all(isinstance(s, Tier1StepRow) for s in steps)

    def test_run_ids_sequential(self) -> None:
        """Run IDs are sequential starting from 0."""
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

        runs, _ = generate_tier1_temporal_dataset(cfg)

        run_ids = [r.run_id for r in runs]
        assert run_ids == [0, 1, 2]

    def test_steps_linked_to_runs(self) -> None:
        """Step rows reference valid run IDs."""
        from satnet.simulation.monte_carlo import (
            Tier1MonteCarloConfig,
            generate_tier1_temporal_dataset,
        )

        cfg = Tier1MonteCarloConfig(
            num_runs=2,
            num_planes_range=(2, 2),
            sats_per_plane_range=(3, 3),
            duration_minutes=1,
            step_seconds=60,
        )

        runs, steps = generate_tier1_temporal_dataset(cfg)

        run_ids = {r.run_id for r in runs}
        step_run_ids = {s.run_id for s in steps}

        assert step_run_ids.issubset(run_ids)

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces identical results."""
        from satnet.simulation.monte_carlo import (
            Tier1MonteCarloConfig,
            generate_tier1_temporal_dataset,
        )

        cfg = Tier1MonteCarloConfig(
            num_runs=2,
            num_planes_range=(2, 3),
            sats_per_plane_range=(3, 4),
            duration_minutes=1,
            step_seconds=60,
            seed=12345,
        )

        runs1, steps1 = generate_tier1_temporal_dataset(cfg)
        runs2, steps2 = generate_tier1_temporal_dataset(cfg)

        # Compare run summaries
        assert len(runs1) == len(runs2)
        for r1, r2 in zip(runs1, runs2):
            assert r1.num_planes == r2.num_planes
            assert r1.gcc_frac_min == r2.gcc_frac_min

    def test_fixed_constellation_mode(self) -> None:
        """sample_constellation=False uses midpoint of ranges."""
        from satnet.simulation.monte_carlo import (
            Tier1MonteCarloConfig,
            generate_tier1_temporal_dataset,
        )

        cfg = Tier1MonteCarloConfig(
            num_runs=3,
            num_planes_range=(4, 6),  # midpoint = 5
            sats_per_plane_range=(6, 10),  # midpoint = 8
            duration_minutes=1,
            step_seconds=60,
            sample_constellation=False,
        )

        runs, _ = generate_tier1_temporal_dataset(cfg)

        # All runs should have same constellation
        for r in runs:
            assert r.num_planes == 5
            assert r.sats_per_plane == 8

    def test_schema_version_in_rows(self) -> None:
        """Run rows include schema version."""
        from satnet.simulation.monte_carlo import (
            SCHEMA_VERSION,
            Tier1MonteCarloConfig,
            generate_tier1_temporal_dataset,
        )

        cfg = Tier1MonteCarloConfig(
            num_runs=1,
            num_planes_range=(2, 2),
            sats_per_plane_range=(3, 3),
            duration_minutes=1,
            step_seconds=60,
        )

        runs, _ = generate_tier1_temporal_dataset(cfg)

        assert runs[0].schema_version == SCHEMA_VERSION


class TestConversionFunctions:
    """Tests for runs_to_dicts and steps_to_dicts."""

    def test_runs_to_dicts(self) -> None:
        """runs_to_dicts converts to list of dicts."""
        from satnet.simulation.monte_carlo import (
            Tier1MonteCarloConfig,
            generate_tier1_temporal_dataset,
            runs_to_dicts,
        )

        cfg = Tier1MonteCarloConfig(
            num_runs=2,
            num_planes_range=(2, 2),
            sats_per_plane_range=(3, 3),
            duration_minutes=1,
            step_seconds=60,
        )

        runs, _ = generate_tier1_temporal_dataset(cfg)
        dicts = runs_to_dicts(runs)

        assert isinstance(dicts, list)
        assert len(dicts) == 2
        assert all(isinstance(d, dict) for d in dicts)
        assert "run_id" in dicts[0]
        assert "gcc_frac_min" in dicts[0]

    def test_steps_to_dicts(self) -> None:
        """steps_to_dicts converts to list of dicts."""
        from satnet.simulation.monte_carlo import (
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
        dicts = steps_to_dicts(steps)

        assert isinstance(dicts, list)
        assert all(isinstance(d, dict) for d in dicts)
        assert "run_id" in dicts[0]
        assert "gcc_frac" in dicts[0]


class TestNoToyTopologyImport:
    """Verify monte_carlo module has no toy topology dependency."""

    def test_no_topology_import(self) -> None:
        """Module does not import satnet.network.topology."""
        import satnet.simulation.monte_carlo as module

        # Check module source
        module_source = module.__file__
        assert module_source is not None
        with open(module_source) as f:
            source = f.read()
        assert "satnet.network.topology" not in source
        assert "from satnet.network.topology" not in source


class TestSchemaValidation:
    """Tests for schema validation functions."""

    def test_validate_runs_schema_valid(self) -> None:
        """Valid runs data passes validation."""
        from satnet.simulation.monte_carlo import (
            Tier1MonteCarloConfig,
            generate_tier1_temporal_dataset,
            runs_to_dicts,
            validate_runs_schema,
        )

        cfg = Tier1MonteCarloConfig(
            num_runs=2,
            num_planes_range=(2, 2),
            sats_per_plane_range=(3, 3),
            duration_minutes=1,
            step_seconds=60,
        )
        runs, _ = generate_tier1_temporal_dataset(cfg)
        runs_dicts = runs_to_dicts(runs)

        # Should not raise
        validate_runs_schema(runs_dicts)

    def test_validate_runs_schema_missing_column(self) -> None:
        """Missing required column raises SchemaValidationError."""
        from satnet.simulation.monte_carlo import (
            SchemaValidationError,
            validate_runs_schema,
        )

        invalid_data = [{"run_id": 0, "num_planes": 3}]  # Missing most columns

        with pytest.raises(SchemaValidationError, match="missing required columns"):
            validate_runs_schema(invalid_data)

    def test_validate_runs_schema_invalid_range(self) -> None:
        """Out-of-range value raises SchemaValidationError."""
        from satnet.simulation.monte_carlo import (
            RUNS_REQUIRED_COLUMNS,
            SCHEMA_VERSION,
            DATASET_VERSION,
            SchemaValidationError,
            validate_runs_schema,
        )

        # Create a row with all required columns but invalid gcc_frac_min
        row = {col: 0 for col in RUNS_REQUIRED_COLUMNS}
        row["gcc_frac_min"] = 1.5  # Invalid: > 1.0
        row["gcc_frac_mean"] = 0.5
        row["partition_fraction"] = 0.5
        row["partition_any"] = 0
        row["schema_version"] = SCHEMA_VERSION
        row["dataset_version"] = DATASET_VERSION
        row["config_hash"] = "abc"

        with pytest.raises(SchemaValidationError, match="out of range"):
            validate_runs_schema([row])

    def test_validate_steps_schema_valid(self) -> None:
        """Valid steps data passes validation."""
        from satnet.simulation.monte_carlo import (
            Tier1MonteCarloConfig,
            generate_tier1_temporal_dataset,
            steps_to_dicts,
            validate_steps_schema,
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

        # Should not raise
        validate_steps_schema(steps_dicts)

    def test_validate_steps_schema_missing_column(self) -> None:
        """Missing required column raises SchemaValidationError."""
        from satnet.simulation.monte_carlo import (
            SchemaValidationError,
            validate_steps_schema,
        )

        invalid_data = [{"run_id": 0, "t": 0}]  # Missing most columns

        with pytest.raises(SchemaValidationError, match="missing required columns"):
            validate_steps_schema(invalid_data)

    def test_validate_empty_data(self) -> None:
        """Empty data passes validation."""
        from satnet.simulation.monte_carlo import (
            validate_runs_schema,
            validate_steps_schema,
        )

        # Should not raise
        validate_runs_schema([])
        validate_steps_schema([])
