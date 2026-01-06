"""Tier 1 Guardrail Tests — Step 0 of Structural Cleanup.

These tests enforce two critical invariants:

1. **No datetime.utcnow() default epoch reachable from Tier 1 entrypoints.**
   The Tier 1 pipeline must always use an explicit epoch (DEFAULT_EPOCH_ISO)
   to guarantee deterministic, reproducible results.

2. **No static t=0-only code paths in Tier 1.**
   Tier 1 evaluation must iterate over t=0..T, not just grab a single snapshot.

These are regression tests that fail loudly if legacy patterns are reintroduced.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Set

import pytest


# ---------------------------------------------------------------------------
# Test 1: No datetime.utcnow() default epoch reachable from Tier 1 entrypoints
# ---------------------------------------------------------------------------


class TestNoUtcnowEpochInTier1:
    """Verify Tier 1 entrypoints always pass explicit epoch to HypatiaAdapter."""

    def test_tier1_rollout_passes_explicit_epoch(self) -> None:
        """run_tier1_rollout passes cfg.epoch to HypatiaAdapter."""
        from satnet.simulation.tier1_rollout import run_tier1_rollout
        
        # Read the source code of run_tier1_rollout
        source = inspect.getsource(run_tier1_rollout)
        
        # Check that HypatiaAdapter is called with epoch=cfg.epoch
        assert "epoch=cfg.epoch" in source, (
            "run_tier1_rollout must pass epoch=cfg.epoch to HypatiaAdapter. "
            "Found no explicit epoch parameter in HypatiaAdapter construction."
        )

    def test_tier1_rollout_config_has_explicit_epoch_default(self) -> None:
        """Tier1RolloutConfig defaults to DEFAULT_EPOCH_ISO, not utcnow()."""
        from satnet.simulation.tier1_rollout import (
            DEFAULT_EPOCH_ISO,
            Tier1RolloutConfig,
        )
        
        # Create config with minimal params
        cfg = Tier1RolloutConfig(num_planes=2, sats_per_plane=3)
        
        # Verify epoch_iso is the fixed default, not dynamic
        assert cfg.epoch_iso == DEFAULT_EPOCH_ISO, (
            f"Tier1RolloutConfig.epoch_iso should default to {DEFAULT_EPOCH_ISO}, "
            f"got {cfg.epoch_iso}"
        )
        
        # Verify epoch property parses correctly
        assert cfg.epoch.year == 2000
        assert cfg.epoch.month == 1
        assert cfg.epoch.day == 1

    def test_monte_carlo_uses_tier1_rollout_epoch(self) -> None:
        """Monte Carlo dataset generation uses Tier1RolloutConfig (which has explicit epoch)."""
        from satnet.simulation.monte_carlo import generate_tier1_temporal_dataset
        
        source = inspect.getsource(generate_tier1_temporal_dataset)
        
        # Should use Tier1RolloutConfig and run_tier1_rollout
        assert "Tier1RolloutConfig" in source, (
            "generate_tier1_temporal_dataset must use Tier1RolloutConfig"
        )
        assert "run_tier1_rollout" in source, (
            "generate_tier1_temporal_dataset must use run_tier1_rollout"
        )

    def test_walker_delta_config_default_epoch_is_utcnow(self) -> None:
        """WalkerDeltaConfig defaults to utcnow() — this is the footgun we guard against.
        
        This test documents the current (problematic) behavior. The fix is to ensure
        all Tier 1 code paths pass an explicit epoch to HypatiaAdapter, bypassing
        this default.
        """
        from satnet.network.hypatia_adapter import WalkerDeltaConfig
        from datetime import datetime
        
        # Create two configs in quick succession
        cfg1 = WalkerDeltaConfig()
        cfg2 = WalkerDeltaConfig()
        
        # Both should have epochs close to now (within 1 second)
        now = datetime.utcnow()
        delta1 = abs((cfg1.epoch - now).total_seconds())
        delta2 = abs((cfg2.epoch - now).total_seconds())
        
        # This test passes if the default is utcnow() (documenting the footgun)
        # If this test fails, it means someone fixed the default — update the test!
        assert delta1 < 2.0, (
            "WalkerDeltaConfig.epoch default should be datetime.utcnow(). "
            "If this changed, update this test and the audit docs."
        )

    def test_hypatia_adapter_without_epoch_uses_utcnow(self) -> None:
        """HypatiaAdapter without epoch param falls back to utcnow() — the footgun.
        
        This test documents the current behavior. Tier 1 code must always pass
        an explicit epoch to avoid this fallback.
        """
        from satnet.network.hypatia_adapter import HypatiaAdapter
        from datetime import datetime
        
        # Create adapter without epoch
        adapter = HypatiaAdapter(num_planes=2, sats_per_plane=2)
        
        # The config.epoch should be close to now
        now = datetime.utcnow()
        delta = abs((adapter.config.epoch - now).total_seconds())
        
        # This test passes if the fallback is utcnow() (documenting the footgun)
        assert delta < 2.0, (
            "HypatiaAdapter without epoch should fall back to utcnow(). "
            "If this changed, update this test."
        )


class TestEpochContractEnforced:
    """Verify all HypatiaAdapter callers pass explicit epoch (Step 2 fixes)."""

    def test_gnn_dataset_passes_explicit_epoch(self) -> None:
        """SatNetTemporalDataset.get() passes explicit epoch to HypatiaAdapter.
        
        Step 2 fixed the H2 audit issue: epoch is now passed explicitly.
        """
        from satnet.models.gnn_dataset import SatNetTemporalDataset
        
        source = inspect.getsource(SatNetTemporalDataset.get)
        
        # Verify HypatiaAdapter is called with epoch=
        assert "epoch=" in source, (
            "SatNetTemporalDataset.get() must pass epoch= to HypatiaAdapter. "
            "This is required for Tier 1 determinism."
        )
        
        # Verify DEFAULT_EPOCH_ISO is used as fallback
        assert "DEFAULT_EPOCH_ISO" in source or "epoch_iso" in source, (
            "SatNetTemporalDataset.get() should use DEFAULT_EPOCH_ISO or epoch_iso from CSV."
        )


class TestLegacyEngineQuarantined:
    """Verify legacy engine has been moved to satnet.legacy namespace."""

    def test_simulation_engine_not_in_simulation_namespace(self) -> None:
        """SimulationEngine is no longer importable from satnet.simulation.engine.
        
        Step 1 moved it to satnet.legacy.engine.
        """
        engine_path = Path(__file__).parent.parent / "src" / "satnet" / "simulation" / "engine.py"
        assert not engine_path.exists(), (
            f"Legacy engine still exists at {engine_path}. "
            "It should be moved to satnet/legacy/engine.py"
        )

    def test_failures_not_in_simulation_namespace(self) -> None:
        """failures.py is no longer in satnet.simulation.
        
        Step 1 moved it to satnet.legacy.failures.
        """
        failures_path = Path(__file__).parent.parent / "src" / "satnet" / "simulation" / "failures.py"
        assert not failures_path.exists(), (
            f"Legacy failures module still exists at {failures_path}. "
            "It should be moved to satnet/legacy/failures.py"
        )

    def test_legacy_engine_exists_in_legacy_namespace(self) -> None:
        """SimulationEngine is available in satnet.legacy.engine."""
        legacy_engine_path = Path(__file__).parent.parent / "src" / "satnet" / "legacy" / "engine.py"
        assert legacy_engine_path.exists(), (
            f"Legacy engine not found at {legacy_engine_path}. "
            "It should be quarantined in satnet/legacy/"
        )

    def test_legacy_failures_exists_in_legacy_namespace(self) -> None:
        """failures.py is available in satnet.legacy.failures."""
        legacy_failures_path = Path(__file__).parent.parent / "src" / "satnet" / "legacy" / "failures.py"
        assert legacy_failures_path.exists(), (
            f"Legacy failures not found at {legacy_failures_path}. "
            "It should be quarantined in satnet/legacy/"
        )


# ---------------------------------------------------------------------------
# Test 2: No static t=0-only code paths in Tier 1
# ---------------------------------------------------------------------------


class TestNoStaticSnapshotInTier1:
    """Verify Tier 1 entrypoints iterate over time, not just t=0."""

    def test_tier1_rollout_iterates_over_time(self) -> None:
        """run_tier1_rollout uses iter_graphs() for temporal evaluation."""
        from satnet.simulation.tier1_rollout import run_tier1_rollout
        
        source = inspect.getsource(run_tier1_rollout)
        
        # Should use iter_graphs() for temporal iteration
        assert "iter_graphs()" in source, (
            "run_tier1_rollout must use iter_graphs() for temporal evaluation. "
            "Found no iter_graphs() call."
        )
        
        # Should NOT just use get_graph_at_step(0) and stop
        # (It's OK to use get_graph_at_step(0) for failure sampling, but must also iterate)
        assert "for t, G_t in" in source or "for t," in source, (
            "run_tier1_rollout must loop over time steps. "
            "Found no temporal loop pattern."
        )

    def test_tier1_rollout_produces_multiple_steps(self) -> None:
        """run_tier1_rollout actually produces multiple time steps."""
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
        
        steps, summary = run_tier1_rollout(cfg)
        
        # 2 minutes at 60s = 3 steps (0, 1, 2)
        assert len(steps) == 3, f"Expected 3 steps, got {len(steps)}"
        assert summary.num_steps == 3
        
        # Verify step indices are sequential
        step_indices = [s.t for s in steps]
        assert step_indices == [0, 1, 2], f"Expected [0, 1, 2], got {step_indices}"

    def test_monte_carlo_produces_temporal_steps(self) -> None:
        """Monte Carlo generates steps for all time points, not just t=0."""
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
        
        # 1 minute at 60s = 2 steps per run (0, 1)
        steps_per_run = 2
        expected_total_steps = 2 * steps_per_run
        
        assert len(steps) == expected_total_steps, (
            f"Expected {expected_total_steps} steps (2 runs × {steps_per_run} steps), "
            f"got {len(steps)}"
        )


class TestLegacyEngineIsStatic:
    """Verify quarantined legacy engine still uses static snapshots (for documentation)."""

    def test_legacy_engine_run_uses_graph_at_t0(self) -> None:
        """Quarantined SimulationEngine.run() uses graph_at_t0 (static snapshot).
        
        This documents the legacy behavior in the quarantined module.
        """
        from satnet.legacy.engine import SimulationEngine
        
        source = inspect.getsource(SimulationEngine.run)
        
        # Legacy engine uses graph_at_t0 (static snapshot)
        uses_static = "graph_at_t0" in source or "get_graph_at_step(0)" in source
        
        assert uses_static, (
            "Quarantined SimulationEngine.run() no longer uses graph_at_t0. "
            "The legacy module should preserve its original behavior for reference."
        )


# ---------------------------------------------------------------------------
# Test 3: Tier 1 scripts don't import legacy engine
# ---------------------------------------------------------------------------


class TestTier1ScriptsNoLegacyImports:
    """Verify Tier 1 scripts don't import the legacy simulation engine."""

    @pytest.fixture
    def tier1_scripts(self) -> list[Path]:
        """Get the Tier 1 script files."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        return [
            scripts_dir / "export_design_dataset.py",
            scripts_dir / "export_failure_dataset.py",
            scripts_dir / "failure_sweep.py",
        ]

    def test_tier1_scripts_no_engine_import(self, tier1_scripts) -> None:
        """Tier 1 scripts don't import satnet.simulation.engine."""
        violations = []
        
        for script_path in tier1_scripts:
            if not script_path.exists():
                continue
            
            content = script_path.read_text()
            
            if "satnet.simulation.engine" in content:
                violations.append(str(script_path))
            if "from satnet.simulation.engine" in content:
                violations.append(str(script_path))
            if "SimulationEngine" in content:
                violations.append(str(script_path))
        
        assert not violations, (
            f"Tier 1 scripts should not import legacy engine:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_simulate_script_removed(self) -> None:
        """scripts/simulate.py has been removed (Step 1 cleanup).
        
        The legacy simulate.py script used the deprecated SimulationEngine.
        It has been removed as part of the Tier 1 cleanup.
        """
        simulate_path = Path(__file__).parent.parent / "scripts" / "simulate.py"
        
        assert not simulate_path.exists(), (
            f"scripts/simulate.py still exists at {simulate_path}. "
            "It should be removed as part of Step 1 cleanup."
        )
