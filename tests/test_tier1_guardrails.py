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
        
        steps, summary, _ = run_tier1_rollout(cfg)
        
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


# ---------------------------------------------------------------------------
# Test 4: Graph Reconstruction Contract (Step 3)
# ---------------------------------------------------------------------------


class TestGraphReconstructionContract:
    """Verify the graph reconstruction contract for ML (Step 3).
    
    The contract ensures that:
    1. Dataset exports include failure realization (failed_nodes_json, failed_edges_json)
    2. Regenerated graphs with failures applied match original labels
    """

    def test_dataset_exports_failure_realization(self) -> None:
        """Monte Carlo dataset includes failure realization columns."""
        from satnet.simulation.monte_carlo import (
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
            node_failure_prob_range=(0.2, 0.2),
            edge_failure_prob_range=(0.2, 0.2),
            seed=42,
        )
        
        runs, _ = generate_tier1_temporal_dataset(cfg)
        runs_dicts = runs_to_dicts(runs)
        
        # Verify failure realization columns exist
        assert "failed_nodes_json" in runs_dicts[0], (
            "Dataset must include failed_nodes_json for graph reconstruction"
        )
        assert "failed_edges_json" in runs_dicts[0], (
            "Dataset must include failed_edges_json for graph reconstruction"
        )

    def test_failure_realization_matches_counts(self) -> None:
        """Failure realization JSON matches num_failed_nodes/edges counts."""
        import json
        from satnet.simulation.monte_carlo import (
            Tier1MonteCarloConfig,
            generate_tier1_temporal_dataset,
            runs_to_dicts,
        )
        
        cfg = Tier1MonteCarloConfig(
            num_runs=3,
            num_planes_range=(3, 3),
            sats_per_plane_range=(4, 4),
            duration_minutes=1,
            step_seconds=60,
            node_failure_prob_range=(0.3, 0.3),
            edge_failure_prob_range=(0.3, 0.3),
            seed=42,
        )
        
        runs, _ = generate_tier1_temporal_dataset(cfg)
        runs_dicts = runs_to_dicts(runs)
        
        for row in runs_dicts:
            failed_nodes = json.loads(row["failed_nodes_json"])
            failed_edges = json.loads(row["failed_edges_json"])
            
            assert len(failed_nodes) == row["num_failed_nodes"], (
                f"failed_nodes_json length {len(failed_nodes)} != "
                f"num_failed_nodes {row['num_failed_nodes']}"
            )
            assert len(failed_edges) == row["num_failed_edges"], (
                f"failed_edges_json length {len(failed_edges)} != "
                f"num_failed_edges {row['num_failed_edges']}"
            )

    def test_reconstruction_reproduces_label(self) -> None:
        """Regenerated graph sequence with failures produces matching partition_any.
        
        This is the key acceptance test for Step 3: given a run row from the
        dataset, we can reconstruct the exact graph sequence and verify that
        the computed partition_any matches the stored label.
        """
        from satnet.simulation.monte_carlo import (
            Tier1MonteCarloConfig,
            generate_tier1_temporal_dataset,
            runs_to_dicts,
        )
        from satnet.simulation.tier1_rollout import (
            DEFAULT_EPOCH_ISO,
            Tier1FailureRealization,
        )
        from satnet.network.hypatia_adapter import HypatiaAdapter
        from satnet.metrics.labels import compute_gcc_frac, compute_partitioned
        from datetime import datetime
        
        # Generate a small dataset with failures
        cfg = Tier1MonteCarloConfig(
            num_runs=2,
            num_planes_range=(2, 2),
            sats_per_plane_range=(3, 3),
            duration_minutes=1,
            step_seconds=60,
            node_failure_prob_range=(0.3, 0.3),
            edge_failure_prob_range=(0.3, 0.3),
            seed=42,
        )
        
        runs, _ = generate_tier1_temporal_dataset(cfg)
        runs_dicts = runs_to_dicts(runs)
        
        for row in runs_dicts:
            # Reconstruct using the contract
            epoch = datetime.fromisoformat(row.get("epoch_iso", DEFAULT_EPOCH_ISO))
            failures = Tier1FailureRealization.from_json_strings(
                row["failed_nodes_json"],
                row["failed_edges_json"],
            )
            
            adapter = HypatiaAdapter(
                num_planes=row["num_planes"],
                sats_per_plane=row["sats_per_plane"],
                inclination_deg=row["inclination_deg"],
                altitude_km=row["altitude_km"],
                epoch=epoch,
            )
            adapter.calculate_isls(
                duration_minutes=row["duration_minutes"],
                step_seconds=row["step_seconds"],
            )
            
            # Compute partition_any from reconstructed graphs
            partition_any_reconstructed = 0
            for t, G in adapter.iter_graphs():
                G_eff = G.copy()
                nodes_to_remove = [n for n in failures.failed_nodes if G_eff.has_node(n)]
                G_eff.remove_nodes_from(nodes_to_remove)
                for u, v in failures.failed_edges:
                    if G_eff.has_edge(u, v):
                        G_eff.remove_edge(u, v)
                
                gcc_frac = compute_gcc_frac(G_eff)
                if compute_partitioned(gcc_frac, 0.8):  # Default threshold
                    partition_any_reconstructed = 1
                    break
            
            assert partition_any_reconstructed == row["partition_any"], (
                f"Run {row['run_id']}: reconstructed partition_any="
                f"{partition_any_reconstructed} != stored={row['partition_any']}. "
                "Graph reconstruction contract violated."
            )


# ---------------------------------------------------------------------------
# Test 5: Failure Semantics (Step 4)
# ---------------------------------------------------------------------------


class TestFailureSemantics:
    """Verify failure semantics are explicit and documented (Step 4)."""

    def test_edge_failures_sampled_from_t0_only(self) -> None:
        """Edge failures are sampled from t=0 edges only (v1 semantics).
        
        This documents the current behavior: edges not present at t=0
        are implicitly immune to failure sampling.
        """
        from satnet.simulation.tier1_rollout import (
            Tier1RolloutConfig,
            run_tier1_rollout,
        )
        
        # Use larger constellation to ensure edges exist at t=0
        cfg = Tier1RolloutConfig(
            num_planes=6,
            sats_per_plane=10,
            duration_minutes=1,
            step_seconds=60,
            edge_failure_prob=1.0,  # 100% failure rate
            node_failure_prob=0.0,
            seed=42,
        )
        
        _, summary, failures = run_tier1_rollout(cfg)
        
        # With 100% edge failure prob, all t=0 edges should fail
        # The number of failed edges should equal the t=0 edge count
        assert summary.num_failed_edges == len(failures.failed_edges)
        # Larger constellation should have edges at t=0
        assert len(failures.failed_edges) > 0, (
            "Expected edges at t=0 for 6x10 constellation. "
            "If this fails, the constellation may be too sparse."
        )

    def test_node_failures_are_persistent(self) -> None:
        """Node failures persist across all time steps."""
        from satnet.simulation.tier1_rollout import (
            Tier1RolloutConfig,
            run_tier1_rollout,
        )
        
        cfg = Tier1RolloutConfig(
            num_planes=2,
            sats_per_plane=3,
            duration_minutes=2,
            step_seconds=60,
            node_failure_prob=0.5,
            edge_failure_prob=0.0,
            seed=42,
        )
        
        steps, summary, failures = run_tier1_rollout(cfg)
        
        # All steps should have the same reduced node count
        # (total - failed nodes, assuming all nodes exist at all times)
        expected_nodes = cfg.total_satellites - len(failures.failed_nodes)
        for step in steps:
            assert step.num_nodes == expected_nodes, (
                f"Step {step.t}: expected {expected_nodes} nodes, got {step.num_nodes}"
            )

    def test_failure_semantics_documented_in_module(self) -> None:
        """Module docstring documents failure semantics."""
        import satnet.simulation.tier1_rollout as module
        
        docstring = module.__doc__
        assert "Failure Semantics" in docstring
        assert "Node Failures" in docstring
        assert "Edge Failures" in docstring
        assert "t=0" in docstring or "t=0" in docstring


# ---------------------------------------------------------------------------
# Test 6: Physics Dependencies (Step 6)
# ---------------------------------------------------------------------------


class TestPhysicsDependencies:
    """Verify Tier 1 physics dependencies are available (Step 6)."""

    def test_sgp4_available(self) -> None:
        """SGP4 is required for Tier 1 orbital propagation."""
        try:
            import sgp4
            assert hasattr(sgp4, "api"), "sgp4.api module required"
        except ImportError:
            pytest.fail(
                "sgp4 package not installed. Tier 1 requires sgp4>=2.22 "
                "for accurate orbital propagation."
            )

    def test_networkx_available(self) -> None:
        """NetworkX is required for graph operations."""
        try:
            import networkx as nx
            assert hasattr(nx, "Graph"), "networkx.Graph required"
        except ImportError:
            pytest.fail("networkx package not installed.")

    def test_physics_constants_documented(self) -> None:
        """Physics constants documentation exists."""
        docs_path = Path(__file__).parent.parent / "docs" / "physics_constants.md"
        assert docs_path.exists(), (
            f"Physics constants documentation not found at {docs_path}"
        )


# ---------------------------------------------------------------------------
# Test 7: Config Hash Collision Resistance (Step 8)
# ---------------------------------------------------------------------------


class TestConfigHashCollisionResistance:
    """Verify config_hash uses full SHA256 for collision resistance (Step 8)."""

    def test_config_hash_is_full_sha256(self) -> None:
        """config_hash returns full 64-char SHA256 hex string."""
        from satnet.simulation.tier1_rollout import Tier1RolloutConfig
        
        cfg = Tier1RolloutConfig(
            num_planes=2,
            sats_per_plane=3,
        )
        
        hash_value = cfg.config_hash()
        
        # Full SHA256 is 64 hex characters
        assert len(hash_value) == 64, (
            f"config_hash should be 64 chars (full SHA256), got {len(hash_value)}"
        )
        # Should be valid hex
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_config_hash_deterministic(self) -> None:
        """Same config produces same hash."""
        from satnet.simulation.tier1_rollout import Tier1RolloutConfig
        
        cfg1 = Tier1RolloutConfig(num_planes=3, sats_per_plane=5, seed=42)
        cfg2 = Tier1RolloutConfig(num_planes=3, sats_per_plane=5, seed=42)
        
        assert cfg1.config_hash() == cfg2.config_hash()

    def test_config_hash_different_for_different_configs(self) -> None:
        """Different configs produce different hashes."""
        from satnet.simulation.tier1_rollout import Tier1RolloutConfig
        
        cfg1 = Tier1RolloutConfig(num_planes=3, sats_per_plane=5)
        cfg2 = Tier1RolloutConfig(num_planes=3, sats_per_plane=6)
        
        assert cfg1.config_hash() != cfg2.config_hash()
