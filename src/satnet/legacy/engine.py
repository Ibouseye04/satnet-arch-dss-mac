"""Legacy Simulation Engine — QUARANTINED.

This module is preserved for reference but should NOT be used in Tier 1 code.
It uses static t=0 snapshots and lacks explicit epoch handling.

For Tier 1 temporal evaluation, use:
- satnet.simulation.tier1_rollout.run_tier1_rollout()
- satnet.simulation.monte_carlo.generate_tier1_temporal_dataset()

Issues with this module (from audit):
- HypatiaAdapter constructed without explicit epoch (falls back to utcnow())
- SimulationEngine.run() uses graph_at_t0 (static snapshot, not temporal)
- Placeholder "fake load" logic
"""

import logging
from dataclasses import dataclass
from typing import Optional

import networkx as nx

from satnet.network.hypatia_adapter import HypatiaAdapter
from satnet.legacy.failures import (
    FailureConfig,
    sample_failures,
    apply_failures,
    compute_impact,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    num_nodes: int
    num_edges: int
    num_bottlenecks: int


class SimulationEngine:
    """
    DEPRECATED: Legacy simulation engine using HypatiaAdapter for network topology.
    
    This engine uses static t=0 snapshots and lacks explicit epoch handling.
    For Tier 1 temporal evaluation, use run_tier1_rollout() instead.
    """
    
    def __init__(
        self,
        num_planes: int = 24,
        sats_per_plane: int = 24,
        inclination_deg: float = 53.0,
        altitude_km: float = 550.0,
        duration_minutes: int = 10,
        step_seconds: int = 60,
    ):
        """
        Initialize the simulation engine with Hypatia Adapter.
        
        WARNING: This does NOT pass an explicit epoch to HypatiaAdapter,
        so it falls back to datetime.utcnow() (nondeterministic).
        """
        self._duration_minutes = duration_minutes
        self._step_seconds = step_seconds
        
        # Initialize HypatiaAdapter with constellation parameters
        # NOTE: No epoch passed - this is the bug documented in the audit
        self.adapter = HypatiaAdapter(
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            inclination_deg=inclination_deg,
            altitude_km=altitude_km,
        )
        
        # Generate TLEs and compute ISLs on startup
        self.adapter.generate_tles()
        self.adapter.calculate_isls(
            duration_minutes=duration_minutes,
            step_seconds=step_seconds,
        )
        
        logger.info(
            "Engine initialized with %d satellites, %d time steps",
            self.adapter.total_satellites,
            self.adapter.num_steps,
        )
    
    @property
    def graph_at_t0(self) -> nx.Graph:
        """Return the network graph at t=0.
        
        DEPRECATED: Use iter_graphs() or get_graph_at_step() for temporal evaluation.
        This property exists only for backward compatibility.
        """
        return self.adapter.get_graph_at_step(0)
    
    # Backward compatibility aliases (deprecated)
    @property
    def network_graph(self) -> nx.Graph:
        """DEPRECATED: Use graph_at_t0 or iter_graphs() instead."""
        return self.graph_at_t0
    
    def get_graph(self) -> nx.Graph:
        """DEPRECATED: Use get_graph_at_step() or iter_graphs() instead."""
        return self.graph_at_t0
    
    def get_graph_at_step(self, time_step: int) -> nx.Graph:
        """Return the network graph at a specific time step."""
        return self.adapter.get_graph_at_step(time_step)
    
    def iter_graphs(self):
        """Iterate over network graphs for all time steps."""
        return self.adapter.iter_graphs()
    
    @property
    def num_steps(self) -> int:
        """Return the number of computed time steps."""
        return self.adapter.num_steps
    
    @property
    def step_seconds(self) -> int:
        """Return the time step interval in seconds."""
        return self._step_seconds
    
    @property
    def duration_minutes(self) -> int:
        """Return the simulation duration in minutes."""
        return self._duration_minutes
    
    def run(self) -> SimulationResult:
        """
        Run the simulation: assign load, detect bottlenecks, inject failures.
        
        DEPRECATED: This method uses graph_at_t0 (static snapshot). For temporal
        evaluation, use run_tier1_rollout() instead.
        """
        G = self.graph_at_t0
        
        logger.info("Simulation started: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        
        # --- Fake load assignment ---
        for u, v, data in G.edges(data=True):
            cap = data.get("capacity", 10.0)
            data["load"] = 0.6 * cap  # 60% utilization everywhere for now
        
        # --- Bottleneck detection ---
        bottlenecks = []
        for u, v, data in G.edges(data=True):
            cap = data.get("capacity", 10.0)
            load = data.get("load", 0.0)
            if cap > 0 and load / cap > 0.8:
                bottlenecks.append((u, v))
        
        logger.debug("Bottlenecks detected: %d", len(bottlenecks))
        
        # --- Failure injection + impact analysis ---
        fail_cfg = FailureConfig(
            node_failure_prob=0.02,
            edge_failure_prob=0.05,
            seed=42,
        )
        failures = sample_failures(G, fail_cfg)
        G_failed = apply_failures(G, failures)
        impact = compute_impact(G, G_failed)
        
        logger.debug(
            "Failures: %d nodes, %d edges. Components: %d -> %d",
            len(failures.failed_nodes),
            len(failures.failed_edges),
            impact.num_components_before,
            impact.num_components_after,
        )
        
        logger.info("Simulation complete")
        
        return SimulationResult(
            num_nodes=G.number_of_nodes(),
            num_edges=G.number_of_edges(),
            num_bottlenecks=len(bottlenecks),
        )


def run_simulation() -> SimulationResult:
    """
    DEPRECATED: Legacy function for backward compatibility.
    
    Use satnet.simulation.tier1_rollout.run_tier1_rollout() instead.
    """
    engine = SimulationEngine()
    return engine.run()
