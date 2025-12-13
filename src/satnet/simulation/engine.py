"""
Simulation Engine for SatNet Architecture DSS

This module provides the main simulation engine that integrates the Hypatia
satellite network adapter with failure injection and impact analysis.

The engine uses Tier 1 physics from HypatiaAdapter:
    - SGP4 orbital propagation
    - Link budget analysis (Optical + RF)
    - Earth obscuration checks
"""

import sys
from pathlib import Path

# Add src directory to path for direct script execution
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent.parent.parent  # src/satnet/simulation -> src
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dataclasses import dataclass
from typing import Optional

import networkx as nx

from satnet.network.hypatia_adapter import HypatiaAdapter
from satnet.simulation.failures import (
    FailureConfig,
    sample_failures,
    apply_failures,
    compute_impact,
)


@dataclass
class SimulationResult:
    num_nodes: int
    num_edges: int
    num_bottlenecks: int


class SimulationEngine:
    """
    Main simulation engine using HypatiaAdapter for network topology.
    
    Replaces the legacy toy topology generator with a realistic Walker Delta
    constellation using Tier 1 physics (SGP4, link budget, Earth obscuration).
    """
    
    def __init__(
        self,
        num_planes: int = 24,
        sats_per_plane: int = 24,
        inclination_deg: float = 53.0,
        altitude_km: float = 550.0,
        duration_minutes: int = 10,
    ):
        """
        Initialize the simulation engine with Hypatia Adapter.
        
        Args:
            num_planes: Number of orbital planes (default: 24 for Starlink-like)
            sats_per_plane: Satellites per plane (default: 24)
            inclination_deg: Orbital inclination in degrees
            altitude_km: Orbital altitude above Earth surface
            duration_minutes: ISL computation duration
        """
        print("Initializing SimulationEngine with HypatiaAdapter...")
        
        # Initialize HypatiaAdapter with constellation parameters
        self.adapter = HypatiaAdapter(
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            inclination_deg=inclination_deg,
            altitude_km=altitude_km,
        )
        
        # Generate TLEs and compute ISLs on startup
        self.adapter.generate_tles()
        self.adapter.calculate_isls(duration_minutes=duration_minutes)
        
        # Load network graph at t=0 (static for now, preserves compatibility)
        self.network_graph: nx.Graph = self.adapter.get_graph_at_step(0)
        
        print(f"Engine initialized with {self.network_graph.number_of_nodes()} nodes, "
              f"{self.network_graph.number_of_edges()} edges")
    
    def get_graph(self) -> nx.Graph:
        """Return the current network graph."""
        return self.network_graph
    
    def run(self) -> SimulationResult:
        """
        Run the simulation: assign load, detect bottlenecks, inject failures.
        
        Returns:
            SimulationResult with node/edge counts and bottleneck count
        """
        G = self.network_graph
        
        print("=== Simulation Started ===")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")
        
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
        
        print(f"Bottlenecks detected: {len(bottlenecks)}")
        
        # --- Failure injection + impact analysis ---
        fail_cfg = FailureConfig(
            node_failure_prob=0.02,
            edge_failure_prob=0.05,
            seed=42,
        )
        failures = sample_failures(G, fail_cfg)
        G_failed = apply_failures(G, failures)
        impact = compute_impact(G, G_failed)
        
        print(f"Failed nodes: {len(failures.failed_nodes)}")
        print(f"Failed edges: {len(failures.failed_edges)}")
        print(
            f"Components: {impact.num_components_before} -> "
            f"{impact.num_components_after}, "
            f"largest component: {impact.largest_component_before} -> "
            f"{impact.largest_component_after}"
        )
        
        print("=== Simulation Complete ===")
        
        return SimulationResult(
            num_nodes=G.number_of_nodes(),
            num_edges=G.number_of_edges(),
            num_bottlenecks=len(bottlenecks),
        )


def run_simulation() -> SimulationResult:
    """
    Legacy function for backward compatibility.
    
    Creates a SimulationEngine and runs the simulation.
    """
    engine = SimulationEngine()
    return engine.run()


# ---------------------------------------------------------------------------
# Main Test Block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Simulation Engine - Hypatia Adapter Integration Test")
    print("=" * 60)
    
    # Initialize engine with Starlink-like parameters
    engine = SimulationEngine(
        num_planes=24,
        sats_per_plane=24,
        inclination_deg=53.0,
        altitude_km=550.0,
        duration_minutes=10,
    )
    
    print("\n" + "=" * 60)
    print("Engine initialized with Hypatia Adapter")
    print("=" * 60)
    
    G = engine.get_graph()
    print(f"Network Graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    # Show link type breakdown
    intra = sum(1 for _, _, d in G.edges(data=True) if d.get('link_type') == 'intra_plane')
    inter = sum(1 for _, _, d in G.edges(data=True) if d.get('link_type') == 'inter_plane')
    seam = sum(1 for _, _, d in G.edges(data=True) if d.get('link_type') == 'seam_link')
    print(f"  Intra-plane links: {intra}")
    print(f"  Inter-plane links: {inter}")
    print(f"  Seam links: {seam}")
    
    # Show link mode breakdown (Tier 1 physics)
    optical = sum(1 for _, _, d in G.edges(data=True) if d.get('link_mode') == 'optical')
    rf = sum(1 for _, _, d in G.edges(data=True) if d.get('link_mode') == 'rf')
    print(f"  Optical links: {optical}")
    print(f"  RF links: {rf}")
    
    print("\n" + "=" * 60)
    print("Integration Test Complete!")
    print("=" * 60)
