from dataclasses import dataclass

import networkx as nx

from satnet.network.topology import TopologyConfig, generate_topology
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


def run_simulation() -> SimulationResult:
    """Build topology, assign load, detect bottlenecks, inject failures."""

    # --- Build topology ---
    cfg = TopologyConfig()
    G: nx.Graph = generate_topology(cfg)

    print("=== Simulation Started ===")
    print(f"Nodes: {len(G.nodes)}")
    print(f"Edges: {len(G.edges)}")

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
        num_nodes=len(G.nodes),
        num_edges=len(G.edges),
        num_bottlenecks=len(bottlenecks),
    )
