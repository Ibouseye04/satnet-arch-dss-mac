from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Literal, Dict, Any
import random

import networkx as nx

FailureType = Literal["node", "edge"]


@dataclass
class FailureConfig:
    """Configuration for random failure injection."""

    node_failure_prob: float = 0.02   # per-node failure probability
    edge_failure_prob: float = 0.05   # per-edge failure probability
    max_failures: int | None = None   # optional global cap
    seed: int | None = 42             # reproducibility


@dataclass
class FailureSet:
    """Concrete set of sampled failures on a given graph."""

    failed_nodes: List[str]
    failed_edges: List[Tuple[str, str]]


@dataclass
class FailureImpact:
    """Simple impact summary for one failure scenario."""

    nodes_before: int
    nodes_after: int
    edges_before: int
    edges_after: int
    num_components_before: int
    num_components_after: int
    largest_component_before: int
    largest_component_after: int


def sample_failures(G: nx.Graph, cfg: FailureConfig) -> FailureSet:
    """
    Sample random node + edge failures given per-element probabilities.
    Does NOT modify the graph.
    """
    rng = random.Random(cfg.seed)

    failed_nodes: List[str] = []
    failed_edges: List[Tuple[str, str]] = []

    # node failures
    for n in G.nodes:
        if rng.random() < cfg.node_failure_prob:
            failed_nodes.append(n)

    # edge failures
    for u, v in G.edges:
        if rng.random() < cfg.edge_failure_prob:
            failed_edges.append((u, v))

    # optional global cap
    if cfg.max_failures is not None:
        combined: List[Tuple[FailureType, Any]] = [
            ("node", n) for n in failed_nodes
        ] + [
            ("edge", e) for e in failed_edges
        ]
        rng.shuffle(combined)
        combined = combined[: cfg.max_failures]

        failed_nodes = [x for t, x in combined if t == "node"]
        failed_edges = [x for t, x in combined if t == "edge"]

    return FailureSet(failed_nodes=failed_nodes, failed_edges=failed_edges)


def apply_failures(G: nx.Graph, failures: FailureSet) -> nx.Graph:
    """
    Return a NEW graph with the given failures applied.
    Original graph is not mutated.
    """
    H = G.copy()
    # remove nodes first (this also removes incident edges)
    if failures.failed_nodes:
        H.remove_nodes_from(failures.failed_nodes)
    # then remove explicit failed edges (if they still exist)
    for u, v in failures.failed_edges:
        if H.has_edge(u, v):
            H.remove_edge(u, v)
    return H


def _component_stats(G: nx.Graph) -> Tuple[int, int]:
    """Number of connected components, and size of largest component."""
    if G.number_of_nodes() == 0:
        return 0, 0

    comps = list(nx.connected_components(G))
    num_components = len(comps)
    largest = max(len(c) for c in comps)
    return num_components, largest


def compute_impact(G_before: nx.Graph, G_after: nx.Graph) -> FailureImpact:
    """Compute a simple structural impact summary."""
    num_components_before, largest_before = _component_stats(G_before)
    num_components_after, largest_after = _component_stats(G_after)

    return FailureImpact(
        nodes_before=G_before.number_of_nodes(),
        nodes_after=G_after.number_of_nodes(),
        edges_before=G_before.number_of_edges(),
        edges_after=G_after.number_of_edges(),
        num_components_before=num_components_before,
        num_components_after=num_components_after,
        largest_component_before=largest_before,
        largest_component_after=largest_after,
    )