"""Pure labeling functions for satellite network connectivity analysis.

All functions in this module are pure (stateless) and operate only on
graph state. They have no knowledge of failure parameters or simulation
context — labels are computed solely from the graph structure.

This module has no dependency on Hypatia or any simulation code.
"""

from __future__ import annotations

import networkx as nx


def compute_num_components(G: nx.Graph) -> int:
    """Compute the number of connected components in the graph.

    Args:
        G: A NetworkX graph.

    Returns:
        Number of connected components. Returns 0 for an empty graph.
    """
    if G.number_of_nodes() == 0:
        return 0
    return nx.number_connected_components(G)


def compute_gcc_size(G: nx.Graph) -> int:
    """Compute the size (node count) of the Giant Connected Component.

    The GCC is the largest connected component by node count.

    Args:
        G: A NetworkX graph.

    Returns:
        Number of nodes in the largest connected component.
        Returns 0 for an empty graph.
    """
    if G.number_of_nodes() == 0:
        return 0
    return len(max(nx.connected_components(G), key=len))


def compute_gcc_frac(G: nx.Graph) -> float:
    """Compute the fraction of nodes in the Giant Connected Component.

    Args:
        G: A NetworkX graph.

    Returns:
        Fraction of nodes in the GCC: |GCC| / |V|.
        Returns 0.0 for an empty graph (avoids division by zero).
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    return compute_gcc_size(G) / n


def compute_partitioned(gcc_frac: float, threshold: float) -> int:
    """Determine if the network is partitioned based on GCC fraction.

    A network is considered "partitioned" if the GCC fraction falls
    below the specified threshold.

    Args:
        gcc_frac: The GCC fraction (0.0 to 1.0).
        threshold: The threshold below which the network is partitioned.

    Returns:
        1 if gcc_frac < threshold (partitioned), 0 otherwise.
    """
    return 1 if gcc_frac < threshold else 0


def aggregate_partition_streaks(partitioned: list[int]) -> int:
    """Compute the maximum consecutive streak of partitioned time steps.

    Args:
        partitioned: A list of partition indicators (0 or 1) for each time step.

    Returns:
        The length of the longest consecutive run of 1s.
        Returns 0 if the list is empty or contains no partitions.
    """
    if not partitioned:
        return 0

    max_streak = 0
    current_streak = 0

    for p in partitioned:
        if p == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak
