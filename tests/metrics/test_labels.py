"""Unit tests for satnet.metrics.labels module.

Tests validate the pure labeling functions for:
- Empty graph handling
- Connected graph → gcc_frac == 1
- Multi-component graph → gcc_frac < 1
- Partition streak aggregation
"""

from __future__ import annotations

import networkx as nx
import pytest

from satnet.metrics.labels import (
    aggregate_partition_streaks,
    compute_gcc_frac,
    compute_gcc_size,
    compute_num_components,
    compute_partitioned,
)


class TestComputeNumComponents:
    """Tests for compute_num_components."""

    def test_empty_graph(self) -> None:
        """Empty graph should have 0 components."""
        G = nx.Graph()
        assert compute_num_components(G) == 0

    def test_single_node(self) -> None:
        """Single isolated node is one component."""
        G = nx.Graph()
        G.add_node(0)
        assert compute_num_components(G) == 1

    def test_connected_graph(self) -> None:
        """Fully connected graph has 1 component."""
        G = nx.complete_graph(5)
        assert compute_num_components(G) == 1

    def test_two_components(self) -> None:
        """Two disconnected cliques form 2 components."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])  # Component 1
        G.add_edges_from([(3, 4), (4, 5)])  # Component 2
        assert compute_num_components(G) == 2

    def test_isolated_nodes(self) -> None:
        """Each isolated node is its own component."""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        assert compute_num_components(G) == 4


class TestComputeGccSize:
    """Tests for compute_gcc_size."""

    def test_empty_graph(self) -> None:
        """Empty graph has GCC size 0."""
        G = nx.Graph()
        assert compute_gcc_size(G) == 0

    def test_single_node(self) -> None:
        """Single node graph has GCC size 1."""
        G = nx.Graph()
        G.add_node(0)
        assert compute_gcc_size(G) == 1

    def test_connected_graph(self) -> None:
        """Connected graph: GCC size equals total nodes."""
        G = nx.path_graph(10)
        assert compute_gcc_size(G) == 10

    def test_two_unequal_components(self) -> None:
        """GCC is the larger of two components."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])  # 4 nodes
        G.add_edges_from([(10, 11)])  # 2 nodes
        assert compute_gcc_size(G) == 4


class TestComputeGccFrac:
    """Tests for compute_gcc_frac."""

    def test_empty_graph(self) -> None:
        """Empty graph returns 0.0 (no division by zero)."""
        G = nx.Graph()
        assert compute_gcc_frac(G) == 0.0

    def test_connected_graph(self) -> None:
        """Fully connected graph has gcc_frac == 1.0."""
        G = nx.complete_graph(10)
        assert compute_gcc_frac(G) == 1.0

    def test_two_equal_components(self) -> None:
        """Two equal components: gcc_frac == 0.5."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])  # 3 nodes
        G.add_edges_from([(3, 4), (4, 5)])  # 3 nodes
        assert compute_gcc_frac(G) == 0.5

    def test_one_large_one_small(self) -> None:
        """Unequal components: gcc_frac reflects larger component."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])  # 4 nodes
        G.add_edge(10, 11)  # 2 nodes
        # Total: 6 nodes, GCC: 4 nodes
        assert compute_gcc_frac(G) == pytest.approx(4 / 6)

    def test_all_isolated(self) -> None:
        """All isolated nodes: gcc_frac == 1/n."""
        G = nx.Graph()
        G.add_nodes_from(range(10))
        # Each node is its own component of size 1
        assert compute_gcc_frac(G) == pytest.approx(1 / 10)


class TestComputePartitioned:
    """Tests for compute_partitioned."""

    def test_above_threshold(self) -> None:
        """gcc_frac >= threshold → not partitioned (0)."""
        assert compute_partitioned(0.9, 0.8) == 0
        assert compute_partitioned(0.8, 0.8) == 0
        assert compute_partitioned(1.0, 0.8) == 0

    def test_below_threshold(self) -> None:
        """gcc_frac < threshold → partitioned (1)."""
        assert compute_partitioned(0.79, 0.8) == 1
        assert compute_partitioned(0.5, 0.8) == 1
        assert compute_partitioned(0.0, 0.8) == 1

    def test_edge_cases(self) -> None:
        """Edge cases for threshold boundaries."""
        assert compute_partitioned(0.0, 0.0) == 0
        assert compute_partitioned(0.0, 0.01) == 1


class TestAggregatePartitionStreaks:
    """Tests for aggregate_partition_streaks."""

    def test_empty_list(self) -> None:
        """Empty list returns 0."""
        assert aggregate_partition_streaks([]) == 0

    def test_no_partitions(self) -> None:
        """All zeros returns 0."""
        assert aggregate_partition_streaks([0, 0, 0, 0]) == 0

    def test_all_partitioned(self) -> None:
        """All ones returns length of list."""
        assert aggregate_partition_streaks([1, 1, 1, 1]) == 4

    def test_single_streak(self) -> None:
        """Single streak in the middle."""
        assert aggregate_partition_streaks([0, 1, 1, 1, 0]) == 3

    def test_multiple_streaks(self) -> None:
        """Multiple streaks: returns the longest."""
        assert aggregate_partition_streaks([1, 1, 0, 1, 1, 1, 0, 1]) == 3

    def test_streak_at_end(self) -> None:
        """Streak at the end of the list."""
        assert aggregate_partition_streaks([0, 0, 1, 1, 1, 1]) == 4

    def test_streak_at_start(self) -> None:
        """Streak at the start of the list."""
        assert aggregate_partition_streaks([1, 1, 0, 0, 0]) == 2

    def test_alternating(self) -> None:
        """Alternating pattern: max streak is 1."""
        assert aggregate_partition_streaks([1, 0, 1, 0, 1, 0]) == 1
