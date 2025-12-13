"""Unit tests for HypatiaAdapter.iter_graphs() method.

Tests validate that the graph iterator:
- Yields graphs without exceptions
- Reuses cached ISL data
- Supports range parameters
- Raises appropriate errors when ISLs not calculated
"""

from __future__ import annotations

import pytest


class TestIterGraphs:
    """Tests for HypatiaAdapter.iter_graphs() method."""

    @pytest.fixture
    def small_adapter(self):
        """Create a small adapter for fast testing."""
        from satnet.network.hypatia_adapter import HypatiaAdapter

        adapter = HypatiaAdapter(
            num_planes=3,
            sats_per_plane=4,
            inclination_deg=53.0,
            altitude_km=550.0,
        )
        adapter.generate_tles()
        adapter.calculate_isls(duration_minutes=2, step_seconds=60)
        return adapter

    def test_iter_graphs_yields_tuples(self, small_adapter) -> None:
        """iter_graphs yields (step, graph) tuples."""
        results = list(small_adapter.iter_graphs())
        assert len(results) > 0

        for step, G in results:
            assert isinstance(step, int)
            assert hasattr(G, "number_of_nodes")
            assert hasattr(G, "number_of_edges")

    def test_iter_graphs_all_steps(self, small_adapter) -> None:
        """iter_graphs with no args yields all steps."""
        results = list(small_adapter.iter_graphs())
        # 2 minutes at 60s intervals = 3 steps (0, 1, 2)
        assert len(results) == 3
        steps = [step for step, _ in results]
        assert steps == [0, 1, 2]

    def test_iter_graphs_start_step(self, small_adapter) -> None:
        """iter_graphs respects start_step parameter."""
        results = list(small_adapter.iter_graphs(start_step=1))
        steps = [step for step, _ in results]
        assert steps == [1, 2]

    def test_iter_graphs_end_step(self, small_adapter) -> None:
        """iter_graphs respects end_step parameter (exclusive)."""
        results = list(small_adapter.iter_graphs(end_step=2))
        steps = [step for step, _ in results]
        assert steps == [0, 1]

    def test_iter_graphs_range(self, small_adapter) -> None:
        """iter_graphs respects both start and end parameters."""
        results = list(small_adapter.iter_graphs(start_step=1, end_step=2))
        steps = [step for step, _ in results]
        assert steps == [1]

    def test_iter_graphs_clamps_range(self, small_adapter) -> None:
        """iter_graphs clamps out-of-range parameters."""
        # Start before 0
        results = list(small_adapter.iter_graphs(start_step=-5, end_step=2))
        steps = [step for step, _ in results]
        assert steps == [0, 1]

        # End beyond available
        results = list(small_adapter.iter_graphs(start_step=1, end_step=100))
        steps = [step for step, _ in results]
        assert steps == [1, 2]

    def test_iter_graphs_empty_range(self, small_adapter) -> None:
        """iter_graphs with invalid range yields nothing."""
        results = list(small_adapter.iter_graphs(start_step=5, end_step=3))
        assert results == []

    def test_iter_graphs_raises_without_isls(self) -> None:
        """iter_graphs raises ValueError if ISLs not calculated."""
        from satnet.network.hypatia_adapter import HypatiaAdapter

        adapter = HypatiaAdapter(num_planes=2, sats_per_plane=3)
        # Don't call calculate_isls

        with pytest.raises(ValueError, match="ISL data not available"):
            list(adapter.iter_graphs())

    def test_iter_graphs_graph_has_correct_nodes(self, small_adapter) -> None:
        """Graphs from iter_graphs have correct node count."""
        expected_nodes = 3 * 4  # num_planes * sats_per_plane

        for step, G in small_adapter.iter_graphs():
            assert G.number_of_nodes() == expected_nodes

    def test_iter_graphs_reuses_cache(self, small_adapter) -> None:
        """iter_graphs reuses cached ISL data (doesn't recompute)."""
        # Get graphs twice - should be identical
        results1 = [(step, G.number_of_edges()) for step, G in small_adapter.iter_graphs()]
        results2 = [(step, G.number_of_edges()) for step, G in small_adapter.iter_graphs()]

        assert results1 == results2


class TestAdapterProperties:
    """Tests for HypatiaAdapter convenience properties."""

    @pytest.fixture
    def configured_adapter(self):
        """Create an adapter with ISLs calculated."""
        from satnet.network.hypatia_adapter import HypatiaAdapter

        adapter = HypatiaAdapter(
            num_planes=2,
            sats_per_plane=3,
        )
        adapter.generate_tles()
        adapter.calculate_isls(duration_minutes=5, step_seconds=60)
        return adapter

    def test_num_steps_property(self, configured_adapter) -> None:
        """num_steps returns correct count."""
        # 5 minutes at 60s = 6 steps (0, 1, 2, 3, 4, 5)
        assert configured_adapter.num_steps == 6

    def test_num_steps_before_isls(self) -> None:
        """num_steps returns 0 before ISLs calculated."""
        from satnet.network.hypatia_adapter import HypatiaAdapter

        adapter = HypatiaAdapter(num_planes=2, sats_per_plane=3)
        assert adapter.num_steps == 0

    def test_step_seconds_property(self, configured_adapter) -> None:
        """step_seconds returns configured value."""
        assert configured_adapter.step_seconds == 60

    def test_duration_seconds_property(self, configured_adapter) -> None:
        """duration_seconds returns configured value."""
        assert configured_adapter.duration_seconds == 5 * 60  # 5 minutes
