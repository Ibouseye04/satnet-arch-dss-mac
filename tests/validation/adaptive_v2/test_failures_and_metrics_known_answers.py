"""Permanent Adaptive-v2 satellite-failure and graph-metric known answers."""

from __future__ import annotations

import networkx as nx
import pytest

from satnet.metrics.labels import (
    compute_gcc_frac,
    compute_gcc_size,
    compute_num_components,
    compute_partitioned,
)
from satnet.simulation.tier1_rollout import (
    FAILURE_MODEL_PERSISTENT_TEMPORAL_UNION_EDGES_V1,
    Tier1RolloutConfig,
    run_tier1_rollout,
)


# ---------------------------------------------------------------------------
# Canonical graph metrics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    (
        "graph",
        "expected_edges",
        "expected_avg_degree",
        "expected_components",
        "expected_gcc_size",
        "expected_gcc_frac",
        "expected_global_efficiency",
    ),
    [
        (
            nx.complete_graph(4),
            6,
            3.0,
            1,
            4,
            1.0,
            1.0,
        ),
        (
            nx.path_graph(4),
            3,
            1.5,
            1,
            4,
            1.0,
            13.0 / 18.0,
        ),
        (
            nx.Graph([(0, 1), (1, 2)]),
            2,
            1.0,
            2,
            3,
            0.75,
            5.0 / 12.0,
        ),
    ],
)
def test_canonical_graph_metric_known_answers(
    graph: nx.Graph,
    expected_edges: int,
    expected_avg_degree: float,
    expected_components: int,
    expected_gcc_size: int,
    expected_gcc_frac: float,
    expected_global_efficiency: float,
) -> None:
    """Lock K4, P4, and P3+isolate independently known graph values."""
    # Explicitly add isolated node 3 to the third controlled graph.
    if set(graph.nodes()) == {0, 1, 2}:
        graph.add_node(3)

    average_degree = (
        sum(dict(graph.degree()).values()) / graph.number_of_nodes()
    )

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == expected_edges
    assert average_degree == pytest.approx(expected_avg_degree, abs=1e-12)

    assert compute_num_components(graph) == expected_components
    assert compute_gcc_size(graph) == expected_gcc_size
    assert compute_gcc_frac(graph) == pytest.approx(
        expected_gcc_frac,
        abs=1e-12,
    )

    assert nx.global_efficiency(graph) == pytest.approx(
        expected_global_efficiency,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("gcc_fraction", "expected_partitioned"),
    [
        (0.79, 1),
        (0.80, 0),
        (0.81, 0),
    ],
)
def test_partition_threshold_semantics_known_answers(
    gcc_fraction: float,
    expected_partitioned: int,
) -> None:
    """SATNET partition status means threshold breach: gcc < threshold."""
    assert compute_partitioned(gcc_fraction, 0.80) == expected_partitioned


# ---------------------------------------------------------------------------
# Controlled rollout adapter
# ---------------------------------------------------------------------------

def _install_fake_adapter(
    monkeypatch: pytest.MonkeyPatch,
    graph_steps: list[tuple[int, nx.Graph]],
) -> None:
    """Replace orbital generation only; retain production failure/metric logic."""
    class FakeHypatiaAdapter:
        def __init__(self, **_kwargs) -> None:
            pass

        def generate_tles(self):
            return None

        def calculate_isls(self, **_kwargs) -> None:
            pass

        def iter_graphs(self):
            for timestep, graph in graph_steps:
                yield timestep, graph.copy()

    monkeypatch.setattr(
        "satnet.network.hypatia_adapter.HypatiaAdapter",
        FakeHypatiaAdapter,
    )


def _four_satellite_config(
    *,
    node_failure_prob: float,
    edge_failure_prob: float,
    seed: int,
    gcc_threshold: float = 0.80,
) -> Tier1RolloutConfig:
    return Tier1RolloutConfig(
        num_planes=1,
        sats_per_plane=4,
        inclination_deg=53.0,
        altitude_km=550.0,
        phasing_factor=1,
        duration_minutes=0,
        step_seconds=60,
        node_failure_prob=node_failure_prob,
        edge_failure_prob=edge_failure_prob,
        failure_model=FAILURE_MODEL_PERSISTENT_TEMPORAL_UNION_EDGES_V1,
        gcc_threshold=gcc_threshold,
        seed=seed,
    )


def test_persistent_temporal_union_includes_late_appearing_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edge (2,3), absent at t0 but present at t1, belongs to failure universe."""
    graph_t0 = nx.Graph()
    graph_t0.add_nodes_from(range(4))
    graph_t0.add_edges_from(((0, 1), (1, 2)))

    graph_t1 = nx.Graph()
    graph_t1.add_nodes_from(range(4))
    graph_t1.add_edges_from(((0, 1), (1, 2), (2, 3)))

    _install_fake_adapter(
        monkeypatch,
        [(0, graph_t0), (1, graph_t1)],
    )

    steps, summary, failures = run_tier1_rollout(
        _four_satellite_config(
            node_failure_prob=0.0,
            edge_failure_prob=1.0,
            seed=123,
        )
    )

    assert failures.failed_nodes == set()
    assert failures.failed_edges == {
        (0, 1),
        (1, 2),
        (2, 3),
    }

    # Late edge (2,3) must be persistently removed when it appears at t1.
    assert steps[0].num_edges == 0
    assert steps[1].num_edges == 0
    assert summary.num_failed_edges == 3
    assert (
        summary.failure_model
        == FAILURE_MODEL_PERSISTENT_TEMPORAL_UNION_EDGES_V1
    )


def test_persistent_node_failure_uses_original_gcc_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed 42 with p=.10 fails only node 1 in the controlled four-node path."""
    graph = nx.path_graph(4)

    _install_fake_adapter(
        monkeypatch,
        [(0, graph)],
    )

    steps, summary, failures = run_tier1_rollout(
        _four_satellite_config(
            node_failure_prob=0.10,
            edge_failure_prob=0.0,
            seed=42,
            gcc_threshold=0.60,
        )
    )

    assert failures.failed_nodes == {1}
    assert failures.failed_edges == set()

    step = steps[0]

    assert step.num_nodes == 3
    assert step.num_edges == 1
    assert step.num_components == 2
    assert step.gcc_size == 2

    # Authoritative dissertation metric:
    # GCC / original constellation = 2 / 4.
    assert step.gcc_frac_original == pytest.approx(0.5, abs=1e-12)
    assert step.gcc_frac == pytest.approx(0.5, abs=1e-12)

    # Diagnostic surviving-node denominator:
    # GCC / surviving nodes = 2 / 3.
    assert step.gcc_frac_surviving == pytest.approx(
        2.0 / 3.0,
        abs=1e-12,
    )

    # 0.5 < frozen controlled threshold 0.60.
    assert step.partitioned == 1

    assert summary.num_failed_nodes == 1
    assert summary.gcc_frac_min_original == pytest.approx(
        0.5,
        abs=1e-12,
    )
    assert summary.partition_any == 1
