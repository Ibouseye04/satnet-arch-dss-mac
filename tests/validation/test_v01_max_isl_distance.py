from __future__ import annotations

from pathlib import Path

from satnet.network.hypatia_adapter import (
    HypatiaAdapter,
    ISLComputationStats,
    SatellitePosition,
    WalkerDeltaConfig,
    _compute_grid_plus_isls,
)


class _CountingBudget:
    def __init__(self) -> None:
        self.distances: list[float] = []

    def evaluate_link(self, distance_km: float):
        self.distances.append(distance_km)
        return "optical", -20.0, 25.0, True


def _run_single_step(
    output_dir: Path,
    max_distance_km: float,
) -> tuple[dict[tuple[int, int], float], ISLComputationStats]:
    with HypatiaAdapter(
        num_planes=3,
        sats_per_plane=10,
        inclination_deg=53.0,
        altitude_km=550.0,
        phasing_factor=1,
        output_dir=output_dir,
    ) as adapter:
        _, stats = adapter.calculate_isls(
            duration_minutes=0,
            step_seconds=60,
            max_isl_distance_km=max_distance_km,
            isl_policy="grid_fixed",
        )
        graph = adapter.get_graph_at_step(0)
        edges = {
            tuple(sorted((int(u), int(v)))): float(attributes["distance_km"])
            for u, v, attributes in graph.edges(data=True)
        }
        return edges, stats


def _assert_fixed_policy_counter_partition(stats: ISLComputationStats) -> None:
    assert stats.total_candidate_links == (
        stats.links_accepted
        + stats.links_rejected_distance
        + stats.links_rejected_los
        + stats.links_rejected_budget
    )


def test_v01_maximum_isl_distance_is_hard_and_inclusive(tmp_path: Path) -> None:
    permissive, permissive_stats = _run_single_step(
        tmp_path / "permissive",
        10_000.0,
    )
    assert permissive

    probe_distance_km = min(permissive.values())
    restrictive, restrictive_stats = _run_single_step(tmp_path / "restrictive", 1.0)
    exact_boundary, exact_stats = _run_single_step(
        tmp_path / "exact",
        probe_distance_km,
    )
    below_boundary, below_stats = _run_single_step(
        tmp_path / "below",
        probe_distance_km - 1e-6,
    )

    assert restrictive == {}
    assert exact_boundary
    assert set(exact_boundary).issubset(permissive)
    assert max(exact_boundary.values()) <= probe_distance_km
    assert probe_distance_km in exact_boundary.values()
    assert below_boundary == {}
    assert restrictive_stats.links_rejected_distance > 0
    assert exact_stats.links_rejected_distance > 0
    assert below_stats.links_rejected_distance > 0
    for stats in (
        permissive_stats,
        restrictive_stats,
        exact_stats,
        below_stats,
    ):
        _assert_fixed_policy_counter_partition(stats)


def test_v01_distance_precedes_budget_and_accepts_exact_boundary() -> None:
    config = WalkerDeltaConfig(num_planes=2, sats_per_plane=2)
    positions = [
        SatellitePosition(0, 7000.0, 0.0, 0.0),
        SatellitePosition(1, 7000.0, 10.0, 0.0),
        SatellitePosition(2, 7000.0, 100.0, 0.0),
        SatellitePosition(3, 7000.0, 110.0, 0.0),
    ]
    exact_budget = _CountingBudget()
    exact_links, exact_stats = _compute_grid_plus_isls(
        config,
        positions,
        exact_budget,
        max_isl_distance_km=10.0,
    )

    assert exact_links
    assert exact_budget.distances
    assert all(link.distance_km <= 10.0 for link in exact_links)
    assert exact_stats.links_rejected_distance > 0
    _assert_fixed_policy_counter_partition(exact_stats)

    below_budget = _CountingBudget()
    below_links, below_stats = _compute_grid_plus_isls(
        config,
        positions,
        below_budget,
        max_isl_distance_km=10.0 - 1e-6,
    )

    assert below_links == []
    assert below_budget.distances == []
    assert below_stats.links_rejected_distance == below_stats.total_candidate_links
    assert below_stats.links_rejected_los == 0
    assert below_stats.links_rejected_budget == 0
    _assert_fixed_policy_counter_partition(below_stats)
