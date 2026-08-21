from __future__ import annotations

from collections import Counter

import pytest

from satnet.network.hypatia_adapter import (
    SatellitePosition,
    WalkerDeltaConfig,
    _compute_grid_plus_isls,
)


class AlwaysViableBudget:
    def evaluate_link(self, distance_km: float):
        return "optical", -20.0, 25.0, True


def _positions_for_adaptive_selection() -> list[SatellitePosition]:
    return [
        SatellitePosition(0, 7000.0, 0.0, 0.0),
        SatellitePosition(1, 7000.0, 10.0, 0.0),
        SatellitePosition(2, 7000.0, -10.0, 0.0),
        SatellitePosition(3, -7000.0, 0.0, 0.0),
        SatellitePosition(4, 7000.0, 100.0, 0.0),
        SatellitePosition(5, 7000.0, -100.0, 0.0),
    ]


def _edge_set(links) -> set[tuple[int, int]]:
    return {tuple(sorted((link.sat_id_1, link.sat_id_2))) for link in links}


def test_grid_fixed_default_matches_explicit_policy() -> None:
    config = WalkerDeltaConfig(num_planes=2, sats_per_plane=3)
    positions = _positions_for_adaptive_selection()
    budget = AlwaysViableBudget()

    default_links, default_stats = _compute_grid_plus_isls(config, positions, budget)
    fixed_links, fixed_stats = _compute_grid_plus_isls(
        config,
        positions,
        budget,
        isl_policy="grid_fixed",
    )

    assert _edge_set(default_links) == _edge_set(fixed_links)
    assert default_stats.total_candidate_links == fixed_stats.total_candidate_links
    assert default_stats.links_rejected_los == fixed_stats.links_rejected_los
    assert default_stats.links_accepted == fixed_stats.links_accepted


def test_grid_adaptive_selects_highest_ranked_viable_candidates_globally() -> None:
    config = WalkerDeltaConfig(num_planes=2, sats_per_plane=3)
    positions = _positions_for_adaptive_selection()

    links, stats = _compute_grid_plus_isls(
        config,
        positions,
        AlwaysViableBudget(),
        isl_policy="grid_adaptive",
        adjacent_search_k=1,
        max_inter_plane_links_per_sat=2,
        collect_adaptive_examples=3,
    )
    edges = _edge_set(links)

    assert (0, 3) not in edges
    assert (0, 4) in edges
    assert stats.accepted_inter_plane_links > 0
    assert stats.adaptive_selection_examples
    first_example = stats.adaptive_selection_examples[0]
    assert first_example["original_adjacent_candidate"] == {
        "sat_id": 3,
        "plane": 1,
        "satellite": 0,
        "candidate_offset": 0,
    }
    assert first_example["selected"][0]["sat_id"] == 4


def test_grid_adaptive_enforces_total_incident_inter_plane_capacity() -> None:
    config = WalkerDeltaConfig(num_planes=2, sats_per_plane=3)
    positions = _positions_for_adaptive_selection()

    links, _ = _compute_grid_plus_isls(
        config,
        positions,
        AlwaysViableBudget(),
        isl_policy="grid_adaptive",
        adjacent_search_k=1,
        max_inter_plane_links_per_sat=1,
    )
    inter_plane_degree: Counter[int] = Counter()
    total_degree: Counter[int] = Counter()
    for link in links:
        total_degree[link.sat_id_1] += 1
        total_degree[link.sat_id_2] += 1
        if link.link_type != "intra_plane":
            inter_plane_degree[link.sat_id_1] += 1
            inter_plane_degree[link.sat_id_2] += 1

    assert inter_plane_degree
    assert max(inter_plane_degree.values()) <= 1
    assert max(total_degree.values()) > 1


def test_grid_adaptive_can_select_two_inter_plane_links_per_sat() -> None:
    config = WalkerDeltaConfig(num_planes=2, sats_per_plane=3)
    positions = _positions_for_adaptive_selection()

    links, _ = _compute_grid_plus_isls(
        config,
        positions,
        AlwaysViableBudget(),
        isl_policy="grid_adaptive",
        adjacent_search_k=1,
        max_inter_plane_links_per_sat=2,
    )
    edges = _edge_set(links)

    assert (0, 4) in edges
    assert (0, 5) in edges


def test_grid_adaptive_rejects_invalid_policy_parameters() -> None:
    config = WalkerDeltaConfig(num_planes=2, sats_per_plane=3)
    positions = _positions_for_adaptive_selection()
    budget = AlwaysViableBudget()

    with pytest.raises(ValueError, match="isl_policy"):
        _compute_grid_plus_isls(config, positions, budget, isl_policy="bad")
    with pytest.raises(ValueError, match="adjacent_search_k"):
        _compute_grid_plus_isls(
            config,
            positions,
            budget,
            isl_policy="grid_adaptive",
            adjacent_search_k=-1,
        )
    with pytest.raises(ValueError, match="max_inter_plane_links_per_sat"):
        _compute_grid_plus_isls(
            config,
            positions,
            budget,
            isl_policy="grid_adaptive",
            max_inter_plane_links_per_sat=3,
        )
