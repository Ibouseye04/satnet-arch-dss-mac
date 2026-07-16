from __future__ import annotations

from collections import Counter
from pathlib import Path

from satnet.network.hypatia_adapter import HypatiaAdapter


def _adaptive_edges(
    output_dir: Path,
) -> tuple[set[tuple[int, int, str]], set[tuple[int, int, str]]]:
    with HypatiaAdapter(
        num_planes=3,
        sats_per_plane=6,
        inclination_deg=53.0,
        altitude_km=550.0,
        phasing_factor=1,
        output_dir=output_dir,
    ) as adapter:
        adapter.calculate_isls(
            duration_minutes=0,
            step_seconds=60,
            isl_policy="grid_adaptive",
            adjacent_search_k=1,
            max_inter_plane_links_per_sat=1,
        )
        graph = adapter.get_graph_at_step(0)
        all_edges = {
            (min(int(u), int(v)), max(int(u), int(v)), str(attributes["link_type"]))
            for u, v, attributes in graph.edges(data=True)
        }
        inter_plane_edges = {
            edge for edge in all_edges if edge[2] != "intra_plane"
        }
        return all_edges, inter_plane_edges


def test_v03_adaptive_graph_respects_total_incident_capacity_deterministically(
    tmp_path: Path,
) -> None:
    first_all, first_inter_plane = _adaptive_edges(tmp_path / "first")
    second_all, second_inter_plane = _adaptive_edges(tmp_path / "second")

    inter_plane_degree: Counter[int] = Counter()
    for u, v, _ in first_inter_plane:
        inter_plane_degree[u] += 1
        inter_plane_degree[v] += 1

    assert first_all == second_all
    assert first_inter_plane == second_inter_plane
    assert first_inter_plane
    assert {link_type for _, _, link_type in first_inter_plane} == {
        "inter_plane",
        "seam_link",
    }
    assert max(inter_plane_degree.values()) <= 1
    assert all(value <= 1 for value in inter_plane_degree.values())
