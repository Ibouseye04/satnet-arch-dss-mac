from __future__ import annotations

import networkx as nx

from satnet.simulation.tier1_rollout import Tier1RolloutConfig, run_tier1_rollout


class FakeAdapter:
    init_kwargs: dict[str, object] = {}
    calculate_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs) -> None:
        FakeAdapter.init_kwargs = kwargs
        self.graph = nx.Graph()
        self.graph.add_nodes_from([0, 1])
        self.graph.add_edge(0, 1)

    def generate_tles(self) -> None:
        return None

    def calculate_isls(self, **kwargs) -> None:
        FakeAdapter.calculate_kwargs = kwargs

    def get_graph_at_step(self, step: int) -> nx.Graph:
        return self.graph.copy()

    def iter_graphs(self):
        yield 0, self.graph.copy()


def test_rollout_config_defaults_to_grid_fixed() -> None:
    cfg = Tier1RolloutConfig(num_planes=2, sats_per_plane=3)

    assert cfg.isl_policy == "grid_fixed"
    assert cfg.adjacent_search_k == 1
    assert cfg.max_inter_plane_links_per_sat == 1


def test_rollout_passes_opt_in_isl_policy_to_adapter(monkeypatch) -> None:
    from satnet.network import hypatia_adapter

    monkeypatch.setattr(hypatia_adapter, "HypatiaAdapter", FakeAdapter)
    cfg = Tier1RolloutConfig(
        num_planes=1,
        sats_per_plane=2,
        duration_minutes=0,
        step_seconds=60,
        isl_policy="grid_adaptive",
        adjacent_search_k=3,
        max_inter_plane_links_per_sat=2,
    )

    run_tier1_rollout(cfg)

    assert FakeAdapter.calculate_kwargs["isl_policy"] == "grid_adaptive"
    assert FakeAdapter.calculate_kwargs["adjacent_search_k"] == 3
    assert FakeAdapter.calculate_kwargs["max_inter_plane_links_per_sat"] == 2
