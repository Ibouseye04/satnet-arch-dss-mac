from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
import torch

from satnet.models.gnn_dataset import SatNetTemporalDataset
from satnet.network.hypatia_adapter import HypatiaAdapter
from satnet.simulation.monte_carlo import (
    Tier1MonteCarloConfig,
    generate_tier1_temporal_dataset,
    write_tier1_dataset_csv,
)
from satnet.simulation.tier1_rollout import DEFAULT_FAILURE_MODEL


def _apply_exported_failures(graph: nx.Graph, run_row) -> nx.Graph:
    effective = graph.copy()
    failed_nodes = {int(node) for node in json.loads(run_row.failed_nodes_json)}
    failed_edges = {
        (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1])))
        for edge in json.loads(run_row.failed_edges_json)
    }
    effective.remove_nodes_from(node for node in failed_nodes if effective.has_node(node))
    effective.remove_edges_from(edge for edge in failed_edges if effective.has_edge(*edge))
    return effective


def test_v04_nominal_export_replays_exactly_and_missing_failures_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_iter_graphs = HypatiaAdapter.iter_graphs
    captured_graphs: list[tuple[int, nx.Graph]] = []
    capture_enabled = True

    def capturing_iter_graphs(self, *args, **kwargs):
        for time_step, graph in original_iter_graphs(self, *args, **kwargs):
            if capture_enabled:
                captured_graphs.append((time_step, graph.copy()))
            yield time_step, graph

    monkeypatch.setattr(HypatiaAdapter, "iter_graphs", capturing_iter_graphs)

    monte_carlo_config = Tier1MonteCarloConfig(
        num_runs=1,
        num_planes_range=(2, 2),
        sats_per_plane_range=(3, 3),
        inclination_deg_range=(53.0, 53.0),
        altitude_km_range=(550.0, 550.0),
        duration_minutes=1,
        step_seconds=60,
        isl_policy="grid_fixed",
        adjacent_search_k=1,
        max_inter_plane_links_per_sat=1,
        gcc_threshold=0.8,
        node_failure_prob_range=(0.1, 0.1),
        edge_failure_prob_range=(0.0, 0.0),
        failure_model=DEFAULT_FAILURE_MODEL,
        seed=42,
        sample_constellation=False,
    )
    runs, steps = generate_tier1_temporal_dataset(monte_carlo_config)
    capture_enabled = False

    assert len(captured_graphs) == 2
    assert json.loads(runs[0].failed_nodes_json) == [1]

    data_dir = tmp_path / "canonical"
    runs_path = data_dir / "tier1_design_runs.csv"
    steps_path = data_dir / "tier1_design_steps.csv"
    write_tier1_dataset_csv(runs, steps, runs_path, steps_path)

    dataset = SatNetTemporalDataset(root=str(data_dir), target_name="partition_any")
    reconstructed = dataset.get(0)

    assert len(reconstructed) == len(captured_graphs) == len(steps)
    for (time_step, original_graph), data, step_row in zip(
        captured_graphs,
        reconstructed,
        steps,
        strict=True,
    ):
        effective_graph = _apply_exported_failures(original_graph, runs[0])
        expected = dataset._networkx_to_pyg_data(
            G=effective_graph,
            time_step=time_step,
            num_planes=runs[0].num_planes,
            sats_per_plane=runs[0].sats_per_plane,
        )

        assert set(effective_graph.nodes()) == set(original_graph.nodes()) - {1}
        assert torch.equal(data.x, expected.x)
        assert torch.equal(data.edge_index, expected.edge_index)
        assert torch.allclose(data.edge_attr, expected.edge_attr, rtol=0.0, atol=1e-7)
        assert int(data.time_step.item()) == time_step == step_row.t
        assert int(data.num_nodes) == effective_graph.number_of_nodes() == step_row.num_nodes
        assert data.edge_index.shape[1] // 2 == effective_graph.number_of_edges() == step_row.num_edges
        assert nx.number_connected_components(effective_graph) == step_row.num_components
        assert float(data.y.item()) == pytest.approx(float(runs[0].partition_any))

    incomplete_dir = tmp_path / "missing_failure_metadata"
    incomplete_dir.mkdir()
    incomplete_runs = pd.read_csv(runs_path).drop(
        columns=["failed_nodes_json", "failed_edges_json"]
    )
    incomplete_runs.to_csv(incomplete_dir / "tier1_design_runs.csv", index=False)

    with pytest.raises(
        ValueError,
        match="failed_edges_json.*failed_nodes_json|failed_nodes_json.*failed_edges_json",
    ):
        SatNetTemporalDataset(
            root=str(incomplete_dir),
            target_name="partition_any",
        )

    assert int(reconstructed[0].num_nodes) == 5
