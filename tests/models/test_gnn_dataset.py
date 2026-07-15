"""Tests for satnet.models.gnn_dataset module.

These tests require torch and torch_geometric to be installed.
Use pytest.importorskip to skip when ML deps are unavailable.
"""

from __future__ import annotations

import json

import pytest


class TestNetworkxToPygData:
    """Tests for _networkx_to_pyg_data node ID mapping."""

    def test_node_mapping_with_noncontiguous_ids(self) -> None:
        """Regression test for B.5: non-contiguous node IDs after failure removal.

        Bug: gnn_dataset.py:312-320 used node_id directly as tensor index.
        When node failures create gaps (e.g., nodes {0,1,2,5,6}), accessing
        x[node_id] with node_id=6 on a tensor of size 5 raises IndexError.

        Evidence (before fix):
            x = torch.zeros((num_nodes, 3))  # num_nodes = 5
            for node_id in G.nodes():        # node_ids = [0, 1, 2, 5, 6]
                x[node_id, 0] = ...          # IndexError when node_id = 5 or 6

        Fix: Create node_mapping to remap to contiguous indices [0, num_nodes).
        """
        torch = pytest.importorskip("torch")
        nx = pytest.importorskip("networkx")
        pytest.importorskip("torch_geometric")

        from satnet.models.gnn_dataset import SatNetTemporalDataset

        # Create a graph with non-contiguous node IDs (simulating node failures)
        # Original constellation had nodes 0-9, but nodes 3,4,7,8,9 failed
        # Remaining nodes: {0, 1, 2, 5, 6}
        G = nx.Graph()
        remaining_nodes = [0, 1, 2, 5, 6]  # Non-contiguous!
        for n in remaining_nodes:
            G.add_node(n, plane=n // 3, sat_in_plane=n % 3)

        # Add some edges between remaining nodes
        G.add_edge(0, 1, distance_km=1000.0, margin_db=10.0, link_type="intra_plane")
        G.add_edge(1, 2, distance_km=1000.0, margin_db=10.0, link_type="intra_plane")
        G.add_edge(5, 6, distance_km=1000.0, margin_db=10.0, link_type="intra_plane")

        # Create dataset instance (just to access the method)
        # We need to call the protected method directly for unit testing
        dataset = SatNetTemporalDataset.__new__(SatNetTemporalDataset)

        # This would have raised IndexError before the fix
        # because node_id=6 > num_nodes=5
        data = dataset._networkx_to_pyg_data(
            G=G,
            label=0,
            run_id=0,
            time_step=0,
            num_planes=4,
            sats_per_plane=3,
        )

        # Verify the result
        assert data.x.shape[0] == 5, "Should have 5 nodes"
        assert data.x.shape[1] == 3, "Should have 3 features"
        assert data.edge_index.shape[0] == 2, "edge_index should be 2xE"
        assert data.edge_index.shape[1] == 6, "Should have 6 edges (3 * 2 bidirectional)"

        # Verify all edge indices are in valid range [0, num_nodes)
        assert data.edge_index.max().item() < 5, "All edge indices should be < num_nodes"
        assert data.edge_index.min().item() >= 0, "All edge indices should be >= 0"

    def test_node_mapping_preserves_features(self) -> None:
        """Verify node features are correctly assigned after mapping."""
        torch = pytest.importorskip("torch")
        nx = pytest.importorskip("networkx")
        pytest.importorskip("torch_geometric")

        from satnet.models.gnn_dataset import SatNetTemporalDataset

        G = nx.Graph()
        # Add nodes with explicit plane/sat_in_plane attributes
        G.add_node(10, plane=2, sat_in_plane=1)  # High node ID
        G.add_node(0, plane=0, sat_in_plane=0)

        dataset = SatNetTemporalDataset.__new__(SatNetTemporalDataset)
        data = dataset._networkx_to_pyg_data(
            G=G, label=1, run_id=0, time_step=0, num_planes=3, sats_per_plane=4
        )

        # Node 0 should map to index 0, node 10 should map to index 1
        # Features are [plane_normalized, sat_normalized, 1.0]
        assert data.x.shape == (2, 3)
        # Node 0: plane=0/(3-1)=0, sat=0/(4-1)=0
        assert data.x[0, 0].item() == pytest.approx(0.0)
        assert data.x[0, 1].item() == pytest.approx(0.0)
        # Node 10 (mapped to idx 1): plane=2/(3-1)=1.0, sat=1/(4-1)=0.333
        assert data.x[1, 0].item() == pytest.approx(1.0)
        assert data.x[1, 1].item() == pytest.approx(1 / 3, rel=0.01)


class TestSatNetTemporalDataset:
    """Tests for SatNetTemporalDataset class initialization and data loading."""

    def test_homogeneous_temporal_metadata_summary_uses_row_values(self) -> None:
        pd = pytest.importorskip("pandas")

        from satnet.models.gnn_dataset import format_dataset_temporal_metadata_summary

        df = pd.DataFrame(
            [
                {"duration_minutes": 5, "step_seconds": 60, "num_steps": 6},
                {"duration_minutes": 5, "step_seconds": 60, "num_steps": 6},
            ]
        )

        summary = format_dataset_temporal_metadata_summary(
            df,
            fallback_duration_minutes=10,
            fallback_step_seconds=60,
        )

        assert summary == "duration=5 min, step=60 s, num_steps=6"
        assert "duration=10 min" not in summary
        assert "fallback" not in summary

    def test_heterogeneous_temporal_metadata_summary_reports_multiple_values(self) -> None:
        pd = pytest.importorskip("pandas")

        from satnet.models.gnn_dataset import format_dataset_temporal_metadata_summary

        df = pd.DataFrame(
            [
                {"duration_minutes": 5, "step_seconds": 60, "num_steps": 6},
                {"duration_minutes": 10, "step_seconds": 60, "num_steps": 11},
            ]
        )

        summary = format_dataset_temporal_metadata_summary(
            df,
            fallback_duration_minutes=10,
            fallback_step_seconds=60,
        )

        assert summary == "multiple durations=[5, 10], steps=[60], num_steps=[6, 11]"

    def test_legacy_temporal_metadata_summary_reports_fallback_values(self) -> None:
        pd = pytest.importorskip("pandas")

        from satnet.models.gnn_dataset import format_dataset_temporal_metadata_summary

        df = pd.DataFrame([{"num_planes": 2, "sats_per_plane": 3}])

        summary = format_dataset_temporal_metadata_summary(
            df,
            fallback_duration_minutes=10,
            fallback_step_seconds=60,
        )

        assert summary == "duration=10 min fallback, step=60 s fallback, num_steps=unknown fallback"

    def test_raises_for_missing_csv(self, tmp_path) -> None:
        """Should raise FileNotFoundError for missing CSV."""
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from satnet.models.gnn_dataset import SatNetTemporalDataset

        with pytest.raises(FileNotFoundError):
            SatNetTemporalDataset(root=str(tmp_path), csv_file="nonexistent.csv")


class TestGnnDatasetCacheContract:
    def _write_minimal_csv(
        self,
        csv_path,
        *,
        include_isl_policy: bool = False,
        failure_model: str | None = None,
        failed_edges_json: str = "[]",
    ) -> None:
        pd = pytest.importorskip("pandas")
        row = {
            "num_planes": 2,
            "sats_per_plane": 3,
            "inclination_deg": 53.0,
            "altitude_km": 550.0,
            "phasing_factor": 1,
            "duration_minutes": 1,
            "step_seconds": 60,
            "failed_nodes_json": "[]",
            "failed_edges_json": failed_edges_json,
            "seed": 42,
            "epoch_iso": "2025-01-01T00:00:00",
            "config_hash": "cfg-001",
            "partition_any": 1,
            "gcc_frac_min": 0.25,
            "max_partition_streak_seconds": 120,
            "max_partition_streak_fraction": 0.5,
        }
        if failure_model is not None:
            row["failure_model"] = failure_model
        if include_isl_policy:
            row.update(
                {
                    "isl_policy": "grid_adaptive",
                    "adjacent_search_k": 1,
                    "max_inter_plane_links_per_sat": 1,
                }
            )
        df = pd.DataFrame([row])
        df.to_csv(csv_path, index=False)

    def test_accepts_new_partition_persistence_target_when_column_exists(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from satnet.models import gnn_dataset as gnn_dataset_module

        csv_path = tmp_path / "tier1_design_runs.csv"
        self._write_minimal_csv(csv_path)
        self._install_fake_adapter(monkeypatch, gnn_dataset_module)

        dataset = gnn_dataset_module.SatNetTemporalDataset(
            root=str(tmp_path),
            target_name="max_partition_streak_seconds",
        )

        assert len(dataset) == 1
        assert dataset[0][0].y.item() == 120

    def test_missing_new_partition_persistence_target_fails_clearly(self, tmp_path) -> None:
        pd = pytest.importorskip("pandas")
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from satnet.models.gnn_dataset import SatNetTemporalDataset

        pd.DataFrame(
            [
                {
                    "num_planes": 2,
                    "sats_per_plane": 3,
                    "inclination_deg": 53.0,
                    "altitude_km": 550.0,
                    "partition_any": 1,
                }
            ]
        ).to_csv(tmp_path / "tier1_design_runs.csv", index=False)

        with pytest.raises(ValueError, match="max_partition_streak_seconds"):
            SatNetTemporalDataset(
                root=str(tmp_path),
                target_name="max_partition_streak_seconds",
            )

    def _install_fake_adapter(self, monkeypatch, module):
        nx = pytest.importorskip("networkx")

        class FakeAdapter:
            calculate_kwargs: dict[str, object] = {}

            def __init__(self, **kwargs):  # noqa: ANN003
                self.kwargs = kwargs

            def calculate_isls(self, **kwargs) -> None:  # noqa: ANN003
                FakeAdapter.calculate_kwargs = kwargs
                self.duration_minutes = kwargs["duration_minutes"]
                self.step_seconds = kwargs["step_seconds"]

            def iter_graphs(self):
                for t in (0, 60):
                    graph = nx.Graph()
                    for node_id in range(6):
                        graph.add_node(
                            node_id,
                            plane=node_id // 3,
                            sat_in_plane=node_id % 3,
                        )
                    graph.add_edge(
                        0,
                        1,
                        distance_km=1000.0,
                        margin_db=10.0,
                        link_type="intra_plane",
                        link_mode="optical",
                    )
                    graph.add_edge(
                        3,
                        4,
                        distance_km=1100.0,
                        margin_db=9.0,
                        link_type="inter_plane",
                        link_mode="optical",
                    )
                    yield t, graph

        monkeypatch.setattr(module, "HypatiaAdapter", FakeAdapter)
        return FakeAdapter

    def test_reconstruction_passes_isl_policy_from_csv(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from satnet.models import gnn_dataset as gnn_dataset_module

        csv_path = tmp_path / "tier1_design_runs.csv"
        self._write_minimal_csv(csv_path, include_isl_policy=True)
        fake_adapter = self._install_fake_adapter(monkeypatch, gnn_dataset_module)

        dataset = gnn_dataset_module.SatNetTemporalDataset(root=str(tmp_path))
        sample = dataset[0]

        assert fake_adapter.calculate_kwargs["isl_policy"] == "grid_adaptive"
        assert fake_adapter.calculate_kwargs["adjacent_search_k"] == 1
        assert fake_adapter.calculate_kwargs["max_inter_plane_links_per_sat"] == 1
        assert sample[0].isl_policy == "grid_adaptive"
        assert sample[0].adjacent_search_k == 1
        assert sample[0].max_inter_plane_links_per_sat == 1

    def test_reconstruction_defaults_legacy_csv_to_grid_fixed(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from satnet.models import gnn_dataset as gnn_dataset_module

        csv_path = tmp_path / "tier1_design_runs.csv"
        self._write_minimal_csv(csv_path)
        fake_adapter = self._install_fake_adapter(monkeypatch, gnn_dataset_module)

        dataset = gnn_dataset_module.SatNetTemporalDataset(root=str(tmp_path))
        sample = dataset[0]

        assert fake_adapter.calculate_kwargs["isl_policy"] == "grid_fixed"
        assert fake_adapter.calculate_kwargs["adjacent_search_k"] == 1
        assert fake_adapter.calculate_kwargs["max_inter_plane_links_per_sat"] == 1
        assert sample[0].isl_policy == "grid_fixed"
        assert sample[0].failure_model == "persistent_t0_edges_v1"

    def test_reconstruction_preserves_new_failure_model_metadata(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from satnet.models import gnn_dataset as gnn_dataset_module

        csv_path = tmp_path / "tier1_design_runs.csv"
        self._write_minimal_csv(
            csv_path,
            failure_model="persistent_temporal_union_edges_v1",
        )
        self._install_fake_adapter(monkeypatch, gnn_dataset_module)

        dataset = gnn_dataset_module.SatNetTemporalDataset(root=str(tmp_path))
        sample = dataset[0]

        assert sample[0].failure_model == "persistent_temporal_union_edges_v1"

    def test_reconstruction_uses_stored_later_only_failed_edge_without_resampling(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")
        nx = pytest.importorskip("networkx")

        from satnet.models import gnn_dataset as gnn_dataset_module

        class FakeAdapter:
            def __init__(self, **kwargs):
                return None

            def calculate_isls(self, **kwargs) -> None:
                return None

            def iter_graphs(self):
                edge_sets = [[(0, 1)], [(0, 1), (2, 3)]]
                for t, edges in enumerate(edge_sets):
                    graph = nx.Graph()
                    for node_id in range(4):
                        graph.add_node(
                            node_id,
                            plane=node_id // 2,
                            sat_in_plane=node_id % 2,
                        )
                    graph.add_edges_from(edges)
                    yield t, graph

        monkeypatch.setattr(gnn_dataset_module, "HypatiaAdapter", FakeAdapter)
        csv_path = tmp_path / "tier1_design_runs.csv"
        self._write_minimal_csv(
            csv_path,
            failure_model="persistent_temporal_union_edges_v1",
            failed_edges_json="[[2, 3]]",
        )

        dataset = gnn_dataset_module.SatNetTemporalDataset(root=str(tmp_path))
        sample = dataset[0]

        assert sample[1].edge_index.shape[1] == 2
        assert sample[1].failure_model == "persistent_temporal_union_edges_v1"

    def test_generated_adaptive_dataset_load_preserves_policy_metadata(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from satnet.models import gnn_dataset as gnn_dataset_module
        from satnet.simulation.monte_carlo import (
            Tier1MonteCarloConfig,
            generate_tier1_temporal_dataset,
            write_tier1_dataset_csv,
        )

        cfg = Tier1MonteCarloConfig(
            num_runs=1,
            num_planes_range=(2, 2),
            sats_per_plane_range=(3, 3),
            duration_minutes=0,
            step_seconds=60,
            isl_policy="grid_adaptive",
            adjacent_search_k=1,
            max_inter_plane_links_per_sat=1,
            node_failure_prob_range=(0.0, 0.0),
            edge_failure_prob_range=(0.0, 0.0),
        )
        runs, steps = generate_tier1_temporal_dataset(cfg)
        write_tier1_dataset_csv(
            runs,
            steps,
            tmp_path / "tier1_design_runs.csv",
            tmp_path / "tier1_design_steps.csv",
        )
        fake_adapter = self._install_fake_adapter(monkeypatch, gnn_dataset_module)

        dataset = gnn_dataset_module.SatNetTemporalDataset(root=str(tmp_path))
        sample = dataset[0]

        assert fake_adapter.calculate_kwargs["isl_policy"] == "grid_adaptive"
        assert sample[0].isl_policy == "grid_adaptive"

    def test_cache_reuses_structure_but_recomputes_target(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from satnet.models import gnn_dataset as gnn_dataset_module

        csv_path = tmp_path / "tier1_design_runs.csv"
        cache_dir = tmp_path / "cache"
        self._write_minimal_csv(csv_path)
        self._install_fake_adapter(monkeypatch, gnn_dataset_module)

        writer_ds = gnn_dataset_module.SatNetTemporalDataset(
            root=str(tmp_path),
            target_name="partition_any",
            write_cache=True,
            cache_dir=str(cache_dir),
        )
        writer_sample = writer_ds[0]
        assert writer_sample[0].y.item() == pytest.approx(1.0)
        meta_path = next(cache_dir.glob("*.meta.json"))
        payload = json.loads(meta_path.read_text())
        assert payload["sample_cache_key"]
        assert payload["generator_provenance"] == (
            gnn_dataset_module.GRAPH_SEQUENCE_GENERATOR_PROVENANCE
        )
        assert payload["sample_cache_key"] != payload["generator_provenance"]
        assert payload["generator_config"]["failure_model"] == "persistent_t0_edges_v1"

        # Ensure second dataset instance hits cache and does not regenerate.
        class FailOnBuildAdapter:
            def __init__(self, **kwargs):  # noqa: ANN003
                raise AssertionError("Expected cache hit, not regeneration")

        monkeypatch.setattr(gnn_dataset_module, "HypatiaAdapter", FailOnBuildAdapter)
        reader_ds = gnn_dataset_module.SatNetTemporalDataset(
            root=str(tmp_path),
            target_name="gcc_frac_min",
            use_cache=True,
            cache_dir=str(cache_dir),
        )
        reader_sample = reader_ds[0]

        assert reader_sample[0].y.item() == pytest.approx(0.25)
        assert reader_sample[1].y.item() == pytest.approx(0.25)

    def test_cache_generator_provenance_mismatch_rejected(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from satnet.models import gnn_dataset as gnn_dataset_module

        csv_path = tmp_path / "tier1_design_runs.csv"
        cache_dir = tmp_path / "cache"
        self._write_minimal_csv(csv_path)
        self._install_fake_adapter(monkeypatch, gnn_dataset_module)

        writer_ds = gnn_dataset_module.SatNetTemporalDataset(
            root=str(tmp_path),
            target_name="partition_any",
            write_cache=True,
            cache_dir=str(cache_dir),
        )
        _ = writer_ds[0]

        meta_path = next(cache_dir.glob("*.meta.json"))
        payload = json.loads(meta_path.read_text())
        payload["generator_provenance"] = "broken-provenance"
        meta_path.write_text(json.dumps(payload))

        reader_ds = gnn_dataset_module.SatNetTemporalDataset(
            root=str(tmp_path),
            target_name="partition_any",
            use_cache=True,
            cache_dir=str(cache_dir),
        )

        with pytest.raises(ValueError, match="generator provenance mismatch"):
            _ = reader_ds[0]

    def test_cache_stale_schema_rejected(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from satnet.models import gnn_dataset as gnn_dataset_module

        csv_path = tmp_path / "tier1_design_runs.csv"
        cache_dir = tmp_path / "cache"
        self._write_minimal_csv(csv_path)
        self._install_fake_adapter(monkeypatch, gnn_dataset_module)

        writer_ds = gnn_dataset_module.SatNetTemporalDataset(
            root=str(tmp_path),
            target_name="partition_any",
            write_cache=True,
            cache_dir=str(cache_dir),
        )
        _ = writer_ds[0]

        meta_path = next(cache_dir.glob("*.meta.json"))
        payload = json.loads(meta_path.read_text())
        payload["cache_schema_version"] = 0
        meta_path.write_text(json.dumps(payload))

        reader_ds = gnn_dataset_module.SatNetTemporalDataset(
            root=str(tmp_path),
            target_name="partition_any",
            use_cache=True,
            cache_dir=str(cache_dir),
        )

        with pytest.raises(ValueError, match="Unsupported cache schema version"):
            _ = reader_ds[0]
