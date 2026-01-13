"""Tests for satnet.models.gnn_dataset module.

These tests require torch and torch_geometric to be installed.
Use pytest.importorskip to skip when ML deps are unavailable.
"""

from __future__ import annotations

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

    def test_raises_for_missing_csv(self, tmp_path) -> None:
        """Should raise FileNotFoundError for missing CSV."""
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from satnet.models.gnn_dataset import SatNetTemporalDataset

        with pytest.raises(FileNotFoundError):
            SatNetTemporalDataset(root=str(tmp_path), csv_file="nonexistent.csv")
