from __future__ import annotations

from pathlib import Path
import os

import networkx as nx
import pytest

from satnet.dss.tgnn_inference import (
    DSSCheckpointError,
    FROZEN_TGNN_CHECKPOINT_SHA256,
    graph_to_pyg_data,
    load_frozen_tgnn,
    predict_sequence,
    resolve_checkpoint_path,
    sequence_to_pyg,
    sha256_file,
)


def graph_sequence():
    result = []
    for timestep in range(11):
        graph = nx.Graph()
        graph.add_node(0, plane=0, sat_in_plane=0)
        graph.add_node(1, plane=0, sat_in_plane=1)
        graph.add_edge(
            0,
            1,
            distance_km=1000.0 + timestep,
            margin_db=10.0,
            link_type="intra_plane",
            link_mode="optical",
        )
        result.append(graph)
    return result


def test_feature_contract_is_exactly_11_by_3_and_4() -> None:
    sequence = sequence_to_pyg(graph_sequence(), num_planes=4, sats_per_plane=5)
    assert len(sequence) == 11
    assert all(tuple(data.x.shape)[1:] == (3,) for data in sequence)
    assert all(tuple(data.edge_attr.shape)[1:] == (4,) for data in sequence)
    assert all(data.edge_weight.equal(data.edge_attr[:, 0]) for data in sequence)
    assert all(float(data.edge_attr[0, 3]) == 1.0 for data in sequence)


def test_prediction_is_not_clipped() -> None:
    import torch

    class OutOfRangeModel:
        def eval(self):
            return self

        def __call__(self, data_list):
            return torch.tensor([[1.25]], dtype=torch.float32)

    prediction = predict_sequence(
        OutOfRangeModel(), sequence_to_pyg(graph_sequence(), num_planes=4, sats_per_plane=5)
    )
    assert prediction == 1.25


def test_checkpoint_mismatch_fails_closed(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not a checkpoint")
    with pytest.raises(DSSCheckpointError, match="SHA-256 mismatch"):
        resolve_checkpoint_path(wrong)


def test_missing_checkpoint_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SATNET_DSS_TGNN_CHECKPOINT", raising=False)
    with pytest.raises(DSSCheckpointError, match="required"):
        resolve_checkpoint_path()


@pytest.mark.skipif(
    not os.environ.get("SATNET_DSS_TGNN_CHECKPOINT")
    or not Path(os.environ["SATNET_DSS_TGNN_CHECKPOINT"]).is_file(),
    reason="Frozen operational checkpoint is supplied by deployment environment",
)
def test_dss_adapter_parity_with_existing_satellite_gnn_forward() -> None:
    import torch
    from satnet.models.gnn_model import SatelliteGNN

    model, checkpoint_hash = load_frozen_tgnn()
    assert checkpoint_hash == FROZEN_TGNN_CHECKPOINT_SHA256
    dss_sequence = sequence_to_pyg(graph_sequence(), num_planes=4, sats_per_plane=5)
    dss_prediction = predict_sequence(model, dss_sequence)

    existing_model = SatelliteGNN(
        node_features=3, hidden_channels=64, out_channels=1, task_type="regression", cheb_k=2
    )
    checkpoint = torch.load(
        os.environ["SATNET_DSS_TGNN_CHECKPOINT"], map_location="cpu", weights_only=False
    )
    existing_model.load_state_dict(checkpoint["model_state_dict"])
    existing_model.eval()
    with torch.no_grad():
        existing_prediction = float(existing_model(dss_sequence).squeeze().item())
    assert dss_prediction == existing_prediction
