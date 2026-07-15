from __future__ import annotations

import pytest


def _sequence():
    torch = pytest.importorskip("torch")
    Data = pytest.importorskip("torch_geometric.data").Data
    return [
        Data(
            x=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.5, 1.0]], dtype=torch.float),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.ones((2, 4), dtype=torch.float),
            y=torch.tensor([1.0], dtype=torch.float),
            time_step=torch.tensor([0], dtype=torch.long),
            run_id=torch.tensor([7], dtype=torch.long),
        )
    ]


def test_full_mode_preserves_x_and_edge_index() -> None:
    torch = pytest.importorskip("torch")
    from satnet.models.gnn_ablation import apply_tgnn_input_mode

    source = _sequence()
    transformed = apply_tgnn_input_mode(source, "full")

    assert torch.equal(transformed[0].x, source[0].x)
    assert torch.equal(transformed[0].edge_index, source[0].edge_index)
    assert transformed[0] is not source[0]


def test_topology_only_preserves_shape_and_edges_with_constant_x() -> None:
    torch = pytest.importorskip("torch")
    from satnet.models.gnn_ablation import apply_tgnn_input_mode

    source = _sequence()
    transformed = apply_tgnn_input_mode(source, "topology_only")

    assert transformed[0].x.shape == source[0].x.shape
    assert torch.equal(transformed[0].x, torch.ones_like(source[0].x))
    assert torch.equal(transformed[0].edge_index, source[0].edge_index)


def test_node_state_only_preserves_x_and_removes_inter_node_edges() -> None:
    torch = pytest.importorskip("torch")
    from satnet.models.gnn_ablation import apply_tgnn_input_mode

    source = _sequence()
    transformed = apply_tgnn_input_mode(source, "node_state_only")

    assert torch.equal(transformed[0].x, source[0].x)
    assert transformed[0].edge_index.shape == (2, 2)
    assert torch.equal(transformed[0].edge_index[0], transformed[0].edge_index[1])
    assert not torch.any(transformed[0].edge_index[0] != transformed[0].edge_index[1])


def test_transformations_do_not_mutate_source_and_preserve_y_metadata() -> None:
    torch = pytest.importorskip("torch")
    from satnet.models.gnn_ablation import apply_tgnn_input_mode

    source = _sequence()
    original_x = source[0].x.clone()
    original_edges = source[0].edge_index.clone()
    transformed = apply_tgnn_input_mode(source, "node_state_only")

    assert torch.equal(source[0].x, original_x)
    assert torch.equal(source[0].edge_index, original_edges)
    assert torch.equal(transformed[0].y, source[0].y)
    assert torch.equal(transformed[0].time_step, source[0].time_step)
    assert torch.equal(transformed[0].run_id, source[0].run_id)


def test_parameter_count_identical_across_modes() -> None:
    pytest.importorskip("torch")
    from satnet.models.gnn_ablation import TGNN_INPUT_MODE_REGISTRY, tgnn_input_mode_metadata
    from satnet.models.gnn_model import SatelliteGNN

    counts = []
    for mode in TGNN_INPUT_MODE_REGISTRY:
        metadata = tgnn_input_mode_metadata(mode=mode, baseline_feature_dim=3)
        model = SatelliteGNN(
            node_features=metadata["transformed_feature_dim"],
            hidden_channels=4,
            out_channels=2,
            task_type="classification",
            cheb_k=2,
        )
        assert model.cheb_k == 2
        counts.append(sum(p.numel() for p in model.parameters()))
    assert len(set(counts)) == 1


def test_unknown_input_mode_fails_clearly() -> None:
    from satnet.models.gnn_ablation import apply_tgnn_input_mode

    with pytest.raises(ValueError, match="Unknown TGNN input mode"):
        apply_tgnn_input_mode(_sequence(), "bad")
