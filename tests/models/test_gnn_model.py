"""Tests for satnet.models.gnn_model module.

These tests require torch and torch_geometric to be installed.
Use pytest.importorskip to skip tests when ML deps are unavailable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def _paired_sequences():
    from torch_geometric.data import Data

    torch.manual_seed(20260714)
    x = torch.randn(8, 3)
    chain = torch.tensor(
        [list(range(7)) + list(range(1, 8)), list(range(1, 8)) + list(range(7))],
        dtype=torch.long,
    )
    star_edges = []
    for node in range(1, 8):
        star_edges.extend([(0, node), (node, 0)])
    star = torch.tensor(star_edges, dtype=torch.long).t().contiguous()
    return (
        [Data(x=x.clone(), edge_index=chain.clone()) for _ in range(3)],
        [Data(x=x.clone(), edge_index=star.clone()) for _ in range(3)],
    )


def _forward_with_hidden(model, sequence):
    from torch_geometric.nn import global_mean_pool

    h = None
    c = None
    model.eval()
    with torch.no_grad():
        for data in sequence:
            h, c = model.recurrent(data.x, data.edge_index, getattr(data, "edge_weight", None), h, c)
        batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        pooled = global_mean_pool(h, batch)
        logits = model.linear(pooled)
    return logits, h


class TestSatelliteGNN:
    """Tests for SatelliteGNN model architecture."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        from satnet.models.gnn_model import SatelliteGNN

        return SatelliteGNN(node_features=3, hidden_channels=16, out_channels=2)

    @pytest.fixture
    def dummy_sequence(self):
        """Create a 3-step sequence of graphs for testing."""
        from torch_geometric.data import Data

        sequence = []
        for _ in range(3):
            x = torch.randn(10, 3)  # 10 nodes, 3 features
            edge_index = torch.randint(0, 10, (2, 20))  # 20 edges
            sequence.append(Data(x=x, edge_index=edge_index))
        return sequence

    def test_default_cheb_k_is_two(self) -> None:
        """Default Chebyshev order is topology-aware."""
        from satnet.models.gnn_model import SatelliteGNN

        model = SatelliteGNN(node_features=3, hidden_channels=16, out_channels=2)

        assert model.cheb_k == 2

    def test_rejects_invalid_cheb_k(self) -> None:
        """Chebyshev order must be positive."""
        from satnet.models.gnn_model import SatelliteGNN

        with pytest.raises(ValueError, match="cheb_k"):
            SatelliteGNN(node_features=3, hidden_channels=16, out_channels=2, cheb_k=0)

    def test_forward_returns_correct_shape(self, model, dummy_sequence) -> None:
        """Forward pass returns logits with correct shape."""
        logits = model(dummy_sequence)

        assert logits.shape == (1, 2), "Should return (batch_size=1, num_classes=2)"

    def test_forward_raises_for_empty_sequence(self, model) -> None:
        """Forward pass raises ValueError for empty sequence."""
        with pytest.raises((ValueError, IndexError)):
            model([])

    def test_predict_returns_class_index(self, model, dummy_sequence) -> None:
        """predict() returns integer class index."""
        pred = model.predict(dummy_sequence)

        assert isinstance(pred, int)
        assert pred in [0, 1]

    def test_predict_proba_returns_probabilities(self, model, dummy_sequence) -> None:
        """predict_proba() returns valid probability distribution."""
        probs = model.predict_proba(dummy_sequence)

        assert probs.shape == (1, 2)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        # All probabilities should be in [0, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_proba_raises_for_regression(self, dummy_sequence) -> None:
        """Regression models should reject predict_proba()."""
        from satnet.models.gnn_model import SatelliteGNN

        model = SatelliteGNN(
            node_features=3,
            hidden_channels=16,
            out_channels=1,
            task_type="regression",
        )

        with pytest.raises(RuntimeError, match="only available when task_type='classification'"):
            model.predict_proba(dummy_sequence)

    def test_model_is_trainable(self, model, dummy_sequence) -> None:
        """Model can be trained with gradient descent."""
        import torch.nn.functional as F

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        label = torch.tensor([1])

        model.train()
        optimizer.zero_grad()
        logits = model(dummy_sequence)
        loss = F.cross_entropy(logits, label)

        # Loss should be computed successfully
        assert not torch.isnan(loss)

        loss.backward()
        optimizer.step()

        # Gradients should have been computed
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_deterministic_eval_mode(self, model, dummy_sequence) -> None:
        """Same input produces same output in eval mode."""
        torch.manual_seed(42)
        model.eval()

        with torch.no_grad():
            logits1 = model(dummy_sequence)
            logits2 = model(dummy_sequence)

        assert torch.allclose(logits1, logits2)

    def test_k1_is_edge_insensitive_for_recurrent_sequence(self) -> None:
        """Installed GCLSTM semantics: K=1 ignores edge_index."""
        from satnet.models.gnn_model import SatelliteGNN

        sequence_a, sequence_b = _paired_sequences()
        torch.manual_seed(123)
        model = SatelliteGNN(node_features=3, hidden_channels=8, out_channels=2, cheb_k=1)
        out_a, hidden_a = _forward_with_hidden(model, sequence_a)
        out_b, hidden_b = _forward_with_hidden(model, sequence_b)

        assert torch.equal(hidden_a, hidden_b)
        assert torch.equal(out_a, out_b)

    def test_k2_is_edge_sensitive_for_recurrent_sequence(self) -> None:
        """Production default K=2 responds to graph topology."""
        from satnet.models.gnn_model import SatelliteGNN

        sequence_a, sequence_b = _paired_sequences()
        torch.manual_seed(123)
        model = SatelliteGNN(node_features=3, hidden_channels=8, out_channels=2, cheb_k=2)
        out_a, hidden_a = _forward_with_hidden(model, sequence_a)
        out_b, hidden_b = _forward_with_hidden(model, sequence_b)

        assert torch.max(torch.abs(hidden_a - hidden_b)).item() > 1e-6
        assert torch.max(torch.abs(out_a - out_b)).item() > 1e-8

    def test_k2_graph_convolution_parameters_receive_gradients_and_update(self) -> None:
        """K=2 graph-convolution parameters participate in optimization."""
        from satnet.models.gnn_model import SatelliteGNN

        sequence_a, _ = _paired_sequences()
        torch.manual_seed(123)
        model = SatelliteGNN(node_features=3, hidden_channels=8, out_channels=2, cheb_k=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = {name: param.detach().clone() for name, param in model.named_parameters()}

        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(sequence_a), torch.tensor([1]))
        loss.backward()
        conv_params = [(name, param) for name, param in model.named_parameters() if ".conv_" in name]

        assert conv_params
        assert any(param.grad is not None and torch.norm(param.grad).item() > 0.0 for _, param in conv_params)

        optimizer.step()

        assert any(
            torch.max(torch.abs(param.detach() - before[name])).item() > 0.0
            for name, param in conv_params
        )

    def test_hidden_state_propagates_across_timesteps_and_resets_between_runs(self) -> None:
        """Forward uses all timesteps and resets sequence state per run."""
        from satnet.models.gnn_model import SatelliteGNN

        sequence_a, _ = _paired_sequences()
        torch.manual_seed(123)
        model = SatelliteGNN(node_features=3, hidden_channels=8, out_channels=2, cheb_k=2)
        model.eval()

        with torch.no_grad():
            full_out = model(sequence_a)
            final_only_out = model([sequence_a[-1]])
            repeat_out = model(sequence_a)

        assert not torch.allclose(full_out, final_only_out)
        assert torch.allclose(full_out, repeat_out)


class TestModelSerialization:
    """Tests for model save/load."""

    def test_save_and_load_state_dict(self, tmp_path) -> None:
        """Model state dict can be saved and loaded."""
        from satnet.models.gnn_model import SatelliteGNN

        # Create and save model
        model1 = SatelliteGNN(node_features=3, hidden_channels=16, out_channels=2)
        model_path = tmp_path / "model.pt"
        torch.save(model1.state_dict(), model_path)

        # Load into new model
        model2 = SatelliteGNN(node_features=3, hidden_channels=16, out_channels=2)
        model2.load_state_dict(torch.load(model_path, weights_only=True))

        # Verify parameters match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
