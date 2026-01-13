"""Tests for satnet.models.gnn_model module.

These tests require torch and torch_geometric to be installed.
Use pytest.importorskip to skip tests when ML deps are unavailable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


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
