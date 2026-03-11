"""Tests for scripts/train_gnn_model.py."""

from __future__ import annotations

from argparse import Namespace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
Data = pytest.importorskip("torch_geometric.data").Data


class _DummyRegressionDataset:
    def __getitem__(self, idx: int):
        label = float(idx) / 10.0
        return [
            Data(
                x=torch.zeros((3, 3), dtype=torch.float),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                y=torch.tensor([label], dtype=torch.float),
            )
        ]


class _DummyRegressionModel:
    task_type = "regression"

    def eval(self):
        return self

    def __call__(self, data_sequence):
        true_val = float(data_sequence[0].y.item())
        return torch.tensor([true_val + 0.05], dtype=torch.float)


def test_regression_evaluate_returns_expected_metrics() -> None:
    import scripts.train_gnn_model as train_gnn

    dataset = _DummyRegressionDataset()
    model = _DummyRegressionModel()
    criterion = nn.SmoothL1Loss()

    avg_loss, accuracy, metrics = train_gnn.evaluate(
        model=model,
        test_indices=[0, 1, 2],
        dataset=dataset,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert isinstance(avg_loss, float)
    assert accuracy == 0.0
    assert {
        "loss",
        "mae",
        "rmse",
        "r2",
        "spearman",
        "kendall",
        "n_samples",
    }.issubset(metrics.keys())
    assert "precision" not in metrics


def test_prediction_export_schema_contains_stable_fields(tmp_path, monkeypatch) -> None:
    import scripts.train_gnn_model as train_gnn

    class FakeDataset:
        def __init__(
            self,
            root: str,
            use_cache: bool,
            write_cache: bool,
            cache_dir: str | None,
            target_name: str,
        ) -> None:
            self._resolved_cache_dir = cache_dir or "cache"
            self.target_name = target_name
            self._rows = [
                {"config_hash": f"cfg-{i}", "run_id": i}
                for i in range(4)
            ]

        def __len__(self) -> int:
            return 4

        def get_label_distribution(self):
            return 2, 2

        def __getitem__(self, idx: int):
            return [
                Data(
                    x=torch.zeros((3, 3), dtype=torch.float),
                    edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                    y=torch.tensor([float(idx % 2)], dtype=torch.float),
                    time_step=torch.tensor([0], dtype=torch.long),
                )
            ]

        def get_run_config(self, idx: int):
            return self._rows[idx]

    class FakeModel(nn.Module):
        def __init__(
            self,
            node_features: int,
            hidden_channels: int,
            out_channels: int,
            task_type: str,
        ) -> None:
            super().__init__()
            self.task_type = task_type
            self.layer = nn.Linear(1, 1)

        def forward(self, data_list):
            _ = data_list
            return torch.tensor([[0.1, 0.9]], dtype=torch.float)

    def fake_train_epoch(model, train_indices, dataset, optimizer, criterion, device):  # noqa: ANN001
        _ = (model, train_indices, dataset, optimizer, criterion, device)
        return 0.1, 1.0

    def fake_evaluate(model, test_indices, dataset, criterion, device):  # noqa: ANN001
        _ = (model, test_indices, dataset, criterion, device)
        return 0.1, 1.0, {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "true_positives": 1,
            "false_positives": 0,
            "true_negatives": 1,
            "false_negatives": 0,
        }

    output_model = tmp_path / "models" / "gnn.pt"
    args = Namespace(
        data_dir=str(tmp_path),
        epochs=1,
        lr=0.01,
        hidden_dim=16,
        test_split=0.5,
        seed=123,
        output_model=str(output_model),
        device="cpu",
        use_cache=False,
        write_cache=False,
        cache_dir=str(tmp_path / "cache"),
        target_name="partition_any",
        experiment_log=str(tmp_path / "experiments" / "gnn_log.jsonl"),
    )

    monkeypatch.setattr(train_gnn, "parse_args", lambda: args)
    monkeypatch.setattr(train_gnn, "SatNetTemporalDataset", FakeDataset)
    monkeypatch.setattr(train_gnn, "SatelliteGNN", FakeModel)
    monkeypatch.setattr(train_gnn, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(train_gnn, "evaluate", fake_evaluate)

    train_gnn.main()

    preds_path = output_model.parent / f"{output_model.stem}_predictions.csv"
    df = pd.read_csv(preds_path)
    required_columns = {
        "config_hash",
        "target_name",
        "task_type",
        "seed",
        "split",
        "sample_idx",
        "y_true",
        "y_pred",
    }
    assert required_columns.issubset(df.columns)


@pytest.mark.parametrize("bad_config_hash", [None, "", "   "])
def test_prediction_export_rejects_invalid_config_hash(
    tmp_path,
    monkeypatch,
    bad_config_hash,
) -> None:
    import scripts.train_gnn_model as train_gnn

    class FakeDataset:
        def __init__(
            self,
            root: str,
            use_cache: bool,
            write_cache: bool,
            cache_dir: str | None,
            target_name: str,
        ) -> None:
            self._resolved_cache_dir = cache_dir or "cache"
            self.target_name = target_name
            self._rows = [
                {"config_hash": "cfg-0", "run_id": 0},
                {"config_hash": bad_config_hash, "run_id": 1},
                {"config_hash": "cfg-2", "run_id": 2},
                {"config_hash": "cfg-3", "run_id": 3},
            ]

        def __len__(self) -> int:
            return 4

        def get_label_distribution(self):
            return 2, 2

        def __getitem__(self, idx: int):
            return [
                Data(
                    x=torch.zeros((3, 3), dtype=torch.float),
                    edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                    y=torch.tensor([float(idx % 2)], dtype=torch.float),
                    time_step=torch.tensor([0], dtype=torch.long),
                )
            ]

        def get_run_config(self, idx: int):
            return self._rows[idx]

    class FakeModel(nn.Module):
        def __init__(
            self,
            node_features: int,
            hidden_channels: int,
            out_channels: int,
            task_type: str,
        ) -> None:
            super().__init__()
            self.task_type = task_type
            self.layer = nn.Linear(1, 1)

        def forward(self, data_list):
            _ = data_list
            return torch.tensor([[0.1, 0.9]], dtype=torch.float)

    output_model = tmp_path / "models" / "gnn.pt"
    args = Namespace(
        data_dir=str(tmp_path),
        epochs=1,
        lr=0.01,
        hidden_dim=16,
        test_split=0.5,
        seed=123,
        output_model=str(output_model),
        device="cpu",
        use_cache=False,
        write_cache=False,
        cache_dir=str(tmp_path / "cache"),
        target_name="partition_any",
        experiment_log=str(tmp_path / "experiments" / "gnn_log.jsonl"),
    )

    monkeypatch.setattr(train_gnn, "parse_args", lambda: args)
    monkeypatch.setattr(train_gnn, "SatNetTemporalDataset", FakeDataset)
    monkeypatch.setattr(train_gnn, "SatelliteGNN", FakeModel)

    with pytest.raises(ValueError, match="stable ranking joins"):
        train_gnn.main()
