"""Tests for scripts/train_design_risk_model.py."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest


class _DummyExperimentLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "_DummyExperimentLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        return None

    def set(self, key: str, value) -> None:  # noqa: ANN001
        _ = (key, value)

    def set_metrics(self, metrics: dict) -> None:  # noqa: ANN401
        _ = metrics

    def start_timer(self, name: str) -> None:
        _ = name

    def stop_timer(self, name: str) -> float:
        _ = name
        return 0.0


def test_partition_any_uses_same_path_logic_for_implicit_and_explicit_default(
    tmp_path,
    monkeypatch,
) -> None:
    pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
    import scripts.train_design_risk_model as train_script

    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "tier1_design_runs.csv"
    csv_path.write_text("num_planes\n4\n")

    captured_csv_paths: list[Path] = []

    def fake_train_rf_model(csv_path: Path, target_name: str, cfg):  # noqa: ANN001
        captured_csv_paths.append(Path(csv_path))
        predictions = pd.DataFrame(
            [
                {
                    "config_hash": "cfg-1",
                    "target_name": target_name,
                    "task_type": "classification",
                    "seed": cfg.random_state,
                    "split": "train",
                    "sample_idx": 0,
                    "y_true": 1.0,
                    "y_pred": 1.0,
                }
            ]
        )
        return object(), {"accuracy": 1.0}, predictions

    monkeypatch.setattr(train_script, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(train_script, "train_rf_model", fake_train_rf_model)
    monkeypatch.setattr(train_script, "save_model", lambda model, path: None)
    monkeypatch.setattr(train_script, "ExperimentLogger", _DummyExperimentLogger)

    def run_once(data_path_arg: str | None) -> None:
        args = Namespace(
            data_path=data_path_arg,
            target_name="partition_any",
            n_estimators=5,
            seed=42,
            test_size=0.2,
            output_dir=str(project_root / "models"),
            experiment_log=str(project_root / "experiments" / "rf_log.jsonl"),
        )
        monkeypatch.setattr(train_script, "parse_args", lambda: args)
        train_script.main()

    run_once(None)
    run_once(str(csv_path))

    assert len(captured_csv_paths) == 2
    assert captured_csv_paths[0] == csv_path
    assert captured_csv_paths[1] == csv_path
