from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")

import satnet.models.autoresearch_rf as autoresearch_rf
from satnet.models.autoresearch_rf import RfExperimentConfig, run_rf_experiment


def _write_runs_csv(csv_path) -> None:
    rows = []
    for i in range(30):
        rows.append(
            {
                "run_id": i,
                "config_hash": f"cfg-{i:04d}",
                "num_planes": 4 + (i % 3),
                "sats_per_plane": 6 + (i % 2),
                "total_satellites": (4 + (i % 3)) * (6 + (i % 2)),
                "inclination_deg": 50.0 + (i % 5),
                "altitude_km": 500.0 + (i * 5),
                "node_failure_prob": 0.01 * (i % 4),
                "edge_failure_prob": 0.02 * (i % 3),
                "duration_minutes": 10,
                "step_seconds": 60,
                "partition_any": i % 2,
                "gcc_frac_min": 0.25 + (i * 0.01),
                "partition_fraction": 0.10 + ((i % 10) * 0.03),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def test_run_rf_experiment_writes_unique_artifacts_and_appends_ledger(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "tier1_design_runs.csv"
    experiments_root = tmp_path / "experiments" / "autoresearch"
    _write_runs_csv(dataset_path)

    def fake_train_rf_once(cfg: RfExperimentConfig):
        metrics = {
            "task_type": "regression",
            "target_name": cfg.target_name,
            "test_rmse": 0.123,
            "test_spearman_rho": 0.75,
            "test_r2": 0.66,
        }
        predictions = pd.DataFrame(
            [
                {
                    "config_hash": "cfg-0000",
                    "target_name": cfg.target_name,
                    "task_type": "regression",
                    "seed": cfg.seed,
                    "split": "test",
                    "sample_idx": 0,
                    "y_true": 0.4,
                    "y_pred": 0.5,
                }
            ]
        )
        return object(), metrics, predictions

    monkeypatch.setattr(autoresearch_rf, "_train_rf_once", fake_train_rf_once)

    config = RfExperimentConfig(
        dataset_path=str(dataset_path),
        target_name="gcc_frac_min",
        feature_mode="tier1_full",
        n_estimators=20,
        seed=7,
        save_model_artifact=False,
    )

    first = run_rf_experiment(config, experiments_root=experiments_root)
    second = run_rf_experiment(
        RfExperimentConfig(
            dataset_path=str(dataset_path),
            target_name="gcc_frac_min",
            feature_mode="design_only",
            n_estimators=30,
            max_depth=12,
            seed=7,
            save_model_artifact=False,
        ),
        experiments_root=experiments_root,
        parent_experiment_id=first["experiment_id"],
    )

    first_dir = experiments_root / first["experiment_id"]
    second_dir = experiments_root / second["experiment_id"]
    ledger_path = experiments_root / "results.jsonl"

    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert first["experiment_id"] != second["experiment_id"]
    assert first_dir.exists()
    assert second_dir.exists()
    assert first_dir != second_dir

    first_metrics = first_dir / "metrics.json"
    first_predictions = first_dir / "predictions.csv"
    second_metrics = second_dir / "metrics.json"
    second_predictions = second_dir / "predictions.csv"

    assert first_metrics.exists()
    assert first_predictions.exists()
    assert second_metrics.exists()
    assert second_predictions.exists()

    first_metrics_payload = json.loads(first_metrics.read_text())
    assert "test_rmse" in first_metrics_payload
    assert ledger_path.exists()

    ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    assert len(ledger_rows) == 2
    assert ledger_rows[0]["experiment_id"] == first["experiment_id"]
    assert ledger_rows[1]["experiment_id"] == second["experiment_id"]
    assert ledger_rows[1]["parent_experiment_id"] == first["experiment_id"]

    assert first_predictions.exists()
    assert second_predictions.exists()
    assert first_metrics.exists()
    assert second_metrics.exists()
