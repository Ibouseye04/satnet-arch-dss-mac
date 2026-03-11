from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

import satnet.models.autoresearch_rf as autoresearch_rf
from satnet.models.autoresearch_rf import (
    RfAutoresearchCommandError,
    RfExperimentConfig,
    load_incumbent_registry,
    run_agent_candidate,
    run_confirmatory_experiment,
    run_rf_experiment,
)


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


def _write_policy_json(policy_path: Path, dataset_path: Path) -> None:
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "satnet_rf_autoresearch_policy_v1",
                "mutable_fields": {
                    "target_name": ["gcc_frac_min", "partition_fraction"],
                    "feature_mode": ["tier1_full", "design_only"],
                    "n_estimators": [100, 300, 600],
                    "max_depth": [None, 12, 20],
                },
                "fixed_fields": {
                    "dataset_path": str(dataset_path),
                    "custom_feature_columns": [],
                    "seed": 42,
                    "experiment_type": "rf_autoresearch_v1",
                    "notes": "",
                    "save_model_artifact": False,
                },
            },
            indent=2,
        )
    )


def _write_candidate_json(
    candidate_path: Path,
    dataset_path: Path,
    *,
    target_name: str = "gcc_frac_min",
    feature_mode: str = "tier1_full",
    n_estimators: int = 300,
    max_depth: int | None = None,
    fidelity_tier: str = "screening",
    extra_fields: dict[str, object] | None = None,
) -> None:
    payload = {
        "dataset_path": str(dataset_path),
        "target_name": target_name,
        "feature_mode": feature_mode,
        "custom_feature_columns": [],
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "seed": 42,
        "fidelity_tier": fidelity_tier,
        "experiment_type": "rf_autoresearch_v1",
        "notes": "",
        "save_model_artifact": False,
    }
    if extra_fields:
        payload.update(extra_fields)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
    )


def test_run_rf_experiment_writes_structured_summary_and_incumbent_registry(
    tmp_path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "tier1_design_runs.csv"
    experiments_root = tmp_path / "experiments" / "autoresearch"
    _write_runs_csv(dataset_path)

    def fake_train_rf_once(cfg: RfExperimentConfig):
        if cfg.feature_mode == "tier1_full":
            rmse = 0.123
            spearman = 0.75
            r2 = 0.66
        else:
            rmse = 0.200
            spearman = 0.50
            r2 = 0.40
        metrics = {
            "task_type": "regression",
            "target_name": cfg.target_name,
            "test_rmse": rmse,
            "test_spearman_rho": spearman,
            "test_r2": r2,
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
    incumbent_path = experiments_root / "incumbent.json"
    last_run_path = experiments_root / "last_run.json"
    first_summary = first_dir / "summary.json"
    second_summary = second_dir / "summary.json"

    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert first["promotion_decision"] == "promoted"
    assert second["promotion_decision"] == "rejected"
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
    assert first_summary.exists()
    assert second_metrics.exists()
    assert second_predictions.exists()
    assert second_summary.exists()

    first_metrics_payload = json.loads(first_metrics.read_text())
    assert "test_rmse" in first_metrics_payload
    assert ledger_path.exists()
    assert incumbent_path.exists()
    assert last_run_path.exists()

    ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    assert len(ledger_rows) == 2
    assert ledger_rows[0]["experiment_id"] == first["experiment_id"]
    assert ledger_rows[1]["experiment_id"] == second["experiment_id"]
    assert ledger_rows[1]["parent_experiment_id"] == first["experiment_id"]

    first_summary_payload = json.loads(first_summary.read_text())
    second_summary_payload = json.loads(second_summary.read_text())
    last_run_payload = json.loads(last_run_path.read_text())
    incumbent_payload = load_incumbent_registry(incumbent_path)

    assert first_summary_payload["schema_version"] == "satnet_rf_autoresearch_summary_v1"
    assert first_summary_payload["experiment_id"] == first["experiment_id"]
    assert first_summary_payload["promotion_decision"] == "promoted"
    assert second_summary_payload["incumbent_comparison"]["incumbent_experiment_id"] == first["experiment_id"]
    assert last_run_payload["schema_version"] == "satnet_rf_autoresearch_last_run_v1"
    assert last_run_payload["experiment_id"] == second["experiment_id"]
    assert last_run_payload["status"] == "succeeded"
    assert last_run_payload["summary_path"] == str(second_summary)
    assert last_run_payload["artifact_paths"]["summary"] == str(second_summary)
    assert last_run_payload["error_type"] is None
    assert last_run_payload["error_message"] is None

    assert len(incumbent_payload["incumbents"]) == 1
    incumbent_entry = incumbent_payload["incumbents"][0]
    assert incumbent_entry["current_best_experiment_id"] == first["experiment_id"]
    assert incumbent_entry["target_name"] == "gcc_frac_min"
    assert incumbent_entry["fidelity_tier"] == "screening"
    assert incumbent_entry["artifact_paths"]["summary"] == str(first_summary)


def test_run_agent_candidate_validates_policy_and_updates_last_run(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "tier1_design_runs.csv"
    experiments_root = tmp_path / "experiments" / "autoresearch"
    candidate_path = tmp_path / "candidate.json"
    policy_path = tmp_path / "mutation_policy.json"
    _write_runs_csv(dataset_path)
    _write_policy_json(policy_path, dataset_path)
    _write_candidate_json(candidate_path, dataset_path)

    def fake_train_rf_once(cfg: RfExperimentConfig):
        metrics = {
            "task_type": "regression",
            "target_name": cfg.target_name,
            "test_rmse": 0.100,
            "test_spearman_rho": 0.80,
            "test_r2": 0.70,
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

    result = run_agent_candidate(
        candidate_path=candidate_path,
        policy_path=policy_path,
        experiments_root=experiments_root,
    )

    summary_path = Path(result["artifact_paths"]["summary"])
    last_run_path = experiments_root / "last_run.json"
    incumbent_path = experiments_root / "incumbent.json"
    summary_payload = json.loads(summary_path.read_text())
    last_run_payload = json.loads(last_run_path.read_text())
    incumbent_payload = load_incumbent_registry(incumbent_path)

    assert result["status"] == "succeeded"
    assert result["promotion_decision"] == "promoted"
    assert summary_payload["experiment_id"] == result["experiment_id"]
    assert summary_payload["fidelity_tier"] == "screening"
    assert last_run_payload["experiment_id"] == result["experiment_id"]
    assert last_run_payload["summary_path"] == str(summary_path)
    assert incumbent_payload["incumbents"][0]["current_best_experiment_id"] == result["experiment_id"]


def test_run_agent_candidate_rejects_illegal_mutation_surface(tmp_path) -> None:
    dataset_path = tmp_path / "tier1_design_runs.csv"
    candidate_path = tmp_path / "candidate.json"
    policy_path = tmp_path / "mutation_policy.json"
    _write_runs_csv(dataset_path)
    _write_policy_json(policy_path, dataset_path)
    _write_candidate_json(candidate_path, dataset_path, n_estimators=999)

    with pytest.raises(ValueError, match="outside the approved mutation surface"):
        run_agent_candidate(
            candidate_path=candidate_path,
            policy_path=policy_path,
            experiments_root=tmp_path / "experiments" / "autoresearch",
        )


def test_run_agent_candidate_rejects_unknown_candidate_fields(tmp_path) -> None:
    dataset_path = tmp_path / "tier1_design_runs.csv"
    candidate_path = tmp_path / "candidate.json"
    policy_path = tmp_path / "mutation_policy.json"
    _write_runs_csv(dataset_path)
    _write_policy_json(policy_path, dataset_path)
    _write_candidate_json(candidate_path, dataset_path, extra_fields={"min_samples_leaf": 2})

    with pytest.raises(ValueError, match="Unsupported config fields"):
        run_agent_candidate(
            candidate_path=candidate_path,
            policy_path=policy_path,
            experiments_root=tmp_path / "experiments" / "autoresearch",
        )


def test_validate_candidate_rejects_missing_required_fixed_policy_fields(tmp_path) -> None:
    dataset_path = tmp_path / "tier1_design_runs.csv"
    candidate_path = tmp_path / "candidate.json"
    policy_path = tmp_path / "mutation_policy.json"
    _write_runs_csv(dataset_path)
    _write_candidate_json(candidate_path, dataset_path)
    policy_payload = {
        "schema_version": "satnet_rf_autoresearch_policy_v1",
        "mutable_fields": {
            "target_name": ["gcc_frac_min", "partition_fraction"],
            "feature_mode": ["tier1_full", "design_only"],
            "n_estimators": [100, 300, 600],
            "max_depth": [None, 12, 20],
        },
        "fixed_fields": {
            "dataset_path": str(dataset_path),
            "custom_feature_columns": [],
            "seed": 42,
            "experiment_type": "rf_autoresearch_v1",
            "notes": "",
        },
    }
    policy_path.write_text(json.dumps(policy_payload, indent=2))

    with pytest.raises(ValueError, match="Mutation policy missing required fixed_fields"):
        run_agent_candidate(
            candidate_path=candidate_path,
            policy_path=policy_path,
            experiments_root=tmp_path / "experiments" / "autoresearch",
        )


def test_run_confirmatory_experiment_keeps_separate_incumbent_scope(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "tier1_design_runs.csv"
    experiments_root = tmp_path / "experiments" / "autoresearch"
    candidate_path = tmp_path / "candidate.json"
    policy_path = tmp_path / "mutation_policy.json"
    _write_runs_csv(dataset_path)
    _write_policy_json(policy_path, dataset_path)
    _write_candidate_json(candidate_path, dataset_path, target_name="partition_fraction")

    def fake_train_rf_once(cfg: RfExperimentConfig):
        metrics = {
            "task_type": "regression",
            "target_name": cfg.target_name,
            "test_rmse": 0.111 if cfg.fidelity_tier == "screening" else 0.109,
            "test_spearman_rho": 0.78,
            "test_r2": 0.68,
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

    screening = run_agent_candidate(
        candidate_path=candidate_path,
        policy_path=policy_path,
        experiments_root=experiments_root,
    )
    confirmatory = run_confirmatory_experiment(
        screening["experiment_id"],
        experiments_root=experiments_root,
        policy_path=policy_path,
    )

    incumbent_payload = load_incumbent_registry(experiments_root / "incumbent.json")
    scopes = {
        (entry["target_name"], entry["fidelity_tier"]): entry
        for entry in incumbent_payload["incumbents"]
    }

    assert confirmatory["status"] == "succeeded"
    assert confirmatory["fidelity_tier"] == "confirmatory"
    assert confirmatory["parent_experiment_id"] == screening["experiment_id"]
    assert ("partition_fraction", "screening") in scopes
    assert ("partition_fraction", "confirmatory") in scopes
    assert scopes[("partition_fraction", "screening")]["current_best_experiment_id"] == screening["experiment_id"]
    assert scopes[("partition_fraction", "confirmatory")]["current_best_experiment_id"] == confirmatory["experiment_id"]


def test_run_confirmatory_experiment_requires_promoted_source(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "tier1_design_runs.csv"
    experiments_root = tmp_path / "experiments" / "autoresearch"
    candidate_path = tmp_path / "candidate.json"
    policy_path = tmp_path / "mutation_policy.json"
    _write_runs_csv(dataset_path)
    _write_policy_json(policy_path, dataset_path)
    _write_candidate_json(candidate_path, dataset_path)

    def fake_train_rf_once(cfg: RfExperimentConfig):
        metrics = {
            "task_type": "regression",
            "target_name": cfg.target_name,
            "test_rmse": 0.100 if cfg.feature_mode == "tier1_full" else 0.300,
            "test_spearman_rho": 0.80,
            "test_r2": 0.70,
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

    screening = run_agent_candidate(
        candidate_path=candidate_path,
        policy_path=policy_path,
        experiments_root=experiments_root,
    )
    rejected_screening = run_rf_experiment(
        RfExperimentConfig(
            dataset_path=str(dataset_path),
            target_name="gcc_frac_min",
            feature_mode="design_only",
            n_estimators=100,
            seed=42,
            save_model_artifact=False,
        ),
        experiments_root=experiments_root,
    )

    with pytest.raises(ValueError, match="promoted screening source run"):
        run_confirmatory_experiment(
            source_experiment_id=rejected_screening["experiment_id"],
            experiments_root=experiments_root,
            policy_path=policy_path,
        )


def test_load_incumbent_registry_rejects_wrong_schema_version(tmp_path) -> None:
    incumbent_path = tmp_path / "incumbent.json"
    incumbent_path.write_text(
        json.dumps(
            {
                "schema_version": "stale_schema",
                "updated_at": None,
                "primary_metric_name": "test_rmse",
                "secondary_metric_name": "test_spearman_rho",
                "tertiary_metric_name": "test_r2",
                "incumbents": [],
            },
            indent=2,
        )
    )

    with pytest.raises(ValueError, match="Unexpected incumbent registry schema"):
        load_incumbent_registry(incumbent_path)


def test_run_rf_experiment_preserves_summary_pointer_on_late_persistence_failure(
    tmp_path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "tier1_design_runs.csv"
    experiments_root = tmp_path / "experiments" / "autoresearch"
    _write_runs_csv(dataset_path)

    def fake_train_rf_once(cfg: RfExperimentConfig):
        metrics = {
            "task_type": "regression",
            "target_name": cfg.target_name,
            "test_rmse": 0.100,
            "test_spearman_rho": 0.80,
            "test_r2": 0.70,
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

    def fake_save_incumbent_registry(*args, **kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(autoresearch_rf, "_train_rf_once", fake_train_rf_once)
    monkeypatch.setattr(autoresearch_rf, "save_incumbent_registry", fake_save_incumbent_registry)

    with pytest.raises(RfAutoresearchCommandError, match="Post-run persistence failed"):
        run_rf_experiment(
            RfExperimentConfig(
                dataset_path=str(dataset_path),
                target_name="gcc_frac_min",
                feature_mode="tier1_full",
                n_estimators=300,
                seed=42,
                save_model_artifact=False,
            ),
            experiments_root=experiments_root,
            command_mode="candidate",
        )

    last_run_payload = json.loads((experiments_root / "last_run.json").read_text())
    summary_path = Path(last_run_payload["summary_path"])

    assert last_run_payload["schema_version"] == "satnet_rf_autoresearch_last_run_v1"
    assert last_run_payload["status"] == "failed"
    assert last_run_payload["promotion_decision"] == "failed"
    assert last_run_payload["error_type"] == "OSError"
    assert summary_path.exists()
    assert last_run_payload["artifact_paths"]["summary"] == str(summary_path)


def test_run_rf_experiment_reports_predictions_artifact_path_on_export_failure(
    tmp_path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "tier1_design_runs.csv"
    experiments_root = tmp_path / "experiments" / "autoresearch"
    _write_runs_csv(dataset_path)

    class FailingPredictions:
        def to_csv(self, path, index=False) -> None:
            raise PermissionError("[WinError 5] Access is denied")

    def fake_train_rf_once(cfg: RfExperimentConfig):
        metrics = {
            "task_type": "regression",
            "target_name": cfg.target_name,
            "test_rmse": 0.100,
            "test_spearman_rho": 0.80,
            "test_r2": 0.70,
        }
        return object(), metrics, FailingPredictions()

    monkeypatch.setattr(autoresearch_rf, "_train_rf_once", fake_train_rf_once)

    result = run_rf_experiment(
        RfExperimentConfig(
            dataset_path=str(dataset_path),
            target_name="gcc_frac_min",
            feature_mode="tier1_full",
            n_estimators=300,
            seed=42,
            save_model_artifact=False,
        ),
        experiments_root=experiments_root,
        command_mode="candidate",
    )

    last_run_payload = json.loads((experiments_root / "last_run.json").read_text())
    predictions_path = (
        experiments_root / result["experiment_id"] / "predictions.csv"
    )

    assert result["status"] == "failed"
    assert result["promotion_decision"] == "failed"
    assert last_run_payload["status"] == "failed"
    assert last_run_payload["error_type"] == "PermissionError"
    assert str(predictions_path) in last_run_payload["error_message"]
    assert "predictions artifact" in last_run_payload["error_message"]
    assert last_run_payload["artifact_paths"]["predictions"] is None
