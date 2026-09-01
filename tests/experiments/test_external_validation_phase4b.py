from __future__ import annotations

import ast
import csv
from pathlib import Path

import numpy as np
import pytest

from satnet.experiments.external_validation import phase4b


@pytest.fixture(scope="module")
def episodes() -> list[phase4b.ExternalEpisode]:
    return phase4b.load_external_episodes()


@pytest.fixture(scope="module")
def artifacts() -> list[phase4b.FrozenArtifact]:
    return phase4b.verify_model_freezes()


def test_frozen_external_bundle_and_inventory_hashes_match() -> None:
    result = phase4b.verify_external_bundle()
    assert result["bundle_sha256"] == "63b098ddc28a1d7ee45f38fc57b22793d876855b832a65c62760371935c577cc"
    assert result["inventory_sha256"] == "7440239bf9df99f59bad7483f072106bdce74954a32c7d02dc1b378d595f24c1"


def test_frozen_phase4_artifact_membership_and_identity(artifacts: list[phase4b.FrozenArtifact]) -> None:
    assert len(artifacts) == 20
    assert {(artifact.task, artifact.seed) for artifact in artifacts} == {(task, seed) for task in phase4b.TASKS for seed in phase4b.SEEDS}
    assert all(artifact.candidate_set_hash == phase4b.CANDIDATE_SET_HASH for artifact in artifacts)
    assert {artifact.config_id for artifact in artifacts if artifact.task == "rf_space_regression"} == {"rf_018"}
    assert {artifact.config_id for artifact in artifacts if artifact.task == "tgnn_space_regression"} == {"tgnn_012"}


def test_external_raw_rf_features_are_preserved_exactly(episodes: list[phase4b.ExternalEpisode]) -> None:
    path = phase4b.EXTERNAL_ROOT / "episodes/external_rf_dataset.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for episode, row in zip(episodes, rows):
        assert episode.rf_feature_text == tuple(row[name] for name in phase4b.RF_FEATURES)
        assert episode.edge_failure_probability_text == row["satellite_edge_failure_probability"]
        assert episode.edge_failure_probability == float(row["satellite_edge_failure_probability"])


def test_all_external_rf_episodes_are_ood_without_remapping(episodes: list[phase4b.ExternalEpisode], artifacts: list[phase4b.FrozenArtifact]) -> None:
    class Tree:
        feature = np.asarray([5, 2, -2], dtype=np.int64)
        threshold = np.asarray([0.24879451841115952, 0.1, -2.0], dtype=float)

    class Estimator:
        tree_ = Tree()

    result = phase4b.verify_rf_ood_domain(episodes, artifacts, model_loader=lambda _: type("Model", (), {"estimators_": [Estimator()]})())
    assert result["ood_episode_count"] == 300
    assert result["ood_all_episodes"] is True
    assert result["all_external_values_exceed_every_learned_threshold"] is True
    assert result["semantic_equivalence_to_synthetic_predictor"] is False


def test_tgnn_graph_input_parity_is_frozen(episodes: list[phase4b.ExternalEpisode]) -> None:
    phase4b.verify_tgnn_target_manifest(episodes)
    payload = phase4b.load_json(episodes[0].tgnn_sequence)
    assert tuple(payload["node_feature_order"]) == phase4b.NODE_FEATURES
    assert tuple(payload["edge_feature_order"]) == phase4b.EDGE_FEATURES
    assert len(payload["snapshots"]) == 11


def test_classification_is_single_class_and_descriptive_only(episodes: list[phase4b.ExternalEpisode]) -> None:
    assert sum(episode.classification_target == 0 for episode in episodes) == 0
    assert sum(episode.classification_target == 1 for episode in episodes) == 300
    metrics = phase4b._classification_metrics(np.ones(300, dtype=int), np.ones(300, dtype=int), np.full(300, 0.9))
    assert metrics["descriptive_only"] is True
    assert metrics["observed_class_balance"] == {"negative": 0, "positive": 300}
    assert "balanced_accuracy" in metrics["excluded_inferential_metrics"]
    assert "roc_auc" in metrics["excluded_inferential_metrics"]


def test_regression_metrics_include_required_residual_statistics() -> None:
    metrics = phase4b._regression_metrics(np.asarray([0.0, 0.5, 1.0]), np.asarray([0.1, 0.4, 0.8]))
    assert set(("mae", "rmse", "r2", "median_absolute_error", "maximum_absolute_error", "residual_mean", "residual_standard_deviation", "prediction_range")) <= set(metrics)
    assert metrics["prediction_range"] == [0.1, 0.8]


def _temporary_output_contract(root: Path) -> phase4b.OutputContract:
    return phase4b.OutputContract(
        tgnn_metrics=root / "tgnn_metrics.json",
        tgnn_predictions=root / "tgnn_predictions.jsonl",
        rf_metrics=root / "rf_metrics.json",
        rf_predictions=root / "rf_predictions.jsonl",
        classification=root / "classification.json",
        comparison=root / "comparison.json",
        summary=root / "summary.json",
        report=root / "report.md",
        inventory=root / "inventory.json",
        identity=root / "identity.json",
    )


def _stub_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        phase4b,
        "preflight",
        lambda: {
            "status": "PASS",
            "model_artifacts": [],
            "output_contract": {},
            "interpretation_contract": {},
        },
    )


def test_inference_requires_explicit_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase4b, "preflight", lambda: pytest.fail("preflight must not run before authorization"))
    with pytest.raises(phase4b.ExternalInferenceError, match="explicit authorization"):
        phase4b.run_inference()


def test_preflight_does_not_create_inference_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "inference"
    monkeypatch.setattr(phase4b, "INFERENCE_ROOT", root)
    monkeypatch.setattr(phase4b, "OUTPUTS", _temporary_output_contract(root))
    monkeypatch.setattr(phase4b, "verify_rf_ood_domain", lambda episodes, artifacts: {"ood_all_episodes": True})
    monkeypatch.setattr(phase4b, "verify_model_freezes", lambda: [])
    monkeypatch.setattr(phase4b, "verify_dataset_bundle", lambda: {"bundle_sha256": phase4b.DATASET_BUNDLE_SHA256})
    monkeypatch.setattr(phase4b, "verify_external_bundle", lambda: {"bundle_sha256": phase4b.EXTERNAL_BUNDLE_SHA256})
    monkeypatch.setattr(phase4b, "load_external_episodes", lambda: [object()] * 300)
    monkeypatch.setattr(phase4b, "verify_tgnn_target_manifest", lambda episodes: None)

    result = phase4b.preflight()

    assert result["status"] == "PASS"
    assert result["inference_root_absent"] is True
    assert not root.exists()


def test_authorized_inference_rejects_existing_inference_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "inference"
    root.mkdir()
    monkeypatch.setattr(phase4b, "INFERENCE_ROOT", root)
    monkeypatch.setattr(phase4b, "OUTPUTS", _temporary_output_contract(root))
    _stub_preflight(monkeypatch)

    with pytest.raises(phase4b.ExternalInferenceError, match="must be absent"):
        phase4b.run_inference(authorized=True)


def test_existing_result_files_cannot_be_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "inference"
    root.mkdir()
    result_file = root / "summary.json"
    result_file.write_text("original\n", encoding="utf-8")
    monkeypatch.setattr(phase4b, "INFERENCE_ROOT", root)
    monkeypatch.setattr(phase4b, "OUTPUTS", _temporary_output_contract(root))
    _stub_preflight(monkeypatch)

    with pytest.raises(phase4b.ExternalInferenceError):
        phase4b.run_inference(authorized=True)

    assert result_file.read_text(encoding="utf-8") == "original\n"


def test_first_authorized_inference_creates_root_after_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "inference"
    outputs = _temporary_output_contract(root)
    monkeypatch.setattr(phase4b, "INFERENCE_ROOT", root)
    monkeypatch.setattr(phase4b, "OUTPUTS", outputs)
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(phase4b, "load_external_episodes", lambda: [])
    monkeypatch.setattr(phase4b, "verify_model_freezes", lambda: [
        phase4b.FrozenArtifact(task, seed, "config", "RF" if task.startswith("rf_") else "TGNN", Path("unused"), "sha", Path("manifest"), "manifest-sha", "selection-sha", "candidate")
        for task in phase4b.TASKS
        for seed in phase4b.SEEDS
    ])
    monkeypatch.setattr(phase4b, "_predict_task", lambda task, seed, artifact, episodes: [])
    monkeypatch.setattr(phase4b, "_metrics_from_records", lambda task, records: {"descriptive_only": "classification" in task})
    monkeypatch.setattr(phase4b, "_five_seed_descriptive_summary", lambda metrics, tasks: {})
    monkeypatch.setattr(phase4b, "paired_bootstrap", lambda rf, tgnn: {"label": phase4b.COMPARISON_LABEL})
    monkeypatch.setattr(phase4b, "write_report", lambda summary, comparison: None)
    identity_root_seen: list[bool] = []
    monkeypatch.setattr(phase4b, "write_execution_identity", lambda result: identity_root_seen.append(root.exists()))

    result = phase4b.run_inference(authorized=True)

    assert result["inference_performed"] is True
    assert root.is_dir()
    assert identity_root_seen == [True]
    assert outputs.summary.exists()
    assert outputs.inventory.exists()


def test_inference_code_has_no_training_or_optimizer_calls() -> None:
    tree = ast.parse(Path(phase4b.__file__).read_text(encoding="utf-8"))
    forbidden = {"fit", "backward", "step", "fit_transform"}
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert not forbidden.intersection(calls)


def test_no_feature_clipping_rescaling_or_risk_bins_are_generated() -> None:
    source = Path(phase4b.__file__).read_text(encoding="utf-8").lower()
    assert "clip(" not in source
    assert "risk_bin" not in source
    assert "normalize(" not in source
    assert "rescale(" not in source


def test_superiority_language_is_rejected_and_report_labels_are_frozen() -> None:
    with pytest.raises(phase4b.ExternalInferenceError):
        phase4b.assert_no_superiority_language("RF is better than TGNN")
    phase4b.assert_no_superiority_language(phase4b.COMPARISON_LABEL)
    assert phase4b.TGNN_LABEL == "Real-input external generalization"
    assert phase4b.RF_LABEL == "Out-of-domain physical-viability proxy stress test"
    assert phase4b.COMPARISON_LABEL == "Descriptive external comparison under mismatched RF feature semantics"
    assert phase4b.PRIMARY_LABEL == "Authoritative held-out model comparison"


def test_historical_bootstrap_is_descriptive_stress_test_only() -> None:
    rf = [{"episode_id": index, "absolute_error": 0.2} for index in range(300)]
    tgnn = [{"episode_id": index, "absolute_error": 0.1} for index in range(300)]
    result = phase4b.paired_bootstrap(rf, tgnn)
    assert result["label"] == phase4b.COMPARISON_LABEL
    assert result["primary_seed"] == 42
    assert result["replicates"] == 2000
    assert "not evidence for choosing one model over the other" in result["interpretation"]


def test_preflight_contract_exposes_no_inference_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase4b, "verify_rf_ood_domain", lambda episodes, artifacts: {"ood_all_episodes": True})
    monkeypatch.setattr(phase4b, "verify_model_freezes", lambda: [])
    monkeypatch.setattr(phase4b, "verify_dataset_bundle", lambda: {"bundle_sha256": phase4b.DATASET_BUNDLE_SHA256})
    monkeypatch.setattr(phase4b, "verify_external_bundle", lambda: {"bundle_sha256": phase4b.EXTERNAL_BUNDLE_SHA256})
    monkeypatch.setattr(phase4b, "load_external_episodes", lambda: [object()] * 300)
    monkeypatch.setattr(phase4b, "verify_tgnn_target_manifest", lambda episodes: None)
    result = phase4b.preflight()
    assert result["status"] == "PASS"
    assert result["inference_performed"] is False
    assert result["training_performed"] is False
    assert result["threshold_tuning_performed"] is False


def test_tgnn_five_seed_regression_summary_is_descriptive() -> None:
    metrics = {
        f"tgnn_space_regression/seed_{seed}": {
            "mae": float(index),
            "rmse": float(index + 1),
            "r2": float(index) / 10.0,
            "median_absolute_error": 0.1,
            "maximum_absolute_error": 0.2,
            "residual_mean": 0.0,
            "residual_standard_deviation": 0.05,
        }
        for index, seed in enumerate(phase4b.SEEDS)
    }
    summary = phase4b._five_seed_descriptive_summary(metrics, ("tgnn_space_regression",))
    assert summary["tgnn_space_regression"]["mae"]["mean"] == 2.0
    assert summary["tgnn_space_regression"]["mae"]["standard_deviation"] == np.sqrt(2.0)


def test_report_contract_contains_labels_without_external_comparison_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = phase4b.OutputContract(report=tmp_path / "external_inference_report.md")
    monkeypatch.setattr(phase4b, "OUTPUTS", outputs)
    phase4b.write_report({"inference_performed": True}, {"label": phase4b.COMPARISON_LABEL, "interpretation": "descriptive only"})
    report = outputs.report.read_text(encoding="utf-8")
    assert phase4b.TGNN_LABEL in report
    assert phase4b.RF_LABEL in report
    assert phase4b.COMPARISON_LABEL in report
    assert phase4b.PRIMARY_LABEL in report
    assert "equivalent real-world validation" in report
