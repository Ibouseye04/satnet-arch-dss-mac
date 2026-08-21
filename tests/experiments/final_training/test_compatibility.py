from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from satnet.experiments.final_training.baselines import classification_baselines_from_train, regression_baselines_from_train
from satnet.experiments.final_training.bootstrap import paired_design_cluster_bootstrap
from satnet.experiments.final_training.checkpoints import CheckpointManifest, EarlyStopping, monitored_metric
from satnet.experiments.final_training.contracts import AUTHORIZED_TASKS, DATASET_ROOT, FINAL_SEEDS, TRAINING_PLAN_BUNDLE_HASH, DATASET_BUNDLE_HASH
from satnet.experiments.final_training.metrics import classification_metrics, regression_metrics
from satnet.experiments.final_training.rf_loader import load_rf_task
from satnet.experiments.final_training.search import candidate_counts, rf_configs, tgnn_configs, validate_rf_configs_without_fit
from satnet.experiments.final_training.selection import ValidationRecord, select_model
from satnet.experiments.final_training.seeds import initialize_determinism
from satnet.experiments.final_training.splits import validate_frozen_split_frame
from satnet.experiments.final_training.test_access import FinalEvaluationAuthorization, TestTargetAccessError
from satnet.experiments.final_training.tgnn_loader import load_tgnn_task, read_sequence
from satnet.experiments.final_training.rf_runner import TrainingDisabledError as RFTrainingDisabledError, train_final as rf_train_final
from satnet.experiments.final_training.tgnn_runner import TrainingDisabledError as TGNNTrainingDisabledError, train_final as tgnn_train_final


def test_authorized_tasks_are_exactly_frozen() -> None:
    assert list(AUTHORIZED_TASKS) == [
        "rf_space_classification", "rf_space_regression", "tgnn_space_classification",
        "tgnn_space_regression", "rf_integrated_regression_mean", "rf_integrated_regression_min",
        "rf_integrated_classification",
    ]


def test_rf_all_schemas_and_split_counts() -> None:
    expected = {
        "rf_space_classification": (6, "space_threshold_breach_any"),
        "rf_space_regression": (6, "space_gcc_fraction_original_min"),
        "rf_integrated_regression_mean": (10, "failure_adjusted_overall_service_fraction_mean"),
        "rf_integrated_regression_min": (10, "failure_adjusted_overall_service_fraction_min"),
        "rf_integrated_classification": (10, "overall_threshold_breach_any"),
    }
    for task_id, (feature_count, target) in expected.items():
        bundle = load_rf_task(task_id, dataset_root=DATASET_ROOT, verify_bundle=False)
        assert len(bundle.feature_order) == feature_count
        assert bundle.task.target == target
        assert {split: len(bundle.splits.indices_for(split)) for split in ("train", "validation", "test")} == {"train": 7000, "validation": 1500, "test": 1500}
        assert set(bundle.feature_order).isdisjoint({"run_id", "design_id", "realization_id", "split"})
        assert bundle.view("train").features.columns.tolist() == list(bundle.feature_order)


def test_rf_test_targets_are_gated() -> None:
    bundle = load_rf_task("rf_space_classification", dataset_root=DATASET_ROOT, verify_bundle=False)
    with pytest.raises(TestTargetAccessError, match="TEST targets"):
        _ = bundle.view("test").targets
    auth = FinalEvaluationAuthorization(True, True, True, True, DATASET_BUNDLE_HASH, TRAINING_PLAN_BUNDLE_HASH)
    assert len(bundle.view("test").authorized_targets(auth)) == 1500


def test_missing_split_metadata_fails_without_resplit() -> None:
    frame = pd.DataFrame({"run_id": [0, 1, 2], "design_id": ["D0"] * 3, "realization_id": ["R00", "R01", "R02"]})
    with pytest.raises(ValueError, match="split"):
        validate_frozen_split_frame(frame)


def test_rf_search_counts_and_parameter_validation() -> None:
    assert candidate_counts() == {"rf_space_classification": 72, "rf_space_regression": 36, "rf_integrated_regression_mean": 36, "rf_integrated_regression_min": 36, "rf_integrated_classification": 72}
    assert sum(candidate_counts().values()) == 252
    assert validate_rf_configs_without_fit("rf_space_classification") == 72
    assert len(tgnn_configs()) == 16


def test_classification_metrics_handle_absent_predicted_class() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])
    assert metrics["confusion_matrix"] == [[2, 0], [2, 0]]
    assert metrics["specificity"] == 1.0
    assert metrics["sensitivity"] == 0.0
    assert metrics["f1_by_class"]["1"] == 0.0
    assert metrics["roc_auc"] == 1.0
    assert classification_metrics([0, 1], [0, 1], [[0.9, 0.1], [0.1, 0.9]])["roc_auc"] == 1.0


def test_regression_metrics_do_not_clip_predictions() -> None:
    metrics = regression_metrics([0.0, 1.0], [-0.5, 1.5])
    assert metrics["mae"] == 0.5
    assert metrics["predictions_below_zero"] == 1
    assert metrics["predictions_above_one"] == 1


def test_baselines_use_train_only() -> None:
    classification = classification_baselines_from_train([0, 0, 1], seed=42)
    assert classification.majority_class == 0
    assert classification.positive_prevalence == pytest.approx(1 / 3)
    assert len(classification.stratified_random_predict(100)) == 100
    regression = regression_baselines_from_train([1.0, 2.0, 100.0])
    assert regression.mean == pytest.approx(103 / 3)
    assert regression.median == 2.0


def test_tgnn_binary_sequence_and_target_manifests() -> None:
    classification = load_tgnn_task("tgnn_space_classification", dataset_root=DATASET_ROOT, verify_bundle=False)
    regression = load_tgnn_task("tgnn_space_regression", dataset_root=DATASET_ROOT, verify_bundle=False)
    assert len(classification.graph_records) == len(classification.target_records) == 10000
    assert classification.task.target == "space_threshold_breach_any"
    assert regression.task.target == "space_gcc_fraction_original_min"
    artifact = classification.structural_sample("train")
    assert artifact.sequence_length == 11
    assert artifact.node_features.shape[1] == 3
    assert artifact.edge_attr.shape[1] == 4
    assert [int(data.timestep_index.item()) for data in artifact.data_list()] == list(range(11))
    assert artifact.header["sequence"]["split"] == "train"


def test_tgnn_test_targets_are_metadata_only() -> None:
    bundle = load_tgnn_task("tgnn_space_regression", dataset_root=DATASET_ROOT, verify_bundle=False)
    with pytest.raises(TestTargetAccessError):
        _ = bundle.view("test").targets
    artifact, target = bundle.load_run(bundle.splits.run_ids["test"][0])
    assert artifact.sequence_length == 11
    assert target is None


def test_early_stopping_monitors_frozen_validation_metric_and_restores() -> None:
    assert monitored_metric("classification") == ("balanced_accuracy", "maximize")
    assert monitored_metric("regression") == ("mae", "minimize")
    stop = EarlyStopping("classification")
    assert stop.update(1, 0.5, {"weight": 1})
    assert stop.update(2, 0.5, {"weight": 2}) is False
    for epoch in range(3, 12):
        stop.update(epoch, 0.5)
    assert stop.should_stop is True
    assert stop.best_epoch == 1 and stop.best_state == {"weight": 1}


def test_selection_uses_frozen_seed_aggregation_and_tie_break() -> None:
    records = []
    for seed in (42, 123, 456):
        records.extend([
            ValidationRecord("complex", seed, {"balanced_accuracy": 0.8, "macro_f1": 0.7}, {"max_depth": 20, "n_estimators": 600}),
            ValidationRecord("simple", seed, {"balanced_accuracy": 0.8, "macro_f1": 0.7}, {"max_depth": 10, "n_estimators": 300}),
        ])
    assert select_model(records, task_type="classification")["selected_candidate_id"] == "simple"


def test_seed_control_rejects_dynamic_seed_and_initializes_frozen_seed() -> None:
    assert initialize_determinism(42)["seed"] == 42
    with pytest.raises(ValueError):
        initialize_determinism(7)


def test_checkpoint_schema_binds_both_frozen_bundles() -> None:
    manifest = CheckpointManifest("rf_space_regression", "RF", {"n_estimators": 300}, 42, DATASET_BUNDLE_HASH, TRAINING_PLAN_BUNDLE_HASH, 7000, 1500, None, {"mae": 0.1}, "0" * 64, {"python": "3.11.9"})
    assert manifest.to_dict()["train_count"] == 7000


def test_design_cluster_bootstrap_is_paired_and_deterministic() -> None:
    rows = [{"design_id": f"D{d}", "realization_id": f"R{r:02d}", "y_true": r % 2, "rf_prediction": r % 2, "tgnn_prediction": 1 - (r % 2)} for d in range(6) for r in range(5)]
    first = paired_design_cluster_bootstrap(rows, task_type="classification", replicates=50)
    second = paired_design_cluster_bootstrap(rows, task_type="classification", replicates=50)
    assert first == second
    assert first.metric == "balanced_accuracy"


def test_training_entry_points_are_hard_disabled() -> None:
    with pytest.raises(RFTrainingDisabledError):
        rf_train_final()
    with pytest.raises(TGNNTrainingDisabledError):
        tgnn_train_final()


def test_new_final_path_has_no_resplit_or_training_calls() -> None:
    root = Path(__file__).parents[3] / "src" / "satnet" / "experiments" / "final_training"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "train_test_split" not in text
    assert "make_run_splits" not in text
    assert ".fit(" not in text
    assert "backward(" not in text
    assert "optimizer.step(" not in text
