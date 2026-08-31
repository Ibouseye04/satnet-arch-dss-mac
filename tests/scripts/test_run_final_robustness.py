from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.final_training import run_final_robustness as phase4


EXPECTED_IDS = {
    "rf_integrated_classification": "rf_015",
    "rf_integrated_regression_mean": "rf_036",
    "rf_integrated_regression_min": "rf_018",
    "rf_space_classification": "rf_020",
    "rf_space_regression": "rf_018",
    "tgnn_space_classification": "tgnn_010",
    "tgnn_space_regression": "tgnn_012",
}


def test_loads_exact_frozen_winners_and_never_runner_ups() -> None:
    winners = phase4.load_frozen_winners()
    assert {task: value["winning_config_id"] for task, value in winners.items()} == EXPECTED_IDS
    assert winners["rf_integrated_classification"]["runner_up"]["config_id"] == "rf_019"
    assert winners["tgnn_space_classification"]["runner_up"]["config_id"] == "tgnn_009"
    assert winners["tgnn_space_regression"]["runner_up"]["config_id"] == "tgnn_011"
    assert all(value["winning_config_id"] != value["runner_up"]["config_id"] for value in winners.values())


def test_seed_and_fit_contract_is_exact() -> None:
    assert phase4.FINAL_SEEDS == (42, 123, 456, 789, 2026)
    assert phase4.PRIMARY_SEED == 42
    assert len(phase4.RF_TASKS) * len(phase4.FINAL_SEEDS) == 25
    assert len(phase4.TGNN_TASKS) * len(phase4.FINAL_SEEDS) == 10
    assert len(phase4.ALL_TASKS) * len(phase4.FINAL_SEEDS) == 35


def test_rf_configuration_overrides_only_robustness_seed() -> None:
    winner = phase4.load_frozen_winners()["rf_space_regression"]
    config = phase4.runtime_rf_configuration("rf_space_regression", winner, 789)
    assert config["random_state"] == 789
    assert config["n_jobs"] == -1
    assert config["n_estimators"] == 300
    assert config["max_depth"] == 20
    assert winner["winning_configuration"]["random_state"] == 42


def test_tgnn_winners_and_validation_monitors_are_exact() -> None:
    winners = phase4.load_frozen_winners()
    assert winners["tgnn_space_classification"]["winning_configuration"] == {
        **phase4.EXPECTED_TGNN_CONFIG,
        "cheb_k": 2,
    }
    assert winners["tgnn_space_regression"]["winning_configuration"] == {
        **phase4.EXPECTED_TGNN_CONFIG,
        "cheb_k": 3,
    }
    assert phase4.EXPECTED_TGNN_CONFIG["max_epochs"] == 100
    selection = phase4.load_json(phase4.PHASE3_ROOT / "summaries" / "tgnn_validation_selection.json")
    assert selection["early_stopping"] == {
        "min_delta": 0.0,
        "monitor_classification": "balanced_accuracy",
        "monitor_regression": "mae",
        "patience": 10,
        "restore_best": True,
    }


def test_test_rows_are_not_retained_or_target_parsed(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps({"run_id": 1, "split": "train", "target": "true"})
        + "\n"
        + json.dumps({"run_id": 2, "split": "test", "target": "not-a-target"})
        + "\n",
        encoding="utf-8",
    )
    rows, targets, test_count = phase4._read_jsonl_split(path, include_targets=True, task="tgnn_space_classification")
    assert [row["run_id"] for row in rows] == [1]
    assert targets == {1: 1}
    assert test_count == 1


def test_completed_fit_is_skipped_only_after_hash_verification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase4, "OUTPUT_ROOT", tmp_path)
    artifact = tmp_path / "rf_space_regression" / "seed_42" / "final_model.joblib"
    manifest_path = artifact.parent / "final_manifest.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"model")
    manifest = {
        "status": "completed",
        "task": "rf_space_regression",
        "seed": 42,
        "selected_config_id": "rf_018",
        "model_family": "RF",
        "selection_freeze_sha256": phase4.RF_SELECTION_HASH,
        "dataset_bundle_hash": phase4.DATASET_BUNDLE_HASH,
        "training_plan_bundle_hash": phase4.TRAINING_PLAN_BUNDLE_HASH,
        "test_accessed": False,
        "model_path": str(artifact),
        "model_sha256": phase4.sha256_file(artifact),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert phase4._verified_completed("rf_space_regression", 42, "rf_018", phase4.RF_SELECTION_HASH) == manifest
    artifact.write_bytes(b"tampered")
    with pytest.raises(phase4.Phase4Error, match="hash mismatch"):
        phase4._verified_completed("rf_space_regression", 42, "rf_018", phase4.RF_SELECTION_HASH)


def test_interrupted_tgnn_attempt_resets_its_owned_log(tmp_path: Path) -> None:
    log = tmp_path / "progress.jsonl"
    log.write_text('{"epoch": 1}\n', encoding="utf-8")
    phase4.reset_attempt_log(log)
    assert not log.exists()


def test_grid_fixed_identity_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = phase4.load_json

    def altered(path: Path):
        value = original(path)
        if path.name == "phase2_manifest.json":
            value["topology"]["grid_fixed_source_count"] = 1
        return value

    monkeypatch.setattr(phase4, "load_json", altered)
    with pytest.raises(phase4.Phase4Error, match="grid_fixed"):
        phase4._verify_dataset_identity()


def test_phase3_sources_are_read_only_during_gate_check() -> None:
    paths = (
        phase4.PHASE3_ROOT / "manifests" / "phase3_manifest.json",
        phase4.PHASE3_ROOT / "manifests" / "rf_selection_freeze.json",
        phase4.PHASE3_ROOT / "manifests" / "tgnn_selection_freeze.json",
    )
    before = {path: phase4.sha256_file(path) for path in paths}
    phase4._verify_phase3()
    assert {path: phase4.sha256_file(path) for path in paths} == before
