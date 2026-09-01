from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.final_training import run_heldout_evaluation as heldout


EXPECTED_IDS = {
    "rf_integrated_classification": "rf_015",
    "rf_integrated_regression_mean": "rf_036",
    "rf_integrated_regression_min": "rf_018",
    "rf_space_classification": "rf_020",
    "rf_space_regression": "rf_018",
    "tgnn_space_classification": "tgnn_010",
    "tgnn_space_regression": "tgnn_012",
}


def _artifact(tmp_path: Path, task: str = "rf_space_regression", seed: int = 42) -> tuple[heldout.FrozenArtifact, dict[str, dict[str, str]]]:
    path = tmp_path / task / f"seed_{seed}" / "final_model.joblib"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"frozen")
    manifest = {
        "status": "completed", "task": task, "seed": seed,
        "selected_config_id": heldout.EXPECTED_RF_IDS[task],
        "configuration": heldout.EXPECTED_RF_CONFIGS[task],
        "dataset_bundle_hash": heldout.DATASET_BUNDLE_HASH,
        "phase3_candidate_set_hash": heldout.CANDIDATE_SET_HASH,
        "test_accessed": False,
        "selection_freeze_sha256": heldout.RF_SELECTION_HASH,
        "model_path": str(path), "model_sha256": heldout.sha256_file(path),
    }
    manifest_path = path.parent / "final_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    phase4_relative = f"{task}/seed_{seed}/final_model.joblib"
    manifest_relative = f"{task}/seed_{seed}/final_manifest.json"
    return heldout.FrozenArtifact(task, seed, path, manifest["model_sha256"], manifest), {manifest_relative: {"sha256": heldout.sha256_file(manifest_path)}, phase4_relative: {"sha256": manifest["model_sha256"]}}


def _metadata_rows(designs: int, split: str = "test") -> list[dict[str, str]]:
    return [{"run_id": str(design * 5 + realization), "run_key": f"D{design:04d}-R{realization:02d}", "design_id": f"D{design:04d}", "realization_id": f"R{realization:02d}", "split": split} for design in range(designs) for realization in range(5)]


def _prediction_rows(designs: int = 2) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in _metadata_rows(designs):
        run_id = int(row["run_id"])
        truth = run_id % 2
        rows.append({**row, "ground_truth": truth, "prediction": truth, "positive_class_score_probability": float(truth), "class_scores": [1.0 - truth, float(truth)]})
    return rows


def test_frozen_plan_resolves_all_five_final_seeds() -> None:
    assert heldout.FINAL_SEEDS == (42, 123, 456, 789, 2026)
    assert heldout.PRIMARY_SEED == 42
    assert heldout.ALL_TASKS == tuple(EXPECTED_IDS)


def test_exact_frozen_winners_are_declared() -> None:
    assert {**heldout.EXPECTED_RF_IDS, **heldout.EXPECTED_TGNN_IDS} == EXPECTED_IDS


def test_no_ensemble_is_declared() -> None:
    assert heldout.PRIMARY_SEED in heldout.FINAL_SEEDS
    assert len(heldout.FINAL_SEEDS) == 5
    assert "ensemble" not in heldout.EXPECTED_RF_CONFIGS


def test_expected_test_cardinality_is_exact() -> None:
    assert heldout.TEST_RUNS == 1500
    assert heldout.TEST_DESIGNS == 300
    assert heldout.REALIZATIONS_PER_DESIGN == 5
    assert heldout.SPLIT_COUNTS["test"] == 1500


def test_training_and_validation_cardinality_is_frozen() -> None:
    assert heldout.SPLIT_COUNTS == {"train": 7000, "validation": 1500, "test": 1500}


def test_evaluator_has_no_training_calls() -> None:
    tree = ast.parse(inspect.getsource(heldout))
    forbidden = {"fit", "backward", "step"}
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert calls.isdisjoint(forbidden)
    assert "torch.optim" not in inspect.getsource(heldout)


def test_tgnn_inference_uses_no_optimizer_or_backward_path() -> None:
    source = inspect.getsource(heldout.evaluate_tgnn) + inspect.getsource(heldout.load_frozen_tgnn)
    assert ".backward" not in source
    assert ".step" not in source
    assert "torch.optim" not in source
    assert "no_grad" in inspect.getsource(heldout.evaluate_tgnn)


def test_rf_inference_cannot_call_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(heldout, "TEST_RUNS", 2)
    artifact, _ = _artifact(tmp_path)
    data = heldout.RFTestData(np.ones((2, 6)), np.array([0, 1]), tuple(_metadata_rows(1)[:2]), np.array([0, 1]))

    class InferenceOnlyRF:
        def fit(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("fit was called")

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.array([0, 1])[: len(x)]

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.array([[0.8, 0.2], [0.2, 0.8]])[: len(x)]

    monkeypatch.setattr(heldout, "load_rf_test_data", lambda *_args: data)
    monkeypatch.setattr(heldout, "load_frozen_rf", lambda _artifact: InferenceOnlyRF())
    metrics, rows = heldout.evaluate_rf("rf_space_classification", 42, artifact, [])
    assert metrics["balanced_accuracy"] == 1.0
    assert len(rows) == 2


def test_train_validation_inputs_do_not_modify_rf_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(heldout, "TEST_RUNS", 2)
    artifact, _ = _artifact(tmp_path, "rf_space_regression")
    data = heldout.RFTestData(np.ones((2, 6)), np.array([0.1, 0.2]), tuple(_metadata_rows(1)[:2]), np.array([0.1]))

    class FrozenRF:
        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.full(len(x), 0.15)

    model = FrozenRF()
    monkeypatch.setattr(heldout, "load_rf_test_data", lambda *_args: data)
    monkeypatch.setattr(heldout, "load_frozen_rf", lambda _artifact: model)
    _, rows = heldout.evaluate_rf("rf_space_regression", 42, artifact, [])
    assert [row["split"] for row in rows] == ["test", "test"]


def test_only_final_phase4_artifact_path_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact, inventory = _artifact(tmp_path)
    monkeypatch.setattr(heldout, "PHASE4_ROOT", tmp_path)
    checked = heldout._verify_frozen_artifact("rf_space_regression", 42, inventory)
    assert checked.model_path.name == "final_model.joblib"


def test_non_final_artifact_path_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact, inventory = _artifact(tmp_path)
    manifest_path = tmp_path / "rf_space_regression" / "seed_42" / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_path"] = str(tmp_path / "runner_up.joblib")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    inventory["rf_space_regression/seed_42/final_manifest.json"]["sha256"] = heldout.sha256_file(manifest_path)
    monkeypatch.setattr(heldout, "PHASE4_ROOT", tmp_path)
    with pytest.raises(heldout.HeldoutError, match="Non-final artifact path"):
        heldout._verify_frozen_artifact("rf_space_regression", 42, inventory)


def test_model_sha_mismatch_is_hard_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, inventory = _artifact(tmp_path)
    relative = "rf_space_regression/seed_42/final_model.joblib"
    inventory[relative]["sha256"] = "0" * 64
    monkeypatch.setattr(heldout, "PHASE4_ROOT", tmp_path)
    with pytest.raises(heldout.HeldoutError, match="inventory SHA mismatch"):
        heldout._verify_frozen_artifact("rf_space_regression", 42, inventory)


def test_phase4_inventory_mismatch_is_hard_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inventory_path = tmp_path / "final_robustness_inventory.json"
    inventory_path.write_text(json.dumps({"bundle_sha256": "x", "test_accessed": False, "artifacts": []}), encoding="utf-8")
    monkeypatch.setattr(heldout, "PHASE4_ROOT", tmp_path)
    monkeypatch.setattr(heldout, "PHASE4_INVENTORY_SHA", heldout.sha256_file(inventory_path))
    monkeypatch.setattr(heldout, "PHASE4_BUNDLE_HASH", "x")
    with pytest.raises(heldout.HeldoutError, match="artifact count"):
        heldout._verify_phase4_inventory()


def test_grid_fixed_lineage_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = heldout.load_json

    def altered(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "phase2_manifest.json":
            value["historical_fixed_source_contamination"] = 1
        return value

    monkeypatch.setattr(heldout, "load_json", altered)
    with pytest.raises(heldout.HeldoutError, match="grid_fixed"):
        heldout._verify_dataset()


def test_adaptive_topology_tuple_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    original = heldout.load_json

    def altered(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "phase2_manifest.json":
            value["topology"]["aggregate"][0]["tuple"] = ["grid_fixed", 1, 1, "persistent_temporal_union_edges_v1"]
        return value

    monkeypatch.setattr(heldout, "load_json", altered)
    with pytest.raises(heldout.HeldoutError, match="topology"):
        heldout._verify_dataset()


def test_split_metadata_rejects_design_crossing_splits() -> None:
    rows: list[dict[str, str]] = []
    for design in range(2000):
        split = "train" if design < 1400 else "validation" if design < 1700 else "test"
        rows.extend(_metadata_rows(design + 1, split)[-5:])
    rows[4]["split"] = "test"
    rows[-1]["split"] = "train"
    with pytest.raises(heldout.HeldoutError, match="crosses splits"):
        heldout._validate_split_metadata(rows)


def test_split_metadata_requires_five_realizations() -> None:
    rows = [{"run_id": str(index), "run_key": f"D0000-R{index:02d}", "design_id": "D0000", "realization_id": f"R{index:02d}", "split": "test"} for index in range(4)]
    with pytest.raises(heldout.HeldoutError, match="row count"):
        heldout._validate_split_metadata(rows)


def test_paired_bootstrap_uses_design_clusters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(heldout, "TEST_RUNS", 10)
    monkeypatch.setattr(heldout, "TEST_DESIGNS", 2)
    monkeypatch.setattr(heldout, "BOOTSTRAP_REPLICATES", 4)
    rf = _prediction_rows()
    tg = [dict(row) for row in rf]
    observed: list[list[int]] = []
    original = heldout._metric_value

    def wrapped(rows: list[dict[str, object]], metric: str, classification: bool) -> float:
        observed.append([sum(str(item["design_id"]) == design for item in rows) for design in ("D0000", "D0001")])
        return original(rows, metric, classification)

    monkeypatch.setattr(heldout, "_metric_value", wrapped)
    result = heldout.paired_bootstrap(rf, tg, True)
    assert result["resampling_unit"] == "design_id"
    assert all(count % 5 == 0 for sample in observed for count in sample)


def test_paired_bootstrap_requires_same_rf_tgnn_clusters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(heldout, "TEST_RUNS", 10)
    monkeypatch.setattr(heldout, "TEST_DESIGNS", 2)
    rf = _prediction_rows()
    tg = [dict(row) for row in reversed(rf)]
    with pytest.raises(heldout.HeldoutError, match="identical TEST runs"):
        heldout.paired_bootstrap(rf, tg, True)


def test_bootstrap_contract_is_frozen() -> None:
    assert heldout.BOOTSTRAP_SEED == 20260812
    assert heldout.BOOTSTRAP_REPLICATES == 2000
    assert heldout.CONFIDENCE_LEVEL == 0.95
    assert heldout.REALIZATIONS_PER_DESIGN == 5


def test_classification_primary_orientation_and_metric_set() -> None:
    source = inspect.getsource(heldout._comparison_statistics)
    assert "balanced_accuracy" in source
    assert "RF minus TGNN balanced accuracy" in source
    assert "rf_space_classification" in source and "tgnn_space_classification" in source
    metric_source = inspect.getsource(heldout._metric_value)
    for metric in ("balanced_accuracy", "macro_f1", "minority_class_recall", "roc_auc", "pr_auc"):
        assert metric in metric_source


def test_regression_primary_orientation_and_no_metric_substitution() -> None:
    source = inspect.getsource(heldout._comparison_statistics)
    assert "RF MAE minus TGNN MAE" in source
    assert "relative_mae_improvement_percent" in source
    assert "rf_space_regression" in source and "tgnn_space_regression" in source


def test_threshold_is_not_optimized() -> None:
    source = inspect.getsource(heldout).lower()
    assert "threshold_selection" not in source
    assert ".optimize" not in source
    assert "no_threshold_optimization" in source
    assert ".predict_proba" in source


def test_risk_category_logic_is_absent() -> None:
    source = inspect.getsource(heldout).lower()
    assert "watchlist" not in source
    assert "critical" not in source
    assert "healthy" not in source


def test_output_root_cannot_alias_phase4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(heldout, "OUTPUT_ROOT", heldout.PHASE4_ROOT)
    with pytest.raises(heldout.HeldoutError, match="aliases"):
        heldout._safe_output_root()


def test_nonempty_output_root_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "heldout"
    output.mkdir()
    (output / "existing.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(heldout, "OUTPUT_ROOT", output)
    with pytest.raises(heldout.HeldoutError, match="absent or empty"):
        heldout._safe_output_root()


def test_output_root_nested_under_phase4_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    phase4 = tmp_path / "phase4"
    output = phase4 / "heldout"
    monkeypatch.setattr(heldout, "PHASE3_ROOT", tmp_path / "phase3")
    monkeypatch.setattr(heldout, "PHASE4_ROOT", phase4)
    monkeypatch.setattr(heldout, "OUTPUT_ROOT", output)
    with pytest.raises(heldout.HeldoutError, match="overlaps"):
        heldout._safe_output_root()


def test_default_mode_does_not_authorize_execution(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    evidence = {"test_accessed": False}
    called = False
    monkeypatch.setattr(heldout, "preflight", lambda: evidence)

    def forbidden(_evidence: dict[str, object]) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("held-out execution was entered")

    monkeypatch.setattr(heldout, "run_authorized", forbidden)
    assert heldout.main([]) == evidence
    assert called is False
    assert json.loads(capsys.readouterr().out)["test_accessed"] is False


def test_explicit_authorization_flag_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "--execute-heldout" in [action.option_strings[0] for action in heldout.build_parser()._actions if action.option_strings]


def test_explicit_authorization_enters_execution_only_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = {"test_accessed": False}
    summary = {"status": "PASS"}
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(heldout, "preflight", lambda: evidence)
    monkeypatch.setattr(heldout, "run_authorized", lambda value: calls.append(value) or summary)
    assert heldout.main(["--execute-heldout"]) == summary
    assert calls == [evidence]


def test_preflight_evidence_explicitly_seals_test_payloads() -> None:
    source = inspect.getsource(heldout.preflight)
    assert "test_targets_loaded" in source
    assert "test_feature_payload_loaded" in source
    assert "test_graph_payload_loaded" in source
    assert "test_accessed" in source


def test_preflight_source_does_not_call_test_payload_loaders() -> None:
    source = inspect.getsource(heldout.preflight)
    assert "load_rf_test_data" not in source
    assert "load_tgnn_test_data" not in source
    assert "load_sequence" not in source


def test_optimizer_is_not_instantiated_during_heldout_inference() -> None:
    source = inspect.getsource(heldout)
    assert "torch.optim" not in source
    assert "Optimizer" not in source


def test_phase3_and_phase4_are_never_output_targets() -> None:
    assert heldout.OUTPUT_ROOT != heldout.PHASE3_ROOT
    assert heldout.OUTPUT_ROOT != heldout.PHASE4_ROOT


def test_no_historical_fixed_artifact_reference_is_authorized() -> None:
    source = inspect.getsource(heldout._verify_frozen_artifact)
    assert "final_model.joblib" in source
    assert "best_validation_checkpoint.pt" in source
    assert "runner_up" not in source
