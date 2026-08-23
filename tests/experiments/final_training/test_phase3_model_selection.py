from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.final_training import run_validation_training as phase3


def test_phase3_candidate_contract_reconstructs_historical_sweep() -> None:
    contract, candidate_hash = phase3.candidate_contract()

    assert contract["candidate_set_hash"] == candidate_hash
    assert contract["rf_fit_count"] == 756
    assert contract["tgnn_fit_count"] == 96
    assert contract["total_fit_count"] == 852
    assert contract["task_fit_counts"] == {
        "rf_space_classification": 216,
        "rf_space_regression": 108,
        "rf_integrated_regression_mean": 108,
        "rf_integrated_regression_min": 108,
        "rf_integrated_classification": 216,
        "tgnn_space_classification": 48,
        "tgnn_space_regression": 48,
    }
    assert len(contract["candidate_set"]) == 852
    assert len({(row["task"], row["candidate_id"], row["seed"]) for row in contract["candidate_set"]}) == 852
    assert contract["test_evaluation"] is False


def test_phase3_test_firewall_rejects_test_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase3, "ROOT", tmp_path)
    candidate = tmp_path / "model_selection" / "rf" / "candidate_manifest.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps({"status": "completed", "test_accessed": False, "validation_metrics": {"mae": 1.0}}), encoding="utf-8")

    evidence = phase3.assert_test_firewall()
    assert evidence["passed"] is True
    assert evidence["candidate_manifests_checked"] == 1

    candidate.write_text(json.dumps({"status": "completed", "test_accessed": False, "test_metrics": {"mae": 0.0}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="TEST evidence"):
        phase3.assert_test_firewall()


def test_phase3_output_root_fails_closed_on_unidentified_existing_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase3, "ROOT", tmp_path)
    (tmp_path / "unexpected.txt").write_text("not a phase 3 output", encoding="utf-8")

    with pytest.raises(RuntimeError, match="resumable identity"):
        phase3.prepare_phase3_root()
