"""CLI target truthfulness tests for training scripts."""

from __future__ import annotations

import sys

import pytest


def test_rf_cli_rejects_removed_num_components_max(monkeypatch) -> None:
    pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
    import scripts.train_design_risk_model as rf_script

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_design_risk_model.py", "--target-name", "num_components_max"],
    )
    with pytest.raises(SystemExit):
        rf_script.parse_args()


def test_gnn_cli_rejects_removed_num_components_max(monkeypatch) -> None:
    pytest.importorskip("torch")
    import scripts.train_gnn_model as gnn_script

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_gnn_model.py", "--target-name", "num_components_max"],
    )
    with pytest.raises(SystemExit):
        gnn_script.parse_args()
