"""Tests for tools/validate_proxy_rankings.py."""

from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

pytest.importorskip("scipy")


def _run_tool(monkeypatch, args: list[str]) -> None:
    import tools.validate_proxy_rankings as validate_tool

    monkeypatch.setattr(sys, "argv", ["validate_proxy_rankings.py", *args])
    validate_tool.main()


def test_duplicate_join_keys_fail_loudly(tmp_path, monkeypatch, capsys) -> None:
    ref_path = tmp_path / "ref.csv"
    proxy_path = tmp_path / "proxy.csv"

    pd.DataFrame(
        [
            {"config_hash": "a", "y_pred": 0.1},
            {"config_hash": "a", "y_pred": 0.2},
        ]
    ).to_csv(ref_path, index=False)
    pd.DataFrame(
        [
            {"config_hash": "a", "y_pred": 0.3},
            {"config_hash": "b", "y_pred": 0.4},
        ]
    ).to_csv(proxy_path, index=False)

    with pytest.raises(SystemExit) as exc:
        _run_tool(monkeypatch, [str(ref_path), str(proxy_path)])

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "Join key is not unique" in output
    assert "left: 1" in output


def test_report_includes_match_and_drop_counts(tmp_path, monkeypatch) -> None:
    ref_path = tmp_path / "ref.csv"
    proxy_path = tmp_path / "proxy.csv"
    report_path = tmp_path / "report.json"

    pd.DataFrame(
        [
            {"config_hash": "a", "y_pred": 0.1},
            {"config_hash": "b", "y_pred": 0.2},
            {"config_hash": "c", "y_pred": 0.3},
        ]
    ).to_csv(ref_path, index=False)
    pd.DataFrame(
        [
            {"config_hash": "b", "y_pred": 0.2},
            {"config_hash": "c", "y_pred": 0.4},
            {"config_hash": "d", "y_pred": 0.5},
        ]
    ).to_csv(proxy_path, index=False)

    _run_tool(
        monkeypatch,
        [
            str(ref_path),
            str(proxy_path),
            "--allow-partial",
            "--output",
            str(report_path),
        ],
    )

    report = json.loads(report_path.read_text())
    assert report["left_row_count"] == 3
    assert report["right_row_count"] == 3
    assert report["matched_row_count"] == 2
    assert report["dropped_left_count"] == 1
    assert report["dropped_right_count"] == 1
    assert report["left_duplicate_key_count"] == 0
    assert report["right_duplicate_key_count"] == 0


def test_null_join_keys_on_left_fail_loudly(tmp_path, monkeypatch, capsys) -> None:
    ref_path = tmp_path / "ref.csv"
    proxy_path = tmp_path / "proxy.csv"

    pd.DataFrame(
        [
            {"config_hash": None, "y_pred": 0.1},
            {"config_hash": "b", "y_pred": 0.2},
        ]
    ).to_csv(ref_path, index=False)
    pd.DataFrame(
        [
            {"config_hash": "a", "y_pred": 0.3},
            {"config_hash": "b", "y_pred": 0.4},
        ]
    ).to_csv(proxy_path, index=False)

    with pytest.raises(SystemExit) as exc:
        _run_tool(monkeypatch, [str(ref_path), str(proxy_path)])

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "null/NaN/empty values" in output
    assert "reference: 1" in output
    assert "proxy: 0" in output


def test_null_join_keys_on_right_fail_loudly(tmp_path, monkeypatch, capsys) -> None:
    ref_path = tmp_path / "ref.csv"
    proxy_path = tmp_path / "proxy.csv"

    pd.DataFrame(
        [
            {"config_hash": "a", "y_pred": 0.1},
            {"config_hash": "b", "y_pred": 0.2},
        ]
    ).to_csv(ref_path, index=False)
    pd.DataFrame(
        [
            {"config_hash": "a", "y_pred": 0.3},
            {"config_hash": None, "y_pred": 0.4},
        ]
    ).to_csv(proxy_path, index=False)

    with pytest.raises(SystemExit) as exc:
        _run_tool(monkeypatch, [str(ref_path), str(proxy_path)])

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "null/NaN/empty values" in output
    assert "reference: 0" in output
    assert "proxy: 1" in output


def test_whitespace_join_keys_fail_loudly(tmp_path, monkeypatch, capsys) -> None:
    ref_path = tmp_path / "ref.csv"
    proxy_path = tmp_path / "proxy.csv"

    pd.DataFrame(
        [
            {"config_hash": "   ", "y_pred": 0.1},
            {"config_hash": "b", "y_pred": 0.2},
        ]
    ).to_csv(ref_path, index=False)
    pd.DataFrame(
        [
            {"config_hash": "a", "y_pred": 0.3},
            {"config_hash": "b", "y_pred": 0.4},
        ]
    ).to_csv(proxy_path, index=False)

    with pytest.raises(SystemExit) as exc:
        _run_tool(monkeypatch, [str(ref_path), str(proxy_path)])

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "null/NaN/empty values" in output
    assert "reference: 1" in output
