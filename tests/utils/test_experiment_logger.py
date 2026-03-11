"""Tests for satnet.utils.experiment_logger."""

from __future__ import annotations

import json
import time

from satnet.utils.experiment_logger import (
    ExperimentLogger,
    compute_target_distribution,
    jsonl_to_csv,
)


class TestExperimentLogger:
    def test_context_manager_creates_jsonl(self, tmp_path) -> None:
        log_path = tmp_path / "log.jsonl"
        with ExperimentLogger(log_path) as log:
            log.set("model_type", "RF")
        assert log_path.exists()
        record = json.loads(log_path.read_text().strip())
        assert record["model_type"] == "RF"
        assert "timestamp" in record
        assert "git_sha" in record
        assert "timings" in record
        assert "total_run_seconds" in record["timings"]

    def test_set_metrics(self, tmp_path) -> None:
        log_path = tmp_path / "log.jsonl"
        with ExperimentLogger(log_path) as log:
            log.set_metrics({"mae": 0.05})
            log.set_metrics({"r2": 0.9})
        record = json.loads(log_path.read_text().strip())
        assert record["metrics"]["mae"] == 0.05
        assert record["metrics"]["r2"] == 0.9

    def test_timer(self, tmp_path) -> None:
        log_path = tmp_path / "log.jsonl"
        with ExperimentLogger(log_path) as log:
            log.start_timer("train")
            time.sleep(0.01)
            elapsed = log.stop_timer("train")
        assert elapsed >= 0.01
        record = json.loads(log_path.read_text().strip())
        assert record["timings"]["train_seconds"] >= 0.01

    def test_multiple_flushes_append(self, tmp_path) -> None:
        log_path = tmp_path / "log.jsonl"
        with ExperimentLogger(log_path) as log:
            log.set("run", 1)
        with ExperimentLogger(log_path) as log:
            log.set("run", 2)
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2


class TestJsonlToCsv:
    def test_conversion(self, tmp_path) -> None:
        jsonl_path = tmp_path / "log.jsonl"
        csv_path = tmp_path / "summary.csv"
        with ExperimentLogger(jsonl_path) as log:
            log.set("model_type", "RF")
            log.set_metrics({"mae": 0.1})
        jsonl_to_csv(jsonl_path, csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "model_type" in content
        assert "metrics.mae" in content


class TestComputeTargetDistribution:
    def test_basic(self) -> None:
        stats = compute_target_distribution([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["count"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0

    def test_empty(self) -> None:
        assert compute_target_distribution([]) == {}
