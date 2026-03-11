"""Lightweight experiment logger — structured local logging to JSONL/CSV.

No external tracking system. Every run appends one record to a JSONL file
and optionally writes a flat CSV summary.

Usage::

    from satnet.utils.experiment_logger import ExperimentLogger

    with ExperimentLogger("experiments/log.jsonl") as log:
        log.set("model_type", "RandomForest")
        log.set("target_name", "gcc_frac_min")
        log.start_timer("training")
        ...
        log.stop_timer("training")
        log.set_metrics({"mae": 0.05, "r2": 0.92})
        # record is auto-flushed on __exit__
"""

from __future__ import annotations

import csv
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


class ExperimentLogger:
    """Accumulates key-value pairs for a single run, then flushes to JSONL."""

    def __init__(self, jsonl_path: str | Path) -> None:
        self._path = Path(jsonl_path)
        self._record: dict[str, Any] = {}
        self._timers: dict[str, float] = {}

        # Pre-fill best-effort metadata
        self._record["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._record["git_sha"] = _git_sha()
        self._record["hostname"] = _hostname()
        self._record["argv"] = sys.argv

    # ── public API ──────────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        self._record[key] = value

    def set_many(self, d: dict[str, Any]) -> None:
        self._record.update(d)

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        self._record.setdefault("metrics", {}).update(metrics)

    def set_cache_metrics(self, cache: dict[str, Any]) -> None:
        self._record.setdefault("cache", {}).update(cache)

    def set_target_stats(self, stats: dict[str, Any]) -> None:
        self._record["target_stats"] = stats

    # ── timers ──────────────────────────────────────────────────────

    def start_timer(self, name: str) -> None:
        self._timers[name] = time.monotonic()

    def stop_timer(self, name: str) -> float:
        elapsed = time.monotonic() - self._timers.pop(name)
        timings = self._record.setdefault("timings", {})
        timings[f"{name}_seconds"] = round(elapsed, 4)
        return elapsed

    # ── flush ───────────────────────────────────────────────────────

    def flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(self._record, default=str) + "\n")
        logger.info("Experiment record appended to %s", self._path)

    # ── context manager ─────────────────────────────────────────────

    def __enter__(self) -> "ExperimentLogger":
        self.start_timer("total_run")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        if "total_run" in self._timers:
            self.stop_timer("total_run")
        self.flush()


# ── CSV summary helper ──────────────────────────────────────────────

def jsonl_to_csv(jsonl_path: str | Path, csv_path: str | Path) -> None:
    """Flatten a JSONL experiment log into a CSV summary."""
    records: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return

    # Flatten nested dicts with dot notation
    flat_records = [_flatten(r) for r in records]

    all_keys: list[str] = []
    seen: set[str] = set()
    for rec in flat_records:
        for k in rec:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_records)


def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ── target distribution stats helper ────────────────────────────────

def compute_target_distribution(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a target variable."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {}
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }
