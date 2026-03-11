from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence
from uuid import uuid4

from satnet.metrics.resilience_targets import infer_task_type

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments" / "autoresearch"
SUPPORTED_SEARCH_TARGETS = ("gcc_frac_min", "partition_fraction")
SUPPORTED_FEATURE_MODES = ("tier1_full", "design_only", "custom")
SUPPORTED_FIDELITY_TIERS = ("screening", "confirmatory")
PRIMARY_METRIC = "test_rmse"
SECONDARY_METRIC = "test_spearman_rho"
TERTIARY_METRIC = "test_r2"


@dataclass(frozen=True)
class RfExperimentConfig:
    dataset_path: str
    target_name: str = "gcc_frac_min"
    feature_mode: Literal["tier1_full", "design_only", "custom"] = "tier1_full"
    custom_feature_columns: tuple[str, ...] = ()
    n_estimators: int = 300
    max_depth: int | None = None
    seed: int = 42
    fidelity_tier: Literal["screening", "confirmatory"] = "screening"
    experiment_type: str = "rf_autoresearch_v1"
    notes: str = ""
    save_model_artifact: bool = True

    def validated(self) -> "RfExperimentConfig":
        task_type = infer_task_type(self.target_name)
        if task_type != "regression":
            raise ValueError(
                "Autoresearch RF v1 is regression-only. "
                f"Target '{self.target_name}' resolves to task_type='{task_type}'."
            )
        if self.feature_mode not in SUPPORTED_FEATURE_MODES:
            raise ValueError(
                f"Unsupported feature_mode '{self.feature_mode}'. "
                f"Expected one of {SUPPORTED_FEATURE_MODES}."
            )
        if self.fidelity_tier not in SUPPORTED_FIDELITY_TIERS:
            raise ValueError(
                f"Unsupported fidelity_tier '{self.fidelity_tier}'. "
                f"Expected one of {SUPPORTED_FIDELITY_TIERS}."
            )
        dataset_path = Path(self.dataset_path).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be positive when provided")
        if self.feature_mode == "custom" and not self.custom_feature_columns:
            raise ValueError("custom_feature_columns must be provided when feature_mode='custom'")
        return replace(self, dataset_path=str(dataset_path))

    def resolved_feature_columns(self) -> list[str]:
        if self.feature_mode == "tier1_full":
            return _tier1_feature_columns()
        if self.feature_mode == "design_only":
            return _tier1_design_feature_columns()
        return list(self.custom_feature_columns)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["custom_feature_columns"] = list(self.custom_feature_columns)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RfExperimentConfig":
        allowed_keys = {
            "dataset_path",
            "target_name",
            "feature_mode",
            "custom_feature_columns",
            "n_estimators",
            "max_depth",
            "seed",
            "fidelity_tier",
            "experiment_type",
            "notes",
            "save_model_artifact",
        }
        data = {key: value for key, value in payload.items() if key in allowed_keys}
        custom_columns = data.get("custom_feature_columns", ())
        if isinstance(custom_columns, list):
            data["custom_feature_columns"] = tuple(str(value) for value in custom_columns)
        return cls(**data)


def load_experiment_config(config_path: str | Path) -> RfExperimentConfig:
    with open(config_path) as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "config" in payload and isinstance(payload["config"], dict):
        payload = payload["config"]
    if not isinstance(payload, dict):
        raise ValueError("Experiment config file must contain a JSON object")
    return RfExperimentConfig.from_dict(payload)


def run_rf_experiment(
    config: RfExperimentConfig,
    *,
    experiments_root: str | Path | None = None,
    parent_experiment_id: str | None = None,
) -> dict[str, Any]:
    cfg = config.validated()
    experiments_root_path = Path(experiments_root or DEFAULT_EXPERIMENTS_ROOT).expanduser().resolve()
    experiments_root_path.mkdir(parents=True, exist_ok=True)
    ledger_path = experiments_root_path / "results.jsonl"

    experiment_id = _make_experiment_id()
    experiment_dir = experiments_root_path / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)

    config_path = experiment_dir / "config.json"
    metrics_path = experiment_dir / "metrics.json"
    predictions_path = experiment_dir / "predictions.csv"
    model_path = experiment_dir / "model.joblib"
    notes_path = experiment_dir / "notes.txt"

    config_payload = {
        "experiment_id": experiment_id,
        "parent_experiment_id": parent_experiment_id,
        "config": cfg.to_dict(),
    }
    _write_json(config_path, config_payload)
    if cfg.notes.strip():
        notes_path.write_text(cfg.notes)

    task_type = infer_task_type(cfg.target_name)
    timestamp = datetime.now(timezone.utc).isoformat()
    git_sha = _git_sha()
    started = time.monotonic()

    artifact_paths: dict[str, str | None] = {
        "experiment_dir": str(experiment_dir),
        "config": str(config_path),
        "metrics": str(metrics_path),
        "predictions": None,
        "model": None,
        "notes": str(notes_path) if cfg.notes.strip() else None,
        "ledger": str(ledger_path),
    }

    status = "succeeded"
    metrics: dict[str, Any]
    promotion_decision = "rejected"
    promotion_reason = "Not evaluated"

    try:
        model, metrics, predictions = _train_rf_once(cfg)
        _write_json(metrics_path, metrics)
        predictions.to_csv(predictions_path, index=False)
        artifact_paths["predictions"] = str(predictions_path)
        if cfg.save_model_artifact:
            _save_model_artifact(model, model_path)
            artifact_paths["model"] = str(model_path)

        incumbent = find_incumbent_result(
            load_results_ledger(ledger_path),
            dataset_path=cfg.dataset_path,
            target_name=cfg.target_name,
            fidelity_tier=cfg.fidelity_tier,
        )
        promoted, promotion_reason = should_promote(metrics, incumbent)
        promotion_decision = "promoted" if promoted else "rejected"
    except Exception as exc:
        status = "failed"
        metrics = {
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
        _write_json(metrics_path, metrics)
        promotion_decision = "failed"
        promotion_reason = f"Run failed with {exc.__class__.__name__}: {exc}"

    runtime_seconds = round(time.monotonic() - started, 4)
    result = {
        "experiment_id": experiment_id,
        "parent_experiment_id": parent_experiment_id,
        "status": status,
        "timestamp": timestamp,
        "git_sha": git_sha,
        "model_family": "RandomForest",
        "experiment_type": cfg.experiment_type,
        "target_name": cfg.target_name,
        "task_type": task_type,
        "dataset_path": cfg.dataset_path,
        "fidelity_tier": cfg.fidelity_tier,
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "metrics": metrics,
        "artifact_paths": artifact_paths,
        "promotion_decision": promotion_decision,
        "promotion_reason": promotion_reason,
        "runtime_seconds": runtime_seconds,
    }
    append_result_ledger(ledger_path, result)
    return result


def run_rf_search_loop(
    baseline_config: RfExperimentConfig,
    *,
    experiments_root: str | Path | None = None,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    baseline = baseline_config.validated()
    candidates = build_screening_candidates(baseline)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    results: list[dict[str, Any]] = []
    baseline_experiment_id: str | None = None
    for index, candidate in enumerate(candidates):
        parent_experiment_id = baseline_experiment_id if index > 0 else None
        result = run_rf_experiment(
            candidate,
            experiments_root=experiments_root,
            parent_experiment_id=parent_experiment_id,
        )
        if index == 0:
            baseline_experiment_id = result["experiment_id"]
        results.append(result)

    successful_results = [row for row in results if row.get("status") == "succeeded"]
    best_result = None
    for row in successful_results:
        if best_result is None:
            best_result = row
            continue
        promoted, _ = should_promote(row.get("metrics", {}), best_result)
        if promoted:
            best_result = row

    return {
        "baseline_experiment_id": baseline_experiment_id,
        "best_experiment_id": best_result.get("experiment_id") if best_result else None,
        "result_count": len(results),
        "results": results,
    }


def build_screening_candidates(baseline_config: RfExperimentConfig) -> list[RfExperimentConfig]:
    baseline = baseline_config.validated()
    candidates = [
        baseline,
        replace(baseline, n_estimators=baseline.n_estimators * 2),
        replace(baseline, max_depth=12),
        replace(baseline, feature_mode="design_only", custom_feature_columns=()),
        replace(baseline, feature_mode="design_only", custom_feature_columns=(), max_depth=12),
    ]
    deduped: list[RfExperimentConfig] = []
    seen: set[str] = set()
    for candidate in candidates:
        signature = json.dumps(candidate.to_dict(), sort_keys=True, default=str)
        if signature not in seen:
            deduped.append(candidate)
            seen.add(signature)
    return deduped


def load_results_ledger(ledger_path: str | Path) -> list[dict[str, Any]]:
    path = Path(ledger_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_result_ledger(ledger_path: str | Path, result: dict[str, Any]) -> None:
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def find_incumbent_result(
    results: Sequence[dict[str, Any]],
    *,
    dataset_path: str,
    target_name: str,
    fidelity_tier: str,
) -> dict[str, Any] | None:
    incumbent: dict[str, Any] | None = None
    canonical_dataset_path = str(Path(dataset_path).expanduser().resolve())
    for row in results:
        if row.get("status") != "succeeded":
            continue
        if row.get("task_type") != "regression":
            continue
        if row.get("model_family") != "RandomForest":
            continue
        if row.get("dataset_path") != canonical_dataset_path:
            continue
        if row.get("target_name") != target_name:
            continue
        if row.get("fidelity_tier") != fidelity_tier:
            continue
        if incumbent is None:
            incumbent = row
            continue
        promoted, _ = should_promote(row.get("metrics", {}), incumbent)
        if promoted:
            incumbent = row
    return incumbent


def should_promote(candidate_metrics: dict[str, Any], incumbent: dict[str, Any] | None) -> tuple[bool, str]:
    candidate_rmse, candidate_spearman, candidate_r2 = _objective_tuple(candidate_metrics)
    if math.isinf(candidate_rmse):
        return False, "Candidate missing valid regression metrics required for promotion"
    if incumbent is None:
        return True, "No prior incumbent for dataset/target/fidelity; promoted as initial incumbent"

    incumbent_metrics = incumbent.get("metrics", {})
    incumbent_rmse, incumbent_spearman, incumbent_r2 = _objective_tuple(incumbent_metrics)
    incumbent_id = incumbent.get("experiment_id")
    tolerance = 1e-12

    if candidate_rmse < incumbent_rmse - tolerance:
        return True, (
            f"Promoted: {PRIMARY_METRIC} improved from {incumbent_rmse:.6f} "
            f"to {candidate_rmse:.6f} vs incumbent {incumbent_id}"
        )
    if abs(candidate_rmse - incumbent_rmse) <= tolerance and candidate_spearman > incumbent_spearman + tolerance:
        return True, (
            f"Promoted on tiebreak: {PRIMARY_METRIC} matched incumbent {incumbent_id} "
            f"and {SECONDARY_METRIC} improved from {incumbent_spearman:.6f} "
            f"to {candidate_spearman:.6f}"
        )
    if (
        abs(candidate_rmse - incumbent_rmse) <= tolerance
        and abs(candidate_spearman - incumbent_spearman) <= tolerance
        and candidate_r2 > incumbent_r2 + tolerance
    ):
        return True, (
            f"Promoted on tertiary tiebreak: {PRIMARY_METRIC} and {SECONDARY_METRIC} "
            f"matched incumbent {incumbent_id}, {TERTIARY_METRIC} improved from "
            f"{incumbent_r2:.6f} to {candidate_r2:.6f}"
        )
    return False, (
        f"Rejected: incumbent {incumbent_id} remains better on {PRIMARY_METRIC} "
        f"(candidate={candidate_rmse:.6f}, incumbent={incumbent_rmse:.6f})"
    )


def _objective_tuple(metrics: dict[str, Any]) -> tuple[float, float, float]:
    rmse = _safe_float(metrics.get(PRIMARY_METRIC), math.inf)
    spearman = _safe_float(metrics.get(SECONDARY_METRIC), -math.inf)
    r2 = _safe_float(metrics.get(TERTIARY_METRIC), -math.inf)
    return rmse, spearman, r2


def _safe_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(parsed):
        return fallback
    return parsed


def _make_experiment_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"rf-{stamp}-{uuid4().hex[:8]}"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _tier1_feature_columns() -> list[str]:
    from satnet.models.risk_model import TIER1_V1_FEATURE_COLUMNS

    return list(TIER1_V1_FEATURE_COLUMNS)


def _tier1_design_feature_columns() -> list[str]:
    from satnet.models.risk_model import TIER1_V1_DESIGN_FEATURE_COLUMNS

    return list(TIER1_V1_DESIGN_FEATURE_COLUMNS)


def _train_rf_once(cfg: RfExperimentConfig) -> tuple[Any, dict[str, Any], Any]:
    from satnet.models.risk_model import RiskModelConfig, train_rf_model

    return train_rf_model(
        csv_path=Path(cfg.dataset_path),
        target_name=cfg.target_name,
        feature_columns=cfg.resolved_feature_columns(),
        cfg=RiskModelConfig(
            random_state=cfg.seed,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
        ),
    )


def _save_model_artifact(model: Any, model_path: Path) -> None:
    from satnet.models.risk_model import save_model

    save_model(model, model_path)
