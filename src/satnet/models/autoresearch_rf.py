from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, NoReturn, Sequence
from uuid import uuid4

from satnet.metrics.resilience_targets import infer_task_type

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments" / "autoresearch"
DEFAULT_MUTATION_POLICY_PATH = DEFAULT_EXPERIMENTS_ROOT / "mutation_policy.json"
DEFAULT_CANDIDATE_PATH = DEFAULT_EXPERIMENTS_ROOT / "candidate.json"
DEFAULT_INCUMBENT_PATH = DEFAULT_EXPERIMENTS_ROOT / "incumbent.json"
DEFAULT_LAST_RUN_PATH = DEFAULT_EXPERIMENTS_ROOT / "last_run.json"
SUPPORTED_SEARCH_TARGETS = ("gcc_frac_min", "partition_fraction")
SUPPORTED_FEATURE_MODES = ("tier1_full", "design_only", "custom")
SUPPORTED_FIDELITY_TIERS = ("screening", "confirmatory")
PRIMARY_METRIC = "test_rmse"
SECONDARY_METRIC = "test_spearman_rho"
TERTIARY_METRIC = "test_r2"
INCUMBENT_SCHEMA_VERSION = "satnet_rf_autoresearch_incumbent_v1"
SUMMARY_SCHEMA_VERSION = "satnet_rf_autoresearch_summary_v1"
LAST_RUN_SCHEMA_VERSION = "satnet_rf_autoresearch_last_run_v1"
POLICY_SCHEMA_VERSION = "satnet_rf_autoresearch_policy_v1"
REQUIRED_MUTABLE_POLICY_FIELDS = ("target_name", "feature_mode", "n_estimators", "max_depth")
REQUIRED_FIXED_POLICY_FIELDS = (
    "dataset_path",
    "custom_feature_columns",
    "seed",
    "experiment_type",
    "notes",
    "save_model_artifact",
)
REQUIRED_INCUMBENT_ENTRY_FIELDS = (
    "scope_key",
    "current_best_experiment_id",
    "dataset_path",
    "target_name",
    "fidelity_tier",
    "metrics",
    "artifact_paths",
)


class RfAutoresearchCommandError(RuntimeError):
    def __init__(self, message: str, *, last_run_payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.last_run_payload = last_run_payload


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
        dataset_path = Path(self.dataset_path).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = (PROJECT_ROOT / dataset_path).resolve()
        else:
            dataset_path = dataset_path.resolve()
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
        unknown_keys = sorted(set(payload) - allowed_keys)
        if unknown_keys:
            raise ValueError(
                "Unsupported config fields: "
                + ", ".join(f"'{field_name}'" for field_name in unknown_keys)
            )
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
    command_mode: str | None = None,
) -> dict[str, Any]:
    cfg = config.validated()
    experiments_root_path = Path(experiments_root or DEFAULT_EXPERIMENTS_ROOT).expanduser().resolve()
    experiments_root_path.mkdir(parents=True, exist_ok=True)
    ledger_path = experiments_root_path / "results.jsonl"
    incumbent_path = experiments_root_path / DEFAULT_INCUMBENT_PATH.name
    last_run_path = experiments_root_path / DEFAULT_LAST_RUN_PATH.name

    experiment_id = _make_experiment_id()
    experiment_dir = experiments_root_path / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)

    config_path = experiment_dir / "config.json"
    metrics_path = experiment_dir / "metrics.json"
    predictions_path = experiment_dir / "predictions.csv"
    model_path = experiment_dir / "model.joblib"
    notes_path = experiment_dir / "notes.txt"
    summary_path = experiment_dir / "summary.json"

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
    incumbent_registry = load_incumbent_registry(incumbent_path)
    prior_incumbent = find_incumbent_entry(
        incumbent_registry,
        dataset_path=cfg.dataset_path,
        target_name=cfg.target_name,
        fidelity_tier=cfg.fidelity_tier,
    )
    ledger_incumbent = None
    if prior_incumbent is None:
        ledger_incumbent = find_incumbent_result(
            load_results_ledger(ledger_path),
            dataset_path=cfg.dataset_path,
            target_name=cfg.target_name,
            fidelity_tier=cfg.fidelity_tier,
        )
        if ledger_incumbent is not None:
            prior_incumbent = incumbent_entry_from_result(ledger_incumbent)

    artifact_paths: dict[str, str | None] = {
        "experiment_dir": str(experiment_dir),
        "config": str(config_path),
        "metrics": str(metrics_path),
        "predictions": None,
        "model": None,
        "notes": str(notes_path) if cfg.notes.strip() else None,
        "summary": str(summary_path),
        "incumbent": str(incumbent_path),
        "last_run": str(last_run_path),
        "ledger": str(ledger_path),
    }

    status = "succeeded"
    metrics: dict[str, Any]
    promotion_decision = "rejected"
    promotion_reason = "Not evaluated"
    incumbent_comparison = _build_incumbent_comparison(prior_incumbent, None)

    try:
        model, metrics, predictions = _train_rf_once(cfg)
        try:
            _write_json(metrics_path, metrics)
        except Exception as exc:
            _raise_artifact_write_error(metrics_path, "metrics artifact", exc)
        try:
            predictions.to_csv(predictions_path, index=False)
        except Exception as exc:
            _raise_artifact_write_error(predictions_path, "predictions artifact", exc)
        artifact_paths["predictions"] = str(predictions_path)
        if cfg.save_model_artifact:
            try:
                _save_model_artifact(model, model_path)
            except Exception as exc:
                _raise_artifact_write_error(model_path, "model artifact", exc)
            artifact_paths["model"] = str(model_path)

        incumbent_comparison = _build_incumbent_comparison(prior_incumbent, metrics)
        promoted, promotion_reason = should_promote(metrics, prior_incumbent)
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
        "incumbent_comparison": incumbent_comparison,
        "artifact_paths": artifact_paths,
        "promotion_decision": promotion_decision,
        "promotion_reason": promotion_reason,
        "runtime_seconds": runtime_seconds,
    }
    summary = build_run_summary(result)
    try:
        try:
            _write_json(summary_path, summary)
        except Exception as exc:
            _raise_artifact_write_error(summary_path, "summary artifact", exc)
        try:
            append_result_ledger(ledger_path, result)
        except Exception as exc:
            _raise_artifact_write_error(ledger_path, "results ledger", exc)
        if prior_incumbent is None and ledger_incumbent is not None:
            incumbent_registry = upsert_incumbent_entry(
                incumbent_registry,
                incumbent_entry_from_result(ledger_incumbent),
            )
        if status == "succeeded" and promotion_decision == "promoted":
            incumbent_registry = upsert_incumbent_entry(
                incumbent_registry,
                incumbent_entry_from_result(result),
            )
        if incumbent_registry.get("incumbents"):
            try:
                save_incumbent_registry(incumbent_path, incumbent_registry)
            except Exception as exc:
                _raise_artifact_write_error(incumbent_path, "incumbent registry", exc)
        try:
            _write_json(
                last_run_path,
                build_last_run_record(
                    result,
                    mode=command_mode,
                    summary_path=str(summary_path),
                    config_snapshot=cfg.to_dict(),
                ),
            )
        except Exception as exc:
            _raise_artifact_write_error(last_run_path, "last run record", exc)
    except Exception as exc:
        failure_reason = f"Post-run persistence failed with {exc.__class__.__name__}: {exc}"
        failure_payload = build_last_run_record(
            {
                **result,
                "status": "failed",
                "promotion_decision": "failed",
                "promotion_reason": failure_reason,
            },
            mode=command_mode,
            summary_path=str(summary_path) if summary_path.exists() else None,
            config_snapshot=cfg.to_dict(),
            error_type=exc.__class__.__name__,
            error_message=str(exc),
        )
        failure_payload["summary_path"] = str(summary_path) if summary_path.exists() else None
        _write_json(last_run_path, failure_payload)
        raise RfAutoresearchCommandError(
            failure_reason,
            last_run_payload=failure_payload,
        ) from exc
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
            command_mode="search",
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


def load_mutation_policy(policy_path: str | Path = DEFAULT_MUTATION_POLICY_PATH) -> dict[str, Any]:
    with open(policy_path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Mutation policy file must contain a JSON object")
    return payload


def validate_candidate_against_policy(
    config: RfExperimentConfig,
    policy: dict[str, Any],
    *,
    required_fidelity_tier: str,
) -> RfExperimentConfig:
    cfg = config.validated()
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected mutation policy schema '{policy.get('schema_version')}'. "
            f"Expected '{POLICY_SCHEMA_VERSION}'."
        )
    if cfg.fidelity_tier != required_fidelity_tier:
        raise ValueError(
            f"Candidate fidelity_tier must be '{required_fidelity_tier}', "
            f"got '{cfg.fidelity_tier}'."
        )
    mutable_fields = policy.get("mutable_fields", {})
    fixed_fields = policy.get("fixed_fields", {})
    missing_mutable_fields = [
        field_name for field_name in REQUIRED_MUTABLE_POLICY_FIELDS if field_name not in mutable_fields
    ]
    if missing_mutable_fields:
        raise ValueError(
            "Mutation policy missing required mutable_fields: "
            + ", ".join(f"'{field_name}'" for field_name in missing_mutable_fields)
        )
    missing_fixed_fields = [
        field_name for field_name in REQUIRED_FIXED_POLICY_FIELDS if field_name not in fixed_fields
    ]
    if missing_fixed_fields:
        raise ValueError(
            "Mutation policy missing required fixed_fields: "
            + ", ".join(f"'{field_name}'" for field_name in missing_fixed_fields)
        )
    if cfg.target_name not in tuple(mutable_fields.get("target_name", ())):
        raise ValueError(f"target_name '{cfg.target_name}' is outside the approved mutation surface")
    if cfg.feature_mode not in tuple(mutable_fields.get("feature_mode", ())):
        raise ValueError(f"feature_mode '{cfg.feature_mode}' is outside the approved mutation surface")
    if cfg.n_estimators not in tuple(mutable_fields.get("n_estimators", ())):
        raise ValueError(f"n_estimators '{cfg.n_estimators}' is outside the approved mutation surface")
    if cfg.max_depth not in tuple(mutable_fields.get("max_depth", ())):
        raise ValueError(f"max_depth '{cfg.max_depth}' is outside the approved mutation surface")
    for field_name, expected_value in fixed_fields.items():
        actual_value = getattr(cfg, field_name)
        if field_name == "dataset_path":
            actual_path = Path(str(actual_value)).expanduser()
            expected_path = Path(str(expected_value)).expanduser()
            if not actual_path.is_absolute():
                actual_path = (PROJECT_ROOT / actual_path).resolve()
            else:
                actual_path = actual_path.resolve()
            if not expected_path.is_absolute():
                expected_path = (PROJECT_ROOT / expected_path).resolve()
            else:
                expected_path = expected_path.resolve()
            actual_value = str(actual_path)
            expected_value = str(expected_path)
        if isinstance(actual_value, tuple) and isinstance(expected_value, list):
            expected_value = tuple(expected_value)
        if actual_value != expected_value:
            raise ValueError(
                f"{field_name} must remain fixed at '{expected_value}', got '{actual_value}'."
            )
    return cfg


def run_agent_candidate(
    *,
    candidate_path: str | Path = DEFAULT_CANDIDATE_PATH,
    policy_path: str | Path = DEFAULT_MUTATION_POLICY_PATH,
    experiments_root: str | Path | None = None,
) -> dict[str, Any]:
    policy = load_mutation_policy(policy_path)
    config = load_experiment_config(candidate_path)
    validated_config = validate_candidate_against_policy(
        config,
        policy,
        required_fidelity_tier="screening",
    )
    return run_rf_experiment(
        validated_config,
        experiments_root=experiments_root,
        command_mode="candidate",
    )


def run_confirmatory_experiment(
    source_experiment_id: str,
    *,
    experiments_root: str | Path | None = None,
    policy_path: str | Path = DEFAULT_MUTATION_POLICY_PATH,
) -> dict[str, Any]:
    experiments_root_path = Path(experiments_root or DEFAULT_EXPERIMENTS_ROOT).expanduser().resolve()
    source_summary_path = experiments_root_path / source_experiment_id / "summary.json"
    source_config_path = experiments_root_path / source_experiment_id / "config.json"
    with open(source_summary_path) as f:
        source_summary = json.load(f)
    if not isinstance(source_summary, dict):
        raise ValueError("Confirmatory source summary must contain a JSON object.")
    if source_summary.get("status") != "succeeded":
        raise ValueError("Confirmatory reruns require a succeeded screening source run.")
    if source_summary.get("promotion_decision") != "promoted":
        raise ValueError("Confirmatory reruns require a promoted screening source run.")
    if source_summary.get("fidelity_tier") != "screening":
        raise ValueError("Confirmatory reruns must originate from a screening summary.")
    source_config = load_experiment_config(source_config_path)
    if source_config.fidelity_tier != "screening":
        raise ValueError(
            "Confirmatory reruns must originate from a screening experiment config."
        )
    confirmatory_config = RfExperimentConfig.from_dict(
        {
            **source_config.to_dict(),
            "fidelity_tier": "confirmatory",
        }
    )
    policy = load_mutation_policy(policy_path)
    validated_config = validate_candidate_against_policy(
        confirmatory_config,
        policy,
        required_fidelity_tier="confirmatory",
    )
    return run_rf_experiment(
        validated_config,
        experiments_root=experiments_root_path,
        parent_experiment_id=source_experiment_id,
        command_mode="confirmatory",
    )


def should_promote(candidate_metrics: dict[str, Any], incumbent: dict[str, Any] | None) -> tuple[bool, str]:
    candidate_rmse, candidate_spearman, candidate_r2 = _objective_tuple(candidate_metrics)
    if math.isinf(candidate_rmse):
        return False, "Candidate missing valid regression metrics required for promotion"
    if incumbent is None:
        return True, "No prior incumbent for dataset/target/fidelity; promoted as initial incumbent"

    incumbent_metrics = incumbent.get("metrics", {})
    incumbent_rmse, incumbent_spearman, incumbent_r2 = _objective_tuple(incumbent_metrics)
    incumbent_id = incumbent.get("experiment_id") or incumbent.get("current_best_experiment_id")
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


def _raise_artifact_write_error(path: Path, artifact_label: str, exc: BaseException) -> NoReturn:
    raise exc.__class__(f"Failed writing {artifact_label} at {path}: {exc}") from exc


def build_run_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment_id": result.get("experiment_id"),
        "status": result.get("status"),
        "target_name": result.get("target_name"),
        "dataset_path": result.get("dataset_path"),
        "fidelity_tier": result.get("fidelity_tier"),
        "metrics": result.get("metrics", {}),
        "incumbent_comparison": result.get("incumbent_comparison", {}),
        "promotion_decision": result.get("promotion_decision"),
        "promotion_reason": result.get("promotion_reason"),
        "artifact_paths": result.get("artifact_paths", {}),
        "timestamp": result.get("timestamp"),
        "git_sha": result.get("git_sha"),
    }


def build_last_run_record(
    result: dict[str, Any],
    *,
    mode: str | None,
    summary_path: str | None = None,
    config_snapshot: dict[str, Any] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    artifact_paths = dict(result.get("artifact_paths", {}) or {})
    metrics_payload = dict(result.get("metrics", {}) or {})
    resolved_error_type = error_type
    if resolved_error_type is None and isinstance(metrics_payload.get("error_type"), str):
        resolved_error_type = metrics_payload["error_type"]
    resolved_error_message = error_message
    if resolved_error_message is None and isinstance(metrics_payload.get("error_message"), str):
        resolved_error_message = metrics_payload["error_message"]
    payload = {
        "schema_version": LAST_RUN_SCHEMA_VERSION,
        "mode": mode,
        "experiment_id": result.get("experiment_id"),
        "parent_experiment_id": result.get("parent_experiment_id"),
        "status": result.get("status"),
        "target_name": result.get("target_name"),
        "task_type": result.get("task_type"),
        "dataset_path": result.get("dataset_path"),
        "fidelity_tier": result.get("fidelity_tier"),
        "seed": result.get("seed"),
        "model_family": result.get("model_family"),
        "experiment_type": result.get("experiment_type"),
        "metrics": metrics_payload,
        "incumbent_comparison": dict(
            result.get("incumbent_comparison", {}) or _build_incumbent_comparison(None, None)
        ),
        "artifact_paths": artifact_paths,
        "promotion_decision": result.get("promotion_decision"),
        "promotion_reason": result.get("promotion_reason"),
        "summary_path": summary_path if summary_path is not None else artifact_paths.get("summary"),
        "incumbent_path": artifact_paths.get("incumbent"),
        "last_run_path": artifact_paths.get("last_run"),
        "timestamp": result.get("timestamp"),
        "git_sha": result.get("git_sha"),
        "config_snapshot": (
            dict(config_snapshot)
            if isinstance(config_snapshot, dict)
            else dict(result.get("config", {})) if isinstance(result.get("config"), dict) else None
        ),
        "error_type": resolved_error_type,
        "error_message": resolved_error_message,
    }
    return payload


def build_command_failure_record(
    *,
    experiments_root: str | Path,
    mode: str,
    exc: BaseException,
) -> dict[str, Any]:
    if isinstance(exc, RfAutoresearchCommandError):
        return dict(exc.last_run_payload)
    experiments_root_path = Path(experiments_root).expanduser().resolve()
    last_run_path = experiments_root_path / DEFAULT_LAST_RUN_PATH.name
    failure_result = {
        "experiment_id": None,
        "parent_experiment_id": None,
        "status": "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": None,
        "model_family": "RandomForest",
        "experiment_type": "rf_autoresearch_v1",
        "target_name": None,
        "task_type": None,
        "dataset_path": None,
        "fidelity_tier": _default_fidelity_tier_for_mode(mode),
        "seed": None,
        "config": {},
        "metrics": {},
        "incumbent_comparison": _build_incumbent_comparison(None, None),
        "artifact_paths": _default_artifact_paths(experiments_root_path, last_run_path=last_run_path),
        "promotion_decision": "failed",
        "promotion_reason": f"{exc.__class__.__name__}: {exc}",
    }
    payload = build_last_run_record(
        failure_result,
        mode=mode,
        summary_path=None,
        config_snapshot=None,
        error_type=exc.__class__.__name__,
        error_message=str(exc),
    )
    payload["summary_path"] = None
    return payload


def load_incumbent_registry(incumbent_path: str | Path = DEFAULT_INCUMBENT_PATH) -> dict[str, Any]:
    path = Path(incumbent_path)
    if not path.exists():
        return {
            "schema_version": INCUMBENT_SCHEMA_VERSION,
            "updated_at": None,
            "primary_metric_name": PRIMARY_METRIC,
            "secondary_metric_name": SECONDARY_METRIC,
            "tertiary_metric_name": TERTIARY_METRIC,
            "incumbents": [],
        }
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Incumbent registry must contain a JSON object")
    if payload.get("schema_version") != INCUMBENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected incumbent registry schema '{payload.get('schema_version')}'. "
            f"Expected '{INCUMBENT_SCHEMA_VERSION}'."
        )
    if "incumbents" not in payload or not isinstance(payload["incumbents"], list):
        raise ValueError("Incumbent registry must contain an 'incumbents' list")
    for index, entry in enumerate(payload["incumbents"]):
        if not isinstance(entry, dict):
            raise ValueError(f"Incumbent entry {index} must contain a JSON object")
        missing_fields = [
            field_name for field_name in REQUIRED_INCUMBENT_ENTRY_FIELDS if field_name not in entry
        ]
        if missing_fields:
            raise ValueError(
                f"Incumbent entry {index} missing required fields: "
                + ", ".join(f"'{field_name}'" for field_name in missing_fields)
            )
        if not isinstance(entry["metrics"], dict):
            raise ValueError(f"Incumbent entry {index} must contain a 'metrics' object")
        if not isinstance(entry["artifact_paths"], dict):
            raise ValueError(f"Incumbent entry {index} must contain an 'artifact_paths' object")
    return payload


def save_incumbent_registry(incumbent_path: str | Path, registry: dict[str, Any]) -> None:
    payload = {
        **registry,
        "schema_version": INCUMBENT_SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric_name": PRIMARY_METRIC,
        "secondary_metric_name": SECONDARY_METRIC,
        "tertiary_metric_name": TERTIARY_METRIC,
    }
    _write_json(Path(incumbent_path), payload)


def find_incumbent_entry(
    registry: dict[str, Any],
    *,
    dataset_path: str,
    target_name: str,
    fidelity_tier: str,
) -> dict[str, Any] | None:
    scope_key = _scope_key(dataset_path, target_name, fidelity_tier)
    for entry in registry.get("incumbents", []):
        if entry.get("scope_key") == scope_key:
            return entry
    return None


def upsert_incumbent_entry(registry: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    incumbents = list(registry.get("incumbents", []))
    scope_key = entry["scope_key"]
    filtered = [row for row in incumbents if row.get("scope_key") != scope_key]
    filtered.append(entry)
    return {
        **registry,
        "incumbents": sorted(filtered, key=lambda row: str(row.get("scope_key"))),
    }


def incumbent_entry_from_result(result: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(result.get("metrics", {}))
    dataset_path = str(Path(str(result["dataset_path"])).expanduser().resolve())
    target_name = str(result["target_name"])
    fidelity_tier = str(result["fidelity_tier"])
    return {
        "scope_key": _scope_key(dataset_path, target_name, fidelity_tier),
        "current_best_experiment_id": result.get("experiment_id"),
        "dataset_path": dataset_path,
        "target_name": target_name,
        "fidelity_tier": fidelity_tier,
        "primary_metric_name": PRIMARY_METRIC,
        "primary_metric_value": _safe_float(metrics.get(PRIMARY_METRIC), math.inf),
        "secondary_metric_name": SECONDARY_METRIC,
        "secondary_metric_value": _safe_float(metrics.get(SECONDARY_METRIC), -math.inf),
        "tertiary_metric_name": TERTIARY_METRIC,
        "tertiary_metric_value": _safe_float(metrics.get(TERTIARY_METRIC), -math.inf),
        "config_snapshot": result.get("config", {}),
        "artifact_paths": result.get("artifact_paths", {}),
        "timestamp": result.get("timestamp"),
        "git_sha": result.get("git_sha"),
        "metrics": metrics,
        "status": result.get("status"),
        "model_family": result.get("model_family"),
        "experiment_type": result.get("experiment_type"),
    }


def _build_incumbent_comparison(
    incumbent: dict[str, Any] | None,
    candidate_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    comparison = {
        "scope_key": None,
        "incumbent_experiment_id": None,
        "primary_metric_name": PRIMARY_METRIC,
        "primary_metric_candidate": None,
        "primary_metric_incumbent": None,
        "secondary_metric_name": SECONDARY_METRIC,
        "secondary_metric_candidate": None,
        "secondary_metric_incumbent": None,
        "tertiary_metric_name": TERTIARY_METRIC,
        "tertiary_metric_candidate": None,
        "tertiary_metric_incumbent": None,
        "incumbent_summary_path": None,
    }
    if candidate_metrics is not None:
        comparison["primary_metric_candidate"] = _optional_float(candidate_metrics.get(PRIMARY_METRIC))
        comparison["secondary_metric_candidate"] = _optional_float(
            candidate_metrics.get(SECONDARY_METRIC)
        )
        comparison["tertiary_metric_candidate"] = _optional_float(candidate_metrics.get(TERTIARY_METRIC))
    if incumbent is None:
        return comparison
    incumbent_metrics = incumbent.get("metrics", {})
    comparison["scope_key"] = incumbent.get("scope_key")
    comparison["incumbent_experiment_id"] = (
        incumbent.get("current_best_experiment_id") or incumbent.get("experiment_id")
    )
    comparison["primary_metric_incumbent"] = _optional_float(
        incumbent_metrics.get(PRIMARY_METRIC, incumbent.get("primary_metric_value"))
    )
    comparison["secondary_metric_incumbent"] = _optional_float(
        incumbent_metrics.get(SECONDARY_METRIC, incumbent.get("secondary_metric_value"))
    )
    comparison["tertiary_metric_incumbent"] = _optional_float(
        incumbent_metrics.get(TERTIARY_METRIC, incumbent.get("tertiary_metric_value"))
    )
    comparison["incumbent_summary_path"] = incumbent.get("artifact_paths", {}).get("summary")
    return comparison


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _scope_key(dataset_path: str, target_name: str, fidelity_tier: str) -> str:
    canonical_dataset_path = str(Path(dataset_path).expanduser().resolve())
    return f"{canonical_dataset_path}::{target_name}::{fidelity_tier}"


def _default_artifact_paths(
    experiments_root_path: Path,
    *,
    last_run_path: Path | None = None,
) -> dict[str, str | None]:
    resolved_last_run_path = last_run_path or experiments_root_path / DEFAULT_LAST_RUN_PATH.name
    return {
        "experiment_dir": None,
        "config": None,
        "metrics": None,
        "predictions": None,
        "model": None,
        "notes": None,
        "summary": None,
        "incumbent": str(experiments_root_path / DEFAULT_INCUMBENT_PATH.name),
        "last_run": str(resolved_last_run_path),
        "ledger": str(experiments_root_path / "results.jsonl"),
    }


def _default_fidelity_tier_for_mode(mode: str) -> str | None:
    if mode == "candidate":
        return "screening"
    if mode == "confirmatory":
        return "confirmatory"
    return None


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
