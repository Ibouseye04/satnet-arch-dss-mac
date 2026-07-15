from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satnet.metrics.resilience_targets import ALL_TARGETS, infer_task_type  # noqa: E402
from satnet.models.gnn_ablation import TGNN_INPUT_MODE_REGISTRY  # noqa: E402
from satnet.models.risk_model import RF_FEATURE_SET_REGISTRY  # noqa: E402
from satnet.utils.split_manifest import build_split_manifest, write_split_manifest  # noqa: E402


RF_CONDITIONS = ("full", "architecture_only", "no_geometry")
TGNN_CONDITIONS = ("full", "topology_only", "node_state_only")
DEFAULT_TARGETS = ("partition_any", "gcc_frac_min_original")


@dataclass(frozen=True)
class ExperimentSpec:
    model: str
    condition: str
    target: str
    output_dir: Path
    command: list[str]
    metrics_path: Path
    predictions_path: Path
    model_path: Path
    config_path: Path
    split_manifest_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SATNET RF/TGNN feature ablation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS), choices=sorted(ALL_TARGETS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="artifacts/ablation")
    parser.add_argument("--skip-rf", action="store_true", default=False)
    parser.add_argument("--skip-tgnn", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--cheb-k", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    return parser.parse_args()


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _target_slug(target: str) -> str:
    return target.replace(".", "_")


def _rf_base_name(target: str) -> str:
    if target == "partition_any":
        return "design_risk_model_tier1"
    return f"rf_{_target_slug(target)}"


def _tgnn_base_name(target: str, condition: str) -> str:
    return f"satellite_gnn_{_target_slug(target)}_{condition}"


def _smoke_subset(smoke: bool) -> int | None:
    return 8 if smoke else None


def _split_sizes(args: argparse.Namespace) -> tuple[float, float]:
    if args.smoke:
        return max(args.test_size, 0.25), max(args.val_size, 0.25)
    return args.test_size, args.val_size


def create_split_manifests(
    *,
    data_dir: Path,
    output_dir: Path,
    targets: list[str],
    seed: int,
    test_size: float,
    val_size: float,
    subset: int | None,
    overwrite: bool,
) -> dict[str, Path]:
    csv_path = data_dir / "tier1_design_runs.csv"
    validate_ablation_dataset_contract(csv_path)
    paths: dict[str, Path] = {}
    for target in targets:
        task_type = infer_task_type(target)
        manifest = build_split_manifest(
            csv_path=csv_path,
            target_name=target,
            task_type=task_type,
            seed=seed,
            test_size=test_size,
            val_size=val_size,
            subset=subset,
        )
        path = output_dir / "splits" / f"{_target_slug(target)}_split.json"
        write_split_manifest(manifest, path, overwrite=overwrite)
        paths[target] = path
    return paths


def validate_ablation_dataset_contract(csv_path: Path) -> None:
    df = pd_read_csv_header(csv_path)
    required_columns = {
        "config_hash",
        "failure_model",
        "isl_policy",
        "adjacent_search_k",
        "max_inter_plane_links_per_sat",
    }
    missing = sorted(column for column in required_columns if column not in df)
    if missing:
        raise ValueError(
            "Current dataset lacks required Tier 1 metadata columns for ablation: "
            f"{missing}"
        )


def pd_read_csv_header(csv_path: Path) -> list[str]:
    import pandas as pd

    return [str(c) for c in pd.read_csv(csv_path, nrows=0).columns.tolist()]


def build_experiment_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    data_dir = _resolve_path(args.data_dir)
    output_dir = _resolve_path(args.output_dir)
    test_size, val_size = _split_sizes(args)
    subset = _smoke_subset(args.smoke)
    split_paths = create_split_manifests(
        data_dir=data_dir,
        output_dir=output_dir,
        targets=list(args.targets),
        seed=args.seed,
        test_size=test_size,
        val_size=val_size,
        subset=subset,
        overwrite=args.overwrite,
    )

    specs: list[ExperimentSpec] = []
    if not args.skip_rf:
        for target in args.targets:
            for condition in RF_CONDITIONS:
                if condition not in RF_FEATURE_SET_REGISTRY:
                    raise ValueError(f"Unknown RF condition: {condition}")
                condition_dir = output_dir / "rf" / _target_slug(target) / condition
                base = _rf_base_name(target)
                metrics_path = condition_dir / f"{base}_metrics.json"
                preds_path = condition_dir / f"{base}_predictions.csv"
                model_path = condition_dir / f"{base}.joblib"
                config_path = condition_dir / f"{base}_config.json"
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_design_risk_model.py"),
                    "--data-path",
                    str(data_dir / "tier1_design_runs.csv"),
                    "--target-name",
                    target,
                    "--feature-set",
                    condition,
                    "--seed",
                    str(args.seed),
                    "--test-size",
                    str(test_size),
                    "--val-size",
                    str(val_size),
                    "--n-estimators",
                    str(min(args.rf_n_estimators, 25) if args.smoke else args.rf_n_estimators),
                    "--output-dir",
                    str(condition_dir),
                    "--split-manifest",
                    str(split_paths[target]),
                    "--config-output",
                    str(config_path),
                    "--experiment-log",
                    str(condition_dir / "experiment_log.jsonl"),
                    "--no-plots",
                ]
                if args.smoke:
                    command.append("--smoke")
                specs.append(
                    ExperimentSpec(
                        model="rf",
                        condition=condition,
                        target=target,
                        output_dir=condition_dir,
                        command=command,
                        metrics_path=metrics_path,
                        predictions_path=preds_path,
                        model_path=model_path,
                        config_path=config_path,
                        split_manifest_path=split_paths[target],
                    )
                )

    if not args.skip_tgnn:
        for target in args.targets:
            for condition in TGNN_CONDITIONS:
                if condition not in TGNN_INPUT_MODE_REGISTRY:
                    raise ValueError(f"Unknown TGNN condition: {condition}")
                condition_dir = output_dir / "tgnn" / _target_slug(target) / condition
                base = _tgnn_base_name(target, condition)
                model_path = condition_dir / f"{base}.pt"
                metrics_path = condition_dir / f"{base}_metrics.json"
                preds_path = condition_dir / f"{base}_predictions.csv"
                config_path = condition_dir / f"{base}_config.json"
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_gnn_model.py"),
                    "--data-dir",
                    str(data_dir),
                    "--target-name",
                    target,
                    "--input-mode",
                    condition,
                    "--seed",
                    str(args.seed),
                    "--test-split",
                    str(test_size),
                    "--val-split",
                    str(val_size),
                    "--epochs",
                    str(1 if args.smoke else args.epochs),
                    "--hidden-dim",
                    str(min(args.hidden_dim, 16) if args.smoke else args.hidden_dim),
                    "--cheb-k",
                    str(args.cheb_k),
                    "--device",
                    args.device,
                    "--output-model",
                    str(model_path),
                    "--metrics-output",
                    str(metrics_path),
                    "--config-output",
                    str(config_path),
                    "--split-manifest",
                    str(split_paths[target]),
                    "--experiment-log",
                    str(condition_dir / "experiment_log.jsonl"),
                ]
                if args.smoke:
                    command.append("--smoke")
                specs.append(
                    ExperimentSpec(
                        model="tgnn",
                        condition=condition,
                        target=target,
                        output_dir=condition_dir,
                        command=command,
                        metrics_path=metrics_path,
                        predictions_path=preds_path,
                        model_path=model_path,
                        config_path=config_path,
                        split_manifest_path=split_paths[target],
                    )
                )
    return specs


def write_run_manifest(*, specs: list[ExperimentSpec], args: argparse.Namespace, output_dir: Path) -> Path:
    payload = {
        "seed": args.seed,
        "targets": list(args.targets),
        "smoke": bool(args.smoke),
        "cheb_k": int(args.cheb_k),
        "output_dir": str(output_dir),
        "num_conditions": len(specs),
        "conditions": [
            {
                "model": spec.model,
                "condition": spec.condition,
                "target": spec.target,
                "output_dir": str(spec.output_dir),
                "metrics_path": str(spec.metrics_path),
                "predictions_path": str(spec.predictions_path),
                "model_path": str(spec.model_path),
                "config_path": str(spec.config_path),
                "split_manifest_path": str(spec.split_manifest_path),
                "command": spec.command,
            }
            for spec in specs
        ],
    }
    path = output_dir / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


def run_specs(specs: list[ExperimentSpec], *, overwrite: bool) -> None:
    for spec in specs:
        if spec.metrics_path.exists() and not overwrite:
            continue
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            spec.command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        (spec.output_dir / "stdout.log").write_text(result.stdout)
        (spec.output_dir / "stderr.log").write_text(result.stderr)
        (spec.output_dir / "invocation.json").write_text(
            json.dumps({"command": spec.command, "returncode": result.returncode}, indent=2)
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Ablation condition failed: {spec.model}/{spec.target}/{spec.condition}. "
                f"See {spec.output_dir}"
            )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _metric(metrics: dict[str, Any], *keys: str) -> Any:
    current: Any = metrics
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def collect_results(specs: list[ExperimentSpec], output_dir: Path) -> None:
    classification_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    metrics_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}

    for spec in specs:
        if not spec.metrics_path.exists():
            continue
        metrics = _read_json(spec.metrics_path)
        metrics_by_key[(spec.model, spec.target, spec.condition)] = metrics

    for spec in specs:
        metrics = metrics_by_key.get((spec.model, spec.target, spec.condition))
        if metrics is None:
            continue
        task_type = metrics.get("task_type", infer_task_type(spec.target))
        baseline = metrics_by_key.get((spec.model, spec.target, "full"), metrics)
        base_row = {
            "model": spec.model,
            "condition": spec.condition,
            "target": spec.target,
            "cheb_k": metrics.get("cheb_k"),
            "sample_count": metrics.get("num_samples"),
            "train_size": metrics.get("train_size"),
            "validation_size": metrics.get("val_size"),
            "test_size": metrics.get("test_size"),
        }
        if task_type == "classification":
            acc = metrics.get("accuracy", _metric(metrics, "test_metrics", "accuracy"))
            precision = metrics.get("precision", _metric(metrics, "test_metrics", "precision"))
            recall = metrics.get("recall", _metric(metrics, "test_metrics", "recall"))
            f1 = metrics.get("f1", _metric(metrics, "test_metrics", "f1"))
            base_acc = baseline.get("accuracy", _metric(baseline, "test_metrics", "accuracy"))
            base_f1 = baseline.get("f1", _metric(baseline, "test_metrics", "f1"))
            row = {
                **base_row,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "baseline_accuracy": base_acc,
                "baseline_f1": base_f1,
                "delta_accuracy": None if acc is None or base_acc is None else acc - base_acc,
                "delta_f1": None if f1 is None or base_f1 is None else f1 - base_f1,
            }
            classification_rows.append(row)
        else:
            mae = metrics.get("test_mae", _metric(metrics, "test_metrics", "mae"))
            rmse = metrics.get("test_rmse", _metric(metrics, "test_metrics", "rmse"))
            r2 = metrics.get("test_r2", _metric(metrics, "test_metrics", "r2"))
            spearman = metrics.get("test_spearman_rho", _metric(metrics, "test_metrics", "spearman"))
            kendall = metrics.get("test_kendall_tau", _metric(metrics, "test_metrics", "kendall"))
            base_mae = baseline.get("test_mae", _metric(baseline, "test_metrics", "mae"))
            base_rmse = baseline.get("test_rmse", _metric(baseline, "test_metrics", "rmse"))
            base_r2 = baseline.get("test_r2", _metric(baseline, "test_metrics", "r2"))
            base_spearman = baseline.get("test_spearman_rho", _metric(baseline, "test_metrics", "spearman"))
            base_kendall = baseline.get("test_kendall_tau", _metric(baseline, "test_metrics", "kendall"))
            row = {
                **base_row,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "spearman_rho": spearman,
                "kendall_tau": kendall,
                "baseline_mae": base_mae,
                "baseline_rmse": base_rmse,
                "baseline_r2": base_r2,
                "delta_mae": None if mae is None or base_mae is None else mae - base_mae,
                "delta_rmse": None if rmse is None or base_rmse is None else rmse - base_rmse,
                "delta_r2": None if r2 is None or base_r2 is None else r2 - base_r2,
                "delta_spearman": None if spearman is None or base_spearman is None else spearman - base_spearman,
                "delta_kendall": None if kendall is None or base_kendall is None else kendall - base_kendall,
            }
            regression_rows.append(row)

    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(comparison_dir / "classification_results.csv", classification_rows)
    _write_csv(comparison_dir / "regression_results.csv", regression_rows)
    _write_csv(comparison_dir / "all_results.csv", classification_rows + regression_rows)
    write_summary(comparison_dir / "summary.md", classification_rows, regression_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    classification_rows: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# SATNET Ablation Study Summary",
        "",
        "Delta convention: classification deltas are ablation - baseline; regression error deltas are ablation - baseline; R²/rank deltas are ablation - baseline.",
        "Positive delta_MAE/RMSE means worse. Negative delta_F1/accuracy/R²/rank means worse.",
        "",
        "Smoke-scale results, when present, are preliminary pipeline checks and not final performance evidence.",
        "",
        "Small degradation can indicate robustness, redundancy, weak use of removed information, or insufficient data. Large degradation can indicate important removed signal. Improvement can indicate noise, overfitting reduction, or sampling variance.",
        "",
        f"Classification rows: {len(classification_rows)}",
        f"Regression rows: {len(regression_rows)}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = _resolve_path(args.output_dir)
    specs = build_experiment_specs(args)
    write_run_manifest(specs=specs, args=args, output_dir=output_dir)
    if args.dry_run:
        print(json.dumps([spec.__dict__ | {"command": spec.command} for spec in specs], indent=2, default=str))
        return
    run_specs(specs, overwrite=args.overwrite)
    collect_results(specs, output_dir)


if __name__ == "__main__":
    main()
