#!/usr/bin/env python3
"""Inspect smoke-run outputs for advisor/demo review.

This tool is intentionally read-only. It summarizes the dataset, target audit,
Random Forest classification metrics, regression metrics, feature importance,
and held-out predictions produced by the smoke workflow.

Usage:
    python tools/inspect_smoke_run.py all
    python tools/inspect_smoke_run.py dataset
    python tools/inspect_smoke_run.py classification
    python tools/inspect_smoke_run.py regression
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUNS_CSV = PROJECT_ROOT / "data" / "tier1_design_runs.csv"
DEFAULT_STEPS_CSV = PROJECT_ROOT / "data" / "tier1_design_steps.csv"
DEFAULT_TARGET_SUMMARY = PROJECT_ROOT / "artifacts" / "smoke" / "target_summary.csv"
DEFAULT_TARGET_CORR = PROJECT_ROOT / "artifacts" / "smoke" / "target_correlations.csv"
DEFAULT_RF_DIR = PROJECT_ROOT / "artifacts" / "smoke" / "rf"

CLASSIFICATION_STEM = "design_risk_model_tier1"
REGRESSION_STEM = "rf_gcc_frac_min"

RUN_COLUMNS = [
    "run_id",
    "num_planes",
    "sats_per_plane",
    "total_satellites",
    "node_failure_prob",
    "edge_failure_prob",
    "partition_any",
    "gcc_frac_min",
    "gcc_frac_mean",
]

STEP_COLUMNS = [
    "run_id",
    "t",
    "num_nodes",
    "num_edges",
    "num_components",
    "gcc_size",
    "gcc_frac",
    "partitioned",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect smoke-run outputs for advisor/demo review",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "section",
        choices=[
            "all",
            "files",
            "dataset",
            "targets",
            "classification",
            "regression",
            "features",
            "predictions",
        ],
        help="Inspection section to run",
    )
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=DEFAULT_RUNS_CSV,
        help="Path to tier1_design_runs.csv",
    )
    parser.add_argument(
        "--steps-csv",
        type=Path,
        default=DEFAULT_STEPS_CSV,
        help="Path to tier1_design_steps.csv",
    )
    parser.add_argument(
        "--target-summary",
        type=Path,
        default=DEFAULT_TARGET_SUMMARY,
        help="Path to target_summary.csv",
    )
    parser.add_argument(
        "--target-correlations",
        type=Path,
        default=DEFAULT_TARGET_CORR,
        help="Path to target_correlations.csv",
    )
    parser.add_argument(
        "--rf-dir",
        type=Path,
        default=DEFAULT_RF_DIR,
        help="Directory containing RF smoke artifacts",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of rows/features/predictions to display",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    require_file(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    require_file(path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def require_file(path: Path) -> None:
    if not path.exists():
        print(f"ERROR: required file not found: {path}")
        print("Run the smoke workflow first, then re-run this inspection.")
        sys.exit(1)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_num(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return f"{number:.4f}"


def format_table(rows: Iterable[dict[str, object]], columns: list[str]) -> str:
    row_list = list(rows)
    if not row_list:
        return "(no rows)"

    rendered_rows = [
        {col: fmt_num(row.get(col, "")) for col in columns}
        for row in row_list
    ]
    widths = {
        col: max(len(col), *(len(row[col]) for row in rendered_rows))
        for col in columns
    }
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    divider = "  ".join("-" * widths[col] for col in columns)
    body = [
        "  ".join(row[col].ljust(widths[col]) for col in columns)
        for row in rendered_rows
    ]
    return "\n".join([header, divider, *body])


def artifact_paths(rf_dir: Path) -> dict[str, Path]:
    return {
        "runs_csv": DEFAULT_RUNS_CSV,
        "steps_csv": DEFAULT_STEPS_CSV,
        "target_summary": DEFAULT_TARGET_SUMMARY,
        "target_correlations": DEFAULT_TARGET_CORR,
        "classifier_model": rf_dir / f"{CLASSIFICATION_STEM}.joblib",
        "classifier_metrics": rf_dir / f"{CLASSIFICATION_STEM}_metrics.json",
        "classifier_predictions": rf_dir / f"{CLASSIFICATION_STEM}_predictions.csv",
        "classifier_feature_importance": rf_dir / f"{CLASSIFICATION_STEM}_feature_importance.csv",
        "classifier_confusion_matrix": rf_dir / f"{CLASSIFICATION_STEM}_confusion_matrix.png",
        "classifier_prediction_plot": rf_dir / f"{CLASSIFICATION_STEM}_prediction_vs_actual.png",
        "regressor_model": rf_dir / f"{REGRESSION_STEM}.joblib",
        "regressor_metrics": rf_dir / f"{REGRESSION_STEM}_metrics.json",
        "regressor_predictions": rf_dir / f"{REGRESSION_STEM}_predictions.csv",
        "regressor_feature_importance": rf_dir / f"{REGRESSION_STEM}_feature_importance.csv",
        "regressor_prediction_plot": rf_dir / f"{REGRESSION_STEM}_prediction_vs_actual.png",
    }


def inspect_files(args: argparse.Namespace) -> None:
    section("Smoke Artifact Files")
    paths = artifact_paths(args.rf_dir)
    paths["runs_csv"] = args.runs_csv
    paths["steps_csv"] = args.steps_csv
    paths["target_summary"] = args.target_summary
    paths["target_correlations"] = args.target_correlations

    rows = []
    for name, path in paths.items():
        exists = path.exists()
        rows.append({
            "artifact": name,
            "exists": "yes" if exists else "NO",
            "size_bytes": path.stat().st_size if exists else "",
            "path": path,
        })
    print(format_table(rows, ["artifact", "exists", "size_bytes", "path"]))


def inspect_dataset(args: argparse.Namespace) -> None:
    section("Dataset Tables")
    runs = read_csv_rows(args.runs_csv)
    steps = read_csv_rows(args.steps_csv)

    step_counts = Counter(row.get("run_id", "") for row in steps)
    partition_counts = Counter(row.get("partition_any", "") for row in runs)
    timestep_counts = sorted(set(step_counts.values()))

    print(f"Run rows: {len(runs)}")
    print(f"Step rows: {len(steps)}")
    print(f"Step rows per run: {timestep_counts}")
    print(f"partition_any counts: {dict(partition_counts)}")
    if runs:
        partitioned = sum(1 for row in runs if row.get("partition_any") == "1")
        print(f"Partition probability: {partitioned / len(runs):.3f}")
    if runs and "gcc_frac_mean" in runs[0]:
        mean_gcc = sum(as_float(row.get("gcc_frac_mean")) for row in runs) / len(runs)
        print(f"Mean gcc_frac_mean: {mean_gcc:.3f}")

    print("\nFirst run-level rows:")
    print(format_table(runs[: args.top_n], RUN_COLUMNS))

    print("\nFirst timestep rows:")
    print(format_table(steps[: args.top_n], STEP_COLUMNS))


def inspect_targets(args: argparse.Namespace) -> None:
    section("Target Audit")
    summary = read_csv_rows(args.target_summary)
    target_columns = [
        "target",
        "count",
        "min",
        "max",
        "mean",
        "std",
        "zeros",
        "ones",
    ]
    print("Target summary:")
    print(format_table(summary, target_columns))

    if args.target_correlations.exists():
        corr_rows = read_csv_rows(args.target_correlations)
        if corr_rows:
            columns = list(corr_rows[0].keys())
            print("\nSpearman correlation matrix:")
            print(format_table(corr_rows, columns))

    print("\nHow to explain this:")
    print("- partition_any needs both zeros and ones to be trainable.")
    print("- Lower gcc_frac_min should correlate with higher partition risk.")
    print("- Smoke metrics are sanity checks, not dissertation-scale results.")


def metrics_path(args: argparse.Namespace, stem: str) -> Path:
    return args.rf_dir / f"{stem}_metrics.json"


def predictions_path(args: argparse.Namespace, stem: str) -> Path:
    return args.rf_dir / f"{stem}_predictions.csv"


def feature_path(args: argparse.Namespace, stem: str) -> Path:
    return args.rf_dir / f"{stem}_feature_importance.csv"


def inspect_classification(args: argparse.Namespace) -> None:
    section("Random Forest Classification")
    metrics = read_json(metrics_path(args, CLASSIFICATION_STEM))

    rows = [{
        "num_samples": metrics.get("num_samples"),
        "train_size": metrics.get("train_size"),
        "val_size": metrics.get("val_size"),
        "test_size": metrics.get("test_size"),
        "split_strategy": metrics.get("split_strategy"),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
    }]
    print(format_table(
        rows,
        [
            "num_samples",
            "train_size",
            "val_size",
            "test_size",
            "split_strategy",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ],
    ))

    print("\nConfusion matrix, read as [[TN, FP], [FN, TP]]:")
    for row in metrics.get("confusion_matrix", []):
        print(" ".join(str(value) for value in row))

    print("\nHow to explain this:")
    print("- Accuracy is overall correctness on the held-out test split.")
    print("- Precision asks: when the model predicts risky, how often is it right?")
    print("- Recall asks: of truly risky designs, how many did the model catch?")
    print("- The split strategy should be run_id_grouped to prevent timestep leakage.")


def inspect_regression(args: argparse.Namespace) -> None:
    section("Random Forest Regression")
    metrics = read_json(metrics_path(args, REGRESSION_STEM))
    rows = [{
        "num_samples": metrics.get("num_samples"),
        "train_size": metrics.get("train_size"),
        "val_size": metrics.get("val_size"),
        "test_size": metrics.get("test_size"),
        "split_strategy": metrics.get("split_strategy"),
        "test_mae": metrics.get("test_mae"),
        "test_rmse": metrics.get("test_rmse"),
        "test_r2": metrics.get("test_r2"),
        "spearman": metrics.get("test_spearman_rho"),
        "kendall": metrics.get("test_kendall_tau"),
    }]
    print(format_table(
        rows,
        [
            "num_samples",
            "train_size",
            "val_size",
            "test_size",
            "split_strategy",
            "test_mae",
            "test_rmse",
            "test_r2",
            "spearman",
            "kendall",
        ],
    ))

    print("\nHow to explain this:")
    print("- MAE is the average absolute miss in GCC-fraction units.")
    print("- RMSE is also in GCC-fraction units and punishes large misses more.")
    print("- R2 compares the model against predicting the mean every time.")
    print("- Spearman/Kendall indicate whether predicted design rankings are useful.")


def inspect_features(args: argparse.Namespace) -> None:
    section("Feature Importances")
    for label, stem in [
        ("classification partition_any", CLASSIFICATION_STEM),
        ("regression gcc_frac_min", REGRESSION_STEM),
    ]:
        path = feature_path(args, stem)
        rows = read_csv_rows(path)
        rows.sort(key=lambda row: as_float(row.get("importance")), reverse=True)
        print(f"\nTop features for {label}:")
        print(format_table(rows[: args.top_n], ["feature", "importance"]))

    print("\nHow to explain this:")
    print("- Feature importance is a Random Forest diagnostic, not causal proof.")
    print("- It helps identify which inputs the baseline used most heavily.")


def inspect_predictions(args: argparse.Namespace) -> None:
    section("Held-Out Test Predictions")
    for label, stem in [
        ("classification partition_any", CLASSIFICATION_STEM),
        ("regression gcc_frac_min", REGRESSION_STEM),
    ]:
        path = predictions_path(args, stem)
        rows = [row for row in read_csv_rows(path) if row.get("split") == "test"]
        print(f"\nTest predictions for {label}:")
        print(format_table(rows[: args.top_n], ["sample_idx", "run_id", "y_true", "y_pred"]))

    print("\nHow to explain this:")
    print("- y_true is the label computed from simulation.")
    print("- y_pred is the model output.")
    print("- Showing only split=test focuses on held-out evaluation.")


def inspect_all(args: argparse.Namespace) -> None:
    inspect_files(args)
    inspect_dataset(args)
    inspect_targets(args)
    inspect_classification(args)
    inspect_regression(args)
    inspect_features(args)
    inspect_predictions(args)


def main() -> None:
    args = parse_args()
    dispatch = {
        "all": inspect_all,
        "files": inspect_files,
        "dataset": inspect_dataset,
        "targets": inspect_targets,
        "classification": inspect_classification,
        "regression": inspect_regression,
        "features": inspect_features,
        "predictions": inspect_predictions,
    }
    dispatch[args.section](args)


if __name__ == "__main__":
    main()
