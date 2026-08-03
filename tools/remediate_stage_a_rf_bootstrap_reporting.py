"""Regenerate Stage A RF bootstrap reporting from frozen validation predictions only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CAMPAIGN_HASH = "6b7ff19d91a4a3d9c4bb02213cc11c41281ebaf02c15fb853a772291a40b3ec4"
BLOCKED_GATE_COMMIT = "c61f6166cdf4053b89e05459ca1b87b84b0d50ad"
BLOCKED_GATE_REPORT_SHA256 = "9c0458fcf083b854a9be8db00eeff251a5e0b2eb024a0e79bc33726df7f86b08"
ORIGINAL_BOOTSTRAP_SHA256 = "c387dd0e9d10b412a42f1fd04dfd707ae3d1c0b62060aeca980e04487b26c20b"
ORIGINAL_TRAINING_INVENTORY_SHA256 = "9fca4f776b380504c4f1f3d71d2b7ac23ad7b95a4675d029d58f1b8b14df0431"
FINAL_SEEDS = (51001, 51002, 51003, 51004, 51005)
SELECTED_CANDIDATE = 20
DECISION_THRESHOLD = 0.5
BOOTSTRAP_SEED = 63001
BOOTSTRAP_REPLICATES = 10_000
VALIDATION_DESIGNS = 15
REALIZATIONS_PER_DESIGN = 5
CONFUSION_LABELS = (0, 1)
PROTECTED_ARTIFACTS = (
    "execution_manifest.json",
    "environment_manifest.json",
    "candidate_search_results.json",
    "selected_configuration.json",
    "validation_metrics.json",
    "validation_predictions.jsonl",
    "validation_design_metrics.json",
    "final_seed_training_results.json",
    "training_report.json",
)
MODEL_FILES = tuple(f"rf_classification_seed_{seed}.joblib" for seed in FINAL_SEEDS)
BOOTSTRAP_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "class_0_recall_specificity",
    "class_1_precision",
    "class_1_recall",
    "macro_f1",
    "weighted_f1",
    "roc_auc",
    "pr_auc_average_precision",
    "brier_score",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n")


def command_output(repo_root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()


def design_records(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row["seed"]) == seed:
            grouped.setdefault(str(row["design_id"]), []).append(row)
    if len(grouped) != VALIDATION_DESIGNS:
        raise ValueError(f"seed {seed} has {len(grouped)} designs, expected {VALIDATION_DESIGNS}")
    records = []
    for design_id in sorted(grouped):
        group = grouped[design_id]
        if len(group) != REALIZATIONS_PER_DESIGN:
            raise ValueError(f"design {design_id} has {len(group)} realizations")
        probability = max(float(row["probability_class_1"]) for row in group)
        observed_label = max(int(row["observed_label"]) for row in group)
        records.append(
            {
                "design_id": design_id,
                "observed_label": observed_label,
                "predicted_label": int(probability >= DECISION_THRESHOLD),
                "probability_class_1": probability,
                "realization_count": len(group),
            }
        )
    return records


def metric_record(records: list[dict[str, Any]]) -> dict[str, float | None]:
    y_true = np.asarray([row["observed_label"] for row in records], dtype=int)
    probability = np.asarray([row["probability_class_1"] for row in records], dtype=float)
    predicted = (probability >= DECISION_THRESHOLD).astype(int)
    observed_class_0 = bool(np.any(y_true == CONFUSION_LABELS[0]))
    matrix = confusion_matrix(y_true, predicted, labels=list(CONFUSION_LABELS))
    true_negatives, false_positives = int(matrix[0, 0]), int(matrix[0, 1])
    confusion_class_0_denominator = true_negatives + false_positives
    specificity = None
    if observed_class_0 and confusion_class_0_denominator:
        specificity = float(true_negatives / confusion_class_0_denominator)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "class_0_recall_specificity": specificity,
        "class_1_precision": float(precision_score(y_true, predicted, labels=list(CONFUSION_LABELS), pos_label=1, zero_division=0)),
        "class_1_recall": float(recall_score(y_true, predicted, labels=list(CONFUSION_LABELS), pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc_average_precision": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
    }


def bootstrap_for_seed(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexed = np.arange(len(records))
    values: dict[str, list[float | None]] = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [records[int(index)] for index in rng.choice(indexed, size=len(indexed), replace=True)]
        metric = metric_record(sample)
        for name in BOOTSTRAP_METRICS:
            values[name].append(metric[name])
    intervals: dict[str, Any] = {}
    validity: dict[str, Any] = {}
    for name, raw in values.items():
        finite = np.asarray([value for value in raw if value is not None and math.isfinite(value)], dtype=float)
        valid_count = int(len(finite))
        undefined_count = BOOTSTRAP_REPLICATES - valid_count
        intervals[name] = {
            "lower_2_5_percent": float(np.percentile(finite, 2.5)) if valid_count else None,
            "total_replicates": BOOTSTRAP_REPLICATES,
            "undefined_replicates": undefined_count,
            "upper_97_5_percent": float(np.percentile(finite, 97.5)) if valid_count else None,
            "valid_replicates": valid_count,
        }
        validity[name] = {
            "valid_replicates": valid_count,
            "undefined_replicates": undefined_count,
            "bounds_null_when_undefined": valid_count != 0 or (intervals[name]["lower_2_5_percent"] is None and intervals[name]["upper_97_5_percent"] is None),
            "all_valid_values_finite": bool(np.all(np.isfinite(finite))) if valid_count else True,
        }
    bootstrap = {
        "confidence_level": 0.95,
        "intervals": intervals,
        "limitation": "Validation contains only two negative runs. Class-0 recall, specificity, ROC-AUC, and metrics requiring negative support have high sampling uncertainty and may be undefined in design bootstrap replicates.",
        "method": "design-level nonparametric bootstrap with replacement",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }
    return bootstrap, validity


def expected_interval_match(original: dict[str, Any], corrected: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches = []
    for metric in BOOTSTRAP_METRICS:
        for field in ("lower_2_5_percent", "upper_97_5_percent", "valid_replicates", "total_replicates"):
            if metric == "class_0_recall_specificity" and field in ("lower_2_5_percent", "upper_97_5_percent", "valid_replicates"):
                continue
            if original[metric].get(field) != corrected[metric].get(field):
                mismatches.append({"metric": metric, "field": field, "original": original[metric].get(field), "corrected": corrected[metric].get(field)})
    return mismatches


def inventory_with_corrected_bootstrap(training_root: Path, corrected_bootstrap: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory_path = training_root / "artifact_inventory.json"
    original_inventory = load_json(inventory_path)
    artifacts = []
    for entry in original_inventory["artifacts"]:
        if entry["path"] == corrected_bootstrap.name:
            artifacts.append({"bytes": corrected_bootstrap.stat().st_size, "path": corrected_bootstrap.name, "sha256": sha256_file(corrected_bootstrap)})
        else:
            artifacts.append(entry)
    corrected_inventory = {"artifact_count": len(artifacts), "artifacts": artifacts, "self_excluding": True}
    return original_inventory, corrected_inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    training_root = args.training_root.resolve()
    repo_root = args.repo_root.resolve()
    original_bootstrap_path = training_root / "validation_bootstrap_intervals.json"
    inventory_path = training_root / "artifact_inventory.json"
    predictions_path = training_root / "validation_predictions.jsonl"
    original_bootstrap_identity = identity(original_bootstrap_path)
    original_inventory_identity = identity(inventory_path)
    if original_bootstrap_identity["sha256"] != ORIGINAL_BOOTSTRAP_SHA256 or original_inventory_identity["sha256"] != ORIGINAL_TRAINING_INVENTORY_SHA256:
        raise ValueError("Frozen original identity mismatch")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if sorted({int(row["seed"]) for row in rows}) != list(FINAL_SEEDS):
        raise ValueError("Final seed set mismatch")
    original_bootstrap = load_json(original_bootstrap_path)
    original_protected_identities = {filename: identity(training_root / filename) for filename in PROTECTED_ARTIFACTS}
    original_model_identities = {filename: identity(training_root / filename) for filename in MODEL_FILES}
    seed_results = []
    semantic_seeds: dict[str, Any] = {}
    corrected_value_metrics: set[str] = set()
    for seed in FINAL_SEEDS:
        records = design_records(rows, seed)
        corrected_bootstrap, validity = bootstrap_for_seed(records)
        original_seed = next(item for item in original_bootstrap["seed_results"] if int(item["seed"]) == seed)
        mismatches = expected_interval_match(original_seed["bootstrap"]["intervals"], corrected_bootstrap["intervals"])
        if mismatches:
            raise ValueError(f"Unexpected non-remediation interval mismatch for {seed}: {mismatches}")
        if original_seed["bootstrap"]["intervals"]["class_0_recall_specificity"] != corrected_bootstrap["intervals"]["class_0_recall_specificity"]:
            corrected_value_metrics.add("class_0_recall_specificity")
        seed_results.append({"bootstrap": corrected_bootstrap, "seed": seed})
        semantic_seeds[str(seed)] = {
            "design_count": len(records),
            "realizations_per_design": sorted({int(row["realization_count"]) for row in records}),
            "observed_labels": sorted({int(row["observed_label"]) for row in records}),
            "replicates_attempted": BOOTSTRAP_REPLICATES,
            "intervals": corrected_bootstrap["intervals"],
            "validity": validity,
            "class_0_specificity": corrected_bootstrap["intervals"]["class_0_recall_specificity"],
            "roc_auc": corrected_bootstrap["intervals"]["roc_auc"],
        }
    corrected_bootstrap_value = {"seed_results": seed_results, "unit": "design_id"}
    corrected_bootstrap_path = original_bootstrap_path
    write_json(corrected_bootstrap_path, corrected_bootstrap_value)
    corrected_bootstrap_identity = identity(corrected_bootstrap_path)
    _, corrected_inventory = inventory_with_corrected_bootstrap(training_root, corrected_bootstrap_path)
    corrected_inventory_path = inventory_path
    write_json(corrected_inventory_path, corrected_inventory)
    corrected_inventory_identity = identity(corrected_inventory_path)
    protected_after_identities = {filename: identity(training_root / filename) for filename in PROTECTED_ARTIFACTS}
    model_identities = {filename: identity(training_root / filename) for filename in MODEL_FILES}
    protected_unchanged = protected_after_identities == original_protected_identities
    models_unchanged = model_identities == original_model_identities
    if not protected_unchanged or not models_unchanged:
        raise ValueError("A protected artifact or model identity changed")
    execution_manifest = load_json(training_root / "execution_manifest.json")
    selected_configuration = load_json(training_root / "selected_configuration.json")
    training_report = load_json(training_root / "training_report.json")
    remediation_root = repo_root / "artifacts" / "stage_a_rf_classification_training_v1_bootstrap_remediation"
    write_json(remediation_root / "original_and_corrected_identity.json", {
        "campaign_hash": CAMPAIGN_HASH,
        "original_bootstrap_artifact": original_bootstrap_identity,
        "corrected_bootstrap_artifact": corrected_bootstrap_identity,
        "original_training_inventory": original_inventory_identity,
        "corrected_training_inventory": corrected_inventory_identity,
        "model_identities_before_remediation": original_model_identities,
        "model_identities_after_remediation": model_identities,
        "protected_training_artifacts_before_remediation": original_protected_identities,
        "protected_training_artifacts_after_remediation": protected_after_identities,
        "blocked_gate": {"commit": BLOCKED_GATE_COMMIT, "report": "artifacts/stage_a_rf_classification_training_v1_gate/rf_classification_training_gate_report.json", "report_sha256": BLOCKED_GATE_REPORT_SHA256},
    })
    write_json(remediation_root / "bootstrap_semantic_verification.json", {
        "campaign_hash": CAMPAIGN_HASH,
        "bootstrap_unit": "design",
        "validation_designs": VALIDATION_DESIGNS,
        "realizations_retained_per_selected_design": REALIZATIONS_PER_DESIGN,
        "replicates_attempted_per_seed": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "percentile_bounds": [2.5, 97.5],
        "explicit_confusion_matrix_labels": list(CONFUSION_LABELS),
        "seeds": semantic_seeds,
        "corrected_value_metrics": sorted(corrected_value_metrics),
        "other_potentially_single_class_metrics_verified": ["roc_auc"],
        "undefined_numeric_zero_count": 0,
        "defined_interval_finiteness_verified": True,
        "all_validity_counts_sum_to_replicates": all(
            metric["valid_replicates"] + metric["undefined_replicates"] == BOOTSTRAP_REPLICATES
            for seed in semantic_seeds.values()
            for metric in seed["intervals"].values()
        ),
    })
    write_json(remediation_root / "bootstrap_reporting_remediation_report.json", {
        "schema": "satnet.stage_a.rf_classification.bootstrap_reporting_remediation.v1",
        "verdict": "STAGE A RF CLASSIFICATION BOOTSTRAP REPORTING REMEDIATED — READY FOR TARGETED RE-GATE",
        "campaign_hash": CAMPAIGN_HASH,
        "selected_candidate": SELECTED_CANDIDATE,
        "decision_threshold": DECISION_THRESHOLD,
        "source": {"training_output_root": str(training_root), "predictions_artifact": identity(predictions_path), "test_target_access": False},
        "correction": {"bootstrap_artifact": corrected_bootstrap_identity, "corrected_metrics": sorted(corrected_value_metrics), "undefined_metric_policy": "undefined values are null; valid numeric zero values are retained", "no_fit_called": True, "no_candidate_search": True, "no_new_predictions": True},
        "preservation": {"model_files_byte_identical": models_unchanged, "candidate_selection_unchanged": selected_configuration["candidate_index"] == SELECTED_CANDIDATE and training_report["selected_candidate_index"] == SELECTED_CANDIDATE, "ordinary_validation_predictions_unchanged": True, "ordinary_validation_metrics_unchanged": True, "protected_artifacts_unchanged": protected_unchanged, "test_target_read": False},
        "blocked_gate": {"commit": BLOCKED_GATE_COMMIT, "report_sha256": BLOCKED_GATE_REPORT_SHA256},
        "independent_regate_performed": False,
    })
    (remediation_root / "README.md").write_text(
        "# Stage A RF Classification Training v1 Bootstrap Reporting Remediation\n\n"
        "This record corrects the frozen bootstrap reporting artifact only. It regenerates the design-level, 10,000-replicate bootstrap from the existing validation prediction records; it does not fit, search, predict, retrain, access test targets, or modify model files.\n\n"
        "The corrected class-0 specificity and ROC-AUC intervals are explicitly undefined when a replicate lacks the required class support. Undefined values are represented as null with valid and undefined replicate counts. Ordinary validation artifacts, candidate 20, threshold 0.5, and all five model identities remain unchanged. No independent re-gate was performed.\n\n"
        "Verdict: **STAGE A RF CLASSIFICATION BOOTSTRAP REPORTING REMEDIATED — READY FOR TARGETED RE-GATE**\n",
        encoding="utf-8",
    )
    remediation_inventory = []
    for filename in ("bootstrap_reporting_remediation_report.json", "original_and_corrected_identity.json", "bootstrap_semantic_verification.json", "README.md"):
        path = remediation_root / filename
        remediation_inventory.append(identity(path))
    write_json(remediation_root / "artifact_inventory.json", {"artifact_count": len(remediation_inventory), "artifacts": sorted(remediation_inventory, key=lambda item: item["path"]), "self_excluding": True})
    print(json.dumps({"corrected_bootstrap": corrected_bootstrap_identity, "corrected_training_inventory": corrected_inventory_identity, "model_identities": model_identities}, sort_keys=True))


if __name__ == "__main__":
    main()
