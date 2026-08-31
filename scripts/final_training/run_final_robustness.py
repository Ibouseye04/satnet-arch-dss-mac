"""Run SATNET Adaptive-v2 Phase-4 final five-seed robustness training.

This runner is deliberately separate from Phase 3.  It consumes only the frozen
Phase-3 selections and the adaptive-v2 TRAIN/VALIDATION payload.  TEST records
are treated as sealed metadata: they are never retained, materialized as model
inputs, evaluated, or used for checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
PHASE3_ROOT = Path(r"C:\Users\johns\external\satnet-10k-final-models-v2-adaptive")
DATASET_ROOT = Path(r"C:\Users\johns\external\satnet-10k-final-ml-datasets-v2-adaptive")
PAYLOAD_INVENTORY = Path(r"C:\Users\johns\external\satnet-10k-final-ml-datasets-v2-adaptive-reconciliation\phase2a_training_payload_inventory.json")
PLAN_ROOT = Path(r"C:\Users\johns\external\satnet-10k-training-test-plan-v1")
OUTPUT_ROOT = PHASE3_ROOT / "final_robustness"
HELDOUT_ROOT = PHASE3_ROOT / "heldout_test"

DATASET_BUNDLE_HASH = "1399f12e6ed7e159076028bc01971cf2274924fc55b43cca0a44219db8c978d1"
TRAINING_PAYLOAD_INVENTORY_HASH = "a9d199a674c20ca31b2a91c6172f371b504603a2e7e09da29c51cd06654c9473"
TRAINING_PAYLOAD_BUNDLE_HASH = "af91fd86702403c232757ab4b5e123b2fabc314b1c4ce26f7bd9a6a0d58d5167"
TRAINING_PLAN_BUNDLE_HASH = "e14ec5e5b2221aa2c9aa187a2baacc517e676f9dc748612c206eb1dc43788a0a"
TRAINING_PLAN_INVENTORY_HASH = "04604051ebc5f4ce659fe558f50ed35689665f7243328ab67a6831948c11f991"
CANDIDATE_SET_HASH = "8d13cbcae2c58c72c9c1ae9da5574423b866097e325bc987ae2bb8d7aae7aaf2"
RF_SELECTION_HASH = "bbdb71469bb88d85ce45720dbaee4e8056589d75e741113e601d52a9f704a2cb"
TGNN_SELECTION_HASH = "81e293f8660c00bf8de0dc55280135c2b116c41c637001a742ec626afdb6108b"
PHASE3_TOOLING_SHA = "9274ba721007c66f0b98aac7e50ba1e9b6b154b5"

FINAL_SEEDS = (42, 123, 456, 789, 2026)
PRIMARY_SEED = 42
TRAIN_COUNT = 7000
VALIDATION_COUNT = 1500
SEALED_TEST_COUNT = 1500
EXPECTED_TOTAL = 35
RF_TASKS = (
    "rf_integrated_classification",
    "rf_integrated_regression_mean",
    "rf_integrated_regression_min",
    "rf_space_classification",
    "rf_space_regression",
)
TGNN_TASKS = ("tgnn_space_classification", "tgnn_space_regression")
ALL_TASKS = RF_TASKS + TGNN_TASKS

SPACE_FEATURES = (
    "num_planes", "sats_per_plane", "altitude_km", "inclination_deg",
    "satellite_node_failure_probability", "satellite_edge_failure_probability",
)
INTEGRATED_FEATURES = SPACE_FEATURES + (
    "civilian_count", "government_count", "military_count",
    "ground_station_failure_probability",
)
RF_DATASETS = {
    "rf_integrated_classification": ("rf_integrated_classification/rf_integrated_classification.csv", INTEGRATED_FEATURES, "overall_threshold_breach_any"),
    "rf_integrated_regression_mean": ("rf_integrated_regression/rf_integrated_regression.csv", INTEGRATED_FEATURES, "failure_adjusted_overall_service_fraction_mean"),
    "rf_integrated_regression_min": ("rf_integrated_regression/rf_integrated_regression.csv", INTEGRATED_FEATURES, "failure_adjusted_overall_service_fraction_min"),
    "rf_space_classification": ("rf_space_classification/rf_space_classification.csv", SPACE_FEATURES, "space_threshold_breach_any"),
    "rf_space_regression": ("rf_space_regression/rf_space_regression.csv", SPACE_FEATURES, "space_gcc_fraction_original_min"),
}
TGNN_TARGETS = {
    "tgnn_space_classification": "tgnn_space_classification/tgnn_space_classification_target_manifest.jsonl",
    "tgnn_space_regression": "tgnn_space_regression/tgnn_space_regression_target_manifest.jsonl",
}
EXPECTED_RF_CONFIGS: dict[str, dict[str, Any]] = {
    "rf_integrated_classification": {"bootstrap": True, "class_weight": "balanced", "max_depth": 10, "max_features": 1.0, "min_samples_leaf": 1, "n_estimators": 300, "n_jobs": -1, "random_state": 42},
    "rf_integrated_regression_mean": {"bootstrap": True, "max_depth": 20, "max_features": 1.0, "min_samples_leaf": 5, "n_estimators": 600, "n_jobs": -1, "random_state": 42},
    "rf_integrated_regression_min": {"bootstrap": True, "max_depth": 20, "max_features": 1.0, "min_samples_leaf": 5, "n_estimators": 300, "n_jobs": -1, "random_state": 42},
    "rf_space_classification": {"bootstrap": True, "class_weight": "balanced", "max_depth": 10, "max_features": 1.0, "min_samples_leaf": 2, "n_estimators": 300, "n_jobs": -1, "random_state": 42},
    "rf_space_regression": {"bootstrap": True, "max_depth": 20, "max_features": 1.0, "min_samples_leaf": 5, "n_estimators": 300, "n_jobs": -1, "random_state": 42},
}
EXPECTED_RF_IDS = {
    "rf_integrated_classification": "rf_015",
    "rf_integrated_regression_mean": "rf_036",
    "rf_integrated_regression_min": "rf_018",
    "rf_space_classification": "rf_020",
    "rf_space_regression": "rf_018",
}
EXPECTED_TGNN_IDS = {
    "tgnn_space_classification": "tgnn_010",
    "tgnn_space_regression": "tgnn_012",
}
EXPECTED_TGNN_CONFIG = {
    "batch_size": 1, "cheb_k": None, "classification_loss": "CrossEntropyLoss",
    "dropout": None, "hidden_dim": 64, "learning_rate": 0.001, "max_epochs": 100,
    "num_layers": 1, "optimizer": "Adam", "regression_loss": "SmoothL1Loss", "weight_decay": 0.0,
}
METADATA_COLUMNS = ("run_id", "run_key", "design_id", "realization_id", "split")


@dataclass(frozen=True)
class RFData:
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    train_metadata: tuple[dict[str, str], ...]
    validation_metadata: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class SequenceRecord:
    run_id: int
    run_key: str
    design_id: str
    realization_id: str
    split: str
    sequence_artifact: str
    sequence_artifact_sha256: str


@dataclass(frozen=True)
class TGNNData:
    records: dict[int, SequenceRecord]
    targets: dict[int, float | int]
    train_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]


class Phase4Error(RuntimeError):
    """Raised when a frozen Phase-4 contract is violated."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase4Error(f"Cannot load JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise Phase4Error(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_binary_dump(path: Path, dump: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    dump(temporary)
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase4Error(message)


def git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def environment_versions() -> dict[str, str]:
    import importlib.metadata as metadata
    import pandas
    import sklearn
    import torch
    import torch_geometric

    def version(distribution: str) -> str:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return "not-installed"

    return {
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__, "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "torch_geometric_temporal_distribution": version("torch-geometric-temporal"),
        "device": "cpu", "torch_cuda_available": str(bool(torch.cuda.is_available())),
    }


def initialize_determinism(seed: int) -> dict[str, Any]:
    require(seed in FINAL_SEEDS, f"Seed {seed} is not in the frozen Phase-4 seed set")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch
    torch.manual_seed(seed)
    cuda = bool(torch.cuda.is_available())
    if cuda:
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {"python_random": seed, "numpy": seed, "pytorch_cpu": seed, "pytorch_cuda": seed if cuda else None, "torch_deterministic_algorithms": True, "ordering": "frozen run_id ascending", "sampler": "none"}


def _phase3_winners(path: Path, family: str) -> dict[str, dict[str, Any]]:
    selection = load_json(path)
    require(selection.get("candidate_set_hash") == CANDIDATE_SET_HASH, f"{family} selection candidate-set hash mismatch")
    tasks = selection.get("tasks")
    require(isinstance(tasks, dict), f"{family} selection has no task map")
    return tasks


def load_frozen_winners() -> dict[str, dict[str, Any]]:
    rf = _phase3_winners(PHASE3_ROOT / "summaries" / "rf_validation_selection.json", "RF")
    tgnn = _phase3_winners(PHASE3_ROOT / "summaries" / "tgnn_validation_selection.json", "TGNN")
    require(set(rf) == set(RF_TASKS), "Frozen RF selection task set mismatch")
    require(set(tgnn) == set(TGNN_TASKS), "Frozen TGNN selection task set mismatch")
    winners: dict[str, dict[str, Any]] = {}
    for task, expected_id in {**EXPECTED_RF_IDS, **EXPECTED_TGNN_IDS}.items():
        record = (rf if task in rf else tgnn)[task]
        require(record.get("winning_config_id") == expected_id, f"Frozen winner mismatch for {task}")
        require(record.get("winning_configuration") != record.get("runner_up", {}).get("configuration"), f"Runner-up substituted for {task}")
        config = dict(record.get("winning_configuration", {}))
        if task in EXPECTED_RF_CONFIGS:
            require(config == EXPECTED_RF_CONFIGS[task], f"RF configuration mismatch for {task}")
        else:
            expected = dict(EXPECTED_TGNN_CONFIG)
            expected["cheb_k"] = 2 if task.endswith("classification") else 3
            require(config == expected, f"TGNN configuration mismatch for {task}")
        winners[task] = {"winning_config_id": expected_id, "winning_configuration": config, "runner_up": record.get("runner_up")}
    # The actual TGNN selection file stores the frozen protocol at top level.
    tgnn_selection = load_json(PHASE3_ROOT / "summaries" / "tgnn_validation_selection.json")
    stopping = tgnn_selection.get("early_stopping", {})
    require(stopping == {"min_delta": 0.0, "monitor_classification": "balanced_accuracy", "monitor_regression": "mae", "patience": 10, "restore_best": True}, "TGNN early-stopping protocol mismatch")
    return winners


def _verify_training_plan() -> dict[str, Any]:
    inventory_path = PLAN_ROOT / "training_plan_inventory.json"
    require(sha256_file(inventory_path) == TRAINING_PLAN_INVENTORY_HASH, "Training-plan inventory SHA mismatch")
    inventory = load_json(inventory_path)
    require(inventory.get("bundle_sha256") == TRAINING_PLAN_BUNDLE_HASH, "Training-plan bundle hash mismatch")
    for artifact in inventory.get("artifacts", []):
        path = PLAN_ROOT / str(artifact["path"])
        require(path.is_file() and path.stat().st_size == int(artifact["bytes"]), f"Missing training-plan artifact: {path}")
        require(sha256_file(path) == artifact["sha256"], f"Training-plan artifact hash mismatch: {path}")
    seeds = load_json(PLAN_ROOT / "seed_registry.json")
    require(tuple(seeds.get("final_robustness_seeds", ())) == FINAL_SEEDS, "Frozen final seeds mismatch")
    require(seeds.get("primary_reporting_seed") == PRIMARY_SEED, "Primary seed mismatch")
    contract = load_json(PLAN_ROOT / "model_selection_contract.json")
    require(contract.get("final_phase", {}).get("seeds") == list(FINAL_SEEDS), "Final contract seed mismatch")
    return {"bundle_hash": TRAINING_PLAN_BUNDLE_HASH, "inventory_sha256": TRAINING_PLAN_INVENTORY_HASH, "final_seeds": list(FINAL_SEEDS), "accounting_discrepancy": {"historical_rf_total_final_fits": contract.get("rf", {}).get("total_final_fits"), "historical_rf_total_planned_rf_fits": contract.get("rf", {}).get("total_planned_rf_fits"), "scientific_rf_final_fits": 25, "scientific_tgnn_final_fits": 10, "scientific_total_final_fits": 35, "resolution": "Scientific task/seed contract is unambiguous; no 45-fit run."}}


def _verify_phase3() -> dict[str, Any]:
    manifest = load_json(PHASE3_ROOT / "manifests" / "phase3_manifest.json")
    conditions = (
        manifest.get("status") == "PASS", manifest.get("rf_expected") == 756, manifest.get("rf_completed") == 756, manifest.get("rf_failed") == 0,
        manifest.get("tgnn_expected") == 96, manifest.get("tgnn_completed") == 96, manifest.get("tgnn_failed") == 0,
        manifest.get("candidate_set_hash") == CANDIDATE_SET_HASH, manifest.get("selection_used_only_validation") is True,
        manifest.get("test_evaluation_performed") is False, manifest.get("test_metrics_calculated") is False,
        manifest.get("test_predictions_generated") is False, manifest.get("test_targets_loaded") is False,
        manifest.get("historical_fixed_data_reused") is False, manifest.get("historical_fixed_checkpoint_reused") is False,
    )
    require(all(conditions), "Phase-3 PASS/test-seal gate failed")
    firewall = load_json(PHASE3_ROOT / "manifests" / "test_firewall_evidence.json")
    require(firewall == {"candidate_manifests_checked": 852, "passed": True, "recorded_at": firewall.get("recorded_at"), "schema_version": "satnet.phase3.test_firewall_evidence.v1", "test_evaluation_performed": False, "test_metrics_present": False, "test_predictions_present": False, "test_targets_loaded": False}, "Phase-3 TEST firewall evidence failed")
    rf_freeze = load_json(PHASE3_ROOT / "manifests" / "rf_selection_freeze.json")
    tgnn_freeze = load_json(PHASE3_ROOT / "manifests" / "tgnn_selection_freeze.json")
    require(rf_freeze.get("sha256") == RF_SELECTION_HASH and rf_freeze.get("candidate_set_hash") == CANDIDATE_SET_HASH and rf_freeze.get("test_accessed") is False, "RF selection freeze evidence mismatch")
    require(tgnn_freeze.get("sha256") == TGNN_SELECTION_HASH and tgnn_freeze.get("candidate_set_hash") == CANDIDATE_SET_HASH and tgnn_freeze.get("test_accessed") is False, "TGNN selection freeze evidence mismatch")
    require(sha256_file(PHASE3_ROOT / "summaries" / "rf_validation_selection.json") == RF_SELECTION_HASH, "RF selection SHA mismatch")
    require(sha256_file(PHASE3_ROOT / "summaries" / "tgnn_validation_selection.json") == TGNN_SELECTION_HASH, "TGNN selection SHA mismatch")
    return {"status": "PASS", "rf_completed": 756, "tgnn_completed": 96, "failures": 0, "test_accessed": False}


def _verify_dataset_identity() -> dict[str, Any]:
    manifest = load_json(DATASET_ROOT / "phase2_manifest.json")
    require(manifest.get("status") == "PASS" and manifest.get("ml_bundle_hash") == DATASET_BUNDLE_HASH, "Adaptive-v2 dataset identity mismatch")
    require(manifest.get("counts", {}).get("split_runs") == {"test": 1500, "train": 7000, "validation": 1500}, "Adaptive-v2 split run counts mismatch")
    require(manifest.get("counts", {}).get("split_designs") == {"test": 300, "train": 1400, "validation": 300}, "Adaptive-v2 split design counts mismatch")
    topology = manifest.get("topology", {})
    require(topology.get("grid_fixed_source_count") == 0, "Historical grid_fixed source contamination detected")
    tuples = topology.get("aggregate", [])
    require(len(tuples) == 1 and tuples[0].get("tuple") == ["grid_adaptive", 1, 1, "persistent_temporal_union_edges_v1"], "Adaptive topology identity mismatch")
    gates = load_json(DATASET_ROOT / "metadata" / "validation_gates.json")
    require(gates.get("adaptive_contract", {}).get("contract_bundle_hash") == "da3c73711b1d60635afcceee8bda0a60d1379e492e1ad0d588d5a0c25e10abe3", "Adaptive contract identity mismatch")
    require(gates.get("historical_root_immutability", {}).get("unchanged") is True, "Historical dataset root changed")
    payload = load_json(PAYLOAD_INVENTORY)
    require(sha256_file(PAYLOAD_INVENTORY) == TRAINING_PAYLOAD_INVENTORY_HASH, "Training-payload inventory SHA mismatch")
    require(payload.get("bundle_sha256") == TRAINING_PAYLOAD_BUNDLE_HASH and payload.get("root") == str(DATASET_ROOT), "Training-payload identity mismatch")
    require(len(payload.get("artifacts", [])) == 10019, "Training-payload inventory cardinality mismatch")
    # Only metadata and file existence are checked here. Rehashing the complete
    # bundle would read sealed TEST payload bytes, which Phase 4 is forbidden to do.
    return {"dataset_bundle_hash": DATASET_BUNDLE_HASH, "training_payload_inventory_hash": TRAINING_PAYLOAD_INVENTORY_HASH, "training_payload_bundle_hash": TRAINING_PAYLOAD_BUNDLE_HASH, "test_payload_rehash": False, "grid_fixed_source_count": 0, "adaptive_topology": True}


def preflight() -> dict[str, Any]:
    require(not HELDOUT_ROOT.exists(), f"Held-out output root already exists: {HELDOUT_ROOT}")
    phase3 = _verify_phase3()
    dataset = _verify_dataset_identity()
    plan = _verify_training_plan()
    require(tuple(FINAL_SEEDS) == (42, 123, 456, 789, 2026), "Robustness seed literal changed")
    require(len(RF_TASKS) * len(FINAL_SEEDS) == 25 and len(TGNN_TASKS) * len(FINAL_SEEDS) == 10 and EXPECTED_TOTAL == 35, "Phase-4 fit count contract failed")
    winners = load_frozen_winners()
    evidence = {"schema_version": "satnet.phase4.preflight_evidence.v1", "completed_at": now(), "phase3": phase3, "dataset": dataset, "training_plan": plan, "seeds": list(FINAL_SEEDS), "primary_reporting_seed": PRIMARY_SEED, "rf_expected": 25, "tgnn_expected": 10, "total_expected": EXPECTED_TOTAL, "selected_winners": {task: {"config_id": v["winning_config_id"], "configuration": v["winning_configuration"]} for task, v in winners.items()}, "exact_ties": {"rf_integrated_classification": ["rf_015", "rf_019"], "tgnn_space_classification": ["tgnn_010", "tgnn_009"], "tgnn_space_regression": ["tgnn_012", "tgnn_011"]}, "tie_resolution": "exact validation ties resolved by the pre-frozen deterministic selection implementation", "test_accessed": False, "test_targets_loaded": False, "test_feature_rows_loaded": False, "heldout_output_root_created": False, "phase3_files_immutable": True, "implementation_git_sha": git_head()}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT_ROOT / "preflight_evidence.json", evidence)
    atomic_json(OUTPUT_ROOT / "phase4_execution_identity.json", {"schema_version": "satnet.phase4.execution_identity.v1", "phase3_root": str(PHASE3_ROOT), "dataset_root": str(DATASET_ROOT), "plan_root": str(PLAN_ROOT), "output_root": str(OUTPUT_ROOT), "phase3_candidate_set_hash": CANDIDATE_SET_HASH, "rf_selection_freeze_hash": RF_SELECTION_HASH, "tgnn_selection_freeze_hash": TGNN_SELECTION_HASH, "phase3_tooling_sha": PHASE3_TOOLING_SHA, "phase4_tooling_git_sha": git_head(), "test_accessed": False})
    return {"winners": winners, "evidence": evidence}


def _parse_classification(value: str) -> int:
    text = value.strip().lower()
    if text in {"true", "1"}: return 1
    if text in {"false", "0"}: return 0
    raise Phase4Error(f"Invalid classification target {value!r}")


def load_rf_data(task: str) -> RFData:
    relative, features, target = RF_DATASETS[task]
    csv_path = DATASET_ROOT / relative
    manifest = load_json(csv_path.with_name(csv_path.stem + "_manifest.json"))
    schema = load_json(csv_path.with_name(csv_path.stem + "_schema.json"))
    require(tuple(manifest.get("predictors", [])) == features and tuple(item["field"] for item in schema.get("predictors", [])) == features, f"RF feature contract mismatch for {task}")
    require(target in manifest.get("targets", []) and csv_path.is_file(), f"RF target/file contract mismatch for {task}")
    train_x: list[list[float]] = []; train_y: list[Any] = []; val_x: list[list[float]] = []; val_y: list[Any] = []
    train_meta: list[dict[str, str]] = []; val_meta: list[dict[str, str]] = []
    counts = {"train": 0, "validation": 0, "test": 0}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == METADATA_COLUMNS + features + tuple(manifest.get("targets", [])), f"RF CSV schema mismatch for {task}")
        for row in reader:
            split = row.get("split", "").strip().lower()
            require(split in counts, f"Unknown RF split {split!r}")
            counts[split] += 1
            if split == "test":
                # The row is not converted into a feature/target structure.
                continue
            values = [float(row[field]) for field in features]
            value: Any = _parse_classification(row[target]) if "classification" in task else float(row[target])
            metadata = {field: row[field] for field in METADATA_COLUMNS}
            (train_x, train_y, train_meta) if split == "train" else (val_x, val_y, val_meta)
            if split == "train": train_x.append(values); train_y.append(value); train_meta.append(metadata)
            else: val_x.append(values); val_y.append(value); val_meta.append(metadata)
    require(counts == {"train": TRAIN_COUNT, "validation": VALIDATION_COUNT, "test": SEALED_TEST_COUNT}, f"RF split counts mismatch for {task}: {counts}")
    _validate_design_groups(train_meta, val_meta)
    return RFData(np.asarray(train_x, dtype=float), np.asarray(train_y), np.asarray(val_x, dtype=float), np.asarray(val_y), tuple(train_meta), tuple(val_meta))


def _validate_design_groups(*metadata_groups: list[dict[str, str]]) -> None:
    by_design: dict[str, set[str]] = {}
    for group in metadata_groups:
        for row in group:
            by_design.setdefault(row["design_id"], set()).add(row["split"])
    require(all(len(splits) == 1 for splits in by_design.values()), "Design crosses TRAIN/VALIDATION")
    for group in metadata_groups:
        counts: dict[str, int] = {}
        for row in group: counts[row["design_id"]] = counts.get(row["design_id"], 0) + 1
        require(all(count == 5 for count in counts.values()), "All five realizations must remain together")


def classification_metrics(y_true: Any, y_pred: Any, scores: Any) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    true = np.asarray(y_true, dtype=int); pred = np.asarray(y_pred, dtype=int); score = np.asarray(scores, dtype=float)[:, 1]
    matrix = confusion_matrix(true, pred, labels=[0, 1]); tn, fp, fn, tp = [int(v) for v in matrix.ravel()]
    return {"balanced_accuracy": float(balanced_accuracy_score(true, pred)), "accuracy": float(accuracy_score(true, pred)), "macro_f1": float(f1_score(true, pred, labels=[0, 1], average="macro", zero_division=0)), "weighted_f1": float(f1_score(true, pred, labels=[0, 1], average="weighted", zero_division=0)), "precision_by_class": {str(k): float(v) for k, v in zip([0, 1], precision_score(true, pred, labels=[0, 1], average=None, zero_division=0))}, "recall_by_class": {str(k): float(v) for k, v in zip([0, 1], recall_score(true, pred, labels=[0, 1], average=None, zero_division=0))}, "f1_by_class": {str(k): float(v) for k, v in zip([0, 1], f1_score(true, pred, labels=[0, 1], average=None, zero_division=0))}, "specificity": float(tn / (tn + fp)) if tn + fp else 0.0, "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0, "confusion_matrix": matrix.tolist(), "roc_auc": float(roc_auc_score(true, score)), "pr_auc": float(average_precision_score(true, score)), "positive_class_count": int(np.sum(true == 1)), "negative_class_count": int(np.sum(true == 0))}


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
    true = np.asarray(y_true, dtype=float); pred = np.asarray(y_pred, dtype=float); error = np.abs(true - pred)
    return {"mae": float(mean_absolute_error(true, pred)), "rmse": float(np.sqrt(mean_squared_error(true, pred))), "r2": float(r2_score(true, pred)), "median_absolute_error": float(median_absolute_error(true, pred)), "maximum_absolute_error": float(np.max(error)), "target_mean": float(np.mean(true)), "target_median": float(np.median(true)), "target_standard_deviation": float(np.std(true)), "prediction_mean": float(np.mean(pred)), "prediction_standard_deviation": float(np.std(pred)), "predictions_below_zero": int(np.sum(pred < 0)), "predictions_above_one": int(np.sum(pred > 1))}


def _read_jsonl_split(path: Path, *, include_targets: bool, task: str) -> tuple[list[dict[str, Any]], dict[int, Any], int]:
    records: list[dict[str, Any]] = []; targets: dict[int, Any] = {}; test_count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = str(row.get("split", "")).lower()
            require(split in {"train", "validation", "test"}, f"Unknown TGNN split {split!r}")
            if split == "test":
                test_count += 1
                continue
            records.append(row)
            if include_targets:
                targets[int(row["run_id"])] = _parse_classification(str(row["target"])) if "classification" in task else float(row["target"])
    return records, targets, test_count


def load_tgnn_data(task: str) -> TGNNData:
    graph_path = DATASET_ROOT / "tgnn_space_classification" / "tgnn_space_graph_manifest.jsonl"
    target_path = DATASET_ROOT / TGNN_TARGETS[task]
    graph_rows, _, graph_test = _read_jsonl_split(graph_path, include_targets=False, task=task)
    target_rows, targets, target_test = _read_jsonl_split(target_path, include_targets=True, task=task)
    require(len(graph_rows) == TRAIN_COUNT + VALIDATION_COUNT and len(target_rows) == TRAIN_COUNT + VALIDATION_COUNT, f"TGNN train/validation manifest count mismatch for {task}")
    require(graph_test == SEALED_TEST_COUNT and target_test == SEALED_TEST_COUNT, f"TGNN sealed TEST count mismatch for {task}")
    records = {int(row["run_id"]): SequenceRecord(int(row["run_id"]), row["run_key"], row["design_id"], row["realization_id"], row["split"], row["sequence_artifact"], row["sequence_artifact_sha256"]) for row in graph_rows}
    target_ids = {int(row["run_id"]) for row in target_rows}
    require(set(records) == target_ids == set(targets), f"TGNN graph/target identity mismatch for {task}")
    train_ids = tuple(sorted(run_id for run_id, record in records.items() if record.split == "train")); val_ids = tuple(sorted(run_id for run_id, record in records.items() if record.split == "validation"))
    require(len(train_ids) == TRAIN_COUNT and len(val_ids) == VALIDATION_COUNT, f"TGNN split counts mismatch for {task}")
    return TGNNData(records, targets, train_ids, val_ids)


MAGIC = b"SATNET-TGNN-V1\x00"
ARRAY_ORDER = ("node_features", "node_identity_index", "edge_index", "edge_attr", "snapshot_node_offsets", "snapshot_edge_offsets", "timestep_index")


def load_sequence(path: Path, expected: SequenceRecord) -> list[Any]:
    import torch
    from torch_geometric.data import Data
    raw = path.read_bytes(); require(raw.startswith(MAGIC), f"Malformed TGNN sequence: {path}")
    cursor = len(MAGIC); header_len = struct.unpack_from("<Q", raw, cursor)[0]; cursor += 8
    header = json.loads(raw[cursor:cursor + header_len].decode("utf-8")); cursor += header_len
    identity = header.get("sequence", {})
    for key in ("run_id", "run_key", "design_id", "realization_id", "split"): require(identity.get(key) == getattr(expected, key), f"TGNN sequence identity mismatch: {path} {key}")
    arrays: dict[str, np.ndarray] = {}
    for name in ARRAY_ORDER:
        size = struct.unpack_from("<Q", raw, cursor)[0]; cursor += 8
        arrays[name] = np.load(io.BytesIO(raw[cursor:cursor + size]), allow_pickle=False); cursor += size
    require(cursor == len(raw) and header.get("sequence_length") == 11 and arrays["node_features"].shape[1] == 3 and arrays["edge_attr"].shape[1] == 4, f"TGNN sequence format mismatch: {path}")
    result = []
    for index in range(11):
        n0, n1 = [int(v) for v in arrays["snapshot_node_offsets"][index:index + 2]]; e0, e1 = [int(v) for v in arrays["snapshot_edge_offsets"][index:index + 2]]
        x = torch.as_tensor(arrays["node_features"][n0:n1], dtype=torch.float32); edge_index = torch.as_tensor(arrays["edge_index"][:, e0:e1], dtype=torch.long); edge_attr = torch.as_tensor(arrays["edge_attr"][e0:e1], dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr); data.edge_weight = edge_attr[:, 0]; data.timestep_index = torch.tensor([index], dtype=torch.long); result.append(data)
    return result


def _tgnn_forward(model: Any, data: TGNNData, run_id: int, device: Any) -> tuple[float, np.ndarray]:
    record = data.records[run_id]; sequence = load_sequence(DATASET_ROOT / "tgnn_space_classification" / "sequences" / Path(record.sequence_artifact).name, record)
    sequence = [item.to(device) for item in sequence]; output = model(sequence)
    return float(data.targets[run_id]), output


def _verified_completed(task: str, seed: int, config_id: str, selection_hash: str) -> dict[str, Any] | None:
    path = OUTPUT_ROOT / task / f"seed_{seed}" / "final_manifest.json"
    if not path.is_file(): return None
    manifest = load_json(path)
    if manifest.get("status") != "completed": return None
    require(manifest.get("task") == task and manifest.get("seed") == seed and manifest.get("selected_config_id") == config_id and manifest.get("selection_freeze_sha256") == selection_hash, f"Completed artifact identity mismatch: {path}")
    require(manifest.get("test_accessed") is False and manifest.get("dataset_bundle_hash") == DATASET_BUNDLE_HASH and manifest.get("training_plan_bundle_hash") == TRAINING_PLAN_BUNDLE_HASH, f"Completed artifact provenance mismatch: {path}")
    artifact_key = "model_path" if task.startswith("rf_") else "checkpoint_path"; hash_key = "model_sha256" if task.startswith("rf_") else "checkpoint_sha256"; artifact = Path(manifest[artifact_key])
    require(artifact.is_file() and sha256_file(artifact) == manifest.get(hash_key), f"Completed artifact hash mismatch: {artifact}")
    return manifest


def _running_manifest(task: str, seed: int, config_id: str, config: dict[str, Any], family: str, selection_hash: str) -> Path:
    path = OUTPUT_ROOT / task / f"seed_{seed}" / "final_manifest.json"; path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, {"schema_version": "satnet.phase4.final_fit_manifest.v1", "status": "running", "started_at": now(), "task": task, "model_family": family, "selected_config_id": config_id, "configuration": config, "seed": seed, "dataset_bundle_hash": DATASET_BUNDLE_HASH, "training_payload_inventory_hash": TRAINING_PAYLOAD_INVENTORY_HASH, "training_payload_bundle_hash": TRAINING_PAYLOAD_BUNDLE_HASH, "training_plan_bundle_hash": TRAINING_PLAN_BUNDLE_HASH, "phase3_candidate_set_hash": CANDIDATE_SET_HASH, "selection_freeze_sha256": selection_hash, "train_count": TRAIN_COUNT, "validation_count": VALIDATION_COUNT, "test_accessed": False})
    return path


def runtime_rf_configuration(task: str, winner: dict[str, Any], seed: int) -> dict[str, Any]:
    require(task in RF_TASKS and seed in FINAL_SEEDS, "Invalid RF robustness task or seed")
    config = dict(winner["winning_configuration"])
    config["random_state"] = seed
    config["n_jobs"] = -1
    return config


def reset_attempt_log(path: Path) -> None:
    """Start a non-completed TGNN attempt with a clean per-fit epoch log."""
    if path.exists():
        path.unlink()


def run_rf(task: str, seed: int, winner: dict[str, Any], phase4_sha: str) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    config_id = winner["winning_config_id"]; selection_hash = RF_SELECTION_HASH
    completed = _verified_completed(task, seed, config_id, selection_hash)
    if completed is not None: return completed
    data = load_rf_data(task); initialize_determinism(seed); config = runtime_rf_configuration(task, winner, seed)
    estimator = RandomForestClassifier(**config) if "classification" in task else RandomForestRegressor(**{k: v for k, v in config.items() if k != "class_weight"})
    manifest_path = _running_manifest(task, seed, config_id, config, "RF", selection_hash); start = time.perf_counter()
    try:
        estimator.fit(data.train_x, data.train_y)
        pred = estimator.predict(data.validation_x); scores = estimator.predict_proba(data.validation_x) if "classification" in task else None
        metrics = classification_metrics(data.validation_y, pred, scores) if scores is not None else regression_metrics(data.validation_y, pred)
        model_path = manifest_path.parent / "final_model.joblib"
        import joblib
        atomic_binary_dump(model_path, lambda path: joblib.dump(estimator, path))
        result = _fit_manifest(task, "RF", config_id, config, seed, selection_hash, phase4_sha, metrics, None, None, model_path, time.perf_counter() - start)
        atomic_json(manifest_path, result); return result
    except Exception as exc:
        atomic_json(manifest_path, {"status": "failed", "failed_at": now(), "task": task, "model_family": "RF", "selected_config_id": config_id, "configuration": config, "seed": seed, "test_accessed": False, "error_type": type(exc).__name__, "error": str(exc)})
        raise


def _fit_manifest(task: str, family: str, config_id: str, config: dict[str, Any], seed: int, selection_hash: str, phase4_sha: str, metrics: dict[str, Any], selected_epoch: int | None, epochs_run: int | None, artifact: Path, runtime: float, early_stopping: dict[str, Any] | None = None) -> dict[str, Any]:
    key = "model_sha256" if family == "RF" else "checkpoint_sha256"; path_key = "model_path" if family == "RF" else "checkpoint_path"; primary = "balanced_accuracy" if "classification" in task else "mae"
    return {"schema_version": "satnet.phase4.final_fit_manifest.v1", "status": "completed", "completed_at": now(), "task": task, "model_family": family, "selected_config_id": config_id, "configuration": config, "seed": seed, "dataset_bundle_hash": DATASET_BUNDLE_HASH, "training_payload_inventory_hash": TRAINING_PAYLOAD_INVENTORY_HASH, "training_payload_bundle_hash": TRAINING_PAYLOAD_BUNDLE_HASH, "training_plan_bundle_hash": TRAINING_PLAN_BUNDLE_HASH, "phase3_candidate_set_hash": CANDIDATE_SET_HASH, "selection_freeze_sha256": selection_hash, "train_count": TRAIN_COUNT, "validation_count": VALIDATION_COUNT, "selected_epoch": selected_epoch, "epochs_run": epochs_run, "validation_primary_metric": metrics[primary], "validation_metrics": metrics, "early_stopping": early_stopping, path_key: str(artifact), key: sha256_file(artifact), "environment_versions": environment_versions(), "phase4_tooling_git_sha": phase4_sha, "test_accessed": False, "fit_runtime_seconds": runtime}


def _append_log(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n"); handle.flush(); os.fsync(handle.fileno())


def run_tgnn(task: str, seed: int, winner: dict[str, Any], phase4_sha: str) -> dict[str, Any]:
    import copy
    import torch
    from torch.nn import CrossEntropyLoss, SmoothL1Loss
    from satnet.models.gnn_model import SatelliteGNN
    config_id = winner["winning_config_id"]; selection_hash = TGNN_SELECTION_HASH
    completed = _verified_completed(task, seed, config_id, selection_hash)
    if completed is not None: return completed
    data = load_tgnn_data(task); seed_metadata = initialize_determinism(seed); torch.set_num_threads(8); task_type = "classification" if "classification" in task else "regression"; config = dict(winner["winning_configuration"])
    manifest_path = _running_manifest(task, seed, config_id, config, "TGNN", selection_hash); log_path = manifest_path.parent / "progress.jsonl"
    # A non-completed attempt owns this log. A restart starts a clean attempt;
    # completed manifests are immutable and never enter this branch.
    reset_attempt_log(log_path)
    start = time.perf_counter(); device = torch.device("cpu")
    model = SatelliteGNN(node_features=3, hidden_channels=int(config["hidden_dim"]), out_channels=2 if task_type == "classification" else 1, task_type=task_type, cheb_k=int(config["cheb_k"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    criterion = CrossEntropyLoss() if task_type == "classification" else SmoothL1Loss()
    monitor_name, direction = ("balanced_accuracy", "maximize") if task_type == "classification" else ("mae", "minimize")
    best_metric: float | None = None; best_epoch: int | None = None; best_state: dict[str, Any] | None = None; bad_epochs = 0; epochs_run = 0; checkpoint = manifest_path.parent / "best_validation_checkpoint.pt"
    try:
        for epoch in range(1, int(config["max_epochs"]) + 1):
            model.train(); train_loss = 0.0
            for run_id in data.train_ids:
                target, output = _tgnn_forward(model, data, run_id, device); label = torch.tensor([target], dtype=torch.long if task_type == "classification" else torch.float32); optimizer.zero_grad()
                loss = criterion(output, label) if task_type == "classification" else criterion(output.view(-1), label.view(-1)); loss.backward(); optimizer.step(); train_loss += float(loss.item())
            model.eval(); y_true: list[Any] = []; y_pred: list[Any] = []; scores: list[Any] = []
            with torch.no_grad():
                for run_id in data.validation_ids:
                    target, output = _tgnn_forward(model, data, run_id, device); y_true.append(target)
                    if task_type == "classification": scores.append(torch.softmax(output, dim=1).cpu().numpy()[0]); y_pred.append(int(output.argmax(dim=1).item()))
                    else: y_pred.append(float(output.squeeze().item()))
            metrics = classification_metrics(y_true, y_pred, np.asarray(scores)) if task_type == "classification" else regression_metrics(y_true, y_pred); monitor = float(metrics[monitor_name]); epochs_run = epoch
            improved = best_metric is None or (monitor > best_metric if direction == "maximize" else monitor < best_metric)
            if improved: best_metric = monitor; best_epoch = epoch; best_state = copy.deepcopy(model.state_dict()); bad_epochs = 0
            else: bad_epochs += 1
            _append_log(log_path, {"epoch": epoch, "train_loss": train_loss / TRAIN_COUNT, "validation_metrics": metrics, "monitor": monitor, "monitor_name": monitor_name, "direction": direction, "improved": improved, "at": now()})
            if improved:
                payload = {"epoch": epoch, "model_state_dict": best_state, "optimizer_state_dict": optimizer.state_dict(), "configuration": config, "task": task, "seed": seed, "validation_primary_metric": best_metric, "dataset_bundle_hash": DATASET_BUNDLE_HASH, "training_plan_bundle_hash": TRAINING_PLAN_BUNDLE_HASH, "phase3_candidate_set_hash": CANDIDATE_SET_HASH, "test_accessed": False, "seed_metadata": seed_metadata}
                temporary = checkpoint.with_name(f"{checkpoint.name}.tmp-{os.getpid()}"); torch.save(payload, temporary); os.replace(temporary, checkpoint)
            if bad_epochs >= 10: break
        require(best_state is not None and best_epoch is not None, f"TGNN produced no validation checkpoint: {task}/{seed}")
        model.load_state_dict(best_state); model.eval(); y_true = []; y_pred = []; scores = []
        with torch.no_grad():
            for run_id in data.validation_ids:
                target, output = _tgnn_forward(model, data, run_id, device); y_true.append(target)
                if task_type == "classification": scores.append(torch.softmax(output, dim=1).cpu().numpy()[0]); y_pred.append(int(output.argmax(dim=1).item()))
                else: y_pred.append(float(output.squeeze().item()))
        metrics = classification_metrics(y_true, y_pred, np.asarray(scores)) if task_type == "classification" else regression_metrics(y_true, y_pred)
        early = {"monitor": monitor_name, "direction": direction, "patience": 10, "min_delta": 0.0, "restore_best": True, "stopped_early": epochs_run < int(config["max_epochs"])}
        result = _fit_manifest(task, "TGNN", config_id, config, seed, selection_hash, phase4_sha, metrics, best_epoch, epochs_run, checkpoint, time.perf_counter() - start, early); result["determinism"] = seed_metadata; atomic_json(manifest_path, result); return result
    except Exception as exc:
        atomic_json(manifest_path, {"status": "failed", "failed_at": now(), "task": task, "model_family": "TGNN", "selected_config_id": config_id, "configuration": config, "seed": seed, "test_accessed": False, "error_type": type(exc).__name__, "error": str(exc)}); raise


def write_progress(winners: dict[str, dict[str, Any]], phase4_sha: str, current: dict[str, Any] | None = None) -> None:
    rf_done = sum(_verified_completed(t, s, winners[t]["winning_config_id"], RF_SELECTION_HASH) is not None for t in RF_TASKS for s in FINAL_SEEDS)
    tgnn_done = sum(_verified_completed(t, s, winners[t]["winning_config_id"], TGNN_SELECTION_HASH) is not None for t in TGNN_TASKS for s in FINAL_SEEDS)
    atomic_json(OUTPUT_ROOT / "progress.json", {"schema_version": "satnet.phase4.progress.v1", "updated_at": now(), "rf": {"expected": 25, "completed": rf_done, "failed": 0}, "tgnn": {"expected": 10, "completed": tgnn_done, "failed": 0}, "total": {"expected": EXPECTED_TOTAL, "completed": rf_done + tgnn_done, "failed": 0}, "current": current, "test_evaluation": False, "test_accessed": False, "phase4_tooling_git_sha": phase4_sha})


def finalize(winners: dict[str, dict[str, Any]], phase4_sha: str) -> None:
    tasks: dict[str, Any] = {}
    for task in ALL_TASKS:
        selection_hash = RF_SELECTION_HASH if task.startswith("rf_") else TGNN_SELECTION_HASH; manifests = [_verified_completed(task, seed, winners[task]["winning_config_id"], selection_hash) for seed in FINAL_SEEDS]; require(all(manifests), f"Missing completed final fit for {task}"); primary = "balanced_accuracy" if "classification" in task else "mae"; values = [float(m["validation_metrics"][primary]) for m in manifests if m]
        tasks[task] = {"selected_config_id": winners[task]["winning_config_id"], "primary_metric": primary, "seed_42": values[0], "mean": statistics.fmean(values), "population_std": statistics.pstdev(values), "min": min(values), "max": max(values), "per_seed": {str(seed): value for seed, value in zip(FINAL_SEEDS, values)}, "checkpoint_or_model_sha256": {str(seed): (m.get("model_sha256") or m.get("checkpoint_sha256")) for seed, m in zip(FINAL_SEEDS, manifests)}}
    summary = {"schema_version": "satnet.phase4.robustness_summary.v1", "generated_at": now(), "tasks": tasks, "seeds": list(FINAL_SEEDS), "primary_reporting_seed": PRIMARY_SEED, "ensemble": False, "test_accessed": False, "test_evaluation": False, "phase4_tooling_git_sha": phase4_sha}; atomic_json(OUTPUT_ROOT / "robustness_summary.json", summary); atomic_json(OUTPUT_ROOT / "acceptance.json", {"schema_version": "satnet.phase4.acceptance.v1", "status": "PASS", "rf": {"expected": 25, "completed": 25, "failed": 0}, "tgnn": {"expected": 10, "completed": 10, "failed": 0}, "total": {"expected": 35, "completed": 35, "failed": 0}, "test_untouched": True, "test_evaluation": False, "ensemble": False, "phase4_tooling_git_sha": phase4_sha}); _write_inventory()


def _write_inventory() -> None:
    entries: list[dict[str, Any]] = []; digest = hashlib.sha256()
    for path in sorted(p for p in OUTPUT_ROOT.rglob("*") if p.is_file() and p.name != "final_robustness_inventory.json"):
        data = path.read_bytes(); relative = path.relative_to(OUTPUT_ROOT).as_posix(); digest.update(relative.encode() + b"\0" + data); entries.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    atomic_json(OUTPUT_ROOT / "final_robustness_inventory.json", {"schema_version": "satnet.phase4.inventory.v1", "inventory_self_excluding": True, "hash_algorithm": "sha256 over sorted relative UTF-8 path + NUL byte + raw file bytes", "bundle_sha256": digest.hexdigest(), "artifacts": entries, "test_accessed": False})


def run(*, limit: int | None = None) -> None:
    phase4_sha = git_head(); preflight_result = preflight(); winners = preflight_result["winners"]; write_progress(winners, phase4_sha)
    fits: list[tuple[str, int]] = [(task, seed) for task in RF_TASKS for seed in FINAL_SEEDS] + [(task, seed) for task in TGNN_TASKS for seed in FINAL_SEEDS]
    if limit is not None: fits = fits[:limit]
    for task, seed in fits:
        write_progress(winners, phase4_sha, {"task": task, "model_family": "RF" if task.startswith("rf_") else "TGNN", "config_id": winners[task]["winning_config_id"], "seed": seed})
        if task.startswith("rf_"): run_rf(task, seed, winners[task], phase4_sha)
        else: run_tgnn(task, seed, winners[task], phase4_sha)
        write_progress(winners, phase4_sha)
    if limit is None: finalize(winners, phase4_sha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS); args = parser.parse_args(); run(limit=args.limit)


if __name__ == "__main__":
    main()
