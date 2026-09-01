"""Run the single authorized SATNET Adaptive-v2 held-out evaluation.

The default command is metadata-only preflight.  TEST feature, graph, and target
payloads are opened only when ``--execute-heldout`` is explicitly supplied.
This module contains inference and statistics code only; it never constructs a
training optimizer or changes a frozen artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import statistics
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DATASET_ROOT = Path(r"C:\Users\johns\external\satnet-10k-final-ml-datasets-v2-adaptive")
PHASE3_ROOT = Path(r"C:\Users\johns\external\satnet-10k-final-models-v2-adaptive")
PHASE4_ROOT = PHASE3_ROOT / "final_robustness"
PLAN_ROOT = Path(r"C:\Users\johns\external\satnet-10k-training-test-plan-v1")
OUTPUT_ROOT = Path(r"C:\Users\johns\external\satnet-10k-heldout-v2-adaptive")

DATASET_BUNDLE_HASH = "1399f12e6ed7e159076028bc01971cf2274924fc55b43cca0a44219db8c978d1"
CANDIDATE_SET_HASH = "8d13cbcae2c58c72c9c1ae9da5574423b866097e325bc987ae2bb8d7aae7aaf2"
RF_SELECTION_HASH = "bbdb71469bb88d85ce45720dbaee4e8056589d75e741113e601d52a9f704a2cb"
TGNN_SELECTION_HASH = "81e293f8660c00bf8de0dc55280135c2b116c41c637001a742ec626afdb6108b"
PHASE4_TOOLING_SHA = "4a49643189a413a5f8c81ce89ff6a08d222f4889"
PHASE4_INVENTORY_SHA = "63ec97e52849b92773887f140a5e95ac3c337837ea7451f9b3df142b4bfd9779"
PHASE4_BUNDLE_HASH = "fa0be3579da38169d4d36663eafca62b4412b4e1b403e7241ae64737e4c6fc83"
TRAINING_PLAN_BUNDLE_HASH = "e14ec5e5b2221aa2c9aa187a2baacc517e676f9dc748612c206eb1dc43788a0a"
TRAINING_PLAN_INVENTORY_SHA = "04604051ebc5f4ce659fe558f50ed35689665f7243328ab67a6831948c11f991"
ADAPTIVE_CONTRACT_BUNDLE_HASH = "da3c73711b1d60635afcceee8bda0a60d1379e492e1ad0d588d5a0c25e10abe3"

FINAL_SEEDS = (42, 123, 456, 789, 2026)
PRIMARY_SEED = 42
TEST_RUNS = 1500
TEST_DESIGNS = 300
REALIZATIONS_PER_DESIGN = 5
SPLIT_COUNTS = {"train": 7000, "validation": 1500, "test": TEST_RUNS}
BOOTSTRAP_SEED = 20260812
BOOTSTRAP_REPLICATES = 2000
CONFIDENCE_LEVEL = 0.95
EXPECTED_TOPOLOGY = ("grid_adaptive", 1, 1, "persistent_temporal_union_edges_v1")
METADATA_COLUMNS = ("run_id", "run_key", "design_id", "realization_id", "split")

RF_FEATURES = (
    "num_planes",
    "sats_per_plane",
    "altitude_km",
    "inclination_deg",
    "satellite_node_failure_probability",
    "satellite_edge_failure_probability",
)
INTEGRATED_FEATURES = RF_FEATURES + (
    "civilian_count",
    "government_count",
    "military_count",
    "ground_station_failure_probability",
)
RF_TASKS = (
    "rf_integrated_classification",
    "rf_integrated_regression_mean",
    "rf_integrated_regression_min",
    "rf_space_classification",
    "rf_space_regression",
)
TGNN_TASKS = ("tgnn_space_classification", "tgnn_space_regression")
ALL_TASKS = RF_TASKS + TGNN_TASKS
RF_DATASETS = {
    "rf_integrated_classification": (
        "rf_integrated_classification/rf_integrated_classification.csv",
        INTEGRATED_FEATURES,
        "overall_threshold_breach_any",
    ),
    "rf_integrated_regression_mean": (
        "rf_integrated_regression/rf_integrated_regression.csv",
        INTEGRATED_FEATURES,
        "failure_adjusted_overall_service_fraction_mean",
    ),
    "rf_integrated_regression_min": (
        "rf_integrated_regression/rf_integrated_regression.csv",
        INTEGRATED_FEATURES,
        "failure_adjusted_overall_service_fraction_min",
    ),
    "rf_space_classification": (
        "rf_space_classification/rf_space_classification.csv",
        RF_FEATURES,
        "space_threshold_breach_any",
    ),
    "rf_space_regression": (
        "rf_space_regression/rf_space_regression.csv",
        RF_FEATURES,
        "space_gcc_fraction_original_min",
    ),
}
TGNN_TARGETS = {
    "tgnn_space_classification": "tgnn_space_classification/tgnn_space_classification_target_manifest.jsonl",
    "tgnn_space_regression": "tgnn_space_regression/tgnn_space_regression_target_manifest.jsonl",
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
EXPECTED_RF_CONFIGS: dict[str, dict[str, Any]] = {
    "rf_integrated_classification": {"bootstrap": True, "class_weight": "balanced", "max_depth": 10, "max_features": 1.0, "min_samples_leaf": 1, "n_estimators": 300, "n_jobs": -1, "random_state": 42},
    "rf_integrated_regression_mean": {"bootstrap": True, "max_depth": 20, "max_features": 1.0, "min_samples_leaf": 5, "n_estimators": 600, "n_jobs": -1, "random_state": 42},
    "rf_integrated_regression_min": {"bootstrap": True, "max_depth": 20, "max_features": 1.0, "min_samples_leaf": 5, "n_estimators": 300, "n_jobs": -1, "random_state": 42},
    "rf_space_classification": {"bootstrap": True, "class_weight": "balanced", "max_depth": 10, "max_features": 1.0, "min_samples_leaf": 2, "n_estimators": 300, "n_jobs": -1, "random_state": 42},
    "rf_space_regression": {"bootstrap": True, "max_depth": 20, "max_features": 1.0, "min_samples_leaf": 5, "n_estimators": 300, "n_jobs": -1, "random_state": 42},
}
EXPECTED_TGNN_CONFIGS = {
    "tgnn_space_classification": {"batch_size": 1, "cheb_k": 2, "classification_loss": "CrossEntropyLoss", "dropout": None, "hidden_dim": 64, "learning_rate": 0.001, "max_epochs": 100, "num_layers": 1, "optimizer": "Adam", "regression_loss": "SmoothL1Loss", "weight_decay": 0.0},
    "tgnn_space_regression": {"batch_size": 1, "cheb_k": 3, "classification_loss": "CrossEntropyLoss", "dropout": None, "hidden_dim": 64, "learning_rate": 0.001, "max_epochs": 100, "num_layers": 1, "optimizer": "Adam", "regression_loss": "SmoothL1Loss", "weight_decay": 0.0},
}


class HeldoutError(RuntimeError):
    """Raised when a frozen held-out evaluation contract is violated."""


@dataclass(frozen=True)
class FrozenArtifact:
    task: str
    seed: int
    model_path: Path
    model_sha256: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class RFTestData:
    x: np.ndarray
    y: np.ndarray
    metadata: tuple[dict[str, str], ...]
    train_y: np.ndarray


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
class TGNNTestData:
    records: tuple[SequenceRecord, ...]
    y: np.ndarray
    train_y: np.ndarray


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HeldoutError(message)


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
        raise HeldoutError(f"Cannot load JSON evidence: {path}") from exc
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def load_jsonl_metadata(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HeldoutError(f"Cannot load metadata manifest: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HeldoutError(f"Malformed metadata manifest {path}:{line_number}") from exc
        require(isinstance(row, dict), f"Metadata row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _verify_plan() -> dict[str, Any]:
    inventory_path = PLAN_ROOT / "training_plan_inventory.json"
    require(sha256_file(inventory_path) == TRAINING_PLAN_INVENTORY_SHA, "Training-plan inventory SHA mismatch")
    inventory = load_json(inventory_path)
    require(inventory.get("bundle_sha256") == TRAINING_PLAN_BUNDLE_HASH, "Training-plan bundle SHA mismatch")
    for artifact in inventory.get("artifacts", []):
        path = PLAN_ROOT / str(artifact["path"])
        require(path.is_file() and path.stat().st_size == int(artifact["bytes"]), f"Missing plan artifact: {path}")
        require(sha256_file(path) == artifact["sha256"], f"Plan artifact SHA mismatch: {path}")
    seeds = load_json(PLAN_ROOT / "seed_registry.json")
    require(tuple(seeds.get("final_robustness_seeds", ())) == FINAL_SEEDS, "Frozen final seed set mismatch")
    require(seeds.get("primary_reporting_seed") == PRIMARY_SEED, "Primary reporting seed mismatch")
    selection = load_json(PLAN_ROOT / "model_selection_contract.json")
    require(selection.get("final_phase", {}).get("seeds") == list(FINAL_SEEDS), "Final plan seed contract mismatch")
    return {"bundle_sha256": TRAINING_PLAN_BUNDLE_HASH, "inventory_sha256": TRAINING_PLAN_INVENTORY_SHA, "final_seeds": list(FINAL_SEEDS), "primary_reporting_seed": PRIMARY_SEED, "interpretation": "all five frozen final seeds are authorized for TEST; seed 42 is primary"}


def _load_selection(path: Path, family: str, expected_ids: dict[str, str], expected_configs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    require(sha256_file(path) == (RF_SELECTION_HASH if family == "RF" else TGNN_SELECTION_HASH), f"{family} selection SHA mismatch")
    value = load_json(path)
    require(value.get("candidate_set_hash") == CANDIDATE_SET_HASH, f"{family} candidate-set SHA mismatch")
    tasks = value.get("tasks")
    require(isinstance(tasks, dict) and set(tasks) == set(expected_ids), f"{family} selection task set mismatch")
    winners: dict[str, dict[str, Any]] = {}
    for task, expected_id in expected_ids.items():
        record = tasks[task]
        require(record.get("winning_config_id") == expected_id, f"Frozen winner mismatch for {task}")
        require(record.get("winning_config_id") != record.get("runner_up", {}).get("config_id"), f"Runner-up selected for {task}")
        require(record.get("winning_configuration") == expected_configs[task], f"Frozen configuration mismatch for {task}")
        winners[task] = record
    return winners


def _verify_phase3() -> dict[str, Any]:
    manifest = load_json(PHASE3_ROOT / "manifests" / "phase3_manifest.json")
    checks = (
        manifest.get("status") == "PASS",
        manifest.get("rf_expected") == 756,
        manifest.get("rf_completed") == 756,
        manifest.get("rf_failed") == 0,
        manifest.get("tgnn_expected") == 96,
        manifest.get("tgnn_completed") == 96,
        manifest.get("tgnn_failed") == 0,
        manifest.get("candidate_set_hash") == CANDIDATE_SET_HASH,
        manifest.get("selection_used_only_validation") is True,
        manifest.get("test_evaluation_performed") is False,
        manifest.get("test_metrics_calculated") is False,
        manifest.get("test_predictions_generated") is False,
        manifest.get("test_targets_loaded") is False,
        manifest.get("historical_fixed_data_reused") is False,
        manifest.get("historical_fixed_checkpoint_reused") is False,
    )
    require(all(checks), "Phase-3 identity or TEST firewall failed")
    firewall = manifest.get("test_firewall", {})
    require(firewall.get("passed") is True and firewall.get("test_evaluation_performed") is False and firewall.get("test_metrics_present") is False and firewall.get("test_predictions_present") is False and firewall.get("test_targets_loaded") is False, "Phase-3 firewall evidence failed")
    for filename, expected_sha in (("rf_selection_freeze.json", RF_SELECTION_HASH), ("tgnn_selection_freeze.json", TGNN_SELECTION_HASH)):
        freeze = load_json(PHASE3_ROOT / "manifests" / filename)
        require(freeze.get("sha256") == expected_sha and freeze.get("candidate_set_hash") == CANDIDATE_SET_HASH and freeze.get("test_accessed") is False, f"Selection freeze evidence failed: {filename}")
    _load_selection(PHASE3_ROOT / "summaries" / "rf_validation_selection.json", "RF", EXPECTED_RF_IDS, EXPECTED_RF_CONFIGS)
    _load_selection(PHASE3_ROOT / "summaries" / "tgnn_validation_selection.json", "TGNN", EXPECTED_TGNN_IDS, EXPECTED_TGNN_CONFIGS)
    stopping = load_json(PHASE3_ROOT / "summaries" / "tgnn_validation_selection.json").get("early_stopping", {})
    require(stopping == {"min_delta": 0.0, "monitor_classification": "balanced_accuracy", "monitor_regression": "mae", "patience": 10, "restore_best": True}, "TGNN frozen stopping protocol mismatch")
    return {"status": "PASS", "rf_fits": "756/756", "tgnn_fits": "96/96", "failures": 0, "selection_used_only_validation": True, "test_evaluation": False, "test_metrics": False, "test_predictions": False, "test_targets_loaded": False, "candidate_set_hash": CANDIDATE_SET_HASH}


def _validate_split_metadata(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    require(len(rows) == sum(SPLIT_COUNTS.values()), f"Split metadata row count mismatch: {len(rows)}")
    run_ids: set[int] = set()
    split_designs: dict[str, set[str]] = {split: set() for split in SPLIT_COUNTS}
    design_splits: dict[str, set[str]] = {}
    realization_counts: dict[str, int] = {}
    realization_ids: dict[str, set[str]] = {}
    split_runs = {split: 0 for split in SPLIT_COUNTS}
    for row in rows:
        try:
            run_id = int(row["run_id"])
            split = str(row["split"]).lower()
            design = str(row["design_id"])
            realization = str(row["realization_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HeldoutError("Malformed split metadata identity") from exc
        require(split in SPLIT_COUNTS, f"Unknown split in metadata: {split}")
        require(run_id not in run_ids, f"Duplicate run_id in metadata: {run_id}")
        run_ids.add(run_id)
        split_runs[split] += 1
        split_designs[split].add(design)
        design_splits.setdefault(design, set()).add(split)
        key = f"{split}:{design}"
        realization_counts[key] = realization_counts.get(key, 0) + 1
        realization_ids.setdefault(key, set()).add(realization)
    require(run_ids == set(range(sum(SPLIT_COUNTS.values()))), "Run IDs are not the complete frozen 0..9999 identity")
    require(split_runs == SPLIT_COUNTS, f"Split run counts mismatch: {split_runs}")
    require(all(len(splits) == 1 for splits in design_splits.values()), "A design crosses splits")
    require(all(count == REALIZATIONS_PER_DESIGN for count in realization_counts.values()), "A design does not have five realizations")
    require(all(len(values) == REALIZATIONS_PER_DESIGN for values in realization_ids.values()), "A design has duplicate realizations")
    require({split: len(designs) for split, designs in split_designs.items()} == {"train": 1400, "validation": 300, "test": TEST_DESIGNS}, "Split design counts mismatch")
    test_designs = sorted(split_designs["test"])
    design_digest = hashlib.sha256("\n".join(test_designs).encode("utf-8")).hexdigest()
    return {"split_runs": split_runs, "split_designs": {split: len(value) for split, value in split_designs.items()}, "realizations_per_design": REALIZATIONS_PER_DESIGN, "designs_disjoint_across_splits": True, "test_design_count": len(test_designs), "test_design_id_sha256": design_digest}


def _verify_dataset() -> dict[str, Any]:
    phase2 = load_json(DATASET_ROOT / "phase2_manifest.json")
    counts = phase2.get("counts", {})
    require(phase2.get("status") == "PASS" and phase2.get("ml_bundle_hash") == DATASET_BUNDLE_HASH, "Adaptive-v2 dataset bundle identity mismatch")
    require(counts.get("split_runs") == SPLIT_COUNTS and counts.get("split_designs") == {"train": 1400, "validation": 300, "test": TEST_DESIGNS}, "Adaptive-v2 cardinality metadata mismatch")
    require(counts.get("realizations_per_design") == REALIZATIONS_PER_DESIGN, "Adaptive-v2 realization contract mismatch")
    require(phase2.get("historical_fixed_source_contamination") == 0, "grid_fixed source contamination detected")
    topology = phase2.get("topology", {})
    require(topology.get("grid_fixed_source_count") == 0, "grid_fixed source count is nonzero")
    aggregates = topology.get("aggregate", [])
    require(len(aggregates) == 1 and tuple(aggregates[0].get("tuple", ())) == EXPECTED_TOPOLOGY, "Adaptive topology identity mismatch")
    gates = load_json(DATASET_ROOT / "metadata" / "validation_gates.json")
    require(gates.get("adaptive_contract", {}).get("contract_bundle_hash") == ADAPTIVE_CONTRACT_BUNDLE_HASH, "Adaptive contract hash mismatch")
    require(gates.get("graph_quality", {}).get("tgnn_space_classification", {}).get("split_completeness") == {"train": {"designs": 1400, "runs": 7000}, "validation": {"designs": 300, "runs": 1500}, "test": {"designs": 300, "runs": 1500}}, "Graph split metadata mismatch")
    graph_manifest = DATASET_ROOT / "tgnn_space_classification" / "tgnn_space_graph_manifest.jsonl"
    metadata = _validate_split_metadata(load_jsonl_metadata(graph_manifest))
    return {"dataset_bundle_hash": DATASET_BUNDLE_HASH, "adaptive_topology": list(EXPECTED_TOPOLOGY), "grid_fixed_source_count": 0, "historical_fixed_source_contamination": 0, "split_metadata": metadata, "test_payload_opened": False, "test_targets_opened": False}


def _verify_phase4_inventory() -> tuple[dict[str, Any], dict[str, Any]]:
    inventory_path = PHASE4_ROOT / "final_robustness_inventory.json"
    require(sha256_file(inventory_path) == PHASE4_INVENTORY_SHA, "Phase-4 inventory file SHA mismatch")
    inventory = load_json(inventory_path)
    require(inventory.get("bundle_sha256") == PHASE4_BUNDLE_HASH and inventory.get("test_accessed") is False, "Phase-4 inventory identity mismatch")
    artifacts = inventory.get("artifacts", [])
    require(len(artifacts) == 85, f"Phase-4 artifact count mismatch: {len(artifacts)}")
    by_path: dict[str, Any] = {}
    for item in artifacts:
        relative = str(item["path"])
        path = PHASE4_ROOT / relative
        require(path.is_file(), f"Missing Phase-4 artifact: {relative}")
        require(path.stat().st_size == int(item["bytes"]), f"Phase-4 artifact size mismatch: {relative}")
        actual = sha256_file(path)
        require(actual == str(item["sha256"]).lower(), f"Phase-4 artifact SHA mismatch: {relative}")
        by_path[relative] = item
    acceptance = load_json(PHASE4_ROOT / "acceptance.json")
    require(acceptance.get("status") == "PASS" and acceptance.get("total") == {"completed": 35, "expected": 35, "failed": 0} and acceptance.get("test_evaluation") is False and acceptance.get("test_untouched") is True, "Phase-4 acceptance gate failed")
    identity = load_json(PHASE4_ROOT / "phase4_execution_identity.json")
    require(identity.get("phase4_tooling_git_sha") == PHASE4_TOOLING_SHA and identity.get("test_accessed") is False and identity.get("phase3_candidate_set_hash") == CANDIDATE_SET_HASH, "Phase-4 execution identity mismatch")
    return by_path, {"inventory_file_sha256": PHASE4_INVENTORY_SHA, "bundle_sha256": PHASE4_BUNDLE_HASH, "artifacts_verified": len(by_path), "status": "PASS", "test_accessed": False, "tooling_sha": PHASE4_TOOLING_SHA}


def _verify_frozen_artifact(task: str, seed: int, inventory: dict[str, Any]) -> FrozenArtifact:
    require(task in ALL_TASKS and seed in FINAL_SEEDS, "Unauthorized task or seed")
    manifest_path = PHASE4_ROOT / task / f"seed_{seed}" / "final_manifest.json"
    manifest_rel = manifest_path.relative_to(PHASE4_ROOT).as_posix()
    require(manifest_rel in inventory, f"Frozen manifest absent from Phase-4 inventory: {manifest_rel}")
    require(sha256_file(manifest_path) == str(inventory[manifest_rel]["sha256"]).lower(), f"Phase-4 manifest SHA mismatch: {manifest_path}")
    manifest = load_json(manifest_path)
    expected_id = (EXPECTED_RF_IDS if task.startswith("rf_") else EXPECTED_TGNN_IDS)[task]
    expected_config = dict((EXPECTED_RF_CONFIGS if task.startswith("rf_") else EXPECTED_TGNN_CONFIGS)[task])
    if task.startswith("rf_"):
        expected_config["random_state"] = seed
    require(manifest.get("status") == "completed" and manifest.get("task") == task and manifest.get("seed") == seed, f"Incomplete frozen model manifest: {manifest_path}")
    require(manifest.get("selected_config_id") == expected_id and manifest.get("configuration") == expected_config, f"Frozen model selection mismatch: {manifest_path}")
    require(manifest.get("dataset_bundle_hash") == DATASET_BUNDLE_HASH and manifest.get("phase3_candidate_set_hash") == CANDIDATE_SET_HASH and manifest.get("test_accessed") is False, f"Frozen model provenance mismatch: {manifest_path}")
    require(manifest.get("selection_freeze_sha256") == (RF_SELECTION_HASH if task.startswith("rf_") else TGNN_SELECTION_HASH), f"Frozen selection hash mismatch: {manifest_path}")
    filename = "final_model.joblib" if task.startswith("rf_") else "best_validation_checkpoint.pt"
    model_path = PHASE4_ROOT / task / f"seed_{seed}" / filename
    require(Path(str(manifest.get("model_path" if task.startswith("rf_") else "checkpoint_path"))).resolve() == model_path.resolve(), f"Non-final artifact path rejected: {manifest_path}")
    relative = model_path.relative_to(PHASE4_ROOT).as_posix()
    require(relative in inventory, f"Frozen model absent from Phase-4 inventory: {relative}")
    model_sha = sha256_file(model_path)
    require(model_sha == str(manifest.get("model_sha256" if task.startswith("rf_") else "checkpoint_sha256")).lower(), f"Frozen model manifest SHA mismatch: {model_path}")
    require(model_sha == str(inventory[relative]["sha256"]).lower(), f"Frozen model inventory SHA mismatch: {model_path}")
    return FrozenArtifact(task, seed, model_path, model_sha, manifest)


def verify_frozen_artifacts(inventory: dict[str, Any]) -> tuple[FrozenArtifact, ...]:
    artifacts = tuple(_verify_frozen_artifact(task, seed, inventory) for task in ALL_TASKS for seed in FINAL_SEEDS)
    require(len(artifacts) == 35, "Frozen final artifact count mismatch")
    return artifacts


def _parse_classification(value: str) -> int:
    text = value.strip().lower()
    require(text in {"true", "false", "0", "1"}, f"Invalid classification target: {value!r}")
    return int(text in {"true", "1"})


def _validate_metadata_against_graph(metadata: Sequence[dict[str, str]], graph_metadata: Sequence[dict[str, Any]]) -> None:
    expected = {(str(row["run_id"]), str(row["run_key"]), str(row["design_id"]), str(row["realization_id"]), str(row["split"]).lower()) for row in graph_metadata if str(row["split"]).lower() == "test"}
    actual = {(str(row["run_id"]), str(row["run_key"]), str(row["design_id"]), str(row["realization_id"]), str(row["split"]).lower()) for row in metadata}
    require(actual == expected, "RF TEST identity differs from authoritative graph metadata")


def load_rf_test_data(task: str, graph_metadata: Sequence[dict[str, Any]]) -> RFTestData:
    relative, features, target = RF_DATASETS[task]
    csv_path = DATASET_ROOT / relative
    manifest = load_json(csv_path.with_name(csv_path.stem + "_manifest.json"))
    schema = load_json(csv_path.with_name(csv_path.stem + "_schema.json"))
    require(tuple(manifest.get("predictors", ())) == features and tuple(item["field"] for item in schema.get("predictors", ())) == features, f"RF feature contract mismatch: {task}")
    require(target in manifest.get("targets", ()) and csv_path.is_file(), f"RF dataset contract mismatch: {task}")
    test_x: list[list[float]] = []
    test_y: list[Any] = []
    test_metadata: list[dict[str, str]] = []
    train_y: list[Any] = []
    counts = {split: 0 for split in SPLIT_COUNTS}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == METADATA_COLUMNS + features + tuple(manifest.get("targets", ())), f"RF CSV schema mismatch: {task}")
        for row in reader:
            split = str(row.get("split", "")).lower()
            require(split in counts, f"Unknown RF split: {split}")
            counts[split] += 1
            value = _parse_classification(str(row[target])) if "classification" in task else float(row[target])
            if split == "train":
                train_y.append(value)
            elif split == "test":
                test_x.append([float(row[field]) for field in features])
                test_y.append(value)
                test_metadata.append({field: str(row[field]) for field in METADATA_COLUMNS})
    require(counts == SPLIT_COUNTS and len(test_metadata) == TEST_RUNS, f"RF split counts mismatch: {task}: {counts}")
    _validate_metadata_against_graph(test_metadata, graph_metadata)
    return RFTestData(np.asarray(test_x, dtype=float), np.asarray(test_y), tuple(test_metadata), np.asarray(train_y))


def _load_graph_metadata() -> list[dict[str, Any]]:
    return load_jsonl_metadata(DATASET_ROOT / "tgnn_space_classification" / "tgnn_space_graph_manifest.jsonl")


def load_tgnn_test_data(task: str, graph_metadata: Sequence[dict[str, Any]]) -> TGNNTestData:
    target_path = DATASET_ROOT / TGNN_TARGETS[task]
    target_field = "space_threshold_breach_any" if "classification" in task else "space_gcc_fraction_original_min"
    target_rows = load_jsonl_metadata(target_path)
    require(len(target_rows) == sum(SPLIT_COUNTS.values()), f"TGNN target manifest count mismatch: {task}")
    graph_by_id = {int(row["run_id"]): row for row in graph_metadata}
    test_pairs: list[tuple[SequenceRecord, Any]] = []
    train_y: list[Any] = []
    target_ids: set[int] = set()
    for row in target_rows:
        run_id = int(row["run_id"])
        require(run_id not in target_ids, f"Duplicate TGNN target run_id: {run_id}")
        target_ids.add(run_id)
        require(row.get("target_field") == target_field and run_id in graph_by_id, f"TGNN target identity mismatch: {task}:{run_id}")
        graph = graph_by_id[run_id]
        for key in ("run_key", "design_id", "realization_id", "split"):
            require(str(row[key]) == str(graph[key]), f"TGNN graph/target metadata mismatch: {task}:{run_id}:{key}")
        value = _parse_classification(str(row["target"])) if "classification" in task else float(row["target"])
        if str(row["split"]).lower() == "train":
            train_y.append(value)
        elif str(row["split"]).lower() == "test":
            test_pairs.append((SequenceRecord(run_id, str(row["run_key"]), str(row["design_id"]), str(row["realization_id"]), "test", str(graph["sequence_artifact"]), str(graph["sequence_artifact_sha256"])), value))
    test_pairs.sort(key=lambda pair: pair[0].run_id)
    test_records = tuple(pair[0] for pair in test_pairs)
    test_y = np.asarray([pair[1] for pair in test_pairs])
    require(len(target_ids) == sum(SPLIT_COUNTS.values()) and len(test_records) == TEST_RUNS and len(train_y) == SPLIT_COUNTS["train"], f"TGNN split counts mismatch: {task}")
    return TGNNTestData(test_records, test_y, np.asarray(train_y))


def load_frozen_rf(artifact: FrozenArtifact) -> Any:
    import joblib
    model = joblib.load(artifact.model_path)
    require(hasattr(model, "predict") and hasattr(model, "predict_proba" if "classification" in artifact.task else "predict"), f"Invalid frozen RF artifact: {artifact.model_path}")
    return model


def load_frozen_tgnn(artifact: FrozenArtifact) -> Any:
    import torch
    from satnet.models.gnn_model import SatelliteGNN
    task_type = "classification" if "classification" in artifact.task else "regression"
    config = artifact.manifest["configuration"]
    model = SatelliteGNN(node_features=3, hidden_channels=int(config["hidden_dim"]), out_channels=2 if task_type == "classification" else 1, task_type=task_type, cheb_k=int(config["cheb_k"]))
    payload = torch.load(artifact.model_path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict), f"Malformed frozen TGNN checkpoint: {artifact.model_path}")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


MAGIC = b"SATNET-TGNN-V1\x00"
ARRAY_ORDER = ("node_features", "node_identity_index", "edge_index", "edge_attr", "snapshot_node_offsets", "snapshot_edge_offsets", "timestep_index")


def load_sequence(record: SequenceRecord) -> list[Any]:
    import torch
    from torch_geometric.data import Data
    path = DATASET_ROOT / Path(record.sequence_artifact)
    require(path.is_file() and sha256_file(path) == record.sequence_artifact_sha256, f"TEST sequence SHA mismatch: {path}")
    raw = path.read_bytes()
    require(raw.startswith(MAGIC), f"Malformed TGNN sequence: {path}")
    cursor = len(MAGIC)
    require(cursor + 8 <= len(raw), f"Malformed TGNN sequence header: {path}")
    header_length = struct.unpack_from("<Q", raw, cursor)[0]
    cursor += 8
    header = json.loads(raw[cursor:cursor + header_length].decode("utf-8"))
    cursor += header_length
    identity = header.get("sequence", {})
    for key in ("run_id", "run_key", "design_id", "realization_id", "split"):
        require(str(identity.get(key)) == str(getattr(record, key)), f"TGNN sequence identity mismatch: {path}:{key}")
    arrays: dict[str, np.ndarray] = {}
    for name in ARRAY_ORDER:
        require(cursor + 8 <= len(raw), f"Malformed TGNN sequence array header: {path}")
        size = struct.unpack_from("<Q", raw, cursor)[0]
        cursor += 8
        require(cursor + size <= len(raw), f"Malformed TGNN sequence array: {path}")
        arrays[name] = np.load(io.BytesIO(raw[cursor:cursor + size]), allow_pickle=False)
        cursor += size
    require(cursor == len(raw) and header.get("sequence_length") == 11 and arrays["node_features"].shape[1] == 3 and arrays["edge_attr"].shape[1] == 4, f"TGNN sequence format mismatch: {path}")
    sequence: list[Any] = []
    for index in range(11):
        node_start, node_end = (int(value) for value in arrays["snapshot_node_offsets"][index:index + 2])
        edge_start, edge_end = (int(value) for value in arrays["snapshot_edge_offsets"][index:index + 2])
        x = torch.as_tensor(arrays["node_features"][node_start:node_end], dtype=torch.float32)
        edge_index = torch.as_tensor(arrays["edge_index"][:, edge_start:edge_end], dtype=torch.long)
        edge_attr = torch.as_tensor(arrays["edge_attr"][edge_start:edge_end], dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.edge_weight = edge_attr[:, 0]
        data.timestep_index = torch.tensor([index], dtype=torch.long)
        sequence.append(data)
    return sequence


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], scores: Sequence[float], train_y: Sequence[int]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    score = np.asarray(scores, dtype=float)
    matrix = confusion_matrix(true, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    unique = np.unique(true)
    require(len(unique) == 2, "Classification TEST target has only one class")
    majority = int(np.bincount(np.asarray(train_y, dtype=int)).argmax())
    return {
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=[0, 1], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, labels=[0, 1], average="weighted", zero_division=0)),
        "precision_by_class": {str(label): float(value) for label, value in zip([0, 1], precision_score(true, pred, labels=[0, 1], average=None, zero_division=0))},
        "recall_by_class": {str(label): float(value) for label, value in zip([0, 1], recall_score(true, pred, labels=[0, 1], average=None, zero_division=0))},
        "f1_by_class": {str(label): float(value) for label, value in zip([0, 1], f1_score(true, pred, labels=[0, 1], average=None, zero_division=0))},
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "confusion_matrix": matrix.tolist(),
        "roc_auc": float(roc_auc_score(true, score)),
        "pr_auc": float(average_precision_score(true, score)),
        "positive_class_count": int(np.sum(true == 1)),
        "negative_class_count": int(np.sum(true == 0)),
        "majority_class_baseline_accuracy": float(np.mean(true == majority)),
        "train_majority_class": majority,
        "minority_class": int(np.bincount(true).argmin()),
    }


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    residual = pred - true
    absolute = np.abs(residual)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "r2": float(r2_score(true, pred)),
        "median_absolute_error": float(median_absolute_error(true, pred)),
        "maximum_absolute_error": float(np.max(absolute)),
        "residual_mean": float(np.mean(residual)),
        "residual_standard_deviation": float(np.std(residual)),
        "target_mean": float(np.mean(true)),
        "target_median": float(np.median(true)),
        "target_standard_deviation": float(np.std(true)),
        "prediction_mean": float(np.mean(pred)),
        "prediction_standard_deviation": float(np.std(pred)),
        "prediction_minimum": float(np.min(pred)),
        "prediction_maximum": float(np.max(pred)),
        "target_minimum": float(np.min(true)),
        "target_maximum": float(np.max(true)),
        "predictions_below_zero": int(np.sum(pred < 0)),
        "predictions_above_one": int(np.sum(pred > 1)),
    }


def _write_prediction_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _classification_rows(task: str, family: str, artifact: FrozenArtifact, metadata: Sequence[dict[str, str]], y_true: Sequence[int], y_pred: Sequence[int], scores: np.ndarray) -> list[dict[str, Any]]:
    return [{"run_id": int(row["run_id"]), "run_key": row["run_key"], "design_id": row["design_id"], "realization_id": row["realization_id"], "split": "test", "task": task, "model_family": family, "selected_config_id": artifact.manifest["selected_config_id"], "model_seed": artifact.seed, "ground_truth": int(y_true[index]), "prediction": int(y_pred[index]), "class_scores": [float(value) for value in scores[index]], "positive_class_score_probability": float(scores[index, 1]), "checkpoint_or_model_sha256": artifact.model_sha256, "dataset_bundle_hash": DATASET_BUNDLE_HASH} for index, row in enumerate(metadata)]


def _regression_rows(task: str, family: str, artifact: FrozenArtifact, metadata: Sequence[dict[str, str]], y_true: Sequence[float], y_pred: Sequence[float]) -> list[dict[str, Any]]:
    return [{"run_id": int(row["run_id"]), "run_key": row["run_key"], "design_id": row["design_id"], "realization_id": row["realization_id"], "split": "test", "task": task, "model_family": family, "selected_config_id": artifact.manifest["selected_config_id"], "model_seed": artifact.seed, "ground_truth": float(y_true[index]), "prediction": float(y_pred[index]), "checkpoint_or_model_sha256": artifact.model_sha256, "dataset_bundle_hash": DATASET_BUNDLE_HASH} for index, row in enumerate(metadata)]


def evaluate_rf(task: str, seed: int, artifact: FrozenArtifact, graph_metadata: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = load_rf_test_data(task, graph_metadata)
    model = load_frozen_rf(artifact)
    prediction = np.asarray(model.predict(data.x))
    if "classification" in task:
        scores = np.asarray(model.predict_proba(data.x), dtype=float)
        require(scores.shape == (TEST_RUNS, 2), f"RF classification score shape mismatch: {task}/{seed}")
        predicted = prediction.astype(int)
        metrics = classification_metrics(data.y.astype(int), predicted, scores[:, 1], data.train_y.astype(int))
        rows = _classification_rows(task, "RF", artifact, data.metadata, data.y.astype(int), predicted, scores)
    else:
        predicted = prediction.astype(float)
        metrics = regression_metrics(data.y.astype(float), predicted)
        rows = _regression_rows(task, "RF", artifact, data.metadata, data.y.astype(float), predicted)
    require(len(rows) == TEST_RUNS, f"RF prediction count mismatch: {task}/{seed}")
    return metrics, rows


def evaluate_tgnn(task: str, seed: int, artifact: FrozenArtifact, graph_metadata: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    data = load_tgnn_test_data(task, graph_metadata)
    model = load_frozen_tgnn(artifact)
    predictions: list[float] = []
    score_rows: list[np.ndarray] = []
    with torch.no_grad():
        for record in data.records:
            sequence = [item.to("cpu") for item in load_sequence(record)]
            output = model(sequence)
            if "classification" in task:
                score_rows.append(torch.softmax(output, dim=1).cpu().numpy()[0])
                predictions.append(float(output.argmax(dim=1).item()))
            else:
                predictions.append(float(output.squeeze().item()))
    if "classification" in task:
        scores = np.asarray(score_rows, dtype=float)
        predicted = np.asarray(predictions, dtype=int)
        metrics = classification_metrics(data.y.astype(int), predicted, scores[:, 1], data.train_y.astype(int))
        metadata = [{"run_id": str(record.run_id), "run_key": record.run_key, "design_id": record.design_id, "realization_id": record.realization_id, "split": record.split} for record in data.records]
        rows = _classification_rows(task, "TGNN", artifact, metadata, data.y.astype(int), predicted, scores)
    else:
        predicted = np.asarray(predictions, dtype=float)
        metrics = regression_metrics(data.y.astype(float), predicted)
        metadata = [{"run_id": str(record.run_id), "run_key": record.run_key, "design_id": record.design_id, "realization_id": record.realization_id, "split": record.split} for record in data.records]
        rows = _regression_rows(task, "TGNN", artifact, metadata, data.y.astype(float), predicted)
    require(len(rows) == TEST_RUNS, f"TGNN prediction count mismatch: {task}/{seed}")
    return metrics, rows


def _metric_value(rows: Sequence[dict[str, Any]], metric: str, classification: bool) -> float:
    y_true = np.asarray([row["ground_truth"] for row in rows])
    y_pred = np.asarray([row["prediction"] for row in rows])
    if classification:
        scores = np.asarray([row["positive_class_score_probability"] for row in rows], dtype=float)
        from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        if metric == "macro_f1":
            return float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0))
        if metric == "minority_class_recall":
            minority = int(np.bincount(y_true.astype(int), minlength=2).argmin())
            return float(recall_score(y_true, y_pred, pos_label=minority, zero_division=0))
        if metric == "roc_auc":
            return float(roc_auc_score(y_true, scores))
        if metric == "pr_auc":
            return float(average_precision_score(y_true, scores))
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        true = y_true.astype(float)
        pred = y_pred.astype(float)
        if metric == "mae":
            return float(mean_absolute_error(true, pred))
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(true, pred)))
        if metric == "r2":
            return float(r2_score(true, pred))
    raise HeldoutError(f"Unauthorized comparison metric: {metric}")


def _design_clusters(rows: Sequence[dict[str, Any]]) -> tuple[list[str], dict[str, np.ndarray]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["design_id"]), []).append(index)
    require(len(groups) == TEST_DESIGNS and all(len(indices) == REALIZATIONS_PER_DESIGN for indices in groups.values()), "Bootstrap clusters are not 300 designs with five realizations")
    return sorted(groups), {design: np.asarray(indices, dtype=int) for design, indices in groups.items()}


def paired_bootstrap(rf_rows: Sequence[dict[str, Any]], tgnn_rows: Sequence[dict[str, Any]], classification: bool) -> dict[str, Any]:
    require(len(rf_rows) == len(tgnn_rows) == TEST_RUNS, "Paired comparison row count mismatch")
    rf_key = [(row["run_id"], row["design_id"], row["realization_id"], row["ground_truth"]) for row in rf_rows]
    tgnn_key = [(row["run_id"], row["design_id"], row["realization_id"], row["ground_truth"]) for row in tgnn_rows]
    require(rf_key == tgnn_key, "RF and TGNN do not share identical TEST runs, targets, or clusters")
    designs, clusters = _design_clusters(rf_rows)
    _design_clusters(tgnn_rows)
    metrics = ("balanced_accuracy", "macro_f1", "minority_class_recall", "roc_auc", "pr_auc") if classification else ("mae", "rmse", "r2")
    rf_point = {metric: _metric_value(rf_rows, metric, classification) for metric in metrics}
    tgnn_point = {metric: _metric_value(tgnn_rows, metric, classification) for metric in metrics}
    differences = {metric: rf_point[metric] - tgnn_point[metric] for metric in metrics}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = {metric: np.empty(BOOTSTRAP_REPLICATES, dtype=float) for metric in metrics}
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(designs, size=len(designs), replace=True)
        indices = np.concatenate([clusters[design] for design in sampled])
        rf_sample = [rf_rows[int(index)] for index in indices]
        tg_sample = [tgnn_rows[int(index)] for index in indices]
        for metric in metrics:
            draws[metric][replicate] = _metric_value(rf_sample, metric, classification) - _metric_value(tg_sample, metric, classification)
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    result: dict[str, Any] = {
        "method": "paired design-cluster bootstrap",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "confidence": CONFIDENCE_LEVEL,
        "resampling_unit": "design_id",
        "cluster_size": REALIZATIONS_PER_DESIGN,
        "paired_sampling": True,
        "same_sampled_designs_rf_tgnn": True,
        "ordinary_run_level_bootstrap_primary": False,
        "metrics": {},
    }
    for metric in metrics:
        lower, upper = np.quantile(draws[metric], [alpha, 1.0 - alpha])
        result["metrics"][metric] = {"rf_point_estimate": rf_point[metric], "tgnn_point_estimate": tgnn_point[metric], "point_estimate_difference": differences[metric], "absolute_paired_difference": abs(differences[metric]), "confidence_interval": {"lower": float(lower), "upper": float(upper)}, "difference_orientation": "RF minus TGNN"}
    return result


def _comparison_statistics(metrics_by_task_seed: dict[str, dict[int, dict[str, Any]]], rows_by_task_seed: dict[str, dict[int, list[dict[str, Any]]]]) -> dict[str, Any]:
    rf_class = metrics_by_task_seed["rf_space_classification"][PRIMARY_SEED]
    tg_class = metrics_by_task_seed["tgnn_space_classification"][PRIMARY_SEED]
    rf_reg = metrics_by_task_seed["rf_space_regression"][PRIMARY_SEED]
    tg_reg = metrics_by_task_seed["tgnn_space_regression"][PRIMARY_SEED]
    classification_bootstrap = paired_bootstrap(rows_by_task_seed["rf_space_classification"][PRIMARY_SEED], rows_by_task_seed["tgnn_space_classification"][PRIMARY_SEED], True)
    regression_bootstrap = paired_bootstrap(rows_by_task_seed["rf_space_regression"][PRIMARY_SEED], rows_by_task_seed["tgnn_space_regression"][PRIMARY_SEED], False)
    class_diff = rf_class["balanced_accuracy"] - tg_class["balanced_accuracy"]
    reg_diff = rf_reg["mae"] - tg_reg["mae"]
    return {
        "primary_reporting_seed": PRIMARY_SEED,
        "classification": {"rf_task": "rf_space_classification", "tgnn_task": "tgnn_space_classification", "primary_metric": "balanced_accuracy", "difference_orientation": "RF minus TGNN balanced accuracy; positive means RF higher", "rf_point_estimate": rf_class["balanced_accuracy"], "tgnn_point_estimate": tg_class["balanced_accuracy"], "point_estimate_difference": class_diff, "absolute_paired_difference": abs(class_diff), "percentage_point_difference": 100.0 * class_diff, "bootstrap": classification_bootstrap},
        "regression": {"rf_task": "rf_space_regression", "tgnn_task": "tgnn_space_regression", "primary_metric": "mae", "difference_orientation": "RF MAE minus TGNN MAE; positive means TGNN lower error", "rf_point_estimate": rf_reg["mae"], "tgnn_point_estimate": tg_reg["mae"], "point_estimate_difference": reg_diff, "absolute_paired_difference": abs(reg_diff), "relative_mae_improvement_percent": 100.0 * reg_diff / rf_reg["mae"], "bootstrap": regression_bootstrap},
    }


def _safe_output_root() -> None:
    output = OUTPUT_ROOT.resolve()
    phase3 = PHASE3_ROOT.resolve()
    phase4 = PHASE4_ROOT.resolve()
    require(output != phase3 and output != phase4, "Held-out output root aliases a frozen source root")
    require(not output.is_relative_to(phase3) and not output.is_relative_to(phase4) and not phase3.is_relative_to(output) and not phase4.is_relative_to(output), "Held-out output root overlaps a frozen source root")
    if OUTPUT_ROOT.exists():
        require(OUTPUT_ROOT.is_dir() and not any(OUTPUT_ROOT.iterdir()), f"Held-out output root must be absent or empty: {OUTPUT_ROOT}")


def preflight() -> dict[str, Any]:
    _safe_output_root()
    plan = _verify_plan()
    phase3 = _verify_phase3()
    dataset = _verify_dataset()
    phase4_inventory, phase4 = _verify_phase4_inventory()
    frozen = verify_frozen_artifacts(phase4_inventory)
    evidence = {
        "schema_version": "satnet.heldout.preflight.v1",
        "completed_at": now(),
        "phase3": phase3,
        "phase4": phase4,
        "phase4_inventory_file_sha256": PHASE4_INVENTORY_SHA,
        "phase4_bundle_sha256": PHASE4_BUNDLE_HASH,
        "dataset": dataset,
        "training_test_plan": plan,
        "frozen_models_verified": len(frozen),
        "tasks": list(ALL_TASKS),
        "authorized_seeds": list(FINAL_SEEDS),
        "primary_reporting_seed": PRIMARY_SEED,
        "ensemble": False,
        "no_retraining": True,
        "no_test_based_tuning": True,
        "no_threshold_optimization": True,
        "test_accessed": False,
        "test_targets_loaded": False,
        "test_feature_payload_loaded": False,
        "test_graph_payload_loaded": False,
        "output_root": str(OUTPUT_ROOT),
        "evaluator_git_sha": git_head(),
    }
    return evidence


def _execution_identity(evidence: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": "satnet.heldout.execution_identity.v1", "executed_at": now(), "evaluator_git_sha": git_head(), "phase4_tooling_sha": PHASE4_TOOLING_SHA, "phase4_inventory_sha256": PHASE4_INVENTORY_SHA, "phase4_bundle_sha256": PHASE4_BUNDLE_HASH, "dataset_bundle_hash": DATASET_BUNDLE_HASH, "preflight_evidence_sha256": hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(), "test_accessed": True, "test_evaluation": True, "authorized_execution_flag": "--execute-heldout"}


def run_authorized(evidence: dict[str, Any]) -> dict[str, Any]:
    _safe_output_root()
    phase4_inventory, _ = _verify_phase4_inventory()
    artifacts = {(artifact.task, artifact.seed): artifact for artifact in verify_frozen_artifacts(phase4_inventory)}
    if not OUTPUT_ROOT.exists():
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    atomic_json(OUTPUT_ROOT / "preflight_evidence.json", evidence)
    atomic_json(OUTPUT_ROOT / "execution_identity.json", _execution_identity(evidence))
    graph_metadata = _load_graph_metadata()
    metrics_by_task_seed: dict[str, dict[int, dict[str, Any]]] = {}
    rows_by_task_seed: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for task in ALL_TASKS:
        metrics_by_task_seed[task] = {}
        rows_by_task_seed[task] = {}
        for seed in FINAL_SEEDS:
            artifact = artifacts[(task, seed)]
            metrics, rows = evaluate_rf(task, seed, artifact, graph_metadata) if task.startswith("rf_") else evaluate_tgnn(task, seed, artifact, graph_metadata)
            metrics_by_task_seed[task][seed] = metrics
            rows_by_task_seed[task][seed] = rows
            task_root = OUTPUT_ROOT / task / f"seed_{seed}"
            atomic_json(task_root / "metrics.json", {"schema_version": "satnet.heldout.task_metrics.v1", "task": task, "seed": seed, "model_family": "RF" if task.startswith("rf_") else "TGNN", "selected_config_id": artifact.manifest["selected_config_id"], "checkpoint_or_model_sha256": artifact.model_sha256, "dataset_bundle_hash": DATASET_BUNDLE_HASH, "split": "test", "test_runs": TEST_RUNS, "metrics": metrics})
            _write_prediction_rows(task_root / "predictions.jsonl", rows)
    comparisons = _comparison_statistics(metrics_by_task_seed, rows_by_task_seed)
    atomic_json(OUTPUT_ROOT / "paired_comparison_statistics.json", {"schema_version": "satnet.heldout.paired_comparison.v1", **comparisons})
    summary = {"schema_version": "satnet.heldout.summary.v1", "status": "PASS", "dataset_bundle_hash": DATASET_BUNDLE_HASH, "phase3_candidate_set_hash": CANDIDATE_SET_HASH, "rf_selection_freeze_hash": RF_SELECTION_HASH, "tgnn_selection_freeze_hash": TGNN_SELECTION_HASH, "phase4_inventory_sha256": PHASE4_INVENTORY_SHA, "phase4_bundle_sha256": PHASE4_BUNDLE_HASH, "phase4_tooling_sha": PHASE4_TOOLING_SHA, "heldout_evaluator_git_sha": git_head(), "test_runs": TEST_RUNS, "test_designs": TEST_DESIGNS, "realizations_per_design": REALIZATIONS_PER_DESIGN, "authorized_model_seeds": list(FINAL_SEEDS), "primary_reporting_seed": PRIMARY_SEED, "ensemble": False, "no_retraining": True, "no_test_based_tuning": True, "no_threshold_optimization": True, "grid_fixed_source_count": 0, "adaptive_topology": list(EXPECTED_TOPOLOGY), "per_task_metrics": {task: {str(seed): metrics_by_task_seed[task][seed] for seed in FINAL_SEEDS} for task in ALL_TASKS}, "primary_rf_tgnn_comparisons": comparisons}
    atomic_json(OUTPUT_ROOT / "heldout_summary.json", summary)
    _write_report(summary)
    _write_heldout_inventory()
    return summary


def _write_report(summary: dict[str, Any]) -> None:
    classification = summary["primary_rf_tgnn_comparisons"]["classification"]
    regression = summary["primary_rf_tgnn_comparisons"]["regression"]
    report = "\n".join(("# SATNET Adaptive-v2 held-out evaluation", "", "Status: PASS", "", f"TEST runs: {TEST_RUNS}", f"TEST designs: {TEST_DESIGNS}", f"Realizations per design: {REALIZATIONS_PER_DESIGN}", f"Authorized seeds: {', '.join(str(seed) for seed in FINAL_SEEDS)}", f"Primary reporting seed: {PRIMARY_SEED}", "Ensemble: false", "Retraining: none", "TEST-based tuning: none", "Threshold optimization: none", "", "## Primary comparisons", "", f"Classification: RF minus TGNN balanced accuracy = {classification['point_estimate_difference']:.12g} percentage-point difference = {classification['percentage_point_difference']:.12g}", f"Classification bootstrap 95% CI: [{classification['bootstrap']['metrics']['balanced_accuracy']['confidence_interval']['lower']:.12g}, {classification['bootstrap']['metrics']['balanced_accuracy']['confidence_interval']['upper']:.12g}]", f"Regression: RF MAE minus TGNN MAE = {regression['point_estimate_difference']:.12g}", f"Relative MAE improvement = {regression['relative_mae_improvement_percent']:.12g}%", f"Regression bootstrap 95% CI: [{regression['bootstrap']['metrics']['mae']['confidence_interval']['lower']:.12g}, {regression['bootstrap']['metrics']['mae']['confidence_interval']['upper']:.12g}]", "", "Bootstrap: paired design-cluster bootstrap; unit design_id; five realizations travel together; 2,000 replicates; seed 20260812; confidence 0.95.", "", "Risk categories are intentionally not implemented in this technical evaluation.", ""))
    (OUTPUT_ROOT / "heldout_report.md").write_text(report, encoding="utf-8")


def _write_heldout_inventory() -> None:
    entries: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(path for path in OUTPUT_ROOT.rglob("*") if path.is_file() and path.name != "heldout_inventory.json"):
        data = path.read_bytes()
        relative = path.relative_to(OUTPUT_ROOT).as_posix()
        digest.update(relative.encode("utf-8") + b"\0" + data)
        entries.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    atomic_json(OUTPUT_ROOT / "heldout_inventory.json", {"schema_version": "satnet.heldout.inventory.v1", "inventory_self_excluding": True, "hash_algorithm": "sha256 over sorted relative UTF-8 path + NUL byte + raw file bytes", "bundle_sha256": digest.hexdigest(), "heldout_evaluator_git_sha": git_head(), "dataset_bundle_hash": DATASET_BUNDLE_HASH, "test_accessed": True, "test_evaluation": True, "artifacts": entries})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-heldout", action="store_true", help="Authorize the one-time TEST evaluation; omitted means metadata-only preflight")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    evidence = preflight()
    if not args.execute_heldout:
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return evidence
    summary = run_authorized(evidence)
    print(json.dumps({"status": summary["status"], "output_root": str(OUTPUT_ROOT), "test_accessed": True, "test_evaluation": True}, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
