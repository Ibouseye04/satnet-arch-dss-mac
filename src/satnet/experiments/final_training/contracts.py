from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PRODUCTION_TOOLING_SHA = "d0515088cf3fca06a6aa2d47059269089dcb10a7"
PRODUCTION_CONTRACT_HASH = "6c7dd365f9e7fb67f5f5e70879a19535ede55468aabfac53d82c2ab35b8307eb"
ML_CONTRACT_BUNDLE_HASH = "056d01886a9b0ac39782780a6e7444fa22638679998058c8145ea9bafb29ad2a"
DATASET_BUNDLE_HASH = "38dacd66432bfa660410ff9cab7f181a53151120e8102dc40df57016f78bbfc3"
TRAINING_PLAN_BUNDLE_HASH = "e14ec5e5b2221aa2c9aa187a2baacc517e676f9dc748612c206eb1dc43788a0a"
TRAINING_PLAN_INVENTORY_HASH = "04604051ebc5f4ce659fe558f50ed35689665f7243328ab67a6831948c11f991"
DATASET_ROOT = Path(r"C:\Users\johns\external\satnet-10k-final-ml-datasets-v1-final")
TRAINING_PLAN_ROOT = Path(r"C:\Users\johns\external\satnet-10k-training-test-plan-v1")

SPACE_FEATURES = (
    "num_planes", "sats_per_plane", "altitude_km", "inclination_deg",
    "satellite_node_failure_probability", "satellite_edge_failure_probability",
)
INTEGRATED_FEATURES = SPACE_FEATURES + (
    "civilian_count", "government_count", "military_count",
    "ground_station_failure_probability",
)
METADATA_COLUMNS = ("run_id", "run_key", "design_id", "realization_id", "split")
SPLITS = ("train", "validation", "test")
SPLIT_RUN_COUNTS = {"train": 7000, "validation": 1500, "test": 1500}
SPLIT_DESIGN_COUNTS = {"train": 1400, "validation": 300, "test": 300}
REALIZATIONS_PER_DESIGN = 5
VALIDATION_SEEDS = (42, 123, 456)
FINAL_SEEDS = (42, 123, 456, 789, 2026)
PRIMARY_REPORTING_SEED = 42
BOOTSTRAP_SEED = 20260812
BOOTSTRAP_REPLICATES = 2000

TaskFamily = Literal["RF", "TGNN"]
TaskType = Literal["classification", "regression"]

@dataclass(frozen=True)
class TaskContract:
    task_id: str
    family: TaskFamily
    task_type: TaskType
    target: str
    features: tuple[str, ...]
    dataset_relative_path: str
    target_relative_path: str | None = None

AUTHORIZED_TASKS: dict[str, TaskContract] = {
    "rf_space_classification": TaskContract("rf_space_classification", "RF", "classification", "space_threshold_breach_any", SPACE_FEATURES, "rf_space_classification/rf_space_classification.csv"),
    "rf_space_regression": TaskContract("rf_space_regression", "RF", "regression", "space_gcc_fraction_original_min", SPACE_FEATURES, "rf_space_regression/rf_space_regression.csv"),
    "tgnn_space_classification": TaskContract("tgnn_space_classification", "TGNN", "classification", "space_threshold_breach_any", (), "tgnn_space_classification/sequences", "tgnn_space_classification/tgnn_space_classification_target_manifest.jsonl"),
    "tgnn_space_regression": TaskContract("tgnn_space_regression", "TGNN", "regression", "space_gcc_fraction_original_min", (), "tgnn_space_classification/sequences", "tgnn_space_regression/tgnn_space_regression_target_manifest.jsonl"),
    "rf_integrated_regression_mean": TaskContract("rf_integrated_regression_mean", "RF", "regression", "failure_adjusted_overall_service_fraction_mean", INTEGRATED_FEATURES, "rf_integrated_regression/rf_integrated_regression.csv"),
    "rf_integrated_regression_min": TaskContract("rf_integrated_regression_min", "RF", "regression", "failure_adjusted_overall_service_fraction_min", INTEGRATED_FEATURES, "rf_integrated_regression/rf_integrated_regression.csv"),
    "rf_integrated_classification": TaskContract("rf_integrated_classification", "RF", "classification", "overall_threshold_breach_any", INTEGRATED_FEATURES, "rf_integrated_classification/rf_integrated_classification.csv"),
}


def get_task(task_id: str) -> TaskContract:
    try:
        return AUTHORIZED_TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"Unauthorized task: {task_id!r}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_dataset_bundle(root: Path = DATASET_ROOT, *, full: bool = True) -> dict[str, Any]:
    """Verify the immutable dataset inventory and, by default, every file hash."""
    inventory_path = root / "final_ml_dataset_inventory.json"
    inventory = load_json(inventory_path)
    if inventory.get("bundle_sha256") != DATASET_BUNDLE_HASH:
        raise ValueError("Dataset inventory bundle hash does not match frozen identity")
    if inventory.get("authoritative_production_sha") != PRODUCTION_TOOLING_SHA:
        raise ValueError("Dataset inventory is bound to a different production SHA")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 10022:
        raise ValueError("Dataset inventory must contain exactly 10022 artifacts")
    checked = 0
    mismatches: list[str] = []
    for artifact in artifacts:
        rel = Path(str(artifact["artifact_path"]))
        path = root / rel
        if not path.is_file() or path.stat().st_size != int(artifact["bytes"]):
            mismatches.append(str(rel))
            continue
        if full and sha256_file(path) != artifact["sha256"]:
            mismatches.append(str(rel))
        checked += 1
    if mismatches:
        raise ValueError(f"Immutable dataset verification failed for {mismatches[:5]}")
    return {"bundle_hash": DATASET_BUNDLE_HASH, "inventory_entries": len(artifacts), "files_checked": checked, "full_hashes": full}


def verify_training_plan(root: Path = TRAINING_PLAN_ROOT) -> dict[str, Any]:
    inventory_path = root / "training_plan_inventory.json"
    inventory = load_json(inventory_path)
    if inventory.get("bundle_sha256") != TRAINING_PLAN_BUNDLE_HASH:
        raise ValueError("Training-plan inventory bundle hash does not match frozen identity")
    payload = inventory_path.read_bytes()
    if sha256_bytes(payload) != TRAINING_PLAN_INVENTORY_HASH:
        raise ValueError("Training-plan inventory SHA-256 does not match frozen identity")
    mismatches: list[str] = []
    for artifact in inventory.get("artifacts", []):
        rel = Path(str(artifact["path"]))
        path = root / rel
        if not path.is_file() or path.stat().st_size != int(artifact["bytes"]) or sha256_file(path) != artifact["sha256"]:
            mismatches.append(str(rel))
    if mismatches:
        raise ValueError(f"Frozen training-plan verification failed for {mismatches[:5]}")
    return {"bundle_hash": TRAINING_PLAN_BUNDLE_HASH, "inventory_entries": len(inventory.get("artifacts", [])), "files_checked": len(inventory.get("artifacts", []))}
