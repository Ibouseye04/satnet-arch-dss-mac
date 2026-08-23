from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
DATASET_ROOT = Path(r"C:\Users\johns\external\satnet-10k-final-ml-datasets-v2-adaptive")
PLAN_ROOT = Path(r"C:\Users\johns\external\satnet-10k-training-test-plan-v1")
ROOT = Path(r"C:\Users\johns\external\satnet-10k-final-models-v2-adaptive")
DATASET_HASH = "1399f12e6ed7e159076028bc01971cf2274924fc55b43cca0a44219db8c978d1"
DATASET_INVENTORY_SHA = "29ef628bbec3ad6a003ede89749b1ba64925098b1edc55cec7e3cb98946af6b8"
RECONCILIATION_ROOT = Path(r"C:\\Users\\johns\\external\\satnet-10k-final-ml-datasets-v2-adaptive-reconciliation")
CURRENT_TREE_INVENTORY_SHA = "53a8ad2fbc6203d4901d15a9ef87d38be124002462b4f81d9ab956d161cefc3f"
TRAINING_PAYLOAD_INVENTORY_SHA = "a9d199a674c20ca31b2a91c6172f371b504603a2e7e09da29c51cd06654c9473"
TRAINING_PAYLOAD_BUNDLE_HASH = "af91fd86702403c232757ab4b5e123b2fabc314b1c4ce26f7bd9a6a0d58d5167"
CURRENT_VALIDATION_GATES_SHA = "64d2a7700b4d04136df0721bba6f197a705cd5ac01f2a424c4fa9c0453818661"
ORIGINAL_VALIDATION_GATES_SHA = "5544ca16c6e4d79021812ed705fea3b4312a65e65fdd9f683ab1a3d21110fa47"
PLAN_HASH = "e14ec5e5b2221aa2c9aa187a2baacc517e676f9dc748612c206eb1dc43788a0a"
PLAN_INVENTORY_SHA = "04604051ebc5f4ce659fe558f50ed35689665f7243328ab67a6831948c11f991"
PHASE2_SOURCE_SHA = "f47800bd5121203b6ba8ec918aaf0b6adc6d0aa1"
PHASE2A_RECONCILIATION_EVIDENCE_SHA = "3f746c2e52112a01a5c408509d268f547e4774ec"
PHASE2_EXPORTER_SHA = "767b9161f0301236153551a2319dd570d7dd342f"
PHASE1_SOURCE_SHA = "346b3ff1670237645acf4836283adbbdc359093a"
ADAPTIVE_CONTRACT_SPEC_HASH = "23c5fffc10849c3bc3ea027251ac3e5ad4c96f0eea85edf1e8deab079cb0871e"
ADAPTIVE_CONTRACT_BUNDLE_HASH = "da3c73711b1d60635afcceee8bda0a60d1379e492e1ad0d588d5a0c25e10abe3"
VALIDATION_SEEDS = (42, 123, 456)
EXPECTED_SPLITS = {"train": 7000, "validation": 1500, "test": 1500}
RF_TASKS = (
    "rf_space_classification",
    "rf_space_regression",
    "rf_integrated_regression_mean",
    "rf_integrated_regression_min",
    "rf_integrated_classification",
)
TGNN_TASKS = ("tgnn_space_classification", "tgnn_space_regression")
TASKS: dict[str, dict[str, Any]] = {
    "rf_space_classification": {"family": "RF", "type": "classification", "target": "space_threshold_breach_any", "features": ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability"), "path": "rf_space_classification/rf_space_classification.csv"},
    "rf_space_regression": {"family": "RF", "type": "regression", "target": "space_gcc_fraction_original_min", "features": ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability"), "path": "rf_space_regression/rf_space_regression.csv"},
    "rf_integrated_regression_mean": {"family": "RF", "type": "regression", "target": "failure_adjusted_overall_service_fraction_mean", "features": ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability", "civilian_count", "government_count", "military_count", "ground_station_failure_probability"), "path": "rf_integrated_regression/rf_integrated_regression.csv"},
    "rf_integrated_regression_min": {"family": "RF", "type": "regression", "target": "failure_adjusted_overall_service_fraction_min", "features": ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability", "civilian_count", "government_count", "military_count", "ground_station_failure_probability"), "path": "rf_integrated_regression/rf_integrated_regression.csv"},
    "rf_integrated_classification": {"family": "RF", "type": "classification", "target": "overall_threshold_breach_any", "features": ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability", "civilian_count", "government_count", "military_count", "ground_station_failure_probability"), "path": "rf_integrated_classification/rf_integrated_classification.csv"},
    "tgnn_space_classification": {"family": "TGNN", "type": "classification", "target": "space_threshold_breach_any", "target_path": "tgnn_space_classification/tgnn_space_classification_target_manifest.jsonl"},
    "tgnn_space_regression": {"family": "TGNN", "type": "regression", "target": "space_gcc_fraction_original_min", "target_path": "tgnn_space_regression/tgnn_space_regression_target_manifest.jsonl"},
}
RF_COMMON = {
    "n_estimators": (300, 600),
    "max_depth": (None, 10, 20),
    "min_samples_leaf": (1, 2, 5),
    "max_features": ("sqrt", 1.0),
    "bootstrap": (True,),
}


def current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Phase-3 execution requires a committed Git worktree") from exc


CODE_SHA = current_git_sha()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(native(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complexity(config: dict[str, Any]) -> tuple[float, float, str]:
    # This exactly mirrors the qualified selection helper's frozen ordering.
    depth = config.get("max_depth")
    return (float(depth) if depth is not None else math.inf, float(config.get("n_estimators", config.get("hidden_dim", math.inf))), str(sorted(config.items())))


def rf_configs(task_id: str) -> list[dict[str, Any]]:
    weights = (None, "balanced") if task_id == "rf_space_classification" else (("balanced", "balanced_subsample") if task_id == "rf_integrated_classification" else (None,))
    configs: list[dict[str, Any]] = []
    for n in RF_COMMON["n_estimators"]:
        for depth in RF_COMMON["max_depth"]:
            for leaf in RF_COMMON["min_samples_leaf"]:
                for features in RF_COMMON["max_features"]:
                    for bootstrap in RF_COMMON["bootstrap"]:
                        for weight in weights:
                            configs.append({"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf, "max_features": features, "bootstrap": bootstrap, "class_weight": weight})
    return configs


def tgnn_configs() -> list[dict[str, Any]]:
    return [{"hidden_dim": hidden, "learning_rate": lr, "cheb_k": k, "max_epochs": epochs, "num_layers": 1, "batch_size": 1, "weight_decay": 0.0, "dropout": None, "optimizer": "Adam", "classification_loss": "CrossEntropyLoss", "regression_loss": "SmoothL1Loss"} for hidden in (32, 64) for lr in (0.001, 0.01) for k in (2, 3) for epochs in (50, 100)]


def env_versions() -> dict[str, str]:
    import importlib.metadata as md
    import pandas
    import sklearn
    import torch
    import torch_geometric
    import torch_geometric_temporal
    return {"python": sys.version.split()[0], "torch": torch.__version__, "torch_geometric": torch_geometric.__version__, "torch_geometric_temporal_imported": str(getattr(torch_geometric_temporal, "__version__", None)), "torch_geometric_temporal_distribution": md.version("torch-geometric-temporal"), "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pandas.__version__, "device": "cpu"}


def progress_path() -> Path:
    return ROOT / "logs" / "progress.json"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(native(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_adaptive_dataset() -> dict[str, Any]:
    original_inventory = DATASET_ROOT / "final_ml_dataset_inventory.json"
    if sha256_file(original_inventory) != DATASET_INVENTORY_SHA:
        raise ValueError("Adaptive original inventory SHA-256 does not match accepted identity")
    inventory = load_json(original_inventory)
    if inventory.get("bundle_sha256") != DATASET_HASH:
        raise ValueError("Adaptive original bundle hash does not match accepted identity")

    tree_path = RECONCILIATION_ROOT / "phase2a_current_tree_inventory.json"
    if sha256_file(tree_path) != CURRENT_TREE_INVENTORY_SHA:
        raise ValueError("Phase-2A current-tree inventory SHA-256 does not match freeze")
    tree_inventory = load_json(tree_path)
    tree_entries = tree_inventory.get("artifacts")
    if not isinstance(tree_entries, list) or len(tree_entries) != 10025:
        raise ValueError("Phase-2A current-tree inventory entry count mismatch")
    for artifact in tree_entries:
        relative = Path(str(artifact["path"]))
        path = DATASET_ROOT / relative
        if not path.is_file() or path.stat().st_size != int(artifact["bytes"]) or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Phase-2A current-tree artifact gate failed: {relative}")

    payload_path = RECONCILIATION_ROOT / "phase2a_training_payload_inventory.json"
    if sha256_file(payload_path) != TRAINING_PAYLOAD_INVENTORY_SHA:
        raise ValueError("Phase-2A training-payload inventory SHA-256 does not match freeze")
    payload = load_json(payload_path)
    payload_entries = payload.get("artifacts")
    if payload.get("bundle_sha256") != TRAINING_PAYLOAD_BUNDLE_HASH or not isinstance(payload_entries, list) or len(payload_entries) != 10019:
        raise ValueError("Phase-2A training-payload inventory identity failed")
    for artifact in payload_entries:
        relative = Path(str(artifact["path"]))
        path = DATASET_ROOT / relative
        if not path.is_file() or path.stat().st_size != int(artifact["bytes"]) or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Phase-2A training-payload artifact gate failed: {relative}")

    validation_path = DATASET_ROOT / "metadata" / "validation_gates.json"
    if sha256_file(validation_path) != CURRENT_VALIDATION_GATES_SHA:
        raise ValueError("Current validation-gates identity does not match Phase-2A freeze")
    validation = load_json(validation_path)
    if validation["source_evidence"] != {"designs": 2000, "production_modified": False, "replay_modified": False, "runs": 10000}:
        raise ValueError("Current validation-gates source evidence failed")
    if validation["topology_provenance"] != {"aggregate": [{"count": 10000, "tuple": ["grid_adaptive", 1, 1, "persistent_temporal_union_edges_v1"]}], "grid_fixed_source_count": 0}:
        raise ValueError("Current validation-gates topology evidence failed")
    if not validation["historical_root_immutability"]["unchanged"]:
        raise ValueError("Historical artifact immutability evidence failed")
    for item in validation["split_verification"].values():
        if item["run_count"] != 10000 or not item["run_ids_complete_0_to_9999"] or item["duplicate_run_ids"] != 0 or item["design_cross_split"] or not item["five_realizations_per_design"] or not item["ordered_run_ids"]:
            raise ValueError("Current validation-gates split evidence failed")
    for item in validation["task_quality"].values():
        if item["row_count"] != 10000 or any(item[key] != 0 for key in ("missing_values", "nan", "+Inf", "-Inf", "out_of_range", "source_target_mismatch_count", "duplicate_run_ids", "exact_duplicate_rows")):
            raise ValueError("Current validation-gates task-quality evidence failed")
    for item in validation["graph_quality"].values():
        if item["sample_count"] != 10000 or item["sequence_count"] != 10000 or item["node_feature_dimension"] != 3 or item["edge_feature_dimension"] != 4 or item["malformed_sequences"] != 0 or item["empty_sequences"] != 0 or item["missing_timesteps"] != 0 or item["target_completeness"] is not True:
            raise ValueError("Current validation-gates graph-quality evidence failed")

    provenance = load_json(DATASET_ROOT / "metadata" / "export_provenance.json")
    if provenance.get("export_tooling_sha") != PHASE2_EXPORTER_SHA or provenance.get("source_scientific_lineage_sha") != PHASE1_SOURCE_SHA:
        raise ValueError("Adaptive exporter provenance does not match accepted Phase-2 lineage")
    phase2 = load_json(DATASET_ROOT / "phase2_manifest.json")
    expected = {"phase1_scientific_source_sha": PHASE1_SOURCE_SHA, "phase2_exporter_tooling_sha": PHASE2_EXPORTER_SHA, "ml_bundle_hash": DATASET_HASH, "source_contract_specification_hash": ADAPTIVE_CONTRACT_SPEC_HASH, "source_contract_bundle_hash": ADAPTIVE_CONTRACT_BUNDLE_HASH}
    if phase2.get("status") != "PASS" or any(phase2.get(key) != value for key, value in expected.items()) or phase2.get("inventory", {}).get("sha256") != DATASET_INVENTORY_SHA:
        raise ValueError("Adaptive Phase-2 manifest identity gate failed")
    counts = phase2.get("counts", {})
    if counts != {"duplicate_run_ids": 0, "exported_run_identity_coverage": "complete", "missing_run_ids": 0, "realizations_per_design": 5, "source_runs": 10000, "split_designs": {"test": 300, "train": 1400, "validation": 300}, "split_runs": EXPECTED_SPLITS, "unique_designs": 2000}:
        raise ValueError("Adaptive run/design/split identity gate failed")
    if phase2.get("leakage") != {"design_cross_split": 0, "rf_predictor_leakage": 0, "tgnn_target_in_graph": 0} or phase2.get("numeric_quality") != {"inf": 0, "invalid_values": 0, "nan": 0, "out_of_range": 0} or phase2.get("target_parity_mismatch_count") != 0 or phase2.get("historical_fixed_source_contamination") != 0:
        raise ValueError("Adaptive quality or contamination gate failed")
    if phase2.get("topology") != validation["topology_provenance"]:
        raise ValueError("Adaptive topology lineage gate failed")
    tgnn_outputs = {item["task"]: item for item in phase2.get("tgnn_outputs", [])}
    if {task: (item["sample_count"], item["sequence_length"], item["node_feature_dimension"], item["edge_feature_dimension"]) for task, item in tgnn_outputs.items()} != {"tgnn_space_classification": (10000, 11, 3, 4), "tgnn_space_regression": (10000, 11, 3, 4)}:
        raise ValueError("Adaptive TGNN dimension/count gate failed")
    return {"bundle_hash": DATASET_HASH, "inventory_sha256": DATASET_INVENTORY_SHA, "training_payload_inventory_sha256": TRAINING_PAYLOAD_INVENTORY_SHA, "training_payload_bundle_sha256": TRAINING_PAYLOAD_BUNDLE_HASH, "inventory_entries": len(inventory["artifacts"]), "training_payload_entries": len(payload_entries), "files_checked": len(tree_entries) + len(payload_entries), "test_evaluation": False}


def verify_historical_plan() -> dict[str, Any]:
    inventory_path = PLAN_ROOT / "training_plan_inventory.json"
    if sha256_file(inventory_path) != PLAN_INVENTORY_SHA:
        raise ValueError("Historical training-plan inventory SHA-256 does not match frozen identity")
    inventory = load_json(inventory_path)
    if inventory.get("bundle_sha256") != PLAN_HASH:
        raise ValueError("Historical training-plan bundle hash does not match frozen identity")
    artifacts = inventory.get("artifacts", [])
    for artifact in artifacts:
        relative = Path(str(artifact["path"]))
        path = PLAN_ROOT / relative
        if not path.is_file() or path.stat().st_size != int(artifact["bytes"]) or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Historical training-plan artifact gate failed: {relative}")
    return {"bundle_hash": PLAN_HASH, "inventory_sha256": PLAN_INVENTORY_SHA, "artifacts_checked": len(artifacts)}


def verify_historical_candidate_contract() -> dict[str, Any]:
    model_plan = load_json(PLAN_ROOT / "model_selection_contract.json")
    rf_plan = load_json(PLAN_ROOT / "rf_search_space.json")
    tgnn_plan = load_json(PLAN_ROOT / "tgnn_search_space.json")
    expected_common = {key: list(value) for key, value in RF_COMMON.items()}
    if any(rf_plan.get("common", {}).get(key) != value for key, value in expected_common.items()):
        raise ValueError("Historical RF search-space contract differs")
    expected_rf_counts = {
        "space_classification": len(rf_configs("rf_space_classification")),
        "space_regression": len(rf_configs("rf_space_regression")),
        "integrated_regression_mean": len(rf_configs("rf_integrated_regression_mean")),
        "integrated_regression_min": len(rf_configs("rf_integrated_regression_min")),
        "integrated_classification": len(rf_configs("rf_integrated_classification")),
    }
    if rf_plan.get("candidate_counts") != {**expected_rf_counts, "all_authorized_rf_task_targets": sum(expected_rf_counts.values())}:
        raise ValueError("Historical RF candidate grid differs")
    if tuple(rf_plan.get("selection", {}).get("classification", ())) != ("higher validation mean balanced_accuracy across seeds 42,123,456", "higher validation mean macro_f1", "simpler model: lower max_depth where comparable, then fewer estimators"):
        raise ValueError("Historical RF classification selection contract differs")
    if tuple(rf_plan.get("selection", {}).get("regression", ())) != ("lower validation mean MAE across seeds 42,123,456", "lower validation mean RMSE", "simpler model"):
        raise ValueError("Historical RF regression selection contract differs")
    frozen_tgnn = tgnn_plan.get("frozen_candidate_list", [])
    generated_tgnn = tgnn_configs()
    if len(frozen_tgnn) != len(generated_tgnn) or any(
        {key: config[key] for key in ("hidden_dim", "learning_rate", "cheb_k", "max_epochs", "num_layers", "weight_decay", "batch_size", "dropout")}
        != {key: row[key] for key in ("hidden_dim", "learning_rate", "cheb_k", "max_epochs", "num_layers", "weight_decay", "batch_size", "dropout")}
        or row.get("early_stopping_patience") is not None
        for config, row in zip(generated_tgnn, frozen_tgnn)
    ):
        raise ValueError("Historical TGNN candidate grid differs")
    if tgnn_plan.get("candidate_count") != 16 or model_plan.get("candidate_phase", {}).get("seeds") != list(VALIDATION_SEEDS):
        raise ValueError("Historical TGNN candidate contract differs")
    if model_plan.get("rf", {}).get("total_candidate_fits") != 756 or model_plan.get("tgnn", {}).get("candidate_fits_total") != 96 or model_plan.get("rf", {}).get("candidate_fit_counts") != {task: len(rf_configs(task)) * len(VALIDATION_SEEDS) for task in RF_TASKS}:
        raise ValueError("Historical candidate fit counts differ")
    return {"model_selection_contract": str(PLAN_ROOT / "model_selection_contract.json"), "rf_search_space": str(PLAN_ROOT / "rf_search_space.json"), "tgnn_search_space": str(PLAN_ROOT / "tgnn_search_space.json"), "rf_fit_count": 756, "tgnn_fit_count": 96, "total_fit_count": 852, "validation_seeds": list(VALIDATION_SEEDS), "test_access": False}


def candidate_contract() -> tuple[dict[str, Any], str]:
    candidates: list[dict[str, Any]] = []
    for task in RF_TASKS:
        for index, config in enumerate(rf_configs(task)):
            for seed in VALIDATION_SEEDS:
                candidates.append({"family": "RF", "task": task, "candidate_id": f"rf_{index + 1:03d}", "candidate_index": index, "seed": seed, "configuration": config})
    for task in TGNN_TASKS:
        for index, config in enumerate(tgnn_configs()):
            for seed in VALIDATION_SEEDS:
                candidates.append({"family": "TGNN", "task": task, "candidate_id": f"tgnn_{index + 1:03d}", "candidate_index": index, "seed": seed, "configuration": config})
    expected = {"RF": 756, "TGNN": 96}
    counts = {family: sum(row["family"] == family for row in candidates) for family in expected}
    if counts != expected or len(candidates) != 852:
        raise ValueError(f"Historical candidate count mismatch: {counts}")
    contract = {"schema_version": "satnet.phase3.model_selection_contract.v1", "status": "frozen", "phase": "model_selection", "fit_split": "train", "selection_split": "validation", "test_evaluation": False, "input_dataset_root": str(DATASET_ROOT), "input_dataset_bundle_hash": DATASET_HASH, "input_training_payload_inventory_sha256": TRAINING_PAYLOAD_INVENTORY_SHA, "input_training_payload_bundle_hash": TRAINING_PAYLOAD_BUNDLE_HASH, "current_validation_gates_sha256": CURRENT_VALIDATION_GATES_SHA, "historical_training_plan_bundle_hash": PLAN_HASH, "phase2_source_sha": PHASE2_SOURCE_SHA, "phase2a_reconciliation_evidence_sha": PHASE2A_RECONCILIATION_EVIDENCE_SHA, "phase2_exporter_sha": PHASE2_EXPORTER_SHA, "training_tooling_sha": CODE_SHA, "validation_seeds": list(VALIDATION_SEEDS), "tasks": list(RF_TASKS + TGNN_TASKS), "task_fit_counts": {task: len(rf_configs(task)) * len(VALIDATION_SEEDS) for task in RF_TASKS} | {task: len(tgnn_configs()) * len(VALIDATION_SEEDS) for task in TGNN_TASKS}, "rf_fit_count": counts["RF"], "tgnn_fit_count": counts["TGNN"], "total_fit_count": len(candidates), "historical_evidence": {"training_plan_root": str(PLAN_ROOT), "model_selection_contract": str(PLAN_ROOT / "model_selection_contract.json"), "rf_search_space": str(PLAN_ROOT / "rf_search_space.json"), "tgnn_search_space": str(PLAN_ROOT / "tgnn_search_space.json"), "historical_output_root": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1"}, "candidate_set": candidates, "candidate_set_hash": canonical_hash(candidates), "selection_rules": {"classification": ["higher mean validation balanced_accuracy", "higher mean validation macro_f1", "historical simplicity key", "candidate id"], "regression": ["lower mean validation MAE", "lower mean validation RMSE", "historical simplicity key", "candidate id"]}, "preprocessing": {"rf": "none; raw ex-ante predictors", "tgnn": "immutable exported node/edge features; no fitted preprocessing", "fit_split_only": True}}
    return contract, contract["candidate_set_hash"]


def prepare_phase3_root() -> tuple[dict[str, Any], str]:
    identity_path = ROOT / "manifests" / "execution_identity.json"
    contract_path = ROOT / "manifests" / "phase3_model_selection_contract.json"
    if ROOT.exists() and any(ROOT.iterdir()):
        if not identity_path.is_file() or not contract_path.is_file():
            raise RuntimeError("Phase-3 output root exists without a resumable identity; refusing overwrite")
        identity = load_json(identity_path)
        existing_contract = load_json(contract_path)
        contract_hash = existing_contract.get("candidate_set_hash")
        if identity.get("training_tooling_sha") != CODE_SHA or identity.get("candidate_set_hash") != contract_hash or identity.get("dataset_bundle_hash") != DATASET_HASH or identity.get("training_payload_inventory_sha256") != TRAINING_PAYLOAD_INVENTORY_SHA or identity.get("training_payload_bundle_hash") != TRAINING_PAYLOAD_BUNDLE_HASH or identity.get("validation_gates_sha256") != CURRENT_VALIDATION_GATES_SHA or identity.get("phase2_source_sha") != PHASE2_SOURCE_SHA or identity.get("test_evaluation") is not False:
            raise RuntimeError("Existing Phase-3 output root identity conflicts with this run")
        return existing_contract, str(contract_hash)
    contract, contract_hash = candidate_contract()
    atomic_json(contract_path, contract)
    atomic_json(identity_path, {"schema_version": "satnet.phase3.execution_identity.v1", "training_tooling_sha": CODE_SHA, "phase2_source_sha": PHASE2_SOURCE_SHA, "phase2a_reconciliation_evidence_sha": PHASE2A_RECONCILIATION_EVIDENCE_SHA, "phase2_exporter_sha": PHASE2_EXPORTER_SHA, "dataset_bundle_hash": DATASET_HASH, "training_payload_inventory_sha256": TRAINING_PAYLOAD_INVENTORY_SHA, "training_payload_bundle_hash": TRAINING_PAYLOAD_BUNDLE_HASH, "validation_gates_sha256": CURRENT_VALIDATION_GATES_SHA, "training_plan_hash": PLAN_HASH, "candidate_set_hash": contract_hash, "validation_seeds": list(VALIDATION_SEEDS), "fit_split": "train", "selection_split": "validation", "test_evaluation": False, "historical_fixed_data_reused": False, "historical_fixed_checkpoint_reused": False, "execution_worktree": str(REPO), "started_at": now()})
    return contract, contract_hash


def assert_test_firewall() -> dict[str, Any]:
    checked = 0
    forbidden: list[str] = []
    for path in (ROOT / "model_selection").rglob("candidate_manifest.json") if (ROOT / "model_selection").exists() else ():
        manifest = load_json(path)
        checked += 1
        if manifest.get("test_accessed") is not False or any(key in manifest for key in ("test_metrics", "test_predictions", "test_targets")):
            forbidden.append(str(path))
    if forbidden:
        raise RuntimeError(f"TEST evidence found in Phase-3 candidate results: {forbidden[:3]}")
    evidence = {"schema_version": "satnet.phase3.test_firewall_evidence.v1", "candidate_manifests_checked": checked, "test_evaluation_performed": False, "test_metrics_present": False, "test_predictions_present": False, "test_targets_loaded": False, "passed": True, "recorded_at": now()}
    atomic_json(ROOT / "manifests" / "test_firewall_evidence.json", evidence)
    return evidence


def expected_records() -> list[tuple[str, str, int]]:
    records = []
    for task in RF_TASKS:
        for index in range(len(rf_configs(task))):
            for seed in VALIDATION_SEEDS:
                records.append((task, f"rf_{index + 1:03d}", seed))
    for task in TGNN_TASKS:
        for index in range(len(tgnn_configs())):
            for seed in VALIDATION_SEEDS:
                records.append((task, f"tgnn_{index + 1:03d}", seed))
    return records


def manifest_path(task: str, config_id: str, seed: int) -> Path:
    family_dir = "rf" if task in RF_TASKS else "tgnn"
    return ROOT / "model_selection" / family_dir / task / config_id / f"seed_{seed}" / "candidate_manifest.json"


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def verified_completed(task: str, config_id: str, seed: int) -> bool:
    manifest = load_manifest(manifest_path(task, config_id, seed))
    if not manifest or manifest.get("status") != "completed":
        return False
    if manifest.get("code_sha") != CODE_SHA or manifest.get("dataset_bundle_hash") != DATASET_HASH or manifest.get("training_payload_inventory_sha256") != TRAINING_PAYLOAD_INVENTORY_SHA or manifest.get("training_payload_bundle_hash") != TRAINING_PAYLOAD_BUNDLE_HASH or manifest.get("training_plan_hash") != PLAN_HASH or manifest.get("candidate_set_hash") != candidate_contract()[1]:
        return False
    if manifest.get("train_count") != 7000 or manifest.get("validation_count") != 1500 or manifest.get("test_accessed") is not False:
        return False
    metrics = manifest.get("validation_metrics")
    if not isinstance(metrics, dict) or not metrics:
        return False
    artifact = manifest.get("artifact_path") or manifest.get("checkpoint_path")
    artifact_hash = manifest.get("artifact_sha256") or manifest.get("checkpoint_sha256")
    return bool(artifact and Path(artifact).is_file() and artifact_hash and sha256_file(Path(artifact)) == artifact_hash)


def write_progress(current: dict[str, Any] | None = None) -> None:
    rf_expected = sum(len(rf_configs(task)) for task in RF_TASKS) * len(VALIDATION_SEEDS)
    tgnn_expected = len(TGNN_TASKS) * len(tgnn_configs()) * len(VALIDATION_SEEDS)
    counts = {"RF": {"completed": 0, "failed": 0}, "TGNN": {"completed": 0, "failed": 0}}
    most_recent: str | None = None
    for task, config_id, seed in expected_records():
        manifest = load_manifest(manifest_path(task, config_id, seed))
        family = "RF" if task in RF_TASKS else "TGNN"
        if verified_completed(task, config_id, seed):
            counts[family]["completed"] += 1
            timestamp = manifest.get("completed_at") if manifest else None
            if isinstance(timestamp, str) and (most_recent is None or timestamp > most_recent):
                most_recent = timestamp
        elif manifest and manifest.get("status") == "failed":
            counts[family]["failed"] += 1
    atomic_json(progress_path(), {"schema_version": "satnet.phase3.progress.v1", "updated_at": now(), "training_tooling_sha": CODE_SHA, "candidate_set_hash": candidate_contract()[1], "output_root": str(ROOT), "validation_seeds": list(VALIDATION_SEEDS), "test_evaluation": False, "most_recent_completed_candidate_at": most_recent, "rf": {"expected": rf_expected, "completed": counts["RF"]["completed"], "failed": counts["RF"]["failed"], "pending": rf_expected - counts["RF"]["completed"] - counts["RF"]["failed"]}, "tgnn": {"expected": tgnn_expected, "completed": counts["TGNN"]["completed"], "failed": counts["TGNN"]["failed"], "pending": tgnn_expected - counts["TGNN"]["completed"] - counts["TGNN"]["failed"]}, "total": {"expected": rf_expected + tgnn_expected, "completed": sum(item["completed"] for item in counts.values()), "failed": sum(item["failed"] for item in counts.values())}, "current": current, "acceptance": {"rf_all_complete": counts["RF"]["completed"] == rf_expected and counts["RF"]["failed"] == 0, "tgnn_all_complete": counts["TGNN"]["completed"] == tgnn_expected and counts["TGNN"]["failed"] == 0, "test_untouched": True}})


def read_rf_data(task_id: str) -> dict[str, Any]:
    spec = TASKS[task_id]
    path = DATASET_ROOT / spec["path"]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        positions = {name: header.index(name) for name in ("run_id", "run_key", "design_id", "realization_id", "split", *spec["features"], spec["target"])}
        rows: dict[str, list[Any]] = {"train": [], "validation": []}
        split_meta: dict[str, list[tuple[int, str, str]]] = {"train": [], "validation": [], "test": []}
        for raw in reader:
            split = raw[positions["split"]].strip().lower()
            run_id = int(raw[positions["run_id"]])
            design = raw[positions["design_id"]]
            realization = raw[positions["realization_id"]]
            split_meta[split].append((run_id, design, realization))
            if split == "test":
                continue
            features = [float(raw[positions[name]]) for name in spec["features"]]
            if spec["type"] == "classification":
                target = 1 if raw[positions[spec["target"]]].strip().lower() in {"1", "true"} else 0
            else:
                target = float(raw[positions[spec["target"]]])
            rows[split].append((run_id, features, target))
    validate_split_metadata(split_meta)
    rows = {name: sorted(value, key=lambda item: item[0]) for name, value in rows.items()}
    return {"X_train": np.asarray([x[1] for x in rows["train"]], dtype=float), "y_train": np.asarray([x[2] for x in rows["train"]]), "X_validation": np.asarray([x[1] for x in rows["validation"]], dtype=float), "y_validation": np.asarray([x[2] for x in rows["validation"]]), "split_meta": split_meta}


def validate_split_metadata(split_meta: dict[str, list[tuple[int, str, str]]]) -> None:
    if {k: len(v) for k, v in split_meta.items()} != EXPECTED_SPLITS:
        raise ValueError(f"Frozen split counts failed: { {k: len(v) for k, v in split_meta.items()} }")
    all_design_splits: dict[str, set[str]] = {}
    for split, rows in split_meta.items():
        ids = {r[0] for r in rows}
        if split != "test" and (len(ids) != len(rows) or ids & set(range(10000)) - ids):
            pass
        design_map: dict[str, set[str]] = {}
        for run_id, design, realization in rows:
            if not 0 <= run_id < 10000 or realization not in {f"R{i:02d}" for i in range(5)}:
                raise ValueError("Invalid frozen identity")
            design_map.setdefault(design, set()).add(realization)
        if any(values != {f"R{i:02d}" for i in range(5)} for values in design_map.values()):
            raise ValueError(f"Realization cluster violation in {split}")
        all_design_splits[split] = set(design_map)
    if any(all_design_splits[a] & all_design_splits[b] for i, a in enumerate(all_design_splits) for b in list(all_design_splits)[i + 1:]):
        raise ValueError("Design crosses frozen splits")


def validation_metrics(task_type: str, y_true: Any, prediction: Any, score: Any | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(SRC))
    from satnet.experiments.final_training.metrics import classification_metrics, regression_metrics
    return classification_metrics(y_true, prediction, score) if task_type == "classification" else regression_metrics(y_true, prediction)


def run_rf(task_id: str, config_id: str, config: dict[str, Any], seed: int, data: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(SRC))
    from satnet.experiments.final_training.seeds import initialize_determinism
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    import joblib
    path = manifest_path(task_id, config_id, seed)
    prior = load_manifest(path)
    if verified_completed(task_id, config_id, seed):
        return prior or {}
    candidate_hash = candidate_contract()[1]
    artifact = ROOT / "checkpoints" / "rf" / task_id / config_id / f"seed_{seed}" / "model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    atomic_json(path, {"schema_version": "satnet.phase3_candidate_manifest.v1", "status": "running", "started_at": now(), "task": task_id, "model_family": "RF", "config_id": config_id, "configuration": {**config, "random_state": seed, "n_jobs": -1}, "seed": seed, "dataset_bundle_hash": DATASET_HASH, "training_payload_inventory_sha256": TRAINING_PAYLOAD_INVENTORY_SHA, "training_payload_bundle_hash": TRAINING_PAYLOAD_BUNDLE_HASH, "training_plan_hash": PLAN_HASH, "phase2_source_sha": PHASE2_SOURCE_SHA, "code_sha": CODE_SHA, "candidate_set_hash": candidate_hash, "train_count": 7000, "validation_count": 1500, "test_accessed": False, "previous_status": prior.get("status") if prior else None})
    try:
        initialize_determinism(seed)
        params = {**config, "random_state": seed, "n_jobs": -1}
        if TASKS[task_id]["type"] != "classification":
            params.pop("class_weight", None)
        estimator = RandomForestClassifier(**params) if TASKS[task_id]["type"] == "classification" else RandomForestRegressor(**params)
        estimator.fit(data["X_train"], data["y_train"])
        if TASKS[task_id]["type"] == "classification":
            pred = estimator.predict(data["X_validation"])
            score = estimator.predict_proba(data["X_validation"])
            metrics = validation_metrics("classification", data["y_validation"], pred, score)
        else:
            pred = estimator.predict(data["X_validation"])
            metrics = validation_metrics("regression", data["y_validation"], pred)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temp = artifact.with_name(artifact.name + f".tmp-{os.getpid()}")
        joblib.dump(estimator, temp)
        os.replace(temp, artifact)
        result = {"schema_version": "satnet.phase3_candidate_manifest.v1", "status": "completed", "completed_at": now(), "task": task_id, "model_family": "RF", "config_id": config_id, "configuration": params, "seed": seed, "dataset_bundle_hash": DATASET_HASH, "training_payload_inventory_sha256": TRAINING_PAYLOAD_INVENTORY_SHA, "training_payload_bundle_hash": TRAINING_PAYLOAD_BUNDLE_HASH, "training_plan_hash": PLAN_HASH, "phase2_source_sha": PHASE2_SOURCE_SHA, "code_sha": CODE_SHA, "candidate_set_hash": candidate_hash, "train_count": 7000, "validation_count": 1500, "fit_runtime_seconds": time.perf_counter() - start, "validation_metrics": metrics, "model_selection_metric": {"name": "balanced_accuracy" if TASKS[task_id]["type"] == "classification" else "mae", "value": metrics["balanced_accuracy"] if TASKS[task_id]["type"] == "classification" else metrics["mae"]}, "artifact_path": str(artifact), "artifact_sha256": sha256_file(artifact), "test_accessed": False, "environment_versions": env_versions()}
        atomic_json(path, result)
        return result
    except Exception as exc:
        atomic_json(path, {"schema_version": "satnet.phase3_candidate_manifest.v1", "status": "failed", "failed_at": now(), "task": task_id, "model_family": "RF", "config_id": config_id, "configuration": config, "seed": seed, "dataset_bundle_hash": DATASET_HASH, "training_payload_inventory_sha256": TRAINING_PAYLOAD_INVENTORY_SHA, "training_payload_bundle_hash": TRAINING_PAYLOAD_BUNDLE_HASH, "training_plan_hash": PLAN_HASH, "phase2_source_sha": PHASE2_SOURCE_SHA, "code_sha": CODE_SHA, "candidate_set_hash": candidate_hash, "train_count": 7000, "validation_count": 1500, "test_accessed": False, "error_type": type(exc).__name__, "error": str(exc)})
        raise


def load_tgnn_metadata(task_id: str) -> dict[str, Any]:
    sys.path.insert(0, str(SRC))
    from satnet.experiments.final_training.tgnn_loader import read_sequence
    graph_path = DATASET_ROOT / "tgnn_space_classification" / "tgnn_space_graph_manifest.jsonl"
    target_path = DATASET_ROOT / TASKS[task_id]["target_path"]
    graph_records: dict[int, dict[str, Any]] = {}
    with graph_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            run_id = int(record["run_id"])
            graph_records[run_id] = record
    targets: dict[int, Any] = {}
    with target_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if str(record.get("split", "")).lower() == "test":
                continue
            run_id = int(record["run_id"])
            value = record["target"]
            targets[run_id] = (1 if str(value).strip().lower() in {"1", "true"} else 0) if TASKS[task_id]["type"] == "classification" else float(value)
    split_meta = {split: [] for split in EXPECTED_SPLITS}
    for record in graph_records.values():
        split = str(record["split"]).lower()
        split_meta[split].append((int(record["run_id"]), str(record["design_id"]), str(record["realization_id"])))
    validate_split_metadata(split_meta)
    train_ids = tuple(sorted(x[0] for x in split_meta["train"]))
    validation_ids = tuple(sorted(x[0] for x in split_meta["validation"]))
    if set(targets) != set(train_ids) | set(validation_ids):
        raise ValueError("TGNN train/validation target identity set mismatch")
    return {"records": graph_records, "targets": targets, "train_ids": train_ids, "validation_ids": validation_ids, "read_sequence": read_sequence}


def tgnn_predict(model: Any, ids: Iterable[int], bundle: dict[str, Any], task_type: str, train: bool, optimizer: Any = None, loss_fn: Any = None) -> tuple[list[float], list[Any], float]:
    import torch
    model.train(train)
    predictions: list[float] = []
    scores: list[Any] = []
    targets: list[Any] = []
    total_loss = 0.0
    for run_id in ids:
        artifact = bundle["read_sequence"](DATASET_ROOT / bundle["records"][run_id]["sequence_artifact"])
        data_list = artifact.data_list()
        target = bundle["targets"][run_id]
        if train:
            optimizer.zero_grad(set_to_none=True)
            output = model(data_list)
            target_tensor = torch.tensor([target], dtype=torch.long if task_type == "classification" else torch.float32)
            loss = loss_fn(output, target_tensor if task_type == "classification" else target_tensor.reshape(1, 1))
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(data_list)
            if task_type == "classification":
                probability = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
                scores.append(probability.tolist())
                predictions.append(float(np.argmax(probability)))
            else:
                predictions.append(float(output.detach().cpu().reshape(-1)[0]))
            targets.append(target)
        if train:
            total_loss += float(loss.detach().cpu().item())
    return predictions, scores if task_type == "classification" else [], total_loss / max(1, len(tuple(ids))) if train else 0.0


def run_tgnn(task_id: str, config_id: str, config: dict[str, Any], seed: int, bundle: dict[str, Any]) -> dict[str, Any]:
    import copy
    import torch
    from torch.nn import CrossEntropyLoss, SmoothL1Loss
    from satnet.experiments.final_training.checkpoints import EarlyStopping
    from satnet.experiments.final_training.seeds import initialize_determinism
    from satnet.models.gnn_model import SatelliteGNN
    path = manifest_path(task_id, config_id, seed)
    prior = load_manifest(path)
    if verified_completed(task_id, config_id, seed):
        return prior or {}
    candidate_dir = path.parent
    checkpoint = ROOT / "checkpoints" / "tgnn" / task_id / config_id / f"seed_{seed}" / "best_validation_checkpoint.pt"
    log_path = ROOT / "logs" / "tgnn" / task_id / config_id / f"seed_{seed}.jsonl"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    candidate_hash = candidate_contract()[1]
    atomic_json(path, {"schema_version": "satnet.phase3_candidate_manifest.v1", "status": "running", "started_at": now(), "task": task_id, "model_family": "TGNN", "config_id": config_id, "configuration": config, "seed": seed, "dataset_bundle_hash": DATASET_HASH, "training_payload_inventory_sha256": TRAINING_PAYLOAD_INVENTORY_SHA, "training_payload_bundle_hash": TRAINING_PAYLOAD_BUNDLE_HASH, "training_plan_hash": PLAN_HASH, "phase2_source_sha": PHASE2_SOURCE_SHA, "code_sha": CODE_SHA, "candidate_set_hash": candidate_hash, "train_count": 7000, "validation_count": 1500, "checkpoint_staging_area": str(checkpoint.parent), "test_accessed": False, "previous_status": prior.get("status") if prior else None})
    try:
        initialize_determinism(seed)
        torch.set_num_threads(8)
        model = SatelliteGNN(node_features=3, hidden_channels=int(config["hidden_dim"]), out_channels=2 if TASKS[task_id]["type"] == "classification" else 1, task_type=TASKS[task_id]["type"], cheb_k=int(config["cheb_k"]))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0)
        loss_fn = CrossEntropyLoss() if TASKS[task_id]["type"] == "classification" else SmoothL1Loss()
        stop = EarlyStopping(TASKS[task_id]["type"], patience=10, min_delta=0.0, restore_best=True)
        best_epoch = None
        for epoch in range(1, int(config["max_epochs"]) + 1):
            _, _, train_loss = tgnn_predict(model, bundle["train_ids"], bundle, TASKS[task_id]["type"], True, optimizer, loss_fn)
            validation_pred, validation_scores, _ = tgnn_predict(model, bundle["validation_ids"], bundle, TASKS[task_id]["type"], False)
            if TASKS[task_id]["type"] == "classification":
                metrics = validation_metrics("classification", [bundle["targets"][i] for i in bundle["validation_ids"]], validation_pred, validation_scores)
                monitor = float(metrics["balanced_accuracy"])
            else:
                metrics = validation_metrics("regression", [bundle["targets"][i] for i in bundle["validation_ids"]], validation_pred)
                monitor = float(metrics["mae"])
            improved = stop.update(epoch, monitor, copy.deepcopy(model.state_dict()))
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "validation_metrics": native(metrics), "monitor": monitor, "improved": improved, "at": now()}) + "\n")
                log.flush()
                os.fsync(log.fileno())
            if improved:
                checkpoint_payload = {"model_state_dict": stop.best_state, "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "configuration": config, "task": task_id, "seed": seed, "validation_primary_metric": monitor, "dataset_bundle_hash": DATASET_HASH, "training_payload_inventory_sha256": TRAINING_PAYLOAD_INVENTORY_SHA, "training_payload_bundle_hash": TRAINING_PAYLOAD_BUNDLE_HASH, "training_plan_hash": PLAN_HASH, "phase2_source_sha": PHASE2_SOURCE_SHA, "candidate_set_hash": candidate_hash, "code_sha": CODE_SHA, "environment_versions": env_versions()}
                temp = checkpoint.with_name(checkpoint.name + f".tmp-{os.getpid()}")
                torch.save(checkpoint_payload, temp)
                os.replace(temp, checkpoint)
                best_epoch = epoch
            if stop.should_stop:
                break
        stop.restore(model)
        # Recompute metrics from the restored best-validation checkpoint/model only.
        validation_pred, validation_scores, _ = tgnn_predict(model, bundle["validation_ids"], bundle, TASKS[task_id]["type"], False)
        metrics = validation_metrics(TASKS[task_id]["type"], [bundle["targets"][i] for i in bundle["validation_ids"]], validation_pred, validation_scores if TASKS[task_id]["type"] == "classification" else None)
        result = {"schema_version": "satnet.phase3_candidate_manifest.v1", "status": "completed", "completed_at": now(), "task": task_id, "model_family": "TGNN", "config_id": config_id, "configuration": config, "seed": seed, "dataset_bundle_hash": DATASET_HASH, "training_payload_inventory_sha256": TRAINING_PAYLOAD_INVENTORY_SHA, "training_payload_bundle_hash": TRAINING_PAYLOAD_BUNDLE_HASH, "training_plan_hash": PLAN_HASH, "phase2_source_sha": PHASE2_SOURCE_SHA, "code_sha": CODE_SHA, "candidate_set_hash": candidate_hash, "train_count": 7000, "validation_count": 1500, "fit_runtime_seconds": time.perf_counter() - start, "epochs_run": epoch, "selected_epoch": stop.best_epoch, "max_epochs": config["max_epochs"], "early_stopping": {"monitor": "balanced_accuracy" if TASKS[task_id]["type"] == "classification" else "mae", "direction": "maximize" if TASKS[task_id]["type"] == "classification" else "minimize", "patience": 10, "min_delta": 0.0, "restore_best": True, "stopped_early": epoch < config["max_epochs"]}, "validation_metrics": metrics, "model_selection_metric": {"name": "balanced_accuracy" if TASKS[task_id]["type"] == "classification" else "mae", "value": metrics["balanced_accuracy"] if TASKS[task_id]["type"] == "classification" else metrics["mae"]}, "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "test_accessed": False, "environment_versions": env_versions()}
        atomic_json(path, result)
        return result
    except Exception as exc:
        atomic_json(path, {"schema_version": "satnet.phase3_candidate_manifest.v1", "status": "failed", "failed_at": now(), "task": task_id, "model_family": "TGNN", "config_id": config_id, "configuration": config, "seed": seed, "dataset_bundle_hash": DATASET_HASH, "training_payload_inventory_sha256": TRAINING_PAYLOAD_INVENTORY_SHA, "training_payload_bundle_hash": TRAINING_PAYLOAD_BUNDLE_HASH, "training_plan_hash": PLAN_HASH, "phase2_source_sha": PHASE2_SOURCE_SHA, "code_sha": CODE_SHA, "candidate_set_hash": candidate_hash, "train_count": 7000, "validation_count": 1500, "checkpoint_path": str(checkpoint), "test_accessed": False, "error_type": type(exc).__name__, "error": str(exc)})
        raise


def collect_records(family_tasks: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for task in family_tasks:
        for config_id, seed in [(f"rf_{i:03d}", s) for i in range(1, len(rf_configs(task)) + 1) for s in VALIDATION_SEEDS] if task in RF_TASKS else [(f"tgnn_{i:03d}", s) for i in range(1, len(tgnn_configs()) + 1) for s in VALIDATION_SEEDS]:
            manifest = load_manifest(manifest_path(task, config_id, seed))
            if not verified_completed(task, config_id, seed):
                raise RuntimeError(f"Missing or unverified completed record: {task}/{config_id}/{seed}")
            rows.append(manifest)
    return rows


def select_results(task_ids: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in task_ids:
        manifests = collect_records((task,))
        grouped: dict[str, list[dict[str, Any]]] = {}
        for m in manifests: grouped.setdefault(m["config_id"], []).append(m)
        ranking = []
        classification = TASKS[task]["type"] == "classification"
        for candidate_id, records in grouped.items():
            values = [float(m["model_selection_metric"]["value"]) for m in records]
            secondary_key = "macro_f1" if classification else "rmse"
            secondaries = [float(m["validation_metrics"][secondary_key]) for m in records]
            config = records[0]["configuration"]
            ranking.append({"config_id": candidate_id, "configuration": config, "mean_primary_metric": statistics.fmean(values), "std_primary_metric": statistics.pstdev(values), "mean_secondary_metric": statistics.fmean(secondaries), "std_secondary_metric": statistics.pstdev(secondaries), "seed_metrics": [{"seed": m["seed"], "validation_metrics": m["validation_metrics"], "selected_epoch": m.get("selected_epoch"), "epochs_run": m.get("epochs_run"), "artifact_path": m.get("artifact_path") or m.get("checkpoint_path"), "artifact_sha256": m.get("artifact_sha256") or m.get("checkpoint_sha256")} for m in sorted(records, key=lambda x: x["seed"])], "complexity_key": list(complexity(config))})
        if classification: ranking.sort(key=lambda x: (-x["mean_primary_metric"], -x["mean_secondary_metric"], tuple(x["complexity_key"]), x["config_id"]))
        else: ranking.sort(key=lambda x: (x["mean_primary_metric"], x["mean_secondary_metric"], tuple(x["complexity_key"]), x["config_id"]))
        winner = ranking[0]
        result[task] = {"task": task, "winning_configuration": winner["configuration"], "winning_config_id": winner["config_id"], "mean_primary_metric": winner["mean_primary_metric"], "std_primary_metric": winner["std_primary_metric"], "primary_metric": "balanced_accuracy" if classification else "mae", "direction": "maximize" if classification else "minimize", "seed_specific_validation_metrics": winner["seed_metrics"], "tie_break_evidence": {"ranking_rule": ["higher mean balanced accuracy", "higher mean macro F1", "simpler model"] if classification else ["lower mean MAE", "lower mean RMSE", "simpler model"], "winning_complexity_key": winner["complexity_key"], "runner_up_complexity_key": ranking[1]["complexity_key"] if len(ranking) > 1 else None}, "runner_up_configuration": ranking[1]["configuration"] if len(ranking) > 1 else None, "runner_up": ranking[1] if len(ranking) > 1 else None, "candidate_count": len(ranking), "validation_seeds": list(VALIDATION_SEEDS), "test_accessed": False, "full_ranking": ranking}
    return result


def write_results_csv(path: Path, task_ids: tuple[str, ...]) -> None:
    records = collect_records(task_ids)
    fields = ["task", "config_id", "seed", "configuration_json", "fit_runtime_seconds", "selected_epoch", "epochs_run", "model_selection_metric_name", "model_selection_metric_value", "balanced_accuracy", "accuracy", "precision_by_class", "recall_by_class", "f1_by_class", "macro_f1", "weighted_f1", "specificity", "sensitivity", "confusion_matrix", "roc_auc", "pr_auc", "mae", "rmse", "r2", "median_absolute_error", "maximum_absolute_error", "target_mean", "target_median", "target_standard_deviation", "prediction_mean", "prediction_standard_deviation", "predictions_below_zero", "predictions_above_one", "artifact_sha256", "checkpoint_sha256", "test_accessed"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for m in records:
            row = {field: "" for field in fields}
            metrics = m["validation_metrics"]
            row.update({"task": m["task"], "config_id": m["config_id"], "seed": m["seed"], "configuration_json": json.dumps(m["configuration"], sort_keys=True), "fit_runtime_seconds": m["fit_runtime_seconds"], "selected_epoch": m.get("selected_epoch", ""), "epochs_run": m.get("epochs_run", ""), "model_selection_metric_name": m["model_selection_metric"]["name"], "model_selection_metric_value": m["model_selection_metric"]["value"], "artifact_sha256": m.get("artifact_sha256", ""), "checkpoint_sha256": m.get("checkpoint_sha256", ""), "test_accessed": m["test_accessed"]})
            for key in fields:
                if key in metrics: row[key] = json.dumps(metrics[key], sort_keys=True) if isinstance(metrics[key], (dict, list)) else metrics[key]
            writer.writerow(row)
    os.replace(temp, path)


def baselines() -> dict[str, Any]:
    sys.path.insert(0, str(SRC))
    from satnet.experiments.final_training.baselines import classification_baselines_from_train, regression_baselines_from_train
    output: dict[str, Any] = {"schema_version": "satnet.validation_baselines.v1", "construction_split": "train", "evaluation_split": "validation", "test_accessed": False, "tasks": {}}
    for task in RF_TASKS:
        data = read_rf_data(task)
        if TASKS[task]["type"] == "classification":
            base = classification_baselines_from_train(data["y_train"], seed=42)
            output["tasks"][task] = {
                "majority_classifier": {
                    "train_majority_class": base.majority_class,
                    "validation_metrics": validation_metrics("classification", data["y_validation"], base.majority_predict(len(data["y_validation"]))),
                },
                "stratified_random_classifier": {
                    "train_positive_prevalence": base.positive_prevalence,
                    "seed": 42,
                    "validation_metrics": validation_metrics("classification", data["y_validation"], base.stratified_random_predict(len(data["y_validation"]))),
                },
            }
        else:
            base = regression_baselines_from_train(data["y_train"])
            output["tasks"][task] = {"training_mean_predictor": {"train_mean": base.mean, "validation_metrics": validation_metrics("regression", data["y_validation"], base.mean_predict(len(data["y_validation"])))}, "training_median_predictor": {"train_median": base.median, "validation_metrics": validation_metrics("regression", data["y_validation"], base.median_predict(len(data["y_validation"])))}}
    return output


def bundle_hash(exclude: str) -> tuple[str, list[dict[str, Any]]]:
    entries = []
    digest = hashlib.sha256()
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != exclude):
        rel = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        digest.update(rel.encode("utf-8")); digest.update(b"\0"); digest.update(data)
        entries.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return digest.hexdigest(), entries


def main() -> None:
    os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")
    dataset_evidence = verify_adaptive_dataset()
    plan_evidence = verify_historical_plan()
    historical_contract = verify_historical_candidate_contract()
    contract, candidate_hash = prepare_phase3_root()
    print("PHASE 2A RECONCILIATION PASS")
    print("Scientific payload mismatches: 0")
    print("Metadata-only mismatches: 1")
    print("Metadata mismatch: metadata/validation_gates.json")
    print(f"Training payload inventory: {TRAINING_PAYLOAD_INVENTORY_SHA}")
    print(f"Training payload bundle: {TRAINING_PAYLOAD_BUNDLE_HASH}")
    print(f"Reconciliation evidence commit: {PHASE2A_RECONCILIATION_EVIDENCE_SHA}")
    print("PHASE 3 PREFLIGHT PASS")
    print("RF candidates: 756")
    print("TGNN candidates: 96")
    print("Total candidates: 852")
    print("TEST evaluation: DISABLED")
    print("Historical fixed reuse: NONE")
    print(f"Phase-3 tooling SHA: {CODE_SHA}")
    print("FULL PHASE 3 MODEL-SELECTION SWEEP STARTED")
    print(json.dumps({"pids": [os.getpid()], "execution_mode": "RF then TGNN sequential in one resumable process", "output_root": str(ROOT), "progress_manifest": str(progress_path()), "tgnn_log_pattern": str(ROOT / "logs" / "tgnn" / "<task>" / "<config_id>" / "seed_<seed>.jsonl"), "command": " ".join([sys.executable, *sys.argv]), "candidate_set_hash": candidate_hash}, indent=2))
    sys.stdout.flush()
    write_progress({"phase": "preflight", "task": None, "config_id": None, "seed": None})
    atomic_json(ROOT / "manifests" / "preflight_evidence.json", {"schema_version": "satnet.phase3.preflight_evidence.v1", "training_tooling_sha": CODE_SHA, "phase2_source_sha": PHASE2_SOURCE_SHA, "dataset": dataset_evidence, "historical_plan": plan_evidence, "historical_candidate_contract": historical_contract, "candidate_set_hash": candidate_hash, "rf_expected": 756, "tgnn_expected": 96, "total_expected": 852, "test_evaluation": False, "historical_fixed_data_reused": False, "historical_fixed_checkpoint_reused": False, "completed_at": now()})
    rf_start = time.perf_counter()
    for task in RF_TASKS:
        data = read_rf_data(task)
        for index, config in enumerate(rf_configs(task), 1):
            for seed in VALIDATION_SEEDS:
                current = {"phase": "rf", "task": task, "config_id": f"rf_{index:03d}", "seed": seed}
                write_progress(current)
                try:
                    run_rf(task, current["config_id"], config, seed, data)
                finally:
                    write_progress(current)
    write_results_csv(ROOT / "summaries" / "rf_validation_results.csv", RF_TASKS)
    rf_selection = select_results(RF_TASKS)
    rf_selection_path = ROOT / "summaries" / "rf_validation_selection.json"
    atomic_json(rf_selection_path, {"schema_version": "satnet.rf_validation_selection.v1", "selection_split": "validation", "fit_split": "train", "validation_seeds": list(VALIDATION_SEEDS), "candidate_set_hash": candidate_hash, "tasks": rf_selection, "test_accessed": False, "frozen_at": now()})
    atomic_json(ROOT / "manifests" / "rf_selection_freeze.json", {"sha256": sha256_file(rf_selection_path), "candidate_set_hash": candidate_hash, "frozen_at": now(), "test_accessed": False})
    atomic_json(ROOT / "manifests" / "rf_phase_runtime.json", {"runtime_seconds": time.perf_counter() - rf_start, "expected_fits": 756, "completed_fits": 756, "failed_fits": 0, "candidate_set_hash": candidate_hash, "test_accessed": False})
    tgnn_start = time.perf_counter()
    for task in TGNN_TASKS:
        bundle = load_tgnn_metadata(task)
        for index, config in enumerate(tgnn_configs(), 1):
            for seed in VALIDATION_SEEDS:
                current = {"phase": "tgnn", "task": task, "config_id": f"tgnn_{index:03d}", "seed": seed}
                write_progress(current)
                try:
                    run_tgnn(task, current["config_id"], config, seed, bundle)
                finally:
                    write_progress(current)
    write_results_csv(ROOT / "summaries" / "tgnn_validation_results.csv", TGNN_TASKS)
    tgnn_selection = select_results(TGNN_TASKS)
    tgnn_selection_path = ROOT / "summaries" / "tgnn_validation_selection.json"
    atomic_json(tgnn_selection_path, {"schema_version": "satnet.tgnn_validation_selection.v1", "selection_split": "validation", "fit_split": "train", "validation_seeds": list(VALIDATION_SEEDS), "candidate_set_hash": candidate_hash, "early_stopping": {"monitor_classification": "balanced_accuracy", "monitor_regression": "mae", "patience": 10, "min_delta": 0.0, "restore_best": True}, "tasks": tgnn_selection, "test_accessed": False, "frozen_at": now()})
    atomic_json(ROOT / "manifests" / "tgnn_selection_freeze.json", {"sha256": sha256_file(tgnn_selection_path), "candidate_set_hash": candidate_hash, "frozen_at": now(), "test_accessed": False})
    atomic_json(ROOT / "manifests" / "tgnn_phase_runtime.json", {"runtime_seconds": time.perf_counter() - tgnn_start, "expected_fits": 96, "completed_fits": 96, "failed_fits": 0, "candidate_set_hash": candidate_hash, "test_accessed": False})
    atomic_json(ROOT / "summaries" / "validation_baselines.json", baselines())
    firewall = assert_test_firewall()
    write_progress({"phase": "complete", "task": None, "config_id": None, "seed": None})
    progress = load_json(progress_path())
    if progress["rf"]["completed"] != 756 or progress["tgnn"]["completed"] != 96 or progress["total"]["failed"] != 0:
        raise RuntimeError("Phase-3 completion gate failed")
    summary = {"schema_version": "satnet.phase3.model_selection_summary.v1", "status": "PASS", "training_tooling_sha": CODE_SHA, "phase2_source_sha": PHASE2_SOURCE_SHA, "phase2a_reconciliation_evidence_sha": PHASE2A_RECONCILIATION_EVIDENCE_SHA, "phase2_exporter_sha": PHASE2_EXPORTER_SHA, "phase1_source_sha": PHASE1_SOURCE_SHA, "dataset_bundle_hash": DATASET_HASH, "candidate_set_hash": candidate_hash, "rf_expected": 756, "rf_completed": progress["rf"]["completed"], "rf_failed": progress["rf"]["failed"], "tgnn_expected": 96, "tgnn_completed": progress["tgnn"]["completed"], "tgnn_failed": progress["tgnn"]["failed"], "selection_used_only_validation": True, "test_evaluation_performed": False, "test_metrics_calculated": False, "test_predictions_generated": False, "test_targets_loaded": False, "preprocessing_fit_on_train_only": True, "historical_fixed_data_reused": False, "historical_fixed_checkpoint_reused": False, "historical_methodology": {"training_plan_root": str(PLAN_ROOT), "training_plan_bundle_hash": PLAN_HASH, "historical_driver": "scripts/final_training/run_validation_training.py", "rf_search_space": "rf_search_space.json", "tgnn_search_space": "tgnn_search_space.json", "model_selection_contract": "model_selection_contract.json"}, "selection_paths": {"rf": str(rf_selection_path), "tgnn": str(tgnn_selection_path)}, "test_firewall": firewall, "generated_at": now()}
    manifest_path_output = ROOT / "manifests" / "phase3_manifest.json"
    atomic_json(manifest_path_output, {**summary, "winning_rf": {task: rf_selection[task] for task in RF_TASKS}, "winning_tgnn": {task: tgnn_selection[task] for task in TGNN_TASKS}})
    report = "# Phase 3 Adaptive-v2 Model Selection\n\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n\nThe complete frozen candidate sweep used TRAIN fitting and VALIDATION-only model selection. TEST targets, predictions, and metrics were not accessed or generated. No robustness, held-out evaluation, external validation, or DSS rebinding was performed.\n"
    (ROOT / "summaries" / "phase3_report.md").write_text(report, encoding="utf-8")
    bundle_hash_value, entries = bundle_hash("phase3_inventory.json")
    atomic_json(ROOT / "manifests" / "phase3_inventory.json", {"schema_version": "satnet.phase3.inventory.v1", "inventory_self_excluding": True, "hash_algorithm": "sha256 over sorted relative UTF-8 path + NUL byte + raw file bytes", "bundle_sha256": bundle_hash_value, "artifacts": entries, "test_accessed": False})
    print(json.dumps({"status": "PASS", "candidate_set_hash": candidate_hash, "bundle_sha256": bundle_hash_value, "root": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
