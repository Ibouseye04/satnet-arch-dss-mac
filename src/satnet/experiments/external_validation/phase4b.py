"""Guarded Adaptive-v2 external inference tooling.

The default command path performs preflight only.  Model predictions require an
explicit authorization flag and are never produced by importing this module.
The external edge-failure column is consumed as a frozen raw RF feature: no
replacement, clipping, rescaling, normalization fitting, or remapping occurs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

EXTERNAL_ROOT = Path(r"C:\Users\johns\external\satnet-external-validation-v2-adaptive")
INFERENCE_ROOT = Path(r"C:\Users\johns\external\satnet-external-inference-v2-adaptive")
MODEL_ROOT = Path(r"C:\Users\johns\external\satnet-10k-final-models-v2-adaptive")
FINAL_ROBUSTNESS_ROOT = MODEL_ROOT / "final_robustness"
DATASET_ROOT = Path(r"C:\Users\johns\external\satnet-10k-final-ml-datasets-v2-adaptive")
HELDOUT_ROOT = Path(r"C:\Users\johns\external\satnet-10k-heldout-v2-adaptive")

EXTERNAL_BUNDLE_SHA256 = "63b098ddc28a1d7ee45f38fc57b22793d876855b832a65c62760371935c577cc"
EXTERNAL_INVENTORY_SHA256 = "7440239bf9df99f59bad7483f072106bdce74954a32c7d02dc1b378d595f24c1"
DATASET_BUNDLE_SHA256 = "1399f12e6ed7e159076028bc01971cf2274924fc55b43cca0a44219db8c978d1"
TRAINING_PLAN_BUNDLE_SHA256 = "e14ec5e5b2221aa2c9aa187a2baacc517e676f9dc748612c206eb1dc43788a0a"
SYNTHETIC_EDGE_TRAIN_MAX = 0.25
PHASE4_INVENTORY_SHA256 = "63ec97e52849b92773887f140a5e95ac3c337837ea7451f9b3df142b4bfd9779"
PHASE4_BUNDLE_SHA256 = "fa0be3579da38169d4d36663eafca62b4412b4e1b403e7241ae64737e4c6fc83"
CANDIDATE_SET_HASH = "8d13cbcae2c58c72c9c1ae9da5574423b866097e325bc987ae2bb8d7aae7aaf2"
RF_SELECTION_HASH = "bbdb71469bb88d85ce45720dbaee4e8056589d75e741113e601d52a9f704a2cb"
TGNN_SELECTION_HASH = "81e293f8660c00bf8de0dc55280135c2b116c41c637001a742ec626afdb6108b"
RF_MAX_LEARNED_EDGE_THRESHOLD = 0.24879451841115952
RF_EXTERNAL_EDGE_MIN = 0.6619127516778524
RF_EXTERNAL_EDGE_MAX = 0.843945720250522

SEEDS = (42, 123, 456, 789, 2026)
PRIMARY_SEED = 42
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_REPLICATES = 2000
EPISODE_COUNT = 300
TIMESTEPS = 11

RF_FEATURES = (
    "num_planes",
    "sats_per_plane",
    "altitude_km",
    "inclination_deg",
    "satellite_node_failure_probability",
    "satellite_edge_failure_probability",
)
NODE_FEATURES = ("plane_idx_normalized", "sat_in_plane_normalized", "node_exists_constant")
EDGE_FEATURES = (
    "distance_km_scaled_10000",
    "margin_db_scaled_100",
    "link_type_code_scaled_2",
    "link_mode_binary",
)
TASKS = (
    "rf_space_classification",
    "tgnn_space_classification",
    "rf_space_regression",
    "tgnn_space_regression",
)
RF_TASKS = TASKS[:1] + TASKS[2:3]
TGNN_TASKS = TASKS[1:2] + TASKS[3:4]
TASK_TYPE = {task: ("classification" if "classification" in task else "regression") for task in TASKS}
MODEL_FAMILY = {task: ("RF" if task.startswith("rf_") else "TGNN") for task in TASKS}
TARGET = {
    "rf_space_classification": "space_threshold_breach_any",
    "tgnn_space_classification": "space_threshold_breach_any",
    "rf_space_regression": "space_gcc_fraction_original_min",
    "tgnn_space_regression": "space_gcc_fraction_original_min",
}
EXPECTED_CONFIG_IDS = {
    "rf_space_classification": "rf_020",
    "tgnn_space_classification": "tgnn_010",
    "rf_space_regression": "rf_018",
    "tgnn_space_regression": "tgnn_012",
}
EXPECTED_RF_CONFIGS = {
    "rf_space_classification": {
        "bootstrap": True,
        "class_weight": "balanced",
        "max_depth": 10,
        "max_features": 1.0,
        "min_samples_leaf": 2,
        "n_estimators": 300,
        "n_jobs": -1,
        "random_state": 42,
    },
    "rf_space_regression": {
        "bootstrap": True,
        "max_depth": 20,
        "max_features": 1.0,
        "min_samples_leaf": 5,
        "n_estimators": 300,
        "n_jobs": -1,
        "random_state": 42,
    },
}
EXPECTED_TGNN_CONFIG = {
    "batch_size": 1,
    "cheb_k": None,
    "classification_loss": "CrossEntropyLoss",
    "dropout": None,
    "hidden_dim": 64,
    "learning_rate": 0.001,
    "max_epochs": 100,
    "num_layers": 1,
    "optimizer": "Adam",
    "regression_loss": "SmoothL1Loss",
    "weight_decay": 0.0,
}

TGNN_LABEL = "Real-input external generalization"
RF_LABEL = "Out-of-domain physical-viability proxy stress test"
COMPARISON_LABEL = "Descriptive external comparison under mismatched RF feature semantics"
PRIMARY_LABEL = "Authoritative held-out model comparison"
NO_SUPERIORITY_TERMS = ("superior", "superiority", "better", "winner", "advantage")
AUTHORITATIVE_HELDOUT_COMPARISON = {
    "classification": {
        "rf_balanced_accuracy": 0.9224385838025269,
        "tgnn_balanced_accuracy": 0.8421571714808752,
        "rf_minus_tgnn": 0.08028141232165165,
        "paired_design_cluster_ci_95": [0.04200399022838036, 0.11981099042922246],
        "conclusion": "RF classification advantage statistically supported",
    },
    "regression": {
        "rf_seed_42_mae": 0.05149723018092795,
        "tgnn_seed_42_mae": 0.05078137591317217,
        "rf_minus_tgnn_mae": 0.0007158542677557778,
        "paired_design_cluster_ci_95": [-0.0055607897035179245, 0.007111047568497697],
        "conclusion": "no statistically distinguishable primary-MAE winner",
    },
}


class ExternalInferenceError(RuntimeError):
    """Raised when a frozen external inference contract is violated."""


@dataclass(frozen=True)
class ExternalEpisode:
    episode_id: int
    timestamp: str
    rf_features: tuple[float, ...]
    rf_feature_text: tuple[str, ...]
    edge_failure_probability: float
    edge_failure_probability_text: str
    regression_target: float
    classification_target: int
    tgnn_sequence: Path


@dataclass(frozen=True)
class FrozenArtifact:
    task: str
    seed: int
    config_id: str
    family: str
    path: Path
    sha256: str
    manifest_path: Path
    manifest_sha256: str
    selection_freeze_sha256: str
    candidate_set_hash: str


@dataclass(frozen=True)
class OutputContract:
    tgnn_metrics: Path = INFERENCE_ROOT / "tgnn_external_metrics.json"
    tgnn_predictions: Path = INFERENCE_ROOT / "tgnn_external_predictions.jsonl"
    rf_metrics: Path = INFERENCE_ROOT / "rf_ood_stress_metrics.json"
    rf_predictions: Path = INFERENCE_ROOT / "rf_ood_stress_predictions.jsonl"
    classification: Path = INFERENCE_ROOT / "classification_descriptive_results.json"
    comparison: Path = INFERENCE_ROOT / "external_descriptive_comparison.json"
    summary: Path = INFERENCE_ROOT / "external_inference_summary.json"
    report: Path = INFERENCE_ROOT / "external_inference_report.md"
    inventory: Path = INFERENCE_ROOT / "external_inference_inventory.json"
    identity: Path = INFERENCE_ROOT / "external_execution_identity.json"


OUTPUTS = OutputContract()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ExternalInferenceError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def bundle_hash(root: Path, excluded_names: set[str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in excluded_names):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def inventory_hash(root: Path, inventory_name: str) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != inventory_name):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalInferenceError(f"cannot load JSON {path}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalInferenceError(message)


def finite(value: Any, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ExternalInferenceError(f"non-finite value at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite(item, f"{location}[{index}]")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_external_bundle() -> dict[str, Any]:
    inventory_path = EXTERNAL_ROOT / "external_validation_inventory.json"
    observed_bundle = bundle_hash(EXTERNAL_ROOT, {inventory_path.name})
    observed_inventory = sha256_file(inventory_path)
    require(observed_bundle == EXTERNAL_BUNDLE_SHA256, f"external bundle hash mismatch: {observed_bundle}")
    require(observed_inventory == EXTERNAL_INVENTORY_SHA256, f"external inventory hash mismatch: {observed_inventory}")
    inventory = load_json(inventory_path)
    require(inventory.get("external_pre_inference_bundle_sha256") == EXTERNAL_BUNDLE_SHA256, "external inventory bundle identity mismatch")
    require(inventory.get("model_inference_performed") is False, "external package already records model inference")
    require(inventory.get("external_predictions_generated") is False, "external package already records predictions")
    require(inventory.get("external_metrics_calculated") is False, "external package already records metrics")
    require(inventory.get("episode_count") == EPISODE_COUNT, "external episode count mismatch")
    return {
        "bundle_sha256": observed_bundle,
        "inventory_sha256": observed_inventory,
        "inventory_path": str(inventory_path),
        "model_inference_performed": False,
    }


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames is not None, f"missing CSV header: {path}")
            return list(reader.fieldnames), list(reader)
    except OSError as exc:
        raise ExternalInferenceError(f"cannot read CSV {path}") from exc


def _verify_sequence(path: Path, episode_id: int) -> None:
    payload = load_json(path)
    require(payload.get("episode_id") == episode_id, f"TGNN episode identity mismatch: {path}")
    require(tuple(payload.get("node_feature_order", ())) == NODE_FEATURES, f"TGNN node feature order mismatch: {path}")
    require(tuple(payload.get("edge_feature_order", ())) == EDGE_FEATURES, f"TGNN edge feature order mismatch: {path}")
    snapshots = payload.get("snapshots")
    require(isinstance(snapshots, list) and len(snapshots) == TIMESTEPS, f"TGNN snapshot count mismatch: {path}")
    for snapshot_index, snapshot in enumerate(snapshots):
        nodes = snapshot.get("nodes", [])
        edges = snapshot.get("edges", [])
        require(isinstance(nodes, list) and isinstance(edges, list), f"TGNN snapshot shape mismatch: {path}")
        for node in nodes:
            features = node.get("features")
            require(isinstance(features, list) and len(features) == len(NODE_FEATURES), f"TGNN node dimension mismatch: {path}")
            finite(features, f"{path}:snapshot={snapshot_index}:node")
        for edge in edges:
            values = [edge.get(name) for name in EDGE_FEATURES]
            require(all(isinstance(value, (int, float)) for value in values), f"TGNN edge feature missing: {path}")
            finite(values, f"{path}:snapshot={snapshot_index}:edge")


def load_external_episodes() -> list[ExternalEpisode]:
    path = EXTERNAL_ROOT / "episodes/external_rf_dataset.csv"
    fields, rows = _read_csv(path)
    required = ["episode_id", "episode_timestamp", *RF_FEATURES, "space_gcc_fraction_original_min", "space_threshold_breach_any", "tgnn_sequence"]
    require(fields[: len(required)] == required, "external RF feature/target order mismatch")
    require(len(rows) == EPISODE_COUNT, "external episode count is not 300")
    episodes: list[ExternalEpisode] = []
    for expected_id, row in enumerate(rows):
        episode_id = int(row["episode_id"])
        require(episode_id == expected_id, "external episode IDs are not ordered 0..299")
        raw_features = tuple(row[name] for name in RF_FEATURES)
        features = tuple(float(value) for value in raw_features)
        edge_text = row["satellite_edge_failure_probability"]
        edge_value = float(edge_text)
        target = float(row["space_gcc_fraction_original_min"])
        classification = int(row["space_threshold_breach_any"])
        require(classification == 1, "external classification is not the frozen 300-positive set")
        require(0.0 <= target <= 1.0, f"external target outside [0,1]: {episode_id}")
        sequence = EXTERNAL_ROOT / row["tgnn_sequence"]
        require(sequence.is_file(), f"missing TGNN sequence: {sequence}")
        _verify_sequence(sequence, episode_id)
        episodes.append(ExternalEpisode(episode_id, row["episode_timestamp"], features, raw_features, edge_value, edge_text, target, classification, sequence))
    edge_values = np.asarray([episode.edge_failure_probability for episode in episodes], dtype=float)
    require(float(np.min(edge_values)) == RF_EXTERNAL_EDGE_MIN, "external RF edge proxy minimum changed")
    require(float(np.max(edge_values)) == RF_EXTERNAL_EDGE_MAX, "external RF edge proxy maximum changed")
    return episodes


def verify_tgnn_target_manifest(episodes: Sequence[ExternalEpisode]) -> None:
    path = EXTERNAL_ROOT / "episodes/tgnn_space_target_manifest.jsonl"
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalInferenceError(f"cannot read TGNN target manifest: {path}") from exc
    require(len(rows) == EPISODE_COUNT, "TGNN target manifest count mismatch")
    for episode, row in zip(episodes, rows):
        require(int(row["episode_id"]) == episode.episode_id, "TGNN target manifest episode mismatch")
        require(float(row["regression_target"]) == episode.regression_target, "TGNN target changed")
        require(int(row["target"]) == episode.classification_target, "TGNN classification target changed")


def _phase4_inventory_entries() -> dict[str, str]:
    inventory_path = FINAL_ROBUSTNESS_ROOT / "final_robustness_inventory.json"
    require(sha256_file(inventory_path) == PHASE4_INVENTORY_SHA256, "Phase-4 inventory SHA mismatch")
    inventory = load_json(inventory_path)
    require(inventory.get("bundle_sha256") == PHASE4_BUNDLE_SHA256, "Phase-4 bundle identity mismatch")
    require(inventory.get("inventory_self_excluding") is True and inventory.get("test_accessed") is False, "Phase-4 inventory seal failed")
    return {str(item["path"]): str(item["sha256"]) for item in inventory.get("artifacts", [])}


def _selection_hash(task: str) -> str:
    return RF_SELECTION_HASH if task.startswith("rf_") else TGNN_SELECTION_HASH


def _artifact_path(task: str, seed: int) -> Path:
    filename = "final_model.joblib" if task.startswith("rf_") else "best_validation_checkpoint.pt"
    return FINAL_ROBUSTNESS_ROOT / task / f"seed_{seed}" / filename


def verify_model_artifact(task: str, seed: int, inventory_entries: Mapping[str, str]) -> FrozenArtifact:
    require(task in TASKS and seed in SEEDS, f"unsupported frozen artifact identity: {task}/{seed}")
    manifest_path = FINAL_ROBUSTNESS_ROOT / task / f"seed_{seed}" / "final_manifest.json"
    manifest_relative = manifest_path.relative_to(FINAL_ROBUSTNESS_ROOT).as_posix()
    require(inventory_entries.get(manifest_relative) == sha256_file(manifest_path), f"Phase-4 manifest absent or changed: {manifest_relative}")
    manifest = load_json(manifest_path)
    config_id = EXPECTED_CONFIG_IDS[task]
    selection_hash = _selection_hash(task)
    require(manifest.get("status") == "completed", f"model is not completed: {task}/{seed}")
    require(manifest.get("task") == task and manifest.get("seed") == seed, f"model identity mismatch: {task}/{seed}")
    require(manifest.get("selected_config_id") == config_id, f"selected config mismatch: {task}/{seed}")
    require(manifest.get("selection_freeze_sha256") == selection_hash, f"selection freeze mismatch: {task}/{seed}")
    require(manifest.get("phase3_candidate_set_hash") == CANDIDATE_SET_HASH, f"candidate set mismatch: {task}/{seed}")
    require(manifest.get("dataset_bundle_hash") == DATASET_BUNDLE_SHA256, f"dataset bundle mismatch: {task}/{seed}")
    require(manifest.get("training_plan_bundle_hash") == TRAINING_PLAN_BUNDLE_SHA256, f"training plan mismatch: {task}/{seed}")
    require(manifest.get("test_accessed") is False, f"TEST access recorded for {task}/{seed}")
    if task.startswith("rf_"):
        expected_config = {**EXPECTED_RF_CONFIGS[task], "random_state": seed}
    else:
        expected_config = {**EXPECTED_TGNN_CONFIG, "cheb_k": 2 if task.endswith("classification") else 3}
    require(manifest.get("configuration") == expected_config, f"frozen configuration mismatch: {task}/{seed}")
    artifact = _artifact_path(task, seed)
    path_key = "model_path" if task.startswith("rf_") else "checkpoint_path"
    require(manifest.get(path_key) == str(artifact), f"frozen artifact path mismatch: {artifact}")
    require(artifact.is_file(), f"missing frozen model artifact: {artifact}")
    artifact_relative = artifact.relative_to(FINAL_ROBUSTNESS_ROOT).as_posix()
    expected_sha = manifest.get("model_sha256" if task.startswith("rf_") else "checkpoint_sha256")
    require(isinstance(expected_sha, str) and sha256_file(artifact) == expected_sha, f"model SHA mismatch: {artifact}")
    require(inventory_entries.get(artifact_relative) == expected_sha, f"model absent from Phase-4 inventory: {artifact_relative}")
    return FrozenArtifact(task, seed, config_id, MODEL_FAMILY[task], artifact, expected_sha, manifest_path, sha256_file(manifest_path), selection_hash, CANDIDATE_SET_HASH)


def verify_model_freezes() -> list[FrozenArtifact]:
    inventory_entries = _phase4_inventory_entries()
    selection_paths = {
        "rf": MODEL_ROOT / "summaries/rf_validation_selection.json",
        "tgnn": MODEL_ROOT / "summaries/tgnn_validation_selection.json",
    }
    freeze_paths = {
        "rf": MODEL_ROOT / "manifests/rf_selection_freeze.json",
        "tgnn": MODEL_ROOT / "manifests/tgnn_selection_freeze.json",
    }
    for family, path in selection_paths.items():
        expected = RF_SELECTION_HASH if family == "rf" else TGNN_SELECTION_HASH
        require(sha256_file(path) == expected, f"{family} selection summary SHA mismatch")
        freeze = load_json(freeze_paths[family])
        require(freeze.get("sha256") == expected and freeze.get("candidate_set_hash") == CANDIDATE_SET_HASH and freeze.get("test_accessed") is False, f"{family} selection freeze contract failed")
    phase3 = load_json(MODEL_ROOT / "manifests/phase3_manifest.json")
    require(phase3.get("status") == "PASS" and phase3.get("candidate_set_hash") == CANDIDATE_SET_HASH, "Phase-3 candidate freeze failed")
    require(phase3.get("test_evaluation_performed") is False and phase3.get("test_predictions_generated") is False and phase3.get("test_targets_loaded") is False, "Phase-3 TEST firewall failed")
    return [verify_model_artifact(task, seed, inventory_entries) for task in TASKS for seed in SEEDS]


def verify_dataset_bundle() -> dict[str, Any]:
    inventory_path = DATASET_ROOT / "final_ml_dataset_inventory.json"
    inventory = load_json(inventory_path)
    require(inventory.get("bundle_sha256") == DATASET_BUNDLE_SHA256, "synthetic dataset bundle SHA mismatch")
    phase3 = load_json(MODEL_ROOT / "manifests/phase3_manifest.json")
    require(phase3.get("dataset_bundle_hash") == DATASET_BUNDLE_SHA256, "Phase-3 dataset bundle reference mismatch")
    return {"bundle_sha256": DATASET_BUNDLE_SHA256, "inventory_path": str(inventory_path)}


def _rf_edge_thresholds(model: Any) -> tuple[float, ...]:
    values: list[float] = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values.extend(float(value) for value in tree.threshold[tree.feature == len(RF_FEATURES) - 1])
    return tuple(values)


def verify_rf_ood_domain(episodes: Sequence[ExternalEpisode], artifacts: Sequence[FrozenArtifact], model_loader: Callable[[Path], Any] | None = None) -> dict[str, Any]:
    edge_values = np.asarray([episode.edge_failure_probability for episode in episodes], dtype=float)
    require(len(edge_values) == EPISODE_COUNT, "RF OOD episode count mismatch")
    require(np.all(edge_values > SYNTHETIC_EDGE_TRAIN_MAX), "RF external edge proxy entered synthetic training range")
    rf_artifacts = [artifact for artifact in artifacts if artifact.family == "RF"]
    require(len(rf_artifacts) == len(RF_TASKS) * len(SEEDS), "RF artifact count mismatch")
    loader = model_loader
    if loader is None:
        import joblib
        loader = joblib.load
    thresholds: dict[str, list[float]] = {}
    all_thresholds: list[float] = []
    for artifact in rf_artifacts:
        values = _rf_edge_thresholds(loader(artifact.path))
        thresholds[f"{artifact.task}/seed_{artifact.seed}"] = list(values)
        all_thresholds.extend(values)
    require(all_thresholds, "RF models contain no learned edge-feature thresholds")
    maximum = max(all_thresholds)
    require(maximum == RF_MAX_LEARNED_EDGE_THRESHOLD, f"maximum RF edge split threshold changed: {maximum}")
    require(all(float(value) > maximum for value in edge_values), "external RF edge proxy does not exceed every learned threshold")
    return {
        "feature": "satellite_edge_failure_probability",
        "external_min": float(np.min(edge_values)),
        "external_max": float(np.max(edge_values)),
        "synthetic_train_max": SYNTHETIC_EDGE_TRAIN_MAX,
        "external_episode_count": int(edge_values.size),
        "ood_episode_count": int(np.sum(edge_values > 0.25)),
        "ood_all_episodes": True,
        "all_external_values_exceed_every_learned_threshold": True,
        "maximum_learned_split_threshold": maximum,
        "thresholds_by_artifact": thresholds,
        "semantic_equivalence_to_synthetic_predictor": False,
        "interpretation": RF_LABEL,
    }


def preflight() -> dict[str, Any]:
    external = verify_external_bundle()
    episodes = load_external_episodes()
    verify_tgnn_target_manifest(episodes)
    dataset = verify_dataset_bundle()
    artifacts = verify_model_freezes()
    ood = verify_rf_ood_domain(episodes, artifacts)
    return {
        "status": "PASS",
        "inference_performed": False,
        "training_performed": False,
        "retraining_performed": False,
        "fine_tuning_performed": False,
        "threshold_tuning_performed": False,
        "feature_remapping_performed": False,
        "external": external,
        "dataset": dataset,
        "ood": ood,
        "episode_count": len(episodes),
        "model_artifacts": [artifact.__dict__ | {"path": str(artifact.path), "manifest_path": str(artifact.manifest_path)} for artifact in artifacts],
        "selected_configs": EXPECTED_CONFIG_IDS,
        "seeds": list(SEEDS),
        "primary_reporting_seed": PRIMARY_SEED,
        "candidate_set_hash": CANDIDATE_SET_HASH,
        "selection_freeze_hashes": {"rf": RF_SELECTION_HASH, "tgnn": TGNN_SELECTION_HASH},
        "phase4_inventory_sha256": PHASE4_INVENTORY_SHA256,
        "authoritative_heldout_model_comparison": AUTHORITATIVE_HELDOUT_COMPARISON,
        "output_contract": {name: str(path) for name, path in OUTPUTS.__dict__.items()},
        "interpretation_contract": {
            "tgnn": TGNN_LABEL,
            "rf": RF_LABEL,
            "comparison": COMPARISON_LABEL,
            "primary": PRIMARY_LABEL,
            "external_target": "SATNET frozen GCC resilience target reconstructed from real Starlink-derived orbital/status inputs through the Adaptive-v2 physics/topology pipeline",
            "rf_edge_feature_semantics": "deterministic 30-day candidate-rejection fraction, not a stochastic observed link-failure variable",
        },
    }


def write_execution_identity(preflight_result: Mapping[str, Any]) -> None:
    identity = {
        "schema_version": "satnet.adaptive_v2.external_execution_identity.v1",
        "status": "PRE_INFERENCE_ONLY",
        "external_root": str(EXTERNAL_ROOT),
        "inference_root": str(INFERENCE_ROOT),
        "external_pre_inference_bundle_sha256": EXTERNAL_BUNDLE_SHA256,
        "external_inventory_sha256": EXTERNAL_INVENTORY_SHA256,
        "dataset_bundle_sha256": DATASET_BUNDLE_SHA256,
        "phase4_inventory_sha256": PHASE4_INVENTORY_SHA256,
        "phase4_inventory_bundle_sha256": PHASE4_BUNDLE_SHA256,
        "candidate_set_hash": CANDIDATE_SET_HASH,
        "authoritative_heldout_model_comparison": AUTHORITATIVE_HELDOUT_COMPARISON,
        "selection_freeze_hashes": {"rf": RF_SELECTION_HASH, "tgnn": TGNN_SELECTION_HASH},
        "selected_configs": EXPECTED_CONFIG_IDS,
        "seeds": list(SEEDS),
        "primary_reporting_seed": PRIMARY_SEED,
        "model_artifacts": preflight_result["model_artifacts"],
        "classification": {"negative": 0, "positive": 300, "descriptive_only": True},
        "inference_performed": False,
        "predictions_generated": False,
        "metrics_calculated": False,
        "no_training_or_tuning": True,
        "output_contract": preflight_result["output_contract"],
        "interpretation_contract": preflight_result["interpretation_contract"],
    }
    atomic_json(OUTPUTS.identity, identity)


def _load_rf(path: Path) -> Any:
    import joblib
    return joblib.load(path)


def _load_tgnn(path: Path, task: str, config: Mapping[str, Any]) -> Any:
    import torch
    from satnet.models.gnn_model import SatelliteGNN
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = SatelliteGNN(node_features=3, hidden_channels=int(config["hidden_dim"]), out_channels=2 if TASK_TYPE[task] == "classification" else 1, task_type=TASK_TYPE[task], cheb_k=int(config["cheb_k"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_tgnn_sequence(path: Path) -> list[Any]:
    import torch
    from torch_geometric.data import Data
    payload = load_json(path)
    result = []
    for snapshot in payload["snapshots"]:
        nodes = sorted(snapshot["nodes"], key=lambda item: int(item["node_id"]))
        index = {int(node["node_id"]): position for position, node in enumerate(nodes)}
        edges = snapshot["edges"]
        edge_index = torch.tensor([[index[int(edge["source"])] for edge in edges], [index[int(edge["target"])] for edge in edges]], dtype=torch.long).reshape(2, -1)
        edge_attr = torch.tensor([[float(edge[name]) for name in EDGE_FEATURES] for edge in edges], dtype=torch.float32).reshape(-1, len(EDGE_FEATURES))
        data = Data(x=torch.tensor([node["features"] for node in nodes], dtype=torch.float32), edge_index=edge_index, edge_attr=edge_attr)
        data.edge_weight = edge_attr[:, 0]
        result.append(data)
    require(len(result) == TIMESTEPS, f"TGNN sequence length changed: {path}")
    return result


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    residual = prediction - target
    absolute = np.abs(residual)
    centered = target - np.mean(target)
    denominator = float(np.sum(centered * centered))
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "r2": float(1.0 - np.sum(residual * residual) / denominator) if denominator else None,
        "median_absolute_error": float(np.median(absolute)),
        "maximum_absolute_error": float(np.max(absolute)),
        "residual_mean": float(np.mean(residual)),
        "residual_standard_deviation": float(np.std(residual)),
        "prediction_range": [float(np.min(prediction)), float(np.max(prediction))],
    }


def _classification_metrics(target: np.ndarray, prediction: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    positive = target == 1
    predicted_positive = prediction == 1
    return {
        "descriptive_only": True,
        "observed_class_balance": {"negative": int(np.sum(target == 0)), "positive": int(np.sum(target == 1))},
        "predicted_positive_count": int(np.sum(predicted_positive)),
        "predicted_negative_count": int(np.sum(~predicted_positive)),
        "positive_class_recall": float(np.sum(positive & predicted_positive) / np.sum(positive)) if np.any(positive) else None,
        "positive_class_score_distribution": {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "standard_deviation": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        },
        "excluded_inferential_metrics": ["balanced_accuracy", "specificity", "negative_class_recall", "roc_auc", "pr_auc"],
    }


def _five_seed_descriptive_summary(metrics: Mapping[str, Mapping[str, Any]], tasks: Sequence[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for task in tasks:
        values = [metrics[f"{task}/seed_{seed}"] for seed in SEEDS]
        if TASK_TYPE[task] == "regression":
            fields = ("mae", "rmse", "r2", "median_absolute_error", "maximum_absolute_error", "residual_mean", "residual_standard_deviation")
        else:
            fields = ("positive_class_recall",)
        summary[task] = {
            field: {
                "mean": float(np.mean([float(item[field]) for item in values])),
                "standard_deviation": float(np.std([float(item[field]) for item in values])),
            }
            for field in fields
        }
    return summary


def _prediction_record(episode: ExternalEpisode, task: str, seed: int, prediction: float, score: float | None = None) -> dict[str, Any]:
    record = {
        "episode_id": episode.episode_id,
        "timestamp": episode.timestamp,
        "task": task,
        "seed": seed,
        "model_family": MODEL_FAMILY[task],
        "target": episode.classification_target if TASK_TYPE[task] == "classification" else episode.regression_target,
        "prediction": int(prediction) if TASK_TYPE[task] == "classification" else float(prediction),
    }
    if score is not None:
        record["positive_class_score_probability"] = float(score)
    if MODEL_FAMILY[task] == "RF":
        record["interpretation_label"] = RF_LABEL
        record["satellite_edge_failure_probability"] = float(episode.edge_failure_probability)
        record["satellite_edge_failure_probability_source_text"] = episode.edge_failure_probability_text
        record["edge_feature_semantically_equivalent_to_synthetic_predictor"] = False
    else:
        record["interpretation_label"] = TGNN_LABEL
    if TASK_TYPE[task] == "regression":
        record["error"] = float(prediction - episode.regression_target)
        record["absolute_error"] = float(abs(prediction - episode.regression_target))
    return record


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _predict_task(task: str, seed: int, artifact: FrozenArtifact, episodes: Sequence[ExternalEpisode]) -> list[dict[str, Any]]:
    manifest = load_json(artifact.manifest_path)
    if MODEL_FAMILY[task] == "RF":
        model = _load_rf(artifact.path)
        matrix = np.asarray([episode.rf_features for episode in episodes], dtype=float)
        predictions = np.asarray(model.predict(matrix))
        if TASK_TYPE[task] == "classification":
            scores = np.asarray(model.predict_proba(matrix))[:, 1]
            return [_prediction_record(episode, task, seed, predictions[index], scores[index]) for index, episode in enumerate(episodes)]
        return [_prediction_record(episode, task, seed, predictions[index]) for index, episode in enumerate(episodes)]
    import torch
    model = _load_tgnn(artifact.path, task, manifest["configuration"])
    predictions: list[float] = []
    scores: list[float | None] = []
    with torch.no_grad():
        for episode in episodes:
            sequence = _load_tgnn_sequence(episode.tgnn_sequence)
            output = model(sequence)
            if TASK_TYPE[task] == "classification":
                probabilities = torch.softmax(output, dim=1)
                predictions.append(float(torch.argmax(probabilities, dim=1).item()))
                scores.append(float(probabilities[0, 1].item()))
            else:
                predictions.append(float(output.squeeze().item()))
                scores.append(None)
    return [_prediction_record(episode, task, seed, predictions[index], scores[index]) for index, episode in enumerate(episodes)]


def _metrics_from_records(task: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if TASK_TYPE[task] == "classification":
        target = np.asarray([record["target"] for record in records], dtype=int)
        prediction = np.asarray([record["prediction"] for record in records], dtype=int)
        score = np.asarray([record["positive_class_score_probability"] for record in records], dtype=float)
        return _classification_metrics(target, prediction, score)
    target = np.asarray([record["target"] for record in records], dtype=float)
    prediction = np.asarray([record["prediction"] for record in records], dtype=float)
    return _regression_metrics(target, prediction)


def paired_bootstrap(rf_records: Sequence[Mapping[str, Any]], tgnn_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rf_by_id = {int(record["episode_id"]): float(record["absolute_error"]) for record in rf_records}
    tgnn_by_id = {int(record["episode_id"]): float(record["absolute_error"]) for record in tgnn_records}
    ids = tuple(sorted(set(rf_by_id) & set(tgnn_by_id)))
    require(ids == tuple(range(EPISODE_COUNT)), "paired bootstrap episode identity mismatch")
    rf = np.asarray([rf_by_id[episode_id] for episode_id in ids], dtype=float)
    tgnn = np.asarray([tgnn_by_id[episode_id] for episode_id in ids], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(ids), size=len(ids))
        deltas[index] = float(np.mean(rf[sample]) - np.mean(tgnn[sample]))
    return {
        "label": COMPARISON_LABEL,
        "metric": "MAE",
        "difference": "RF MAE minus TGNN MAE",
        "paired_unit": "episode_id",
        "primary_seed": PRIMARY_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_interval": "95% percentile",
        "observed_rf_mae": float(np.mean(rf)),
        "observed_tgnn_mae": float(np.mean(tgnn)),
        "observed_delta_mae_rf_minus_tgnn": float(np.mean(rf) - np.mean(tgnn)),
        "percentile_2_5": float(np.percentile(deltas, 2.5)),
        "percentile_97_5": float(np.percentile(deltas, 97.5)),
        "interpretation": "Descriptive stress-test statistic for mismatched RF feature semantics; not evidence for choosing one model over the other or for equivalent real-world validation.",
    }


def assert_no_superiority_language(text: str) -> None:
    lowered = text.lower()
    require(not any(term in lowered for term in NO_SUPERIORITY_TERMS), "prohibited RF-vs-TGNN superiority language detected")


def write_report(summary: Mapping[str, Any], comparison: Mapping[str, Any]) -> None:
    report = "\n".join([
        "# Adaptive-v2 External Inference",
        "",
        "## Analysis labels",
        f"- TGNN: **{TGNN_LABEL}**",
        f"- RF: **{RF_LABEL}**",
        f"- RF-vs-TGNN: **{COMPARISON_LABEL}**",
        f"- Primary synthetic comparison: **{PRIMARY_LABEL}**",
        "",
        "## Scientific interpretation",
        "The external regression target is SATNET's frozen GCC resilience target reconstructed from real Starlink-derived orbital/status inputs through the Adaptive-v2 physics/topology pipeline. It is not observed Starlink network resilience ground truth.",
        "",
        "The external satellite_edge_failure_probability column is a deterministic 30-day candidate-rejection fraction. It is not semantically equivalent to the synthetic stochastic Bernoulli edge-failure predictor. The frozen raw sources contain no direct observation of that synthetic edge-failure probability.",
        "",
        "All 300 external RF episodes are outside the synthetic RF training range on this feature, and all external values exceed every learned RF split threshold on the feature. RF outputs therefore measure an OOD proxy stress test, not equivalent real-world validation.",
        "",
        "Classification is single-class (0 negatives, 300 positives) and descriptive only. Inferential discrimination metrics and model-comparison claims are not reported.",
        "",
        "The paired statistic, when present, is historical-compatibility evidence labeled as a descriptive stress test under mismatched RF feature semantics. It is not evidence of validated RF-vs-TGNN real-world performance.",
        "",
        "## Frozen execution summary",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Historical-compatibility statistic",
        "```json",
        json.dumps(comparison, indent=2, sort_keys=True),
        "```",
        "",
        "No training, retraining, fine-tuning, threshold tuning, feature remapping, normalization fitting, model selection, or dataset modification was performed.",
    ]) + "\n"
    assert_no_superiority_language(json.dumps(comparison, sort_keys=True))
    OUTPUTS.report.write_text(report, encoding="utf-8")


def run_inference(*, authorized: bool = False) -> dict[str, Any]:
    require(authorized, "external inference requires explicit authorization")
    preflight_result = preflight()
    write_execution_identity(preflight_result)
    episodes = load_external_episodes()
    artifacts = {(artifact.task, artifact.seed): artifact for artifact in verify_model_freezes()}
    all_records: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        for seed in SEEDS:
            records = _predict_task(task, seed, artifacts[(task, seed)], episodes)
            all_records[f"{task}/seed_{seed}"] = records
            metrics[f"{task}/seed_{seed}"] = {"label": RF_LABEL if task.startswith("rf_") else TGNN_LABEL, **_metrics_from_records(task, records)}
    tgnn_records = [record for key, records in all_records.items() if key.startswith("tgnn_") for record in records]
    rf_records = [record for key, records in all_records.items() if key.startswith("rf_") for record in records]
    _write_jsonl(OUTPUTS.tgnn_predictions, tgnn_records)
    _write_jsonl(OUTPUTS.rf_predictions, rf_records)
    tgnn_metrics = {"per_seed": {key: value for key, value in metrics.items() if key.startswith("tgnn_")}, "five_seed_descriptive_summary": _five_seed_descriptive_summary(metrics, TGNN_TASKS)}
    rf_metrics = {"per_seed": {key: value for key, value in metrics.items() if key.startswith("rf_")}, "five_seed_descriptive_summary": _five_seed_descriptive_summary(metrics, RF_TASKS)}
    classification = {key: value for key, value in metrics.items() if "classification" in key}
    comparison = paired_bootstrap(all_records[f"rf_space_regression/seed_{PRIMARY_SEED}"], all_records[f"tgnn_space_regression/seed_{PRIMARY_SEED}"])
    summary = {"status": "COMPLETE", "inference_performed": True, "preflight": preflight_result, "tgnn": tgnn_metrics, "rf_ood_stress": rf_metrics, "classification_descriptive": classification, "comparison": comparison, "primary_synthetic_comparison": PRIMARY_LABEL, "authoritative_heldout_model_comparison": AUTHORITATIVE_HELDOUT_COMPARISON}
    atomic_json(OUTPUTS.tgnn_metrics, tgnn_metrics)
    atomic_json(OUTPUTS.rf_metrics, rf_metrics)
    atomic_json(OUTPUTS.classification, classification)
    atomic_json(OUTPUTS.comparison, comparison)
    atomic_json(OUTPUTS.summary, summary)
    write_report(summary, comparison)
    entries = []
    for path in sorted(item for item in INFERENCE_ROOT.rglob("*") if item.is_file() and item.name != OUTPUTS.inventory.name):
        entries.append({"path": path.relative_to(INFERENCE_ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    atomic_json(OUTPUTS.inventory, {"schema_version": "satnet.adaptive_v2.external_inference_inventory.v1", "status": "COMPLETE", "artifacts": entries, "inference_performed": True, "external_pre_inference_bundle_sha256": EXTERNAL_BUNDLE_SHA256, "external_inventory_sha256": EXTERNAL_INVENTORY_SHA256})
    return summary


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-inference", action="store_true", help="run only after explicit authorization")
    parser.add_argument("--authorize-external-inference", action="store_true", help="required with --run-inference")
    parser.add_argument("--write-identity", action="store_true", help="write preflight identity into the separate inference root")
    args = parser.parse_args()
    if args.run_inference:
        require(args.authorize_external_inference, "--authorize-external-inference is required")
        run_inference(authorized=True)
        return
    result = preflight()
    if args.write_identity:
        write_execution_identity(result)
    else:
        print(json.dumps({"status": result["status"], "inference_performed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
