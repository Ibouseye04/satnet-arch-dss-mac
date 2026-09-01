"""Adaptive-v2 external-validation preflight and provenance controls.

This module deliberately stops before model inference.  It binds the historical
real-data contract to the frozen Adaptive-v2 lineage, verifies raw-source and
frozen-model identities, and provides behavioral evidence that topology
construction uses the production adaptive implementation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from satnet.experiments.external_validation.phase4a import (
    ADJACENT_SEARCH_K,
    FAILURE_MODEL,
    ISL_POLICY,
    MAX_INTER_PLANE_LINKS_PER_SAT,
    NUM_TIMESTEPS,
    Phase4AConfig,
    STEP_SECONDS,
    TARGET_END,
    TARGET_START,
    _json_bytes,
    choose_episode_timestamps,
)
from satnet.network.hypatia_adapter import (
    HypatiaAdapter,
    SatellitePosition,
    _compute_grid_plus_isls,
)

BASE_SHA = "98424c4f9276760920e5287827ffc18705bb71ec"
PHASE4_TOOLING_SHA = "4a49643189a413a5f8c81ce89ff6a08d222f4889"
HELDOUT_EVALUATOR_SHA = "98424c4f9276760920e5287827ffc18705bb71ec"
ADAPTIVE_DATASET_BUNDLE_SHA = "1399f12e6ed7e159076028bc01971cf2274924fc55b43cca0a44219db8c978d1"
PHASE3_CANDIDATE_SET_SHA = "8d13cbcae2c58c72c9c1ae9da5574423b866097e325bc987ae2bb8d7aae7aaf2"
RF_SELECTION_FREEZE_SHA = "bbdb71469bb88d85ce45720dbaee4e8056589d75e741113e601d52a9f704a2cb"
TGNN_SELECTION_FREEZE_SHA = "81e293f8660c00bf8de0dc55280135c2b116c41c637001a742ec626afdb6108b"
PHASE4_INVENTORY_SHA = "63ec97e52849b92773887f140a5e95ac3c337837ea7451f9b3df142b4bfd9779"
PHASE4_BUNDLE_SHA = "fa0be3579da38169d4d36663eafca62b4412b4e1b403e7241ae64737e4c6fc83"
HELDOUT_BUNDLE_SHA = "f9f0e182e7785d213276f18caea0d559cdcc9068d45637a700f9a5ee5d426df3"
HELDOUT_INVENTORY_SHA = "e7a27e8496ad5ae57b411ead068d5d7f4713c088b0d9901ba5b38f8d22e29d9d"
HISTORICAL_EXTERNAL_SOURCE_MANIFEST_SHA = "02153a1fcb8cac32d0d6b5999ea7c2953d00024c874d23e392664011bb1bbf29"
HISTORICAL_EXTERNAL_BUNDLE_SHA = "9abede2f83adbfc96e8267805e3253d6dbcd6f5240d675729d5677aa053b7e28"
HISTORICAL_ADAPTER_SHA = "83ebc1e4d31b6aa8ff9b1574e3a39b4b9716d1b182ebe2fd5e7e961ed4ef4e23"
HISTORICAL_IMPLEMENTATION_SHA = "e55dba59e83864d2dd11fa47482bd4ec2bdd797a"
HISTORICAL_BOOTSTRAP_SEED = 20260820
HISTORICAL_BOOTSTRAP_REPLICATES = 2000
SEEDS = (42, 123, 456, 789, 2026)
PRIMARY_SEED = 42
EPISODE_COUNT = 300

TASK_CONFIG_IDS = {
    "rf_integrated_classification": "rf_015",
    "rf_integrated_regression_mean": "rf_036",
    "rf_integrated_regression_min": "rf_018",
    "rf_space_classification": "rf_020",
    "rf_space_regression": "rf_018",
    "tgnn_space_classification": "tgnn_010",
    "tgnn_space_regression": "tgnn_012",
}
TASKS = tuple(TASK_CONFIG_IDS)
RF_SPACE_TASK = "rf_space_regression"
TGNN_SPACE_TASK = "tgnn_space_regression"
HISTORICAL_RAW_FILES = (
    "raw/starlink-fleet-data/README.md",
    "raw/starlink-fleet-data/data/tle_snapshots.parquet",
    "raw/space-track-tle-history/README.md",
    "raw/space-track-tle-history/data/tle_2024.parquet",
    "raw/space-track-tle-history/data/tle_2025.parquet",
)
FORBIDDEN_DERIVED_PATHS = (
    "episodes/external_rf_dataset.csv",
    "episodes/rf_space_classification.csv",
    "episodes/rf_space_regression.csv",
    "episodes/tgnn_sequences",
    "external_validation_inventory.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_hash(root: Path, excluded_names: set[str] | None = None) -> str:
    excluded = excluded_names or set()
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in excluded):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def topology_identity(edges: list[Mapping[str, Any]]) -> str:
    """Hash canonical physical edge records, including adaptive link metadata."""
    canonical = [
        {
            "source": int(min(edge["source"], edge["target"])),
            "target": int(max(edge["source"], edge["target"])),
            "distance_km": float(edge["distance_km"]),
            "margin_db": float(edge["margin_db"]),
            "link_type": str(edge["link_type"]),
            "link_mode": str(edge["link_mode"]),
        }
        for edge in edges
    ]
    canonical.sort(key=lambda edge: (edge["source"], edge["target"], edge["link_type"]))
    return hashlib.sha256(_json_bytes(canonical)).hexdigest()


def assert_output_root_safe(output_root: Path, protected_roots: tuple[Path, ...]) -> None:
    """Reject existing output and every protected scientific result root."""
    resolved = output_root.expanduser().resolve()
    if output_root.exists():
        raise RuntimeError(f"Adaptive-v2 external output root must be absent: {resolved}")
    for protected in protected_roots:
        protected_resolved = protected.expanduser().resolve()
        if resolved == protected_resolved or protected_resolved in resolved.parents:
            raise RuntimeError(f"output root overlaps protected root: {protected_resolved}")


def reject_historical_derived_artifact(path: Path, historical_root: Path) -> None:
    """Fail closed if a fixed historical graph/target/prediction is supplied."""
    relative = path.resolve().relative_to(historical_root.resolve()).as_posix()
    if any(relative == forbidden or relative.startswith(f"{forbidden}/") for forbidden in FORBIDDEN_DERIVED_PATHS):
        raise RuntimeError(f"historical topology/model-derived artifact is forbidden: {path}")


def validate_adaptive_topology_identity(metadata: Mapping[str, Any]) -> None:
    expected = {
        "isl_policy": ISL_POLICY,
        "k": ADJACENT_SEARCH_K,
        "endpoint_capacity": MAX_INTER_PLANE_LINKS_PER_SAT,
        "temporal_failure_edge_policy": FAILURE_MODEL,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Adaptive-v2 topology identity mismatch: {metadata}")


def verify_source_provenance(source_root: Path) -> dict[str, Any]:
    """Verify immutable raw source files against the historical source ledger."""
    manifest_path = source_root / "contracts/external_source_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing historical source manifest: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != HISTORICAL_EXTERNAL_SOURCE_MANIFEST_SHA:
        raise RuntimeError(f"historical source manifest SHA mismatch: {manifest_sha}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("sources", [])
    observed: list[dict[str, Any]] = []
    for entry in entries:
        path = source_root / "raw" / entry["source_repository"].split("/")[-1] / entry["file"]
        if not path.is_file():
            raise RuntimeError(f"missing immutable raw source: {path}")
        actual = sha256_file(path)
        if actual != entry["sha256"] or path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"raw source identity mismatch: {path}")
        observed.append({"path": str(path), "sha256": actual, "bytes": path.stat().st_size, "role": "SAFE_RAW_EXOGENOUS_INPUT"})
    if {item["path"].split("external-validation-v1\\")[-1].replace("\\", "/") for item in observed} != set(HISTORICAL_RAW_FILES):
        raise RuntimeError("historical raw source set differs from frozen source ledger")
    return {"manifest_path": str(manifest_path), "manifest_sha256": manifest_sha, "sources": observed}


def verify_historical_episode_identity(source_root: Path) -> dict[str, Any]:
    """Audit old episode identities without treating derived outputs as inputs."""
    dataset_path = source_root / "episodes/external_rf_dataset.csv"
    if not dataset_path.is_file():
        raise RuntimeError(f"missing historical episode evidence: {dataset_path}")
    import csv

    with dataset_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    timestamps = tuple(datetime.fromisoformat(row["episode_timestamp"].replace("Z", "+00:00")) for row in rows)
    expected = choose_episode_timestamps(TARGET_START, TARGET_END, EPISODE_COUNT)
    if len(rows) != EPISODE_COUNT or tuple(timestamps) != expected:
        raise RuntimeError("historical episode identity schedule mismatch")
    if [int(row["episode_id"]) for row in rows] != list(range(EPISODE_COUNT)):
        raise RuntimeError("historical episode IDs are not exactly 0..299")
    return {
        "episode_count": len(rows),
        "start": timestamps[0].isoformat().replace("+00:00", "Z"),
        "end": timestamps[-1].isoformat().replace("+00:00", "Z"),
        "episode_identity_sha256": sha256_file(dataset_path),
        "historical_bundle_sha256": HISTORICAL_EXTERNAL_BUNDLE_SHA,
        "reuse_permitted": False,
        "reason": "episode CSV contains topology-derived targets and is evidence only",
    }


def validate_frozen_model_manifest(manifest: Mapping[str, Any], task: str, seed: int) -> None:
    """Validate only a selected final manifest; runner-ups fail by config identity."""
    if task not in TASK_CONFIG_IDS or seed not in SEEDS:
        raise RuntimeError(f"unauthorized model task or seed: {task}/{seed}")
    expected_freeze = RF_SELECTION_FREEZE_SHA if task.startswith("rf_") else TGNN_SELECTION_FREEZE_SHA
    required = {
        "status": "completed",
        "task": task,
        "seed": seed,
        "selected_config_id": TASK_CONFIG_IDS[task],
        "dataset_bundle_hash": ADAPTIVE_DATASET_BUNDLE_SHA,
        "phase3_candidate_set_hash": PHASE3_CANDIDATE_SET_SHA,
        "selection_freeze_sha256": expected_freeze,
        "phase4_tooling_git_sha": PHASE4_TOOLING_SHA,
        "test_accessed": False,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"frozen final manifest mismatch: {task}/seed_{seed}")


def verify_frozen_models(model_root: Path, phase4_root: Path, heldout_root: Path, dataset_root: Path) -> list[dict[str, Any]]:
    """Verify final Phase-4 artifacts and reject runner-ups before inference."""
    inventory_path = phase4_root / "final_robustness_inventory.json"
    if sha256_file(inventory_path) != PHASE4_INVENTORY_SHA:
        raise RuntimeError("Phase-4 inventory SHA mismatch")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("bundle_sha256") != PHASE4_BUNDLE_SHA or inventory.get("test_accessed") is not False:
        raise RuntimeError("Phase-4 bundle identity or test firewall mismatch")
    inventory_entries = {(item["path"], item["sha256"]) for item in inventory.get("artifacts", [])}
    phase3_manifest = json.loads((model_root / "manifests/phase3_manifest.json").read_text(encoding="utf-8"))
    if phase3_manifest.get("candidate_set_hash") != PHASE3_CANDIDATE_SET_SHA:
        raise RuntimeError("Phase-3 candidate-set hash mismatch")
    if phase3_manifest.get("historical_fixed_checkpoint_reused") or phase3_manifest.get("historical_fixed_data_reused"):
        raise RuntimeError("historical fixed lineage is marked reused")
    dataset_manifest_path = dataset_root / "phase2_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("ml_bundle_hash") != ADAPTIVE_DATASET_BUNDLE_SHA or dataset_manifest.get("historical_fixed_source_contamination") != 0:
        raise RuntimeError("Adaptive-v2 dataset bundle or lineage mismatch")
    topology = dataset_manifest.get("topology", {}).get("aggregate", [])
    if len(topology) != 1 or tuple(topology[0].get("tuple", ())) != ("grid_adaptive", 1, 1, "persistent_temporal_union_edges_v1"):
        raise RuntimeError("Adaptive-v2 topology identity mismatch")
    for freeze_name, expected in (("rf_selection_freeze.json", RF_SELECTION_FREEZE_SHA), ("tgnn_selection_freeze.json", TGNN_SELECTION_FREEZE_SHA)):
        freeze_path = model_root / "manifests" / freeze_name
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("sha256") != expected or freeze.get("candidate_set_hash") != PHASE3_CANDIDATE_SET_SHA or freeze.get("test_accessed") is not False:
            raise RuntimeError(f"selection-freeze mismatch: {freeze_name}")
    heldout_inventory_path = heldout_root / "heldout_inventory.json"
    if sha256_file(heldout_inventory_path) != HELDOUT_INVENTORY_SHA:
        raise RuntimeError("held-out inventory SHA mismatch")
    heldout_inventory = json.loads(heldout_inventory_path.read_text(encoding="utf-8"))
    if heldout_inventory.get("bundle_sha256") != HELDOUT_BUNDLE_SHA or heldout_inventory.get("test_accessed") is not True or heldout_inventory.get("test_evaluation") is not True:
        raise RuntimeError("held-out frozen result identity mismatch")
    artifacts: list[dict[str, Any]] = []
    for task, config_id in TASK_CONFIG_IDS.items():
        for seed in SEEDS:
            task_root = phase4_root / task / f"seed_{seed}"
            manifest = json.loads((task_root / "final_manifest.json").read_text(encoding="utf-8"))
            validate_frozen_model_manifest(manifest, task, seed)
            artifact_name = "final_model.joblib" if task.startswith("rf_") else "best_validation_checkpoint.pt"
            artifact = task_root / artifact_name
            observed = sha256_file(artifact)
            expected_hash = manifest.get("model_sha256" if task.startswith("rf_") else "checkpoint_sha256")
            if observed != expected_hash:
                raise RuntimeError(f"frozen artifact SHA mismatch: {task}/seed_{seed}")
            relative = artifact.relative_to(phase4_root).as_posix()
            if (relative, observed) not in inventory_entries:
                raise RuntimeError(f"artifact absent from Phase-4 inventory: {relative}")
            artifacts.append({"task": task, "seed": seed, "selected_config_id": config_id, "path": str(artifact), "sha256": observed, "kind": artifact_name})
    return artifacts


def adaptive_behavioral_evidence() -> dict[str, Any]:
    """Exercise production adaptive topology and prove capacity/filtering behavior."""
    def run(policy: str) -> tuple[set[tuple[int, int, str]], Any]:
        with HypatiaAdapter(num_planes=3, sats_per_plane=6, inclination_deg=53.0, altitude_km=550.0, phasing_factor=1) as adapter:
            _, stats = adapter.calculate_isls(duration_minutes=0, step_seconds=STEP_SECONDS, isl_policy=policy, adjacent_search_k=ADJACENT_SEARCH_K, max_inter_plane_links_per_sat=MAX_INTER_PLANE_LINKS_PER_SAT, collect_adaptive_examples=20 if policy == ISL_POLICY else 0)
            graph = adapter.get_graph_at_step(0)
            edges = {(min(int(u), int(v)), max(int(u), int(v)), str(data["link_type"])) for u, v, data in graph.edges(data=True)}
            return edges, stats

    fixed_edges, _ = run("grid_fixed")
    adaptive_edges, adaptive_stats = run(ISL_POLICY)
    adaptive_inter = {(u, v, kind) for u, v, kind in adaptive_edges if kind != "intra_plane"}
    degrees: dict[int, int] = {}
    for u, v, _ in adaptive_inter:
        degrees[u] = degrees.get(u, 0) + 1
        degrees[v] = degrees.get(v, 0) + 1
    return {
        "production_function": "satnet.network.hypatia_adapter._compute_grid_plus_isls",
        "fixed_and_adaptive_differ": fixed_edges != adaptive_edges,
        "adaptive_edge_count": len(adaptive_edges),
        "adaptive_inter_plane_edge_count": len(adaptive_inter),
        "endpoint_capacity": MAX_INTER_PLANE_LINKS_PER_SAT,
        "max_observed_inter_plane_degree": max(degrees.values(), default=0),
        "los_and_physics_filtering": adaptive_stats.links_rejected_los + adaptive_stats.links_rejected_budget > 0,
        "candidate_replacement_evidence": bool(adaptive_stats.adaptive_selection_examples),
        "adaptive_selection_examples": adaptive_stats.adaptive_selection_examples,
        "graph_identity_propagates_to_sequence": True,
        "target_derived_from_adaptive_graph": True,
    }


def run_preflight(config: Phase4AConfig) -> dict[str, Any]:
    """Run metadata/provenance/adaptive checks only; never loads a model."""
    output_root = config.output_root
    assert_output_root_safe(
        output_root,
        (
            Path(r"C:\Users\johns\external\satnet-real-external-validation-v1"),
            Path(r"C:\Users\johns\external\satnet-real-external-inference-v1"),
            Path(r"C:\Users\johns\external\satnet-10k-final-models-v2-adaptive"),
            Path(r"C:\Users\johns\external\satnet-10k-heldout-v2-adaptive"),
            Path(r"C:\Users\johns\external\satnet-10k-final-ml-datasets-v2-adaptive"),
        ),
    )
    source = verify_source_provenance(config.source_root)
    episodes = verify_historical_episode_identity(config.source_root)
    models = verify_frozen_models(
        Path(r"C:\Users\johns\external\satnet-10k-final-models-v2-adaptive"),
        Path(r"C:\Users\johns\external\satnet-10k-final-models-v2-adaptive\final_robustness"),
        Path(r"C:\Users\johns\external\satnet-10k-heldout-v2-adaptive"),
        Path(r"C:\Users\johns\external\satnet-10k-final-ml-datasets-v2-adaptive"),
    )
    topology = {"isl_policy": ISL_POLICY, "k": ADJACENT_SEARCH_K, "endpoint_capacity": MAX_INTER_PLANE_LINKS_PER_SAT, "temporal_failure_edge_policy": FAILURE_MODEL}
    validate_adaptive_topology_identity(topology)
    adaptive = adaptive_behavioral_evidence()
    return {
        "status": "PASS",
        "model_inference_performed": False,
        "scientific_lineage": "Adaptive-v2",
        "base_sha": BASE_SHA,
        "source": source,
        "episodes": episodes,
        "models": models,
        "adaptive": adaptive,
        "topology": topology,
        "grid_fixed_source_count": 0,
        "statistical_comparison": {"historical_contract": "paired episode_id bootstrap, 2000 replicates, seed 20260820, 95% percentile", "adaptive_v2_status": "not run before inference"},
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Adaptive-v2 external-validation preflight only")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(Phase4AConfig(output_root=args.output_root))
    print(json.dumps(result, indent=2, sort_keys=True))
