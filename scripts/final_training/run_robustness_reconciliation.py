from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(r"C:\Users\johns\external\satnet-10k-model-training-v1")
FINAL_ROOT = ROOT / "final_robustness"
RECON_ROOT = ROOT / "robustness_reconciliation"
REPO = Path(r"C:\Users\johns\external\satnet-10k-training-worktree-v1")
QUALIFIED_PYTHON = Path(r"C:\Users\johns\venvs\satnet-10k-qualification\Scripts\python.exe")
EXPECTED_BUNDLE_SHA = "12d8db5f7b96f3b2e920911c0c877525153e2a23fab3b3d21321af5b79f3ab33"
EXPECTED_SEEDS = (42, 123, 456, 789, 2026)
RF_TASKS = (
    "rf_space_classification",
    "rf_space_regression",
    "rf_integrated_regression_mean",
    "rf_integrated_regression_min",
    "rf_integrated_classification",
)
TGNN_TASKS = ("tgnn_space_classification", "tgnn_space_regression")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def self_excluding_bundle_hash(root: Path, exclude_name: str) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != exclude_name):
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        entries.append({"path": relative, "bytes": len(payload), "sha256": sha256_bytes(payload)})
    return digest.hexdigest(), entries


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_frozen_bundle() -> dict[str, Any]:
    inventory_path = FINAL_ROOT / "final_robustness_inventory.json"
    inventory = load_json(inventory_path)
    observed, entries = self_excluding_bundle_hash(FINAL_ROOT, inventory_path.name)
    if observed != EXPECTED_BUNDLE_SHA:
        raise RuntimeError(f"Frozen robustness bundle mismatch: {observed} != {EXPECTED_BUNDLE_SHA}")
    if inventory.get("bundle_sha256") != EXPECTED_BUNDLE_SHA:
        raise RuntimeError("Existing robustness inventory does not contain the expected bundle hash")
    expected_entries = inventory.get("artifacts")
    if expected_entries != entries:
        raise RuntimeError("Existing robustness inventory entries do not match the byte-identical bundle")
    return {"expected": EXPECTED_BUNDLE_SHA, "observed": observed, "files": len(entries)}


def load_estimator(path: Path) -> Any:
    try:
        import joblib
        return joblib.load(path)
    except ImportError:
        with path.open("rb") as handle:
            return pickle.load(handle)


def reconcile_rf(task: str, seed: int) -> dict[str, Any]:
    run_root = FINAL_ROOT / task / f"seed_{seed}"
    manifest_path = run_root / "final_manifest.json"
    model_path = run_root / "final_model.joblib"
    manifest = load_json(manifest_path)
    checks = {
        "status_completed": manifest.get("status") == "completed",
        "manifest_seed_matches": manifest.get("seed") == seed,
        "test_accessed_false": manifest.get("test_accessed") is False,
        "model_sha256_matches": sha256_file(model_path) == manifest.get("model_sha256"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"RF provenance check failed for {task}/seed_{seed}: {checks}")
    estimator = load_estimator(model_path)
    random_state = getattr(estimator, "random_state", None)
    checks["persisted_random_state_matches"] = random_state == seed
    if not checks["persisted_random_state_matches"]:
        raise RuntimeError(f"RF random_state mismatch for {task}/seed_{seed}: {random_state} != {seed}")
    return {
        "task": task,
        "seed": seed,
        "manifest_path": str(manifest_path),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "manifest_configuration": manifest.get("configuration"),
        "persisted_random_state": random_state,
        "checks": checks,
    }


def parse_progress(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid progress JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict) or not isinstance(value.get("epoch"), int):
                raise RuntimeError(f"Invalid progress epoch at {path}:{line_number}")
            records.append(value)
    epochs = [int(record["epoch"]) for record in records]
    expected = list(range(1, len(epochs) + 1))
    if not epochs or epochs != expected:
        raise RuntimeError(f"Progress epochs are not exactly monotonic 1..N in {path}: {epochs}")
    return records, epochs[-1]


def reconcile_tgnn(task: str, seed: int) -> dict[str, Any]:
    run_root = FINAL_ROOT / task / f"seed_{seed}"
    manifest_path = run_root / "final_manifest.json"
    progress_path = run_root / "progress.jsonl"
    checkpoint_path = run_root / "best_validation_checkpoint.pt"
    manifest = load_json(manifest_path)
    checks = {
        "status_completed": manifest.get("status") == "completed",
        "manifest_seed_matches": manifest.get("seed") == seed,
        "test_accessed_false": manifest.get("test_accessed") is False,
        "checkpoint_sha256_matches": sha256_file(checkpoint_path) == manifest.get("checkpoint_sha256"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"TGNN provenance check failed for {task}/seed_{seed}: {checks}")
    records, actual_epochs_run = parse_progress(progress_path)
    checks["progress_valid"] = True
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("epoch"), int):
        raise RuntimeError(f"Checkpoint metadata missing integer epoch for {task}/seed_{seed}")
    checkpoint_selected_epoch = int(checkpoint["epoch"])
    checks["checkpoint_selected_epoch_matches_manifest"] = checkpoint_selected_epoch == manifest.get("selected_epoch")
    checks["selected_epoch_within_actual_run"] = int(manifest["selected_epoch"]) <= actual_epochs_run
    if not checks["checkpoint_selected_epoch_matches_manifest"] or not checks["selected_epoch_within_actual_run"]:
        raise RuntimeError(f"TGNN epoch provenance check failed for {task}/seed_{seed}: {checks}")
    max_epochs = int(manifest["max_epochs"])
    return {
        "task": task,
        "seed": seed,
        "manifest_path": str(manifest_path),
        "progress_path": str(progress_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_epoch": checkpoint_selected_epoch,
        "reported_epochs_run": int(manifest["epochs_run"]),
        "actual_epochs_run": actual_epochs_run,
        "max_epochs": max_epochs,
        "reported_stopped_early": bool(manifest.get("early_stopping", {}).get("stopped_early")),
        "actual_stopped_early": actual_epochs_run < max_epochs,
        "progress_record_count": len(records),
        "checks": checks,
    }


def main() -> None:
    frozen_bundle = verify_frozen_bundle()
    rf_results = [reconcile_rf(task, seed) for task in RF_TASKS for seed in EXPECTED_SEEDS]
    tgnn_results = [reconcile_tgnn(task, seed) for task in TGNN_TASKS for seed in EXPECTED_SEEDS]
    driver_sha = sha256_file(Path(__file__))
    result = {
        "schema_version": "satnet.robustness_provenance_reconciliation.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authoritative_paths": {
            "execution_worktree": str(REPO),
            "qualified_python": str(QUALIFIED_PYTHON),
            "final_robustness_root": str(FINAL_ROOT),
        },
        "identities": {
            "implementation_sha": "e55dba59e83864d2dd11fa47482bd4ec2bdd797a",
            "production_sha": "d0515088cf3fca06a6aa2d47059269089dcb10a7",
            "dataset_bundle_sha": "38dacd66432bfa660410ff9cab7f181a53151120e8102dc40df57016f78bbfc3",
            "training_plan_bundle_sha": "e14ec5e5b2221aa2c9aa187a2baacc517e676f9dc748612c206eb1dc43788a0a",
            "rf_selection_sha": "a1b7076197f123ecf62b0b79ddd092bdb66f0d7d8b94d6c113dc201744163911",
            "tgnn_selection_sha": "c85dfdcd3b6a62d741675ac522fda76bd3de7dd4c311a82bc1b12e079dab2bfb",
            "frozen_robustness_bundle_sha": EXPECTED_BUNDLE_SHA,
        },
        "frozen_bundle_check": frozen_bundle,
        "rf": {
            "models_checked": len(rf_results),
            "random_state_matches": sum(bool(item["checks"]["persisted_random_state_matches"]) for item in rf_results),
            "mismatches": 0,
            "results": rf_results,
            "driver_note": "The robustness driver constructed each RF estimator using params = {**spec, 'random_state': seed, 'n_jobs': -1}, but serialized configuration: spec. The persisted fitted estimator is authoritative for per-run RF randomness; the manifest configuration is not authoritative for random_state.",
        },
        "tgnn": {
            "runs_checked": len(tgnn_results),
            "checkpoint_selected_epoch_matches": sum(bool(item["checks"]["checkpoint_selected_epoch_matches_manifest"]) for item in tgnn_results),
            "progress_logs_valid": sum(bool(item["checks"]["progress_valid"]) for item in tgnn_results),
            "mismatches": 0,
            "results": tgnn_results,
            "driver_note": "The robustness driver serialized epochs_run and selected_epoch as best_epoch and set stopped_early from best_epoch < max_epochs. selected_epoch/checkpoint epoch are authoritative; actual_epochs_run and actual_stopped_early are reconstructed from progress.jsonl.",
        },
        "scientific_training_invalidated": False,
        "retraining_required": False,
        "robustness_bundle_modified": False,
        "test_accessed": False,
        "reconciliation_driver_sha256": driver_sha,
    }
    json_path = RECON_ROOT / "robustness_provenance_reconciliation.json"
    report_path = RECON_ROOT / "robustness_provenance_reconciliation_report.md"
    inventory_path = RECON_ROOT / "robustness_reconciliation_inventory.json"
    atomic_json(json_path, result)
    report = "# Robustness Provenance Reconciliation\n\n" + json.dumps(result, indent=2, sort_keys=True) + "\n"
    report += "\n## TGNN Actual Epochs\n\n| task | seed | selected_epoch | reported_epochs_run | actual_epochs_run | max_epochs | reported_stopped_early | actual_stopped_early | checkpoint_sha256 |\n|---|---:|---:|---:|---:|---:|---|---|---|\n"
    for item in tgnn_results:
        report += f"| {item['task']} | {item['seed']} | {item['selected_epoch']} | {item['reported_epochs_run']} | {item['actual_epochs_run']} | {item['max_epochs']} | {item['reported_stopped_early']} | {item['actual_stopped_early']} | {item['checkpoint_sha256']} |\n"
    report_path.write_text(report, encoding="utf-8")
    bundle_sha, entries = self_excluding_bundle_hash(RECON_ROOT, inventory_path.name)
    atomic_json(inventory_path, {
        "schema_version": "satnet.robustness_reconciliation_inventory.v1",
        "inventory_self_excluding": True,
        "hash_algorithm": "sha256 over sorted relative UTF-8 path + NUL byte + raw file bytes",
        "bundle_sha256": bundle_sha,
        "artifacts": entries,
        "driver_sha256": driver_sha,
        "test_accessed": False,
    })
    print("ROBUSTNESS PROVENANCE RECONCILIATION PASS — HELD-OUT TEST AUTHORIZED")


if __name__ == "__main__":
    main()
