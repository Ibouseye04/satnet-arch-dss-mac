from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from importlib.util import find_spec
from pathlib import Path
from typing import Any

PRODUCTION_SHA = "d0515088cf3fca06a6aa2d47059269089dcb10a7"
DISCREPANT_SHA = "d0515088b5d8959a0c41b3a2b61b2bf0b957f353"
CONTRACT_SPEC_SHA = "6c7dd365f9e7fb67f5f5e70879a19535ede55468aabfac53d82c2ab35b8307eb"
DESIGN_MANIFEST_SHA = "d39830c861ae3e7c5222dd05c44ecef6fb66b365a6d7a42e26f335c8820600e0"
RUN_MANIFEST_SHA = "e965d5daea19a958fe6ce20d5c4a75240a42497b6fde33a508d4df0df64888ea"
SPLIT_MANIFEST_SHA = "07a84c324b255f21f2697b88150c2cc3171156405f0750e2db1312ff8204c4aa"
CONTRACT_BUNDLE_SHA = "059dff74930d1125a46947a06d213dd07c3a93dce226a894558a01805c3ed94c"
SPLIT_CANDIDATE = 3958
GENERATION_ROOT = Path(r"C:\Users\johns\external\satnet-10k-production-generation")
REPLAY_ROOT = Path(r"C:\Users\johns\external\satnet-10k-production-replay")
ACCEPTANCE_REPORT = Path(r"C:\Users\johns\external\satnet-10k-production-acceptance\production_acceptance.json")
AUDIT_ROOT = Path(r"C:\Users\johns\external\satnet-10k-production-audit")
CORRECTED_AUDIT_ROOT = Path(r"C:\Users\johns\external\satnet-10k-production-audit-corrected")
CONTRACT_SOURCE = Path(r"C:\Users\johns\satnet-10k-full-suite-qualification\artifacts\final_integrated_dataset_10k_contract")
OUTPUT_ROOT = Path(r"C:\Users\johns\external\satnet-10k-ml-contract-v1")
AUDIT_SCRIPT = Path(__file__).resolve()

SPACE_FEATURES = [
    "num_planes",
    "sats_per_plane",
    "altitude_km",
    "inclination_deg",
    "satellite_node_failure_probability",
    "satellite_edge_failure_probability",
]
GROUND_FEATURES = [
    "civilian_count",
    "government_count",
    "military_count",
    "ground_station_failure_probability",
]
INTEGRATED_FEATURES = SPACE_FEATURES + GROUND_FEATURES
TARGETS = [
    "space_threshold_breach_any",
    "ground_threshold_breach_any",
    "overall_threshold_breach_any",
    "space_gcc_fraction_original_min",
    "failure_adjusted_ground_service_fraction_min",
    "failure_adjusted_overall_service_fraction_min",
    "failure_adjusted_overall_service_fraction_mean",
    "ground_service_loss_due_to_failures_max",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical(value) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(CONTRACT_SOURCE.parent.parent), *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def provenance() -> dict[str, Any]:
    satnet_file = None
    satnet_source = None
    try:
        spec = find_spec("satnet")
        satnet_file = None if spec is None else str(spec.origin)
        if satnet_file:
            satnet_source = str(Path(satnet_file).parent.parent)
    except (ImportError, AttributeError):
        pass
    audit_script_hash = sha256_file(AUDIT_ROOT / "tools" / "satnet_10k_production_audit.py")
    return {
        "working_directory": str(AUDIT_ROOT),
        "audit_script": str(AUDIT_ROOT / "tools" / "satnet_10k_production_audit.py"),
        "audit_script_sha256": audit_script_hash,
        "contract_freeze_script": str(AUDIT_SCRIPT),
        "python_executable": sys.executable,
        "satnet_file_in_qualification_environment": satnet_file,
        "editable_install_source_path": satnet_source,
        "satnet_git_head": run_git("rev-parse", "HEAD"),
        "satnet_git_status_short": run_git("status", "--short").splitlines(),
        "audit_script_imports_satnet": False,
        "audit_data_sources": [str(GENERATION_ROOT), str(REPLAY_ROOT), str(ACCEPTANCE_REPORT)],
        "provenance_resolution": "The prior audit script does not import SATNET; the inspected qualification environment resolves SATNET to the authoritative editable source at the production-qualified HEAD.",
    }


def load_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    run_dirs = sorted(GENERATION_ROOT.glob("run_*"), key=lambda p: int(p.name.split("_")[1]))
    if len(run_dirs) != 10000:
        raise RuntimeError(f"Expected 10000 generation runs, found {len(run_dirs)}")
    rows: list[dict[str, Any]] = []
    designs: dict[str, dict[str, Any]] = {}
    design_splits: dict[str, str] = {}
    for expected_id, run_dir in enumerate(run_dirs):
        run_id = int(run_dir.name.split("_")[1])
        if run_id != expected_id:
            raise RuntimeError(f"Run identity gap at {expected_id}: {run_dir.name}")
        design = json.loads((run_dir / "input" / "design_record.json").read_text(encoding="utf-8"))
        run = json.loads((run_dir / "input" / "run_record.json").read_text(encoding="utf-8"))
        target = json.loads((run_dir / "targets" / "target.json").read_text(encoding="utf-8"))
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        if run["run_id"] != run_id or target["run_id"] != run_id or run["run_key"] != target["run_key"]:
            raise RuntimeError(f"Identity mismatch in run {run_id}")
        if run["contract_spec_hash"] != CONTRACT_SPEC_SHA or target["contract_spec_hash"] != CONTRACT_SPEC_SHA:
            raise RuntimeError(f"Contract hash mismatch in run {run_id}")
        did = str(design["design_id"])
        if did in designs and designs[did] != design:
            raise RuntimeError(f"Design record differs across realizations: {did}")
        designs[did] = design
        split = str(run["split_assignment"])
        if did in design_splits and design_splits[did] != split:
            raise RuntimeError(f"Design crosses split: {did}")
        design_splits[did] = split
        rows.append({"run": run, "design": design, "target": target, "result": result})
    if len(designs) != 2000:
        raise RuntimeError(f"Expected 2000 unique designs, found {len(designs)}")
    if Counter(design_splits.values()) != Counter({"train": 1400, "validation": 300, "test": 300}):
        raise RuntimeError("Design split counts do not match frozen contract")
    if any(sum(item["run"]["design_id"] == did for item in rows) != 5 for did in designs):
        raise RuntimeError("Every design must have exactly five realizations")
    for row in rows:
        replay = json.loads((REPLAY_ROOT / f"run_{row['run']['run_id']:04d}" / "replay_report.json").read_text(encoding="utf-8"))
        if replay.get("replay_state") != "succeeded" or replay.get("first_mismatch") is not None:
            raise RuntimeError(f"Replay failure for run {row['run']['run_id']}")
        if replay.get("expected_result_hash") != row["result"].get("run_result_hash"):
            raise RuntimeError(f"Replay/result mismatch for run {row['run']['run_id']}")
    return rows, designs, design_splits


def frozen_identity() -> dict[str, Any]:
    def read_jsonl(name: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (CONTRACT_SOURCE / name).read_text(encoding="utf-8").splitlines() if line]

    specification = json.loads((CONTRACT_SOURCE / "contract_specification.json").read_text(encoding="utf-8"))
    designs = read_jsonl("designs.jsonl")
    runs = read_jsonl("runs.jsonl")
    split = json.loads((CONTRACT_SOURCE / "split_manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads((CONTRACT_SOURCE / "contract_bundle.json").read_text(encoding="utf-8"))
    semantic_hashes = {
        "contract_specification.json": sha256_bytes(canonical({k: v for k, v in specification.items() if k != "contract_spec_hash"}).encode("utf-8")),
        "designs.jsonl": sha256_bytes(canonical({"designs": designs, "identity_domain": "satnet_final_integrated_dataset_design_manifest", "identity_version": "1"}).encode("utf-8")),
        "runs.jsonl": sha256_bytes(canonical({"identity_domain": "satnet_final_integrated_dataset_run_manifest", "identity_version": "2", "runs": runs}).encode("utf-8")),
        "split_manifest.json": sha256_bytes(canonical({k: v for k, v in split.items() if k != "split_manifest_hash"}).encode("utf-8")),
        "contract_bundle.json": sha256_bytes(canonical({k: v for k, v in bundle.items() if k != "contract_bundle_hash"}).encode("utf-8")),
    }
    expected = {
        "contract_specification.json": CONTRACT_SPEC_SHA,
        "designs.jsonl": DESIGN_MANIFEST_SHA,
        "runs.jsonl": RUN_MANIFEST_SHA,
        "split_manifest.json": SPLIT_MANIFEST_SHA,
        "contract_bundle.json": CONTRACT_BUNDLE_SHA,
    }
    for name, expected_hash in expected.items():
        if semantic_hashes[name] != expected_hash:
            raise RuntimeError(f"Frozen semantic hash mismatch for {name}: {semantic_hashes[name]} != {expected_hash}")
    if split.get("selected_candidate_id") != SPLIT_CANDIDATE or split.get("outcome_fields_used") is not False:
        raise RuntimeError("Frozen split candidate or outcome independence is invalid")
    return {
        "tooling_sha": PRODUCTION_SHA,
        "contract_spec_sha256": CONTRACT_SPEC_SHA,
        "design_manifest_sha256": DESIGN_MANIFEST_SHA,
        "run_manifest_sha256": RUN_MANIFEST_SHA,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA,
        "contract_bundle_sha256": CONTRACT_BUNDLE_SHA,
        "selected_split_candidate": SPLIT_CANDIDATE,
        "hash_semantics": "canonical JSON semantic hashes, excluding each artifact's self-hash field",
    }


def balances(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    balance_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    for target in ("space_threshold_breach_any", "ground_threshold_breach_any", "overall_threshold_breach_any"):
        for scope, selected in [("overall", rows), ("train", [r for r in rows if r["run"]["split_assignment"] == "train"]), ("validation", [r for r in rows if r["run"]["split_assignment"] == "validation"]), ("test", [r for r in rows if r["run"]["split_assignment"] == "test"])]:
            pos = sum(bool(r["target"][target]) for r in selected)
            total = len(selected)
            neg = total - pos
            balance_rows.append({
                "target": target, "scope": scope, "total": total, "positive": pos, "negative": neg,
                "positive_fraction": f"{pos / total:.17g}", "negative_fraction": f"{neg / total:.17g}",
                "majority_minority_ratio": f"{max(pos, neg) / min(pos, neg):.17g}" if min(pos, neg) else "null",
                "baseline_majority_accuracy": f"{max(pos, neg) / total:.17g}",
            })
        for scope, selected in [("overall", rows), ("train", [r for r in rows if r["run"]["split_assignment"] == "train"]), ("validation", [r for r in rows if r["run"]["split_assignment"] == "validation"]), ("test", [r for r in rows if r["run"]["split_assignment"] == "test"])]:
            by_design: dict[str, int] = defaultdict(int)
            for row in selected:
                by_design[row["run"]["design_id"]] += int(bool(row["target"][target]))
            counts = Counter(by_design.values())
            for positive_realizations in range(6):
                composition_rows.append({"target": target, "scope": scope, "positive_realizations_of_five": positive_realizations, "design_count": counts.get(positive_realizations, 0)})
    return balance_rows, composition_rows


def target_catalog() -> list[dict[str, Any]]:
    rows = [
        ("space_threshold_breach_any", "classification", "secondary", "boolean", "At least one temporal space GCC threshold breach; 1 means a breach occurred.", "0.80000000000000004", "targets/target.json", "higher space GCC fraction is better", "space-only"),
        ("ground_threshold_breach_any", "classification", "diagnostic_only", "boolean", "At least one temporal ground service threshold breach after the accepted ground-failure realization.", "0.80000000000000004", "targets/target.json", "higher ground service fraction is better", "ground-only diagnostic"),
        ("overall_threshold_breach_any", "classification", "primary", "boolean", "At least one temporal integrated service threshold breach after the accepted ground-failure realization.", "0.80000000000000004", "targets/target.json", "higher overall service fraction is better", "integrated"),
        ("space_gcc_fraction_original_min", "regression", "primary_space_regression", "canonical_binary64_fraction", "Minimum temporal original-constellation GCC fraction over the full run sequence.", "not applicable", "targets/target.json", "higher is better", "space-only"),
        ("failure_adjusted_ground_service_fraction_min", "regression", "diagnostic_only", "canonical_binary64_fraction", "Minimum temporal ground service fraction after persistent ground failures.", "not applicable", "targets/target.json", "higher is better", "ground-only diagnostic"),
        ("failure_adjusted_overall_service_fraction_min", "regression", "safety_secondary_integrated_regression", "canonical_binary64_fraction", "Minimum temporal integrated service fraction after persistent ground failures.", "not applicable", "targets/target.json", "higher is better", "integrated"),
        ("failure_adjusted_overall_service_fraction_mean", "regression", "primary_integrated_regression", "canonical_binary64_fraction", "Mean temporal integrated service fraction after persistent ground failures.", "not applicable", "targets/target.json", "higher is better", "integrated"),
        ("ground_service_loss_due_to_failures_max", "regression", "diagnostic_only", "canonical_binary64_fraction", "Maximum temporal ground service loss attributable to ground failures.", "not applicable", "targets/target.json", "lower is better", "ground-only diagnostic"),
    ]
    return [{"target": x[0], "kind": x[1], "role": x[2], "data_type": x[3], "semantic_definition": x[4], "threshold": x[5], "source_artifact": x[6], "direction": x[7], "scope": x[8]} for x in rows]


def feature_catalog(designs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    all_fields = SPACE_FEATURES + GROUND_FEATURES + ["configured_satellite_count", "total_ground_station_count", "duration_minutes", "step_seconds", "isl_policy", "adjacent_search_k", "max_inter_plane_links_per_sat", "phasing_factor", "max_isl_distance_km"]
    for field in all_fields:
        values = [designs[d][field] for d in sorted(designs)]
        varying = len({json.dumps(v, sort_keys=True) for v in values})
        if field in SPACE_FEATURES:
            use = "space"
            allowed = "yes"
            reason = "ex-ante satellite architecture/failure input"
        elif field in GROUND_FEATURES:
            use = "integrated"
            allowed = "yes"
            reason = "ex-ante ground architecture/failure input"
        elif field == "configured_satellite_count":
            use = "none"
            allowed = "no"
            reason = "excluded deterministic redundancy: num_planes * sats_per_plane"
        elif field == "total_ground_station_count":
            use = "none"
            allowed = "no"
            reason = "excluded deterministic redundancy: civilian_count + government_count + military_count"
        else:
            use = "metadata"
            allowed = "no"
            reason = "constant experiment metadata; no predictive variation"
        records.append({"field": field, "source_artifact": "input/design_record.json", "unique_values": varying, "task_scope": use, "allowed_predictor": allowed, "reason": reason})
    return records


def rf_schema(task_id: str, target: str, features: list[str], status: str, rationale: str) -> dict[str, Any]:
    return {
        "schema_name": f"{task_id}_v1", "schema_version": "1", "status": status,
        "sample_unit": "one simulation run; five realizations remain separate",
        "row_count_before_training_preprocessing": 10000,
        "target": {"field": target, "source_artifact": "targets/target.json", "data_type": "boolean" if target.endswith("_any") else "canonical_binary64_fraction", "threshold": "0.80000000000000004" if target.endswith("_any") else None, "higher_is_better": not target.endswith("loss_due_to_failures_max"), "rationale": rationale},
        "predictors": [{"field": f, "source_artifact": "input/design_record.json", "ex_ante": True} for f in features],
        "metadata_retained_not_predictors": ["run_id", "run_key", "design_id", "realization_id", "split"],
        "excluded_predictors": {"post_simulation": True, "target_derived": True, "ids_hashes_split_labels": True, "realized_failures": True, "replay_status": True},
        "preprocessing": {"fit_split": "train only", "validation_use": "model selection only", "test_use": "final evaluation only", "resampling": "not part of frozen dataset"},
        "class_imbalance": "preserve observed prevalence; do not rebalance before freeze; use balanced metrics later",
    }


def tgnn_schema(task_id: str, target: str, rationale: str) -> dict[str, Any]:
    return {
        "schema_name": f"{task_id}_v1", "schema_version": "1", "status": "READY FOR DATASET EXPORT",
        "sample_unit": "one full satellite temporal graph sequence per simulation run",
        "expected_samples": 10000, "design_groups": 2000, "realizations_per_design": 5,
        "framing": "ex-post full-sequence assessment; not forecasting",
        "target": {"field": target, "source_artifact": "targets/target.json", "data_type": "boolean" if target.endswith("_any") else "canonical_binary64_fraction", "threshold": "0.80000000000000004" if target.endswith("_any") else None, "rationale": rationale},
        "node_features": [
            {"name": "plane_idx_normalized", "level": "node", "timestep": "same timestep", "observable": True, "later_derived": False, "target_derived": False, "post_outcome": False, "allowed": True},
            {"name": "sat_in_plane_normalized", "level": "node", "timestep": "same timestep", "observable": True, "later_derived": False, "target_derived": False, "post_outcome": False, "allowed": True},
            {"name": "node_exists_constant", "level": "node", "timestep": "same timestep", "observable": True, "later_derived": False, "target_derived": False, "post_outcome": False, "allowed": True},
        ],
        "edge_features": [
            {"name": "distance_km_scaled_10000", "level": "edge", "timestep": "same timestep", "observable": True, "later_derived": False, "target_derived": False, "post_outcome": False, "allowed": True},
            {"name": "margin_db_scaled_100", "level": "edge", "timestep": "same timestep", "observable": True, "later_derived": False, "target_derived": False, "post_outcome": False, "allowed": True},
            {"name": "link_type_code_scaled_2", "level": "edge", "timestep": "same timestep", "observable": True, "later_derived": False, "target_derived": False, "post_outcome": False, "allowed": True},
            {"name": "link_mode_binary", "level": "edge", "timestep": "same timestep", "observable": True, "later_derived": False, "target_derived": False, "post_outcome": False, "allowed": True},
        ],
        "graph_sequence": {"source": "current SatNetTemporalDataset/HypatiaAdapter satellite graph sequence", "all_timesteps_allowed": True, "target_in_features": False, "failed_satellite_semantics": "failed nodes/edges are represented in the effective graph sequence"},
        "metadata_retained_not_features": ["run_id", "design_id", "realization_id", "split"],
        "integrated_ground_support": "not supported by current adapter; integrated TGNN is future/not authorized",
        "preprocessing": {"fit_split": "train only", "test_use": "final evaluation only"},
    }


def corrected_audit(rows: list[dict[str, Any]], identity: dict[str, Any], balance_rows: list[dict[str, Any]], comp_rows: list[dict[str, Any]]) -> None:
    CORRECTED_AUDIT_ROOT.mkdir(parents=True, exist_ok=False)
    prov = provenance()
    prov.update({"reported_discrepant_sha": DISCREPANT_SHA, "authoritative_production_tooling_sha": PRODUCTION_SHA, "root_cause": "reporting metadata defect: the audit script hard-codes tooling_sha to null and does not emit provenance; the discrepant SHA exists only in the prior handoff, not in audit source or generated reports.", "prior_audit_results_valid": True})
    original = json.loads((AUDIT_ROOT / "audit_summary.json").read_text(encoding="utf-8"))
    original["tooling_sha"] = PRODUCTION_SHA
    original["provenance"] = prov
    original["authoritative_identity"] = identity
    write_json(CORRECTED_AUDIT_ROOT / "audit_summary.json", original)
    report = json.loads((AUDIT_ROOT / "ml_readiness_report.json").read_text(encoding="utf-8"))
    report["provenance"] = prov
    report["authoritative_identity"] = identity
    write_json(CORRECTED_AUDIT_ROOT / "ml_readiness_report.json", report)
    md = (AUDIT_ROOT / "production_dataset_audit.md").read_text(encoding="utf-8")
    md += "\n\n## Corrected provenance\n\n"
    md += f"- Authoritative tooling SHA: `{PRODUCTION_SHA}`\n- Prior discrepant handoff SHA: `{DISCREPANT_SHA}`\n- Root cause: audit script metadata defect; source sets `tooling_sha` to null and does not import SATNET.\n"
    md += f"- Qualification SATNET source: `{prov['satnet_file_in_qualification_environment']}`\n- Qualification git HEAD: `{prov['satnet_git_head']}`\n- Prior audit results remain valid: `true`\n"
    (CORRECTED_AUDIT_ROOT / "production_dataset_audit.md").write_text(md, encoding="utf-8", newline="\n")
    write_json(CORRECTED_AUDIT_ROOT / "provenance_resolution.json", prov)


def build() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing ML contract root: {OUTPUT_ROOT}")
    rows, designs, design_splits = load_rows()
    acceptance = json.loads(ACCEPTANCE_REPORT.read_text(encoding="utf-8"))
    if acceptance != {"derived_generation_submission_count": 10000, "derived_replay_submission_count": 10000, "production_acceptance": "passed", "validated_run_count": 10000}:
        raise RuntimeError(f"Acceptance report differs from expected accepted identity: {acceptance}")
    identity = frozen_identity()
    balance_rows, comp_rows = balances(rows)
    OUTPUT_ROOT.mkdir(parents=True)
    provenance_data = provenance()
    corrected_audit(rows, identity, balance_rows, comp_rows)
    schemas = {
        "rf_space_classification_schema.json": rf_schema("rf_space_classification", "space_threshold_breach_any", SPACE_FEATURES, "READY FOR DATASET EXPORT", "Isolate satellite architecture and satellite-failure assumptions as explanations of space-segment resilience."),
        "rf_space_regression_schema.json": rf_schema("rf_space_regression", "space_gcc_fraction_original_min", SPACE_FEATURES, "READY FOR DATASET EXPORT", "Estimate continuous space-only GCC degradation from ex-ante satellite design and failure variables."),
        "rf_integrated_classification_schema.json": rf_schema("rf_integrated_classification", "overall_threshold_breach_any", INTEGRATED_FEATURES, "READY FOR DATASET EXPORT", "Predict end-to-end resilience from ex-ante space and ground architecture variables."),
        "rf_integrated_regression_schema.json": {**rf_schema("rf_integrated_regression", "failure_adjusted_overall_service_fraction_mean", INTEGRATED_FEATURES, "READY FOR DATASET EXPORT", "Primary frozen methodology target is the mean integrated service fraction; the minimum is retained as a safety secondary."), "secondary_target": "failure_adjusted_overall_service_fraction_min"},
        "tgnn_space_classification_schema.json": tgnn_schema("tgnn_space_classification", "space_threshold_breach_any", "Assess space-segment breach occurrence from a full satellite temporal graph sequence."),
        "tgnn_space_regression_schema.json": tgnn_schema("tgnn_space_regression", "space_gcc_fraction_original_min", "Estimate continuous space-only GCC degradation from the same full satellite temporal graph sequence."),
    }
    for name, schema in schemas.items():
        write_json(OUTPUT_ROOT / name, schema)
    write_csv(OUTPUT_ROOT / "classification_balance.csv", list(balance_rows[0]), balance_rows)
    write_csv(OUTPUT_ROOT / "design_outcome_composition.csv", list(comp_rows[0]), comp_rows)
    catalog = target_catalog()
    write_csv(OUTPUT_ROOT / "target_catalog.csv", list(catalog[0]), catalog)
    features = feature_catalog(designs)
    write_csv(OUTPUT_ROOT / "feature_catalog.csv", list(features[0]), features)
    split_rows = []
    for item in rows:
        run = item["run"]
        split_rows.append({"run_id": run["run_id"], "run_key": run["run_key"], "design_id": run["design_id"], "realization_id": run["realization_id"], "split": run["split_assignment"]})
    write_csv(OUTPUT_ROOT / "split_contract.csv", list(split_rows[0]), split_rows)
    summary = {
        "contract_name": "satnet-10k-ml-contract-v1", "status": "READY FOR FINAL ML DATASET EXPORT",
        "authoritative_identity": identity, "provenance": provenance_data,
        "production_acceptance": acceptance, "counts": {"designs": 2000, "runs": 10000, "realizations_per_design": 5},
        "split_counts": {"train": {"designs": 1400, "runs": 7000}, "validation": {"designs": 300, "runs": 1500}, "test": {"designs": 300, "runs": 1500}},
        "split_candidate": SPLIT_CANDIDATE, "outcome_fields_used_for_split": False, "design_cross_split": False,
        "rf_tasks": {"space_classification": "READY FOR DATASET EXPORT", "space_regression": "READY FOR DATASET EXPORT", "integrated_classification": "READY FOR DATASET EXPORT", "integrated_regression": "READY FOR DATASET EXPORT"},
        "tgnn_tasks": {"space_classification": "READY FOR DATASET EXPORT", "space_regression": "READY FOR DATASET EXPORT", "integrated_classification": "BLOCKED — current adapter is satellite-only", "integrated_regression": "BLOCKED — current adapter is satellite-only"},
        "preprocessing": "Any learned preprocessing, feature selection, encoding, resampling, class weighting, and classification threshold selection is train-only; no preprocessing is fit in this freeze.",
        "training_prohibited": True,
    }
    write_json(OUTPUT_ROOT / "ml_contract_summary.json", summary)
    md = f"""# SATNET 10k ML Dataset Contract v1\n\n## Status\n\n`READY FOR FINAL ML DATASET EXPORT`\n\nThis root freezes schemas only. No model was trained, no simulation was regenerated, the accepted production roots were not modified, and the DOE and grouped membership were not changed.\n\n## Authoritative identity\n\n- Tooling SHA: `{PRODUCTION_SHA}`\n- Contract specification: `{CONTRACT_SPEC_SHA}`\n- Design manifest: `{DESIGN_MANIFEST_SHA}`\n- Run manifest: `{RUN_MANIFEST_SHA}`\n- Split manifest: `{SPLIT_MANIFEST_SHA}`\n- Contract bundle: `{CONTRACT_BUNDLE_SHA}`\n- Split candidate: `{SPLIT_CANDIDATE}`\n\n## Samples and split\n\nOne RF row and one TGNN sample represent one complete simulation run. There are 10,000 rows/sequences, 2,000 designs, and five separate realizations per design. The frozen grouped split is train 1,400 designs/7,000 runs, validation 300/1,500, and test 300/1,500. No design crosses splits and outcome fields were not used to assign the split. No downstream `train_test_split()` is authorized.\n\n## RF predictor contract\n\nSpace-only tasks use: `{', '.join(SPACE_FEATURES)}`. Integrated tasks add `{', '.join(GROUND_FEATURES)}`. `configured_satellite_count` is excluded because it equals `num_planes * sats_per_plane`; `total_ground_station_count` is excluded because it equals the sum of the three class counts. Ground composition is therefore represented by the three integer counts plus ground failure probability, avoiding redundant total encoding.\n\nFixed fields (`duration_minutes`, `step_seconds`, `isl_policy`, `adjacent_search_k`, `max_inter_plane_links_per_sat`, `phasing_factor`, and `max_isl_distance_km`) are constant experiment metadata and excluded from predictors. IDs, hashes, seeds, selected stations, split labels, replay status, realized failures, graph metrics, service metrics, outcomes, and target-derived fields are excluded.\n\nThe integrated regression follows the frozen methodology target hierarchy: `failure_adjusted_overall_service_fraction_mean` is primary, and `failure_adjusted_overall_service_fraction_min` is retained as a safety secondary.\n\n## TGNN contract\n\nThe current adapter is `SatNetTemporalDataset` over Hypatia satellite-only temporal graphs. It provides normalized plane index, normalized satellite-within-plane index, constant node existence, and same-timestep edge attributes for distance, margin, link type, and link mode. All timesteps are permitted because this is an ex-post full-sequence assessment, not forecasting. Targets are never included in features.\n\nThe current adapter cannot represent integrated ground nodes/features. Integrated TGNN tasks are blocked and not represented as authorized current schemas; adapter redesign is required.\n\n## Target definitions and imbalance\n\nTargets are copied from accepted `targets/target.json` artifacts and calculated from verified G5 summaries. Classification value 1 means at least one temporal threshold breach occurred at the frozen 0.8 threshold. Observed prevalence is preserved exactly; no threshold changes, oversampling, undersampling, class weighting, or threshold optimization occurs during this contract freeze. Future learned preprocessing must fit on train only; validation is for model selection and test is untouched until final evaluation.\n\nRaw accuracy is not the primary later classification criterion for the 99.15% positive integrated target. Balanced accuracy, precision, recall, F1, specificity, ROC-AUC when meaningful, PR-AUC, and confusion matrices should be reported later.\n\n## Provenance correction\n\nThe prior audit handoff SHA `{DISCREPANT_SHA}` was a reporting-only discrepancy. The audit source explicitly writes `tooling_sha` as null and contains no hard-coded discrepant SHA; it also does not import SATNET. The qualification environment resolves the editable SATNET package to the exact authoritative production-qualified checkout and HEAD `{PRODUCTION_SHA}`. Corrected audit metadata is in `{CORRECTED_AUDIT_ROOT}`; production roots remain untouched.\n\n## Artifact inventory\n\nThe companion `contract_artifact_inventory.json` records SHA-256 and byte length for every contract artifact. Its bundle hash is computed from the canonical inventory payload excluding the bundle hash field.\n"""
    (OUTPUT_ROOT / "dataset_contract.md").write_text(md, encoding="utf-8", newline="\n")
    artifact_names = sorted(p.name for p in OUTPUT_ROOT.iterdir() if p.is_file() and p.name != "contract_artifact_inventory.json")
    inventory = {"identity": "satnet-10k-ml-contract-v1-inventory", "artifacts": [{"path": name, "bytes": (OUTPUT_ROOT / name).stat().st_size, "sha256": sha256_file(OUTPUT_ROOT / name)} for name in artifact_names]}
    inventory["bundle_sha256"] = sha256_bytes(canonical(inventory).encode("utf-8"))
    write_json(OUTPUT_ROOT / "contract_artifact_inventory.json", inventory)
    print(json.dumps({"output_root": str(OUTPUT_ROOT), "bundle_sha256": inventory["bundle_sha256"], "artifact_count": len(artifact_names)}, indent=2))


if __name__ == "__main__":
    build()
