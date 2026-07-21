from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from satnet.experiments.stage_a_contract.proposal import (
    ARTIFACT_SCHEMAS,
    build_artifact_payloads,
    build_inventory as build_proposal_inventory,
)

APPROVED_PROPOSAL_COMMIT = "509f2449dbbaf4c1f5153ecfa4bc1652f24f75da"
APPROVED_PROPOSAL_INVENTORY = "69fd2a9fbcc1b461ae4230a282cfcaaadc26ba7b65563bc90a81e86e1ca98127"
APPROVED_SEED_MANIFEST = "ab37564cca81e926b6e43caf74f1ff2be641b8bc203e32c3ae6a219ea6f7bace"
AUDIT_COMMIT = "a1514a654fe76518db16001b98e899f773eb9d1e"
AUDIT_IMPLEMENTATION_COMMIT = "27b889b83e26158778f4031df9084c3c5d825d5f"
AUDIT_INVENTORY = "16ce1a1b138567a144cbf9b1d715c74b30f05e1d8262841d9580240080339d10"
AUDIT_INVENTORY_BYTE_LENGTH = 3691
AUDIT_INVENTORY_LF_COUNT = 107
AUDIT_INVENTORY_CRLF_COUNT = 0
BYTE_POLICY_COMMIT = "9c55e5999f308a84b857fdf6828cf48a9ca5b360"
PRODUCTION_TOOLING_SHA = "9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab"
BASE_CONTRACT_TAG = "final-integrated-dataset-contract-v1"
BASE_CONTRACT_COMMIT = "a1967185e80327e4b00c1831828dc975ab6819fc"
BASE_CONTRACT_SPECIFICATION = "482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b"
GENERATION_LEDGER = "a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1"
REPLAY_LEDGER = "4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd"
EVIDENCE_FREEZE_ARCHIVE = "375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc"
CONTRACT_NAME = "SATNET Stage A Discovery Contract"
CONTRACT_VERSION = "1"
CORPUS_NAMESPACE = "stage_a_discovery_v1"
FREEZE_DATE = "2026-07-21"
FREEZE_STATUS = "FROZEN_PENDING_INDEPENDENT_AUDIT"
FREEZE_BRANCH = "freeze/stage-a-discovery-contract-v1-restart"
TAG_NAME = "stage-a-discovery-contract-v1"
REQUIRED_NEXT_TASK = "Independent audit of the frozen SATNET Stage A Discovery Contract v1"
SOURCE_ROOT_RELATIVE = Path("artifacts/stage_a_discovery_contract_proposal")
AUDIT_ROOT_RELATIVE = Path("artifacts/stage_a_discovery_contract_freeze_audit")
SOURCE_ARTIFACTS = (
    "stage_a_contract_proposal.json",
    "stage_a_design_manifest.csv",
    "stage_a_discovery_criteria.json",
    "stage_a_near_neighbor_policy.json",
    "stage_a_output_root_manifest.json",
    "stage_a_partition_manifest.json",
    "stage_a_proposal_inventory.json",
    "stage_a_region_bounds.json",
    "stage_a_run_manifest.csv",
    "stage_a_seed_manifest.csv",
    "stage_a_seed_policy.json",
)
SPECIFICATION_NAME = "stage_a_frozen_contract_specification.json"
INVENTORY_NAME = "stage_a_frozen_contract_inventory.json"
HASH_NAME = "stage_a_frozen_contract.sha256"
DECLARATION_NAME = "stage_a_contract_freeze_declaration.json"
README_NAME = "STAGE_A_CONTRACT_FREEZE_README.txt"
TOP_LEVEL_ARTIFACTS = frozenset(
    {SPECIFICATION_NAME, INVENTORY_NAME, HASH_NAME, DECLARATION_NAME, README_NAME}
)
OUTPUT_ROOTS = {
    "production_generation": r"C:\Users\johns\satnet-stage-a-discovery-v1-production",
    "production_replay": r"C:\Users\johns\satnet-stage-a-discovery-v1-replay",
    "production_acceptance": r"C:\Users\johns\satnet-stage-a-discovery-v1-acceptance",
    "evidence_freeze": r"C:\Users\johns\satnet-stage-a-discovery-v1-freeze",
}
FROZEN_EVIDENCE_ROOTS = (
    r"C:\Users\johns\satnet-final-production-20260720",
    r"C:\Users\johns\satnet-final-production-replay-20260720",
    r"C:\Users\johns\satnet-final-production-v1-freeze-20260721",
    r"C:\Users\johns\satnet-final-production-v1-freeze-20260721.zip",
)
PROTECTED_PATHS = (
    "src/satnet/ground",
    "src/satnet/network",
    "src/satnet/simulation/tier1_rollout.py",
    "src/satnet/models/gnn_dataset.py",
    "src/satnet/models/gnn_model.py",
    "src/satnet/models/risk_model.py",
    "src/satnet/utils/graph_cache.py",
    "artifacts/final_integrated_dataset_contract",
    "src/satnet/experiments/final_dataset",
    "src/satnet/experiments/final_generation",
)
SOURCE_SCHEMAS = ARTIFACT_SCHEMAS | {
    "stage_a_proposal_inventory.json": "satnet.stage_a.proposal_inventory.v1"
}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def csv_rows(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))


def git(repo_root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"Git command failed: {' '.join(args)}: {detail}")
    return result.stdout.strip()


def verify_approved_audit_bundle(repo_root: Path) -> list[dict[str, Any]]:
    audit_root = repo_root / AUDIT_ROOT_RELATIVE
    inventory_path = audit_root / "audit_inventory.json"
    inventory_payload = inventory_path.read_bytes()
    if len(inventory_payload) != AUDIT_INVENTORY_BYTE_LENGTH:
        raise ValueError("Freeze-readiness audit inventory byte length mismatch")
    if sha256_bytes(inventory_payload) != AUDIT_INVENTORY:
        raise ValueError("Freeze-readiness audit inventory SHA-256 mismatch")
    if inventory_payload.count(b"\n") != AUDIT_INVENTORY_LF_COUNT:
        raise ValueError("Freeze-readiness audit inventory LF count mismatch")
    if inventory_payload.count(b"\r\n") != AUDIT_INVENTORY_CRLF_COUNT:
        raise ValueError("Freeze-readiness audit inventory CRLF conversion detected")
    inventory = json.loads(inventory_payload)
    records = inventory.get("artifacts")
    if not isinstance(records, list) or len(records) != 16:
        raise ValueError("Freeze-readiness audit inventory membership mismatch")
    if inventory.get("artifact_count_excluding_inventory") != 16:
        raise ValueError("Freeze-readiness audit inventory count mismatch")
    if inventory.get("proposal_commit_audited") != APPROVED_PROPOSAL_COMMIT:
        raise ValueError("Freeze-readiness audit proposal identity mismatch")
    if inventory.get("schema_identifier") != "satnet.stage_a.freeze_readiness_audit_inventory.v1":
        raise ValueError("Freeze-readiness audit inventory schema mismatch")
    paths = [record.get("relative_path") for record in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Freeze-readiness audit inventory ordering or uniqueness mismatch")
    expected_names = {*paths, "audit_inventory.json"}
    actual_entries = {path.name for path in audit_root.iterdir()}
    if actual_entries != expected_names or any(not path.is_file() for path in audit_root.iterdir()):
        raise ValueError("Freeze-readiness audit bundle contains missing or extra artifacts")
    results: list[dict[str, Any]] = []
    for record in records:
        relative = record.get("relative_path")
        if not isinstance(relative, str) or "\\" in relative or Path(relative).is_absolute():
            raise ValueError("Freeze-readiness audit inventory path format mismatch")
        path = audit_root / relative
        payload = path.read_bytes()
        digest = sha256_bytes(payload)
        if len(payload) != record.get("byte_length") or digest != record.get("sha256"):
            raise ValueError(f"Freeze-readiness audit artifact byte mismatch: {relative}")
        results.append(
            {
                "relative_path": relative,
                "byte_length": len(payload),
                "sha256": digest,
                "lf_count": payload.count(b"\n"),
                "crlf_count": payload.count(b"\r\n"),
                "inventory_match": True,
            }
        )
    tracked = set(
        git(
            repo_root,
            "ls-files",
            "--",
            AUDIT_ROOT_RELATIVE.as_posix(),
        ).splitlines()
    )
    expected_tracked = {
        f"{AUDIT_ROOT_RELATIVE.as_posix()}/{name}" for name in expected_names
    }
    if tracked != expected_tracked:
        raise ValueError("Freeze-readiness audit tracked-file membership mismatch")
    results.append(
        {
            "relative_path": "audit_inventory.json",
            "byte_length": len(inventory_payload),
            "sha256": sha256_bytes(inventory_payload),
            "lf_count": inventory_payload.count(b"\n"),
            "crlf_count": inventory_payload.count(b"\r\n"),
            "inventory_match": True,
        }
    )
    return results


def verify_byte_preservation_attributes(repo_root: Path) -> None:
    expected = {
        f"{AUDIT_ROOT_RELATIVE.as_posix()}/audit_inventory.json": "unset",
        "artifacts/stage_a_discovery_contract_v1/stage_a_frozen_contract_inventory.json": "unset",
        f"{SOURCE_ROOT_RELATIVE.as_posix()}/stage_a_proposal_inventory.json": "unset",
        "README.md": "unspecified",
    }
    for path, expected_value in expected.items():
        output = git(repo_root, "check-attr", "text", "--", path)
        observed = output.rsplit(": ", 1)[-1]
        if observed != expected_value:
            raise ValueError(f"Stage A byte-preservation attribute mismatch: {path}")


def _overlaps(first: Path, second: Path) -> bool:
    left = os.path.normcase(os.path.abspath(first))
    right = os.path.normcase(os.path.abspath(second))
    try:
        common = os.path.commonpath((left, right))
    except ValueError:
        return False
    return common == left or common == right


def reproduce_approved_proposal(repo_root: Path) -> dict[str, bytes]:
    source_root = repo_root / SOURCE_ROOT_RELATIVE
    actual_names = tuple(sorted(path.name for path in source_root.iterdir() if path.is_file()))
    if actual_names != SOURCE_ARTIFACTS:
        raise ValueError("Approved proposal artifact set mismatch")
    generated = build_artifact_payloads(repo_root)
    generated["stage_a_proposal_inventory.json"] = canonical_json_bytes(
        build_proposal_inventory(generated)
    )
    if tuple(sorted(generated)) != SOURCE_ARTIFACTS:
        raise ValueError("Proposal reproduction artifact set mismatch")
    for name in SOURCE_ARTIFACTS:
        if source_root.joinpath(name).read_bytes() != generated[name]:
            raise ValueError(f"Approved proposal reproduction mismatch: {name}")
    if sha256_file(source_root / "stage_a_proposal_inventory.json") != APPROVED_PROPOSAL_INVENTORY:
        raise ValueError("Approved proposal inventory SHA-256 mismatch")
    if sha256_file(source_root / "stage_a_seed_manifest.csv") != APPROVED_SEED_MANIFEST:
        raise ValueError("Approved seed-manifest SHA-256 mismatch")
    return generated


def verify_repository_identity(
    repo_root: Path,
    expected_proposal_commit: str,
    expected_audit_commit: str,
    expected_audit_inventory: str,
    expected_byte_policy_commit: str,
) -> None:
    if expected_proposal_commit != APPROVED_PROPOSAL_COMMIT:
        raise ValueError("Wrong approved proposal commit")
    if expected_audit_commit != AUDIT_COMMIT:
        raise ValueError("Wrong freeze-readiness audit commit")
    if expected_audit_inventory != AUDIT_INVENTORY:
        raise ValueError("Wrong freeze-readiness audit inventory")
    if expected_byte_policy_commit != BYTE_POLICY_COMMIT:
        raise ValueError("Wrong byte-policy correction commit")
    for label, commit in (
        ("approved proposal", expected_proposal_commit),
        ("freeze-readiness audit", expected_audit_commit),
        ("byte-policy correction", expected_byte_policy_commit),
    ):
        if git(repo_root, "cat-file", "-t", commit) != "commit":
            raise ValueError(f"{label} commit unavailable")
        git(repo_root, "merge-base", "--is-ancestor", commit, "HEAD")
    if git(repo_root, "rev-list", "-n", "1", BASE_CONTRACT_TAG) != BASE_CONTRACT_COMMIT:
        raise ValueError("Frozen final-integrated contract tag target mismatch")
    base_specification = read_json(
        repo_root / "artifacts/final_integrated_dataset_contract/contract_specification.json"
    )
    if base_specification.get("contract_spec_hash") != BASE_CONTRACT_SPECIFICATION:
        raise ValueError("Frozen final-integrated contract specification hash mismatch")
    verify_byte_preservation_attributes(repo_root)
    verify_approved_audit_bundle(repo_root)
    audit_root = repo_root / AUDIT_ROOT_RELATIVE
    if git(
        repo_root,
        "diff",
        "--name-only",
        AUDIT_COMMIT,
        "--",
        AUDIT_ROOT_RELATIVE.as_posix(),
    ):
        raise ValueError("Freeze-readiness audit bundle changed relative to approved commit")
    findings = read_json(audit_root / "audit_findings.json")
    if findings.get("final_verdict") != "APPROVED FOR STAGE A CONTRACT FREEZE":
        raise ValueError("Freeze-readiness audit verdict mismatch")
    if any(
        findings.get(field)
        for field in ("binding_findings", "nonbinding_findings", "required_corrections")
    ):
        raise ValueError("Freeze-readiness audit contains unresolved findings")


def verify_repository_boundaries(repo_root: Path) -> None:
    source_path = SOURCE_ROOT_RELATIVE.as_posix()
    if git(repo_root, "diff", "--name-only", APPROVED_PROPOSAL_COMMIT, "--", source_path):
        raise ValueError("Approved proposal source changed relative to approved commit")
    if git(repo_root, "status", "--short", "--untracked-files=all", "--", source_path):
        raise ValueError("Approved proposal source has tracked or untracked changes")
    changed = git(
        repo_root,
        "diff",
        "--name-only",
        APPROVED_PROPOSAL_COMMIT,
        "--",
        *PROTECTED_PATHS,
    )
    status = git(repo_root, "status", "--short", "--untracked-files=all", "--", *PROTECTED_PATHS)
    if changed or status:
        raise ValueError("Protected science or frozen production content changed")


def verify_output_roots(repo_root: Path) -> dict[str, bool]:
    worktrees = [
        Path(line.removeprefix("worktree "))
        for line in git(repo_root, "worktree", "list", "--porcelain").splitlines()
        if line.startswith("worktree ")
    ]
    outputs = {name: Path(value) for name, value in OUTPUT_ROOTS.items()}
    for name, path in outputs.items():
        if path.exists():
            raise ValueError(f"Reserved Stage A output root exists: {name}")
        others = [value for key, value in outputs.items() if key != name]
        protected = [repo_root, *worktrees, *(Path(value) for value in FROZEN_EVIDENCE_ROOTS)]
        if any(_overlaps(path, candidate) for candidate in [*others, *protected]):
            raise ValueError(f"Reserved Stage A output root overlaps protected path: {name}")
    return {name: False for name in sorted(outputs)}


def _source_payloads(source_root: Path) -> dict[str, bytes]:
    entries = tuple(sorted(source_root.iterdir(), key=lambda path: path.name))
    names = tuple(path.name for path in entries)
    if names != SOURCE_ARTIFACTS or any(not path.is_file() for path in entries):
        raise ValueError("Frozen source-bundle artifact set mismatch")
    return {name: source_root.joinpath(name).read_bytes() for name in SOURCE_ARTIFACTS}


def _source_records(payloads: Mapping[str, bytes]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in SOURCE_ARTIFACTS:
        payload = payloads[name]
        record_count = len(csv_rows(payload)) if name.endswith(".csv") else None
        records.append(
            {
                "relative_path": f"source_bundle/{name}",
                "byte_length": len(payload),
                "sha256": sha256_bytes(payload),
                "schema_identifier": SOURCE_SCHEMAS[name],
                "record_count": record_count,
                "source_classification": "approved_proposal_source_copy",
            }
        )
    return records


def build_frozen_specification(repo_root: Path, source_root: Path) -> dict[str, Any]:
    payloads = _source_payloads(source_root)
    proposal = json.loads(payloads["stage_a_contract_proposal.json"])
    partition = json.loads(payloads["stage_a_partition_manifest.json"])
    neighbor = json.loads(payloads["stage_a_near_neighbor_policy.json"])
    roots = json.loads(payloads["stage_a_output_root_manifest.json"])
    designs = csv_rows(payloads["stage_a_design_manifest.csv"])
    runs = csv_rows(payloads["stage_a_run_manifest.csv"])
    seeds = csv_rows(payloads["stage_a_seed_manifest.csv"])
    audit_minima = read_json(repo_root / AUDIT_ROOT_RELATIVE / "audit_neighbor_minima.json")
    region_counts = Counter(row["region"] for row in designs)
    partition_counts = Counter(row["partition"] for row in designs)
    region_partition = Counter((row["region"], row["partition"]) for row in designs)
    if len(designs) != 30 or len(runs) != 150 or len(seeds) != 150:
        raise ValueError("Stage A design/run/seed cardinality mismatch")
    expected_designs = [f"SA-D{index:03d}" for index in range(30)]
    if [row["design_id"] for row in designs] != expected_designs:
        raise ValueError("Stage A design identity range mismatch")
    if [int(row["global_run_id"]) for row in runs] != list(range(500, 650)):
        raise ValueError("Stage A global run identity range mismatch")
    expected_keys = [f"SA-D{design:03d}-R{realization:02d}" for design in range(30) for realization in range(5)]
    if [row["run_key"] for row in runs] != expected_keys:
        raise ValueError("Stage A run-key identity range mismatch")
    if len({row["run_key"] for row in runs}) != 150:
        raise ValueError("Duplicate Stage A run key")
    if len({row["global_run_id"] for row in runs}) != 150:
        raise ValueError("Duplicate Stage A global run ID")
    if len({(row["design_id"], row["realization_id"]) for row in runs}) != 150:
        raise ValueError("Duplicate Stage A design-realization pair")
    if roots["proposed_resolved_paths"] != OUTPUT_ROOTS:
        raise ValueError("Stage A reserved output-root identity mismatch")
    if region_counts != Counter({"resilient_core": 12, "boundary": 12, "global_control": 6}):
        raise ValueError("Stage A region allocation mismatch")
    if partition_counts != Counter({"development": 20, "validation": 5, "sealed_holdout": 5}):
        raise ValueError("Stage A partition allocation mismatch")
    expected_region_partition = Counter(
        {
            ("resilient_core", "development"): 8,
            ("resilient_core", "validation"): 2,
            ("resilient_core", "sealed_holdout"): 2,
            ("boundary", "development"): 8,
            ("boundary", "validation"): 2,
            ("boundary", "sealed_holdout"): 2,
            ("global_control", "development"): 4,
            ("global_control", "validation"): 1,
            ("global_control", "sealed_holdout"): 1,
        }
    )
    if region_partition != expected_region_partition:
        raise ValueError("Stage A region-by-partition allocation mismatch")
    if Counter(row["design_id"] for row in runs) != Counter(
        {design_id: 5 for design_id in expected_designs}
    ):
        raise ValueError("Stage A realizations-per-design mismatch")
    if [row["run_key"] for row in seeds] != expected_keys:
        raise ValueError("Stage A seed identity range mismatch")
    if len({row["run_key"] for row in seeds}) != 150:
        raise ValueError("Duplicate Stage A seed identity")
    if proposal.get("threshold_definition") != {
        "value": "0.80",
        "margin": "failure_adjusted_overall_service_fraction_min - 0.80",
        "non_breach": "margin >= 0",
        "breach": "margin < 0",
    }:
        raise ValueError("Stage A target threshold or margin rule mismatch")
    boundary = proposal.get("boundary_definitions", {})
    if boundary.get("observed_boundary_design") != (
        "Five margins straddle zero or at least two have absolute margin <= 0.05"
    ) or boundary.get("endpoint_behavior") != "Absolute margin 0.05 qualifies; margin 0 is non-breach":
        raise ValueError("Stage A observed-boundary rule mismatch")
    margin = proposal.get("distinct_margin_definition", {})
    if margin.get("quantization_increment") != "0.000001" or margin.get("rounding_mode") != "ROUND_HALF_EVEN":
        raise ValueError("Stage A canonical-margin quantization rule mismatch")
    classes = proposal.get("class_support_definitions", {})
    if classes.get("mixed_design") != (
        "at least one of each class; counts once by majority and may separately count as observed boundary"
    ):
        raise ValueError("Stage A mixed-design majority-class rule mismatch")
    final_corpus = proposal.get("final_corpus_membership", {})
    if final_corpus.get("primary_final_classification_corpus") != [
        "original frozen 500-run corpus",
        "future frozen Stage B corpus",
    ] or any(
        final_corpus.get(field) != "EXCLUDED"
        for field in ("stage_a_development", "stage_a_validation", "stage_a_sealed_holdout")
    ):
        raise ValueError("Stage A final-corpus exclusion rule mismatch")
    stage_b = proposal.get("stage_b_adaptation_boundary", {})
    if stage_b.get("adaptation_sources") != ["Stage A development", "Stage A validation"] or stage_b.get("prohibited_source") != "Stage A sealed holdout":
        raise ValueError("Stage B adaptation boundary mismatch")
    holdout = partition.get("partitions", {}).get("sealed_holdout", {})
    if holdout.get("sealed") is not True or holdout.get("design_count") != 5 or holdout.get("run_count") != 25:
        raise ValueError("Stage A sealed-holdout cardinality or state mismatch")
    required_prohibitions = {
        "Select Stage B parameter bounds",
        "Select Stage B regions",
        "Select Stage B design density",
        "Select Stage B sample size",
        "Select Stage B split allocation",
        "Select Stage B seeds",
        "Select Stage B acceptance gates",
        "Modify a frozen Stage B contract",
    }
    if not required_prohibitions.issubset(set(holdout.get("prohibited_uses", []))):
        raise ValueError("Stage A sealed-holdout leakage prohibition mismatch")
    if neighbor.get("pending_scientific_reviews") != [] or neighbor.get("justified_exceptions") != []:
        raise ValueError("Stage A near-neighbor review or exception remains pending")
    selected = neighbor.get("candidate_review", {}).get("ranked_admissible_alternatives", [None])[0]
    if not isinstance(selected, dict) or selected.get("selected") is not True or selected.get("candidate_values") != {
        "ground_station_failure_probability": "0.10000000000000001"
    }:
        raise ValueError("Stage A SA-D020 correction mismatch")
    expected_minima = {
        "original_sa_d013_sa_d020_distance": 0.07681919236933395,
        "corrected_sa_d013_sa_d020_distance": 0.12039492645571381,
    }
    if any(audit_minima.get(name) != value for name, value in expected_minima.items()):
        raise ValueError("Stage A SA-D013/SA-D020 near-neighbor distance mismatch")
    for name, value in (
        ("minimum_development_validation", 0.12039492645571381),
        ("minimum_development_holdout", 0.1205683310788027),
        ("minimum_validation_holdout", 0.1500946005884104),
    ):
        if audit_minima.get(name, {}).get("distance") != value:
            raise ValueError(f"Stage A near-neighbor minimum mismatch: {name}")
    source_records = _source_records(payloads)
    return {
        "schema_identifier": "satnet.stage_a.frozen_contract_specification.v1",
        "contract_name": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "corpus_namespace": CORPUS_NAMESPACE,
        "freeze_date": FREEZE_DATE,
        "freeze_status": FREEZE_STATUS,
        "contract_frozen": True,
        "simulation_authorized": False,
        "production_authorized": False,
        "execution_authorized": False,
        "approved_proposal": {
            "branch": "correction/stage-a-near-neighbor-resolution-v1",
            "commit": APPROVED_PROPOSAL_COMMIT,
            "proposal_inventory_sha256": APPROVED_PROPOSAL_INVENTORY,
            "seed_manifest_sha256": APPROVED_SEED_MANIFEST,
        },
        "byte_preservation_policy": {
            "commit": BYTE_POLICY_COMMIT,
            "audit_bundle_attribute": "text unset",
            "frozen_contract_bundle_attribute": "text unset",
            "approved_proposal_bundle_attribute": "text unset",
        },
        "freeze_readiness_audit": {
            "branch": "audit/stage-a-contract-freeze-readiness-v1",
            "commit": AUDIT_COMMIT,
            "implementation_commit": AUDIT_IMPLEMENTATION_COMMIT,
            "report_commit": AUDIT_COMMIT,
            "inventory_sha256": AUDIT_INVENTORY,
            "verdict": "APPROVED FOR STAGE A CONTRACT FREEZE",
            "binding_findings": [],
            "nonbinding_findings": [],
            "required_corrections": [],
        },
        "base_production_provenance": {
            "production_tooling_sha": PRODUCTION_TOOLING_SHA,
            "frozen_contract_tag": BASE_CONTRACT_TAG,
            "frozen_contract_commit": BASE_CONTRACT_COMMIT,
            "contract_specification_sha256": BASE_CONTRACT_SPECIFICATION,
            "generation_ledger_sha256": GENERATION_LEDGER,
            "replay_ledger_sha256": REPLAY_LEDGER,
            "evidence_freeze_archive_sha256": EVIDENCE_FREEZE_ARCHIVE,
        },
        "source_bundle": {
            "artifact_count": 11,
            "artifacts": source_records,
            "preservation": "Byte-exact copies of the independently audited proposal artifacts.",
            "historical_status_explanation": (
                "The source bundle preserves the exact audited proposal bytes. Its internal "
                "NOT_FROZEN fields represent the source proposal's historical state. The frozen "
                "contract envelope binds that exact source bundle as Stage A Discovery Contract "
                "v1. Simulation authorization remains false."
            ),
        },
        "cardinality": {
            "design_count": 30,
            "run_count": 150,
            "realizations_per_design": 5,
            "seed_record_count": 150,
        },
        "region_allocation": dict(sorted(region_counts.items())),
        "partition_allocation": {
            name: {
                "design_count": partition["partitions"][name]["design_count"],
                "run_count": partition["partitions"][name]["run_count"],
            }
            for name in ("development", "validation", "sealed_holdout")
        },
        "region_by_partition_allocation": {
            region: {
                name: region_partition[(region, name)]
                for name in ("development", "validation", "sealed_holdout")
            }
            for region in ("resilient_core", "boundary", "global_control")
        },
        "identity_namespace": {
            "design_ids": "SA-D000 through SA-D029",
            "design_indexes": "0 through 29",
            "realization_ids": "R00 through R04",
            "realization_indexes": "0 through 4",
            "global_run_ids": "500 through 649",
            "run_keys": "SA-D000-R00 through SA-D029-R04",
            "global_run_id_formula": (
                "500 + stage_a_design_index * 5 + realization_index"
            ),
            "collision_policy": {
                "original_identity_collision": "PROHIBITED",
                "stage_a_identity_collision": "PROHIBITED",
                "duplicate_design_id": "PROHIBITED",
                "duplicate_run_id": "PROHIBITED",
                "duplicate_run_key": "PROHIBITED",
                "duplicate_design_realization_pair": "PROHIBITED",
            },
        },
        "threshold_definition": proposal["threshold_definition"],
        "boundary_definitions": {
            "preassigned_boundary_region": (
                "DOE designation only; not automatically an observed boundary result"
            ),
            "observed_boundary_design": {
                "condition_a": "minimum realization margin < 0 and maximum realization margin >= 0",
                "condition_b": "at least two of five realizations have absolute margin <= 0.05",
                "combination": "condition_a OR condition_b",
            },
            "endpoint_behavior": "absolute margin exactly 0.05 qualifies",
        },
        "design_level_class_rules": {
            "non_breach_majority": "at least 3 of 5 realizations are non-breach",
            "breach_majority": "at least 3 of 5 realizations are breach",
            "mixed_design": "at least one breach and at least one non-breach realization",
            "counting_rule": (
                "Each design counts toward exactly one majority-class minimum; mixed designs are "
                "assigned by majority class, are not double-counted, and may separately count as "
                "observed boundary designs."
            ),
        },
        "canonical_margin": {
            "quantization_increment": "0.000001",
            "rounding": "ROUND_HALF_EVEN",
            "distinct_margin_count": "number of unique quantized margins",
        },
        "sealed_holdout_policy": {
            "partition": partition["partitions"]["sealed_holdout"],
            "unsealing_conditions": partition["unsealing_conditions"],
            "change_rule": partition["holdout_change_rule"],
            "may_confirm_or_reject_only": True,
            "may_alter_frozen_stage_b_contract": False,
        },
        "final_corpus_membership": proposal["final_corpus_membership"],
        "stage_b_adaptation_boundary": proposal["stage_b_adaptation_boundary"],
        "near_neighbor_policy": {
            "duplicate_policy": neighbor["duplicate_policy"],
            "cross_partition_policy": neighbor["cross_partition_policy"],
            "sealed_holdout_policy": neighbor["sealed_holdout_policy"],
            "changed_design": "SA-D020",
            "changed_parameter": "ground_station_failure_probability",
            "original_value": "0.075",
            "corrected_value": "0.100",
            "original_sa_d013_sa_d020_distance": audit_minima["original_sa_d013_sa_d020_distance"],
            "corrected_sa_d013_sa_d020_distance": audit_minima["corrected_sa_d013_sa_d020_distance"],
            "minimum_development_validation": audit_minima["minimum_development_validation"]["distance"],
            "minimum_development_holdout": audit_minima["minimum_development_holdout"]["distance"],
            "minimum_validation_holdout": audit_minima["minimum_validation_holdout"]["distance"],
            "minimum_stage_a_to_original": audit_minima["minimum_stage_a_to_original"],
            "pending_scientific_reviews": neighbor["pending_scientific_reviews"],
            "remaining_exceptions": neighbor["justified_exceptions"],
        },
        "future_output_roots": {
            "paths": OUTPUT_ROOTS,
            "required_state_at_freeze": "ABSENT",
            "creation_authorized": False,
            "must_be_outside_git_worktrees": True,
            "must_be_outside_frozen_evidence": True,
            "must_be_mutually_non_overlapping": True,
        },
        "future_requirements": {
            "required_next_step": REQUIRED_NEXT_TASK,
            "independent_frozen_contract_audit_required": True,
            "separate_execution_tooling_task_required": True,
            "separate_production_authorization_required": True,
        },
        "protected_science_boundary": {
            "scientific_content_changes_authorized": False,
            "proposal_source_changes_authorized": False,
            "frozen_production_evidence_changes_authorized": False,
            "protected_paths": list(PROTECTED_PATHS),
        },
        "git_provenance": {
            "freeze_branch": FREEZE_BRANCH,
            "freeze_base_audit_commit": AUDIT_COMMIT,
            "byte_policy_correction_commit": BYTE_POLICY_COMMIT,
            "annotated_tag": TAG_NAME,
        },
    }


def build_frozen_inventory(
    source_payloads: Mapping[str, bytes], specification_payload: bytes
) -> dict[str, Any]:
    records = _source_records(source_payloads)
    records.append(
        {
            "relative_path": SPECIFICATION_NAME,
            "byte_length": len(specification_payload),
            "sha256": sha256_bytes(specification_payload),
            "schema_identifier": "satnet.stage_a.frozen_contract_specification.v1",
            "record_count": None,
            "source_classification": "frozen_contract_specification",
        }
    )
    records.sort(key=lambda record: record["relative_path"])
    return {
        "schema_identifier": "satnet.stage_a.frozen_contract_inventory.v1",
        "contract_name": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "ordering": "relative_path ordinal lexical ascending",
        "path_format": "relative POSIX forward-slash paths",
        "hash_algorithm": "SHA-256",
        "hash_encoding": "lowercase hexadecimal",
        "encoding": "UTF-8 without BOM",
        "newline": "LF with one terminal newline",
        "self_reference_policy": "This inventory excludes its own bytes.",
        "contract_bound_artifact_count": len(records),
        "artifacts": records,
    }


def build_freeze_declaration(contract_hash: str, inventory_length: int) -> dict[str, Any]:
    return {
        "schema_identifier": "satnet.stage_a.contract_freeze_declaration.v1",
        "contract_name": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "freeze_date": FREEZE_DATE,
        "contract_hash": contract_hash,
        "contract_inventory_path": INVENTORY_NAME,
        "contract_inventory_byte_length": inventory_length,
        "contract_inventory_sha256": contract_hash,
        "freeze_branch": FREEZE_BRANCH,
        "freeze_base_audit_commit": AUDIT_COMMIT,
        "approved_proposal_commit": APPROVED_PROPOSAL_COMMIT,
        "approved_proposal_inventory_sha256": APPROVED_PROPOSAL_INVENTORY,
        "audit_commit": AUDIT_COMMIT,
        "audit_inventory_sha256": AUDIT_INVENTORY,
        "audit_verdict": "APPROVED FOR STAGE A CONTRACT FREEZE",
        "byte_policy_correction_commit": BYTE_POLICY_COMMIT,
        "contract_frozen": True,
        "freeze_status": FREEZE_STATUS,
        "simulation_authorized": False,
        "production_authorized": False,
        "execution_authorized": False,
        "tag_name": TAG_NAME,
        "tag_pushed": False,
        "required_next_step": REQUIRED_NEXT_TASK,
        "prohibited_actions": [
            "Stage A simulation",
            "Stage A production",
            "Stage A replay",
            "Stage A acceptance",
            "Stage B simulation",
            "RF training",
            "TGNN training",
        ],
    }


def build_freeze_readme(contract_hash: str) -> bytes:
    text = f"""STAGE A CONTRACT FROZEN
SIMULATION NOT AUTHORIZED
INDEPENDENT FROZEN-CONTRACT AUDIT REQUIRED

SATNET Stage A Discovery Contract v1

Freeze state:
{FREEZE_STATUS}

Simulation authorization:
FALSE

Production authorization:
FALSE

Execution authorization:
FALSE

Approved proposal commit:
{APPROVED_PROPOSAL_COMMIT}

Freeze-readiness audit commit:
{AUDIT_COMMIT}

Proposal inventory hash:
{APPROVED_PROPOSAL_INVENTORY}

Seed-manifest hash:
{APPROVED_SEED_MANIFEST}

Audit inventory hash:
{AUDIT_INVENTORY}

Frozen contract hash:
{contract_hash}

Frozen contract inventory path:
{INVENTORY_NAME}

Git tag:
{TAG_NAME}

Counts:
30 designs
150 runs
150 seed records
5 realizations per design

Region allocation:
12 resilient_core
12 boundary
6 global_control

Partition allocation:
20 development designs / 100 runs
5 validation designs / 25 runs
5 sealed_holdout designs / 25 runs

Identity ranges:
SA-D000 through SA-D029
Global run IDs 500 through 649
SA-D000-R00 through SA-D029-R04

Holdout status:
SEALED

Final-corpus membership:
All Stage A partitions are excluded from the primary final classification corpus.

Output-root status:
All four reserved future output roots are absent and creation is unauthorized.

Required next task:
{REQUIRED_NEXT_TASK}

This freeze does not authorize simulation.

Independent frozen-contract audit is mandatory before execution-tooling work.
"""
    return text.encode("utf-8")


def _expected_bundle_payloads(repo_root: Path, source_root: Path) -> dict[str, bytes]:
    source_payloads = _source_payloads(source_root)
    specification_payload = canonical_json_bytes(
        build_frozen_specification(repo_root, source_root)
    )
    inventory_payload = canonical_json_bytes(
        build_frozen_inventory(source_payloads, specification_payload)
    )
    contract_hash = sha256_bytes(inventory_payload)
    return {
        SPECIFICATION_NAME: specification_payload,
        INVENTORY_NAME: inventory_payload,
        HASH_NAME: f"{contract_hash}  {INVENTORY_NAME}\n".encode("utf-8"),
        DECLARATION_NAME: canonical_json_bytes(
            build_freeze_declaration(contract_hash, len(inventory_payload))
        ),
        README_NAME: build_freeze_readme(contract_hash),
    }


def create_frozen_contract(repo_root: Path, contract_root: Path) -> dict[str, Any]:
    verify_repository_identity(
        repo_root,
        APPROVED_PROPOSAL_COMMIT,
        AUDIT_COMMIT,
        AUDIT_INVENTORY,
        BYTE_POLICY_COMMIT,
    )
    if git(repo_root, "branch", "--show-current") != FREEZE_BRANCH:
        raise ValueError("Stage A contract freeze must run on the restart freeze branch")
    if git(repo_root, "tag", "--list", TAG_NAME):
        raise ValueError("Stage A frozen contract tag already exists")
    verify_repository_boundaries(repo_root)
    verify_output_roots(repo_root)
    proposal_payloads = reproduce_approved_proposal(repo_root)
    if contract_root.exists():
        raise FileExistsError(f"Frozen contract root already exists: {contract_root}")
    source_root = contract_root / "source_bundle"
    source_root.mkdir(parents=True)
    approved_root = repo_root / SOURCE_ROOT_RELATIVE
    for name in SOURCE_ARTIFACTS:
        shutil.copyfile(approved_root / name, source_root / name)
        if source_root.joinpath(name).read_bytes() != proposal_payloads[name]:
            raise ValueError(f"Frozen source-copy byte mismatch: {name}")
    for name, payload in _expected_bundle_payloads(repo_root, source_root).items():
        contract_root.joinpath(name).write_bytes(payload)
    return validate_frozen_contract(
        repo_root=repo_root,
        contract_root=contract_root,
        expected_proposal_commit=APPROVED_PROPOSAL_COMMIT,
        expected_proposal_inventory=APPROVED_PROPOSAL_INVENTORY,
        expected_audit_commit=AUDIT_COMMIT,
        expected_audit_inventory=AUDIT_INVENTORY,
    )


def _validate_inventory_records(contract_root: Path, inventory: Mapping[str, Any]) -> None:
    records = inventory.get("artifacts")
    if not isinstance(records, list) or len(records) != 12:
        raise ValueError("Frozen contract inventory must bind exactly 12 artifacts")
    paths = [record.get("relative_path") for record in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Frozen contract inventory path ordering or uniqueness mismatch")
    expected_paths = sorted(
        [f"source_bundle/{name}" for name in SOURCE_ARTIFACTS] + [SPECIFICATION_NAME]
    )
    if paths != expected_paths:
        raise ValueError("Frozen contract inventory membership mismatch")
    for record in records:
        relative = record["relative_path"]
        if "\\" in relative or Path(relative).is_absolute():
            raise ValueError("Frozen contract inventory path format mismatch")
        path = contract_root / Path(relative)
        if not path.is_file():
            raise ValueError(f"Frozen contract-bound artifact missing: {relative}")
        if path.stat().st_size != record["byte_length"]:
            raise ValueError(f"Frozen contract-bound artifact length mismatch: {relative}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Frozen contract-bound artifact SHA-256 mismatch: {relative}")


def _validate_authorization(specification: Mapping[str, Any], declaration: Mapping[str, Any]) -> None:
    for label, value in (("specification", specification), ("declaration", declaration)):
        if value.get("contract_frozen") is not True:
            raise ValueError(f"Frozen contract marked unfrozen in {label}")
        if value.get("freeze_status") != FREEZE_STATUS:
            raise ValueError(f"Frozen contract status mismatch in {label}")
        for field in (
            "simulation_authorized",
            "production_authorized",
            "execution_authorized",
        ):
            if value.get(field) is not False:
                raise ValueError(f"Stage A authorization enabled: {label}.{field}")
    if declaration.get("tag_name") != TAG_NAME:
        raise ValueError("Frozen contract tag name mismatch")


def validate_frozen_contract(
    *,
    repo_root: Path,
    contract_root: Path,
    expected_proposal_commit: str,
    expected_proposal_inventory: str,
    expected_audit_commit: str,
    expected_audit_inventory: str,
    expected_byte_policy_commit: str = BYTE_POLICY_COMMIT,
    enforce_repository: bool = True,
    enforce_output_roots: bool = True,
) -> dict[str, Any]:
    if expected_proposal_inventory != APPROVED_PROPOSAL_INVENTORY:
        raise ValueError("Wrong approved proposal inventory SHA-256")
    if not contract_root.is_dir():
        raise FileNotFoundError(contract_root)
    actual_top_files = {path.name for path in contract_root.iterdir() if path.is_file()}
    actual_top_dirs = {path.name for path in contract_root.iterdir() if path.is_dir()}
    if actual_top_files != TOP_LEVEL_ARTIFACTS or actual_top_dirs != {"source_bundle"}:
        raise ValueError("Frozen contract bundle contains missing or extra artifacts")
    if enforce_repository:
        verify_repository_identity(
            repo_root,
            expected_proposal_commit,
            expected_audit_commit,
            expected_audit_inventory,
            expected_byte_policy_commit,
        )
        verify_repository_boundaries(repo_root)
        approved = reproduce_approved_proposal(repo_root)
    else:
        if expected_proposal_commit != APPROVED_PROPOSAL_COMMIT:
            raise ValueError("Wrong approved proposal commit")
        if expected_audit_commit != AUDIT_COMMIT or expected_audit_inventory != AUDIT_INVENTORY:
            raise ValueError("Wrong freeze-readiness audit identity")
        if expected_byte_policy_commit != BYTE_POLICY_COMMIT:
            raise ValueError("Wrong byte-policy correction commit")
        approved = {
            name: repo_root.joinpath(SOURCE_ROOT_RELATIVE, name).read_bytes()
            for name in SOURCE_ARTIFACTS
        }
    source_root = contract_root / "source_bundle"
    frozen_sources = _source_payloads(source_root)
    for name in SOURCE_ARTIFACTS:
        if frozen_sources[name] != approved[name]:
            raise ValueError(f"Frozen source-copy byte mismatch: {name}")
    expected_payloads = _expected_bundle_payloads(repo_root, source_root)
    for name, expected in expected_payloads.items():
        if contract_root.joinpath(name).read_bytes() != expected:
            raise ValueError(f"Frozen contract derived artifact mismatch: {name}")
    inventory_payload = contract_root.joinpath(INVENTORY_NAME).read_bytes()
    inventory = json.loads(inventory_payload)
    _validate_inventory_records(contract_root, inventory)
    contract_hash = sha256_bytes(inventory_payload)
    hash_record = contract_root.joinpath(HASH_NAME).read_text(encoding="utf-8")
    if hash_record != f"{contract_hash}  {INVENTORY_NAME}\n":
        raise ValueError("Frozen contract hash record mismatch")
    specification = read_json(contract_root / SPECIFICATION_NAME)
    declaration = read_json(contract_root / DECLARATION_NAME)
    _validate_authorization(specification, declaration)
    if declaration.get("contract_hash") != contract_hash:
        raise ValueError("Freeze declaration contract hash mismatch")
    output_states = verify_output_roots(repo_root) if enforce_output_roots else {
        name: False for name in sorted(OUTPUT_ROOTS)
    }
    return {
        "contract_artifact_count": inventory["contract_bound_artifact_count"],
        "source_artifact_count": specification["source_bundle"]["artifact_count"],
        "design_count": specification["cardinality"]["design_count"],
        "run_count": specification["cardinality"]["run_count"],
        "seed_count": specification["cardinality"]["seed_record_count"],
        "contract_inventory_sha256": contract_hash,
        "contract_hash": contract_hash,
        "authorization_state": {
            "simulation_authorized": False,
            "production_authorized": False,
            "execution_authorized": False,
        },
        "output_root_state": output_states,
        "validation_verdict": "PASSED",
    }


def verify_frozen_production_evidence() -> dict[str, Any]:
    from satnet.experiments.final_class_support_audit.audit import verify_frozen_evidence

    return verify_frozen_evidence(
        production_tooling_root=Path(r"C:\Users\johns\satnet-production-tooling-20260720"),
        generation_root=Path(FROZEN_EVIDENCE_ROOTS[0]),
        replay_root=Path(FROZEN_EVIDENCE_ROOTS[1]),
        freeze_root=Path(FROZEN_EVIDENCE_ROOTS[2]),
        freeze_archive=Path(FROZEN_EVIDENCE_ROOTS[3]),
        freeze_archive_hash_file=Path(
            r"C:\Users\johns\satnet-final-production-v1-freeze-20260721.zip.sha256"
        ),
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-proposal-commit", default=APPROVED_PROPOSAL_COMMIT)
    parser.add_argument("--expected-proposal-inventory", default=APPROVED_PROPOSAL_INVENTORY)
    parser.add_argument("--expected-audit-commit", default=AUDIT_COMMIT)
    parser.add_argument("--expected-audit-inventory", default=AUDIT_INVENTORY)
    parser.add_argument("--expected-byte-policy-commit", default=BYTE_POLICY_COMMIT)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m satnet.experiments.stage_a_contract.freeze")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--contract-root", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--contract-root", type=Path, required=True)
    _add_identity_arguments(validate_parser)
    evidence_parser = subparsers.add_parser("verify-frozen-evidence")
    args = parser.parse_args(argv)
    repo_root = _repo_root()
    if args.command == "create":
        result = create_frozen_contract(repo_root, args.contract_root)
    elif args.command == "validate":
        result = validate_frozen_contract(
            repo_root=repo_root,
            contract_root=args.contract_root,
            expected_proposal_commit=args.expected_proposal_commit,
            expected_proposal_inventory=args.expected_proposal_inventory,
            expected_audit_commit=args.expected_audit_commit,
            expected_audit_inventory=args.expected_audit_inventory,
            expected_byte_policy_commit=args.expected_byte_policy_commit,
        )
    else:
        result = verify_frozen_production_evidence()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
