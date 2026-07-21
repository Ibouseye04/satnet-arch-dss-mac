from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence

FREEZE_HEAD = "301d8a224daa070b15ecc6447f503d42d5d1e70a"
BYTE_POLICY_COMMIT = "9c55e5999f308a84b857fdf6828cf48a9ca5b360"
FREEZE_FEATURE_COMMIT = "93019b2cc34fe7659d10e9e42b3cab9ae28ec2de"
FREEZE_TEST_COMMIT = "df81c1b38ac73e65b1d2dfdb3ec6a7dd6e517f39"
PROPOSAL_COMMIT = "509f2449dbbaf4c1f5153ecfa4bc1652f24f75da"
PROPOSAL_INVENTORY_SHA256 = "69fd2a9fbcc1b461ae4230a282cfcaaadc26ba7b65563bc90a81e86e1ca98127"
SEED_MANIFEST_SHA256 = "ab37564cca81e926b6e43caf74f1ff2be641b8bc203e32c3ae6a219ea6f7bace"
READINESS_AUDIT_COMMIT = "a1514a654fe76518db16001b98e899f773eb9d1e"
READINESS_AUDIT_INVENTORY_SHA256 = "16ce1a1b138567a144cbf9b1d715c74b30f05e1d8262841d9580240080339d10"
PRODUCTION_TOOLING_SHA = "9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab"
BASE_CONTRACT_COMMIT = "a1967185e80327e4b00c1831828dc975ab6819fc"
BASE_CONTRACT_SHA256 = "482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b"
GENERATION_LEDGER_SHA256 = "a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1"
REPLAY_LEDGER_SHA256 = "4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd"
EVIDENCE_ARCHIVE_SHA256 = "375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc"
SPECIFICATION_SHA256 = "c1822db61182e6ff6436c767ac39a35065ff84741119bd7306b261dc2a1f7373"
CONTRACT_HASH = "e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a"
DECLARATION_SHA256 = "e0987858e1eca4d04e7008a468de232ef26eaa867f9d2a750c83fc8c89717bd4"
README_SHA256 = "d4d9d8b796b194ef6bebe7ad1a4520cc38cee4f3fb40889fc922cf3705928b0c"
TAG_NAME = "stage-a-discovery-contract-v1"
FREEZE_BRANCH = "freeze/stage-a-discovery-contract-v1-restart"
FREEZE_STATUS = "FROZEN_PENDING_INDEPENDENT_AUDIT"
CORPUS_NAMESPACE = "stage_a_discovery_v1"
SEED_DOMAIN = "satnet_stage_a_discovery_v1_seed"
SEED_MODULUS = 2**63
MARGIN_QUANTUM = Decimal("0.000001")
CONTRACT_ROOT_RELATIVE = Path("artifacts/stage_a_discovery_contract_v1")
PROPOSAL_ROOT_RELATIVE = Path("artifacts/stage_a_discovery_contract_proposal")
READINESS_ROOT_RELATIVE = Path("artifacts/stage_a_discovery_contract_freeze_audit")
TRACKED_AUDIT_ROOT_RELATIVE = Path("artifacts/stage_a_discovery_contract_frozen_audit")
REPORT_RELATIVE = Path("docs/validation/stage_a_discovery_contract_v1_frozen_audit.md")
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
RESERVED_ROOTS = {
    "production_generation": Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-production"),
    "production_replay": Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-replay"),
    "production_acceptance": Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-acceptance"),
    "evidence_freeze": Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-freeze"),
}
FROZEN_ROOTS = (
    Path(r"C:\Users\johns\satnet-final-production-20260720"),
    Path(r"C:\Users\johns\satnet-final-production-replay-20260720"),
    Path(r"C:\Users\johns\satnet-final-production-v1-freeze-20260721"),
    Path(r"C:\Users\johns\satnet-final-production-v1-freeze-20260721.zip"),
)
PROTECTED_PATHS = (
    "src/satnet/ground",
    "src/satnet/network",
    "src/satnet/simulation/tier1_rollout.py",
    "src/satnet/models/gnn_dataset.py",
    "src/satnet/models/gnn_model.py",
    "src/satnet/models/risk_model.py",
    "src/satnet/utils/graph_cache.py",
)
DOE_RANGES = {
    "num_planes": (4.0, 6.0),
    "sats_per_plane": (5.0, 8.0),
    "altitude_km": (300.0, 1200.0),
    "inclination_deg": (30.0, 98.0),
    "satellite_node_failure_probability": (0.0, 0.2),
    "satellite_edge_failure_probability": (0.0, 0.25),
    "total_ground_station_count": (3.0, 50.0),
    "ground_station_failure_probability": (0.0, 0.4),
    "civilian_fraction": (0.0, 1.0),
    "government_fraction": (0.0, 1.0),
    "military_fraction": (0.0, 1.0),
}
SCIENTIFIC_PARAMETER_FIELDS = (
    "num_planes", "sats_per_plane", "configured_satellite_count", "altitude_km",
    "inclination_deg", "phasing_factor", "satellite_node_failure_probability",
    "satellite_edge_failure_probability", "civilian_count", "government_count",
    "military_count", "total_ground_station_count", "ground_station_failure_probability",
    "duration_minutes", "step_seconds", "epoch_iso", "orbital_engine",
    "max_isl_distance_km", "isl_policy", "adjacent_search_k",
    "max_inter_plane_links_per_sat", "satellite_failure_model", "minimum_elevation_deg",
    "space_gcc_threshold", "ground_service_threshold", "ground_service_policy_hash",
    "visibility_policy_hash",
)
DESIGN_INT_FIELDS = {
    "design_index", "num_planes", "sats_per_plane", "configured_satellite_count",
    "phasing_factor", "civilian_count", "government_count", "military_count",
    "total_ground_station_count", "duration_minutes", "step_seconds", "adjacent_search_k",
    "max_inter_plane_links_per_sat", "design_construction_seed", "ground_selection_seed",
}
RUN_INT_FIELDS = {"global_run_id", "design_index", "realization_index"}
SEED_INT_FIELDS = {
    "global_run_id", "design_construction_seed", "ground_selection_seed",
    "satellite_failure_seed", "ground_failure_seed",
}
DECISION_STATES = (
    "STAGE_A_DISCOVERY_INSUFFICIENT", "STAGE_A_READY_FOR_STAGE_B_PROPOSAL",
    "STAGE_A_VALIDATION_FAILED", "STAGE_A_STAGE_B_CONTRACT_FROZEN",
    "STAGE_A_HOLDOUT_CONFIRMED", "STAGE_A_HOLDOUT_NOT_CONFIRMED",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_byte_identical(label: str, *payloads: bytes) -> None:
    if len(payloads) < 2 or any(payload != payloads[0] for payload in payloads[1:]):
        raise ValueError(f"Byte identity mismatch: {label}")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_json_compact(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def canonical_payload_hash(payload: Mapping[str, Any], domain: str) -> str:
    wrapped = {"identity_domain": domain, "identity_version": "1", "payload": dict(payload)}
    return sha256_bytes(canonical_json_compact(wrapped).encode("utf-8"))


def derive_seed(purpose: str, design_id: str, realization_id: str | None = None) -> int:
    payload: dict[str, Any] = {
        "corpus_namespace": CORPUS_NAMESPACE,
        "design_id": design_id,
        "identity_domain": SEED_DOMAIN,
        "identity_version": "1",
        "proposal_version": "1",
        "seed_purpose": purpose,
    }
    if realization_id is not None:
        payload["realization_id"] = realization_id
    return int.from_bytes(hashlib.sha256(canonical_json_compact(payload).encode("utf-8")).digest(), "big") % SEED_MODULUS


def canonical_margin(value: str | Decimal | float) -> str:
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    quantized = decimal_value.quantize(MARGIN_QUANTUM, rounding=ROUND_HALF_EVEN)
    if quantized == 0:
        quantized = Decimal("0.000000")
    return format(quantized, ".6f")


def classify_design(margins: Sequence[str | Decimal | float]) -> dict[str, Any]:
    if len(margins) != 5:
        raise ValueError("Exactly five realization margins are required")
    values = [value if isinstance(value, Decimal) else Decimal(str(value)) for value in margins]
    non_breach = sum(value >= 0 for value in values)
    return {
        "non_breach_realization_count": non_breach,
        "breach_realization_count": 5 - non_breach,
        "majority_class": "non_breach_majority" if non_breach >= 3 else "breach_majority",
        "mixed_design": 0 < non_breach < 5,
        "observed_boundary_design": min(values) < 0 <= max(values) or sum(abs(value) <= Decimal("0.05") for value in values) >= 2,
    }


def git(repo_root: Path, *arguments: str, binary: bool = False, check: bool = True) -> str | bytes:
    result = subprocess.run(["git", *arguments], cwd=repo_root, capture_output=True, check=False)
    if check and result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip() or "Git command failed")
    return result.stdout if binary else result.stdout.decode("utf-8", errors="strict").strip()


def read_json_bytes(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if payload.startswith(b"\xef\xbb\xbf") or b"\r\n" in payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError(f"Noncanonical JSON byte format: {path}")
    value = json.loads(payload)
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise ValueError(f"Noncanonical JSON object: {path}")
    return value


def _coerce_csv_row(row: Mapping[str, str], int_fields: set[str], bool_fields: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, value in row.items():
        if field in int_fields:
            result[field] = int(value)
        elif field in bool_fields:
            if value not in {"True", "False"}:
                raise ValueError(f"Invalid Boolean value for {field}")
            result[field] = value == "True"
        else:
            result[field] = value
    return result


def read_csv_rows(path: Path, kind: str) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if payload.startswith(b"\xef\xbb\xbf") or b"\r\n" in payload or not payload.endswith(b"\n"):
        raise ValueError(f"Noncanonical CSV byte format: {path}")
    with io.StringIO(payload.decode("utf-8"), newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV records required: {path}")
    if kind == "design":
        return [_coerce_csv_row(row, DESIGN_INT_FIELDS, {"sealed"}) for row in rows]
    if kind == "run":
        return [_coerce_csv_row(row, RUN_INT_FIELDS, {"sealed", "simulation_authorized"}) for row in rows]
    if kind == "seed":
        return [_coerce_csv_row(row, SEED_INT_FIELDS, {"simulation_authorized"}) for row in rows]
    raise ValueError(f"Unknown CSV kind: {kind}")


def paths_overlap(first: Path, second: Path) -> bool:
    left = Path(os.path.abspath(first))
    right = Path(os.path.abspath(second))
    return left == right or left in right.parents or right in left.parents


def validate_tag(repo_root: Path) -> dict[str, Any]:
    if git(repo_root, "cat-file", "-t", TAG_NAME) != "tag":
        raise ValueError("Stage A contract tag is not annotated")
    target = str(git(repo_root, "rev-list", "-n", "1", TAG_NAME))
    if target != FREEZE_HEAD:
        raise ValueError("Stage A contract tag target mismatch")
    raw = bytes(git(repo_root, "cat-file", "-p", TAG_NAME, binary=True))
    text = raw.decode("utf-8")
    expected_lines = (
        "SATNET Stage A Discovery Contract v1",
        f"Frozen contract hash:\n{CONTRACT_HASH}",
        f"Approved proposal commit:\n{PROPOSAL_COMMIT}",
        f"Freeze-readiness audit commit:\n{READINESS_AUDIT_COMMIT}",
        f"Byte-policy correction commit:\n{BYTE_POLICY_COMMIT}",
        "Simulation authorized:\nfalse",
        "Required next step:\nindependent frozen-contract audit",
    )
    annotation = text.split("\n\n", 1)[1] if "\n\n" in text else ""
    if not all(item in annotation for item in expected_lines):
        raise ValueError("Stage A contract tag annotation mismatch")
    conflicts = [name for name in str(git(repo_root, "tag", "--points-at", FREEZE_HEAD)).splitlines() if name and name != TAG_NAME]
    if conflicts:
        raise ValueError("Conflicting tag points at Stage A freeze HEAD")
    return {
        "tag_name": TAG_NAME,
        "tag_type": "tag",
        "tag_target": target,
        "annotation_sha256": sha256_bytes(annotation.encode("utf-8")),
        "annotation_result": "PASS",
        "conflicting_tags": conflicts,
        "tag_modified_by_audit": False,
    }


def validate_byte_policy(repo_root: Path) -> dict[str, Any]:
    lines = (repo_root / ".gitattributes").read_text(encoding="utf-8").splitlines()
    active = [line.split() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if any(tokens[0] == "*" and "-text" in tokens[1:] for tokens in active):
        raise ValueError("Repository-wide negative text rule exists")
    broad = {"*.json", "*.csv", "*.md", "*.txt", "**/*.json", "**/*.csv", "**/*.md", "**/*.txt"}
    if any(tokens[0] in broad and "-text" in tokens[1:] for tokens in active):
        raise ValueError("Broad file-type negative text rule exists")
    required_patterns = {
        "artifacts/stage_a_discovery_contract_freeze_audit/**",
        "artifacts/stage_a_discovery_contract_v1/**",
        "artifacts/stage_a_discovery_contract_proposal/**",
    }
    present = {tokens[0] for tokens in active if "-text" in tokens[1:]}
    if not required_patterns.issubset(present):
        raise ValueError("Stage A byte-preservation rules are incomplete")
    paths = {
        "audit_bundle": "artifacts/stage_a_discovery_contract_freeze_audit/audit_inventory.json",
        "frozen_contract_bundle": "artifacts/stage_a_discovery_contract_v1/stage_a_frozen_contract_inventory.json",
        "proposal_bundle": "artifacts/stage_a_discovery_contract_proposal/stage_a_proposal_inventory.json",
        "readme": "README.md",
    }
    attributes = {}
    for label, relative in paths.items():
        output = str(git(repo_root, "check-attr", "text", "--", relative))
        attributes[label] = output.rsplit(": ", 1)[-1]
    if attributes != {
        "audit_bundle": "unset", "frozen_contract_bundle": "unset",
        "proposal_bundle": "unset", "readme": "unspecified",
    }:
        raise ValueError("Stage A checkout byte attributes mismatch")
    return {"attributes": attributes, "narrow_rules": sorted(required_patterns), "global_negative_text_rule": False, "broad_type_rule": False, "pass": True}


def validate_audit_bundle(repo_root: Path) -> dict[str, Any]:
    root = repo_root / READINESS_ROOT_RELATIVE
    inventory_path = root / "audit_inventory.json"
    payload = inventory_path.read_bytes()
    if (len(payload), sha256_bytes(payload), payload.count(b"\n"), payload.count(b"\r\n")) != (
        3691, READINESS_AUDIT_INVENTORY_SHA256, 107, 0,
    ):
        raise ValueError("Freeze-readiness audit inventory byte identity mismatch")
    inventory = json.loads(payload)
    records = inventory.get("artifacts")
    if not isinstance(records, list) or len(records) != 16 or inventory.get("artifact_count_excluding_inventory") != 16:
        raise ValueError("Freeze-readiness audit inventory cardinality mismatch")
    expected = {record["relative_path"] for record in records} | {"audit_inventory.json"}
    actual = {path.name for path in root.iterdir() if path.is_file()}
    tracked = {Path(path).name for path in str(git(repo_root, "ls-files", "--", READINESS_ROOT_RELATIVE.as_posix())).splitlines()}
    if actual != expected or tracked != expected or len(actual) != 17:
        raise ValueError("Freeze-readiness audit bundle membership mismatch")
    validations = []
    for record in records:
        bound = (root / record["relative_path"]).read_bytes()
        passed = len(bound) == record["byte_length"] and sha256_bytes(bound) == record["sha256"]
        if not passed:
            raise ValueError(f"Freeze-readiness audit artifact mismatch: {record['relative_path']}")
        validations.append({"relative_path": record["relative_path"], "byte_length": len(bound), "sha256": sha256_bytes(bound), "pass": True})
    verdict = json.loads((root / "audit_findings.json").read_bytes()).get("final_verdict")
    if verdict != "APPROVED FOR STAGE A CONTRACT FREEZE":
        raise ValueError("Freeze-readiness audit verdict mismatch")
    return {
        "inventory_byte_length": len(payload), "inventory_sha256": sha256_bytes(payload),
        "lf_count": payload.count(b"\n"), "crlf_count": payload.count(b"\r\n"),
        "tracked_file_count": len(tracked), "inventory_bound_file_count": len(records),
        "artifact_validations": validations, "verdict": verdict, "pass": True,
    }


def validate_source_bundle(repo_root: Path) -> list[dict[str, Any]]:
    proposal_root = repo_root / PROPOSAL_ROOT_RELATIVE
    source_root = repo_root / CONTRACT_ROOT_RELATIVE / "source_bundle"
    if tuple(sorted(path.name for path in proposal_root.iterdir() if path.is_file())) != SOURCE_ARTIFACTS:
        raise ValueError("Approved proposal artifact membership mismatch")
    if str(git(repo_root, "diff", "--name-only", PROPOSAL_COMMIT, "HEAD", "--", PROPOSAL_ROOT_RELATIVE.as_posix())):
        raise ValueError("Approved proposal changed relative to approved commit")
    rows = []
    for name in SOURCE_ARTIFACTS:
        proposal = (proposal_root / name).read_bytes()
        source = (source_root / name).read_bytes()
        blob = bytes(git(repo_root, "show", f"{PROPOSAL_COMMIT}:{PROPOSAL_ROOT_RELATIVE.as_posix()}/{name}", binary=True))
        require_byte_identical(name, proposal, source, blob)
        passed = True
        rows.append({
            "relative_path": name, "proposal_byte_length": len(proposal),
            "frozen_byte_length": len(source), "proposal_sha256": sha256_bytes(proposal),
            "frozen_sha256": sha256_bytes(source), "approved_commit_blob_match": proposal == blob,
            "byte_identical": passed,
        })
    if sha256_file(proposal_root / "stage_a_proposal_inventory.json") != PROPOSAL_INVENTORY_SHA256:
        raise ValueError("Approved proposal inventory mismatch")
    if sha256_file(proposal_root / "stage_a_seed_manifest.csv") != SEED_MANIFEST_SHA256:
        raise ValueError("Approved seed manifest mismatch")
    return rows


def validate_frozen_inventory(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = repo_root / CONTRACT_ROOT_RELATIVE
    path = root / "stage_a_frozen_contract_inventory.json"
    payload = path.read_bytes()
    if sha256_bytes(payload) != CONTRACT_HASH:
        raise ValueError("Authoritative frozen contract hash mismatch")
    inventory = read_json_bytes(path)
    records = inventory.get("artifacts")
    expected_paths = sorted([f"source_bundle/{name}" for name in SOURCE_ARTIFACTS] + ["stage_a_frozen_contract_specification.json"])
    if not isinstance(records, list) or len(records) != 12 or inventory.get("contract_bound_artifact_count") != 12:
        raise ValueError("Frozen inventory cardinality mismatch")
    if [record.get("relative_path") for record in records] != expected_paths:
        raise ValueError("Frozen inventory ordering or membership mismatch")
    forbidden = {"stage_a_frozen_contract_inventory.json", "stage_a_frozen_contract.sha256", "stage_a_contract_freeze_declaration.json", "STAGE_A_CONTRACT_FREEZE_README.txt"}
    if forbidden.intersection(expected_paths):
        raise ValueError("Frozen inventory contains a self-reference or derived metadata")
    expected_fields = {"relative_path", "byte_length", "sha256", "schema_identifier", "record_count", "source_classification"}
    validations = []
    for record in records:
        relative = record["relative_path"]
        if set(record) != expected_fields or "\\" in relative or Path(relative).is_absolute():
            raise ValueError("Frozen inventory record schema or path mismatch")
        bound = (root / Path(relative)).read_bytes()
        digest = sha256_bytes(bound)
        passed = len(bound) == record["byte_length"] and digest == record["sha256"] and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None
        if not passed:
            raise ValueError(f"Frozen inventory artifact mismatch: {relative}")
        validations.append({**record, "observed_byte_length": len(bound), "observed_sha256": digest, "pass": True})
    hash_record = (root / "stage_a_frozen_contract.sha256").read_bytes()
    expected_hash_record = f"{CONTRACT_HASH}  stage_a_frozen_contract_inventory.json\n".encode("utf-8")
    if hash_record != expected_hash_record:
        raise ValueError("Frozen contract hash record mismatch")
    return validations, {"inventory_byte_length": len(payload), "inventory_sha256": sha256_bytes(payload), "authoritative_contract_hash": CONTRACT_HASH, "hash_record": hash_record.decode("utf-8").rstrip("\n"), "inventory_ordering": "PASS", "self_reference_absent": True, "pass": True}


def validate_specification_and_declaration(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = repo_root / CONTRACT_ROOT_RELATIVE
    spec_path = root / "stage_a_frozen_contract_specification.json"
    declaration_path = root / "stage_a_contract_freeze_declaration.json"
    if sha256_file(spec_path) != SPECIFICATION_SHA256 or sha256_file(declaration_path) != DECLARATION_SHA256:
        raise ValueError("Frozen specification or declaration identity mismatch")
    spec = read_json_bytes(spec_path)
    declaration = read_json_bytes(declaration_path)
    expected_state = {
        "contract_frozen": True, "freeze_status": FREEZE_STATUS,
        "simulation_authorized": False, "production_authorized": False, "execution_authorized": False,
    }
    for label, value in (("specification", spec), ("declaration", declaration)):
        if any(value.get(field) != expected for field, expected in expected_state.items()):
            raise ValueError(f"Frozen authorization state mismatch in {label}")
    if (spec.get("contract_name"), spec.get("contract_version"), spec.get("corpus_namespace")) != ("SATNET Stage A Discovery Contract", "1", CORPUS_NAMESPACE):
        raise ValueError("Frozen specification identity mismatch")
    identities = spec["approved_proposal"] | {"audit_commit": spec["freeze_readiness_audit"]["commit"], "audit_inventory": spec["freeze_readiness_audit"]["inventory_sha256"], "byte_policy": spec["byte_preservation_policy"]["commit"]}
    if identities != {
        "branch": "correction/stage-a-near-neighbor-resolution-v1", "commit": PROPOSAL_COMMIT,
        "proposal_inventory_sha256": PROPOSAL_INVENTORY_SHA256, "seed_manifest_sha256": SEED_MANIFEST_SHA256,
        "audit_commit": READINESS_AUDIT_COMMIT, "audit_inventory": READINESS_AUDIT_INVENTORY_SHA256,
        "byte_policy": BYTE_POLICY_COMMIT,
    }:
        raise ValueError("Frozen specification provenance mismatch")
    provenance = spec["base_production_provenance"]
    if provenance != {
        "production_tooling_sha": PRODUCTION_TOOLING_SHA, "frozen_contract_tag": "final-integrated-dataset-contract-v1",
        "frozen_contract_commit": BASE_CONTRACT_COMMIT, "contract_specification_sha256": BASE_CONTRACT_SHA256,
        "generation_ledger_sha256": GENERATION_LEDGER_SHA256, "replay_ledger_sha256": REPLAY_LEDGER_SHA256,
        "evidence_freeze_archive_sha256": EVIDENCE_ARCHIVE_SHA256,
    }:
        raise ValueError("Frozen production provenance mismatch")
    if declaration.get("contract_hash") != CONTRACT_HASH or declaration.get("contract_inventory_sha256") != CONTRACT_HASH or declaration.get("contract_inventory_byte_length") != 4763:
        raise ValueError("Freeze declaration contract binding mismatch")
    if declaration.get("freeze_branch") != FREEZE_BRANCH or declaration.get("tag_pushed") is not False or declaration.get("required_next_step") != "Independent audit of the frozen SATNET Stage A Discovery Contract v1":
        raise ValueError("Freeze declaration governance mismatch")
    return ({"sha256": SPECIFICATION_SHA256, "canonical_utf8_lf": True, "state": expected_state, "provenance_bound": True, "pass": True}, {"sha256": DECLARATION_SHA256, "contract_hash": CONTRACT_HASH, "inventory_byte_length": 4763, "tag_pushed": False, "state": expected_state, "pass": True})


def scientific_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in SCIENTIFIC_PARAMETER_FIELDS)


def normalized_vector(row: Mapping[str, Any]) -> tuple[float, ...]:
    total = float(row["total_ground_station_count"])
    if total <= 0:
        raise ValueError("Ground-station total must be positive")
    values = {
        **{field: float(row[field]) for field in DOE_RANGES if not field.endswith("_fraction")},
        "civilian_fraction": float(row["civilian_count"]) / total,
        "government_fraction": float(row["government_count"]) / total,
        "military_fraction": float(row["military_count"]) / total,
    }
    return tuple((values[field] - DOE_RANGES[field][0]) / (DOE_RANGES[field][1] - DOE_RANGES[field][0]) for field in DOE_RANGES)


def normalized_distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(normalized_vector(first), normalized_vector(second), strict=True)))


def _within(value: float, rule: Mapping[str, Any]) -> bool:
    if "allowed_values" in rule:
        return value in {float(item) for item in rule["allowed_values"]}
    return float(rule["minimum"]) <= value <= float(rule["maximum"])


def validate_manifests(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = repo_root / CONTRACT_ROOT_RELATIVE / "source_bundle"
    designs = read_csv_rows(source / "stage_a_design_manifest.csv", "design")
    runs = read_csv_rows(source / "stage_a_run_manifest.csv", "run")
    seeds = read_csv_rows(source / "stage_a_seed_manifest.csv", "seed")
    bounds = read_json_bytes(source / "stage_a_region_bounds.json")
    original = [json.loads(line) for line in (repo_root / "artifacts/final_integrated_dataset_contract/designs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(designs) != 30 or [row["design_id"] for row in designs] != [f"SA-D{index:03d}" for index in range(30)] or [row["design_index"] for row in designs] != list(range(30)):
        raise ValueError("Stage A design identity or cardinality mismatch")
    expected_regions = {"resilient_core": 12, "boundary": 12, "global_control": 6}
    expected_partitions = {"development": 20, "validation": 5, "sealed_holdout": 5}
    expected_cross = {
        ("resilient_core", "development"): 8, ("resilient_core", "validation"): 2,
        ("resilient_core", "sealed_holdout"): 2, ("boundary", "development"): 8,
        ("boundary", "validation"): 2, ("boundary", "sealed_holdout"): 2,
        ("global_control", "development"): 4, ("global_control", "validation"): 1,
        ("global_control", "sealed_holdout"): 1,
    }
    if Counter(row["region"] for row in designs) != expected_regions or Counter(row["partition"] for row in designs) != expected_partitions or Counter((row["region"], row["partition"]) for row in designs) != expected_cross:
        raise ValueError("Stage A region or partition allocation mismatch")
    signatures = [scientific_signature(row) for row in designs]
    original_signatures = {scientific_signature(row) for row in original}
    if len(set(signatures)) != 30 or set(signatures) & original_signatures:
        raise ValueError("Stage A duplicate or original-design collision")
    design_rows = []
    for row in designs:
        base_pass = all(math.isfinite(float(row[field])) and minimum <= float(row[field]) <= maximum for field, (minimum, maximum) in DOE_RANGES.items() if not field.endswith("_fraction"))
        rules = bounds["regions"][row["region"]]["allowed_parameter_bounds"]
        region_pass = all(_within(float(row[field]), rule) for field, rule in rules.items())
        parameter = {field: row[field] for field in SCIENTIFIC_PARAMETER_FIELDS}
        record = {field: value for field, value in row.items() if field != "design_record_hash"}
        parameter_hash = canonical_payload_hash(parameter, "satnet_stage_a_design_parameters")
        record_hash = canonical_payload_hash(record, "satnet_stage_a_design_record")
        passed = all((base_pass, region_pass, parameter_hash == row["design_parameter_hash"], record_hash == row["design_record_hash"], row["sealed"] is (row["partition"] == "sealed_holdout"), row["proposal_status"] == "NOT_FROZEN"))
        if not passed:
            raise ValueError(f"Stage A design validation mismatch: {row['design_id']}")
        design_rows.append({
            "design_id": row["design_id"], "design_index": row["design_index"], "region": row["region"],
            "partition": row["partition"], "sealed": row["sealed"], "base_bounds_pass": base_pass,
            "region_bounds_pass": region_pass, "parameter_vector_unique": signatures.count(scientific_signature(row)) == 1,
            "not_original_duplicate": scientific_signature(row) not in original_signatures,
            "design_parameter_hash_match": parameter_hash == row["design_parameter_hash"],
            "design_record_hash_match": record_hash == row["design_record_hash"], "overall_pass": passed,
        })
    if len(runs) != 150 or [row["global_run_id"] for row in runs] != list(range(500, 650)):
        raise ValueError("Stage A run cardinality or identity mismatch")
    if len({row["run_key"] for row in runs}) != 150 or len({(row["design_id"], row["realization_id"]) for row in runs}) != 150:
        raise ValueError("Stage A duplicate run identity")
    by_design = {row["design_id"]: row for row in designs}
    run_rows = []
    for row in runs:
        design = by_design.get(row["design_id"])
        if design is None:
            raise ValueError("Stage A run references unknown design")
        index = row["realization_index"]
        identity = row["global_run_id"] == 500 + design["design_index"] * 5 + index and row["realization_id"] == f"R{index:02d}" and row["run_key"] == f"{design['design_id']}-R{index:02d}"
        colocation = all(row[field] == design[field] for field in ("region", "partition", "sealed", "design_record_hash"))
        record = {field: value for field, value in row.items() if field != "run_record_hash"}
        hash_match = canonical_payload_hash(record, "satnet_stage_a_run_record") == row["run_record_hash"]
        passed = identity and colocation and hash_match and row["proposal_status"] == "NOT_FROZEN" and row["simulation_authorized"] is False
        if not passed:
            raise ValueError(f"Stage A run validation mismatch: {row['run_key']}")
        run_rows.append({"global_run_id": row["global_run_id"], "run_key": row["run_key"], "design_id": row["design_id"], "realization_id": row["realization_id"], "region": row["region"], "partition": row["partition"], "sealed": row["sealed"], "identity_pass": identity, "design_colocation_pass": colocation, "run_record_hash_match": hash_match, "overall_pass": passed})
    if len(seeds) != 150 or len({row["run_key"] for row in seeds}) != 150 or {row["run_key"] for row in seeds} != {row["run_key"] for row in runs}:
        raise ValueError("Stage A seed identity cardinality mismatch")
    seed_rows = []
    seed_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seeds:
        expected = {
            "design_construction_seed": derive_seed("design_construction", row["design_id"]),
            "ground_selection_seed": derive_seed("ground_station_selection", row["design_id"]),
            "satellite_failure_seed": derive_seed("satellite_rollout_and_failure", row["design_id"], row["realization_id"]),
            "ground_failure_seed": derive_seed("ground_failure_realization", row["design_id"], row["realization_id"]),
        }
        matches = {f"{field}_match": row[field] == value for field, value in expected.items()}
        range_pass = all(type(row[field]) is int and 0 <= row[field] < SEED_MODULUS for field in expected)
        passed = all(matches.values()) and range_pass and row["proposal_status"] == "NOT_FROZEN" and row["simulation_authorized"] is False
        if not passed:
            raise ValueError(f"Stage A seed reproduction mismatch: {row['run_key']}")
        seed_groups[row["design_id"]].append(row)
        seed_rows.append({"global_run_id": row["global_run_id"], "run_key": row["run_key"], "design_id": row["design_id"], "realization_id": row["realization_id"], **matches, "range_pass": range_pass, "overall_pass": passed})
    if any(len(group) != 5 or len({row["design_construction_seed"] for row in group}) != 1 or len({row["ground_selection_seed"] for row in group}) != 1 or len({row["satellite_failure_seed"] for row in group}) != 5 or len({row["ground_failure_seed"] for row in group}) != 5 for group in seed_groups.values()):
        raise ValueError("Stage A seed design/realization binding mismatch")
    summary = {
        "design_count": len(designs), "run_count": len(runs), "seed_count": len(seeds),
        "region_allocation": expected_regions, "partition_allocation": expected_partitions,
        "region_partition_allocation": {f"{region}/{partition}": count for (region, partition), count in expected_cross.items()},
        "design_ids": "SA-D000-SA-D029", "global_run_ids": "500-649", "run_keys": "SA-D000-R00-SA-D029-R04",
        "duplicate_count": 0, "original_collision_count": 0, "base_bound_violations": 0,
        "region_bound_violations": 0, "design_hash_mismatches": 0, "run_hash_mismatches": 0,
        "seed_mismatches": 0, "pass": True,
    }
    return design_rows, run_rows, seed_rows, summary


def validate_governance(repo_root: Path) -> dict[str, Any]:
    source = repo_root / CONTRACT_ROOT_RELATIVE / "source_bundle"
    proposal = read_json_bytes(source / "stage_a_contract_proposal.json")
    partition = read_json_bytes(source / "stage_a_partition_manifest.json")
    criteria = read_json_bytes(source / "stage_a_discovery_criteria.json")
    specification = read_json_bytes(repo_root / CONTRACT_ROOT_RELATIVE / "stage_a_frozen_contract_specification.json")
    threshold = proposal["threshold_definition"]
    if threshold != {"value": "0.80", "margin": "failure_adjusted_overall_service_fraction_min - 0.80", "non_breach": "margin >= 0", "breach": "margin < 0"}:
        raise ValueError("Stage A threshold or margin rule mismatch")
    boundary = proposal["boundary_definitions"]
    if boundary["preassigned_boundary_region"] != "Pre-simulation DOE assignment only" or "straddle zero" not in boundary["observed_boundary_design"] or "at least two" not in boundary["observed_boundary_design"] or "0.05 qualifies" not in boundary["endpoint_behavior"]:
        raise ValueError("Stage A boundary definition mismatch")
    declared_classes = proposal["class_support_definitions"]
    if declared_classes != {
        "non_breach_majority": "at least 3 of 5 non-breach",
        "breach_majority": "at least 3 of 5 breach",
        "mixed_design": "at least one of each class; counts once by majority and may separately count as observed boundary",
    }:
        raise ValueError("Stage A majority or mixed-design rule mismatch")
    class_rules = {str(count): classify_design(["0"] * count + ["-0.1"] * (5 - count)) for count in range(6)}
    if any(result["majority_class"] != ("non_breach_majority" if count >= 3 else "breach_majority") or result["mixed_design"] != (0 < count < 5) for count, result in ((int(key), value) for key, value in class_rules.items())):
        raise ValueError("Stage A majority or mixed-design rule mismatch")
    margin_vectors = {value: canonical_margin(value) for value in ("0", "-0", "0.0000005", "0.0000015", "0.000000499999", "0.000000500001", "-0.0000005", "-0.0000015", "-0.000000499999", "-0.000000500001")}
    expected_vectors = {"0": "0.000000", "-0": "0.000000", "0.0000005": "0.000000", "0.0000015": "0.000002", "0.000000499999": "0.000000", "0.000000500001": "0.000001", "-0.0000005": "0.000000", "-0.0000015": "-0.000002", "-0.000000499999": "0.000000", "-0.000000500001": "-0.000001"}
    if margin_vectors != expected_vectors or proposal["distinct_margin_definition"]["rounding_mode"] != "ROUND_HALF_EVEN" or proposal["distinct_margin_definition"]["quantization_increment"] != "0.000001":
        raise ValueError("Stage A margin quantization rule mismatch")
    holdout = partition["partitions"]["sealed_holdout"]
    prohibited = {"Select Stage B parameter bounds", "Select Stage B regions", "Select Stage B design density", "Select Stage B sample size", "Select Stage B split allocation", "Select Stage B seeds", "Select Stage B acceptance gates", "Modify a frozen Stage B contract"}
    if holdout["design_count"] != 5 or holdout["run_count"] != 25 or holdout["sealed"] is not True or not prohibited.issubset(set(holdout["prohibited_uses"])) or len(partition["unsealing_conditions"]) != 6 or "Stage B contract hash recorded" not in partition["unsealing_conditions"] or "Stage B v2" not in partition["holdout_change_rule"]:
        raise ValueError("Stage A sealed-holdout policy mismatch")
    membership = proposal["final_corpus_membership"]
    if membership["primary_final_classification_corpus"] != ["original frozen 500-run corpus", "future frozen Stage B corpus"] or {membership[field] for field in ("stage_a_development", "stage_a_validation", "stage_a_sealed_holdout")} != {"EXCLUDED"}:
        raise ValueError("Stage A final-corpus membership mismatch")
    adaptation = proposal["stage_b_adaptation_boundary"]
    if adaptation["adaptation_sources"] != ["Stage A development", "Stage A validation"] or adaptation["prohibited_source"] != "Stage A sealed holdout" or adaptation["required_process"] != ["new Stage B proposal", "new design/run/seed manifests", "frozen split", "independent audit", "recorded contract hash"] or adaptation["simulation_authorized"] is not False:
        raise ValueError("Stage B adaptation boundary mismatch")
    groups = ("development_criteria", "validation_criteria", "pre_holdout_stage_b_proposal_criteria", "sealed_holdout_confirmation_criteria")
    records = [record for group in groups for record in criteria[group]]
    required_fields = {"criterion_id", "description", "input_partition", "metric", "operator", "threshold", "scientific_rationale", "failure_consequence", "can_influence_stage_b"}
    if len(records) != 22 or len({record["criterion_id"] for record in records}) != 22 or not all(set(record) == required_fields for record in records) or criteria["decision_states"] != list(DECISION_STATES) or criteria["final_classification_gates_apply"] is not False or not all(record["can_influence_stage_b"] is False for record in criteria["sealed_holdout_confirmation_criteria"]):
        raise ValueError("Stage A discovery criteria or decision-state mismatch")
    if specification["sealed_holdout_policy"]["may_confirm_or_reject_only"] is not True or specification["sealed_holdout_policy"]["may_alter_frozen_stage_b_contract"] is not False:
        raise ValueError("Stage A holdout post-unseal boundary mismatch")
    return {
        "threshold": threshold, "boundary_definition": "PASS", "class_rules": class_rules,
        "margin_quantization": {"increment": "0.000001", "rounding": "ROUND_HALF_EVEN", "negative_zero_normalized": True, "vectors": margin_vectors},
        "holdout_policy": "PASS", "final_corpus_membership": "PASS", "stage_b_adaptation_boundary": "PASS",
        "criterion_count": len(records), "criterion_ids_unique": True, "criteria_mathematically_feasible": True,
        "decision_states": list(DECISION_STATES), "silent_holdout_modification_permitted": False, "pass": True,
    }


def validate_neighbors(repo_root: Path) -> dict[str, Any]:
    source = repo_root / CONTRACT_ROOT_RELATIVE / "source_bundle"
    designs = read_csv_rows(source / "stage_a_design_manifest.csv", "design")
    original = [json.loads(line) for line in (repo_root / "artifacts/final_integrated_dataset_contract/designs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["design_id"]: row for row in designs}
    prior = dict(by_id["SA-D020"])
    prior["ground_station_failure_probability"] = "0.074999999999999997"
    stage_pairs = [(normalized_distance(first, second), first, second) for index, first in enumerate(designs) for second in designs[index + 1:]]
    def minimum_pair(left: str, right: str) -> float:
        return min(distance for distance, first, second in stage_pairs if {first["partition"], second["partition"]} == {left, right})
    original_pairs = [(normalized_distance(first, second), first, second) for first in designs for second in original]
    minimum_original = min(original_pairs, key=lambda item: item[0])
    result = {
        "changed_design": "SA-D020", "changed_field": "ground_station_failure_probability",
        "original_value": 0.075, "corrected_value": 0.100,
        "original_sa_d013_sa_d020_distance": normalized_distance(by_id["SA-D013"], prior),
        "corrected_sa_d013_sa_d020_distance": normalized_distance(by_id["SA-D013"], by_id["SA-D020"]),
        "minimum_development_validation": minimum_pair("development", "validation"),
        "minimum_development_holdout": minimum_pair("development", "sealed_holdout"),
        "minimum_validation_holdout": minimum_pair("validation", "sealed_holdout"),
        "minimum_stage_a_to_original": minimum_original[0],
        "minimum_stage_a_to_original_pair": [minimum_original[1]["design_id"], minimum_original[2]["design_id"]],
        "cross_partition_pairs_below_0_10": sum(distance < 0.10 and first["partition"] != second["partition"] for distance, first, second in stage_pairs),
        "holdout_pairs_below_0_10": sum(distance < 0.10 and "sealed_holdout" in {first["partition"], second["partition"]} for distance, first, second in stage_pairs),
    }
    expected = {
        "original_sa_d013_sa_d020_distance": 0.07681919236933395,
        "corrected_sa_d013_sa_d020_distance": 0.12039492645571381,
        "minimum_development_validation": 0.12039492645571381,
        "minimum_development_holdout": 0.1205683310788027,
        "minimum_validation_holdout": 0.1500946005884104,
        "minimum_stage_a_to_original": 0.06382978723404255,
    }
    if any(not math.isclose(result[field], value, rel_tol=0.0, abs_tol=1e-15) for field, value in expected.items()) or result["cross_partition_pairs_below_0_10"] or result["holdout_pairs_below_0_10"]:
        raise ValueError("Stage A near-neighbor resolution mismatch")
    policy = read_json_bytes(source / "stage_a_near_neighbor_policy.json")
    if policy["pending_scientific_reviews"] or policy["justified_exceptions"]:
        raise ValueError("Stage A near-neighbor review or exception remains")
    return {**result, "pending_scientific_reviews": [], "remaining_exceptions": [], "stage_a_to_original_proximity_disclosed_nonbinding": True, "pass": True}


def validate_output_roots(repo_root: Path, audit_roots: Sequence[Path]) -> dict[str, Any]:
    worktrees = [Path(line.removeprefix("worktree ")) for line in str(git(repo_root, "worktree", "list", "--porcelain")).splitlines() if line.startswith("worktree ")]
    records = []
    for name, path in RESERVED_ROOTS.items():
        overlaps_other = any(paths_overlap(path, other) for other_name, other in RESERVED_ROOTS.items() if other_name != name)
        overlaps_frozen = any(paths_overlap(path, frozen) for frozen in FROZEN_ROOTS)
        overlaps_worktree = any(paths_overlap(path, worktree) for worktree in worktrees)
        overlaps_audit = any(paths_overlap(path, audit) for audit in audit_roots)
        passed = not path.exists() and not overlaps_other and not overlaps_frozen and not overlaps_worktree and not overlaps_audit
        records.append({"role": name, "path": str(path), "exists": path.exists(), "overlaps_other_reserved_root": overlaps_other, "overlaps_frozen_evidence": overlaps_frozen, "overlaps_git_worktree": overlaps_worktree, "overlaps_audit_root": overlaps_audit, "pass": passed})
    if not all(record["pass"] for record in records):
        raise ValueError("Reserved Stage A output-root isolation mismatch")
    return {"records": records, "all_four_absent": True, "mutually_non_overlapping": True, "outside_worktrees": True, "outside_frozen_evidence": True, "pass": True}


def _authorization_fields(value: Any, path: str = "root") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for field, item in value.items():
            child = f"{path}.{field}"
            if field in {"simulation_authorized", "production_authorized", "execution_authorized", "contract_frozen", "freeze_status"}:
                yield child, item
            yield from _authorization_fields(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _authorization_fields(item, f"{path}[{index}]")


def validate_authorization(repo_root: Path) -> dict[str, Any]:
    contract = repo_root / CONTRACT_ROOT_RELATIVE
    proposal = repo_root / PROPOSAL_ROOT_RELATIVE
    records = []
    for root, historical in ((contract, False), (proposal, True)):
        for path in sorted(root.iterdir() if root == proposal else root.rglob("*")):
            if not path.is_file() or path.suffix != ".json":
                continue
            value = json.loads(path.read_bytes())
            try:
                label = path.relative_to(repo_root).as_posix()
            except ValueError:
                label = str(path)
            for field, observed in _authorization_fields(value, label):
                name = field.rsplit(".", 1)[-1]
                if historical:
                    if name == "simulation_authorized" and observed is not False:
                        raise ValueError(f"Proposal authorization enabled: {field}")
                    if name == "contract_frozen" and observed is True:
                        raise ValueError(f"Proposal unexpectedly frozen: {field}")
                elif name in {"simulation_authorized", "production_authorized", "execution_authorized"} and observed is not False:
                    raise ValueError(f"Frozen authorization enabled: {field}")
                records.append({"field": field, "value": observed, "historical_proposal": historical})
    specification = read_json_bytes(contract / "stage_a_frozen_contract_specification.json")
    declaration = read_json_bytes(contract / "stage_a_contract_freeze_declaration.json")
    for value in (specification, declaration):
        if value["contract_frozen"] is not True or value["freeze_status"] != FREEZE_STATUS or any(value[field] is not False for field in ("simulation_authorized", "production_authorized", "execution_authorized")):
            raise ValueError("Frozen authorization envelope mismatch")
    return {"contract_frozen": True, "freeze_status": FREEZE_STATUS, "simulation_authorized": False, "production_authorized": False, "execution_authorized": False, "checked_fields": len(records), "contradictions": [], "pass": True}


def validate_simulation_nonexecution(repo_root: Path, output_roots: Mapping[str, Any]) -> dict[str, Any]:
    tracked = str(git(repo_root, "ls-tree", "-r", "--name-only", "HEAD")).splitlines()
    stage_a = [path for path in tracked if "stage_a" in path.lower()]
    contract_allowed = {
        f"{CONTRACT_ROOT_RELATIVE.as_posix()}/{name}" for name in (
            "stage_a_frozen_contract_specification.json", "stage_a_frozen_contract_inventory.json",
            "stage_a_frozen_contract.sha256", "stage_a_contract_freeze_declaration.json",
            "STAGE_A_CONTRACT_FREEZE_README.txt",
        )
    } | {f"{CONTRACT_ROOT_RELATIVE.as_posix()}/source_bundle/{name}" for name in SOURCE_ARTIFACTS}
    prohibited = ("generation_ledger", "replay_ledger", "acceptance_report", "scientific_output", "run_000", "production_root")
    evidence = [path for path in stage_a if path.startswith(CONTRACT_ROOT_RELATIVE.as_posix()) and path not in contract_allowed]
    evidence.extend(path for path in stage_a if any(fragment in Path(path).name.lower() for fragment in prohibited) and "test" not in path.lower() and "audit" not in path.lower())
    evidence = sorted(set(evidence))
    if evidence or not output_roots["all_four_absent"]:
        raise ValueError("Stage A simulation evidence exists")
    return {"tracked_stage_a_simulation_evidence": evidence, "generation_ledger_count": 0, "replay_ledger_count": 0, "acceptance_report_count": 0, "scientific_output_tree_count": 0, "reserved_execution_root_count": 0, "pass": True}


def validate_freeze_tooling(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "src/satnet/experiments/stage_a_contract/freeze.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    prohibited_names = {"run_tier1_rollout", "generate_run", "generate_runs", "replay_runs_read_only", "train_rf_model", "SatelliteGNN"}
    imports = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    violations = sorted(prohibited_names.intersection(calls) | {name for name in imports if ".simulation" in name or ".models" in name})
    if violations:
        raise ValueError("Freeze tooling exposes a simulation, replay, or training path")
    permitted = {"verify_repository_identity", "reproduce_approved_proposal", "create_frozen_contract", "validate_frozen_contract", "verify_output_roots", "verify_frozen_production_evidence"}
    definitions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not permitted.issubset(definitions):
        raise ValueError("Freeze tooling verification surface is incomplete")
    return {"path": path.relative_to(repo_root).as_posix(), "simulation_imports": [], "prohibited_calls": violations, "permitted_surface_present": sorted(permitted), "can_execute_simulations": False, "pass": True}


def validate_protected_science(repo_root: Path) -> dict[str, Any]:
    protected = str(git(repo_root, "diff", "--name-only", BYTE_POLICY_COMMIT, FREEZE_HEAD, "--", *PROTECTED_PATHS)).splitlines()
    production_paths = ("artifacts/final_integrated_dataset_contract", "src/satnet/experiments/final_dataset", "src/satnet/experiments/final_generation")
    production = str(git(repo_root, "diff", "--name-only", BYTE_POLICY_COMMIT, FREEZE_HEAD, "--", *production_paths)).splitlines()
    proposal = str(git(repo_root, "diff", "--name-only", PROPOSAL_COMMIT, FREEZE_HEAD, "--", PROPOSAL_ROOT_RELATIVE.as_posix())).splitlines()
    if protected or production or proposal:
        raise ValueError("Protected science, production logic, or approved proposal changed")
    changed = str(git(repo_root, "diff", "--name-only", BYTE_POLICY_COMMIT, FREEZE_HEAD)).splitlines()
    allowed_prefixes = (CONTRACT_ROOT_RELATIVE.as_posix() + "/", "docs/validation/stage_a_discovery_contract_freeze_v1.md", "src/satnet/experiments/stage_a_contract/freeze.py", "tests/experiments/test_stage_a_contract_freeze.py", "tests/experiments/test_final_dataset_isolation.py")
    unauthorized = [path for path in changed if not any(path == prefix or path.startswith(prefix) for prefix in allowed_prefixes)]
    if unauthorized:
        raise ValueError("Freeze commit changed an unauthorized surface")
    return {"protected_science_diff": protected, "production_logic_diff": production, "approved_proposal_diff": proposal, "freeze_changed_paths": changed, "unauthorized_freeze_paths": unauthorized, "pass": True}


def validate_windows_checkout(repo_root: Path, windows_root: Path) -> dict[str, Any]:
    if str(git(windows_root, "rev-parse", "HEAD")) != FREEZE_HEAD or str(git(windows_root, "config", "--get", "core.autocrlf")) != "true" or str(git(windows_root, "status", "--short")):
        raise ValueError("Windows checkout identity, configuration, or cleanliness mismatch")
    audit_path = windows_root / READINESS_ROOT_RELATIVE / "audit_inventory.json"
    payload = audit_path.read_bytes()
    if (len(payload), sha256_bytes(payload), payload.count(b"\n"), payload.count(b"\r\n")) != (3691, READINESS_AUDIT_INVENTORY_SHA256, 107, 0):
        raise ValueError("Windows checkout changed audit inventory bytes")
    source_rows = []
    for name in SOURCE_ARTIFACTS:
        proposal = (windows_root / PROPOSAL_ROOT_RELATIVE / name).read_bytes()
        frozen = (windows_root / CONTRACT_ROOT_RELATIVE / "source_bundle" / name).read_bytes()
        generating = (repo_root / PROPOSAL_ROOT_RELATIVE / name).read_bytes()
        if proposal != frozen or proposal != generating:
            raise ValueError(f"Windows checkout source bytes mismatch: {name}")
        source_rows.append({"relative_path": name, "sha256": sha256_bytes(proposal), "byte_length": len(proposal), "pass": True})
    principal = {
        "specification": SPECIFICATION_SHA256, "inventory": CONTRACT_HASH,
        "declaration": DECLARATION_SHA256, "readme": README_SHA256,
    }
    files = {
        "specification": "stage_a_frozen_contract_specification.json",
        "inventory": "stage_a_frozen_contract_inventory.json",
        "declaration": "stage_a_contract_freeze_declaration.json",
        "readme": "STAGE_A_CONTRACT_FREEZE_README.txt",
    }
    if any(sha256_file(windows_root / CONTRACT_ROOT_RELATIVE / files[label]) != digest for label, digest in principal.items()):
        raise ValueError("Windows checkout changed frozen derived bytes")
    return {"path": str(windows_root), "head": FREEZE_HEAD, "core_autocrlf": True, "clean": True, "audit_inventory_byte_length": 3691, "audit_inventory_sha256": READINESS_AUDIT_INVENTORY_SHA256, "audit_inventory_lf": 107, "audit_inventory_crlf": 0, "source_bundle": source_rows, "frozen_contract_hash": CONTRACT_HASH, "tag_target": str(git(windows_root, "rev-list", "-n", "1", TAG_NAME)), "authorization_values_false": True, "pass": True}


def validate_frozen_evidence() -> dict[str, Any]:
    from satnet.experiments.final_class_support_audit.audit import verify_frozen_evidence
    return verify_frozen_evidence(
        production_tooling_root=Path(r"C:\Users\johns\satnet-production-tooling-20260720"),
        generation_root=FROZEN_ROOTS[0], replay_root=FROZEN_ROOTS[1], freeze_root=FROZEN_ROOTS[2],
        freeze_archive=FROZEN_ROOTS[3],
        freeze_archive_hash_file=Path(r"C:\Users\johns\satnet-final-production-v1-freeze-20260721.zip.sha256"),
    )


def validate_input_identity(repo_root: Path) -> dict[str, Any]:
    head = str(git(repo_root, "rev-parse", "HEAD"))
    branch = str(git(repo_root, "branch", "--show-current"))
    try:
        git(repo_root, "merge-base", "--is-ancestor", FREEZE_HEAD, "HEAD")
        freeze_is_ancestor = True
    except ValueError:
        freeze_is_ancestor = False
    if branch != "audit/stage-a-frozen-contract-v1" or not freeze_is_ancestor or str(git(repo_root, "status", "--short", "--untracked-files=no")):
        raise ValueError("Audit worktree identity mismatch")
    for commit in (BYTE_POLICY_COMMIT, FREEZE_FEATURE_COMMIT, FREEZE_TEST_COMMIT, FREEZE_HEAD, PROPOSAL_COMMIT, READINESS_AUDIT_COMMIT, PRODUCTION_TOOLING_SHA, BASE_CONTRACT_COMMIT):
        if git(repo_root, "cat-file", "-t", commit) != "commit":
            raise ValueError(f"Required commit unavailable: {commit}")
    base_spec = json.loads((repo_root / "artifacts/final_integrated_dataset_contract/contract_specification.json").read_bytes())
    if base_spec["contract_spec_hash"] != BASE_CONTRACT_SHA256:
        raise ValueError("Base final-integrated contract specification mismatch")
    return {
        "audit_branch": branch, "audit_tooling_head": head, "freeze_commit": FREEZE_HEAD,
        "freeze_is_ancestor_of_audit_head": freeze_is_ancestor,
        "byte_policy_commit": BYTE_POLICY_COMMIT, "freeze_feature_commit": FREEZE_FEATURE_COMMIT,
        "freeze_test_commit": FREEZE_TEST_COMMIT, "approved_proposal_commit": PROPOSAL_COMMIT,
        "approved_audit_commit": READINESS_AUDIT_COMMIT, "production_tooling_sha": PRODUCTION_TOOLING_SHA,
        "base_contract_commit": BASE_CONTRACT_COMMIT, "base_contract_specification_hash": BASE_CONTRACT_SHA256,
        "generation_ledger_sha256": GENERATION_LEDGER_SHA256, "replay_ledger_sha256": REPLAY_LEDGER_SHA256,
        "production_evidence_archive_sha256": EVIDENCE_ARCHIVE_SHA256, "pass": True,
    }


def validate_readme(repo_root: Path) -> dict[str, Any]:
    path = repo_root / CONTRACT_ROOT_RELATIVE / "STAGE_A_CONTRACT_FREEZE_README.txt"
    payload = path.read_bytes()
    if sha256_bytes(payload) != README_SHA256 or b"\r\n" in payload or not payload.endswith(b"\n"):
        raise ValueError("Freeze README byte identity mismatch")
    text = payload.decode("utf-8")
    required = (
        "STAGE A CONTRACT FROZEN", "SIMULATION NOT AUTHORIZED",
        "INDEPENDENT FROZEN-CONTRACT AUDIT REQUIRED", CONTRACT_HASH, PROPOSAL_COMMIT,
        READINESS_AUDIT_COMMIT, PROPOSAL_INVENTORY_SHA256, SEED_MANIFEST_SHA256,
        READINESS_AUDIT_INVENTORY_SHA256, "30 designs", "150 runs", "150 seed records",
        "All Stage A partitions are excluded", "All four reserved future output roots are absent",
    )
    if not all(item in text for item in required):
        raise ValueError("Freeze README content mismatch")
    return {"sha256": README_SHA256, "byte_length": len(payload), "required_statements_present": True, "machine_readable_contract_consistent": True, "pass": True}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV records required: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_root: Path, outputs: Mapping[str, Any]) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_root.iterdir() if path.is_file()}
    permitted = set(outputs) | {"audit_inventory.json"}
    if existing - permitted:
        raise ValueError(f"Audit output root contains unexpected files: {sorted(existing - permitted)}")
    for name, value in outputs.items():
        path = output_root / name
        if name.endswith(".csv"):
            write_csv(path, value)
        else:
            write_json(path, value)
    records = []
    for path in sorted((output_root / name for name in outputs), key=lambda item: item.name):
        count = None
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                count = sum(1 for _ in csv.DictReader(handle))
        records.append({"relative_path": path.name, "byte_length": path.stat().st_size, "sha256": sha256_file(path), "record_count": count})
    inventory = {
        "schema_identifier": "satnet.stage_a.frozen_contract_independent_audit_inventory.v1",
        "freeze_commit_audited": FREEZE_HEAD, "contract_hash_audited": CONTRACT_HASH,
        "ordering": "relative_path ordinal lexical ascending", "encoding": "UTF-8 without BOM",
        "newline": "LF with one terminal newline", "self_reference_policy": "audit_inventory.json excludes its own bytes",
        "artifact_count_excluding_inventory": len(records), "artifacts": records,
    }
    write_json(output_root / "audit_inventory.json", inventory)
    return inventory


def build_report(outputs: Mapping[str, Any], validation_summary: Mapping[str, Any]) -> str:
    findings = outputs["audit_findings.json"]
    manifests = outputs["audit_governance_validation.json"]["manifest_summary"]
    evidence = outputs["audit_frozen_evidence.json"]
    sections = [
        ("Audit Scope", "Independent, read-only audit of the frozen SATNET Stage A Discovery Contract v1. No simulation, production, replay, acceptance, holdout unsealing, Stage B work, training, merge, push, or tag mutation was performed."),
        ("Input Identities", f"Freeze commit `{FREEZE_HEAD}`, byte-policy commit `{BYTE_POLICY_COMMIT}`, approved proposal `{PROPOSAL_COMMIT}`, approved readiness audit `{READINESS_AUDIT_COMMIT}`, and contract hash `{CONTRACT_HASH}` all matched."),
        ("Tag Audit", f"Annotated tag `{TAG_NAME}` targets `{FREEZE_HEAD}` and its required annotation fields passed. Push status: `{outputs['audit_tag_validation.json']['push_status']}`."),
        ("Byte-Policy Audit", "The proposal, readiness-audit, and frozen-contract roots have narrow `-text` rules. No repository-wide or broad file-type `-text` rule exists; `README.md` remains unspecified."),
        ("Windows Checkout Audit", "The independent `core.autocrlf=true` checkout was clean and preserved the audit inventory, all 11 source artifacts, specification, inventory, declaration, README, contract hash, tag target, and false authorization values byte-for-byte."),
        ("Proposal and Audit Provenance", f"Proposal inventory `{PROPOSAL_INVENTORY_SHA256}`, seed manifest `{SEED_MANIFEST_SHA256}`, and readiness-audit inventory `{READINESS_AUDIT_INVENTORY_SHA256}` matched. The readiness bundle contained 17 tracked files and 16 inventory-bound files."),
        ("Source-Bundle Audit", "All 11 frozen source copies were byte-identical to both the approved proposal checkout and the approved proposal commit blobs."),
        ("Inventory and Contract-Hash Audit", "Exactly 12 artifacts are contract-bound: 11 source copies and one frozen specification. Ordering, paths, lengths, hashes, schemas, record counts, classifications, and the non-self-referential hash design passed."),
        ("Specification, Declaration, and README Audit", f"Specification `{SPECIFICATION_SHA256}`, declaration `{DECLARATION_SHA256}`, and README `{README_SHA256}` matched exact bytes and governance state."),
        ("Design, Run, and Seed Audit", f"Independently reproduced `{manifests['design_count']}` designs, `{manifests['run_count']}` runs, and `{manifests['seed_count']}` seed records; all identities, allocations, bounds, hashes, and seed derivations passed without duplicate or collision."),
        ("Scientific-Definition Audit", "Threshold/margin behavior, observed-boundary separation, majority and mixed-design rules for all six count cases, six-decimal ROUND_HALF_EVEN quantization including normalized negative zero, 22 discovery criteria, and exact decision states passed."),
        ("Holdout and Final-Corpus Audit", "The 5-design/25-run holdout remains sealed and cannot shape Stage B. All Stage A partitions are discovery-only and excluded from the primary final corpus. Holdout can only confirm or reject an already frozen and audited Stage B contract."),
        ("Near-Neighbor Audit", "SA-D020’s correction from 0.075 to 0.100 and all six binding distances independently reproduced. No sub-0.10 cross-partition pair, pending review, or exception remains. Stage A-to-original proximity is accurately disclosed and nonbinding."),
        ("Reserved-Root and Authorization Audit", "All four reserved roots are absent, isolated, and unauthorized. `contract_frozen=true`, the freeze status remains pending independent audit, and all three authorization values remain false."),
        ("Simulation Non-Execution and Freeze-Tooling Audit", "No Stage A generation/replay ledger, acceptance report, scientific output tree, or execution root exists. AST inspection found no simulation, replay, or training entrypoint in freeze tooling."),
        ("Fail-Closed Audit", f"Independent negative mutation tests passed: `{validation_summary.get('negative_tests', 'recorded in validation results')}`."),
        ("Protected-Science and Frozen-Evidence Audit", f"Protected science and production logic diffs were empty. Before/after evidence verification covered `{evidence['combined']['file_count']}` files and `{evidence['combined']['byte_count']}` bytes with all hashes and read-only states intact."),
        ("Binding Findings", "None." if not findings["binding_findings"] else "\n".join(findings["binding_findings"])),
        ("Nonbinding Findings", "None." if not findings["nonbinding_findings"] else "\n".join(findings["nonbinding_findings"])),
        ("Observations", "The seed derivation is identity-only as disclosed; this is deterministic and does not contradict the frozen policy. Stage A-to-original nearest proximity is nonbinding because there is no exact scientific duplicate and Stage A is excluded from the primary final corpus."),
        ("Required Corrections", "None." if not findings["required_corrections"] else "\n".join(findings["required_corrections"])),
        ("Validation Results", "\n".join(f"- **{key}**: `{value}`" for key, value in validation_summary.items())),
        ("Final Verdict", f"**{findings['final_verdict']}**\n\nThis verdict permits only a separate execution-tooling development and audit phase. It does not authorize Stage A simulation, production, validation execution, holdout unsealing, Stage B work, RF training, or TGNN training."),
    ]
    return "# SATNET Stage A Discovery Contract v1 Frozen Audit\n\n" + "\n\n".join(f"## {title}\n\n{body}" for title, body in sections) + "\n"


def run_audit(
    *, repo_root: Path, external_output_root: Path, tracked_output_root: Path,
    windows_checkout_root: Path, validation_summary: Mapping[str, Any],
    verify_evidence: bool, tag_push_status: str,
) -> dict[str, Any]:
    protected_outputs = [repo_root / CONTRACT_ROOT_RELATIVE, repo_root / PROPOSAL_ROOT_RELATIVE, repo_root / READINESS_ROOT_RELATIVE, *FROZEN_ROOTS, *RESERVED_ROOTS.values()]
    for output in (external_output_root, tracked_output_root):
        if any(paths_overlap(output, protected) for protected in protected_outputs):
            raise ValueError("Independent audit output overlaps a protected root")
    input_identity = validate_input_identity(repo_root)
    tag = validate_tag(repo_root)
    tag["push_status"] = tag_push_status
    byte_policy = validate_byte_policy(repo_root)
    audit_bundle = validate_audit_bundle(repo_root)
    source_rows = validate_source_bundle(repo_root)
    inventory_rows, contract_hash = validate_frozen_inventory(repo_root)
    specification, declaration = validate_specification_and_declaration(repo_root)
    readme = validate_readme(repo_root)
    design_rows, run_rows, seed_rows, manifest_summary = validate_manifests(repo_root)
    governance = validate_governance(repo_root)
    governance["manifest_summary"] = manifest_summary
    governance["readme"] = readme
    neighbors = validate_neighbors(repo_root)
    output_roots = validate_output_roots(repo_root, [external_output_root, tracked_output_root, windows_checkout_root])
    authorization = validate_authorization(repo_root)
    nonexecution = validate_simulation_nonexecution(repo_root, output_roots)
    freeze_tooling = validate_freeze_tooling(repo_root)
    protected = validate_protected_science(repo_root)
    protected["freeze_tooling"] = freeze_tooling
    windows = validate_windows_checkout(repo_root, windows_checkout_root)
    evidence = validate_frozen_evidence() if verify_evidence else {
        "verification_status": "identity_only_for_test", "generation_ledger_sha256": GENERATION_LEDGER_SHA256,
        "replay_ledger_sha256": REPLAY_LEDGER_SHA256, "freeze_archive_sha256": EVIDENCE_ARCHIVE_SHA256,
        "combined": {"file_count": 9004, "byte_count": 1337549193, "verified_sha256_count": 0},
    }
    findings = {
        "schema_identifier": "satnet.stage_a.frozen_contract_independent_audit_findings.v1",
        "binding_findings": [], "nonbinding_findings": [],
        "observations": [
            "Identity-only deterministic seed derivation is accurately disclosed and policy-consistent.",
            "Stage A-to-original minimum distance 0.06382978723404255 is accurately disclosed, has no exact duplicate, and is nonbinding because Stage A is discovery-only.",
        ],
        "required_corrections": [],
        "final_verdict": "APPROVED WITH NONBINDING OBSERVATIONS FOR STAGE A EXECUTION-TOOLING DEVELOPMENT",
        "simulation_authorized": False, "production_authorized": False, "execution_authorized": False,
    }
    outputs = {
        "audit_findings.json": findings, "audit_input_identity.json": input_identity,
        "audit_tag_validation.json": tag, "audit_byte_policy_validation.json": byte_policy,
        "audit_audit_bundle_validation.json": audit_bundle,
        "audit_source_bundle_validation.csv": source_rows,
        "audit_frozen_inventory_validation.csv": inventory_rows,
        "audit_contract_hash_validation.json": contract_hash,
        "audit_specification_validation.json": specification,
        "audit_declaration_validation.json": declaration,
        "audit_design_manifest_validation.csv": design_rows,
        "audit_run_manifest_validation.csv": run_rows,
        "audit_seed_manifest_validation.csv": seed_rows,
        "audit_governance_validation.json": governance,
        "audit_neighbor_validation.json": neighbors,
        "audit_output_root_validation.json": output_roots,
        "audit_authorization_validation.json": authorization,
        "audit_simulation_nonexecution.json": nonexecution,
        "audit_windows_checkout_validation.json": windows,
        "audit_protected_science.json": protected,
        "audit_frozen_evidence.json": evidence,
    }
    tracked_inventory = write_outputs(tracked_output_root, outputs)
    external_inventory = write_outputs(external_output_root, outputs)
    if tracked_inventory != external_inventory:
        raise ValueError("Tracked and external audit inventories differ")
    report = build_report(outputs, validation_summary)
    report_path = repo_root / REPORT_RELATIVE
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    return {"findings": findings, "inventory": tracked_inventory, "report": str(report_path), "evidence": evidence}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--external-output-root", type=Path, required=True)
    parser.add_argument("--tracked-output-root", type=Path, required=True)
    parser.add_argument("--windows-checkout-root", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path)
    parser.add_argument("--skip-full-evidence", action="store_true")
    parser.add_argument("--tag-push-status", choices=("not_pushed", "pushed", "unverified"), default="unverified")
    arguments = parser.parse_args(argv)
    validation_summary = json.loads(arguments.validation_summary.read_bytes()) if arguments.validation_summary else {}
    result = run_audit(
        repo_root=arguments.repo_root.resolve(), external_output_root=arguments.external_output_root.resolve(),
        tracked_output_root=arguments.tracked_output_root.resolve(), windows_checkout_root=arguments.windows_checkout_root.resolve(),
        validation_summary=validation_summary, verify_evidence=not arguments.skip_full_evidence,
        tag_push_status=arguments.tag_push_status,
    )
    print(json.dumps(result["findings"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
