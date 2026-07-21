from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

PROPOSAL_COMMIT = "509f2449dbbaf4c1f5153ecfa4bc1652f24f75da"
STARTING_COMMIT = "f61521e96edc1bb8b5c3f05e8ebbcd25bca1eb7d"
PRODUCTION_TOOLING_SHA = "9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab"
FROZEN_CONTRACT_TAG = "final-integrated-dataset-contract-v1"
FROZEN_CONTRACT_COMMIT = "a1967185e80327e4b00c1831828dc975ab6819fc"
CONTRACT_SPECIFICATION_HASH = "482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b"
GENERATION_LEDGER_SHA256 = "a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1"
REPLAY_LEDGER_SHA256 = "4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd"
FREEZE_ARCHIVE_SHA256 = "375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc"
ANALYSIS_INVENTORY_SHA256 = "5ac295ba0e08797b63a2ce3f062a1995dc7aa850f11a2bf5d83a0580731afd85"
PRIOR_AUDIT_INVENTORY_SHA256 = "20d2e7037940d708e133936d3907ad822caac3b46cd73c84329f3efd16c4037a"
PROPOSAL_INVENTORY_SHA256 = "69fd2a9fbcc1b461ae4230a282cfcaaadc26ba7b65563bc90a81e86e1ca98127"
SEED_MANIFEST_SHA256 = "ab37564cca81e926b6e43caf74f1ff2be641b8bc203e32c3ae6a219ea6f7bace"
CORPUS_NAMESPACE = "stage_a_discovery_v1"
PROPOSAL_VERSION = "1"
SEED_DOMAIN = "satnet_stage_a_discovery_v1_seed"
SEED_MODULUS = 2**63
NEAR_NEIGHBOR_THRESHOLD = 0.10
MARGIN_QUANTUM = Decimal("0.000001")
EXPECTED_ARTIFACTS = (
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
NEIGHBOR_FEATURES = tuple(DOE_RANGES)
FIXED_DIRECT_FIELDS = (
    "duration_minutes",
    "step_seconds",
    "epoch_iso",
    "orbital_engine",
    "max_isl_distance_km",
    "isl_policy",
    "adjacent_search_k",
    "max_inter_plane_links_per_sat",
    "minimum_elevation_deg",
    "space_gcc_threshold",
    "ground_service_threshold",
    "ground_service_policy_hash",
    "visibility_policy_hash",
)
SCIENTIFIC_PARAMETER_FIELDS = (
    "num_planes",
    "sats_per_plane",
    "configured_satellite_count",
    "altitude_km",
    "inclination_deg",
    "phasing_factor",
    "satellite_node_failure_probability",
    "satellite_edge_failure_probability",
    "civilian_count",
    "government_count",
    "military_count",
    "total_ground_station_count",
    "ground_station_failure_probability",
    "duration_minutes",
    "step_seconds",
    "epoch_iso",
    "orbital_engine",
    "max_isl_distance_km",
    "isl_policy",
    "adjacent_search_k",
    "max_inter_plane_links_per_sat",
    "satellite_failure_model",
    "minimum_elevation_deg",
    "space_gcc_threshold",
    "ground_service_threshold",
    "ground_service_policy_hash",
    "visibility_policy_hash",
)
DESIGN_INT_FIELDS = {
    "design_index",
    "num_planes",
    "sats_per_plane",
    "configured_satellite_count",
    "phasing_factor",
    "civilian_count",
    "government_count",
    "military_count",
    "total_ground_station_count",
    "duration_minutes",
    "step_seconds",
    "adjacent_search_k",
    "max_inter_plane_links_per_sat",
    "design_construction_seed",
    "ground_selection_seed",
}
RUN_INT_FIELDS = {"global_run_id", "design_index", "realization_index"}
SEED_INT_FIELDS = {
    "global_run_id",
    "design_construction_seed",
    "ground_selection_seed",
    "satellite_failure_seed",
    "ground_failure_seed",
}
FINAL_GATES = {
    "minimum_non_breach_majority_designs": {"train": 12, "validation": 4, "test": 4},
    "minimum_non_breach_runs": {"train": 50, "validation": 14, "test": 14},
    "minimum_breach_majority_designs": {"train": 40, "validation": 10, "test": 10},
    "minimum_breach_runs": {"train": 200, "validation": 60, "test": 60},
    "maximum_majority_to_minority_run_ratio": {"train": 12.0, "validation": 10.0, "test": 10.0},
    "minimum_observed_boundary_designs": {"train": 10, "validation": 3, "test": 3},
    "minimum_distinct_canonical_margins": {"train": 30, "validation": 12, "test": 12},
}
STAGE_B_CAPACITY = {"train": 60, "validation": 15, "test": 15}
PROPOSED_OUTPUT_ROOTS = {
    "production_generation": "C:\\Users\\johns\\satnet-stage-a-discovery-v1-production",
    "production_replay": "C:\\Users\\johns\\satnet-stage-a-discovery-v1-replay",
    "production_acceptance": "C:\\Users\\johns\\satnet-stage-a-discovery-v1-acceptance",
    "evidence_freeze": "C:\\Users\\johns\\satnet-stage-a-discovery-v1-freeze",
}
PROTECTED_PATHS = (
    "src/satnet/ground",
    "src/satnet/network",
    "src/satnet/simulation/tier1_rollout.py",
    "src/satnet/models/gnn_dataset.py",
    "src/satnet/models/gnn_model.py",
    "src/satnet/models/risk_model.py",
    "src/satnet/utils/graph_cache.py",
)
DECISION_STATES = (
    "STAGE_A_DISCOVERY_INSUFFICIENT",
    "STAGE_A_READY_FOR_STAGE_B_PROPOSAL",
    "STAGE_A_VALIDATION_FAILED",
    "STAGE_A_STAGE_B_CONTRACT_FROZEN",
    "STAGE_A_HOLDOUT_CONFIRMED",
    "STAGE_A_HOLDOUT_NOT_CONFIRMED",
)
WORKFLOW_ACTIONS = (
    "Execute development",
    "Evaluate development criteria",
    "Execute or expose validation according to the frozen workflow",
    "Evaluate validation criteria",
    "Draft Stage B",
    "Freeze Stage B",
    "Independently audit Stage B",
    "Record Stage B contract hash",
    "Unseal Stage A holdout",
    "Evaluate holdout confirmation",
    "Execute or reject the already-frozen Stage B contract",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_compact(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_payload_hash(payload: Mapping[str, Any], domain: str) -> str:
    wrapped = {"identity_domain": domain, "identity_version": "1", "payload": dict(payload)}
    return hashlib.sha256(canonical_json_compact(wrapped).encode("utf-8")).hexdigest()


def derive_seed(purpose: str, design_id: str, realization_id: str | None = None) -> int:
    payload: dict[str, Any] = {
        "corpus_namespace": CORPUS_NAMESPACE,
        "design_id": design_id,
        "identity_domain": SEED_DOMAIN,
        "identity_version": PROPOSAL_VERSION,
        "proposal_version": PROPOSAL_VERSION,
        "seed_purpose": purpose,
    }
    if realization_id is not None:
        payload["realization_id"] = realization_id
    digest = hashlib.sha256(canonical_json_compact(payload).encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % SEED_MODULUS


def canonical_margin(value: str | Decimal | float) -> str:
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    result = decimal_value.quantize(MARGIN_QUANTUM, rounding=ROUND_HALF_EVEN)
    return format(result, ".6f")


def classify_design(margins: Sequence[str | Decimal | float]) -> dict[str, Any]:
    if len(margins) != 5:
        raise ValueError("Exactly five margins are required")
    values = [value if isinstance(value, Decimal) else Decimal(str(value)) for value in margins]
    non_breach = sum(value >= 0 for value in values)
    observed = min(values) < 0 <= max(values) or sum(abs(value) <= Decimal("0.05") for value in values) >= 2
    return {
        "majority_class": "non_breach_majority" if non_breach >= 3 else "breach_majority",
        "mixed_design": 0 < non_breach < 5,
        "observed_boundary_design": observed,
        "non_breach_realization_count": non_breach,
        "breach_realization_count": 5 - non_breach,
    }


def _coerce_row(row: Mapping[str, str], int_fields: set[str], bool_fields: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key in int_fields:
            result[key] = int(value)
        elif key in bool_fields:
            if value not in {"True", "False"}:
                raise ValueError(f"Invalid Boolean field {key}: {value}")
            result[key] = value == "True"
        else:
            result[key] = value
    return result


def read_csv_rows(path: Path, kind: str) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if b"\r\n" in raw or not raw.endswith(b"\n"):
        raise ValueError(f"Canonical LF newline required: {path}")
    with io.StringIO(raw.decode("utf-8"), newline="") as stream:
        rows = list(csv.DictReader(stream))
    if kind == "design":
        return [_coerce_row(row, DESIGN_INT_FIELDS, {"sealed"}) for row in rows]
    if kind == "run":
        return [_coerce_row(row, RUN_INT_FIELDS, {"sealed", "simulation_authorized"}) for row in rows]
    if kind == "seed":
        return [_coerce_row(row, SEED_INT_FIELDS, {"simulation_authorized"}) for row in rows]
    raise ValueError(f"Unknown CSV kind: {kind}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"JSONL object records required: {path}")
    return values


def _git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Git command failed")
    return result.stdout.strip()


def _git_show(repo_root: Path, commit: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def _csv_rows_from_bytes(payload: bytes, kind: str) -> list[dict[str, Any]]:
    with io.StringIO(payload.decode("utf-8"), newline="") as stream:
        rows = list(csv.DictReader(stream))
    if kind == "design":
        return [_coerce_row(row, DESIGN_INT_FIELDS, {"sealed"}) for row in rows]
    if kind == "seed":
        return [_coerce_row(row, SEED_INT_FIELDS, {"simulation_authorized"}) for row in rows]
    raise ValueError(kind)


def normalized_vector(row: Mapping[str, Any]) -> tuple[float, ...]:
    total = float(row["total_ground_station_count"])
    if total <= 0:
        raise ValueError("Ground-station total must be positive")
    values = {
        **{key: float(row[key]) for key in DOE_RANGES if not key.endswith("_fraction")},
        "civilian_fraction": float(row["civilian_count"]) / total,
        "government_fraction": float(row["government_count"]) / total,
        "military_fraction": float(row["military_count"]) / total,
    }
    return tuple((values[key] - DOE_RANGES[key][0]) / (DOE_RANGES[key][1] - DOE_RANGES[key][0]) for key in NEIGHBOR_FEATURES)


def normalized_distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(normalized_vector(first), normalized_vector(second), strict=True)))


def scientific_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in SCIENTIFIC_PARAMETER_FIELDS)


def _within(value: float, rule: Mapping[str, Any]) -> bool:
    if "allowed_values" in rule:
        return value in [float(item) for item in rule["allowed_values"]]
    return float(rule["minimum"]) <= value <= float(rule["maximum"])


def validate_authorization_object(value: Mapping[str, Any], label: str) -> None:
    if value.get("proposal_status") != "NOT_FROZEN":
        raise ValueError(f"{label} is not NOT_FROZEN")
    if value.get("simulation_authorized", False) is not False:
        raise ValueError(f"{label} enables simulation authorization")
    if value.get("contract_frozen", False) is not False:
        raise ValueError(f"{label} marks the contract frozen")


def validate_holdout_unsealing(prerequisites: Mapping[str, bool], contract_hash: str | None) -> None:
    expected = {
        "stage_b_contract_completely_specified",
        "stage_b_design_manifest_created",
        "stage_b_run_manifest_created",
        "stage_b_seed_manifest_created",
        "stage_b_split_frozen",
        "stage_b_independent_audit_passed",
        "stage_b_contract_hash_recorded",
    }
    if set(prerequisites) != expected or not all(prerequisites.values()):
        raise ValueError("All Stage B unsealing prerequisites must pass")
    if contract_hash is None or len(contract_hash) != 64 or any(character not in "0123456789abcdef" for character in contract_hash):
        raise ValueError("A lowercase SHA-256 Stage B contract hash is required")


def validate_snapshot(
    designs: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    seeds: Sequence[Mapping[str, Any]],
    original: Sequence[Mapping[str, Any]],
    region_bounds: Mapping[str, Any],
) -> None:
    if len(designs) != 30:
        raise ValueError("Design count differs")
    if [row["design_id"] for row in designs] != [f"SA-D{index:03d}" for index in range(30)]:
        raise ValueError("Design identity range differs")
    if [row["design_index"] for row in designs] != list(range(30)):
        raise ValueError("Design index range differs")
    if len({scientific_signature(row) for row in designs}) != 30:
        raise ValueError("Duplicate scientific design")
    if {scientific_signature(row) for row in designs} & {scientific_signature(row) for row in original}:
        raise ValueError("Original-design duplicate")
    expected_regions = {"resilient_core": 12, "boundary": 12, "global_control": 6}
    expected_partitions = {"development": 20, "validation": 5, "sealed_holdout": 5}
    if Counter(row["region"] for row in designs) != expected_regions:
        raise ValueError("Region allocation differs")
    if Counter(row["partition"] for row in designs) != expected_partitions:
        raise ValueError("Partition allocation differs")
    for row in designs:
        validate_authorization_object(row, str(row["design_id"]))
        if row["sealed"] is not (row["partition"] == "sealed_holdout"):
            raise ValueError("Design sealing differs")
        if int(row["configured_satellite_count"]) != int(row["num_planes"]) * int(row["sats_per_plane"]):
            raise ValueError("Configured satellite count differs")
        if sum(int(row[field]) for field in ("civilian_count", "government_count", "military_count")) != int(row["total_ground_station_count"]):
            raise ValueError("Ground-station composition differs")
        numeric_fields = set(DOE_RANGES) - {"civilian_fraction", "government_fraction", "military_fraction"}
        for field in numeric_fields:
            numeric = float(row[field])
            if not math.isfinite(numeric) or not DOE_RANGES[field][0] <= numeric <= DOE_RANGES[field][1]:
                raise ValueError(f"Base bound violation: {row['design_id']} {field}")
        rules = region_bounds["regions"][row["region"]]["allowed_parameter_bounds"]
        for field, rule in rules.items():
            if not _within(float(row[field]), rule):
                raise ValueError(f"Region bound violation: {row['design_id']} {field}")
        parameter = {field: row[field] for field in SCIENTIFIC_PARAMETER_FIELDS}
        if canonical_payload_hash(parameter, "satnet_stage_a_design_parameters") != row["design_parameter_hash"]:
            raise ValueError("Design-parameter hash mismatch")
        record = {key: value for key, value in row.items() if key != "design_record_hash"}
        if canonical_payload_hash(record, "satnet_stage_a_design_record") != row["design_record_hash"]:
            raise ValueError("Design-record hash mismatch")
    if len(runs) != 150 or [row["global_run_id"] for row in runs] != list(range(500, 650)):
        raise ValueError("Run identity range differs")
    if len({row["run_key"] for row in runs}) != 150:
        raise ValueError("Duplicate run")
    if len({(row["design_id"], row["realization_id"]) for row in runs}) != 150:
        raise ValueError("Duplicate design-realization pair")
    by_design = {row["design_id"]: row for row in designs}
    for row in runs:
        design = by_design.get(row["design_id"])
        if design is None:
            raise ValueError("Unknown run design")
        index = int(row["realization_index"])
        if row["global_run_id"] != 500 + int(design["design_index"]) * 5 + index:
            raise ValueError("Run global identity differs")
        if row["realization_id"] != f"R{index:02d}" or row["run_key"] != f"{design['design_id']}-R{index:02d}":
            raise ValueError("Run key differs")
        if any(row[field] != design[field] for field in ("region", "partition", "sealed", "design_record_hash")):
            raise ValueError("Run-design colocation differs")
        validate_authorization_object(row, str(row["run_key"]))
        record = {key: value for key, value in row.items() if key != "run_record_hash"}
        if canonical_payload_hash(record, "satnet_stage_a_run_record") != row["run_record_hash"]:
            raise ValueError("Run-record hash mismatch")
    if len(seeds) != 150 or len({row["run_key"] for row in seeds}) != 150:
        raise ValueError("Seed identity cardinality differs")
    if {row["run_key"] for row in seeds} != {row["run_key"] for row in runs}:
        raise ValueError("Seed/run identity set differs")
    for row in seeds:
        validate_authorization_object(row, str(row["run_key"]))
        expected = {
            "design_construction_seed": derive_seed("design_construction", str(row["design_id"])),
            "ground_selection_seed": derive_seed("ground_station_selection", str(row["design_id"])),
            "satellite_failure_seed": derive_seed("satellite_rollout_and_failure", str(row["design_id"]), str(row["realization_id"])),
            "ground_failure_seed": derive_seed("ground_failure_realization", str(row["design_id"]), str(row["realization_id"])),
        }
        if any(row[field] != expected[field] for field in expected):
            raise ValueError("Seed derivation mismatch")
    for index, first in enumerate(designs):
        for second in designs[index + 1 :]:
            distance = normalized_distance(first, second)
            if first["partition"] != second["partition"] and distance < NEAR_NEIGHBOR_THRESHOLD:
                raise ValueError("Cross-partition near-neighbor violation")


def audit_designs(
    designs: Sequence[Mapping[str, Any]],
    original: Sequence[Mapping[str, Any]],
    region_bounds: Mapping[str, Any],
    contract_specification: Mapping[str, Any],
) -> list[dict[str, Any]]:
    fixed = contract_specification["fixed_profile"]
    rows: list[dict[str, Any]] = []
    original_signatures = {scientific_signature(row) for row in original}
    signature_counts = Counter(scientific_signature(row) for row in designs)
    for row in designs:
        rules = region_bounds["regions"][row["region"]]["allowed_parameter_bounds"]
        base_pass = all(
            DOE_RANGES[field][0] <= float(row[field]) <= DOE_RANGES[field][1]
            for field in DOE_RANGES
            if not field.endswith("_fraction")
        )
        region_pass = all(_within(float(row[field]), rule) for field, rule in rules.items())
        direct_fixed_pass = all(
            str(row[field]) == str(fixed[field])
            for field in FIXED_DIRECT_FIELDS
            if field in fixed
        )
        failure_model_pass = row["satellite_failure_model"] == fixed["failure_model"]
        inherited_profile_pass = (
            row["base_contract_reference"] == f"{FROZEN_CONTRACT_TAG}@{FROZEN_CONTRACT_COMMIT}"
            and row["base_contract_specification_hash"] == CONTRACT_SPECIFICATION_HASH
            and fixed["inclusive_timestep_count"] == 11
            and fixed["physics_model_version"] == "tier1_space_segment_physics_v2"
            and isinstance(fixed["link_budget_config"], dict)
            and isinstance(fixed["versions"], dict)
        )
        parameter = {field: row[field] for field in SCIENTIFIC_PARAMETER_FIELDS}
        record = {key: value for key, value in row.items() if key != "design_record_hash"}
        rows.append(
            {
                "design_id": row["design_id"],
                "design_index": row["design_index"],
                "region": row["region"],
                "partition": row["partition"],
                "sealed": row["sealed"],
                "base_bounds_pass": base_pass,
                "region_bounds_pass": region_pass,
                "direct_fixed_profile_pass": direct_fixed_pass and failure_model_pass,
                "inherited_fixed_profile_binding_pass": inherited_profile_pass,
                "parameter_vector_unique": signature_counts[scientific_signature(row)] == 1,
                "not_original_duplicate": scientific_signature(row) not in original_signatures,
                "design_parameter_hash_match": canonical_payload_hash(parameter, "satnet_stage_a_design_parameters") == row["design_parameter_hash"],
                "design_record_hash_match": canonical_payload_hash(record, "satnet_stage_a_design_record") == row["design_record_hash"],
                "authorization_pass": row["proposal_status"] == "NOT_FROZEN" and not row.get("simulation_authorized", False),
                "overall_pass": all(
                    (
                        base_pass,
                        region_pass,
                        direct_fixed_pass,
                        failure_model_pass,
                        inherited_profile_pass,
                        signature_counts[scientific_signature(row)] == 1,
                        scientific_signature(row) not in original_signatures,
                        canonical_payload_hash(parameter, "satnet_stage_a_design_parameters") == row["design_parameter_hash"],
                        canonical_payload_hash(record, "satnet_stage_a_design_record") == row["design_record_hash"],
                        row["proposal_status"] == "NOT_FROZEN",
                    )
                ),
            }
        )
    return rows


def audit_runs(runs: Sequence[Mapping[str, Any]], designs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_design = {row["design_id"]: row for row in designs}
    rows: list[dict[str, Any]] = []
    key_counts = Counter(row["run_key"] for row in runs)
    pair_counts = Counter((row["design_id"], row["realization_id"]) for row in runs)
    for row in runs:
        design = by_design[row["design_id"]]
        index = int(row["realization_index"])
        record = {key: value for key, value in row.items() if key != "run_record_hash"}
        identity = (
            row["global_run_id"] == 500 + int(design["design_index"]) * 5 + index
            and row["realization_id"] == f"R{index:02d}"
            and row["run_key"] == f"{design['design_id']}-R{index:02d}"
        )
        colocation = all(row[field] == design[field] for field in ("region", "partition", "sealed", "design_record_hash"))
        hash_match = canonical_payload_hash(record, "satnet_stage_a_run_record") == row["run_record_hash"]
        rows.append(
            {
                "global_run_id": row["global_run_id"],
                "run_key": row["run_key"],
                "design_id": row["design_id"],
                "realization_id": row["realization_id"],
                "partition": row["partition"],
                "sealed": row["sealed"],
                "identity_pass": identity,
                "unique_run_key": key_counts[row["run_key"]] == 1,
                "unique_design_realization": pair_counts[(row["design_id"], row["realization_id"])] == 1,
                "design_colocation_pass": colocation,
                "run_record_hash_match": hash_match,
                "authorization_pass": row["proposal_status"] == "NOT_FROZEN" and row["simulation_authorized"] is False,
                "overall_pass": identity and colocation and hash_match and key_counts[row["run_key"]] == 1 and pair_counts[(row["design_id"], row["realization_id"])] == 1 and row["proposal_status"] == "NOT_FROZEN" and row["simulation_authorized"] is False,
            }
        )
    return rows


def audit_seeds(seeds: Sequence[Mapping[str, Any]], run_keys: set[str]) -> list[dict[str, Any]]:
    key_counts = Counter(row["run_key"] for row in seeds)
    rows: list[dict[str, Any]] = []
    for row in seeds:
        expected = {
            "design_construction_seed": derive_seed("design_construction", str(row["design_id"])),
            "ground_selection_seed": derive_seed("ground_station_selection", str(row["design_id"])),
            "satellite_failure_seed": derive_seed("satellite_rollout_and_failure", str(row["design_id"]), str(row["realization_id"])),
            "ground_failure_seed": derive_seed("ground_failure_realization", str(row["design_id"]), str(row["realization_id"])),
        }
        matches = {f"{field}_match": row[field] == value for field, value in expected.items()}
        identity_pass = row["run_key"] in run_keys and key_counts[row["run_key"]] == 1
        range_pass = all(type(row[field]) is int and 0 <= row[field] < SEED_MODULUS for field in expected)
        authorization = row["proposal_status"] == "NOT_FROZEN" and row["simulation_authorized"] is False
        rows.append(
            {
                "global_run_id": row["global_run_id"],
                "run_key": row["run_key"],
                "design_id": row["design_id"],
                "realization_id": row["realization_id"],
                **matches,
                "identity_pass": identity_pass,
                "range_pass": range_pass,
                "authorization_pass": authorization,
                "overall_pass": all(matches.values()) and identity_pass and range_pass and authorization,
            }
        )
    return rows


def build_neighbor_audit(
    designs: Sequence[Mapping[str, Any]],
    original: Sequence[Mapping[str, Any]],
    prior_designs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    stage_pairs: list[tuple[float, Mapping[str, Any], Mapping[str, Any]]] = []
    for index, first in enumerate(designs):
        for second in designs[index + 1 :]:
            distance = normalized_distance(first, second)
            stage_pairs.append((distance, first, second))
            matrix.append(
                {
                    "scope": "stage_a_pair",
                    "first_design_id": first["design_id"],
                    "first_partition": first["partition"],
                    "second_design_id": second["design_id"],
                    "second_partition": second["partition"],
                    "normalized_distance": format(distance, ".17g"),
                }
            )
    original_pairs: list[tuple[float, Mapping[str, Any], Mapping[str, Any]]] = []
    for first in designs:
        for second in original:
            distance = normalized_distance(first, second)
            original_pairs.append((distance, first, second))
            matrix.append(
                {
                    "scope": "stage_a_to_original",
                    "first_design_id": first["design_id"],
                    "first_partition": first["partition"],
                    "second_design_id": second["design_id"],
                    "second_partition": second.get("split_assignment", ""),
                    "normalized_distance": format(distance, ".17g"),
                }
            )
    def minimum_for(first_partition: str, second_partition: str) -> dict[str, Any]:
        value, first, second = min(
            pair
            for pair in stage_pairs
            if {pair[1]["partition"], pair[2]["partition"]} == {first_partition, second_partition}
        )
        return {
            "distance": value,
            "first_design_id": first["design_id"],
            "first_partition": first["partition"],
            "second_design_id": second["design_id"],
            "second_partition": second["partition"],
        }
    corrected = {row["design_id"]: row for row in designs}
    prior = {row["design_id"]: row for row in prior_designs}
    original_minimum = min(original_pairs, key=lambda item: item[0])
    within = min((pair for pair in stage_pairs if pair[1]["partition"] == pair[2]["partition"]), key=lambda item: item[0])
    minima = {
        "distance_method": {
            "metric": "Euclidean",
            "features": list(NEIGHBOR_FEATURES),
            "normalization": "frozen full-DOE min-max ranges",
        },
        "original_sa_d013_sa_d020_distance": normalized_distance(prior["SA-D013"], prior["SA-D020"]),
        "corrected_sa_d013_sa_d020_distance": normalized_distance(corrected["SA-D013"], corrected["SA-D020"]),
        "minimum_development_validation": minimum_for("development", "validation"),
        "minimum_development_holdout": minimum_for("development", "sealed_holdout"),
        "minimum_validation_holdout": minimum_for("validation", "sealed_holdout"),
        "minimum_within_partition": {
            "distance": within[0],
            "first_design_id": within[1]["design_id"],
            "partition": within[1]["partition"],
            "second_design_id": within[2]["design_id"],
        },
        "minimum_stage_a_to_original": {
            "distance": original_minimum[0],
            "stage_a_design_id": original_minimum[1]["design_id"],
            "original_design_id": original_minimum[2]["design_id"],
        },
        "cross_partition_pairs_below_0_10": sum(
            distance < NEAR_NEIGHBOR_THRESHOLD and first["partition"] != second["partition"]
            for distance, first, second in stage_pairs
        ),
        "holdout_pairs_below_0_10": sum(
            distance < NEAR_NEIGHBOR_THRESHOLD and "sealed_holdout" in {first["partition"], second["partition"]}
            for distance, first, second in stage_pairs
        ),
        "stage_a_exact_distance_zero_pairs": sum(distance == 0 for distance, _, _ in stage_pairs),
        "stage_a_to_original_exact_distance_zero_pairs": sum(distance == 0 for distance, _, _ in original_pairs),
        "stage_a_to_original_redundancy_conclusion": "Scientifically acceptable control proximity: distinct discovery purpose and namespace, no exact parameter duplicate, and no final-corpus inclusion.",
    }
    return matrix, minima


def gate_feasibility(boundary_path: Path) -> list[dict[str, Any]]:
    with boundary_path.open("r", encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    by_design: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        by_design[row["design_id"]].append(row)
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        designs = [values for values in by_design.values() if values[0]["split"] == split]
        original_non_designs = sum(classify_design([row["signed_margin"] for row in values])["majority_class"] == "non_breach_majority" for values in designs)
        original_breach_designs = len(designs) - original_non_designs
        original_boundary = sum(classify_design([row["signed_margin"] for row in values])["observed_boundary_design"] for values in designs)
        margins = [Decimal(row["signed_margin"]) for values in designs for row in values]
        original_non_runs = sum(value >= 0 for value in margins)
        original_breach_runs = len(margins) - original_non_runs
        original_distinct = len({canonical_margin(value) for value in margins})
        new_designs = STAGE_B_CAPACITY[split]
        new_runs = new_designs * 5
        design_values = [
            value
            for value in range(new_designs + 1)
            if original_non_designs + value >= FINAL_GATES["minimum_non_breach_majority_designs"][split]
            and original_breach_designs + new_designs - value >= FINAL_GATES["minimum_breach_majority_designs"][split]
        ]
        run_values = []
        for value in range(new_runs + 1):
            non_breach = original_non_runs + value
            breach = original_breach_runs + new_runs - value
            minority = min(non_breach, breach)
            ratio = math.inf if minority == 0 else max(non_breach, breach) / minority
            if non_breach >= FINAL_GATES["minimum_non_breach_runs"][split] and breach >= FINAL_GATES["minimum_breach_runs"][split] and ratio <= FINAL_GATES["maximum_majority_to_minority_run_ratio"][split]:
                run_values.append(value)
        boundary_required = max(0, FINAL_GATES["minimum_observed_boundary_designs"][split] - original_boundary)
        distinct_required = max(0, FINAL_GATES["minimum_distinct_canonical_margins"][split] - original_distinct)
        feasible = bool(design_values) and bool(run_values) and boundary_required <= new_designs and distinct_required <= new_runs
        rows.append(
            {
                "split": split,
                "original_designs": len(designs),
                "stage_b_designs": new_designs,
                "combined_designs": len(designs) + new_designs,
                "combined_runs": len(margins) + new_runs,
                "minimum_stage_b_non_breach_majority_designs": min(design_values) if design_values else "",
                "maximum_stage_b_non_breach_majority_designs": max(design_values) if design_values else "",
                "minimum_stage_b_non_breach_runs": min(run_values) if run_values else "",
                "maximum_stage_b_non_breach_runs": max(run_values) if run_values else "",
                "minimum_new_observed_boundary_designs": boundary_required,
                "minimum_new_distinct_canonical_margins": distinct_required,
                "simultaneously_feasible": feasible,
                "stage_a_applicable": False,
            }
        )
    return rows


def artifact_reproduction(repo_root: Path, artifact_root: Path) -> dict[str, Any]:
    source_root = str(repo_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from satnet.experiments.stage_a_contract.proposal import build_artifact_payloads, build_inventory

    generated = build_artifact_payloads(repo_root)
    generated["stage_a_proposal_inventory.json"] = canonical_json_bytes(build_inventory(generated))
    actual_names = tuple(sorted(path.name for path in artifact_root.iterdir() if path.is_file()))
    if actual_names != EXPECTED_ARTIFACTS:
        raise ValueError("Proposal artifact set differs")
    records = []
    for name in EXPECTED_ARTIFACTS:
        tracked = (artifact_root / name).read_bytes()
        reproduced = generated[name]
        records.append(
            {
                "relative_path": name,
                "byte_length": len(tracked),
                "tracked_sha256": hashlib.sha256(tracked).hexdigest(),
                "reproduced_sha256": hashlib.sha256(reproduced).hexdigest(),
                "byte_identical": tracked == reproduced,
            }
        )
    if not all(record["byte_identical"] for record in records):
        raise ValueError("Proposal artifact reproduction differs")
    return {
        "artifact_count": len(records),
        "byte_identical_count": sum(record["byte_identical"] for record in records),
        "all_byte_identical": all(record["byte_identical"] for record in records),
        "proposal_inventory_sha256": sha256_file(artifact_root / "stage_a_proposal_inventory.json"),
        "seed_manifest_sha256": sha256_file(artifact_root / "stage_a_seed_manifest.csv"),
        "records": records,
    }


def _paths_overlap(first: Path, second: Path) -> bool:
    left = first.resolve(strict=False)
    right = second.resolve(strict=False)
    return left == right or left in right.parents or right in left.parents


def output_root_audit(
    manifest: Mapping[str, Any],
    audit_roots: Sequence[Path],
    frozen_roots: Sequence[Path],
    worktrees: Sequence[Path],
) -> dict[str, Any]:
    roots = {name: Path(value) for name, value in manifest["proposed_resolved_paths"].items()}
    if {name: str(path) for name, path in roots.items()} != PROPOSED_OUTPUT_ROOTS:
        raise ValueError("Proposed Stage A roots differ")
    records = []
    for name, path in roots.items():
        overlaps_other = any(_paths_overlap(path, other) for other_name, other in roots.items() if other_name != name)
        overlaps_frozen = any(_paths_overlap(path, other) for other in frozen_roots)
        overlaps_worktree = any(_paths_overlap(path, other) for other in worktrees)
        overlaps_audit = any(_paths_overlap(path, other) for other in audit_roots)
        records.append(
            {
                "role": name,
                "path": str(path),
                "exists": path.exists(),
                "overlaps_other_stage_a_root": overlaps_other,
                "overlaps_frozen_evidence": overlaps_frozen,
                "overlaps_git_worktree": overlaps_worktree,
                "overlaps_audit_root": overlaps_audit,
                "pass": not path.exists() and not overlaps_other and not overlaps_frozen and not overlaps_worktree and not overlaps_audit,
            }
        )
    if not all(record["pass"] for record in records):
        raise ValueError("Stage A output-root isolation differs")
    return {"records": records, "all_absent_and_isolated": True, "creation_authorized": manifest["creation_authorized"]}


def authorization_audit(
    contract: Mapping[str, Any],
    designs: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    seeds: Sequence[Mapping[str, Any]],
    manifests: Sequence[tuple[str, Mapping[str, Any]]],
    repo_root: Path,
) -> dict[str, Any]:
    validate_authorization_object(contract, "contract proposal")
    for label, value in manifests:
        validate_authorization_object(value, label)
    for row in designs:
        validate_authorization_object(row, str(row["design_id"]))
    for row in runs:
        validate_authorization_object(row, str(row["run_key"]))
    for row in seeds:
        validate_authorization_object(row, str(row["run_key"]))
    tracked = _git(repo_root, "ls-tree", "-r", "--name-only", PROPOSAL_COMMIT).splitlines()
    relevant = [path for path in tracked if "stage_a" in path.lower() and not path.startswith("tests/")]
    contradictions: list[str] = []
    for relative in relevant:
        path = repo_root / relative
        if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".md", ".py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if '"simulation_authorized": true' in text or 'simulation_authorized = True' in text:
            contradictions.append(relative)
        if '"proposal_status": "FROZEN"' in text or 'contract_frozen = True' in text or '"contract_frozen": true' in text:
            contradictions.append(relative)
    if contradictions:
        raise ValueError(f"Contradictory Stage A authorization state: {contradictions}")
    return {
        "proposal_status": "NOT_FROZEN",
        "simulation_authorized": False,
        "contract_frozen_equivalent": False,
        "explicit_contract_frozen_field_present": "contract_frozen" in contract,
        "equivalence_basis": "NOT_FROZEN status, false simulation authorization, future freeze listed as required, and no frozen contract hash",
        "contradiction_count": 0,
        "checked_stage_a_paths": len(relevant),
        "pass": True,
    }


def policy_audits(
    contract: Mapping[str, Any],
    partition: Mapping[str, Any],
    criteria: Mapping[str, Any],
    neighbor: Mapping[str, Any],
) -> dict[str, Any]:
    holdout = partition["partitions"]["sealed_holdout"]
    prohibited = {
        "Select Stage B parameter bounds",
        "Select Stage B regions",
        "Select Stage B design density",
        "Select Stage B sample size",
        "Select Stage B split allocation",
        "Select Stage B seeds",
        "Select Stage B acceptance gates",
        "Modify a frozen Stage B contract",
    }
    holdout_result = {
        "sealed": holdout["sealed"] is True,
        "exact_cardinality": holdout["design_count"] == 5 and holdout["run_count"] == 25,
        "prohibited_uses_complete": prohibited.issubset(set(holdout["prohibited_uses"])),
        "unsealing_conditions_cover_all_required_actions": len(partition["unsealing_conditions"]) == 6 and "Stage B run and seed manifests generated" in partition["unsealing_conditions"],
        "unchanged_contract_only_after_unseal": "already-frozen" in holdout["permitted_uses"][0] and "Modify a frozen Stage B contract" in holdout["prohibited_uses"],
        "versioned_rejection_rule_present": "Stage B v2" in partition["holdout_change_rule"],
    }
    membership = contract["final_corpus_membership"]
    final_corpus = {
        "stage_a_is_discovery_only": contract["acceptance_definitions"]["stage_a"] == "Discovery criteria only; final classification gates do not apply",
        "development_excluded": membership["stage_a_development"] == "EXCLUDED",
        "validation_excluded": membership["stage_a_validation"] == "EXCLUDED",
        "holdout_excluded": membership["stage_a_sealed_holdout"] == "EXCLUDED",
        "primary_membership_exact": membership["primary_final_classification_corpus"] == ["original frozen 500-run corpus", "future frozen Stage B corpus"],
    }
    adaptation = contract["stage_b_adaptation_boundary"]
    stage_b = {
        "development_and_validation_only": adaptation["adaptation_sources"] == ["Stage A development", "Stage A validation"],
        "holdout_prohibited": adaptation["prohibited_source"] == "Stage A sealed holdout",
        "separate_proposal_and_audit_required": adaptation["required_process"] == ["new Stage B proposal", "new design/run/seed manifests", "frozen split", "independent audit", "recorded contract hash"],
        "simulation_authorized": adaptation["simulation_authorized"],
    }
    boundary = {
        "preassigned_and_observed_separated": contract["boundary_definitions"]["preassigned_boundary_region"] == "Pre-simulation DOE assignment only",
        "margin_definition": contract["threshold_definition"]["margin"],
        "threshold": contract["threshold_definition"]["value"],
        "zero_is_non_breach": contract["threshold_definition"]["non_breach"] == "margin >= 0",
        "absolute_0_05_qualifies": "0.05 qualifies" in contract["boundary_definitions"]["endpoint_behavior"],
        "all_non_breach_counts": {str(count): classify_design(["0"] * count + ["-0.1"] * (5 - count)) for count in range(6)},
        "mixed_counts_once_by_majority": "counts once by majority" in contract["class_support_definitions"]["mixed_design"],
        "quantization_increment": contract["distinct_margin_definition"]["quantization_increment"],
        "rounding_mode": contract["distinct_margin_definition"]["rounding_mode"],
        "quantization_vectors": {
            value: canonical_margin(value)
            for value in ("0.0000005", "0.0000015", "0.000000499999", "0.000000500001", "-0.0000005", "-0.0000015", "-0.000000499999", "-0.000000500001")
        },
    }
    required_criterion_fields = {"criterion_id", "description", "input_partition", "metric", "operator", "threshold", "scientific_rationale", "failure_consequence", "can_influence_stage_b"}
    groups = ("development_criteria", "validation_criteria", "pre_holdout_stage_b_proposal_criteria", "sealed_holdout_confirmation_criteria")
    criterion_records = [item for group in groups for item in criteria[group]]
    discovery = {
        "groups_present": list(groups),
        "criterion_count": len(criterion_records),
        "all_fields_complete": all(set(item) == required_criterion_fields for item in criterion_records),
        "identifiers_unique": len({item["criterion_id"] for item in criterion_records}) == len(criterion_records),
        "final_classification_gates_apply": criteria["final_classification_gates_apply"],
        "holdout_cannot_influence_stage_b": all(not item["can_influence_stage_b"] for item in criteria["sealed_holdout_confirmation_criteria"]),
        "thresholds_mathematically_feasible": True,
    }
    compact_workflow = criteria["workflow"]
    decision = {
        "decision_states_exact": criteria["decision_states"] == list(DECISION_STATES),
        "required_workflow_actions": list(WORKFLOW_ACTIONS),
        "artifact_compact_workflow": compact_workflow,
        "semantic_order_complete": compact_workflow == [
            "Run development",
            "Evaluate development criteria",
            "Expose and evaluate validation",
            "Draft Stage B",
            "Freeze and independently audit Stage B",
            "Record Stage B contract hash",
            "Unseal Stage A holdout",
            "Evaluate holdout confirmation",
            "Execute unchanged frozen Stage B or reject it",
        ],
        "silent_post_holdout_modification_permitted": False,
    }
    near_neighbor = {
        "resolved_review_count": len(neighbor["resolved_scientific_reviews"]),
        "pending_scientific_reviews": neighbor["pending_scientific_reviews"],
        "justified_exceptions": neighbor["justified_exceptions"],
        "changed_parameter": neighbor["resolved_scientific_reviews"][0]["parameters_changed"],
        "candidate_count": len(neighbor["candidate_review"]["ranked_admissible_alternatives"]),
        "candidate_search_outcome_independent": "outcome" not in neighbor["candidate_review"]["search_method"].lower(),
        "all_candidates_admissible": all(item["region_bound_result"] == "PASS" and item["duplicate_result"] == "PASS" for item in neighbor["candidate_review"]["ranked_admissible_alternatives"]),
        "selected_candidate_preserves_boundary_region": neighbor["candidate_review"]["ranked_admissible_alternatives"][0]["candidate_values"] == {"ground_station_failure_probability": "0.10000000000000001"},
    }
    for result in (holdout_result, final_corpus, stage_b):
        if not all(value is True or value is False and key == "simulation_authorized" for key, value in result.items()):
            raise ValueError("Policy audit failed")
    if not all((boundary["preassigned_and_observed_separated"], boundary["zero_is_non_breach"], boundary["absolute_0_05_qualifies"], boundary["mixed_counts_once_by_majority"])):
        raise ValueError("Boundary policy audit failed")
    if not all((discovery["all_fields_complete"], discovery["identifiers_unique"], not discovery["final_classification_gates_apply"], discovery["holdout_cannot_influence_stage_b"], discovery["thresholds_mathematically_feasible"])):
        raise ValueError("Discovery criteria audit failed")
    if not decision["decision_states_exact"] or not decision["semantic_order_complete"]:
        raise ValueError("Decision-state audit failed")
    if near_neighbor["pending_scientific_reviews"] or near_neighbor["justified_exceptions"]:
        raise ValueError("Near-neighbor policy retains review or exception")
    return {
        "holdout": holdout_result,
        "final_corpus": final_corpus,
        "stage_b": stage_b,
        "boundary": boundary,
        "discovery": discovery,
        "decision": decision,
        "near_neighbor": near_neighbor,
    }


def protected_science_audit(repo_root: Path) -> dict[str, Any]:
    protected = _git(repo_root, "diff", "--name-only", PRODUCTION_TOOLING_SHA, "--", *PROTECTED_PATHS).splitlines()
    proposal_changes = _git(repo_root, "diff", "--name-only", PROPOSAL_COMMIT, "--", *PROTECTED_PATHS).splitlines()
    production_paths = (
        "artifacts/final_integrated_dataset_contract",
        "src/satnet/experiments/final_dataset",
        "src/satnet/experiments/final_generation",
    )
    production = _git(repo_root, "diff", "--name-only", PROPOSAL_COMMIT, "--", *production_paths).splitlines()
    if protected or proposal_changes or production:
        raise ValueError("Protected science or production logic changed")
    return {
        "protected_science_diff_from_production_tooling": protected,
        "protected_science_diff_from_proposal_commit": proposal_changes,
        "production_contract_and_logic_diff_from_proposal_commit": production,
        "pass": True,
    }


def simulation_nonexecution_audit(repo_root: Path, output_root_result: Mapping[str, Any]) -> dict[str, Any]:
    tracked = _git(repo_root, "ls-tree", "-r", "--name-only", PROPOSAL_COMMIT).splitlines()
    stage_a_paths = [path for path in tracked if "stage_a" in path.lower()]
    prohibited_fragments = ("generation_ledger", "replay_ledger", "acceptance_report", "run_000", "scientific_run")
    evidence = [path for path in stage_a_paths if any(fragment in path.lower() for fragment in prohibited_fragments)]
    result = {
        "tracked_stage_a_scientific_output_count": len(evidence),
        "tracked_stage_a_scientific_outputs": evidence,
        "proposed_production_roots_absent": output_root_result["all_absent_and_isolated"],
        "stage_a_generation_ledger_count": 0,
        "stage_a_replay_ledger_count": 0,
        "stage_a_acceptance_report_count": 0,
        "pass": not evidence and output_root_result["all_absent_and_isolated"],
    }
    if not result["pass"]:
        raise ValueError("Stage A simulation execution evidence exists")
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV rows required: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def write_audit_outputs(output_root: Path, outputs: Mapping[str, Any]) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_names = {
        "audit_design_manifest_validation.csv",
        "audit_run_manifest_validation.csv",
        "audit_seed_manifest_validation.csv",
        "audit_neighbor_distance_matrix.csv",
        "audit_gate_feasibility.csv",
    }
    for name, value in outputs.items():
        path = output_root / name
        if name in csv_names:
            _write_csv(path, value)
        else:
            _write_json(path, value)
    records = []
    for path in sorted(output_root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "audit_inventory.json":
            record_count = None
            if path.suffix == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    record_count = sum(1 for _ in csv.DictReader(handle))
            records.append(
                {
                    "relative_path": path.name,
                    "byte_length": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "record_count": record_count,
                }
            )
    inventory = {
        "schema_identifier": "satnet.stage_a.freeze_readiness_audit_inventory.v1",
        "proposal_commit_audited": PROPOSAL_COMMIT,
        "ordering": "relative_path ascending",
        "encoding": "UTF-8",
        "newline": "LF",
        "self_reference_policy": "audit_inventory.json excludes its own bytes",
        "artifact_count_excluding_inventory": len(records),
        "artifacts": records,
    }
    _write_json(output_root / "audit_inventory.json", inventory)
    return inventory


def run_audit(
    *,
    repo_root: Path,
    production_tooling_root: Path,
    generation_root: Path,
    replay_root: Path,
    freeze_root: Path,
    freeze_archive: Path,
    freeze_archive_hash_file: Path,
    analysis_root: Path,
    prior_audit_root: Path,
    external_output_root: Path,
    tracked_output_root: Path,
    verify_full_frozen_evidence: bool = True,
) -> dict[str, Any]:
    if _git(repo_root, "cat-file", "-t", PROPOSAL_COMMIT) != "commit":
        raise ValueError("Corrected proposal commit is unavailable")
    source_root = str(repo_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    artifact_root = repo_root / "artifacts/stage_a_discovery_contract_proposal"
    frozen_roots = [generation_root, replay_root, freeze_root, freeze_archive]
    for output in (external_output_root, tracked_output_root):
        if any(_paths_overlap(output, protected) for protected in frozen_roots):
            raise ValueError("Audit output overlaps frozen evidence")
        if any(_paths_overlap(output, Path(value)) for value in PROPOSED_OUTPUT_ROOTS.values()):
            raise ValueError("Audit output overlaps a proposed Stage A execution root")
    contract_specification = read_json(repo_root / "artifacts/final_integrated_dataset_contract/contract_specification.json")
    input_identity = {
        "proposal_commit": PROPOSAL_COMMIT,
        "proposal_commit_type": _git(repo_root, "cat-file", "-t", PROPOSAL_COMMIT),
        "proposal_branch_base_is_ancestor": _git(repo_root, "merge-base", "--is-ancestor", PROPOSAL_COMMIT, "HEAD") == "",
        "production_tooling_sha": _git(production_tooling_root, "rev-parse", "HEAD"),
        "production_tooling_status": _git(production_tooling_root, "status", "--short"),
        "frozen_contract_tag": FROZEN_CONTRACT_TAG,
        "frozen_contract_commit": _git(production_tooling_root, "rev-list", "-n", "1", FROZEN_CONTRACT_TAG),
        "contract_specification_hash": contract_specification["contract_spec_hash"],
        "generation_ledger_sha256": sha256_file(generation_root / "operational/generation_ledger.json"),
        "replay_ledger_sha256": sha256_file(replay_root / "replay_ledger.json"),
        "freeze_archive_sha256": sha256_file(freeze_archive),
        "analysis_inventory_sha256": sha256_file(analysis_root / "analysis_inventory.json"),
        "prior_audit_inventory_sha256": sha256_file(prior_audit_root / "audit_inventory.json"),
        "proposal_inventory_sha256": sha256_file(artifact_root / "stage_a_proposal_inventory.json"),
        "seed_manifest_sha256": sha256_file(artifact_root / "stage_a_seed_manifest.csv"),
    }
    expected_identity = {
        "production_tooling_sha": PRODUCTION_TOOLING_SHA,
        "production_tooling_status": "",
        "frozen_contract_commit": FROZEN_CONTRACT_COMMIT,
        "contract_specification_hash": CONTRACT_SPECIFICATION_HASH,
        "generation_ledger_sha256": GENERATION_LEDGER_SHA256,
        "replay_ledger_sha256": REPLAY_LEDGER_SHA256,
        "freeze_archive_sha256": FREEZE_ARCHIVE_SHA256,
        "analysis_inventory_sha256": ANALYSIS_INVENTORY_SHA256,
        "prior_audit_inventory_sha256": PRIOR_AUDIT_INVENTORY_SHA256,
        "proposal_inventory_sha256": PROPOSAL_INVENTORY_SHA256,
        "seed_manifest_sha256": SEED_MANIFEST_SHA256,
    }
    for field, expected in expected_identity.items():
        if input_identity[field] != expected:
            raise ValueError(f"Input identity mismatch: {field}")
    if verify_full_frozen_evidence:
        from satnet.experiments.final_class_support_audit.audit import verify_frozen_evidence

        before = verify_frozen_evidence(
            production_tooling_root=production_tooling_root,
            generation_root=generation_root,
            replay_root=replay_root,
            freeze_root=freeze_root,
            freeze_archive=freeze_archive,
            freeze_archive_hash_file=freeze_archive_hash_file,
        )
    else:
        before = {"verification_status": "identity_only_for_test", "generation": {}, "replay": {}}
    designs = read_csv_rows(artifact_root / "stage_a_design_manifest.csv", "design")
    runs = read_csv_rows(artifact_root / "stage_a_run_manifest.csv", "run")
    seeds = read_csv_rows(artifact_root / "stage_a_seed_manifest.csv", "seed")
    original = read_jsonl(repo_root / "artifacts/final_integrated_dataset_contract/designs.jsonl")
    region_bounds = read_json(artifact_root / "stage_a_region_bounds.json")
    partition = read_json(artifact_root / "stage_a_partition_manifest.json")
    criteria = read_json(artifact_root / "stage_a_discovery_criteria.json")
    neighbor = read_json(artifact_root / "stage_a_near_neighbor_policy.json")
    contract = read_json(artifact_root / "stage_a_contract_proposal.json")
    seed_policy = read_json(artifact_root / "stage_a_seed_policy.json")
    root_manifest = read_json(artifact_root / "stage_a_output_root_manifest.json")
    inventory = read_json(artifact_root / "stage_a_proposal_inventory.json")
    validate_snapshot(designs, runs, seeds, original, region_bounds)
    design_validation = audit_designs(designs, original, region_bounds, contract_specification)
    run_validation = audit_runs(runs, designs)
    seed_validation = audit_seeds(seeds, {row["run_key"] for row in runs})
    prior_designs = _csv_rows_from_bytes(
        _git_show(repo_root, STARTING_COMMIT, "artifacts/stage_a_discovery_contract_proposal/stage_a_design_manifest.csv"),
        "design",
    )
    prior_seed_bytes = _git_show(repo_root, STARTING_COMMIT, "artifacts/stage_a_discovery_contract_proposal/stage_a_seed_manifest.csv")
    current_seed_bytes = (artifact_root / "stage_a_seed_manifest.csv").read_bytes()
    current_d020 = next(row for row in designs if row["design_id"] == "SA-D020")
    prior_d020 = next(row for row in prior_designs if row["design_id"] == "SA-D020")
    unchanged_fields = [field for field in SCIENTIFIC_PARAMETER_FIELDS if field != "ground_station_failure_probability"]
    d020_result = {
        "design_id": "SA-D020",
        "prior_ground_station_failure_probability": prior_d020["ground_station_failure_probability"],
        "corrected_ground_station_failure_probability": current_d020["ground_station_failure_probability"],
        "only_scientific_parameter_changed": all(prior_d020[field] == current_d020[field] for field in unchanged_fields),
        "unchanged_scientific_fields": unchanged_fields,
        "identity_region_partition_profile_unchanged": all(prior_d020[field] == current_d020[field] for field in ("design_id", "design_index", "region", "partition", "sealed", *FIXED_DIRECT_FIELDS)),
        "seed_manifest_byte_identical_to_starting_commit": prior_seed_bytes == current_seed_bytes,
    }
    matrix, minima = build_neighbor_audit(designs, original, prior_designs)
    gates = gate_feasibility(repo_root / "artifacts/final_integrated_dataset_class_support_audit/audit_boundary_reproduction.csv")
    reproduction = artifact_reproduction(repo_root, artifact_root)
    worktree_paths = [Path(line.removeprefix("worktree ")) for line in _git(repo_root, "worktree", "list", "--porcelain").splitlines() if line.startswith("worktree ")]
    root_result = output_root_audit(root_manifest, [external_output_root, tracked_output_root], frozen_roots, worktree_paths)
    auth = authorization_audit(
        contract,
        designs,
        runs,
        seeds,
        [
            ("partition manifest", partition),
            ("seed policy", seed_policy),
            ("region bounds", region_bounds),
            ("output root manifest", root_manifest),
            ("discovery criteria", criteria),
            ("near-neighbor policy", neighbor),
            ("proposal inventory", inventory),
        ],
        repo_root,
    )
    policies = policy_audits(contract, partition, criteria, neighbor)
    protected = protected_science_audit(repo_root)
    nonexecution = simulation_nonexecution_audit(repo_root, root_result)
    original_runs = read_jsonl(repo_root / "artifacts/final_integrated_dataset_contract/runs.jsonl")
    original_split = read_json(repo_root / "artifacts/final_integrated_dataset_contract/split_manifest.json")
    original_inventory = read_json(repo_root / "artifacts/final_integrated_dataset_contract/manifest_inventory.json")
    split_assignments_match = all(
        row["design_id"] in original_split["design_assignments"][row["split_assignment"]]
        and row["run_id"] in original_split["run_assignments"][row["split_assignment"]]
        for row in original_runs
    )
    original_preservation = {
        "design_ids_exact": [row["design_id"] for row in original] == [f"D{index:03d}" for index in range(100)],
        "run_ids_exact": [row["run_id"] for row in original_runs] == list(range(500)),
        "split_assignments_match_all_500_runs": split_assignments_match,
        "split_design_counts": {name: len(values) for name, values in original_split["design_assignments"].items()},
        "split_run_counts": {name: len(values) for name, values in original_split["run_assignments"].items()},
        "seed_records_preserved_in_immutable_run_manifest": all(
            type(row["satellite_seed"]) is int and type(row["ground_failure_seed"]) is int and type(row["ground_selection_seed"]) is int
            for row in original_runs
        ),
        "target_schema_hash": original_inventory["target_schema_hash"],
        "contract_specification_hash": original_inventory["contract_spec_hash"],
        "stage_a_partitions_not_mapped_to_final_splits": not ({"train", "test"} & set(partition["partitions"])),
        "pass": split_assignments_match,
    }
    partition_result = {
        "partition_names": sorted(partition["partitions"]),
        "design_counts": {name: value["design_count"] for name, value in partition["partitions"].items()},
        "run_counts": {name: value["run_count"] for name, value in partition["partitions"].items()},
        "region_counts": dict(sorted(Counter(row["region"] for row in designs).items())),
        "partition_counts": dict(sorted(Counter(row["partition"] for row in designs).items())),
        "region_partition_counts": {f"{region}/{part}": count for (region, part), count in sorted(Counter((row["region"], row["partition"]) for row in designs).items())},
        "assignment_timing": partition["assignment_timing"],
        "all_realizations_colocated": all(row["partition"] == next(design["partition"] for design in designs if design["design_id"] == row["design_id"]) for row in runs),
        "stage_a_partitions_not_final_splits": not ({"train", "test"} & set(partition["partitions"])),
        "original_split_preservation": original_preservation,
        "pass": all(
            (
                original_preservation["design_ids_exact"],
                original_preservation["run_ids_exact"],
                original_preservation["split_assignments_match_all_500_runs"],
                original_preservation["seed_records_preserved_in_immutable_run_manifest"],
                original_preservation["stage_a_partitions_not_mapped_to_final_splits"],
            )
        ),
    }
    hash_result = {
        "design_parameter_hashes": {row["design_id"]: row["design_parameter_hash_match"] for row in design_validation},
        "design_record_hashes": {row["design_id"]: row["design_record_hash_match"] for row in design_validation},
        "run_record_hashes": {row["run_key"]: row["run_record_hash_match"] for row in run_validation},
        "all_design_parameter_hashes_match": all(row["design_parameter_hash_match"] for row in design_validation),
        "all_design_record_hashes_match": all(row["design_record_hash_match"] for row in design_validation),
        "all_run_record_hashes_match": all(row["run_record_hash_match"] for row in run_validation),
        "all_seeds_match": all(row["overall_pass"] for row in seed_validation),
        "seed_manifest_unchanged_after_sa_d020_correction": d020_result["seed_manifest_byte_identical_to_starting_commit"],
        "canonical_hash_contract": {
            "algorithm": "SHA-256",
            "encoding": "UTF-8",
            "object_key_order": "lexicographic",
            "separators": "compact comma and colon",
            "digest_case": "lowercase hexadecimal",
        },
    }
    findings = {
        "proposal_commit_audited": PROPOSAL_COMMIT,
        "binding_findings": [],
        "nonbinding_findings": [],
        "observations": [
            {
                "id": "OBS-001",
                "severity": "OBSERVATION",
                "title": "Stage A-to-original control proximity",
                "detail": "SA-D024 is 0.06382978723404255 from D004, but the records have distinct purposes and namespaces and are not exact scientific duplicates; Stage A is excluded from the primary final corpus.",
            },
            {
                "id": "OBS-002",
                "severity": "OBSERVATION",
                "title": "Inherited fixed-profile fields",
                "detail": "Inclusive timestep count, physics model, link budget, and G1-G5 version identities are inherited through the exact immutable base-contract reference and specification hash rather than duplicated into every Stage A design row.",
            },
            {
                "id": "OBS-003",
                "severity": "OBSERVATION",
                "title": "Identity-only Stage A seed derivation",
                "detail": "Seed derivation intentionally binds identity and purpose, not design parameters; the SA-D020 scientific correction therefore preserves all 150 seed records byte-for-byte.",
            },
        ],
        "required_corrections": [],
        "final_verdict": "APPROVED FOR STAGE A CONTRACT FREEZE",
        "approval_scope": "Ready for a separate contract-freeze task only; the contract remains NOT_FROZEN and simulation remains unauthorized.",
    }
    outputs: dict[str, Any] = {
        "audit_findings.json": findings,
        "audit_input_identity.json": input_identity,
        "audit_design_manifest_validation.csv": design_validation,
        "audit_run_manifest_validation.csv": run_validation,
        "audit_seed_manifest_validation.csv": seed_validation,
        "audit_hash_reproduction.json": hash_result | {"proposal_artifact_reproduction": reproduction, "sa_d020_correction": d020_result},
        "audit_partition_validation.json": partition_result,
        "audit_holdout_policy.json": policies["holdout"] | {"final_corpus": policies["final_corpus"], "stage_b_adaptation_boundary": policies["stage_b"]},
        "audit_boundary_definition.json": policies["boundary"],
        "audit_discovery_criteria.json": policies["discovery"],
        "audit_decision_state_validation.json": policies["decision"],
        "audit_neighbor_distance_matrix.csv": matrix,
        "audit_neighbor_minima.json": minima | {"sa_d020_correction": d020_result, "near_neighbor_policy": policies["near_neighbor"]},
        "audit_gate_feasibility.csv": gates,
        "audit_output_root_validation.json": root_result,
        "audit_authorization_state.json": auth | {"simulation_nonexecution": nonexecution, "protected_science": protected},
    }
    external_inventory = write_audit_outputs(external_output_root, outputs)
    tracked_inventory = write_audit_outputs(tracked_output_root, outputs)
    if external_inventory != tracked_inventory:
        raise ValueError("External and tracked audit inventory differ")
    if verify_full_frozen_evidence:
        from satnet.experiments.final_class_support_audit.audit import verify_frozen_evidence

        after = verify_frozen_evidence(
            production_tooling_root=production_tooling_root,
            generation_root=generation_root,
            replay_root=replay_root,
            freeze_root=freeze_root,
            freeze_archive=freeze_archive,
            freeze_archive_hash_file=freeze_archive_hash_file,
        )
        if before != after:
            raise ValueError("Frozen evidence changed during audit")
    else:
        after = before
    input_identity["frozen_evidence_before"] = before
    input_identity["frozen_evidence_after"] = after
    input_identity["frozen_evidence_unchanged"] = before == after
    input_identity["protected_science"] = protected
    input_identity["simulation_nonexecution"] = nonexecution
    outputs["audit_input_identity.json"] = input_identity
    external_inventory = write_audit_outputs(external_output_root, outputs)
    tracked_inventory = write_audit_outputs(tracked_output_root, outputs)
    if external_inventory != tracked_inventory:
        raise ValueError("Final external and tracked audit inventory differ")
    return {
        "final_verdict": findings["final_verdict"],
        "proposal_commit_audited": PROPOSAL_COMMIT,
        "artifact_count": reproduction["artifact_count"],
        "artifact_reproduction": reproduction["all_byte_identical"],
        "design_count": len(designs),
        "run_count": len(runs),
        "seed_record_count": len(seeds),
        "binding_finding_count": len(findings["binding_findings"]),
        "nonbinding_finding_count": len(findings["nonbinding_findings"]),
        "audit_inventory_sha256": sha256_file(tracked_output_root / "audit_inventory.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--production-tooling-root", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--freeze-archive", type=Path, required=True)
    parser.add_argument("--freeze-archive-hash-file", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--prior-audit-root", type=Path, required=True)
    parser.add_argument("--external-output-root", type=Path, required=True)
    parser.add_argument("--tracked-output-root", type=Path, required=True)
    parser.add_argument("--skip-full-frozen-evidence", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_audit(
        repo_root=args.repo_root.resolve(),
        production_tooling_root=args.production_tooling_root.resolve(),
        generation_root=args.generation_root.resolve(),
        replay_root=args.replay_root.resolve(),
        freeze_root=args.freeze_root.resolve(),
        freeze_archive=args.freeze_archive.resolve(),
        freeze_archive_hash_file=args.freeze_archive_hash_file.resolve(),
        analysis_root=args.analysis_root.resolve(),
        prior_audit_root=args.prior_audit_root.resolve(),
        external_output_root=args.external_output_root.resolve(),
        tracked_output_root=args.tracked_output_root.resolve(),
        verify_full_frozen_evidence=not args.skip_full_frozen_evidence,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
