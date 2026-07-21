from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from satnet.ground.canonical import canonical_float_string, canonical_json

CORPUS_NAMESPACE = "stage_a_discovery_v1"
PROPOSAL_VERSION = "1"
PROPOSAL_STATUS = "NOT_FROZEN"
SIMULATION_AUTHORIZED = False
PROPOSAL_SCHEMA = "satnet.stage_a_discovery_contract_proposal.v1"
SEED_POLICY_VERSION = "satnet.stage_a_seed_policy.v1"
SEED_DOMAIN = "satnet_stage_a_discovery_v1_seed"
SEED_MODULUS = 2**63
SERVICE_THRESHOLD = Decimal("0.80")
MARGIN_QUANTUM = Decimal("0.000001")
AUDIT_COMMIT = "5add33fcd8205f59e8694b104c9b34f82bd66cd1"
AUDIT_IMPLEMENTATION_COMMIT = "47d0fa6d428bdbf8796e0d9db2eb5f3767b7dd94"
ANALYSIS_COMMIT = "340af3f22f2bc7facc4f0e749b10490f06e80cde"
PRODUCTION_TOOLING_SHA = "9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab"
FROZEN_CONTRACT_COMMIT = "a1967185e80327e4b00c1831828dc975ab6819fc"
CONTRACT_SPECIFICATION_HASH = "482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b"
GENERATION_LEDGER_SHA256 = "a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1"
REPLAY_LEDGER_SHA256 = "4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd"
FREEZE_ARCHIVE_SHA256 = "375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc"
ANALYSIS_INVENTORY_SHA256 = "5ac295ba0e08797b63a2ce3f062a1995dc7aa850f11a2bf5d83a0580731afd85"
AUDIT_INVENTORY_SHA256 = "20d2e7037940d708e133936d3907ad822caac3b46cd73c84329f3efd16c4037a"
FIXED_PROFILE = {
    "duration_minutes": 10,
    "step_seconds": 60,
    "epoch_iso": "2000-01-01T12:00:00+00:00",
    "orbital_engine": "sgp4",
    "max_isl_distance_km": 10000.0,
    "isl_policy": "grid_fixed",
    "adjacent_search_k": 1,
    "max_inter_plane_links_per_sat": 1,
    "satellite_failure_model": "persistent_temporal_union_edges_v1",
    "minimum_elevation_deg": 10.0,
    "space_gcc_threshold": 0.8,
    "ground_service_threshold": 0.8,
    "ground_service_policy_hash": "e314b9d4123ef84832965b1baadd04a118805f8cab368bd548a3f8f65ae02950",
    "visibility_policy_hash": "5dda243a39d13746d4e9d16922318df1774c4cf29a95fbc70677b618a41c45b1",
}
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
    *FIXED_PROFILE,
)
FINAL_GATES = {
    "minimum_non_breach_majority_designs": {"train": 12, "validation": 4, "test": 4},
    "minimum_non_breach_runs": {"train": 50, "validation": 14, "test": 14},
    "minimum_breach_majority_designs": {"train": 40, "validation": 10, "test": 10},
    "minimum_breach_runs": {"train": 200, "validation": 60, "test": 60},
    "maximum_majority_to_minority_run_ratio": {"train": 12.0, "validation": 10.0, "test": 10.0},
    "minimum_observed_boundary_designs": {"train": 10, "validation": 3, "test": 3},
    "minimum_distinct_canonical_margins": {"train": 30, "validation": 12, "test": 12},
}
OUTPUT_ROOTS = {
    "production_generation": "C:\\Users\\johns\\satnet-stage-a-discovery-v1-production",
    "production_replay": "C:\\Users\\johns\\satnet-stage-a-discovery-v1-replay",
    "production_acceptance": "C:\\Users\\johns\\satnet-stage-a-discovery-v1-acceptance",
    "evidence_freeze": "C:\\Users\\johns\\satnet-stage-a-discovery-v1-freeze",
}
FROZEN_ROOTS = (
    "C:\\Users\\johns\\satnet-final-production-20260720",
    "C:\\Users\\johns\\satnet-final-production-replay-20260720",
    "C:\\Users\\johns\\satnet-final-production-v1-freeze-20260721",
    "C:\\Users\\johns\\satnet-final-production-v1-freeze-20260721.zip",
)


def canonical_payload_hash(payload: Mapping[str, Any], *, domain: str, version: str = "1") -> str:
    wrapped = {"identity_domain": domain, "identity_version": version, "payload": dict(payload)}
    return hashlib.sha256(canonical_json(wrapped).encode("utf-8")).hexdigest()


def derive_seed(*, purpose: str, design_id: str, realization_id: str | None = None) -> int:
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
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % SEED_MODULUS


def canonical_margin(value: float | Decimal | str) -> str:
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    return format(decimal_value.quantize(MARGIN_QUANTUM, rounding=ROUND_HALF_EVEN), ".6f")


def observed_boundary_design(margins: Sequence[float | Decimal | str]) -> bool:
    if len(margins) != 5:
        raise ValueError("Observed-boundary evaluation requires exactly five margins")
    values = [value if isinstance(value, Decimal) else Decimal(str(value)) for value in margins]
    return min(values) < 0 <= max(values) or sum(abs(value) <= Decimal("0.05") for value in values) >= 2


def design_outcome(margins: Sequence[float | Decimal | str]) -> dict[str, Any]:
    if len(margins) != 5:
        raise ValueError("Design outcome requires exactly five margins")
    values = [value if isinstance(value, Decimal) else Decimal(str(value)) for value in margins]
    non_breach = sum(value >= 0 for value in values)
    return {
        "majority_class": "non_breach_majority" if non_breach >= 3 else "breach_majority",
        "mixed_design": 0 < non_breach < 5,
        "observed_boundary_design": observed_boundary_design(values),
        "non_breach_realization_count": non_breach,
        "breach_realization_count": 5 - non_breach,
    }


def normalized_vector(row: Mapping[str, Any]) -> tuple[float, ...]:
    total = float(row["total_ground_station_count"])
    values = {
        **{key: float(row[key]) for key in DOE_RANGES if not key.endswith("_fraction")},
        "civilian_fraction": float(row["civilian_count"]) / total,
        "government_fraction": float(row["government_count"]) / total,
        "military_fraction": float(row["military_count"]) / total,
    }
    return tuple((values[key] - DOE_RANGES[key][0]) / (DOE_RANGES[key][1] - DOE_RANGES[key][0]) for key in NEIGHBOR_FEATURES)


def normalized_distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    return sum((left - right) ** 2 for left, right in zip(normalized_vector(first), normalized_vector(second), strict=True)) ** 0.5


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_float(value: float) -> str:
    return canonical_float_string(value)


def paths_overlap(first: str | Path, second: str | Path) -> bool:
    left = Path(first).resolve()
    right = Path(second).resolve()
    return left == right or left in right.parents or right in left.parents
