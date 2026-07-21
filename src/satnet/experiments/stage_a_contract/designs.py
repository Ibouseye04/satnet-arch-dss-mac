from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from satnet.experiments.stage_a_contract.semantics import (
    CONTRACT_SPECIFICATION_HASH,
    CORPUS_NAMESPACE,
    DOE_RANGES,
    FIXED_PROFILE,
    PROPOSAL_STATUS,
    SCIENTIFIC_PARAMETER_FIELDS,
    canonical_float,
    canonical_payload_hash,
    derive_seed,
    normalized_distance,
)

DESIGN_VALUES = (
    ("resilient_core", "development", 6, 8, 1200, 98, 0.000, 0.000, 9, 6, 5, 0.000),
    ("resilient_core", "development", 6, 8, 1100, 90, 0.010, 0.010, 10, 8, 7, 0.010),
    ("resilient_core", "development", 6, 8, 1000, 80, 0.020, 0.025, 12, 9, 9, 0.020),
    ("resilient_core", "development", 6, 8, 900, 70, 0.030, 0.040, 14, 11, 10, 0.030),
    ("resilient_core", "development", 6, 8, 1150, 75, 0.010, 0.030, 8, 8, 4, 0.040),
    ("resilient_core", "development", 6, 8, 1050, 95, 0.025, 0.015, 15, 5, 5, 0.020),
    ("resilient_core", "development", 6, 8, 950, 85, 0.040, 0.050, 6, 12, 12, 0.050),
    ("resilient_core", "development", 6, 8, 850, 65, 0.050, 0.060, 7, 7, 6, 0.060),
    ("resilient_core", "validation", 6, 8, 1175, 88, 0.015, 0.020, 10, 10, 5, 0.015),
    ("resilient_core", "validation", 6, 8, 875, 72, 0.045, 0.055, 12, 6, 7, 0.055),
    ("resilient_core", "sealed_holdout", 6, 8, 1125, 92, 0.020, 0.035, 9, 12, 9, 0.025),
    ("resilient_core", "sealed_holdout", 6, 8, 825, 62, 0.055, 0.065, 9, 5, 6, 0.070),
    ("boundary", "development", 6, 8, 805, 60, 0.050, 0.050, 11, 5, 4, 0.050),
    ("boundary", "development", 6, 8, 750, 58, 0.055, 0.060, 8, 5, 5, 0.060),
    ("boundary", "development", 6, 8, 700, 56, 0.060, 0.070, 6, 5, 4, 0.080),
    ("boundary", "development", 6, 8, 650, 55, 0.070, 0.080, 5, 3, 4, 0.100),
    ("boundary", "development", 5, 8, 850, 65, 0.040, 0.050, 8, 6, 6, 0.040),
    ("boundary", "development", 5, 8, 775, 60, 0.050, 0.070, 6, 6, 6, 0.070),
    ("boundary", "development", 6, 7, 825, 62, 0.045, 0.055, 10, 5, 5, 0.060),
    ("boundary", "development", 6, 7, 700, 57, 0.065, 0.090, 5, 6, 4, 0.120),
    ("boundary", "validation", 6, 8, 725, 59, 0.055, 0.065, 9, 6, 5, 0.075),
    ("boundary", "validation", 5, 8, 675, 54, 0.075, 0.095, 7, 4, 4, 0.110),
    ("boundary", "sealed_holdout", 6, 8, 775, 63, 0.060, 0.080, 9, 4, 5, 0.090),
    ("boundary", "sealed_holdout", 5, 7, 625, 52, 0.085, 0.110, 4, 4, 4, 0.140),
    ("global_control", "development", 4, 5, 300, 30, 0.200, 0.250, 2, 2, 2, 0.400),
    ("global_control", "development", 5, 6, 500, 45, 0.150, 0.200, 5, 5, 5, 0.250),
    ("global_control", "development", 4, 8, 1200, 98, 0.000, 0.000, 20, 15, 15, 0.000),
    ("global_control", "development", 6, 5, 900, 75, 0.100, 0.125, 18, 6, 6, 0.200),
    ("global_control", "validation", 5, 7, 400, 90, 0.180, 0.050, 5, 25, 10, 0.350),
    ("global_control", "sealed_holdout", 4, 6, 1050, 35, 0.020, 0.220, 2, 3, 5, 0.100),
)

REGION_BOUNDS = {
    "resilient_core": {
        "num_planes": {"allowed_values": [6]},
        "sats_per_plane": {"allowed_values": [8]},
        "altitude_km": {"minimum": 800.0, "maximum": 1200.0},
        "inclination_deg": {"minimum": 60.0, "maximum": 98.0},
        "satellite_node_failure_probability": {"minimum": 0.0, "maximum": 0.06},
        "satellite_edge_failure_probability": {"minimum": 0.0, "maximum": 0.07},
        "total_ground_station_count": {"minimum": 18, "maximum": 35},
        "ground_station_failure_probability": {"minimum": 0.0, "maximum": 0.08},
    },
    "boundary": {
        "num_planes": {"allowed_values": [5, 6]},
        "sats_per_plane": {"allowed_values": [7, 8]},
        "altitude_km": {"minimum": 600.0, "maximum": 900.0},
        "inclination_deg": {"minimum": 50.0, "maximum": 75.0},
        "satellite_node_failure_probability": {"minimum": 0.03, "maximum": 0.10},
        "satellite_edge_failure_probability": {"minimum": 0.03, "maximum": 0.12},
        "total_ground_station_count": {"minimum": 10, "maximum": 30},
        "ground_station_failure_probability": {"minimum": 0.03, "maximum": 0.15},
    },
    "global_control": {
        "num_planes": {"allowed_values": [4, 5, 6]},
        "sats_per_plane": {"allowed_values": [5, 6, 7, 8]},
        "altitude_km": {"minimum": 300.0, "maximum": 1200.0},
        "inclination_deg": {"minimum": 30.0, "maximum": 98.0},
        "satellite_node_failure_probability": {"minimum": 0.0, "maximum": 0.20},
        "satellite_edge_failure_probability": {"minimum": 0.0, "maximum": 0.25},
        "total_ground_station_count": {"allowed_values": [6, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
        "ground_station_failure_probability": {"minimum": 0.0, "maximum": 0.40},
    },
}


def _within(value: float, rule: Mapping[str, Any]) -> bool:
    if "allowed_values" in rule:
        return value in rule["allowed_values"]
    return float(rule["minimum"]) <= value <= float(rule["maximum"])


def scientific_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in SCIENTIFIC_PARAMETER_FIELDS)


def build_design_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, values in enumerate(DESIGN_VALUES):
        region, partition, planes, sats, altitude, inclination, node, edge, civilian, government, military, ground = values
        row: dict[str, Any] = {
            "corpus_namespace": CORPUS_NAMESPACE,
            "design_id": f"SA-D{index:03d}",
            "design_index": index,
            "region": region,
            "partition": partition,
            "sealed": partition == "sealed_holdout",
            "proposal_status": PROPOSAL_STATUS,
            "num_planes": planes,
            "sats_per_plane": sats,
            "configured_satellite_count": planes * sats,
            "altitude_km": canonical_float(float(altitude)),
            "inclination_deg": canonical_float(float(inclination)),
            "phasing_factor": 1,
            "satellite_node_failure_probability": canonical_float(float(node)),
            "satellite_edge_failure_probability": canonical_float(float(edge)),
            "civilian_count": civilian,
            "government_count": government,
            "military_count": military,
            "total_ground_station_count": civilian + government + military,
            "ground_station_failure_probability": canonical_float(float(ground)),
            "ground_selection_policy": "deterministic_catalog_selection_v1",
            "fixed_simulation_profile_reference": "SATNET Final Integrated Production Corpus v1 fixed_profile",
            "base_contract_reference": "final-integrated-dataset-contract-v1@a1967185e80327e4b00c1831828dc975ab6819fc",
            "base_contract_specification_hash": CONTRACT_SPECIFICATION_HASH,
        }
        for field, value in FIXED_PROFILE.items():
            row[field] = canonical_float(value) if isinstance(value, float) else value
        parameter_payload = {field: row[field] for field in SCIENTIFIC_PARAMETER_FIELDS}
        row["design_parameter_hash"] = canonical_payload_hash(parameter_payload, domain="satnet_stage_a_design_parameters")
        row["design_construction_seed"] = derive_seed(purpose="design_construction", design_id=row["design_id"])
        row["ground_selection_seed"] = derive_seed(purpose="ground_station_selection", design_id=row["design_id"])
        row["design_record_hash"] = canonical_payload_hash(row, domain="satnet_stage_a_design_record")
        rows.append(row)
    validate_design_rows(rows)
    return rows


def validate_design_rows(rows: Sequence[Mapping[str, Any]], original_rows: Iterable[Mapping[str, Any]] = ()) -> None:
    if len(rows) != 30:
        raise ValueError("Stage A requires exactly 30 designs")
    expected_ids = [f"SA-D{index:03d}" for index in range(30)]
    ids = [str(row["design_id"]) for row in rows]
    if ids != expected_ids or len(ids) != len(set(ids)):
        raise ValueError("Stage A design identities must be unique SA-D000 through SA-D029")
    if any(str(value).startswith("D") and not str(value).startswith("SA-") for value in ids):
        raise ValueError("Stage A design identity collides with original namespace")
    if Counter(row["region"] for row in rows) != {"resilient_core": 12, "boundary": 12, "global_control": 6}:
        raise ValueError("Stage A region allocation is invalid")
    if Counter(row["partition"] for row in rows) != {"development": 20, "validation": 5, "sealed_holdout": 5}:
        raise ValueError("Stage A partition allocation is invalid")
    expected_cross = {
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
    if Counter((row["region"], row["partition"]) for row in rows) != expected_cross:
        raise ValueError("Stage A region-by-partition allocation is invalid")
    signatures = [scientific_signature(row) for row in rows]
    if len(signatures) != len(set(signatures)):
        raise ValueError("Stage A contains a duplicate parameter vector")
    original_signatures = {scientific_signature(row) for row in original_rows}
    if original_signatures.intersection(signatures):
        raise ValueError("Stage A contains an exact original design duplicate")
    for row in rows:
        if row["proposal_status"] != PROPOSAL_STATUS:
            raise ValueError("Every Stage A design must remain NOT_FROZEN")
        if bool(row["sealed"]) is (row["partition"] != "sealed_holdout"):
            raise ValueError("Stage A sealed flag differs from partition")
        if int(row["configured_satellite_count"]) != int(row["num_planes"]) * int(row["sats_per_plane"]):
            raise ValueError("Configured satellite count is invalid")
        if sum(int(row[field]) for field in ("civilian_count", "government_count", "military_count")) != int(row["total_ground_station_count"]):
            raise ValueError("Ground-station class counts do not sum to total")
        for field, bounds in DOE_RANGES.items():
            if field.endswith("_fraction"):
                continue
            if not bounds[0] <= float(row[field]) <= bounds[1]:
                raise ValueError(f"{row['design_id']} violates base DOE bounds for {field}")
        for field, rule in REGION_BOUNDS[str(row["region"])].items():
            if not _within(float(row[field]), rule):
                raise ValueError(f"{row['design_id']} violates assigned region bounds for {field}")
        expected_parameter = canonical_payload_hash({field: row[field] for field in SCIENTIFIC_PARAMETER_FIELDS}, domain="satnet_stage_a_design_parameters")
        if row["design_parameter_hash"] != expected_parameter:
            raise ValueError("Design-parameter hash mismatch")
        expected_record = canonical_payload_hash({key: value for key, value in row.items() if key != "design_record_hash"}, domain="satnet_stage_a_design_record")
        if row["design_record_hash"] != expected_record:
            raise ValueError("Design-record hash mismatch")


def minimum_distances(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    cross = [normalized_distance(first, second) for index, first in enumerate(rows) for second in rows[index + 1 :] if first["partition"] != second["partition"]]
    sealed = [normalized_distance(first, second) for first in rows if first["partition"] == "sealed_holdout" for second in rows if second["partition"] != "sealed_holdout"]
    return {"minimum_cross_partition_distance": min(cross), "minimum_sealed_to_unsealed_distance": min(sealed)}
