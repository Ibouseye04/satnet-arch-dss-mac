from __future__ import annotations

from collections import Counter
from fractions import Fraction
import math
from typing import Any, Iterable

from satnet.experiments.final_dataset.deterministic import (
    canonical_digest,
    split_candidate_payload,
)
from satnet.ground.canonical import canonical_hash

SPLIT_MANIFEST_IDENTITY_DOMAIN = "satnet_final_integrated_dataset_split_manifest"
SPLIT_MANIFEST_IDENTITY_VERSION = "2"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_SIZES = {"train": 1400, "validation": 300, "test": 300}
SPLIT_RUN_SIZES = {"train": 7000, "validation": 1500, "test": 1500}
SPLIT_CANDIDATE_COUNT = 4096
FROZEN_SPLIT_CANDIDATE_ID: int | None = 3958
SELECTED_SPLIT_CANDIDATE_ID: int | None = None
MARGINAL_FIELDS = (
    "num_planes",
    "sats_per_plane",
    "doe_stratum",
    "total_ground_station_count",
    "reduced_composition_ratio",
    "altitude_bin",
    "inclination_bin",
    "node_failure_bin",
    "edge_failure_bin",
    "ground_failure_bin",
)
BINS = {
    "altitude_bin": ((300.0, 525.0, False), (525.0, 750.0, False), (750.0, 975.0, False), (975.0, 1200.0, True)),
    "inclination_bin": ((30.0, 47.0, False), (47.0, 64.0, False), (64.0, 81.0, False), (81.0, 98.0, True)),
    "node_failure_bin": ((0.0, 0.05, False), (0.05, 0.10, False), (0.10, 0.15, False), (0.15, 0.20, True)),
    "edge_failure_bin": ((0.0, 0.0625, False), (0.0625, 0.125, False), (0.125, 0.1875, False), (0.1875, 0.25, True)),
    "ground_failure_bin": ((0.0, 0.10, False), (0.10, 0.20, False), (0.20, 0.30, False), (0.30, 0.40, True)),
}


def reduced_composition_ratio(record: dict[str, Any]) -> str:
    values = (
        record["civilian_count"],
        record["government_count"],
        record["military_count"],
    )
    divisor = math.gcd(values[0], math.gcd(values[1], values[2]))
    if divisor <= 0:
        raise ValueError("Composition counts must have a positive common divisor")
    return ":".join(str(value // divisor) for value in values)


def _bin_label(value: float, bins: tuple[tuple[float, float, bool], ...]) -> str:
    for index, (lower, upper, inclusive_upper) in enumerate(bins):
        if lower <= value < upper or inclusive_upper and lower <= value <= upper:
            return f"B{index}"
    raise ValueError(f"Value {value} is outside the locked split bins")


def marginal_values(record: dict[str, Any]) -> dict[str, str | int]:
    return {
        "num_planes": record["num_planes"],
        "sats_per_plane": record["sats_per_plane"],
        "doe_stratum": record["doe_stratum"],
        "total_ground_station_count": record["total_ground_station_count"],
        "reduced_composition_ratio": reduced_composition_ratio(record),
        "altitude_bin": _bin_label(float(record["altitude_km"]), BINS["altitude_bin"]),
        "inclination_bin": _bin_label(float(record["inclination_deg"]), BINS["inclination_bin"]),
        "node_failure_bin": _bin_label(
            float(record["satellite_node_failure_probability"]), BINS["node_failure_bin"]
        ),
        "edge_failure_bin": _bin_label(
            float(record["satellite_edge_failure_probability"]), BINS["edge_failure_bin"]
        ),
        "ground_failure_bin": _bin_label(
            float(record["ground_station_failure_probability"]), BINS["ground_failure_bin"]
        ),
    }


def _candidate_assignments(
    designs: tuple[dict[str, Any], ...], candidate_id: int
) -> dict[str, tuple[dict[str, Any], ...]]:
    ordered = sorted(
        designs,
        key=lambda record: (
            canonical_digest(
                split_candidate_payload(
                    candidate_id=candidate_id, design_id=record["design_id"]
                )
            ),
            record["design_id"],
        ),
    )
    return {
        "train": tuple(ordered[:SPLIT_SIZES["train"]]),
        "validation": tuple(
            ordered[SPLIT_SIZES["train"] : SPLIT_SIZES["train"] + SPLIT_SIZES["validation"]]
        ),
        "test": tuple(ordered[SPLIT_SIZES["train"] + SPLIT_SIZES["validation"] :]),
    }


def _meets_hard_requirements(assignments: dict[str, tuple[dict[str, Any], ...]]) -> bool:
    required_planes = {4, 5, 6}
    required_sats = {5, 6, 7, 8}
    required_strata = {"pilot_anchor", "transition", "global"}
    for records in assignments.values():
        if {record["num_planes"] for record in records} != required_planes:
            return False
        if {record["sats_per_plane"] for record in records} != required_sats:
            return False
        if {record["doe_stratum"] for record in records} != required_strata:
            return False
    anchor_counts = {
        split: sum(record["doe_stratum"] == "pilot_anchor" for record in records)
        for split, records in assignments.items()
    }
    return (
        anchor_counts["train"] >= 3
        and anchor_counts["validation"] >= 1
        and anchor_counts["test"] >= 1
    )


def _score(
    designs: tuple[dict[str, Any], ...],
    assignments: dict[str, tuple[dict[str, Any], ...]],
) -> tuple[Fraction, Fraction, Fraction]:
    """Score marginals with exact fractions and O(N) counting per candidate."""
    values_by_design = {
        record["design_id"]: marginal_values(record) for record in designs
    }
    counts_by_split = {
        split: {
            field: Counter(values_by_design[record["design_id"]][field] for record in assignments[split])
            for field in MARGINAL_FIELDS
        }
        for split in SPLIT_NAMES
    }
    total_design_count = sum(SPLIT_SIZES.values())
    normalized: list[Fraction] = []
    absolute: list[Fraction] = []
    for field in MARGINAL_FIELDS:
        full_counts = Counter(values[field] for values in values_by_design.values())
        for category, full_count in sorted(full_counts.items(), key=lambda item: str(item[0])):
            if full_count <= 0:
                raise RuntimeError("Split marginal category has no designs")
            for split in SPLIT_NAMES:
                expected = Fraction(SPLIT_SIZES[split] * full_count, total_design_count)
                actual = counts_by_split[split][field][category]
                difference = abs(Fraction(actual, 1) - expected)
                absolute.append(difference)
                normalized.append(difference / expected)
    return max(normalized), sum(value**2 for value in normalized), sum(absolute)


def _fraction_object(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def select_frozen_split(
    designs: Iterable[dict[str, Any]],
) -> tuple[
    tuple[Fraction, Fraction, Fraction],
    dict[str, tuple[dict[str, Any], ...]],
]:
    normalized = tuple(designs)
    if len(normalized) != sum(SPLIT_SIZES.values()):
        raise ValueError("Grouped split requires exactly 2,000 designs")
    valid_candidates: list[
        tuple[
            tuple[Fraction, Fraction, Fraction],
            int,
            dict[str, tuple[dict[str, Any], ...]],
        ]
    ] = []
    for candidate_id in range(SPLIT_CANDIDATE_COUNT):
        assignments = _candidate_assignments(normalized, candidate_id)
        if _meets_hard_requirements(assignments):
            valid_candidates.append((_score(normalized, assignments), candidate_id, assignments))
    if not valid_candidates:
        raise RuntimeError("No split candidate satisfies every hard requirement")
    score, candidate_id, assignments = min(
        valid_candidates, key=lambda item: (*item[0], item[1])
    )
    if FROZEN_SPLIT_CANDIDATE_ID is not None and candidate_id != FROZEN_SPLIT_CANDIDATE_ID:
        raise RuntimeError("Deterministic split selection differs from the frozen 10k candidate")
    global SELECTED_SPLIT_CANDIDATE_ID
    SELECTED_SPLIT_CANDIDATE_ID = candidate_id
    return score, assignments


def design_split_mapping(
    assignments: dict[str, tuple[dict[str, Any], ...]],
) -> dict[str, str]:
    return {
        record["design_id"]: split
        for split in SPLIT_NAMES
        for record in assignments[split]
    }


def build_split_manifest(
    *,
    designs: Iterable[dict[str, Any]],
    runs: Iterable[dict[str, Any]],
    contract_spec_hash: str,
    design_manifest_hash: str,
) -> dict[str, Any]:
    normalized_designs = tuple(designs)
    normalized_runs = tuple(runs)
    score, assignments = select_frozen_split(normalized_designs)
    candidate_id = SELECTED_SPLIT_CANDIDATE_ID
    if candidate_id is None:
        raise RuntimeError("Split candidate selection did not produce an identity")
    design_assignments = {
        split: sorted(record["design_id"] for record in assignments[split])
        for split in SPLIT_NAMES
    }
    design_split = {
        design_id: split
        for split, design_ids in design_assignments.items()
        for design_id in design_ids
    }
    if any(
        record.get("split_assignment") != design_split[record["design_id"]]
        for record in normalized_runs
    ):
        raise ValueError("Run split assignment differs from the frozen design split")
    run_assignments = {
        split: [
            record["run_id"]
            for record in normalized_runs
            if design_split[record["design_id"]] == split
        ]
        for split in SPLIT_NAMES
    }
    payload: dict[str, Any] = {
        "identity_domain": SPLIT_MANIFEST_IDENTITY_DOMAIN,
        "identity_version": SPLIT_MANIFEST_IDENTITY_VERSION,
        "contract_spec_hash": contract_spec_hash,
        "design_manifest_hash": design_manifest_hash,
        "strategy": "pre_outcome_grouped_marginal_balance",
        "candidate_count": SPLIT_CANDIDATE_COUNT,
        "selected_candidate_id": candidate_id,
        "selected_candidate_score": {
            "maximum_normalized_deviation": _fraction_object(score[0]),
            "sum_squared_normalized_deviation": _fraction_object(score[1]),
            "total_absolute_deviation": _fraction_object(score[2]),
        },
        "marginal_fields": list(MARGINAL_FIELDS),
        "design_assignments": design_assignments,
        "run_assignments": run_assignments,
        "outcome_fields_used": False,
    }
    result = dict(payload)
    result["split_manifest_hash"] = canonical_hash(payload)
    validate_split_manifest(result, designs=normalized_designs, runs=normalized_runs)
    return result


def validate_split_manifest(
    manifest: dict[str, Any],
    *,
    designs: Iterable[dict[str, Any]],
    runs: Iterable[dict[str, Any]],
) -> None:
    payload = {key: value for key, value in manifest.items() if key != "split_manifest_hash"}
    if canonical_hash(payload) != manifest.get("split_manifest_hash"):
        raise ValueError("Split-manifest hash mismatch")
    if manifest.get("outcome_fields_used") is not False:
        raise ValueError("Split manifest must be pre-outcome")
    if manifest.get("selected_candidate_id") != FROZEN_SPLIT_CANDIDATE_ID:
        raise ValueError("Split manifest candidate differs from the frozen 10k candidate")
    normalized_designs = tuple(designs)
    design_assignments = manifest["design_assignments"]
    if {split: len(design_assignments[split]) for split in SPLIT_NAMES} != SPLIT_SIZES:
        raise ValueError("Split design counts are invalid")
    all_design_ids = [design_id for split in SPLIT_NAMES for design_id in design_assignments[split]]
    expected_design_ids = [record["design_id"] for record in normalized_designs]
    if len(all_design_ids) != len(set(all_design_ids)) or set(all_design_ids) != set(expected_design_ids):
        raise ValueError("Split designs are not a disjoint complete partition")
    expected_assignments = _candidate_assignments(
        normalized_designs, manifest["selected_candidate_id"]
    )
    expected_design_assignments = {
        split: sorted(record["design_id"] for record in expected_assignments[split])
        for split in SPLIT_NAMES
    }
    if design_assignments != expected_design_assignments:
        raise ValueError("Split design assignments differ from deterministic reconstruction")
    expected_score = _score(normalized_designs, expected_assignments)
    persisted_score = manifest["selected_candidate_score"]
    if {
        "maximum_normalized_deviation": _fraction_object(expected_score[0]),
        "sum_squared_normalized_deviation": _fraction_object(expected_score[1]),
        "total_absolute_deviation": _fraction_object(expected_score[2]),
    } != persisted_score:
        raise ValueError("Split candidate score differs from deterministic reconstruction")
    design_split = {
        design_id: split
        for split in SPLIT_NAMES
        for design_id in design_assignments[split]
    }
    run_assignments = manifest["run_assignments"]
    if {split: len(run_assignments[split]) for split in SPLIT_NAMES} != SPLIT_RUN_SIZES:
        raise ValueError("Split run counts are invalid")
    normalized_runs = tuple(runs)
    if any(type(record.get("run_id")) is not int for record in normalized_runs):
        raise TypeError("Split run IDs must be exact integers")
    if any(
        type(run_id) is not int
        for split in SPLIT_NAMES
        for run_id in run_assignments[split]
    ):
        raise TypeError("Split assignments must contain exact integer run IDs")
    run_by_id = {record["run_id"]: record for record in normalized_runs}
    all_run_ids = [run_id for split in SPLIT_NAMES for run_id in run_assignments[split]]
    if len(all_run_ids) != len(set(all_run_ids)) or set(all_run_ids) != set(run_by_id):
        raise ValueError("Split runs are not a disjoint complete partition")
    for split in SPLIT_NAMES:
        if any(
            design_split[run_by_id[run_id]["design_id"]] != split
            or run_by_id[run_id].get("split_assignment") != split
            for run_id in run_assignments[split]
        ):
            raise ValueError("A design realization crossed split boundaries")
