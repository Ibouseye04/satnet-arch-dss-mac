from __future__ import annotations

from fractions import Fraction
import hashlib
import math
from typing import Any, Sequence, TypeVar

import numpy as np

from satnet.experiments.final_dataset.specification import (
    CANONICAL_DIMENSION_ORDER,
    CONTRACT_MASTER_SEED,
    SEED_MODULUS,
    SPLIT_MASTER_SEED,
)
from satnet.ground.canonical import canonical_float_string, canonical_json

T = TypeVar("T")


def parse_canonical_float(value: object, field_name: str) -> float:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a canonical float string")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid float string") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    if parsed == 0.0 and value.startswith("-"):
        raise ValueError(f"{field_name} must use positive zero")
    if canonical_float_string(parsed) != value:
        raise ValueError(f"{field_name} is not canonical")
    return parsed


def canonical_digest(payload: dict[str, Any]) -> bytes:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()


def derived_seed(payload: dict[str, Any]) -> int:
    return int.from_bytes(canonical_digest(payload), byteorder="big", signed=False) % SEED_MODULUS


def ground_selection_seed_payload(design_id: str) -> dict[str, Any]:
    return {
        "contract_master_seed": CONTRACT_MASTER_SEED,
        "design_id": design_id,
        "identity_domain": "satnet_final_dataset_ground_selection_seed",
        "identity_version": "1",
    }


def satellite_seed_payload(design_id: str, realization_id: str) -> dict[str, Any]:
    return {
        "contract_master_seed": CONTRACT_MASTER_SEED,
        "design_id": design_id,
        "identity_domain": "satnet_final_dataset_satellite_seed",
        "identity_version": "1",
        "realization_id": realization_id,
        "seed_purpose": "satellite_rollout_and_failure",
    }


def ground_failure_seed_payload(design_id: str, realization_id: str) -> dict[str, Any]:
    return {
        "contract_master_seed": CONTRACT_MASTER_SEED,
        "design_id": design_id,
        "identity_domain": "satnet_final_dataset_ground_failure_seed",
        "identity_version": "1",
        "realization_id": realization_id,
    }


def lhs_permutation_payload(
    *, stratum_id: str, candidate_id: int, dimension_name: str, row_index: int
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "contract_master_seed": CONTRACT_MASTER_SEED,
        "dimension_name": dimension_name,
        "identity_domain": "satnet_final_dataset_lhs_permutation",
        "identity_version": "1",
        "row_index": row_index,
        "stratum_id": stratum_id,
    }


def lhs_jitter_payload(
    *, stratum_id: str, candidate_id: int, dimension_name: str, row_index: int
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "contract_master_seed": CONTRACT_MASTER_SEED,
        "dimension_name": dimension_name,
        "identity_domain": "satnet_final_dataset_lhs_jitter",
        "identity_version": "1",
        "row_index": row_index,
        "stratum_id": stratum_id,
    }


def schedule_pairing_payload(
    *, stratum_id: str, schedule_purpose: str, record_index: int
) -> dict[str, Any]:
    allowed = {"continuous_rows", "ground_schedule", "satellite_pair_schedule"}
    if schedule_purpose not in allowed:
        raise ValueError(f"Unsupported schedule purpose: {schedule_purpose}")
    return {
        "contract_master_seed": CONTRACT_MASTER_SEED,
        "identity_domain": "satnet_final_dataset_schedule_pairing",
        "identity_version": "1",
        "record_index": record_index,
        "schedule_purpose": schedule_purpose,
        "stratum_id": stratum_id,
    }


def split_candidate_payload(*, candidate_id: int, design_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "design_id": design_id,
        "identity_domain": "satnet_final_dataset_split_candidate",
        "identity_version": "1",
        "split_master_seed": SPLIT_MASTER_SEED,
    }


def allocate_ground_classes(total: int, weights: tuple[int, int, int]) -> tuple[int, int, int]:
    if type(total) is not int or total < 3:
        raise ValueError("total must be an integer of at least three")
    if len(weights) != 3 or any(type(weight) is not int or weight <= 0 for weight in weights):
        raise ValueError("weights must contain three positive integers")
    weight_sum = sum(weights)
    remaining = total - 3
    quotas = tuple(Fraction(remaining * weight, weight_sum) for weight in weights)
    floors = [quota.numerator // quota.denominator for quota in quotas]
    unassigned = remaining - sum(floors)
    remainder_order = sorted(
        range(3),
        key=lambda index: (-(quotas[index] - floors[index]), index),
    )
    for index in remainder_order[:unassigned]:
        floors[index] += 1
    result = tuple(value + 1 for value in floors)
    if sum(result) != total or any(value < 1 or value > 50 for value in result):
        raise RuntimeError("Ground allocation violated the contract")
    return result


def deterministic_order(
    values: Sequence[T], *, stratum_id: str, schedule_purpose: str
) -> tuple[T, ...]:
    indexed = list(enumerate(values))
    indexed.sort(
        key=lambda item: (
            canonical_digest(
                schedule_pairing_payload(
                    stratum_id=stratum_id,
                    schedule_purpose=schedule_purpose,
                    record_index=item[0],
                )
            ),
            item[0],
        )
    )
    return tuple(value for _, value in indexed)


def _jitter(digest: bytes) -> float:
    digest_int = int.from_bytes(digest, byteorder="big", signed=False)
    value = digest_int >> (256 - 53)
    if not 0 <= value < 2**53:
        raise RuntimeError("Jitter integer is outside its 53-bit range")
    result = value / 2**53
    if not 0.0 <= result < 1.0:
        raise RuntimeError("Jitter is outside [0, 1)")
    return result


def _lhs_candidate(row_count: int, stratum_id: str, candidate_id: int) -> tuple[dict[str, float], ...]:
    rows = [{dimension: 0.0 for dimension in CANONICAL_DIMENSION_ORDER} for _ in range(row_count)]
    for dimension in CANONICAL_DIMENSION_ORDER:
        ordered_indices = sorted(
            range(row_count),
            key=lambda row_index: (
                canonical_digest(
                    lhs_permutation_payload(
                        stratum_id=stratum_id,
                        candidate_id=candidate_id,
                        dimension_name=dimension,
                        row_index=row_index,
                    )
                ),
                row_index,
            ),
        )
        rank_by_row = {row_index: rank for rank, row_index in enumerate(ordered_indices)}
        for row_index in range(row_count):
            jitter = _jitter(
                canonical_digest(
                    lhs_jitter_payload(
                        stratum_id=stratum_id,
                        candidate_id=candidate_id,
                        dimension_name=dimension,
                        row_index=row_index,
                    )
                )
            )
            rows[row_index][dimension] = (rank_by_row[row_index] + jitter) / row_count
    return tuple(rows)


def _lhs_score(rows: Sequence[dict[str, float]]) -> tuple[float, float]:
    """Score one LHS candidate without Python-level O(N²) pair iteration.

    The candidate rows, dimension ordering, Euclidean metric, upper-triangle pair
    order, and min/mean score semantics are unchanged. NumPy performs the same
    pairwise arithmetic in vectorized form so the 1,600-row contract remains
    materializable without weakening the 256-candidate search.
    """
    values = np.asarray(
        [[row[dimension] for dimension in CANONICAL_DIMENSION_ORDER] for row in rows],
        dtype=np.float64,
    )
    if values.ndim != 2 or values.shape[1] != len(CANONICAL_DIMENSION_ORDER):
        raise RuntimeError("LHS candidate has an invalid shape")
    squared_norms = np.einsum("ij,ij->i", values, values)
    squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (values @ values.T)
    upper = np.triu_indices(len(rows), k=1)
    distances = np.sqrt(np.maximum(squared_distances[upper], 0.0))
    if not np.all(np.isfinite(distances)):
        raise RuntimeError("LHS distance is not finite")
    minimum = float(np.min(distances))
    mean = float(math.fsum(float(value) for value in distances) / len(distances))
    if not math.isfinite(minimum) or not math.isfinite(mean):
        raise RuntimeError("LHS score is not finite")
    return minimum, mean


def select_lhs(
    *, row_count: int, stratum_id: str, ranges: dict[str, tuple[float, float]]
) -> tuple[tuple[dict[str, float], ...], dict[str, Any]]:
    if tuple(ranges) != CANONICAL_DIMENSION_ORDER:
        raise ValueError("LHS ranges must use canonical dimension order")
    candidates: list[tuple[float, float, int, tuple[dict[str, float], ...]]] = []
    for candidate_id in range(256):
        unit_rows = _lhs_candidate(row_count, stratum_id, candidate_id)
        minimum, mean = _lhs_score(unit_rows)
        candidates.append((minimum, mean, candidate_id, unit_rows))
    minimum, mean, candidate_id, unit_rows = max(
        candidates, key=lambda item: (item[0], item[1], -item[2])
    )
    physical_rows: list[dict[str, float]] = []
    for unit_row in unit_rows:
        row: dict[str, float] = {}
        for dimension in CANONICAL_DIMENSION_ORDER:
            lower, upper = ranges[dimension]
            value = lower + unit_row[dimension] * (upper - lower)
            if not math.isfinite(value):
                raise RuntimeError("Mapped LHS value is not finite")
            row[dimension] = 0.0 if value == 0.0 else value
        physical_rows.append(row)
    evidence = {
        "candidate_id": candidate_id,
        "minimum_pairwise_distance": canonical_float_string(minimum),
        "mean_pairwise_distance": canonical_float_string(mean),
        "row_count": row_count,
        "stratum_id": stratum_id,
    }
    return tuple(physical_rows), evidence
