from __future__ import annotations

from fractions import Fraction
import math

import pytest

from satnet.experiments.final_dataset.deterministic import (
    _jitter,
    _lhs_candidate,
    _lhs_score,
    allocate_ground_classes,
    canonical_digest,
    derived_seed,
    ground_failure_seed_payload,
    ground_selection_seed_payload,
    lhs_jitter_payload,
    lhs_permutation_payload,
    parse_canonical_float,
    satellite_seed_payload,
    schedule_pairing_payload,
    split_candidate_payload,
)
from satnet.experiments.final_dataset.specification import CANONICAL_DIMENSION_ORDER
from satnet.ground.canonical import canonical_float_string, canonical_json


def test_canonical_float_examples_and_strict_reader() -> None:
    expected = {
        600.0: "600",
        55.0: "55",
        0.05: "0.050000000000000003",
        0.8: "0.80000000000000004",
    }
    for value, text in expected.items():
        assert canonical_float_string(value) == text
        assert parse_canonical_float(text, "value") == value
    with pytest.raises(ValueError, match="not canonical"):
        parse_canonical_float("600.0", "value")
    for text in ("nan", "inf", "-inf"):
        with pytest.raises(ValueError, match="finite"):
            parse_canonical_float(text, "value")
    with pytest.raises(ValueError, match="positive zero"):
        parse_canonical_float("-0", "value")
    with pytest.raises(TypeError, match="canonical float string"):
        parse_canonical_float(600.0, "value")


def test_seed_and_digest_golden_vectors() -> None:
    cases = (
        (
            ground_selection_seed_payload("D000"),
            "15d84731821505f8fa48c8bce17f5637549d51908dd8508afda931e107a94bf3",
            9054823368371031027,
        ),
        (
            satellite_seed_payload("D000", "R01"),
            "24e1fe32a6364a745651c89a026fa59d2a531c60f29520df98b602a0fe20599e",
            1780613593153165726,
        ),
        (
            ground_failure_seed_payload("D000", "R01"),
            "e9398c741d333c08e4e3aa647088a32c5bd701c8fd26b6cf8f92cdccd3429e72",
            1122185536770055794,
        ),
    )
    for payload, digest, seed in cases:
        assert canonical_digest(payload).hex() == digest
        assert derived_seed(payload) == seed
        assert 0 <= seed < 2**63
    digest_cases = (
        (
            lhs_permutation_payload(
                stratum_id="transition",
                candidate_id=0,
                dimension_name="altitude_km",
                row_index=0,
            ),
            "7a67fd34932867869c325c2715eba0e5343760e60407b8920880d4790e88a951",
        ),
        (
            lhs_jitter_payload(
                stratum_id="transition",
                candidate_id=0,
                dimension_name="altitude_km",
                row_index=0,
            ),
            "92fe1c1895aa2c008b6d96b19106b8dd79390a3e1bb6f0347baf749a79300080",
        ),
        (
            schedule_pairing_payload(
                stratum_id="transition",
                schedule_purpose="continuous_rows",
                record_index=0,
            ),
            "0a7ea6344701d1a8e7511ccb56fe5346b562283c49cb18f3ff3a90a3f1629f44",
        ),
        (
            split_candidate_payload(candidate_id=0, design_id="D000"),
            "0307fd48dd37131924474b10740882f8029f2b6bb4b53ee84fdc949aeaa14746",
        ),
    )
    for payload, digest in digest_cases:
        assert canonical_digest(payload).hex() == digest
        assert canonical_json(payload).encode("utf-8")


def test_ground_allocation_uses_minimum_first_exact_remainders() -> None:
    assert allocate_ground_classes(6, (8, 1, 1)) == (4, 1, 1)
    assert allocate_ground_classes(20, (1, 1, 1)) == (7, 7, 6)
    assert allocate_ground_classes(50, (8, 1, 1)) == (38, 6, 6)
    assert allocate_ground_classes(6, (9, 9, 2)) == (3, 2, 1)
    with pytest.raises(ValueError):
        allocate_ground_classes(2, (1, 1, 1))
    with pytest.raises(ValueError):
        allocate_ground_classes(6, (1, 0, 1))


def test_lhs_rank_semantics_and_canonical_dimension_order() -> None:
    assert CANONICAL_DIMENSION_ORDER == (
        "altitude_km",
        "inclination_deg",
        "satellite_node_failure_probability",
        "satellite_edge_failure_probability",
        "ground_station_failure_probability",
    )
    rows = _lhs_candidate(7, "transition", 0)
    for dimension in CANONICAL_DIMENSION_ORDER:
        strata = sorted(math.floor(row[dimension] * 7) for row in rows)
        assert strata == list(range(7))
        assert all(0.0 <= row[dimension] < 1.0 for row in rows)


def test_jitter_uses_most_significant_53_bits() -> None:
    assert _jitter(bytes(32)) == 0.0
    expected = (2**53 - 1) / 2**53
    assert _jitter(bytes([255]) * 32) == expected
    digest = bytes.fromhex(
        "92fe1c1895aa2c008b6d96b19106b8dd79390a3e1bb6f0347baf749a79300080"
    )
    digest_int = int.from_bytes(digest, "big", signed=False)
    assert _jitter(digest) == (digest_int >> 203) / 2**53


def test_lhs_pair_enumeration_and_score() -> None:
    rows = []
    for altitude in (0.0, 1.0, 2.0):
        row = {dimension: 0.0 for dimension in CANONICAL_DIMENSION_ORDER}
        row["altitude_km"] = altitude
        rows.append(row)
    minimum, mean = _lhs_score(rows)
    assert minimum == 1.0
    assert mean == math.fsum((1.0, 2.0, 1.0)) / 3


def test_fraction_arithmetic_example_is_exact() -> None:
    expected = Fraction(15 * 7, 100)
    actual = Fraction(2, 1)
    normalized = abs(actual - expected) / expected
    assert normalized == Fraction(19, 21)
