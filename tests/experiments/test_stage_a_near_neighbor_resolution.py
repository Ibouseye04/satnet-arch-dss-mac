from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from satnet.experiments.stage_a_contract.designs import (
    build_design_rows,
    minimum_distances,
    scientific_signature,
    validate_design_rows,
)
from satnet.experiments.stage_a_contract.proposal import (
    APPROVED_EXCEPTION_STATUS,
    build_artifact_payloads,
    build_inventory,
    build_near_neighbor_policy,
    build_run_rows,
    build_seed_rows,
    load_original_designs,
    validate_near_neighbor_separation,
)
from satnet.experiments.stage_a_contract.semantics import (
    OUTPUT_ROOTS,
    canonical_float,
    canonical_json_bytes,
    normalized_distance,
)

ROOT = Path(__file__).parents[2]
PRIOR_DISTANCE = 0.07681919236933395
CORRECTED_DISTANCE = 0.12039492645571381


def _by_id(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["design_id"]): row for row in rows}


def _prior_designs() -> list[dict[str, object]]:
    rows = deepcopy(build_design_rows())
    rows[20]["ground_station_failure_probability"] = canonical_float(0.075)
    return rows


def test_original_blocker_and_corrected_distance_are_exactly_reproduced() -> None:
    prior = _by_id(_prior_designs())
    corrected = _by_id(build_design_rows())
    assert prior["SA-D013"]["partition"] == "development"
    assert prior["SA-D020"]["partition"] == "validation"
    assert normalized_distance(prior["SA-D013"], prior["SA-D020"]) == pytest.approx(PRIOR_DISTANCE, abs=1e-15)
    assert normalized_distance(corrected["SA-D013"], corrected["SA-D020"]) == pytest.approx(CORRECTED_DISTANCE, abs=1e-15)
    assert corrected["SA-D020"]["ground_station_failure_probability"] == canonical_float(0.100)


def test_corrected_cross_partition_and_holdout_separation_passes() -> None:
    designs = build_design_rows()
    distances = minimum_distances(designs)
    assert distances["minimum_development_validation_distance"] >= 0.10
    assert distances["minimum_development_holdout_distance"] >= 0.10
    assert distances["minimum_validation_holdout_distance"] >= 0.10
    validate_near_neighbor_separation(designs)


def test_resolution_preserves_counts_identities_regions_partitions_and_seeds() -> None:
    designs = build_design_rows()
    runs = build_run_rows(designs)
    seeds = build_seed_rows(runs)
    assert len(designs) == 30
    assert len(runs) == 150
    assert len(seeds) == 150
    assert Counter(row["region"] for row in designs) == {"resilient_core": 12, "boundary": 12, "global_control": 6}
    assert Counter(row["partition"] for row in designs) == {"development": 20, "validation": 5, "sealed_holdout": 5}
    assert [row["design_id"] for row in designs] == [f"SA-D{index:03d}" for index in range(30)]
    assert [row["global_run_id"] for row in runs] == list(range(500, 650))
    assert build_seed_rows(build_run_rows(build_design_rows())) == seeds
    assert hashlib.sha256(build_artifact_payloads(ROOT)["stage_a_seed_manifest.csv"]).hexdigest() == "ab37564cca81e926b6e43caf74f1ff2be641b8bc203e32c3ae6a219ea6f7bace"


def test_resolution_has_no_duplicate_or_original_collision() -> None:
    original = load_original_designs(ROOT / "artifacts/final_integrated_dataset_contract/designs.jsonl")
    designs = build_design_rows()
    validate_design_rows(designs, original)
    assert len({scientific_signature(row) for row in designs}) == 30
    assert not ({scientific_signature(row) for row in designs} & {scientific_signature(row) for row in original})


def test_policy_records_resolved_review_candidates_and_no_pending_review() -> None:
    original = load_original_designs(ROOT / "artifacts/final_integrated_dataset_contract/designs.jsonl")
    policy = build_near_neighbor_policy(build_design_rows(), original)
    assert policy["proposal_status"] == "NOT_FROZEN"
    assert policy["simulation_authorized"] is False
    assert policy["pending_scientific_reviews"] == []
    assert policy["justified_exceptions"] == []
    assert len(policy["resolved_scientific_reviews"]) == 1
    review = policy["resolved_scientific_reviews"][0]
    assert review["prior_normalized_distance"] == pytest.approx(PRIOR_DISTANCE, abs=1e-15)
    assert review["corrected_normalized_distance"] == pytest.approx(CORRECTED_DISTANCE, abs=1e-15)
    candidates = policy["candidate_review"]["ranked_admissible_alternatives"]
    assert len(candidates) == 10
    assert candidates[0]["selected"] is True
    assert candidates[0]["candidate_values"] == {"ground_station_failure_probability": canonical_float(0.100)}
    assert all(candidate["region_bound_result"] == "PASS" for candidate in candidates)
    assert all(candidate["duplicate_result"] == "PASS" for candidate in candidates)
    assert all(candidate["minimum_development_distance"] >= 0.11 for candidate in candidates)
    assert all(candidate["minimum_sealed_holdout_distance"] >= 0.10 for candidate in candidates)


def test_subthreshold_development_validation_pair_requires_exact_valid_exception() -> None:
    prior = _prior_designs()
    by_id = _by_id(prior)
    distance = normalized_distance(by_id["SA-D013"], by_id["SA-D020"])
    with pytest.raises(ValueError, match="exact explicit exception"):
        validate_near_neighbor_separation(prior)
    exception = {
        "first_design_id": "SA-D013",
        "second_design_id": "SA-D020",
        "normalized_distance": distance,
        "review_status": APPROVED_EXCEPTION_STATUS,
        "scientific_justification": "Recorded test-only scientific justification.",
        "approval_reference": "TEST-ONLY-APPROVAL",
    }
    validate_near_neighbor_separation(prior, [exception])
    invalid = deepcopy(exception)
    invalid["approval_reference"] = ""
    with pytest.raises(ValueError, match="approval reference"):
        validate_near_neighbor_separation(prior, [invalid])


def test_subthreshold_holdout_pair_is_never_exception_eligible() -> None:
    designs = build_design_rows()
    by_id = _by_id(designs)
    holdout = by_id["SA-D022"]
    validation = by_id["SA-D020"]
    for field in (
        "num_planes",
        "sats_per_plane",
        "altitude_km",
        "inclination_deg",
        "satellite_node_failure_probability",
        "satellite_edge_failure_probability",
        "civilian_count",
        "government_count",
        "military_count",
        "total_ground_station_count",
        "ground_station_failure_probability",
    ):
        holdout[field] = validation[field]
    with pytest.raises(ValueError, match="sealed holdout"):
        validate_near_neighbor_separation(designs)


def test_design_run_and_inventory_hashes_regenerate_deterministically() -> None:
    first_designs = build_design_rows()
    second_designs = build_design_rows()
    assert [row["design_parameter_hash"] for row in first_designs] == [row["design_parameter_hash"] for row in second_designs]
    assert [row["design_record_hash"] for row in first_designs] == [row["design_record_hash"] for row in second_designs]
    first_runs = build_run_rows(first_designs)
    second_runs = build_run_rows(second_designs)
    assert [row["run_record_hash"] for row in first_runs] == [row["run_record_hash"] for row in second_runs]
    first_payloads = build_artifact_payloads(ROOT)
    second_payloads = build_artifact_payloads(ROOT)
    assert canonical_json_bytes(build_inventory(first_payloads)) == canonical_json_bytes(build_inventory(second_payloads))


def test_proposal_remains_unauthorized_and_external_roots_absent() -> None:
    payloads = build_artifact_payloads(ROOT)
    assert b'"proposal_status": "NOT_FROZEN"' in payloads["stage_a_contract_proposal.json"]
    assert b'"simulation_authorized": false' in payloads["stage_a_contract_proposal.json"]
    assert all(not Path(path).exists() for path in OUTPUT_ROOTS.values())
