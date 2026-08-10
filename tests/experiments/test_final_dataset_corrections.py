from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from satnet.experiments.final_dataset.design import validate_run_records
from satnet.experiments.final_dataset.materialize import read_json, read_jsonl
from satnet.experiments.final_dataset.specification import (
    build_contract_specification,
    build_later_generation_acceptance_gates,
    validate_contract_specification,
)
from satnet.ground.canonical import canonical_hash

ROOT = Path(__file__).parents[2]
OUTPUT = ROOT / "artifacts" / "final_integrated_dataset_10k_contract"


@pytest.fixture(scope="module")
def designs() -> tuple[dict[str, Any], ...]:
    return read_jsonl(OUTPUT / "designs.jsonl")


@pytest.fixture(scope="module")
def runs() -> tuple[dict[str, Any], ...]:
    return read_jsonl(OUTPUT / "runs.jsonl")


def _changed_runs(
    runs: tuple[dict[str, Any], ...],
    *,
    record_index: int,
    field: str,
    value: object,
) -> tuple[dict[str, Any], ...]:
    changed = [dict(record) for record in runs]
    changed[record_index][field] = value
    return tuple(changed)


def test_authoritative_numeric_run_identity_and_run_key(designs, runs) -> None:
    assert len({design["design_id"] for design in designs}) == 2000
    assert len(runs) == 10000
    assert [record["run_id"] for record in runs] == list(range(10000))
    assert {record["run_id"] for record in runs} == set(range(10000))
    assert all(type(record["run_id"]) is int for record in runs)
    assert all(type(record["run_key"]) is str for record in runs)
    assert all("run_index" not in record for record in runs)
    assert len({record["run_key"] for record in runs}) == 10000
    assert len({(record["design_id"], record["realization_id"]) for record in runs}) == 10000
    for record in runs:
        assert record["run_id"] == record["design_index"] * 5 + record["realization_index"]
        assert record["realization_id"] == f"R{record['realization_index']:02d}"
        assert record["run_key"] == f"{record['design_id']}-{record['realization_id']}"
    assert runs[0]["run_key"] == "D0000-R00"
    assert runs[4]["run_key"] == "D0000-R04"
    assert runs[5]["run_key"] == "D0001-R00"
    assert runs[-1]["run_key"] == "D1999-R04"


def test_split_manifest_uses_integer_run_ids_and_preserves_frozen_designs(runs) -> None:
    split = read_json(OUTPUT / "split_manifest.json")
    assert split["selected_candidate_id"] == 3958
    assert canonical_hash({"design_assignments": split["design_assignments"]}) == (
        "7a775735bd85981694c0fbbc5460c7ede7a0a9b236fffd5b65605367bf0e2b7b"
    )
    assigned = [
        run_id
        for split_name in ("train", "validation", "test")
        for run_id in split["run_assignments"][split_name]
    ]
    assert all(type(run_id) is int for run_id in assigned)
    assert set(assigned) == set(range(10000))
    run_by_id = {record["run_id"]: record for record in runs}
    for split_name in ("train", "validation", "test"):
        assert all(
            run_by_id[run_id]["split_assignment"] == split_name
            for run_id in split["run_assignments"][split_name]
        )


@pytest.mark.parametrize("value", [True, "0"])
def test_non_integer_run_id_fails(value, designs, runs) -> None:
    changed = _changed_runs(runs, record_index=0, field="run_id", value=value)
    with pytest.raises(TypeError, match="run_id must be an exact integer"):
        validate_run_records(changed, designs=designs)


@pytest.mark.parametrize("value", [-1, 10000])
def test_out_of_range_run_id_fails(value, designs, runs) -> None:
    changed = _changed_runs(runs, record_index=0, field="run_id", value=value)
    with pytest.raises(ValueError, match="run_id must be within"):
        validate_run_records(changed, designs=designs)


def test_missing_run_id_fails(designs, runs) -> None:
    changed = [dict(record) for record in runs]
    del changed[0]["run_id"]
    with pytest.raises(ValueError, match="fields do not match"):
        validate_run_records(changed, designs=designs)


def test_duplicate_numeric_run_id_fails(designs, runs) -> None:
    changed = _changed_runs(runs, record_index=1, field="run_id", value=0)
    with pytest.raises(ValueError, match="cover ordered integers"):
        validate_run_records(changed, designs=designs)


def test_duplicate_run_key_fails(designs, runs) -> None:
    changed = _changed_runs(
        runs, record_index=1, field="run_key", value=runs[0]["run_key"]
    )
    with pytest.raises(ValueError, match="Run keys must be unique"):
        validate_run_records(changed, designs=designs)


def test_non_string_run_key_fails(designs, runs) -> None:
    changed = _changed_runs(runs, record_index=0, field="run_key", value=0)
    with pytest.raises(TypeError, match="run_key must be a string"):
        validate_run_records(changed, designs=designs)


def test_later_generation_acceptance_gates_are_complete() -> None:
    gates = build_later_generation_acceptance_gates()
    required_values = {
        "expected_run_count": 10000,
        "required_generation_attempt_count": 10000,
        "required_successful_generation_count": 10000,
        "required_authoritative_replay_count": 10000,
        "required_successful_replay_count": 10000,
        "allow_seed_substitution": False,
        "allow_run_omission": False,
        "allow_replacement_runs": False,
        "require_failure_evidence_preservation": True,
        "require_all_numeric_targets_finite": True,
        "require_all_target_fractions_in_unit_interval": True,
        "require_primary_classification_both_classes_per_split": True,
        "require_primary_regression_nonzero_standard_deviation_per_split": True,
        "require_primary_regression_minimum_unique_values_per_split": 5,
        "allow_outcome_driven_split_reshuffle": False,
        "require_frozen_split_manifest": True,
        "require_all_five_realizations_colocated_by_design": True,
        "require_zero_missing_stage_artifacts": True,
        "require_zero_duplicate_run_ids": True,
        "require_zero_duplicate_design_realization_pairs": True,
        "require_exact_g1_g5_replay": True,
        "require_protected_science_diff_empty": True,
    }
    for field, expected in required_values.items():
        assert gates[field] == expected
    assert gates["count_semantics"] == {
        "expected_run_count": "records_in_frozen_run_manifest",
        "required_generation_attempt_count": "frozen_runs_submitted_to_generation",
        "required_successful_generation_count": "frozen_runs_successfully_generated",
        "required_authoritative_replay_count": "frozen_runs_submitted_to_authoritative_replay",
        "required_successful_replay_count": "frozen_runs_successfully_replayed",
    }
    assert gates["failure_behavior"] == {
        "preserve_failed_run_evidence": True,
        "keep_frozen_run_manifest_unchanged": True,
        "dataset_status_on_any_failure": "incomplete",
        "retry_requires_same_run_id_and_frozen_seeds": True,
    }
    assert gates["numeric_target_validity"] == {
        "reject_nan": True,
        "reject_positive_infinity": True,
        "reject_negative_infinity": True,
        "fraction_interval": ["0", "1"],
        "fraction_interval_closed": True,
    }
    assert gates["primary_classification_gate"] == {
        "target_field": "overall_threshold_breach_any",
        "required_splits": ["train", "validation", "test"],
        "required_values_per_split": [False, True],
        "failure_action": "dataset_acceptance_failed_pending_explicit_contract_version_review",
        "allow_threshold_tuning": False,
    }
    assert gates["primary_regression_gate"] == {
        "target_field": "failure_adjusted_overall_service_fraction_mean",
        "required_splits": ["train", "validation", "test"],
        "standard_deviation_semantics": "population_standard_deviation_over_finite_binary64_values",
        "require_nonzero_standard_deviation": True,
        "unique_value_semantics": "exact_binary64_equality_after_finite_parse",
        "minimum_unique_values_per_split": 5,
    }
    assert gates["split_immutability"] == {
        "selected_candidate_id": 3958,
        "outcomes_or_labels_may_modify_assignments": False,
        "all_realizations_grouped_by_design": True,
    }
    assert gates["required_stage_artifacts"] == [
        "satellite_rollout",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
    ]


def test_contract_separates_contract_phase_and_generation_gates() -> None:
    specification = build_contract_specification()
    assert "acceptance_gates" not in specification
    assert specification["contract_phase_validation_gates"]["simulation_artifacts_required"] is False
    assert specification["later_generation_acceptance_gates"] == (
        build_later_generation_acceptance_gates()
    )
    assert specification["run_identity"]["authoritative_field"] == "run_id"
    assert specification["run_identity"]["run_index_present"] is False


def test_required_gate_changes_specification_and_bundle_hashes() -> None:
    specification = build_contract_specification()
    bundle = read_json(OUTPUT / "contract_bundle.json")
    baseline_bundle_payload = {
        key: value for key, value in bundle.items() if key != "contract_bundle_hash"
    }
    mutations = (
        ("allow_seed_substitution", True),
        ("required_successful_replay_count", 499),
    )
    for field, value in mutations:
        changed_payload = deepcopy(
            {key: item for key, item in specification.items() if key != "contract_spec_hash"}
        )
        changed_payload["later_generation_acceptance_gates"][field] = value
        changed_spec_hash = canonical_hash(changed_payload)
        assert changed_spec_hash != specification["contract_spec_hash"]
        changed_specification = dict(changed_payload)
        changed_specification["contract_spec_hash"] = changed_spec_hash
        with pytest.raises(ValueError, match="Later-generation acceptance gates"):
            validate_contract_specification(changed_specification)
        changed_bundle_payload = dict(baseline_bundle_payload)
        changed_bundle_payload["contract_spec_hash"] = changed_spec_hash
        assert canonical_hash(changed_bundle_payload) != bundle["contract_bundle_hash"]


def test_removing_required_gate_changes_specification_and_bundle_hashes() -> None:
    specification = build_contract_specification()
    changed_payload = deepcopy(
        {key: item for key, item in specification.items() if key != "contract_spec_hash"}
    )
    del changed_payload["later_generation_acceptance_gates"]["allow_run_omission"]
    changed_spec_hash = canonical_hash(changed_payload)
    assert changed_spec_hash != specification["contract_spec_hash"]
    bundle = read_json(OUTPUT / "contract_bundle.json")
    changed_bundle_payload = {
        key: value for key, value in bundle.items() if key != "contract_bundle_hash"
    }
    changed_bundle_payload["contract_spec_hash"] = changed_spec_hash
    assert canonical_hash(changed_bundle_payload) != bundle["contract_bundle_hash"]
