from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

import scripts.validation.audit_stage_a_discovery_contract as audit

ROOT = Path(__file__).parents[2]
ARTIFACT_ROOT = ROOT / "artifacts/stage_a_discovery_contract_proposal"


@pytest.fixture()
def snapshot() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    designs = audit.read_csv_rows(ARTIFACT_ROOT / "stage_a_design_manifest.csv", "design")
    runs = audit.read_csv_rows(ARTIFACT_ROOT / "stage_a_run_manifest.csv", "run")
    seeds = audit.read_csv_rows(ARTIFACT_ROOT / "stage_a_seed_manifest.csv", "seed")
    original = audit.read_jsonl(ROOT / "artifacts/final_integrated_dataset_contract/designs.jsonl")
    bounds = audit.read_json(ARTIFACT_ROOT / "stage_a_region_bounds.json")
    return designs, runs, seeds, original, bounds


def test_exact_manifest_cardinality_identity_and_allocations(snapshot) -> None:
    designs, runs, seeds, original, bounds = snapshot
    audit.validate_snapshot(designs, runs, seeds, original, bounds)
    assert len(designs) == 30
    assert len(runs) == 150
    assert len(seeds) == 150
    assert [row["design_id"] for row in designs] == [f"SA-D{index:03d}" for index in range(30)]
    assert [row["design_index"] for row in designs] == list(range(30))
    assert [row["global_run_id"] for row in runs] == list(range(500, 650))
    assert Counter(row["region"] for row in designs) == {"resilient_core": 12, "boundary": 12, "global_control": 6}
    assert Counter(row["partition"] for row in designs) == {"development": 20, "validation": 5, "sealed_holdout": 5}
    assert Counter((row["region"], row["partition"]) for row in designs) == {
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
    assert all(sum(row["design_id"] == design["design_id"] for row in runs) == 5 for design in designs)


def test_independent_hash_seed_and_artifact_reproduction(snapshot) -> None:
    designs, runs, seeds, original, bounds = snapshot
    contract_specification = audit.read_json(ROOT / "artifacts/final_integrated_dataset_contract/contract_specification.json")
    design_rows = audit.audit_designs(designs, original, bounds, contract_specification)
    run_rows = audit.audit_runs(runs, designs)
    seed_rows = audit.audit_seeds(seeds, {row["run_key"] for row in runs})
    assert all(row["overall_pass"] for row in design_rows)
    assert all(row["overall_pass"] for row in run_rows)
    assert all(row["overall_pass"] for row in seed_rows)
    assert audit.sha256_file(ARTIFACT_ROOT / "stage_a_seed_manifest.csv") == audit.SEED_MANIFEST_SHA256
    reproduction = audit.artifact_reproduction(ROOT, ARTIFACT_ROOT)
    assert reproduction["artifact_count"] == 11
    assert reproduction["byte_identical_count"] == 11
    assert reproduction["all_byte_identical"] is True
    assert reproduction["proposal_inventory_sha256"] == audit.PROPOSAL_INVENTORY_SHA256


def test_sa_d020_correction_and_all_neighbor_minima(snapshot) -> None:
    designs, _, seeds, original, _ = snapshot
    prior = audit._csv_rows_from_bytes(
        audit._git_show(ROOT, audit.STARTING_COMMIT, "artifacts/stage_a_discovery_contract_proposal/stage_a_design_manifest.csv"),
        "design",
    )
    matrix, minima = audit.build_neighbor_audit(designs, original, prior)
    current_by_id = {row["design_id"]: row for row in designs}
    prior_by_id = {row["design_id"]: row for row in prior}
    assert len(matrix) == 3435
    assert prior_by_id["SA-D020"]["ground_station_failure_probability"] == "0.074999999999999997"
    assert current_by_id["SA-D020"]["ground_station_failure_probability"] == "0.10000000000000001"
    assert minima["original_sa_d013_sa_d020_distance"] == pytest.approx(0.07681919236933395, abs=1e-15)
    assert minima["corrected_sa_d013_sa_d020_distance"] == pytest.approx(0.12039492645571381, abs=1e-15)
    assert minima["minimum_development_validation"]["distance"] == pytest.approx(0.12039492645571381, abs=1e-15)
    assert minima["minimum_development_holdout"]["distance"] == pytest.approx(0.1205683310788027, abs=1e-15)
    assert minima["minimum_validation_holdout"]["distance"] == pytest.approx(0.1500946005884104, abs=1e-15)
    assert minima["minimum_stage_a_to_original"]["distance"] == pytest.approx(0.06382978723404255, abs=1e-15)
    assert minima["cross_partition_pairs_below_0_10"] == 0
    assert minima["holdout_pairs_below_0_10"] == 0
    starting_seed_bytes = audit._git_show(ROOT, audit.STARTING_COMMIT, "artifacts/stage_a_discovery_contract_proposal/stage_a_seed_manifest.csv")
    assert starting_seed_bytes == (ARTIFACT_ROOT / "stage_a_seed_manifest.csv").read_bytes()
    assert len(seeds) == 150


def test_holdout_final_corpus_boundary_discovery_and_decision_policies() -> None:
    contract = audit.read_json(ARTIFACT_ROOT / "stage_a_contract_proposal.json")
    partition = audit.read_json(ARTIFACT_ROOT / "stage_a_partition_manifest.json")
    criteria = audit.read_json(ARTIFACT_ROOT / "stage_a_discovery_criteria.json")
    neighbor = audit.read_json(ARTIFACT_ROOT / "stage_a_near_neighbor_policy.json")
    result = audit.policy_audits(contract, partition, criteria, neighbor)
    assert all(result["holdout"].values())
    assert all(result["final_corpus"].values())
    assert result["stage_b"]["development_and_validation_only"] is True
    assert result["stage_b"]["holdout_prohibited"] is True
    assert result["stage_b"]["simulation_authorized"] is False
    assert result["boundary"]["preassigned_and_observed_separated"] is True
    assert result["boundary"]["rounding_mode"] == "ROUND_HALF_EVEN"
    assert result["discovery"]["final_classification_gates_apply"] is False
    assert result["decision"]["decision_states_exact"] is True
    assert result["decision"]["semantic_order_complete"] is True
    assert result["near_neighbor"]["pending_scientific_reviews"] == []
    assert result["near_neighbor"]["justified_exceptions"] == []


@pytest.mark.parametrize("non_breach_count", range(6))
def test_design_level_class_support_for_every_non_breach_count(non_breach_count: int) -> None:
    outcome = audit.classify_design(["0"] * non_breach_count + ["-0.1"] * (5 - non_breach_count))
    assert outcome["non_breach_realization_count"] == non_breach_count
    assert outcome["breach_realization_count"] == 5 - non_breach_count
    assert outcome["majority_class"] == ("non_breach_majority" if non_breach_count >= 3 else "breach_majority")
    assert outcome["mixed_design"] is (0 < non_breach_count < 5)


def test_boundary_endpoints_and_canonical_margin_half_even() -> None:
    assert audit.classify_design(["-0.01", "0", "0.1", "0.2", "0.3"])["observed_boundary_design"] is True
    assert audit.classify_design(["0.05", "0.05", "0.2", "0.3", "0.4"])["observed_boundary_design"] is True
    assert audit.classify_design(["-0.05", "-0.05", "-0.2", "-0.3", "-0.4"])["observed_boundary_design"] is True
    assert audit.canonical_margin("0.0000005") == "0.000000"
    assert audit.canonical_margin("0.0000015") == "0.000002"
    assert audit.canonical_margin("0.000000499999") == "0.000000"
    assert audit.canonical_margin("0.000000500001") == "0.000001"
    assert audit.canonical_margin("-0.0000005") == "-0.000000"
    assert audit.canonical_margin("-0.0000015") == "-0.000002"
    assert audit.canonical_margin("-0.000000499999") == "-0.000000"
    assert audit.canonical_margin("-0.000000500001") == "-0.000001"


def test_original_identity_split_seed_and_target_contract_remain_preserved() -> None:
    designs = audit.read_jsonl(ROOT / "artifacts/final_integrated_dataset_contract/designs.jsonl")
    runs = audit.read_jsonl(ROOT / "artifacts/final_integrated_dataset_contract/runs.jsonl")
    split = audit.read_json(ROOT / "artifacts/final_integrated_dataset_contract/split_manifest.json")
    inventory = audit.read_json(ROOT / "artifacts/final_integrated_dataset_contract/manifest_inventory.json")
    assert [row["design_id"] for row in designs] == [f"D{index:03d}" for index in range(100)]
    assert [row["run_id"] for row in runs] == list(range(500))
    assert all(row["design_id"] in split["design_assignments"][row["split_assignment"]] for row in runs)
    assert all(row["run_id"] in split["run_assignments"][row["split_assignment"]] for row in runs)
    assert {name: len(values) for name, values in split["design_assignments"].items()} == {"train": 70, "validation": 15, "test": 15}
    assert {name: len(values) for name, values in split["run_assignments"].items()} == {"train": 350, "validation": 75, "test": 75}
    assert all(type(row["satellite_seed"]) is int and type(row["ground_failure_seed"]) is int for row in runs)
    assert inventory["target_schema_hash"] == "9088948d6b03db59877a129179ac3091c3690aeab9d2849a921bfa90f05da528"


def test_final_gate_scope_and_simultaneous_feasibility() -> None:
    rows = audit.gate_feasibility(ROOT / "artifacts/final_integrated_dataset_class_support_audit/audit_boundary_reproduction.csv")
    assert [row["split"] for row in rows] == ["train", "validation", "test"]
    assert [(row["combined_designs"], row["combined_runs"]) for row in rows] == [(130, 650), (30, 150), (30, 150)]
    assert all(row["simultaneously_feasible"] for row in rows)
    assert all(row["stage_a_applicable"] is False for row in rows)


def test_authorization_and_output_roots_are_fail_closed(snapshot, tmp_path: Path) -> None:
    designs, runs, seeds, _, _ = snapshot
    contract = audit.read_json(ARTIFACT_ROOT / "stage_a_contract_proposal.json")
    manifests = [
        ("partition", audit.read_json(ARTIFACT_ROOT / "stage_a_partition_manifest.json")),
        ("seed policy", audit.read_json(ARTIFACT_ROOT / "stage_a_seed_policy.json")),
        ("region bounds", audit.read_json(ARTIFACT_ROOT / "stage_a_region_bounds.json")),
        ("output roots", audit.read_json(ARTIFACT_ROOT / "stage_a_output_root_manifest.json")),
        ("criteria", audit.read_json(ARTIFACT_ROOT / "stage_a_discovery_criteria.json")),
        ("neighbors", audit.read_json(ARTIFACT_ROOT / "stage_a_near_neighbor_policy.json")),
        ("inventory", audit.read_json(ARTIFACT_ROOT / "stage_a_proposal_inventory.json")),
    ]
    result = audit.authorization_audit(contract, designs, runs, seeds, manifests, ROOT)
    assert result["proposal_status"] == "NOT_FROZEN"
    assert result["simulation_authorized"] is False
    assert result["contract_frozen_equivalent"] is False
    root_result = audit.output_root_audit(
        manifests[3][1],
        [tmp_path / "external", ROOT / "artifacts/stage_a_discovery_contract_freeze_audit"],
        [Path("C:/Users/johns/satnet-final-production-20260720"), Path("C:/Users/johns/satnet-final-production-replay-20260720")],
        [ROOT],
    )
    assert root_result["all_absent_and_isolated"] is True


@pytest.mark.parametrize(
    ("status", "authorized", "frozen"),
    [("FROZEN", False, False), ("NOT_FROZEN", True, False), ("NOT_FROZEN", False, True)],
)
def test_rejects_invalid_authorization_states(status: str, authorized: bool, frozen: bool) -> None:
    with pytest.raises(ValueError):
        audit.validate_authorization_object(
            {"proposal_status": status, "simulation_authorized": authorized, "contract_frozen": frozen},
            "mutation",
        )


def test_rejects_every_missing_holdout_prerequisite_and_contract_hash() -> None:
    valid = {
        "stage_b_contract_completely_specified": True,
        "stage_b_design_manifest_created": True,
        "stage_b_run_manifest_created": True,
        "stage_b_seed_manifest_created": True,
        "stage_b_split_frozen": True,
        "stage_b_independent_audit_passed": True,
        "stage_b_contract_hash_recorded": True,
    }
    audit.validate_holdout_unsealing(valid, "a" * 64)
    for field in valid:
        mutated = dict(valid)
        mutated[field] = False
        with pytest.raises(ValueError, match="prerequisites"):
            audit.validate_holdout_unsealing(mutated, "a" * 64)
    with pytest.raises(ValueError, match="hash"):
        audit.validate_holdout_unsealing(valid, None)


def test_snapshot_rejects_duplicate_identity_count_bound_seed_and_neighbor_mutations(snapshot) -> None:
    designs, runs, seeds, original, bounds = snapshot
    mutations = []
    duplicate_design = deepcopy(designs)
    duplicate_design[1] = deepcopy(duplicate_design[0])
    mutations.append((duplicate_design, runs, seeds))
    duplicate_index = deepcopy(designs)
    duplicate_index[1]["design_index"] = 0
    mutations.append((duplicate_index, runs, seeds))
    identity_collision = deepcopy(designs)
    identity_collision[0]["design_id"] = "D000"
    mutations.append((identity_collision, runs, seeds))
    duplicate_vector = deepcopy(designs)
    for field in audit.SCIENTIFIC_PARAMETER_FIELDS:
        duplicate_vector[1][field] = duplicate_vector[0][field]
    mutations.append((duplicate_vector, runs, seeds))
    original_duplicate = deepcopy(designs)
    for field in audit.SCIENTIFIC_PARAMETER_FIELDS:
        original_duplicate[0][field] = original[0][field]
    mutations.append((original_duplicate, runs, seeds))
    wrong_region = deepcopy(designs)
    wrong_region[0]["region"] = "boundary"
    mutations.append((wrong_region, runs, seeds))
    wrong_partition = deepcopy(designs)
    wrong_partition[0]["partition"] = "validation"
    mutations.append((wrong_partition, runs, seeds))
    out_of_bounds = deepcopy(designs)
    out_of_bounds[0]["altitude_km"] = "1201"
    mutations.append((out_of_bounds, runs, seeds))
    region_violation = deepcopy(designs)
    region_violation[0]["altitude_km"] = "700"
    mutations.append((region_violation, runs, seeds))
    duplicate_run = deepcopy(runs)
    duplicate_run[1]["run_key"] = duplicate_run[0]["run_key"]
    mutations.append((designs, duplicate_run, seeds))
    missing_run = deepcopy(runs[:-1])
    mutations.append((designs, missing_run, seeds))
    substituted_seed = deepcopy(seeds)
    substituted_seed[0]["satellite_failure_seed"] += 1
    mutations.append((designs, runs, substituted_seed))
    neighbor_violation = deepcopy(designs)
    source = next(row for row in neighbor_violation if row["design_id"] == "SA-D013")
    target = next(row for row in neighbor_violation if row["design_id"] == "SA-D020")
    for field in audit.SCIENTIFIC_PARAMETER_FIELDS:
        target[field] = source[field]
    mutations.append((neighbor_violation, runs, seeds))
    for mutated_designs, mutated_runs, mutated_seeds in mutations:
        with pytest.raises(ValueError):
            audit.validate_snapshot(mutated_designs, mutated_runs, mutated_seeds, original, bounds)


def test_rejects_infeasible_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(audit.FINAL_GATES["minimum_non_breach_runs"], "validation", 151)
    rows = audit.gate_feasibility(ROOT / "artifacts/final_integrated_dataset_class_support_audit/audit_boundary_reproduction.csv")
    assert next(row for row in rows if row["split"] == "validation")["simultaneously_feasible"] is False
