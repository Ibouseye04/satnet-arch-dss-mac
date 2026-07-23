from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from satnet.experiments.stage_a_contract.designs import (
    build_design_rows,
    minimum_distances,
    scientific_signature,
    validate_design_rows,
)
from satnet.experiments.stage_a_contract.proposal import (
    build_artifact_payloads,
    build_discovery_criteria,
    build_final_gate_feasibility,
    build_partition_manifest,
    build_run_rows,
    build_seed_rows,
    load_original_designs,
    validate_holdout_policy,
)
from satnet.experiments.stage_a_contract.semantics import (
    CORPUS_NAMESPACE,
    OUTPUT_ROOTS,
    canonical_margin,
    design_outcome,
    observed_boundary_design,
)

ROOT = Path(__file__).parents[2]


def _csv_rows(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))


def test_exact_design_run_region_partition_and_identity_contract() -> None:
    original = load_original_designs(ROOT / "artifacts/final_integrated_dataset_contract/designs.jsonl")
    designs = build_design_rows()
    validate_design_rows(designs, original)
    runs = build_run_rows(designs)
    assert len(designs) == 30
    assert len(runs) == 150
    assert [row["design_id"] for row in designs] == [f"SA-D{index:03d}" for index in range(30)]
    assert [row["global_run_id"] for row in runs] == list(range(500, 650))
    assert len({row["run_key"] for row in runs}) == 150
    assert len({(row["design_id"], row["realization_id"]) for row in runs}) == 150
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
    for design in designs:
        group = [row for row in runs if row["design_id"] == design["design_id"]]
        assert len(group) == 5
        assert {row["partition"] for row in group} == {design["partition"]}
        assert {row["region"] for row in group} == {design["region"]}
        assert all(row["sealed"] is (design["partition"] == "sealed_holdout") for row in group)
        assert all(row["proposal_status"] == "NOT_FROZEN" for row in group)
        assert all(row["simulation_authorized"] is False for row in group)
    assert not ({row["design_id"] for row in designs} & {f"D{index:03d}" for index in range(100)})
    assert not ({row["global_run_id"] for row in runs} & set(range(500)))


def test_design_bounds_duplicates_hashes_and_near_neighbor_isolation() -> None:
    original = load_original_designs(ROOT / "artifacts/final_integrated_dataset_contract/designs.jsonl")
    first = build_design_rows()
    second = build_design_rows()
    validate_design_rows(first, original)
    assert [row["design_parameter_hash"] for row in first] == [row["design_parameter_hash"] for row in second]
    assert [row["design_record_hash"] for row in first] == [row["design_record_hash"] for row in second]
    assert len({scientific_signature(row) for row in first}) == 30
    assert not ({scientific_signature(row) for row in first} & {scientific_signature(row) for row in original})
    distances = minimum_distances(first)
    assert distances["minimum_sealed_to_unsealed_distance"] > 0.10


def test_seed_manifest_is_domain_separated_deterministic_and_design_fixed() -> None:
    runs = build_run_rows(build_design_rows())
    first = build_seed_rows(runs)
    second = build_seed_rows(runs)
    assert first == second
    assert len(first) == 150
    for design_id in [f"SA-D{index:03d}" for index in range(30)]:
        group = [row for row in first if row["design_id"] == design_id]
        assert len(group) == 5
        assert len({row["design_construction_seed"] for row in group}) == 1
        assert len({row["ground_selection_seed"] for row in group}) == 1
        assert len({row["satellite_failure_seed"] for row in group}) == 5
        assert len({row["ground_failure_seed"] for row in group}) == 5
        assert all(0 <= int(row[field]) < 2**63 for row in group for field in ("design_construction_seed", "ground_selection_seed", "satellite_failure_seed", "ground_failure_seed"))
    for row in first:
        assert len({row["design_construction_seed"], row["ground_selection_seed"], row["satellite_failure_seed"], row["ground_failure_seed"]}) == 4


def test_boundary_majority_mixed_and_half_even_semantics() -> None:
    assert observed_boundary_design(["-0.01", "0", "0.1", "0.2", "0.3"]) is True
    assert observed_boundary_design(["0.05", "0.05", "0.2", "0.3", "0.4"]) is True
    assert observed_boundary_design(["-0.05", "-0.05", "-0.2", "-0.3", "-0.4"]) is True
    assert observed_boundary_design(["0.051", "0.051", "0.2", "0.3", "0.4"]) is False
    assert design_outcome(["-0.1", "-0.1", "-0.1", "0", "0.1"]) == {
        "majority_class": "breach_majority",
        "mixed_design": True,
        "observed_boundary_design": True,
        "non_breach_realization_count": 2,
        "breach_realization_count": 3,
    }
    assert canonical_margin("0.0000005") == "0.000000"
    assert canonical_margin("0.0000015") == "0.000002"
    assert canonical_margin("-0.0000005") == "-0.000000"
    assert design_outcome(["0", "0", "0", "-0.1", "-0.2"])["majority_class"] == "non_breach_majority"


def test_holdout_stage_b_boundary_and_final_corpus_semantics() -> None:
    designs = build_design_rows()
    runs = build_run_rows(designs)
    partition = build_partition_manifest(designs, runs)
    holdout = partition["partitions"]["sealed_holdout"]
    assert holdout["sealed"] is True
    assert holdout["design_count"] == 5
    assert holdout["run_count"] == 25
    assert partition["holdout_not_final_test_set"] is True
    with pytest.raises(ValueError, match="contract hash"):
        validate_holdout_policy(partition)
    validate_holdout_policy(partition, stage_b_contract_hash="a" * 64)
    criteria = build_discovery_criteria()
    assert criteria["final_classification_gates_apply"] is False
    assert all(item["can_influence_stage_b"] is False for item in criteria["sealed_holdout_confirmation_criteria"])
    payloads = build_artifact_payloads(ROOT)
    contract = json.loads(payloads["stage_a_contract_proposal.json"])
    membership = contract["final_corpus_membership"]
    assert membership["primary_final_classification_corpus"] == ["original frozen 500-run corpus", "future frozen Stage B corpus"]
    assert membership["stage_a_development"] == "EXCLUDED"
    assert membership["stage_a_validation"] == "EXCLUDED"
    assert membership["stage_a_sealed_holdout"] == "EXCLUDED"
    assert membership["expected_if_stage_b_remains_90_designs"] == {"designs": 190, "runs": 950}


def test_final_gate_feasibility_uses_corrected_definitions() -> None:
    result = build_final_gate_feasibility(ROOT / "artifacts/final_integrated_dataset_class_support_audit/audit_boundary_reproduction.csv")
    assert result["overall_feasible"] is True
    assert result["stage_a_excluded"] is True
    assert set(result["by_split"]) == {"train", "validation", "test"}
    assert all(value["all_gates_mathematically_feasible"] for value in result["by_split"].values())
    assert result["by_split"]["train"]["original"]["non_breach_majority_designs"] == 0
    assert result["by_split"]["test"]["original"]["non_breach_majority_designs"] == 1
    assert result["by_split"]["train"]["exact_feasible_stage_b_non_breach_run_interval"]["minimum"] == 48
    assert "majority" in result["gate_definitions"]["design_class"]
    assert "quantized" in result["gate_definitions"]["distinct_margin"]


def test_machine_readable_payloads_are_complete_deterministic_and_unauthorized() -> None:
    first = build_artifact_payloads(ROOT)
    second = build_artifact_payloads(ROOT)
    assert first == second
    assert set(first) == {
        "stage_a_design_manifest.csv",
        "stage_a_run_manifest.csv",
        "stage_a_partition_manifest.json",
        "stage_a_seed_policy.json",
        "stage_a_seed_manifest.csv",
        "stage_a_region_bounds.json",
        "stage_a_output_root_manifest.json",
        "stage_a_discovery_criteria.json",
        "stage_a_near_neighbor_policy.json",
        "stage_a_contract_proposal.json",
    }
    assert len(_csv_rows(first["stage_a_design_manifest.csv"])) == 30
    assert len(_csv_rows(first["stage_a_run_manifest.csv"])) == 150
    assert len(_csv_rows(first["stage_a_seed_manifest.csv"])) == 150
    for name, payload in first.items():
        assert hashlib.sha256(payload).hexdigest() == hashlib.sha256(second[name]).hexdigest()
    contract = json.loads(first["stage_a_contract_proposal.json"])
    assert contract["corpus_namespace"] == CORPUS_NAMESPACE
    assert contract["proposal_status"] == "NOT_FROZEN"
    assert contract["simulation_authorized"] is False
    roots = json.loads(first["stage_a_output_root_manifest.json"])
    assert roots["proposed_resolved_paths"] == OUTPUT_ROOTS
    assert roots["current_existence"] == {name: False for name in OUTPUT_ROOTS}


def test_stage_a_source_has_no_simulation_replay_or_training_entrypoint() -> None:
    package = ROOT / "src/satnet/experiments/stage_a_contract"
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(package.glob("*.py")))
    prohibited = ("run_tier1_rollout(", "generate_run(", "generate_runs(", "replay_runs_read_only(", "train_rf_model(", "SatelliteGNN(")
    assert not any(value in source for value in prohibited)
