from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

import pytest

from satnet.experiments.final_dataset.design import (
    build_design_records,
    build_run_records,
    design_manifest_hash,
    run_manifest_hash,
)
from satnet.experiments.final_dataset.materialize import (
    materialize_final_contract_manifests,
    read_json,
    read_jsonl,
    validate_materialized_contract,
)
from satnet.experiments.final_dataset.specification import (
    CATALOG_HASH,
    build_contract_specification,
    materialize_machine_specification,
)
from satnet.experiments.final_dataset.split import (
    _candidate_assignments,
    _score,
    build_split_manifest,
    reduced_composition_ratio,
)
from satnet.experiments.integrated_ground_manifest import read_pilot_design_manifest
from satnet.ground.catalog import load_ground_station_catalog

ROOT = Path(__file__).parents[2]
OUTPUT = ROOT / "artifacts" / "final_integrated_dataset_contract"
PILOT_DESIGNS = ROOT / "artifacts" / "integrated_ground_pilot_25" / "inputs" / "pilot_designs.json"
PILOT_CATALOG = ROOT / "artifacts" / "integrated_ground_pilot_25" / "inputs" / "pilot_catalog.csv"


@pytest.fixture(scope="module")
def designs() -> tuple[dict, ...]:
    return read_jsonl(OUTPUT / "designs.jsonl")


@pytest.fixture(scope="module")
def runs() -> tuple[dict, ...]:
    return read_jsonl(OUTPUT / "runs.jsonl")


def test_manifest_counts_ordering_and_hash_inventory(designs, runs) -> None:
    assert len(designs) == 100
    assert len(runs) == 500
    assert [record["design_id"] for record in designs] == [f"D{index:03d}" for index in range(100)]
    assert [record["run_id"] for record in runs] == list(range(500))
    assert all(type(record["run_id"]) is int for record in runs)
    assert all("run_index" not in record for record in runs)
    assert len({record["run_key"] for record in runs}) == 500
    inventory = read_json(OUTPUT / "manifest_inventory.json")
    assert inventory["catalog_hash"] == CATALOG_HASH
    assert inventory["design_manifest_hash"] == design_manifest_hash(designs)
    assert inventory["run_manifest_hash"] == run_manifest_hash(runs)
    assert inventory["split_design_counts"] == {"train": 70, "validation": 15, "test": 15}


def test_pilot_anchor_scientific_equality_and_new_identity(designs) -> None:
    pilot_designs = read_pilot_design_manifest(PILOT_DESIGNS)
    for final, pilot in zip(designs[:5], pilot_designs, strict=True):
        assert final["pilot_design_id"] == pilot.design_id
        assert final["pilot_design_hash"] == pilot.design_hash
        assert final["design_id"] != pilot.design_id
        assert final["design_record_hash"] != pilot.design_hash
        assert final["num_planes"] == pilot.num_planes
        assert final["sats_per_plane"] == pilot.sats_per_plane
        assert float(final["altitude_km"]) == pilot.altitude_km
        assert float(final["inclination_deg"]) == pilot.inclination_deg
        assert float(final["satellite_node_failure_probability"]) == pilot.node_failure_probability
        assert float(final["satellite_edge_failure_probability"]) == pilot.edge_failure_probability
        assert final["civilian_count"] == pilot.civilian_count
        assert final["government_count"] == pilot.government_count
        assert final["military_count"] == pilot.military_count
        assert float(final["ground_station_failure_probability"]) == pilot.ground_failure_probability
    assert designs[0]["total_ground_station_count"] == 20
    assert (designs[0]["civilian_count"], designs[0]["government_count"], designs[0]["military_count"]) == (8, 6, 6)
    assert designs[4]["total_ground_station_count"] == 3


def test_design_level_ground_selection_is_materialized_and_fixed(designs, runs) -> None:
    for design in designs:
        selected = (
            design["civilian_selected_station_ids"]
            + design["government_selected_station_ids"]
            + design["military_selected_station_ids"]
        )
        assert selected == design["selected_station_ids"]
        assert len(selected) == design["total_ground_station_count"]
        group = [record for record in runs if record["design_id"] == design["design_id"]]
        assert len(group) == 5
        for field in (
            "ground_selection_seed",
            "ground_selection_hash",
            "ground_design_hash",
        ):
            assert {record[field] for record in group} == {design[field]}
        assert len({record["satellite_seed"] for record in group}) == 5
        assert len({record["ground_failure_seed"] for record in group}) == 5


def test_exact_doe_coverage(designs) -> None:
    assert Counter(record["doe_stratum"] for record in designs) == {
        "pilot_anchor": 5,
        "transition": 35,
        "global": 60,
    }
    transition = designs[5:40]
    assert len(
        {
            (record["total_ground_station_count"], tuple(record["composition_weights"]))
            for record in transition
        }
    ) == 35
    assert Counter((record["num_planes"], record["sats_per_plane"]) for record in transition) == {
        (5, 6): 6,
        (5, 7): 6,
        (5, 8): 6,
        (6, 6): 6,
        (6, 7): 6,
        (6, 8): 5,
    }
    global_records = designs[40:]
    assert set(Counter((record["num_planes"], record["sats_per_plane"]) for record in global_records).values()) == {5}
    assert Counter(record["total_ground_station_count"] for record in global_records) == {
        total: 6 for total in (6, 10, 15, 20, 25, 30, 35, 40, 45, 50)
    }
    assert set(Counter(tuple(record["composition_weights"]) for record in global_records).values()) == {6}
    assert {record["lhs_candidate_id"] for record in transition} == {72}
    assert {record["lhs_candidate_id"] for record in global_records} == {158}


def test_reduced_compositions_unify_anchor_and_non_anchor(designs) -> None:
    assert reduced_composition_ratio(designs[0]) == "4:3:3"
    assert reduced_composition_ratio(designs[1]) == "3:1:1"
    assert reduced_composition_ratio(designs[2]) == "4:3:3"
    assert reduced_composition_ratio(designs[3]) == "1:2:2"
    assert reduced_composition_ratio(designs[4]) == "1:1:1"
    assert any(
        reduced_composition_ratio(record) == "1:1:1" for record in designs[5:]
    )


def test_grouped_split_is_pre_outcome_and_exact(designs, runs) -> None:
    split = read_json(OUTPUT / "split_manifest.json")
    assert split["selected_candidate_id"] == 164
    assert split["outcome_fields_used"] is False
    assert {name: len(split["design_assignments"][name]) for name in ("train", "validation", "test")} == {
        "train": 70,
        "validation": 15,
        "test": 15,
    }
    assert {name: len(split["run_assignments"][name]) for name in ("train", "validation", "test")} == {
        "train": 350,
        "validation": 75,
        "test": 75,
    }
    assignments = _candidate_assignments(designs, 164)
    score = _score(designs, assignments)
    persisted = split["selected_candidate_score"]
    assert (score[0].numerator, score[0].denominator) == (
        persisted["maximum_normalized_deviation"]["numerator"],
        persisted["maximum_normalized_deviation"]["denominator"],
    )
    assert (score[1].numerator, score[1].denominator) == (
        persisted["sum_squared_normalized_deviation"]["numerator"],
        persisted["sum_squared_normalized_deviation"]["denominator"],
    )
    assert (score[2].numerator, score[2].denominator) == (
        persisted["total_absolute_deviation"]["numerator"],
        persisted["total_absolute_deviation"]["denominator"],
    )
    anchor_counts = {
        name: sum(design_id in {"D000", "D001", "D002", "D003", "D004"} for design_id in split["design_assignments"][name])
        for name in ("train", "validation", "test")
    }
    assert anchor_counts == {"train": 3, "validation": 1, "test": 1}


def test_split_fails_without_a_valid_hard_candidate(monkeypatch, designs, runs) -> None:
    monkeypatch.setattr(
        "satnet.experiments.final_dataset.split._meets_hard_requirements",
        lambda assignments: False,
    )
    specification = build_contract_specification()
    with pytest.raises(RuntimeError, match="No split candidate"):
        build_split_manifest(
            designs=designs,
            runs=runs,
            contract_spec_hash=specification["contract_spec_hash"],
            design_manifest_hash=design_manifest_hash(designs),
        )


def test_split_digest_tie_uses_design_id(monkeypatch, designs) -> None:
    monkeypatch.setattr(
        "satnet.experiments.final_dataset.split.canonical_digest",
        lambda payload: bytes(32),
    )
    assignments = _candidate_assignments(tuple(reversed(designs)), 0)
    ordered = [record["design_id"] for name in ("train", "validation", "test") for record in assignments[name]]
    assert ordered == [f"D{index:03d}" for index in range(100)]


def test_catalog_semantic_identity_and_separate_file_identity() -> None:
    catalog = load_ground_station_catalog(PILOT_CATALOG)
    inventory = read_json(OUTPUT / "manifest_inventory.json")
    assert catalog.catalog_hash == CATALOG_HASH
    assert inventory["pilot_catalog_file_sha256"] != catalog.catalog_hash
    assert len(inventory["pilot_catalog_file_sha256"]) == 64


def test_identity_graph_is_acyclic(designs, runs) -> None:
    specification = read_json(OUTPUT / "contract_specification.json")
    bundle = read_json(OUTPUT / "contract_bundle.json")
    assert specification["contract_spec_hash"] == bundle["contract_spec_hash"]
    assert "contract_bundle_hash" not in specification
    assert all("contract_bundle_hash" not in record for record in designs)
    assert all("contract_bundle_hash" not in record for record in runs)
    assert all(record["contract_spec_hash"] == specification["contract_spec_hash"] for record in designs)
    assert all(record["contract_spec_hash"] == specification["contract_spec_hash"] for record in runs)


def test_schemas_have_no_rf_outcome_leakage() -> None:
    rf = read_json(OUTPUT / "integrated_rf_export_schema.json")
    target = read_json(OUTPUT / "target_schema.json")
    predictor_names = {entry["field"] for entry in rf["predictors"]}
    target_names = {entry["field"] for entry in target["targets"]}
    assert predictor_names.isdisjoint(target_names)
    assert all("seed" not in name for name in predictor_names)
    assert all("hash" not in name for name in predictor_names)
    assert "selected_station_ids" not in predictor_names
    assert rf["current_model_changes_authorized"] is False
    tgnn = read_json(OUTPUT / "integrated_tgnn_adapter_schema.json")
    assert tgnn["implementation_status"] == "future_adapter_not_implemented"
    assert tgnn["node_inputs"]["satellite_ecef"] == "omitted_in_v1"
    assert tgnn["current_model_changes_authorized"] is False


def test_contract_rematerializes_byte_for_byte(tmp_path: Path) -> None:
    outputs = (tmp_path / "contract_a", tmp_path / "contract_b")
    for output in outputs:
        materialize_machine_specification(output)
        materialize_final_contract_manifests(
            output_root=output,
            pilot_design_manifest=PILOT_DESIGNS,
            catalog_path=PILOT_CATALOG,
        )
        assert validate_materialized_contract(output) == validate_materialized_contract(OUTPUT)
    expected_names = sorted(path.name for path in OUTPUT.iterdir() if path.is_file())
    for output in outputs:
        assert sorted(path.name for path in output.iterdir() if path.is_file()) == expected_names
    for name in expected_names:
        committed = (OUTPUT / name).read_bytes()
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
        assert (outputs[0] / name).read_bytes() == committed


def test_machine_specification_precedes_manifest_identity(designs) -> None:
    specification = read_json(OUTPUT / "contract_specification.json")
    assert specification == build_contract_specification()
    assert all(record["contract_spec_hash"] == specification["contract_spec_hash"] for record in designs)


def test_no_placeholders_or_outcomes_in_contract_inputs(designs, runs) -> None:
    forbidden_text = ("TODO", "TBD", "placeholder", "unknown")
    for path in OUTPUT.iterdir():
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert not any(value.lower() in text.lower() for value in forbidden_text)
    outcome_names = {
        "overall_threshold_breach_any",
        "ground_threshold_breach_any",
        "space_threshold_breach_any",
        "failure_adjusted_overall_service_fraction_mean",
        "failure_adjusted_overall_service_fraction_min",
        "failure_adjusted_ground_service_fraction_min",
        "space_gcc_fraction_original_min",
        "ground_service_loss_due_to_failures_max",
    }
    assert all(outcome_names.isdisjoint(record) for record in designs)
    assert all(outcome_names.isdisjoint(record) for record in runs)
