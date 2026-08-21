from __future__ import annotations

import json
from pathlib import Path

import pytest

from satnet.experiments.integrated_ground_manifest import (
    PILOT_EXPECTED_RUN_COUNT,
    SEED_PURPOSES,
    IntegratedPilotRun,
    build_pilot_designs,
    build_pilot_runs,
    build_synthetic_pilot_catalog,
    catalog_provenance,
    derive_pilot_seed,
    materialize_pilot_inputs,
    pilot_run_manifest_hash,
    read_pilot_design_manifest,
    read_pilot_run_manifest,
)
from satnet.ground.catalog import GroundStationClass, load_ground_station_catalog


def test_five_by_five_manifest_and_run_mapping() -> None:
    designs = build_pilot_designs()
    runs = build_pilot_runs(designs)
    assert len(designs) == 5
    assert len(runs) == PILOT_EXPECTED_RUN_COUNT
    assert [run.run_id for run in runs] == list(range(25))
    assert [(run.design_id, run.realization_id) for run in runs[:6]] == [
        ("P01", "R01"),
        ("P01", "R02"),
        ("P01", "R03"),
        ("P01", "R04"),
        ("P01", "R05"),
        ("P02", "R01"),
    ]
    assert runs[-1].design_id == "P05"
    assert runs[-1].realization_id == "R05"
    assert {run.design_group_id for run in runs} == {design.design_id for design in designs}
    assert all(sum(run.design_id == design.design_id for run in runs) == 5 for design in designs)


def test_seed_derivation_golden_vectors() -> None:
    assert {
        purpose: derive_pilot_seed(
            design_id="P01",
            realization_id="R01",
            seed_purpose=purpose,
        )
        for purpose in SEED_PURPOSES
    } == {
        "satellite_rollout_and_failure": 7354080756714418247,
        "ground_station_selection": 8343673171039191876,
        "ground_failure_realization": 783383988117819731,
    }


def test_seed_derivation_is_order_independent_and_purpose_specific() -> None:
    values = {
        purpose: derive_pilot_seed(
            design_id="P03",
            realization_id="R04",
            seed_purpose=purpose,
        )
        for purpose in reversed(SEED_PURPOSES)
    }
    assert len(set(values.values())) == 3
    assert all(type(value) is int and 0 <= value < 2**63 for value in values.values())
    with pytest.raises(ValueError):
        derive_pilot_seed(
            design_id="P03",
            realization_id="R04",
            seed_purpose="unsupported",
        )


def test_design_hash_is_constant_within_group_and_seeds_vary() -> None:
    runs = build_pilot_runs()
    p02 = [run for run in runs if run.design_id == "P02"]
    assert len({run.design_hash for run in p02}) == 1
    assert len({run.satellite_rollout_seed for run in p02}) == 5
    assert len({run.ground_station_selection_seed for run in p02}) == 5
    assert len({run.ground_failure_seed for run in p02}) == 5


def test_run_constructor_rejects_wrong_mapping_and_duplicate_manifest(tmp_path: Path) -> None:
    run = build_pilot_runs()[0]
    with pytest.raises(ValueError, match="mapping"):
        IntegratedPilotRun(
            run_id=1,
            design_id=run.design_id,
            realization_id=run.realization_id,
            design_group_id=run.design_group_id,
            design_hash=run.design_hash,
            satellite_rollout_seed=run.satellite_rollout_seed,
            ground_station_selection_seed=run.ground_station_selection_seed,
            ground_failure_seed=run.ground_failure_seed,
        )
    root = tmp_path / "pilot"
    materialize_pilot_inputs(root)
    path = root / "inputs" / "pilot_runs.jsonl"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(first + "\n" + first + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_pilot_run_manifest(path, read_pilot_design_manifest(root / "inputs" / "pilot_designs.json"))


def test_synthetic_catalog_has_exact_pilot_population_and_provenance() -> None:
    catalog = build_synthetic_pilot_catalog()
    assert len(catalog.stations) == 150
    assert {
        station_class.value: len(catalog.eligible(station_class))
        for station_class in GroundStationClass
    } == {"civilian": 50, "government": 50, "military": 50}
    assert {station.country_code for station in catalog.stations} == {"ZZ"}
    assert all(station.enabled for station in catalog.stations)
    assert all(station.name.startswith("Synthetic Pilot") for station in catalog.stations)
    assert len({station.region for station in catalog.stations}) == 10
    provenance = catalog_provenance(catalog)
    assert provenance["synthetic"] is True
    assert provenance["pilot_only"] is True
    assert provenance["scientifically_reviewed"] is False
    assert provenance["production_research_catalog_availability"] == "NOT AVAILABLE"
    assert provenance["production_research_catalog_scientific_review"] == "NOT PERFORMED"


def test_manifest_materialization_roundtrips_and_hashes(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    evidence = materialize_pilot_inputs(root)
    catalog = load_ground_station_catalog(root / "inputs" / "pilot_catalog.csv")
    designs = read_pilot_design_manifest(root / "inputs" / "pilot_designs.json")
    runs = read_pilot_run_manifest(root / "inputs" / "pilot_runs.jsonl", designs)
    provenance = json.loads(
        (root / "inputs" / "pilot_catalog_provenance.json").read_text(encoding="utf-8")
    )
    assert catalog.catalog_hash == evidence["catalog_hash"]
    assert evidence["catalog_hash"] == "810c64dfb030b042311c90f2f42f8dee866a48fc63a6a29e362ee328c52eaa6e"
    assert pilot_run_manifest_hash(runs) == evidence["run_manifest_hash"]
    assert evidence["run_manifest_hash"] == "1b1e128561a7f25233bc89677e5c40da7c4df8945e6070e2827f12d6c42535c7"
    assert provenance["catalog_hash"] == catalog.catalog_hash
    assert len(designs) == 5
    assert len(runs) == 25
    for path in (root / "inputs").iterdir():
        assert path.read_bytes().endswith(b"\n")
        assert b"\r\n" not in path.read_bytes()
    with pytest.raises(FileExistsError):
        materialize_pilot_inputs(root)


def test_committed_run_summary_contains_catalog_and_all_seed_identities() -> None:
    root = Path(__file__).parents[2] / "artifacts" / "integrated_ground_pilot_25"
    provenance = json.loads(
        (root / "inputs" / "pilot_catalog_provenance.json").read_text(encoding="utf-8")
    )
    manifest_rows = [
        json.loads(line)
        for line in (root / "inputs" / "pilot_runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary_rows = [
        json.loads(line)
        for line in (root / "summaries" / "pilot_run_summary.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest_rows) == len(summary_rows) == 25
    for manifest, summary in zip(manifest_rows, summary_rows, strict=True):
        assert summary["run_id"] == manifest["run_id"]
        assert summary["catalog_hash"] == provenance["catalog_hash"]
        assert summary["satellite_rollout_seed"] == manifest["satellite_rollout_seed"]
        assert summary["ground_station_selection_seed"] == manifest["ground_station_selection_seed"]
        assert summary["ground_failure_seed"] == manifest["ground_failure_seed"]


def test_manifest_reader_rejects_unknown_and_duplicate_json_keys(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    materialize_pilot_inputs(root)
    design_path = root / "inputs" / "pilot_designs.json"
    design_value = json.loads(design_path.read_text(encoding="utf-8"))
    design_value["unknown"] = 1
    design_path.write_text(json.dumps(design_value, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_pilot_design_manifest(design_path)
    design_path.write_text(
        '{"pilot_schema_version":"1","pilot_schema_version":"1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        read_pilot_design_manifest(design_path)
