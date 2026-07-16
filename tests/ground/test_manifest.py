from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from satnet.ground.catalog import GroundStationClass, load_ground_station_catalog
from satnet.ground.persistence import (
    GROUND_DESIGN_SCHEMA_VERSION,
    GroundRunDesignRecord,
    make_disabled_ground_design_record,
    make_enabled_ground_design_record,
    read_ground_design_manifest,
    reconstruct_ground_selection,
    validate_manifest_against_satellite_runs,
    write_ground_design_manifest,
)
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "ground_segment"
    / "synthetic_ground_station_catalog.csv"
)
HASH_A = "a" * 64
HASH_B = "b" * 64


def enabled_record(run_id: int = 0, satellite_hash: str = HASH_A) -> GroundRunDesignRecord:
    catalog = load_ground_station_catalog(FIXTURE)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(6, 3, 3, 42),
    )
    return make_enabled_ground_design_record(
        run_id=run_id,
        satellite_config_hash=satellite_hash,
        selection=selection,
    )


def disabled_record(run_id: int = 1, satellite_hash: str = HASH_B) -> GroundRunDesignRecord:
    return make_disabled_ground_design_record(
        run_id=run_id,
        satellite_config_hash=satellite_hash,
    )


def test_enabled_record_round_trips_exactly(tmp_path: Path) -> None:
    path = tmp_path / "ground_designs.jsonl"
    original = enabled_record()
    write_ground_design_manifest([original], path)
    loaded = read_ground_design_manifest(path)
    assert loaded == (original,)
    assert reconstruct_ground_selection(loaded[0], load_ground_station_catalog(FIXTURE)) is not None


def test_disabled_record_round_trips_without_catalog(tmp_path: Path) -> None:
    path = tmp_path / "ground_designs.jsonl"
    original = disabled_record()
    write_ground_design_manifest([original], path)
    loaded = read_ground_design_manifest(path)
    assert loaded == (original,)
    assert reconstruct_ground_selection(loaded[0]) is None


def test_writer_sorts_records_by_numeric_run_id(tmp_path: Path) -> None:
    path = tmp_path / "ground_designs.jsonl"
    write_ground_design_manifest(
        [disabled_record(10, HASH_A), disabled_record(2, HASH_B)],
        path,
    )
    objects = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["run_id"] for item in objects] == [2, 10]
    assert list(objects[0]) == sorted(objects[0])
    assert path.read_bytes().endswith(b"\n")
    assert b"\r\n" not in path.read_bytes()


def test_existing_manifest_not_overwritten_by_default(tmp_path: Path) -> None:
    path = tmp_path / "ground_designs.jsonl"
    write_ground_design_manifest([enabled_record()], path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_ground_design_manifest([disabled_record()], path)
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_explicit_overwrite_replaces_manifest(tmp_path: Path) -> None:
    path = tmp_path / "ground_designs.jsonl"
    write_ground_design_manifest([enabled_record()], path)
    write_ground_design_manifest([disabled_record()], path, overwrite=True)
    assert read_ground_design_manifest(path) == (disabled_record(),)


def test_manifest_requires_jsonl_extension(tmp_path: Path) -> None:
    path = tmp_path / "ground_designs.json"
    with pytest.raises(ValueError, match=".jsonl"):
        write_ground_design_manifest([enabled_record()], path)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=".jsonl"):
        read_ground_design_manifest(path)


@pytest.mark.parametrize("content", ["", "\n", "{}\n\n{}\n"])
def test_empty_manifest_or_empty_lines_fail(tmp_path: Path, content: str) -> None:
    path = tmp_path / "ground_designs.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_ground_design_manifest(path)


def test_duplicate_json_keys_fail_with_line_context(tmp_path: Path) -> None:
    path = tmp_path / "ground_designs.jsonl"
    path.write_text('{"run_id":1,"run_id":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Line 1.*Duplicate JSON key"):
        read_ground_design_manifest(path)


def test_malformed_json_fails_with_line_context(tmp_path: Path) -> None:
    path = tmp_path / "ground_designs.jsonl"
    path.write_text('{"run_id":\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Line 1.*malformed JSON"):
        read_ground_design_manifest(path)


def test_unknown_and_missing_manifest_fields_fail(tmp_path: Path) -> None:
    record = enabled_record().to_manifest_object()
    path = tmp_path / "ground_designs.jsonl"
    with_extra = dict(record)
    with_extra["unexpected"] = "value"
    path.write_text(json.dumps(with_extra) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown=.*unexpected"):
        read_ground_design_manifest(path)
    without_field = dict(record)
    del without_field["catalog_hash"]
    path.write_text(json.dumps(without_field) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing=.*catalog_hash"):
        read_ground_design_manifest(path)


@pytest.mark.parametrize("run_id", ["1", "01", True, 1.0])
def test_manifest_run_id_rejects_nonintegers(tmp_path: Path, run_id: object) -> None:
    record = enabled_record().to_manifest_object()
    record["run_id"] = run_id
    path = tmp_path / "ground_designs.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(TypeError, match="run_id"):
        read_ground_design_manifest(path)


@pytest.mark.parametrize("field_name", ["civilian_count", "government_count", "military_count"])
def test_manifest_counts_reject_booleans(tmp_path: Path, field_name: str) -> None:
    record = enabled_record().to_manifest_object()
    record[field_name] = True
    path = tmp_path / "ground_designs.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(TypeError, match=field_name):
        read_ground_design_manifest(path)


def test_manifest_ground_enabled_requires_boolean(tmp_path: Path) -> None:
    record = enabled_record().to_manifest_object()
    record["ground_segment_enabled"] = 1
    path = tmp_path / "ground_designs.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(TypeError, match="ground_segment_enabled"):
        read_ground_design_manifest(path)


def test_duplicate_run_ids_fail_on_read_and_write(tmp_path: Path) -> None:
    path = tmp_path / "ground_designs.jsonl"
    record = enabled_record()
    lines = [json.dumps(record.to_manifest_object()), json.dumps(record.to_manifest_object())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_ground_design_manifest(path)
    with pytest.raises(ValueError, match="Duplicate"):
        write_ground_design_manifest([record, record], tmp_path / "other.jsonl")


def test_manifest_completeness_and_satellite_hashes() -> None:
    records = [enabled_record(0, HASH_A), disabled_record(1, HASH_B)]
    validate_manifest_against_satellite_runs(records, {0: HASH_A, 1: HASH_B})
    with pytest.raises(ValueError, match=r"missing=\[2\]"):
        validate_manifest_against_satellite_runs(records, {0: HASH_A, 1: HASH_B, 2: HASH_A})
    with pytest.raises(ValueError, match=r"orphan=\[1\]"):
        validate_manifest_against_satellite_runs(records, {0: HASH_A})
    with pytest.raises(ValueError, match="does not match"):
        validate_manifest_against_satellite_runs(records, {0: HASH_B, 1: HASH_B})


def test_enabled_reconstruction_requires_matching_catalog() -> None:
    record = enabled_record()
    with pytest.raises(ValueError, match="requires a validated catalog"):
        reconstruct_ground_selection(record)
    catalog = load_ground_station_catalog(FIXTURE)
    changed = replace(catalog.stations[0], name="Changed Synthetic Name")
    changed_catalog = type(catalog)((changed,) + catalog.stations[1:])
    with pytest.raises(ValueError, match="catalog hash"):
        reconstruct_ground_selection(record, changed_catalog)


def test_reordered_persisted_ids_fail() -> None:
    record = enabled_record()
    civilian_ids = tuple(reversed(record.selected_civilian_station_ids))
    tampered = replace(
        record,
        selected_civilian_station_ids=civilian_ids,
        selected_station_ids=(
            civilian_ids
            + record.selected_government_station_ids
            + record.selected_military_station_ids
        ),
    )
    with pytest.raises(ValueError, match="canonical deterministic prefixes"):
        reconstruct_ground_selection(tampered, load_ground_station_catalog(FIXTURE))


def test_unknown_disabled_and_wrong_class_ids_fail() -> None:
    record = enabled_record()
    catalog = load_ground_station_catalog(FIXTURE)
    unselected_government_id = next(
        station.station_id
        for station in catalog.eligible(GroundStationClass.GOVERNMENT)
        if station.station_id not in record.selected_government_station_ids
    )
    replacements = [
        ("CIV_UNKNOWN_999", "does not exist"),
        ("CIV_TEST_DISABLED", "disabled"),
        (unselected_government_id, "wrong class"),
    ]
    for identifier, message in replacements:
        civilian_ids = (identifier,) + record.selected_civilian_station_ids[1:]
        combined = (
            civilian_ids
            + record.selected_government_station_ids
            + record.selected_military_station_ids
        )
        tampered = replace(
            record,
            selected_civilian_station_ids=civilian_ids,
            selected_station_ids=combined,
        )
        with pytest.raises(ValueError, match=message):
            reconstruct_ground_selection(tampered, catalog)


def test_nonprefix_persisted_ids_fail_without_reselection() -> None:
    record = enabled_record()
    catalog = load_ground_station_catalog(FIXTURE)
    unselected = next(
        station.station_id
        for station in catalog.eligible(GroundStationClass.CIVILIAN)
        if station.station_id not in record.selected_civilian_station_ids
    )
    civilian_ids = (unselected,) + record.selected_civilian_station_ids[1:]
    tampered = replace(
        record,
        selected_civilian_station_ids=civilian_ids,
        selected_station_ids=(
            civilian_ids
            + record.selected_government_station_ids
            + record.selected_military_station_ids
        ),
    )
    with pytest.raises(ValueError, match="canonical deterministic prefixes"):
        reconstruct_ground_selection(tampered, catalog)


def test_ground_design_hash_is_run_and_satellite_independent() -> None:
    first = enabled_record(0, HASH_A)
    second = enabled_record(99, HASH_B)
    assert first.ground_design_hash == second.ground_design_hash
    assert first.selection_hash == second.selection_hash


def test_wrong_persisted_count_fails_closed() -> None:
    record = enabled_record()
    with pytest.raises(ValueError, match="civilian_count"):
        replace(record, civilian_count=record.civilian_count + 1)


def test_selection_hash_mismatch_fails_closed() -> None:
    record = enabled_record()
    with pytest.raises(ValueError, match="ground_design_hash"):
        replace(record, selection_hash="c" * 64)


def test_invalid_schema_and_hashes_fail() -> None:
    record = enabled_record()
    with pytest.raises(ValueError, match="schema"):
        replace(record, ground_design_schema_version="2")
    with pytest.raises(ValueError, match="satellite_config_hash"):
        replace(record, satellite_config_hash="not-a-hash")
    with pytest.raises(ValueError, match="ground_design_hash"):
        replace(record, ground_design_hash="0" * 64)
    assert record.ground_design_schema_version == GROUND_DESIGN_SCHEMA_VERSION
