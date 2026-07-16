from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from satnet.ground.catalog import GroundStation, GroundStationClass, load_ground_station_catalog
from satnet.ground.coordinates import (
    CoordinateFrame,
    FramedSatellitePosition,
    OperationalSatellitePositionSnapshot,
    WGS84_SEMI_MAJOR_AXIS_KM,
)
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.ground.visibility import (
    GroundVisibilityPolicy,
    evaluate_ground_design_visibility,
    evaluate_ground_visibility,
)
from satnet.ground.visibility_persistence import (
    GroundVisibilityRecord,
    make_ground_visibility_record,
    read_ground_visibility_manifest,
    replay_ground_visibility_records,
    validate_visibility_records_against_sources,
    write_ground_visibility_manifest,
)

TIMESTAMP = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
SATELLITE_HASH = "a" * 64
GROUND_HASH = "b" * 64
FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "ground_segment"
    / "synthetic_ground_station_catalog.csv"
)


def station() -> GroundStation:
    return GroundStation(
        station_id="CIV_G2_TEST_001",
        name="Synthetic Persistence Station",
        station_class=GroundStationClass.CIVILIAN,
        latitude_deg=0.0,
        longitude_deg=0.0,
        altitude_m=0.0,
        region="region_alpha",
        country_code="ZZ",
    )


def framed_position(
    satellite_id: int,
    *,
    visible: bool,
    timestamp: datetime = TIMESTAMP,
) -> FramedSatellitePosition:
    return FramedSatellitePosition(
        satellite_id=satellite_id,
        timestamp_utc=timestamp,
        x_km=WGS84_SEMI_MAJOR_AXIS_KM + (500.0 if visible else -100.0),
        y_km=0.0 if visible else 500.0,
        z_km=0.0,
        frame=CoordinateFrame.ECEF,
    )


def pure_snapshot(timestamp: datetime = TIMESTAMP, timestep: int = 0):
    source = OperationalSatellitePositionSnapshot(
        timestep_index=timestep,
        timestamp_utc=timestamp,
        satellite_config_hash=SATELLITE_HASH,
        positions=(
            framed_position(2, visible=True, timestamp=timestamp),
            framed_position(10, visible=False, timestamp=timestamp),
        ),
    )
    return evaluate_ground_visibility(
        selected_stations=[station()],
        satellite_snapshot=source,
        policy=GroundVisibilityPolicy(0.0),
        ground_design_hash=GROUND_HASH,
    )


def canonical_ground_context():
    catalog = load_ground_station_catalog(FIXTURE)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(1, 0, 0, 42),
    )
    ground_design = make_enabled_ground_design_record(
        run_id=7,
        satellite_config_hash=SATELLITE_HASH,
        selection=selection,
    )
    policy = GroundVisibilityPolicy(10.0)
    first = OperationalSatellitePositionSnapshot(
        timestep_index=0,
        timestamp_utc=TIMESTAMP,
        satellite_config_hash=SATELLITE_HASH,
        positions=(framed_position(0, visible=True),),
    )
    second_timestamp = TIMESTAMP + timedelta(minutes=1)
    second = OperationalSatellitePositionSnapshot(
        timestep_index=1,
        timestamp_utc=second_timestamp,
        satellite_config_hash=SATELLITE_HASH,
        positions=(),
    )
    sources = (first, second)
    snapshots = tuple(
        evaluate_ground_design_visibility(
            ground_design=ground_design,
            catalog=catalog,
            satellite_snapshot=source,
            policy=policy,
        )
        for source in sources
    )
    records = tuple(
        make_ground_visibility_record(run_id=7, snapshot=snapshot)
        for snapshot in snapshots
    )
    return catalog, ground_design, policy, sources, snapshots, records


def write_object(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def test_visibility_record_round_trip_and_record_hash(tmp_path: Path) -> None:
    record = make_ground_visibility_record(run_id=3, snapshot=pure_snapshot())
    path = tmp_path / "visibility.jsonl"
    write_ground_visibility_manifest([record], path)
    assert read_ground_visibility_manifest(path) == (record,)
    assert len(record.record_hash) == 64
    assert path.read_bytes().endswith(b"\n")
    assert b"\r\n" not in path.read_bytes()


def test_writer_sorts_by_run_and_timestep(tmp_path: Path) -> None:
    first_time = TIMESTAMP
    second_time = TIMESTAMP + timedelta(minutes=1)
    records = [
        make_ground_visibility_record(run_id=10, snapshot=pure_snapshot(first_time, 0)),
        make_ground_visibility_record(run_id=2, snapshot=pure_snapshot(second_time, 1)),
        make_ground_visibility_record(run_id=2, snapshot=pure_snapshot(first_time, 0)),
    ]
    path = tmp_path / "visibility.jsonl"
    write_ground_visibility_manifest(records, path)
    values = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(value["run_id"], value["timestep_index"]) for value in values] == [
        (2, 0),
        (2, 1),
        (10, 0),
    ]
    assert all(list(value) == sorted(value) for value in values)


def test_writer_refuses_overwrite_and_cleans_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "visibility.jsonl"
    record = make_ground_visibility_record(run_id=0, snapshot=pure_snapshot())
    write_ground_visibility_manifest([record], path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_ground_visibility_manifest([record], path)
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []
    write_ground_visibility_manifest([record], path, overwrite=True)


@pytest.mark.parametrize("content", ["", "\n", "{}\n\n{}\n"])
def test_reader_rejects_empty_files_and_lines(tmp_path: Path, content: str) -> None:
    path = tmp_path / "visibility.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_ground_visibility_manifest(path)


def test_reader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "visibility.jsonl"
    path.write_text('{"run_id":1,"run_id":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        read_ground_visibility_manifest(path)


def test_reader_rejects_unknown_and_missing_fields(tmp_path: Path) -> None:
    record = make_ground_visibility_record(run_id=0, snapshot=pure_snapshot())
    value = record.to_manifest_object()
    path = tmp_path / "visibility.jsonl"
    extra = dict(value)
    extra["unexpected"] = 1
    write_object(path, extra)
    with pytest.raises(ValueError, match="unknown=.*unexpected"):
        read_ground_visibility_manifest(path)
    missing = dict(value)
    del missing["snapshot_hash"]
    write_object(path, missing)
    with pytest.raises(ValueError, match="missing=.*snapshot_hash"):
        read_ground_visibility_manifest(path)


@pytest.mark.parametrize("field_name", ["run_id", "timestep_index"])
@pytest.mark.parametrize("value", [True, "1", 1.0])
def test_record_integer_fields_are_exact(tmp_path: Path, field_name: str, value: object) -> None:
    record = make_ground_visibility_record(run_id=0, snapshot=pure_snapshot())
    manifest = record.to_manifest_object()
    manifest[field_name] = value
    path = tmp_path / "visibility.jsonl"
    write_object(path, manifest)
    with pytest.raises(TypeError, match=field_name):
        read_ground_visibility_manifest(path)


def test_alternate_equivalent_timestamp_string_is_not_canonical(tmp_path: Path) -> None:
    record = make_ground_visibility_record(run_id=0, snapshot=pure_snapshot())
    manifest = record.to_manifest_object()
    manifest["timestamp_utc"] = "2026-07-16T12:00:00Z"
    path = tmp_path / "visibility.jsonl"
    write_object(path, manifest)
    with pytest.raises(ValueError, match="six|YYYY"):
        read_ground_visibility_manifest(path)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("visibility_model_version", "2", "visibility_model_version"),
        ("frame_contract_version", "other", "frame_contract_version"),
        ("wgs84_model_version", "other", "wgs84_model_version"),
        ("snapshot_hash", "c" * 64, "snapshot_hash"),
        ("record_hash", "c" * 64, "record_hash"),
    ],
)
def test_version_and_hash_corruption_fails(
    tmp_path: Path, field_name: str, value: str, message: str
) -> None:
    record = make_ground_visibility_record(run_id=0, snapshot=pure_snapshot())
    manifest = record.to_manifest_object()
    manifest[field_name] = value
    path = tmp_path / "visibility.jsonl"
    write_object(path, manifest)
    with pytest.raises(ValueError, match=message):
        read_ground_visibility_manifest(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "remove_visible",
        "remove_nonvisible",
        "add_observation",
        "station_id",
        "satellite_id",
        "timestamp",
        "elevation",
        "slant_range",
        "visibility_flag",
        "station_mapping",
    ],
)
def test_observation_and_mapping_corruption_fails(tmp_path: Path, mutation: str) -> None:
    record = make_ground_visibility_record(run_id=0, snapshot=pure_snapshot())
    manifest = deepcopy(record.to_manifest_object())
    observations = manifest["link_observations"]
    if mutation == "remove_visible":
        del observations[0]
    elif mutation == "remove_nonvisible":
        del observations[1]
    elif mutation == "add_observation":
        observations.append(deepcopy(observations[0]))
        observations[-1]["satellite_id"] = 99
    elif mutation == "station_id":
        observations[0]["station_id"] = "CIV_G2_TEST_999"
    elif mutation == "satellite_id":
        observations[0]["satellite_id"] = 99
    elif mutation == "timestamp":
        observations[0]["timestamp_utc"] = "2026-07-16T12:00:01.000000Z"
    elif mutation == "elevation":
        observations[0]["elevation_deg"] = "89"
    elif mutation == "slant_range":
        observations[0]["slant_range_km"] = "501"
    elif mutation == "visibility_flag":
        observations[0]["is_visible"] = False
    elif mutation == "station_mapping":
        manifest["visible_satellite_ids_by_station"][0][1] = []
    path = tmp_path / "visibility.jsonl"
    write_object(path, manifest)
    with pytest.raises(ValueError):
        read_ground_visibility_manifest(path)


def test_noncanonical_float_number_is_rejected(tmp_path: Path) -> None:
    record = make_ground_visibility_record(run_id=0, snapshot=pure_snapshot())
    manifest = record.to_manifest_object()
    manifest["link_observations"][0]["elevation_deg"] = 90.0
    path = tmp_path / "visibility.jsonl"
    write_object(path, manifest)
    with pytest.raises(TypeError, match="canonical float string"):
        read_ground_visibility_manifest(path)


def test_duplicate_run_timestep_keys_fail(tmp_path: Path) -> None:
    record = make_ground_visibility_record(run_id=0, snapshot=pure_snapshot())
    with pytest.raises(ValueError, match="Duplicate"):
        write_ground_visibility_manifest([record, record], tmp_path / "visibility.jsonl")


def test_collection_completeness_and_all_failed_record() -> None:
    catalog, ground_design, policy, sources, snapshots, records = canonical_ground_context()
    validate_visibility_records_against_sources(
        records=records,
        run_id=7,
        operational_satellite_snapshots=sources,
        ground_design=ground_design,
        policy=policy,
    )
    empty_record = records[1]
    assert empty_record.link_observations == ()
    assert all(not ids for _, ids in empty_record.visible_satellite_ids_by_station)
    with pytest.raises(ValueError, match="missing"):
        validate_visibility_records_against_sources(
            records=records[:1],
            run_id=7,
            operational_satellite_snapshots=sources,
            ground_design=ground_design,
            policy=policy,
        )
    extra = make_ground_visibility_record(run_id=8, snapshot=snapshots[0])
    with pytest.raises(ValueError, match="extra"):
        validate_visibility_records_against_sources(
            records=records + (extra,),
            run_id=7,
            operational_satellite_snapshots=sources,
            ground_design=ground_design,
            policy=policy,
        )


def test_collection_rejects_timestamp_and_identity_mismatches() -> None:
    _, ground_design, policy, sources, _, records = canonical_ground_context()
    mismatched_source = OperationalSatellitePositionSnapshot(
        timestep_index=0,
        timestamp_utc=sources[0].timestamp_utc + timedelta(seconds=1),
        satellite_config_hash=SATELLITE_HASH,
        positions=(),
    )
    with pytest.raises(ValueError, match="timestamp mismatch"):
        validate_visibility_records_against_sources(
            records=records,
            run_id=7,
            operational_satellite_snapshots=(mismatched_source, sources[1]),
            ground_design=ground_design,
            policy=policy,
        )


def test_exact_replay_matches_all_timesteps() -> None:
    catalog, ground_design, policy, sources, snapshots, records = canonical_ground_context()
    replayed = replay_ground_visibility_records(
        records=records,
        run_id=7,
        operational_satellite_snapshots=sources,
        ground_design=ground_design,
        catalog=catalog,
        policy=policy,
    )
    assert replayed == snapshots
    assert len(replayed) == 2


def test_replay_rejects_different_valid_policy() -> None:
    catalog, ground_design, policy, sources, snapshots, records = canonical_ground_context()
    with pytest.raises(ValueError, match="policy hash"):
        replay_ground_visibility_records(
            records=records,
            run_id=7,
            operational_satellite_snapshots=sources,
            ground_design=ground_design,
            catalog=catalog,
            policy=GroundVisibilityPolicy(20.0),
        )
