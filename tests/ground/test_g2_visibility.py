from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from satnet.ground.catalog import GroundStation, GroundStationCatalog, GroundStationClass, load_ground_station_catalog
from satnet.ground.coordinates import (
    CoordinateFrame,
    FramedSatellitePosition,
    OperationalSatellitePositionSnapshot,
    WGS84_SEMI_MAJOR_AXIS_KM,
)
from satnet.ground.persistence import (
    make_disabled_ground_design_record,
    make_enabled_ground_design_record,
)
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.ground.visibility import (
    MIN_VALID_SLANT_RANGE_KM,
    GroundVisibilityPolicy,
    calculate_topocentric_geometry,
    evaluate_ground_design_visibility,
    evaluate_ground_visibility,
    evaluate_ground_visibility_sequence,
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


def station(
    station_id: str = "CIV_G2_TEST_001",
    *,
    latitude_deg: float = 0.0,
    longitude_deg: float = 0.0,
    altitude_m: float = 0.0,
    enabled: bool = True,
) -> GroundStation:
    return GroundStation(
        station_id=station_id,
        name=f"Synthetic {station_id}",
        station_class=GroundStationClass.CIVILIAN,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        altitude_m=altitude_m,
        region="region_alpha",
        country_code="ZZ",
        enabled=enabled,
    )


def satellite(
    satellite_id: int,
    *,
    x_km: float,
    y_km: float,
    z_km: float,
    timestamp: datetime = TIMESTAMP,
) -> FramedSatellitePosition:
    return FramedSatellitePosition(
        satellite_id=satellite_id,
        timestamp_utc=timestamp,
        x_km=x_km,
        y_km=y_km,
        z_km=z_km,
        frame=CoordinateFrame.ECEF,
    )


def source(
    *positions: FramedSatellitePosition,
    timestep: int = 0,
    timestamp: datetime = TIMESTAMP,
    satellite_hash: str = SATELLITE_HASH,
) -> OperationalSatellitePositionSnapshot:
    return OperationalSatellitePositionSnapshot(
        timestep_index=timestep,
        timestamp_utc=timestamp,
        satellite_config_hash=satellite_hash,
        positions=tuple(positions),
    )


def overhead(
    satellite_id: int = 0,
    height_km: float = 500.0,
    timestamp: datetime = TIMESTAMP,
) -> FramedSatellitePosition:
    return satellite(
        satellite_id,
        x_km=WGS84_SEMI_MAJOR_AXIS_KM + height_km,
        y_km=0.0,
        z_km=0.0,
        timestamp=timestamp,
    )


def horizon(satellite_id: int = 0, east_km: float = 500.0) -> FramedSatellitePosition:
    return satellite(
        satellite_id,
        x_km=WGS84_SEMI_MAJOR_AXIS_KM,
        y_km=east_km,
        z_km=0.0,
    )


def below_horizon(satellite_id: int = 0) -> FramedSatellitePosition:
    return satellite(
        satellite_id,
        x_km=WGS84_SEMI_MAJOR_AXIS_KM - 100.0,
        y_km=500.0,
        z_km=0.0,
    )


@pytest.mark.parametrize("minimum", [0.0, 10.0, 90.0])
def test_overhead_satellite_is_ninety_degrees_and_visible(minimum: float) -> None:
    elevation, slant_range = calculate_topocentric_geometry(station(), overhead())
    assert elevation == pytest.approx(90.0, abs=1e-12)
    assert slant_range == pytest.approx(500.0, abs=1e-12)
    snapshot = evaluate_ground_visibility(
        selected_stations=[station()],
        satellite_snapshot=source(overhead()),
        policy=GroundVisibilityPolicy(minimum),
        ground_design_hash=GROUND_HASH,
    )
    assert snapshot.link_observations[0].is_visible is True


def test_horizon_is_exactly_inclusive() -> None:
    elevation, slant_range = calculate_topocentric_geometry(station(), horizon())
    assert elevation == pytest.approx(0.0, abs=1e-12)
    assert slant_range == pytest.approx(500.0, abs=1e-12)
    snapshot = evaluate_ground_visibility(
        selected_stations=[station()],
        satellite_snapshot=source(horizon()),
        policy=GroundVisibilityPolicy(0.0),
        ground_design_hash=GROUND_HASH,
    )
    assert snapshot.link_observations[0].elevation_deg == 0.0
    assert snapshot.link_observations[0].is_visible is True


def test_below_horizon_is_not_visible() -> None:
    elevation, _ = calculate_topocentric_geometry(station(), below_horizon())
    assert elevation < 0.0
    snapshot = evaluate_ground_visibility(
        selected_stations=[station()],
        satellite_snapshot=source(below_horizon()),
        policy=GroundVisibilityPolicy(0.0),
        ground_design_hash=GROUND_HASH,
    )
    assert snapshot.link_observations[0].is_visible is False
    assert snapshot.visible_links == ()


def test_threshold_below_exact_and_above_have_no_epsilon() -> None:
    local_up_values = (-1e-6, 0.0, 1e-6)
    results = []
    for index, local_up in enumerate(local_up_values):
        position = satellite(
            index,
            x_km=WGS84_SEMI_MAJOR_AXIS_KM + local_up,
            y_km=1000.0,
            z_km=0.0,
        )
        results.append(
            evaluate_ground_visibility(
                selected_stations=[station()],
                satellite_snapshot=source(position),
                policy=GroundVisibilityPolicy(0.0),
                ground_design_hash=GROUND_HASH,
            ).link_observations[0].is_visible
        )
    assert results == [False, True, True]


def test_degenerate_slant_range_fails() -> None:
    colocated = satellite(
        0,
        x_km=WGS84_SEMI_MAJOR_AXIS_KM + MIN_VALID_SLANT_RANGE_KM / 2.0,
        y_km=0.0,
        z_km=0.0,
    )
    with pytest.raises(ValueError, match="Degenerate geometry"):
        calculate_topocentric_geometry(station(), colocated)


def test_slant_range_matches_euclidean_common_frame_distance() -> None:
    position = satellite(
        0,
        x_km=WGS84_SEMI_MAJOR_AXIS_KM + 300.0,
        y_km=400.0,
        z_km=500.0,
    )
    _, slant_range = calculate_topocentric_geometry(station(), position)
    assert slant_range == pytest.approx((300.0**2 + 400.0**2 + 500.0**2) ** 0.5)


def test_station_altitude_and_coordinates_change_geometry() -> None:
    position = overhead()
    _, sea_level_range = calculate_topocentric_geometry(station(), position)
    _, elevated_range = calculate_topocentric_geometry(
        station(altitude_m=1000.0), position
    )
    _, shifted_range = calculate_topocentric_geometry(
        station(latitude_deg=10.0, longitude_deg=20.0), position
    )
    assert elevated_range == pytest.approx(sea_level_range - 1.0)
    assert shifted_range != pytest.approx(sea_level_range)


@pytest.mark.parametrize("minimum", [True, -0.1, 90.1, float("nan"), "10"])
def test_policy_rejects_invalid_minimum(minimum: object) -> None:
    with pytest.raises((TypeError, ValueError), match="minimum_elevation_deg"):
        GroundVisibilityPolicy(minimum)


def test_policy_hash_is_deterministic_and_value_sensitive() -> None:
    assert GroundVisibilityPolicy(10.0).visibility_policy_hash == GroundVisibilityPolicy(
        10
    ).visibility_policy_hash
    assert GroundVisibilityPolicy(10.0).visibility_policy_hash != GroundVisibilityPolicy(
        20.0
    ).visibility_policy_hash


def test_all_pairs_are_retained_and_visible_subset_is_exact() -> None:
    stations = [station("CIV_G2_TEST_002"), station("CIV_G2_TEST_001")]
    satellites = source(overhead(2), below_horizon(10))
    snapshot = evaluate_ground_visibility(
        selected_stations=stations,
        satellite_snapshot=satellites,
        policy=GroundVisibilityPolicy(0.0),
        ground_design_hash=GROUND_HASH,
    )
    assert len(snapshot.link_observations) == 4
    assert len(snapshot.visible_links) == 2
    assert tuple(
        (item.station_id, item.satellite_id) for item in snapshot.link_observations
    ) == (
        ("CIV_G2_TEST_001", 2),
        ("CIV_G2_TEST_001", 10),
        ("CIV_G2_TEST_002", 2),
        ("CIV_G2_TEST_002", 10),
    )
    assert snapshot.visible_links == tuple(
        item for item in snapshot.link_observations if item.is_visible
    )
    assert snapshot.visible_satellite_ids_by_station == (
        ("CIV_G2_TEST_001", (2,)),
        ("CIV_G2_TEST_002", (2,)),
    )


def test_station_and_satellite_input_order_do_not_change_snapshot() -> None:
    first_station = station("CIV_G2_TEST_001")
    second_station = station("CIV_G2_TEST_002")
    first = evaluate_ground_visibility(
        selected_stations=[first_station, second_station],
        satellite_snapshot=source(overhead(2), overhead(10)),
        policy=GroundVisibilityPolicy(0.0),
        ground_design_hash=GROUND_HASH,
    )
    second = evaluate_ground_visibility(
        selected_stations=[second_station, first_station],
        satellite_snapshot=source(overhead(2), overhead(10)),
        policy=GroundVisibilityPolicy(0.0),
        ground_design_hash=GROUND_HASH,
    )
    assert second == first


def test_empty_satellite_snapshot_produces_empty_mapping_for_every_station() -> None:
    snapshot = evaluate_ground_visibility(
        selected_stations=[station("CIV_G2_TEST_002"), station("CIV_G2_TEST_001")],
        satellite_snapshot=source(),
        policy=GroundVisibilityPolicy(10.0),
        ground_design_hash=GROUND_HASH,
    )
    assert snapshot.link_observations == ()
    assert snapshot.visible_links == ()
    assert snapshot.visible_satellite_ids_by_station == (
        ("CIV_G2_TEST_001", ()),
        ("CIV_G2_TEST_002", ()),
    )
    assert len(snapshot.snapshot_hash) == 64


def test_empty_duplicate_and_disabled_station_inputs_fail() -> None:
    source_snapshot = source(overhead())
    policy = GroundVisibilityPolicy(0.0)
    with pytest.raises(ValueError, match="at least one"):
        evaluate_ground_visibility(
            selected_stations=[],
            satellite_snapshot=source_snapshot,
            policy=policy,
            ground_design_hash=GROUND_HASH,
        )
    duplicate = station()
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_ground_visibility(
            selected_stations=[duplicate, duplicate],
            satellite_snapshot=source_snapshot,
            policy=policy,
            ground_design_hash=GROUND_HASH,
        )
    with pytest.raises(ValueError, match="Disabled"):
        evaluate_ground_visibility(
            selected_stations=[station(enabled=False)],
            satellite_snapshot=source_snapshot,
            policy=policy,
            ground_design_hash=GROUND_HASH,
        )


def test_nonvisible_geometry_change_changes_snapshot_hash() -> None:
    first = evaluate_ground_visibility(
        selected_stations=[station()],
        satellite_snapshot=source(below_horizon()),
        policy=GroundVisibilityPolicy(0.0),
        ground_design_hash=GROUND_HASH,
    )
    changed_position = satellite(
        0,
        x_km=WGS84_SEMI_MAJOR_AXIS_KM - 200.0,
        y_km=500.0,
        z_km=0.0,
    )
    second = evaluate_ground_visibility(
        selected_stations=[station()],
        satellite_snapshot=source(changed_position),
        policy=GroundVisibilityPolicy(0.0),
        ground_design_hash=GROUND_HASH,
    )
    assert first.visible_links == second.visible_links == ()
    assert first.snapshot_hash != second.snapshot_hash


def test_identical_ecef_geometry_at_different_times_does_not_rotate_ground() -> None:
    first_position = overhead(timestamp=TIMESTAMP)
    later_timestamp = TIMESTAMP + timedelta(minutes=1)
    second_position = satellite(
        0,
        x_km=first_position.x_km,
        y_km=first_position.y_km,
        z_km=first_position.z_km,
        timestamp=later_timestamp,
    )
    first = evaluate_ground_visibility(
        selected_stations=[station()],
        satellite_snapshot=source(first_position),
        policy=GroundVisibilityPolicy(0.0),
        ground_design_hash=GROUND_HASH,
    )
    second = evaluate_ground_visibility(
        selected_stations=[station()],
        satellite_snapshot=source(
            second_position, timestep=1, timestamp=later_timestamp
        ),
        policy=GroundVisibilityPolicy(0.0),
        ground_design_hash=GROUND_HASH,
    )
    assert first.link_observations[0].elevation_deg == second.link_observations[0].elevation_deg
    assert first.link_observations[0].slant_range_km == second.link_observations[0].slant_range_km
    assert first.snapshot_hash != second.snapshot_hash


def test_visibility_sequence_is_deterministic_and_requires_temporal_order() -> None:
    later = TIMESTAMP + timedelta(minutes=1)
    sources = (
        source(overhead(timestamp=TIMESTAMP)),
        source(
            satellite(
                0,
                x_km=WGS84_SEMI_MAJOR_AXIS_KM + 500.0,
                y_km=0.0,
                z_km=0.0,
                timestamp=later,
            ),
            timestep=1,
            timestamp=later,
        ),
    )
    first = evaluate_ground_visibility_sequence(
        selected_stations=[station()],
        satellite_sequence=sources,
        policy=GroundVisibilityPolicy(10.0),
        ground_design_hash=GROUND_HASH,
    )
    second = evaluate_ground_visibility_sequence(
        selected_stations=[station()],
        satellite_sequence=sources,
        policy=GroundVisibilityPolicy(10.0),
        ground_design_hash=GROUND_HASH,
    )
    assert first == second
    assert tuple(item.timestep_index for item in first) == (0, 1)
    with pytest.raises(ValueError, match="contiguous"):
        evaluate_ground_visibility_sequence(
            selected_stations=[station()],
            satellite_sequence=(replace(sources[0], timestep_index=1),),
            policy=GroundVisibilityPolicy(10.0),
            ground_design_hash=GROUND_HASH,
        )


def test_canonical_orchestration_reconstructs_g1_selection() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(1, 0, 0, 42),
    )
    ground_design = make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash=SATELLITE_HASH,
        selection=selection,
    )
    snapshot = evaluate_ground_design_visibility(
        ground_design=ground_design,
        catalog=catalog,
        satellite_snapshot=source(overhead()),
        policy=GroundVisibilityPolicy(0.0),
    )
    assert tuple(item.station_id for item in snapshot.link_observations) == selection.selected_station_ids
    assert snapshot.ground_design_hash == ground_design.ground_design_hash


def test_canonical_orchestration_rejects_catalog_and_satellite_mismatches() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(1, 0, 0, 42),
    )
    ground_design = make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash=SATELLITE_HASH,
        selection=selection,
    )
    changed_station = replace(catalog.stations[0], name="Changed Synthetic Name")
    changed_catalog = GroundStationCatalog((changed_station,) + catalog.stations[1:])
    with pytest.raises(ValueError, match="catalog hash"):
        evaluate_ground_design_visibility(
            ground_design=ground_design,
            catalog=changed_catalog,
            satellite_snapshot=source(overhead()),
            policy=GroundVisibilityPolicy(0.0),
        )
    with pytest.raises(ValueError, match="satellite hash"):
        evaluate_ground_design_visibility(
            ground_design=ground_design,
            catalog=catalog,
            satellite_snapshot=source(overhead(), satellite_hash="c" * 64),
            policy=GroundVisibilityPolicy(0.0),
        )


def test_canonical_orchestration_rejects_disabled_ground_design() -> None:
    disabled = make_disabled_ground_design_record(
        run_id=0,
        satellite_config_hash=SATELLITE_HASH,
    )
    with pytest.raises(ValueError, match="enabled"):
        evaluate_ground_design_visibility(
            ground_design=disabled,
            catalog=load_ground_station_catalog(FIXTURE),
            satellite_snapshot=source(overhead()),
            policy=GroundVisibilityPolicy(0.0),
        )
