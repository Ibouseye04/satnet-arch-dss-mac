"""Permanent Adaptive-v2 ground-segment known-answer validation tests."""

from __future__ import annotations

from datetime import datetime, timezone
from math import cos, radians, sin

import pytest

from satnet.ground.catalog import GroundStation, GroundStationClass
from satnet.ground.coordinates import (
    CoordinateFrame,
    FramedSatellitePosition,
    OperationalSatellitePositionSnapshot,
    WGS84_SEMI_MAJOR_AXIS_KM,
    WGS84_SEMI_MINOR_AXIS_KM,
    ground_station_to_ecef,
)
from satnet.ground.visibility import (
    GroundVisibilityPolicy,
    calculate_topocentric_geometry,
    evaluate_ground_visibility,
)


TIMESTAMP = datetime(2026, 7, 19, 0, 0, 0, tzinfo=timezone.utc)
SATELLITE_CONFIG_HASH = "a" * 64
GROUND_DESIGN_HASH = "b" * 64


def _station(
    *,
    latitude_deg: float = 0.0,
    longitude_deg: float = 0.0,
    altitude_m: float = 0.0,
) -> GroundStation:
    return GroundStation(
        station_id="CIV_VALIDATION_001",
        name="Validation Station",
        station_class=GroundStationClass.CIVILIAN,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        altitude_m=altitude_m,
        region="validation_region",
        country_code="US",
    )


def _position(
    satellite_id: int,
    *,
    x_km: float,
    y_km: float,
    z_km: float,
) -> FramedSatellitePosition:
    return FramedSatellitePosition(
        satellite_id=satellite_id,
        timestamp_utc=TIMESTAMP,
        x_km=x_km,
        y_km=y_km,
        z_km=z_km,
        frame=CoordinateFrame.ECEF,
    )


def test_wgs84_equator_prime_meridian_known_answer() -> None:
    """Sea-level equator/prime-meridian maps to the WGS-84 semi-major axis."""
    ecef = ground_station_to_ecef(_station())

    assert ecef.x_km == pytest.approx(6378.137, abs=1e-12)
    assert ecef.x_km == pytest.approx(WGS84_SEMI_MAJOR_AXIS_KM, abs=1e-12)
    assert ecef.y_km == pytest.approx(0.0, abs=1e-12)
    assert ecef.z_km == pytest.approx(0.0, abs=1e-12)


def test_wgs84_north_pole_known_answer() -> None:
    """The WGS-84 north pole lies on the semi-minor axis."""
    ecef = ground_station_to_ecef(_station(latitude_deg=90.0))

    assert ecef.x_km == pytest.approx(0.0, abs=1e-9)
    assert ecef.y_km == pytest.approx(0.0, abs=1e-9)
    assert ecef.z_km == pytest.approx(
        WGS84_SEMI_MINOR_AXIS_KM,
        abs=1e-9,
    )


def test_wgs84_altitude_is_applied_in_kilometers() -> None:
    """A 1000-m station altitude adds exactly 1 km at the equator."""
    sea_level = ground_station_to_ecef(_station())
    elevated = ground_station_to_ecef(_station(altitude_m=1000.0))

    assert elevated.x_km - sea_level.x_km == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_topocentric_directly_overhead_known_answer() -> None:
    """A satellite 550 km directly above the station has 90° elevation."""
    station = _station()
    station_ecef = ground_station_to_ecef(station)

    satellite = _position(
        0,
        x_km=station_ecef.x_km + 550.0,
        y_km=station_ecef.y_km,
        z_km=station_ecef.z_km,
    )

    elevation_deg, slant_range_km = calculate_topocentric_geometry(
        station,
        satellite,
    )

    assert elevation_deg == pytest.approx(90.0, abs=1e-12)
    assert slant_range_km == pytest.approx(550.0, abs=1e-12)


def _satellite_at_local_elevation(
    *,
    satellite_id: int,
    elevation_deg: float,
    slant_range_km: float = 1000.0,
) -> FramedSatellitePosition:
    """Construct an ENU-known point for the equator/prime-meridian station.

    At lat=0°, lon=0°:
      local up    -> +ECEF X
      local east  -> +ECEF Y
      local north -> +ECEF Z
    """
    station_ecef = ground_station_to_ecef(_station())

    angle = radians(elevation_deg)
    up_km = slant_range_km * sin(angle)
    east_km = slant_range_km * cos(angle)

    return _position(
        satellite_id,
        x_km=station_ecef.x_km + up_km,
        y_km=station_ecef.y_km + east_km,
        z_km=station_ecef.z_km,
    )


@pytest.mark.parametrize(
    ("requested_elevation_deg", "expected_visible"),
    [
        (9.999, False),
        (10.0, True),
        (10.001, True),
    ],
)
def test_frozen_ten_degree_visibility_boundary(
    requested_elevation_deg: float,
    expected_visible: bool,
) -> None:
    """Ground visibility uses the frozen inclusive elevation >= 10° rule."""
    station = _station()
    satellite = _satellite_at_local_elevation(
        satellite_id=0,
        elevation_deg=requested_elevation_deg,
    )

    elevation_deg, slant_range_km = calculate_topocentric_geometry(
        station,
        satellite,
    )

    assert elevation_deg == pytest.approx(
        requested_elevation_deg,
        abs=1e-10,
    )
    assert slant_range_km == pytest.approx(1000.0, abs=1e-9)

    snapshot = OperationalSatellitePositionSnapshot(
        timestep_index=0,
        timestamp_utc=TIMESTAMP,
        satellite_config_hash=SATELLITE_CONFIG_HASH,
        positions=(satellite,),
    )

    visibility = evaluate_ground_visibility(
        selected_stations=(station,),
        satellite_snapshot=snapshot,
        policy=GroundVisibilityPolicy(minimum_elevation_deg=10.0),
        ground_design_hash=GROUND_DESIGN_HASH,
    )

    assert len(visibility.link_observations) == 1
    assert visibility.link_observations[0].is_visible is expected_visible

    expected_count = 1 if expected_visible else 0
    assert len(visibility.visible_links) == expected_count


# ---------------------------------------------------------------------------
# Deterministic ground-failure sampling known answers
# ---------------------------------------------------------------------------

from satnet.ground.failure_policy import GroundFailurePolicy
from satnet.ground.failure_realization import (
    ground_failure_trial_bytes,
    ground_failure_trial_digest,
    ground_station_fails,
)


GROUND_FAILURE_TEST_STATION_ID = "CIV_VALIDATION_001"
GROUND_FAILURE_TEST_SEED = 20260719

EXPECTED_GROUND_FAILURE_TRIAL_JSON = (
    '{"ground_failure_model_version":"1",'
    '"ground_failure_sampling_version":"1",'
    '"ground_failure_seed":20260719,'
    '"identity_domain":"satnet_ground_failure_trial",'
    '"identity_version":"1",'
    '"station_id":"CIV_VALIDATION_001"}'
)

EXPECTED_GROUND_FAILURE_TRIAL_SHA256 = (
    "21777910faac503731a0a77e69bcd445"
    "319b198db9d538336004d659b9adde8b"
)


def test_ground_failure_trial_canonical_bytes_known_answer() -> None:
    """Freeze the canonical payload used for deterministic station sampling."""
    actual = ground_failure_trial_bytes(
        station_id=GROUND_FAILURE_TEST_STATION_ID,
        ground_failure_seed=GROUND_FAILURE_TEST_SEED,
    )

    assert actual == EXPECTED_GROUND_FAILURE_TRIAL_JSON.encode("utf-8")


def test_ground_failure_trial_sha256_known_answer() -> None:
    """Freeze the SHA-256 digest for the controlled station/seed pair."""
    digest = ground_failure_trial_digest(
        station_id=GROUND_FAILURE_TEST_STATION_ID,
        ground_failure_seed=GROUND_FAILURE_TEST_SEED,
    )

    assert digest.hex() == EXPECTED_GROUND_FAILURE_TRIAL_SHA256


@pytest.mark.parametrize(
    ("failure_probability", "expected_failed"),
    [
        (0.0, False),
        (0.10, False),
        (0.20, True),
        (0.50, True),
        (1.0, True),
    ],
)
def test_ground_station_failure_threshold_known_answers(
    failure_probability: float,
    expected_failed: bool,
) -> None:
    """The controlled digest lies at approximately the 13.07th percentile."""
    policy = GroundFailurePolicy(
        ground_station_failure_probability=failure_probability
    )

    actual = ground_station_fails(
        station_id=GROUND_FAILURE_TEST_STATION_ID,
        ground_failure_seed=GROUND_FAILURE_TEST_SEED,
        policy=policy,
    )

    assert actual is expected_failed


# ---------------------------------------------------------------------------
# Full ground-failure realization / replay known answers
# ---------------------------------------------------------------------------

from satnet.ground.catalog import GroundStationCatalog
from satnet.ground.failure_realization import (
    sample_ground_failure_realization,
    validate_ground_failure_realization_context,
)
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.selection import (
    GroundSegmentEnabledConfig,
    select_ground_stations,
)


def _controlled_six_station_catalog() -> GroundStationCatalog:
    """Six-station architecture used for the controlled G5 failure case."""
    stations = (
        GroundStation(
            station_id="CIV_TEST_001",
            name="Civilian Test 1",
            station_class=GroundStationClass.CIVILIAN,
            latitude_deg=0.0,
            longitude_deg=0.0,
            altitude_m=0.0,
            region="test_region",
            country_code="US",
        ),
        GroundStation(
            station_id="CIV_TEST_002",
            name="Civilian Test 2",
            station_class=GroundStationClass.CIVILIAN,
            latitude_deg=1.0,
            longitude_deg=1.0,
            altitude_m=0.0,
            region="test_region",
            country_code="US",
        ),
        GroundStation(
            station_id="GOV_TEST_001",
            name="Government Test 1",
            station_class=GroundStationClass.GOVERNMENT,
            latitude_deg=2.0,
            longitude_deg=2.0,
            altitude_m=0.0,
            region="test_region",
            country_code="US",
        ),
        GroundStation(
            station_id="GOV_TEST_002",
            name="Government Test 2",
            station_class=GroundStationClass.GOVERNMENT,
            latitude_deg=3.0,
            longitude_deg=3.0,
            altitude_m=0.0,
            region="test_region",
            country_code="US",
        ),
        GroundStation(
            station_id="MIL_TEST_001",
            name="Military Test 1",
            station_class=GroundStationClass.MILITARY,
            latitude_deg=4.0,
            longitude_deg=4.0,
            altitude_m=0.0,
            region="test_region",
            country_code="US",
        ),
        GroundStation(
            station_id="MIL_TEST_002",
            name="Military Test 2",
            station_class=GroundStationClass.MILITARY,
            latitude_deg=5.0,
            longitude_deg=5.0,
            altitude_m=0.0,
            region="test_region",
            country_code="US",
        ),
    )
    return GroundStationCatalog(stations)


def _controlled_ground_failure_context():
    catalog = _controlled_six_station_catalog()

    # Request all six stations. The selection seed remains part of the
    # canonical design identity even though the full population is selected.
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(
            civilian_count=2,
            government_count=2,
            military_count=2,
            station_selection_seed=7,
        ),
    )

    design = make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash="c" * 64,
        selection=selection,
    )

    policy = GroundFailurePolicy(
        ground_station_failure_probability=0.40
    )

    return catalog, selection, design, policy


def test_controlled_six_station_failure_realization_known_answer() -> None:
    """Reproduce the independently verified p=.40, seed=123456789 G5 case."""
    catalog, selection, design, policy = _controlled_ground_failure_context()

    realization = sample_ground_failure_realization(
        ground_design=design,
        catalog=catalog,
        policy=policy,
        ground_failure_seed=123456789,
    )

    assert set(selection.selected_station_ids) == {
        "CIV_TEST_001",
        "CIV_TEST_002",
        "GOV_TEST_001",
        "GOV_TEST_002",
        "MIL_TEST_001",
        "MIL_TEST_002",
    }

    assert realization.selected_station_ids == (
        "CIV_TEST_001",
        "CIV_TEST_002",
        "GOV_TEST_001",
        "GOV_TEST_002",
        "MIL_TEST_001",
        "MIL_TEST_002",
    )

    assert realization.failed_station_ids == (
        "CIV_TEST_002",
        "GOV_TEST_001",
        "GOV_TEST_002",
    )

    assert realization.operational_station_ids == (
        "CIV_TEST_001",
        "MIL_TEST_001",
        "MIL_TEST_002",
    )

    assert realization.total_ground_station_count == 6
    assert realization.failed_ground_station_count == 3
    assert realization.operational_ground_station_count == 3

    assert set(realization.failed_station_ids).isdisjoint(
        realization.operational_station_ids
    )
    assert (
        set(realization.failed_station_ids)
        | set(realization.operational_station_ids)
        == set(realization.selected_station_ids)
    )


def test_ground_failure_realization_replay_is_exact() -> None:
    """Same authoritative context and seed reproduce the identical realization."""
    catalog, _selection, design, policy = _controlled_ground_failure_context()

    first = sample_ground_failure_realization(
        ground_design=design,
        catalog=catalog,
        policy=policy,
        ground_failure_seed=123456789,
    )

    second = sample_ground_failure_realization(
        ground_design=design,
        catalog=catalog,
        policy=policy,
        ground_failure_seed=123456789,
    )

    assert second == first
    assert (
        second.ground_failure_realization_hash
        == first.ground_failure_realization_hash
    )

    # Production replay validator must accept the realization.
    validate_ground_failure_realization_context(
        realization=first,
        ground_design=design,
        catalog=catalog,
        policy=policy,
    )
