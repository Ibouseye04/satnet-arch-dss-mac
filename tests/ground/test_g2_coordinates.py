from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pytest

from satnet.ground.canonical import canonical_utc_timestamp
from satnet.ground.catalog import GroundStation, GroundStationClass
from satnet.ground.coordinates import (
    CoordinateFrame,
    FramedSatellitePosition,
    OperationalSatellitePositionSnapshot,
    WGS84_SEMI_MAJOR_AXIS_KM,
    WGS84_SEMI_MINOR_AXIS_KM,
    ground_station_to_ecef,
)

TIMESTAMP = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
HASH = "a" * 64


def station(
    *, latitude_deg: float = 0.0, longitude_deg: float = 0.0, altitude_m: float = 0.0
) -> GroundStation:
    return GroundStation(
        station_id="CIV_G2_TEST_001",
        name="Synthetic G2 Station",
        station_class=GroundStationClass.CIVILIAN,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        altitude_m=altitude_m,
        region="region_alpha",
        country_code="ZZ",
    )


def position(
    satellite_id: int = 0,
    *,
    timestamp: datetime = TIMESTAMP,
    frame: CoordinateFrame = CoordinateFrame.ECEF,
) -> FramedSatellitePosition:
    return FramedSatellitePosition(
        satellite_id=satellite_id,
        timestamp_utc=timestamp,
        x_km=7000.0,
        y_km=0.0,
        z_km=0.0,
        frame=frame,
    )


def test_wgs84_equator_prime_meridian() -> None:
    result = ground_station_to_ecef(station())
    assert result.x_km == pytest.approx(WGS84_SEMI_MAJOR_AXIS_KM, abs=1e-12)
    assert result.y_km == pytest.approx(0.0, abs=1e-12)
    assert result.z_km == pytest.approx(0.0, abs=1e-12)


def test_wgs84_equator_ninety_degrees_east() -> None:
    result = ground_station_to_ecef(station(longitude_deg=90.0))
    assert result.x_km == pytest.approx(0.0, abs=1e-12)
    assert result.y_km == pytest.approx(WGS84_SEMI_MAJOR_AXIS_KM, abs=1e-12)
    assert result.z_km == pytest.approx(0.0, abs=1e-12)


def test_wgs84_north_pole() -> None:
    result = ground_station_to_ecef(station(latitude_deg=90.0))
    assert result.x_km == pytest.approx(0.0, abs=1e-9)
    assert result.y_km == pytest.approx(0.0, abs=1e-9)
    assert result.z_km == pytest.approx(WGS84_SEMI_MINOR_AXIS_KM, abs=1e-9)


def test_wgs84_altitude_changes_radial_position() -> None:
    sea_level = ground_station_to_ecef(station())
    elevated = ground_station_to_ecef(station(altitude_m=1000.0))
    below = ground_station_to_ecef(station(altitude_m=-100.0))
    assert elevated.x_km - sea_level.x_km == pytest.approx(1.0, abs=1e-12)
    assert below.x_km - sea_level.x_km == pytest.approx(-0.1, abs=1e-12)


def test_wgs84_southern_and_western_coordinates() -> None:
    result = ground_station_to_ecef(station(latitude_deg=-30.0, longitude_deg=-45.0))
    assert result.x_km > 0.0
    assert result.y_km < 0.0
    assert result.z_km < 0.0


def test_canonical_utc_timestamp_has_fixed_six_digit_fraction() -> None:
    assert canonical_utc_timestamp(TIMESTAMP) == "2026-07-16T12:00:00.000000Z"
    with_microseconds = TIMESTAMP.replace(microsecond=123)
    assert canonical_utc_timestamp(with_microseconds) == "2026-07-16T12:00:00.000123Z"


def test_canonical_utc_rejects_naive_and_nonzero_offsets() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        canonical_utc_timestamp(datetime(2026, 7, 16, 12, 0, 0))
    non_utc = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone(timedelta(hours=1)))
    with pytest.raises(ValueError, match="normalized to UTC"):
        canonical_utc_timestamp(non_utc)


@pytest.mark.parametrize("satellite_id", [True, False, -1, "1", 1.0])
def test_framed_position_requires_nonnegative_integer_id(satellite_id: object) -> None:
    with pytest.raises(TypeError, match="satellite_id"):
        position(satellite_id=satellite_id)


@pytest.mark.parametrize("coordinate", [math.nan, math.inf, -math.inf, True, "1.0"])
def test_framed_position_rejects_invalid_coordinates(coordinate: object) -> None:
    with pytest.raises((TypeError, ValueError), match="x_km"):
        FramedSatellitePosition(
            satellite_id=0,
            timestamp_utc=TIMESTAMP,
            x_km=coordinate,
            y_km=0.0,
            z_km=0.0,
            frame=CoordinateFrame.ECEF,
        )


def test_operational_snapshot_accepts_empty_positions() -> None:
    snapshot = OperationalSatellitePositionSnapshot(0, TIMESTAMP, HASH, ())
    assert snapshot.positions == ()
    assert snapshot.timestep_index == 0
    assert snapshot.timestamp_utc == TIMESTAMP


def test_operational_snapshot_requires_numeric_satellite_order() -> None:
    ordered = OperationalSatellitePositionSnapshot(
        0, TIMESTAMP, HASH, (position(2), position(10))
    )
    assert tuple(item.satellite_id for item in ordered.positions) == (2, 10)
    with pytest.raises(ValueError, match="ascending numeric"):
        OperationalSatellitePositionSnapshot(
            0, TIMESTAMP, HASH, (position(10), position(2))
        )


def test_operational_snapshot_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        OperationalSatellitePositionSnapshot(
            0, TIMESTAMP, HASH, (position(2), position(2))
        )


@pytest.mark.parametrize("frame", [CoordinateFrame.TEME, CoordinateFrame.ECI])
def test_operational_snapshot_rejects_non_ecef_frames(frame: CoordinateFrame) -> None:
    with pytest.raises(ValueError, match="only ECEF"):
        OperationalSatellitePositionSnapshot(
            0, TIMESTAMP, HASH, (position(0, frame=frame),)
        )


def test_operational_snapshot_rejects_mixed_timestamp() -> None:
    later = TIMESTAMP + timedelta(seconds=1)
    with pytest.raises(ValueError, match="timestamp"):
        OperationalSatellitePositionSnapshot(
            0, TIMESTAMP, HASH, (position(0, timestamp=later),)
        )


@pytest.mark.parametrize("timestep", [True, -1, "0", 0.0])
def test_operational_snapshot_rejects_invalid_timestep(timestep: object) -> None:
    with pytest.raises(TypeError, match="timestep_index"):
        OperationalSatellitePositionSnapshot(timestep, TIMESTAMP, HASH, ())
