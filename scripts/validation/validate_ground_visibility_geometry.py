from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math

from satnet.ground.canonical import canonical_utc_timestamp
from satnet.ground.catalog import GroundStation, GroundStationClass
from satnet.ground.coordinates import (
    CoordinateFrame,
    FramedSatellitePosition,
    OperationalSatellitePositionSnapshot,
    WGS84_SEMI_MAJOR_AXIS_KM,
    ground_station_to_ecef,
)
from satnet.ground.visibility import (
    GroundVisibilityPolicy,
    calculate_topocentric_geometry,
    evaluate_ground_visibility,
)

GROUND_DESIGN_HASH = "0" * 64
SATELLITE_CONFIG_HASH = "1" * 64
BASE_TIMESTAMP = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


def _station(station_id: str, latitude: float, longitude: float) -> GroundStation:
    return GroundStation(
        station_id=station_id,
        name=f"Synthetic Diagnostic {station_id}",
        station_class=GroundStationClass.CIVILIAN,
        latitude_deg=latitude,
        longitude_deg=longitude,
        altitude_m=0.0,
        region="diagnostic_region",
        country_code="ZZ",
    )


def _equatorial_position(
    satellite_id: int,
    elevation_deg: float,
    timestamp: datetime,
    slant_range_km: float = 500.0,
) -> FramedSatellitePosition:
    elevation = math.radians(elevation_deg)
    up = slant_range_km * math.sin(elevation)
    east = slant_range_km * math.cos(elevation)
    return FramedSatellitePosition(
        satellite_id=satellite_id,
        timestamp_utc=timestamp,
        x_km=WGS84_SEMI_MAJOR_AXIS_KM + up,
        y_km=east,
        z_km=0.0,
        frame=CoordinateFrame.ECEF,
    )


def build_diagnostics() -> list[dict[str, object]]:
    primary = _station("CIV_DIAG_001", 0.0, 0.0)
    secondary = _station("CIV_DIAG_002", 30.0, 45.0)
    policy = GroundVisibilityPolicy(10.0)
    cases = (
        ("overhead", 90.0),
        ("high_elevation", 45.0),
        ("threshold_boundary", 10.0),
        ("near_horizon", 0.0),
        ("below_horizon", -10.0),
    )
    results: list[dict[str, object]] = []
    for satellite_id, (case_name, elevation) in enumerate(cases):
        position = _equatorial_position(
            satellite_id,
            elevation,
            BASE_TIMESTAMP,
        )
        source = OperationalSatellitePositionSnapshot(
            timestep_index=0,
            timestamp_utc=BASE_TIMESTAMP,
            satellite_config_hash=SATELLITE_CONFIG_HASH,
            positions=(position,),
        )
        case_policy = policy
        if case_name == "threshold_boundary":
            computed_elevation, _ = calculate_topocentric_geometry(primary, position)
            case_policy = GroundVisibilityPolicy(computed_elevation)
        snapshot = evaluate_ground_visibility(
            selected_stations=(primary,),
            satellite_snapshot=source,
            policy=case_policy,
            ground_design_hash=GROUND_DESIGN_HASH,
        )
        observation = snapshot.link_observations[0]
        station_ecef = ground_station_to_ecef(primary)
        results.append(
            {
                "case": case_name,
                "station_id": primary.station_id,
                "station_geodetic": {
                    "latitude_deg": primary.latitude_deg,
                    "longitude_deg": primary.longitude_deg,
                    "altitude_m": primary.altitude_m,
                },
                "station_ecef_km": [
                    station_ecef.x_km,
                    station_ecef.y_km,
                    station_ecef.z_km,
                ],
                "satellite_id": position.satellite_id,
                "satellite_ecef_km": [position.x_km, position.y_km, position.z_km],
                "coordinate_frame": position.frame.value,
                "timestamp_utc": canonical_utc_timestamp(position.timestamp_utc),
                "elevation_deg": observation.elevation_deg,
                "slant_range_km": observation.slant_range_km,
                "minimum_elevation_deg": case_policy.minimum_elevation_deg,
                "is_visible": observation.is_visible,
            }
        )
    second_timestamp = BASE_TIMESTAMP + timedelta(minutes=1)
    multi_positions = (
        _equatorial_position(20, 60.0, second_timestamp),
        _equatorial_position(21, 20.0, second_timestamp),
    )
    multi_source = OperationalSatellitePositionSnapshot(
        timestep_index=1,
        timestamp_utc=second_timestamp,
        satellite_config_hash=SATELLITE_CONFIG_HASH,
        positions=multi_positions,
    )
    multi_snapshot = evaluate_ground_visibility(
        selected_stations=(primary, secondary),
        satellite_snapshot=multi_source,
        policy=policy,
        ground_design_hash=GROUND_DESIGN_HASH,
    )
    results.append(
        {
            "case": "multiple_stations_satellites_second_timestamp",
            "timestamp_utc": canonical_utc_timestamp(second_timestamp),
            "coordinate_frame": CoordinateFrame.ECEF.value,
            "observation_count": len(multi_snapshot.link_observations),
            "visible_link_count": len(multi_snapshot.visible_links),
            "station_mapping": [
                [station_id, list(satellite_ids)]
                for station_id, satellite_ids in multi_snapshot.visible_satellite_ids_by_station
            ],
        }
    )
    return results


def main() -> None:
    print(json.dumps(build_diagnostics(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
