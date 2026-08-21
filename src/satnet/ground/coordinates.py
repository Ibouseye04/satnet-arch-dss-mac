from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
import re

from satnet.ground.canonical import canonical_utc_timestamp
from satnet.ground.catalog import GroundStation

GROUND_VISIBILITY_MODEL_VERSION = "1"
GROUND_VISIBILITY_FRAME_CONTRACT_VERSION = "hypatia_ecef_gmst_v1"
GROUND_WGS84_MODEL_VERSION = "wgs84_geodetic_ecef_v1"
WGS84_SEMI_MAJOR_AXIS_KM = 6378.137
WGS84_FLATTENING = 1.0 / 298.257223563
WGS84_FIRST_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
WGS84_SEMI_MINOR_AXIS_KM = WGS84_SEMI_MAJOR_AXIS_KM * (1.0 - WGS84_FLATTENING)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CoordinateFrame(str, Enum):
    ECEF = "ecef"
    TEME = "teme"
    ECI = "eci"


def _finite_coordinate(value: int | float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be an integer or float")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return 0.0 if normalized == 0.0 else normalized


@dataclass(frozen=True)
class EcefPosition:
    x_km: float
    y_km: float
    z_km: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_km", _finite_coordinate(self.x_km, "x_km"))
        object.__setattr__(self, "y_km", _finite_coordinate(self.y_km, "y_km"))
        object.__setattr__(self, "z_km", _finite_coordinate(self.z_km, "z_km"))


@dataclass(frozen=True)
class FramedSatellitePosition:
    satellite_id: int
    timestamp_utc: datetime
    x_km: float
    y_km: float
    z_km: float
    frame: CoordinateFrame

    def __post_init__(self) -> None:
        if type(self.satellite_id) is not int or self.satellite_id < 0:
            raise TypeError("satellite_id must be a nonnegative integer")
        canonical_utc_timestamp(self.timestamp_utc)
        object.__setattr__(self, "x_km", _finite_coordinate(self.x_km, "x_km"))
        object.__setattr__(self, "y_km", _finite_coordinate(self.y_km, "y_km"))
        object.__setattr__(self, "z_km", _finite_coordinate(self.z_km, "z_km"))
        if not isinstance(self.frame, CoordinateFrame):
            raise TypeError("frame must be a CoordinateFrame")


@dataclass(frozen=True)
class OperationalSatellitePositionSnapshot:
    timestep_index: int
    timestamp_utc: datetime
    satellite_config_hash: str
    positions: tuple[FramedSatellitePosition, ...]

    def __post_init__(self) -> None:
        if type(self.timestep_index) is not int or self.timestep_index < 0:
            raise TypeError("timestep_index must be a nonnegative integer")
        canonical_utc_timestamp(self.timestamp_utc)
        if not isinstance(self.satellite_config_hash, str) or not SHA256_PATTERN.fullmatch(
            self.satellite_config_hash
        ):
            raise ValueError(
                "satellite_config_hash must be exactly 64 lowercase hexadecimal characters"
            )
        if not isinstance(self.positions, tuple):
            raise TypeError("positions must be a tuple")
        if any(not isinstance(position, FramedSatellitePosition) for position in self.positions):
            raise TypeError("Every position must be a FramedSatellitePosition")
        identifiers = [position.satellite_id for position in self.positions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Operational position snapshot contains duplicate satellite IDs")
        if identifiers != sorted(identifiers):
            raise ValueError("Operational positions must use ascending numeric satellite order")
        for position in self.positions:
            if position.timestamp_utc != self.timestamp_utc:
                raise ValueError("Every position timestamp must match the snapshot timestamp")
            if position.frame is not CoordinateFrame.ECEF:
                raise ValueError("Ground Visibility Model Version 1 accepts only ECEF positions")


def ground_station_to_ecef(station: GroundStation) -> EcefPosition:
    if not isinstance(station, GroundStation):
        raise TypeError("station must be a GroundStation")
    latitude_rad = math.radians(station.latitude_deg)
    longitude_rad = math.radians(station.longitude_deg)
    altitude_km = station.altitude_m / 1000.0
    sin_latitude = math.sin(latitude_rad)
    cos_latitude = math.cos(latitude_rad)
    sin_longitude = math.sin(longitude_rad)
    cos_longitude = math.cos(longitude_rad)
    prime_vertical_radius = WGS84_SEMI_MAJOR_AXIS_KM / math.sqrt(
        1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED * sin_latitude**2
    )
    return EcefPosition(
        x_km=(prime_vertical_radius + altitude_km) * cos_latitude * cos_longitude,
        y_km=(prime_vertical_radius + altitude_km) * cos_latitude * sin_longitude,
        z_km=(
            prime_vertical_radius * (1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED)
            + altitude_km
        )
        * sin_latitude,
    )
