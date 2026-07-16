from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
import re
from typing import Iterable

from satnet.ground.canonical import (
    canonical_float_string,
    canonical_hash,
    canonical_station_name,
)

CATALOG_IDENTITY_DOMAIN = "satnet_ground_catalog"
CATALOG_IDENTITY_VERSION = "1"
MIN_SUPPORTED_GROUND_ALTITUDE_M = -500.0
MAX_SUPPORTED_GROUND_ALTITUDE_M = 9000.0
MIN_PRODUCTION_ENABLED_STATIONS_PER_CLASS = 50
STATION_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
REGION_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
COUNTRY_CODE_PATTERN = re.compile(r"^[A-Z]{2}$")
CATALOG_COLUMNS = (
    "station_id",
    "name",
    "station_class",
    "latitude_deg",
    "longitude_deg",
    "altitude_m",
    "region",
    "country_code",
    "enabled",
)


class GroundStationClass(str, Enum):
    CIVILIAN = "civilian"
    GOVERNMENT = "government"
    MILITARY = "military"


def _validated_float(value: int | float, field_name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be an integer or float, not {type(value).__name__}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    if not minimum <= normalized <= maximum:
        raise ValueError(f"{field_name} must be within [{minimum}, {maximum}]")
    return 0.0 if normalized == 0.0 else normalized


@dataclass(frozen=True)
class GroundStation:
    station_id: str
    name: str
    station_class: GroundStationClass
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    region: str
    country_code: str
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.station_id, str) or not STATION_ID_PATTERN.fullmatch(
            self.station_id
        ):
            raise ValueError(
                "station_id must match ^[A-Z][A-Z0-9_]{2,63}$"
            )
        object.__setattr__(self, "name", canonical_station_name(self.name))
        if not isinstance(self.station_class, GroundStationClass):
            raise TypeError("station_class must be a GroundStationClass")
        object.__setattr__(
            self,
            "latitude_deg",
            _validated_float(self.latitude_deg, "latitude_deg", -90.0, 90.0),
        )
        object.__setattr__(
            self,
            "longitude_deg",
            _validated_float(self.longitude_deg, "longitude_deg", -180.0, 180.0),
        )
        object.__setattr__(
            self,
            "altitude_m",
            _validated_float(
                self.altitude_m,
                "altitude_m",
                MIN_SUPPORTED_GROUND_ALTITUDE_M,
                MAX_SUPPORTED_GROUND_ALTITUDE_M,
            ),
        )
        if not isinstance(self.region, str) or not REGION_PATTERN.fullmatch(self.region):
            raise ValueError("region must match ^[a-z][a-z0-9_]{1,63}$")
        if not isinstance(self.country_code, str) or not COUNTRY_CODE_PATTERN.fullmatch(
            self.country_code
        ):
            raise ValueError("country_code must contain exactly two uppercase ASCII letters")
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a Boolean")

    def canonical_record(self) -> dict[str, str | bool]:
        return {
            "altitude_m": canonical_float_string(self.altitude_m),
            "country_code": self.country_code,
            "enabled": self.enabled,
            "latitude_deg": canonical_float_string(self.latitude_deg),
            "longitude_deg": canonical_float_string(self.longitude_deg),
            "name": self.name,
            "region": self.region,
            "station_class": self.station_class.value,
            "station_id": self.station_id,
        }


@dataclass(frozen=True)
class GroundStationCatalog:
    stations: tuple[GroundStation, ...]
    catalog_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.stations, tuple):
            raise TypeError("stations must be a tuple")
        if not self.stations:
            raise ValueError("Ground-station catalog must contain at least one station")
        if any(not isinstance(station, GroundStation) for station in self.stations):
            raise TypeError("Every catalog record must be a GroundStation")
        identifiers = [station.station_id for station in self.stations]
        duplicates = sorted(
            identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
        )
        if duplicates:
            raise ValueError(f"Duplicate ground-station IDs: {duplicates}")
        ordered = tuple(sorted(self.stations, key=lambda station: station.station_id))
        object.__setattr__(self, "stations", ordered)
        payload = {
            "identity_domain": CATALOG_IDENTITY_DOMAIN,
            "identity_version": CATALOG_IDENTITY_VERSION,
            "stations": [station.canonical_record() for station in ordered],
        }
        object.__setattr__(self, "catalog_hash", canonical_hash(payload))

    def by_id(self) -> dict[str, GroundStation]:
        return {station.station_id: station for station in self.stations}

    def eligible(self, station_class: GroundStationClass) -> tuple[GroundStation, ...]:
        if not isinstance(station_class, GroundStationClass):
            raise TypeError("station_class must be a GroundStationClass")
        return tuple(
            station
            for station in self.stations
            if station.enabled and station.station_class is station_class
        )


def _parse_station_row(row: dict[str, str], line_number: int) -> GroundStation:
    try:
        station_class = GroundStationClass(row["station_class"])
    except ValueError as exc:
        raise ValueError(
            f"Line {line_number}: unknown station_class '{row['station_class']}'"
        ) from exc
    enabled_text = row["enabled"]
    if enabled_text not in {"true", "false"}:
        raise ValueError(
            f"Line {line_number}: enabled must be exactly 'true' or 'false'"
        )
    try:
        latitude_deg = float(row["latitude_deg"])
        longitude_deg = float(row["longitude_deg"])
        altitude_m = float(row["altitude_m"])
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: invalid numeric station field") from exc
    try:
        return GroundStation(
            station_id=row["station_id"],
            name=row["name"],
            station_class=station_class,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            altitude_m=altitude_m,
            region=row["region"],
            country_code=row["country_code"],
            enabled=enabled_text == "true",
        )
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"Line {line_number}: {exc}") from exc


def load_ground_station_catalog(path: str | Path) -> GroundStationCatalog:
    catalog_path = Path(path)
    try:
        handle = catalog_path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise OSError(f"Unable to open ground-station catalog '{catalog_path}': {exc}") from exc
    with handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Ground-station catalog is empty") from exc
        if len(header) != len(set(header)):
            raise ValueError("Ground-station catalog contains duplicate headers")
        if set(header) != set(CATALOG_COLUMNS) or len(header) != len(CATALOG_COLUMNS):
            missing = sorted(set(CATALOG_COLUMNS) - set(header))
            unexpected = sorted(set(header) - set(CATALOG_COLUMNS))
            raise ValueError(
                f"Ground-station catalog headers must exactly match the canonical schema; "
                f"missing={missing}, unexpected={unexpected}"
            )
        stations: list[GroundStation] = []
        for line_number, values in enumerate(reader, start=2):
            if len(values) != len(header):
                raise ValueError(
                    f"Line {line_number}: expected {len(header)} fields, found {len(values)}"
                )
            row = dict(zip(header, values, strict=True))
            stations.append(_parse_station_row(row, line_number))
    return GroundStationCatalog(tuple(stations))


def validate_production_catalog_readiness(
    catalog: GroundStationCatalog,
    *,
    minimum_enabled_per_class: int = MIN_PRODUCTION_ENABLED_STATIONS_PER_CLASS,
) -> None:
    if not isinstance(catalog, GroundStationCatalog):
        raise TypeError("catalog must be a GroundStationCatalog")
    if type(minimum_enabled_per_class) is not int or minimum_enabled_per_class <= 0:
        raise ValueError("minimum_enabled_per_class must be a positive integer")
    shortfalls = {
        station_class.value: len(catalog.eligible(station_class))
        for station_class in GroundStationClass
        if len(catalog.eligible(station_class)) < minimum_enabled_per_class
    }
    if shortfalls:
        raise ValueError(
            "Catalog is not production-ready; enabled station counts below "
            f"{minimum_enabled_per_class}: {shortfalls}"
        )


def catalog_from_stations(stations: Iterable[GroundStation]) -> GroundStationCatalog:
    return GroundStationCatalog(tuple(stations))
