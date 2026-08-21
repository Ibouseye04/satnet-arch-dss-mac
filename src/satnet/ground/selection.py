from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
from typing import Sequence

from satnet.ground.canonical import canonical_hash
from satnet.ground.catalog import (
    GroundStation,
    GroundStationCatalog,
    GroundStationClass,
)

GROUND_STATION_SELECTION_VERSION = "1"
SELECTION_IDENTITY_DOMAIN = "satnet_ground_selection"
SELECTION_IDENTITY_VERSION = "1"
MAX_SELECTION_SEED = 2**63 - 1


@dataclass(frozen=True)
class GroundSegmentDisabledConfig:
    enabled: bool = field(default=False, init=False)

    @property
    def total_ground_station_count(self) -> int:
        return 0


@dataclass(frozen=True)
class GroundSegmentEnabledConfig:
    civilian_count: int
    government_count: int
    military_count: int
    station_selection_seed: int
    enabled: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        for field_name in ("civilian_count", "government_count", "military_count"):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be nonnegative")
        if self.total_ground_station_count <= 0:
            raise ValueError("Enabled ground segment must request at least one station")
        if type(self.station_selection_seed) is not int:
            raise TypeError("station_selection_seed must be an integer")
        if not 0 <= self.station_selection_seed <= MAX_SELECTION_SEED:
            raise ValueError(
                f"station_selection_seed must be within [0, {MAX_SELECTION_SEED}]"
            )

    @property
    def total_ground_station_count(self) -> int:
        return self.civilian_count + self.government_count + self.military_count

    def count_for(self, station_class: GroundStationClass) -> int:
        if station_class is GroundStationClass.CIVILIAN:
            return self.civilian_count
        if station_class is GroundStationClass.GOVERNMENT:
            return self.government_count
        if station_class is GroundStationClass.MILITARY:
            return self.military_count
        raise TypeError("station_class must be a GroundStationClass")


@dataclass(frozen=True)
class GroundStationSelection:
    catalog_hash: str
    selection_version: str
    selection_seed: int
    civilian_station_ids: tuple[str, ...]
    government_station_ids: tuple[str, ...]
    military_station_ids: tuple[str, ...]
    selected_station_ids: tuple[str, ...]
    selection_hash: str

    def __post_init__(self) -> None:
        tuple_fields = (
            "civilian_station_ids",
            "government_station_ids",
            "military_station_ids",
            "selected_station_ids",
        )
        if any(not isinstance(getattr(self, name), tuple) for name in tuple_fields):
            raise TypeError("Ground-station selection ID collections must be tuples")
        combined = (
            self.civilian_station_ids
            + self.government_station_ids
            + self.military_station_ids
        )
        if self.selected_station_ids != combined:
            raise ValueError("selected_station_ids must match canonical class concatenation")
        if len(set(combined)) != len(combined):
            raise ValueError("Ground-station selection contains duplicate IDs")

    @property
    def total_ground_station_count(self) -> int:
        return len(self.selected_station_ids)


def _rank_digest(*parts: str) -> bytes:
    material = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(material).digest()


def build_region_balanced_order(
    stations: Sequence[GroundStation],
    *,
    station_class: GroundStationClass,
    selection_seed: int,
) -> tuple[GroundStation, ...]:
    if not isinstance(station_class, GroundStationClass):
        raise TypeError("station_class must be a GroundStationClass")
    if type(selection_seed) is not int:
        raise TypeError("selection_seed must be an integer")
    if not 0 <= selection_seed <= MAX_SELECTION_SEED:
        raise ValueError(f"selection_seed must be within [0, {MAX_SELECTION_SEED}]")
    eligible = [
        station
        for station in stations
        if station.enabled and station.station_class is station_class
    ]
    regions: dict[str, list[GroundStation]] = defaultdict(list)
    for station in eligible:
        regions[station.region].append(station)
    class_value = station_class.value
    seed_value = str(selection_seed)
    ordered_regions = sorted(
        regions,
        key=lambda region: (
            _rank_digest(
                GROUND_STATION_SELECTION_VERSION,
                class_value,
                seed_value,
                region,
            ),
            region,
        ),
    )
    for region in ordered_regions:
        regions[region].sort(
            key=lambda station: (
                _rank_digest(
                    GROUND_STATION_SELECTION_VERSION,
                    class_value,
                    seed_value,
                    region,
                    station.station_id,
                ),
                station.station_id,
            )
        )
    result: list[GroundStation] = []
    next_index = {region: 0 for region in ordered_regions}
    while len(result) < len(eligible):
        emitted = False
        for region in ordered_regions:
            index = next_index[region]
            if index < len(regions[region]):
                result.append(regions[region][index])
                next_index[region] = index + 1
                emitted = True
        if not emitted:
            raise RuntimeError("Region-balanced ordering made no progress")
    return tuple(result)


def _selection_hash(
    *,
    catalog_hash: str,
    config: GroundSegmentEnabledConfig,
    civilian_ids: tuple[str, ...],
    government_ids: tuple[str, ...],
    military_ids: tuple[str, ...],
) -> str:
    combined_ids = civilian_ids + government_ids + military_ids
    payload = {
        "catalog_hash": catalog_hash,
        "civilian_count": config.civilian_count,
        "civilian_station_ids": list(civilian_ids),
        "government_count": config.government_count,
        "government_station_ids": list(government_ids),
        "identity_domain": SELECTION_IDENTITY_DOMAIN,
        "identity_version": SELECTION_IDENTITY_VERSION,
        "military_count": config.military_count,
        "military_station_ids": list(military_ids),
        "selected_station_ids": list(combined_ids),
        "selection_seed": config.station_selection_seed,
        "selection_version": GROUND_STATION_SELECTION_VERSION,
    }
    return canonical_hash(payload)


def select_ground_stations(
    *,
    catalog: GroundStationCatalog,
    config: GroundSegmentEnabledConfig,
) -> GroundStationSelection:
    if not isinstance(catalog, GroundStationCatalog):
        raise TypeError("catalog must be a GroundStationCatalog")
    if not isinstance(config, GroundSegmentEnabledConfig):
        raise TypeError("config must be a GroundSegmentEnabledConfig")
    selected_by_class: dict[GroundStationClass, tuple[str, ...]] = {}
    for station_class in GroundStationClass:
        complete_order = build_region_balanced_order(
            catalog.stations,
            station_class=station_class,
            selection_seed=config.station_selection_seed,
        )
        requested_count = config.count_for(station_class)
        if requested_count > len(complete_order):
            raise ValueError(
                f"Requested {requested_count} {station_class.value} stations, "
                f"but only {len(complete_order)} enabled stations are eligible"
            )
        selected_by_class[station_class] = tuple(
            station.station_id for station in complete_order[:requested_count]
        )
    civilian_ids = selected_by_class[GroundStationClass.CIVILIAN]
    government_ids = selected_by_class[GroundStationClass.GOVERNMENT]
    military_ids = selected_by_class[GroundStationClass.MILITARY]
    combined_ids = civilian_ids + government_ids + military_ids
    selection = GroundStationSelection(
        catalog_hash=catalog.catalog_hash,
        selection_version=GROUND_STATION_SELECTION_VERSION,
        selection_seed=config.station_selection_seed,
        civilian_station_ids=civilian_ids,
        government_station_ids=government_ids,
        military_station_ids=military_ids,
        selected_station_ids=combined_ids,
        selection_hash=_selection_hash(
            catalog_hash=catalog.catalog_hash,
            config=config,
            civilian_ids=civilian_ids,
            government_ids=government_ids,
            military_ids=military_ids,
        ),
    )
    if selection.total_ground_station_count != config.total_ground_station_count:
        raise RuntimeError("Selected station count does not match requested count")
    return selection
