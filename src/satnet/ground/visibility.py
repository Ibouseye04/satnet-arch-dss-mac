from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
import re
from typing import Sequence

from satnet.ground.canonical import (
    canonical_float_string,
    canonical_hash,
    canonical_utc_timestamp,
)
from satnet.ground.catalog import GroundStation, GroundStationCatalog, STATION_ID_PATTERN
from satnet.ground.coordinates import (
    GROUND_VISIBILITY_FRAME_CONTRACT_VERSION,
    GROUND_VISIBILITY_MODEL_VERSION,
    GROUND_WGS84_MODEL_VERSION,
    CoordinateFrame,
    FramedSatellitePosition,
    OperationalSatellitePositionSnapshot,
    ground_station_to_ecef,
)
from satnet.ground.persistence import GroundRunDesignRecord, reconstruct_ground_selection

GROUND_VISIBILITY_POLICY_IDENTITY_DOMAIN = "satnet_ground_visibility_policy"
GROUND_VISIBILITY_POLICY_IDENTITY_VERSION = "1"
GROUND_VISIBILITY_SNAPSHOT_IDENTITY_DOMAIN = "satnet_ground_visibility_snapshot"
GROUND_VISIBILITY_SNAPSHOT_IDENTITY_VERSION = "1"
MIN_VALID_SLANT_RANGE_KM = 1.0e-9
GEOMETRY_TEST_ABS_TOL = 1.0e-9
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _validate_hash(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be exactly 64 lowercase hexadecimal characters")


def _finite_float(value: int | float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be an integer or float")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return 0.0 if normalized == 0.0 else normalized


@dataclass(frozen=True)
class GroundVisibilityPolicy:
    minimum_elevation_deg: float
    visibility_policy_hash: str = field(init=False)

    def __post_init__(self) -> None:
        minimum = _finite_float(self.minimum_elevation_deg, "minimum_elevation_deg")
        if not 0.0 <= minimum <= 90.0:
            raise ValueError("minimum_elevation_deg must be within [0.0, 90.0]")
        object.__setattr__(self, "minimum_elevation_deg", minimum)
        payload = {
            "frame_contract_version": GROUND_VISIBILITY_FRAME_CONTRACT_VERSION,
            "identity_domain": GROUND_VISIBILITY_POLICY_IDENTITY_DOMAIN,
            "identity_version": GROUND_VISIBILITY_POLICY_IDENTITY_VERSION,
            "minimum_elevation_deg": canonical_float_string(minimum),
            "visibility_model_version": GROUND_VISIBILITY_MODEL_VERSION,
            "wgs84_model_version": GROUND_WGS84_MODEL_VERSION,
        }
        object.__setattr__(self, "visibility_policy_hash", canonical_hash(payload))


@dataclass(frozen=True)
class SatelliteGroundLinkObservation:
    station_id: str
    satellite_id: int
    timestamp_utc: datetime
    elevation_deg: float
    slant_range_km: float
    is_visible: bool
    satellite_frame: CoordinateFrame
    visibility_model_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.station_id, str) or not STATION_ID_PATTERN.fullmatch(
            self.station_id
        ):
            raise ValueError("station_id is invalid")
        if type(self.satellite_id) is not int or self.satellite_id < 0:
            raise TypeError("satellite_id must be a nonnegative integer")
        canonical_utc_timestamp(self.timestamp_utc)
        elevation = _finite_float(self.elevation_deg, "elevation_deg")
        if not -90.0 <= elevation <= 90.0:
            raise ValueError("elevation_deg must be within [-90.0, 90.0]")
        slant_range = _finite_float(self.slant_range_km, "slant_range_km")
        if slant_range < MIN_VALID_SLANT_RANGE_KM:
            raise ValueError(
                f"slant_range_km must be at least {MIN_VALID_SLANT_RANGE_KM}"
            )
        if type(self.is_visible) is not bool:
            raise TypeError("is_visible must be a Boolean")
        if self.satellite_frame is not CoordinateFrame.ECEF:
            raise ValueError("Ground Visibility Model Version 1 accepts only ECEF observations")
        if self.visibility_model_version != GROUND_VISIBILITY_MODEL_VERSION:
            raise ValueError(
                f"visibility_model_version must be '{GROUND_VISIBILITY_MODEL_VERSION}'"
            )
        object.__setattr__(self, "elevation_deg", elevation)
        object.__setattr__(self, "slant_range_km", slant_range)

    def canonical_record(self) -> dict[str, str | int | bool]:
        return {
            "elevation_deg": canonical_float_string(self.elevation_deg),
            "is_visible": self.is_visible,
            "satellite_frame": self.satellite_frame.value,
            "satellite_id": self.satellite_id,
            "slant_range_km": canonical_float_string(self.slant_range_km),
            "station_id": self.station_id,
            "timestamp_utc": canonical_utc_timestamp(self.timestamp_utc),
            "visibility_model_version": self.visibility_model_version,
        }


@dataclass(frozen=True)
class GroundVisibilitySnapshot:
    timestep_index: int
    timestamp_utc: datetime
    satellite_config_hash: str
    ground_design_hash: str
    visibility_policy_hash: str
    visibility_model_version: str
    frame_contract_version: str
    wgs84_model_version: str
    link_observations: tuple[SatelliteGroundLinkObservation, ...]
    visible_links: tuple[SatelliteGroundLinkObservation, ...]
    visible_satellite_ids_by_station: tuple[tuple[str, tuple[int, ...]], ...]
    snapshot_hash: str

    def __post_init__(self) -> None:
        if type(self.timestep_index) is not int or self.timestep_index < 0:
            raise TypeError("timestep_index must be a nonnegative integer")
        canonical_utc_timestamp(self.timestamp_utc)
        _validate_hash(self.satellite_config_hash, "satellite_config_hash")
        _validate_hash(self.ground_design_hash, "ground_design_hash")
        _validate_hash(self.visibility_policy_hash, "visibility_policy_hash")
        if self.visibility_model_version != GROUND_VISIBILITY_MODEL_VERSION:
            raise ValueError("Unsupported visibility_model_version")
        if self.frame_contract_version != GROUND_VISIBILITY_FRAME_CONTRACT_VERSION:
            raise ValueError("Unsupported frame_contract_version")
        if self.wgs84_model_version != GROUND_WGS84_MODEL_VERSION:
            raise ValueError("Unsupported wgs84_model_version")
        if not isinstance(self.link_observations, tuple) or not isinstance(
            self.visible_links, tuple
        ):
            raise TypeError("Visibility link collections must be tuples")
        if any(
            not isinstance(observation, SatelliteGroundLinkObservation)
            for observation in self.link_observations
        ):
            raise TypeError("Every link observation must be a SatelliteGroundLinkObservation")
        keys = [
            (observation.station_id, observation.satellite_id)
            for observation in self.link_observations
        ]
        if keys != sorted(keys):
            raise ValueError("link_observations must use canonical station and satellite order")
        if len(keys) != len(set(keys)):
            raise ValueError("link_observations contains duplicate station-satellite pairs")
        if any(
            observation.timestamp_utc != self.timestamp_utc
            for observation in self.link_observations
        ):
            raise ValueError("Observation timestamps must match snapshot timestamp")
        expected_visible = tuple(
            observation for observation in self.link_observations if observation.is_visible
        )
        if self.visible_links != expected_visible:
            raise ValueError("visible_links must equal the exact visible observation subset")
        if not isinstance(self.visible_satellite_ids_by_station, tuple):
            raise TypeError("visible_satellite_ids_by_station must be a tuple")
        if any(
            not isinstance(entry, tuple) or len(entry) != 2
            for entry in self.visible_satellite_ids_by_station
        ):
            raise ValueError("Every station visibility mapping must be a two-item tuple")
        station_ids = [entry[0] for entry in self.visible_satellite_ids_by_station]
        if station_ids != sorted(station_ids) or len(station_ids) != len(set(station_ids)):
            raise ValueError("Station visibility mappings must use unique canonical station order")
        observation_station_ids = {observation.station_id for observation in self.link_observations}
        if not observation_station_ids.issubset(set(station_ids)):
            raise ValueError("Station visibility mappings omit an observed station")
        visible_by_station = {
            station_id: tuple(
                observation.satellite_id
                for observation in expected_visible
                if observation.station_id == station_id
            )
            for station_id in station_ids
        }
        for entry in self.visible_satellite_ids_by_station:
            station_id, satellite_ids = entry
            if not isinstance(station_id, str) or not STATION_ID_PATTERN.fullmatch(station_id):
                raise ValueError("Station visibility mapping contains an invalid station ID")
            if not isinstance(satellite_ids, tuple):
                raise TypeError("Mapped satellite IDs must be a tuple")
            if list(satellite_ids) != sorted(satellite_ids) or len(satellite_ids) != len(
                set(satellite_ids)
            ):
                raise ValueError("Mapped satellite IDs must be unique and numerically sorted")
            if satellite_ids != visible_by_station[station_id]:
                raise ValueError("Station visibility mapping does not match visible links")
        _validate_hash(self.snapshot_hash, "snapshot_hash")
        if self.snapshot_hash != _compute_snapshot_hash(
            timestep_index=self.timestep_index,
            timestamp_utc=self.timestamp_utc,
            satellite_config_hash=self.satellite_config_hash,
            ground_design_hash=self.ground_design_hash,
            visibility_policy_hash=self.visibility_policy_hash,
            link_observations=self.link_observations,
            visible_satellite_ids_by_station=self.visible_satellite_ids_by_station,
        ):
            raise ValueError("snapshot_hash does not match canonical visibility snapshot")


def calculate_topocentric_geometry(
    station: GroundStation,
    satellite_position: FramedSatellitePosition,
) -> tuple[float, float]:
    if not isinstance(station, GroundStation):
        raise TypeError("station must be a GroundStation")
    if not isinstance(satellite_position, FramedSatellitePosition):
        raise TypeError("satellite_position must be a FramedSatellitePosition")
    if satellite_position.frame is not CoordinateFrame.ECEF:
        raise ValueError("Ground Visibility Model Version 1 accepts only ECEF positions")
    station_ecef = ground_station_to_ecef(station)
    delta_x = satellite_position.x_km - station_ecef.x_km
    delta_y = satellite_position.y_km - station_ecef.y_km
    delta_z = satellite_position.z_km - station_ecef.z_km
    latitude = math.radians(station.latitude_deg)
    longitude = math.radians(station.longitude_deg)
    sin_latitude = math.sin(latitude)
    cos_latitude = math.cos(latitude)
    sin_longitude = math.sin(longitude)
    cos_longitude = math.cos(longitude)
    east = -sin_longitude * delta_x + cos_longitude * delta_y
    north = (
        -sin_latitude * cos_longitude * delta_x
        - sin_latitude * sin_longitude * delta_y
        + cos_latitude * delta_z
    )
    up = (
        cos_latitude * cos_longitude * delta_x
        + cos_latitude * sin_longitude * delta_y
        + sin_latitude * delta_z
    )
    horizontal_range = math.hypot(east, north)
    slant_range = math.sqrt(east**2 + north**2 + up**2)
    if not math.isfinite(slant_range) or slant_range < MIN_VALID_SLANT_RANGE_KM:
        raise ValueError(
            f"Degenerate geometry: slant range is below {MIN_VALID_SLANT_RANGE_KM} km"
        )
    elevation = math.degrees(math.atan2(up, horizontal_range))
    if not math.isfinite(elevation):
        raise ValueError("Calculated elevation is nonfinite")
    return elevation, slant_range


def _compute_snapshot_hash(
    *,
    timestep_index: int,
    timestamp_utc: datetime,
    satellite_config_hash: str,
    ground_design_hash: str,
    visibility_policy_hash: str,
    link_observations: tuple[SatelliteGroundLinkObservation, ...],
    visible_satellite_ids_by_station: tuple[tuple[str, tuple[int, ...]], ...],
) -> str:
    payload = {
        "frame_contract_version": GROUND_VISIBILITY_FRAME_CONTRACT_VERSION,
        "ground_design_hash": ground_design_hash,
        "identity_domain": GROUND_VISIBILITY_SNAPSHOT_IDENTITY_DOMAIN,
        "identity_version": GROUND_VISIBILITY_SNAPSHOT_IDENTITY_VERSION,
        "link_observations": [
            observation.canonical_record() for observation in link_observations
        ],
        "satellite_config_hash": satellite_config_hash,
        "timestep_index": timestep_index,
        "timestamp_utc": canonical_utc_timestamp(timestamp_utc),
        "visibility_model_version": GROUND_VISIBILITY_MODEL_VERSION,
        "visibility_policy_hash": visibility_policy_hash,
        "visible_satellite_ids_by_station": [
            [station_id, list(satellite_ids)]
            for station_id, satellite_ids in visible_satellite_ids_by_station
        ],
        "wgs84_model_version": GROUND_WGS84_MODEL_VERSION,
    }
    return canonical_hash(payload)


def evaluate_ground_visibility(
    *,
    selected_stations: Sequence[GroundStation],
    satellite_snapshot: OperationalSatellitePositionSnapshot,
    policy: GroundVisibilityPolicy,
    ground_design_hash: str,
) -> GroundVisibilitySnapshot:
    if not isinstance(satellite_snapshot, OperationalSatellitePositionSnapshot):
        raise TypeError("satellite_snapshot must be an OperationalSatellitePositionSnapshot")
    if not isinstance(policy, GroundVisibilityPolicy):
        raise TypeError("policy must be a GroundVisibilityPolicy")
    _validate_hash(ground_design_hash, "ground_design_hash")
    stations = tuple(selected_stations)
    if not stations:
        raise ValueError("Ground visibility requires at least one selected station")
    if any(not isinstance(station, GroundStation) for station in stations):
        raise TypeError("Every selected station must be a GroundStation")
    if any(not station.enabled for station in stations):
        raise ValueError("Disabled stations cannot be evaluated")
    station_ids = [station.station_id for station in stations]
    if len(station_ids) != len(set(station_ids)):
        raise ValueError("Selected stations contain duplicate station IDs")
    stations = tuple(sorted(stations, key=lambda station: station.station_id))
    observations: list[SatelliteGroundLinkObservation] = []
    for station in stations:
        for satellite in satellite_snapshot.positions:
            elevation, slant_range = calculate_topocentric_geometry(station, satellite)
            observations.append(
                SatelliteGroundLinkObservation(
                    station_id=station.station_id,
                    satellite_id=satellite.satellite_id,
                    timestamp_utc=satellite_snapshot.timestamp_utc,
                    elevation_deg=elevation,
                    slant_range_km=slant_range,
                    is_visible=elevation >= policy.minimum_elevation_deg,
                    satellite_frame=satellite.frame,
                    visibility_model_version=GROUND_VISIBILITY_MODEL_VERSION,
                )
            )
    canonical_observations = tuple(observations)
    visible_links = tuple(
        observation for observation in canonical_observations if observation.is_visible
    )
    station_mapping = tuple(
        (
            station.station_id,
            tuple(
                observation.satellite_id
                for observation in visible_links
                if observation.station_id == station.station_id
            ),
        )
        for station in stations
    )
    snapshot_hash = _compute_snapshot_hash(
        timestep_index=satellite_snapshot.timestep_index,
        timestamp_utc=satellite_snapshot.timestamp_utc,
        satellite_config_hash=satellite_snapshot.satellite_config_hash,
        ground_design_hash=ground_design_hash,
        visibility_policy_hash=policy.visibility_policy_hash,
        link_observations=canonical_observations,
        visible_satellite_ids_by_station=station_mapping,
    )
    return GroundVisibilitySnapshot(
        timestep_index=satellite_snapshot.timestep_index,
        timestamp_utc=satellite_snapshot.timestamp_utc,
        satellite_config_hash=satellite_snapshot.satellite_config_hash,
        ground_design_hash=ground_design_hash,
        visibility_policy_hash=policy.visibility_policy_hash,
        visibility_model_version=GROUND_VISIBILITY_MODEL_VERSION,
        frame_contract_version=GROUND_VISIBILITY_FRAME_CONTRACT_VERSION,
        wgs84_model_version=GROUND_WGS84_MODEL_VERSION,
        link_observations=canonical_observations,
        visible_links=visible_links,
        visible_satellite_ids_by_station=station_mapping,
        snapshot_hash=snapshot_hash,
    )


def evaluate_ground_visibility_sequence(
    *,
    selected_stations: Sequence[GroundStation],
    satellite_sequence: Sequence[OperationalSatellitePositionSnapshot],
    policy: GroundVisibilityPolicy,
    ground_design_hash: str,
) -> tuple[GroundVisibilitySnapshot, ...]:
    sources = tuple(satellite_sequence)
    if not sources:
        raise ValueError("Visibility sequence requires at least one satellite snapshot")
    if tuple(source.timestep_index for source in sources) != tuple(range(len(sources))):
        raise ValueError("Satellite sequence timestep indices must be contiguous from zero")
    if any(
        current.timestamp_utc >= following.timestamp_utc
        for current, following in zip(sources, sources[1:])
    ):
        raise ValueError("Satellite sequence timestamps must be strictly increasing")
    config_hashes = {source.satellite_config_hash for source in sources}
    if len(config_hashes) != 1:
        raise ValueError("Satellite sequence must use one satellite configuration hash")
    return tuple(
        evaluate_ground_visibility(
            selected_stations=selected_stations,
            satellite_snapshot=source,
            policy=policy,
            ground_design_hash=ground_design_hash,
        )
        for source in sources
    )


def evaluate_ground_design_visibility(
    *,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    satellite_snapshot: OperationalSatellitePositionSnapshot,
    policy: GroundVisibilityPolicy,
) -> GroundVisibilitySnapshot:
    if not isinstance(ground_design, GroundRunDesignRecord):
        raise TypeError("ground_design must be a GroundRunDesignRecord")
    if ground_design.satellite_config_hash != satellite_snapshot.satellite_config_hash:
        raise ValueError("Ground design satellite hash does not match satellite snapshot")
    selection = reconstruct_ground_selection(ground_design, catalog)
    if selection is None:
        raise ValueError("Ground visibility requires an enabled ground design")
    stations_by_id = catalog.by_id()
    selected_stations = tuple(
        stations_by_id[station_id] for station_id in selection.selected_station_ids
    )
    return evaluate_ground_visibility(
        selected_stations=selected_stations,
        satellite_snapshot=satellite_snapshot,
        policy=policy,
        ground_design_hash=ground_design.ground_design_hash,
    )


def evaluate_ground_design_visibility_sequence(
    *,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    satellite_sequence: Sequence[OperationalSatellitePositionSnapshot],
    policy: GroundVisibilityPolicy,
) -> tuple[GroundVisibilitySnapshot, ...]:
    sources = tuple(satellite_sequence)
    if not sources:
        raise ValueError("Visibility sequence requires at least one satellite snapshot")
    snapshots = tuple(
        evaluate_ground_design_visibility(
            ground_design=ground_design,
            catalog=catalog,
            satellite_snapshot=source,
            policy=policy,
        )
        for source in sources
    )
    if tuple(snapshot.timestep_index for snapshot in snapshots) != tuple(range(len(snapshots))):
        raise ValueError("Visibility sequence timestep indices must be contiguous from zero")
    if any(
        current.timestamp_utc >= following.timestamp_utc
        for current, following in zip(snapshots, snapshots[1:])
    ):
        raise ValueError("Visibility sequence timestamps must be strictly increasing")
    return snapshots
