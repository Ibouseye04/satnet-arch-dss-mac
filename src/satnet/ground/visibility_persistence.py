from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from satnet.ground.canonical import (
    canonical_float_string,
    canonical_hash,
    canonical_json,
    canonical_utc_timestamp,
)
from satnet.ground.catalog import GroundStationCatalog
from satnet.ground.coordinates import (
    GROUND_VISIBILITY_FRAME_CONTRACT_VERSION,
    GROUND_VISIBILITY_MODEL_VERSION,
    GROUND_WGS84_MODEL_VERSION,
    CoordinateFrame,
    OperationalSatellitePositionSnapshot,
)
from satnet.ground.persistence import GroundRunDesignRecord
from satnet.ground.visibility import (
    GroundVisibilityPolicy,
    GroundVisibilitySnapshot,
    SatelliteGroundLinkObservation,
    evaluate_ground_design_visibility,
)

GROUND_VISIBILITY_SCHEMA_VERSION = "1"
GROUND_VISIBILITY_RECORD_IDENTITY_DOMAIN = "satnet_ground_visibility_record"
GROUND_VISIBILITY_RECORD_IDENTITY_VERSION = "1"
GROUND_VISIBILITY_RECORD_FIELDS = frozenset(
    {
        "visibility_schema_version",
        "run_id",
        "timestep_index",
        "timestamp_utc",
        "satellite_config_hash",
        "ground_design_hash",
        "visibility_policy_hash",
        "visibility_model_version",
        "frame_contract_version",
        "wgs84_model_version",
        "snapshot_hash",
        "record_hash",
        "link_observations",
        "visible_satellite_ids_by_station",
    }
)
OBSERVATION_FIELDS = frozenset(
    {
        "station_id",
        "satellite_id",
        "timestamp_utc",
        "elevation_deg",
        "slant_range_km",
        "is_visible",
        "satellite_frame",
        "visibility_model_version",
    }
)


def _parse_canonical_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a canonical UTC string")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must use YYYY-MM-DDTHH:MM:SS.ffffffZ"
        ) from exc
    if canonical_utc_timestamp(parsed) != value:
        raise ValueError(f"{field_name} is not canonical UTC")
    return parsed


def _parse_canonical_float(value: object, field_name: str) -> float:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a canonical float string")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid float string") from exc
    if canonical_float_string(parsed) != value:
        raise ValueError(f"{field_name} is not a canonical float string")
    return parsed


def _compute_record_hash(
    *,
    run_id: int,
    timestep_index: int,
    timestamp_utc: datetime,
    snapshot_hash: str,
) -> str:
    return canonical_hash(
        {
            "frame_contract_version": GROUND_VISIBILITY_FRAME_CONTRACT_VERSION,
            "identity_domain": GROUND_VISIBILITY_RECORD_IDENTITY_DOMAIN,
            "identity_version": GROUND_VISIBILITY_RECORD_IDENTITY_VERSION,
            "run_id": run_id,
            "snapshot_hash": snapshot_hash,
            "timestep_index": timestep_index,
            "timestamp_utc": canonical_utc_timestamp(timestamp_utc),
            "visibility_model_version": GROUND_VISIBILITY_MODEL_VERSION,
            "visibility_schema_version": GROUND_VISIBILITY_SCHEMA_VERSION,
            "wgs84_model_version": GROUND_WGS84_MODEL_VERSION,
        }
    )


@dataclass(frozen=True)
class GroundVisibilityRecord:
    visibility_schema_version: str
    run_id: int
    timestep_index: int
    timestamp_utc: datetime
    satellite_config_hash: str
    ground_design_hash: str
    visibility_policy_hash: str
    visibility_model_version: str
    frame_contract_version: str
    wgs84_model_version: str
    snapshot_hash: str
    record_hash: str
    link_observations: tuple[SatelliteGroundLinkObservation, ...]
    visible_satellite_ids_by_station: tuple[tuple[str, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        if self.visibility_schema_version != GROUND_VISIBILITY_SCHEMA_VERSION:
            raise ValueError("Unsupported visibility_schema_version")
        if type(self.run_id) is not int or self.run_id < 0:
            raise TypeError("run_id must be a nonnegative integer")
        if type(self.timestep_index) is not int or self.timestep_index < 0:
            raise TypeError("timestep_index must be a nonnegative integer")
        canonical_utc_timestamp(self.timestamp_utc)
        if self.visibility_model_version != GROUND_VISIBILITY_MODEL_VERSION:
            raise ValueError("Unsupported visibility_model_version")
        if self.frame_contract_version != GROUND_VISIBILITY_FRAME_CONTRACT_VERSION:
            raise ValueError("Unsupported frame_contract_version")
        if self.wgs84_model_version != GROUND_WGS84_MODEL_VERSION:
            raise ValueError("Unsupported wgs84_model_version")
        if not isinstance(self.link_observations, tuple):
            raise TypeError("link_observations must be a tuple")
        if not isinstance(self.visible_satellite_ids_by_station, tuple):
            raise TypeError("visible_satellite_ids_by_station must be a tuple")
        visible_links = tuple(
            observation for observation in self.link_observations if observation.is_visible
        )
        GroundVisibilitySnapshot(
            timestep_index=self.timestep_index,
            timestamp_utc=self.timestamp_utc,
            satellite_config_hash=self.satellite_config_hash,
            ground_design_hash=self.ground_design_hash,
            visibility_policy_hash=self.visibility_policy_hash,
            visibility_model_version=self.visibility_model_version,
            frame_contract_version=self.frame_contract_version,
            wgs84_model_version=self.wgs84_model_version,
            link_observations=self.link_observations,
            visible_links=visible_links,
            visible_satellite_ids_by_station=self.visible_satellite_ids_by_station,
            snapshot_hash=self.snapshot_hash,
        )
        expected_record_hash = _compute_record_hash(
            run_id=self.run_id,
            timestep_index=self.timestep_index,
            timestamp_utc=self.timestamp_utc,
            snapshot_hash=self.snapshot_hash,
        )
        if self.record_hash != expected_record_hash:
            raise ValueError("record_hash does not match canonical visibility record")

    def to_manifest_object(self) -> dict[str, Any]:
        return {
            "frame_contract_version": self.frame_contract_version,
            "ground_design_hash": self.ground_design_hash,
            "link_observations": [
                observation.canonical_record() for observation in self.link_observations
            ],
            "record_hash": self.record_hash,
            "run_id": self.run_id,
            "satellite_config_hash": self.satellite_config_hash,
            "snapshot_hash": self.snapshot_hash,
            "timestep_index": self.timestep_index,
            "timestamp_utc": canonical_utc_timestamp(self.timestamp_utc),
            "visibility_model_version": self.visibility_model_version,
            "visibility_policy_hash": self.visibility_policy_hash,
            "visibility_schema_version": self.visibility_schema_version,
            "visible_satellite_ids_by_station": [
                [station_id, list(satellite_ids)]
                for station_id, satellite_ids in self.visible_satellite_ids_by_station
            ],
            "wgs84_model_version": self.wgs84_model_version,
        }


def make_ground_visibility_record(
    *,
    run_id: int,
    snapshot: GroundVisibilitySnapshot,
) -> GroundVisibilityRecord:
    if not isinstance(snapshot, GroundVisibilitySnapshot):
        raise TypeError("snapshot must be a GroundVisibilitySnapshot")
    record_hash = _compute_record_hash(
        run_id=run_id,
        timestep_index=snapshot.timestep_index,
        timestamp_utc=snapshot.timestamp_utc,
        snapshot_hash=snapshot.snapshot_hash,
    )
    return GroundVisibilityRecord(
        visibility_schema_version=GROUND_VISIBILITY_SCHEMA_VERSION,
        run_id=run_id,
        timestep_index=snapshot.timestep_index,
        timestamp_utc=snapshot.timestamp_utc,
        satellite_config_hash=snapshot.satellite_config_hash,
        ground_design_hash=snapshot.ground_design_hash,
        visibility_policy_hash=snapshot.visibility_policy_hash,
        visibility_model_version=snapshot.visibility_model_version,
        frame_contract_version=snapshot.frame_contract_version,
        wgs84_model_version=snapshot.wgs84_model_version,
        snapshot_hash=snapshot.snapshot_hash,
        record_hash=record_hash,
        link_observations=snapshot.link_observations,
        visible_satellite_ids_by_station=snapshot.visible_satellite_ids_by_station,
    )


def _object_pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key '{key}'")
        result[key] = value
    return result


def _observation_from_object(value: object, line_number: int) -> SatelliteGroundLinkObservation:
    if not isinstance(value, dict):
        raise ValueError(f"Line {line_number}: link observation must be an object")
    missing = sorted(OBSERVATION_FIELDS - set(value))
    unknown = sorted(set(value) - OBSERVATION_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"Line {line_number}: observation fields invalid; missing={missing}, unknown={unknown}"
        )
    if type(value["satellite_id"]) is not int:
        raise TypeError(f"Line {line_number}: satellite_id must be an integer")
    if type(value["is_visible"]) is not bool:
        raise TypeError(f"Line {line_number}: is_visible must be a Boolean")
    try:
        frame = CoordinateFrame(value["satellite_frame"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Line {line_number}: invalid satellite_frame") from exc
    return SatelliteGroundLinkObservation(
        station_id=value["station_id"],
        satellite_id=value["satellite_id"],
        timestamp_utc=_parse_canonical_timestamp(
            value["timestamp_utc"], f"Line {line_number} observation timestamp_utc"
        ),
        elevation_deg=_parse_canonical_float(
            value["elevation_deg"], f"Line {line_number} elevation_deg"
        ),
        slant_range_km=_parse_canonical_float(
            value["slant_range_km"], f"Line {line_number} slant_range_km"
        ),
        is_visible=value["is_visible"],
        satellite_frame=frame,
        visibility_model_version=value["visibility_model_version"],
    )


def _record_from_object(value: object, line_number: int) -> GroundVisibilityRecord:
    if not isinstance(value, dict):
        raise ValueError(f"Line {line_number}: visibility record must be an object")
    missing = sorted(GROUND_VISIBILITY_RECORD_FIELDS - set(value))
    unknown = sorted(set(value) - GROUND_VISIBILITY_RECORD_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"Line {line_number}: visibility record fields invalid; "
            f"missing={missing}, unknown={unknown}"
        )
    for field_name in ("run_id", "timestep_index"):
        if type(value[field_name]) is not int:
            raise TypeError(f"Line {line_number}: {field_name} must be an integer")
    if not isinstance(value["link_observations"], list):
        raise TypeError(f"Line {line_number}: link_observations must be an array")
    observations = tuple(
        _observation_from_object(observation, line_number)
        for observation in value["link_observations"]
    )
    raw_mapping = value["visible_satellite_ids_by_station"]
    if not isinstance(raw_mapping, list):
        raise TypeError(
            f"Line {line_number}: visible_satellite_ids_by_station must be an array"
        )
    mapping: list[tuple[str, tuple[int, ...]]] = []
    for entry in raw_mapping:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(f"Line {line_number}: invalid station visibility mapping")
        station_id, satellite_ids = entry
        if not isinstance(station_id, str) or not isinstance(satellite_ids, list):
            raise TypeError(f"Line {line_number}: invalid station visibility mapping types")
        if any(type(satellite_id) is not int for satellite_id in satellite_ids):
            raise TypeError(f"Line {line_number}: mapped satellite IDs must be integers")
        mapping.append((station_id, tuple(satellite_ids)))
    try:
        return GroundVisibilityRecord(
            visibility_schema_version=value["visibility_schema_version"],
            run_id=value["run_id"],
            timestep_index=value["timestep_index"],
            timestamp_utc=_parse_canonical_timestamp(
                value["timestamp_utc"], f"Line {line_number} timestamp_utc"
            ),
            satellite_config_hash=value["satellite_config_hash"],
            ground_design_hash=value["ground_design_hash"],
            visibility_policy_hash=value["visibility_policy_hash"],
            visibility_model_version=value["visibility_model_version"],
            frame_contract_version=value["frame_contract_version"],
            wgs84_model_version=value["wgs84_model_version"],
            snapshot_hash=value["snapshot_hash"],
            record_hash=value["record_hash"],
            link_observations=observations,
            visible_satellite_ids_by_station=tuple(mapping),
        )
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"Line {line_number}: {exc}") from exc


def _validate_unique_keys(records: Sequence[GroundVisibilityRecord]) -> None:
    keys = [(record.run_id, record.timestep_index) for record in records]
    duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
    if duplicates:
        raise ValueError(f"Duplicate visibility run/timestep keys: {duplicates}")


def read_ground_visibility_manifest(
    path: str | Path,
) -> tuple[GroundVisibilityRecord, ...]:
    manifest_path = Path(path)
    if manifest_path.suffix != ".jsonl":
        raise ValueError("Canonical visibility manifest must use the .jsonl extension")
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Unable to read visibility manifest '{manifest_path}': {exc}") from exc
    if not text:
        raise ValueError("Visibility manifest is empty")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    if not lines or any(line == "" for line in lines):
        raise ValueError("Visibility manifest contains an empty line")
    records: list[GroundVisibilityRecord] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line, object_pairs_hook=_object_pairs_without_duplicates)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Line {line_number}: malformed JSON: {exc}") from exc
        records.append(_record_from_object(value, line_number))
    _validate_unique_keys(records)
    return tuple(sorted(records, key=lambda record: (record.run_id, record.timestep_index)))


def write_ground_visibility_manifest(
    records: Sequence[GroundVisibilityRecord],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    manifest_path = Path(path)
    if manifest_path.suffix != ".jsonl":
        raise ValueError("Canonical visibility manifest must use the .jsonl extension")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a Boolean")
    if not records:
        raise ValueError("Visibility manifest must contain at least one record")
    if any(not isinstance(record, GroundVisibilityRecord) for record in records):
        raise TypeError("Every item must be a GroundVisibilityRecord")
    _validate_unique_keys(records)
    ordered = sorted(records, key=lambda record: (record.run_id, record.timestep_index))
    serialized = [canonical_json(record.to_manifest_object()) for record in ordered]
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Visibility manifest already exists: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(serialized) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"Visibility manifest already exists: {manifest_path}")
        os.replace(temporary_path, manifest_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def validate_visibility_records_against_sources(
    *,
    records: Sequence[GroundVisibilityRecord],
    run_id: int,
    operational_satellite_snapshots: Sequence[OperationalSatellitePositionSnapshot],
    ground_design: GroundRunDesignRecord,
    policy: GroundVisibilityPolicy,
) -> None:
    if type(run_id) is not int or run_id < 0:
        raise TypeError("run_id must be a nonnegative integer")
    if not records:
        raise ValueError("Visibility records must not be empty")
    if not operational_satellite_snapshots:
        raise ValueError("Operational satellite snapshots must not be empty")
    if not isinstance(ground_design, GroundRunDesignRecord):
        raise TypeError("ground_design must be a GroundRunDesignRecord")
    if not isinstance(policy, GroundVisibilityPolicy):
        raise TypeError("policy must be a GroundVisibilityPolicy")
    if any(
        not isinstance(source, OperationalSatellitePositionSnapshot)
        for source in operational_satellite_snapshots
    ):
        raise TypeError("Every source must be an OperationalSatellitePositionSnapshot")
    source_timesteps = [source.timestep_index for source in operational_satellite_snapshots]
    if len(source_timesteps) != len(set(source_timesteps)):
        raise ValueError("Operational satellite snapshots contain duplicate timesteps")
    source_hashes = {source.satellite_config_hash for source in operational_satellite_snapshots}
    if source_hashes != {ground_design.satellite_config_hash}:
        raise ValueError("Operational satellite snapshots do not match ground-design satellite hash")
    _validate_unique_keys(records)
    expected_keys = {
        (run_id, source.timestep_index) for source in operational_satellite_snapshots
    }
    actual_keys = {(record.run_id, record.timestep_index) for record in records}
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise ValueError(
            f"Visibility records do not match source timesteps; missing={missing}, extra={extra}"
        )
    source_by_timestep = {
        source.timestep_index: source for source in operational_satellite_snapshots
    }
    for record in records:
        source = source_by_timestep[record.timestep_index]
        if record.timestamp_utc != source.timestamp_utc:
            raise ValueError(f"Run {run_id} timestep {record.timestep_index}: timestamp mismatch")
        if record.satellite_config_hash != source.satellite_config_hash:
            raise ValueError(
                f"Run {run_id} timestep {record.timestep_index}: satellite hash mismatch"
            )
        if record.ground_design_hash != ground_design.ground_design_hash:
            raise ValueError(
                f"Run {run_id} timestep {record.timestep_index}: ground-design hash mismatch"
            )
        if record.visibility_policy_hash != policy.visibility_policy_hash:
            raise ValueError(
                f"Run {run_id} timestep {record.timestep_index}: policy hash mismatch"
            )


def replay_ground_visibility_records(
    *,
    records: Sequence[GroundVisibilityRecord],
    run_id: int,
    operational_satellite_snapshots: Sequence[OperationalSatellitePositionSnapshot],
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    policy: GroundVisibilityPolicy,
) -> tuple[GroundVisibilitySnapshot, ...]:
    validate_visibility_records_against_sources(
        records=records,
        run_id=run_id,
        operational_satellite_snapshots=operational_satellite_snapshots,
        ground_design=ground_design,
        policy=policy,
    )
    records_by_timestep = {record.timestep_index: record for record in records}
    replayed: list[GroundVisibilitySnapshot] = []
    for source in sorted(
        operational_satellite_snapshots, key=lambda snapshot: snapshot.timestep_index
    ):
        expected = evaluate_ground_design_visibility(
            ground_design=ground_design,
            catalog=catalog,
            satellite_snapshot=source,
            policy=policy,
        )
        record = records_by_timestep[source.timestep_index]
        comparisons = {
            "snapshot_hash": (record.snapshot_hash, expected.snapshot_hash),
            "link_observations": (record.link_observations, expected.link_observations),
            "visible_station_mapping": (
                record.visible_satellite_ids_by_station,
                expected.visible_satellite_ids_by_station,
            ),
        }
        mismatches = [name for name, values in comparisons.items() if values[0] != values[1]]
        if mismatches:
            raise ValueError(
                f"Run {run_id} timestep {source.timestep_index}: visibility replay mismatch "
                f"for {mismatches}"
            )
        replayed.append(expected)
    return tuple(replayed)
