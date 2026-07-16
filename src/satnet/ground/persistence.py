from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from satnet.ground.canonical import canonical_hash, canonical_json
from satnet.ground.catalog import (
    GroundStationCatalog,
    GroundStationClass,
    STATION_ID_PATTERN,
)
from satnet.ground.selection import (
    GROUND_STATION_SELECTION_VERSION,
    MAX_SELECTION_SEED,
    GroundSegmentEnabledConfig,
    GroundStationSelection,
    select_ground_stations,
)

GROUND_DESIGN_SCHEMA_VERSION = "1"
GROUND_DESIGN_IDENTITY_DOMAIN = "satnet_ground_design"
GROUND_DESIGN_IDENTITY_VERSION = "1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GROUND_RUN_DESIGN_FIELDS = frozenset(
    {
        "ground_design_schema_version",
        "run_id",
        "satellite_config_hash",
        "ground_segment_enabled",
        "catalog_hash",
        "selection_version",
        "station_selection_seed",
        "civilian_count",
        "government_count",
        "military_count",
        "selected_civilian_station_ids",
        "selected_government_station_ids",
        "selected_military_station_ids",
        "selected_station_ids",
        "selection_hash",
        "ground_design_hash",
    }
)


def _validate_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be exactly 64 lowercase hexadecimal characters")


def _ground_design_hash(*, enabled: bool, selection_hash: str | None) -> str:
    payload = {
        "enabled": enabled,
        "ground_design_schema_version": GROUND_DESIGN_SCHEMA_VERSION,
        "identity_domain": GROUND_DESIGN_IDENTITY_DOMAIN,
        "identity_version": GROUND_DESIGN_IDENTITY_VERSION,
        "selection_hash": selection_hash,
    }
    return canonical_hash(payload)


@dataclass(frozen=True)
class GroundRunDesignRecord:
    ground_design_schema_version: str
    run_id: int
    satellite_config_hash: str
    ground_segment_enabled: bool
    catalog_hash: str | None
    selection_version: str | None
    station_selection_seed: int | None
    civilian_count: int
    government_count: int
    military_count: int
    selected_civilian_station_ids: tuple[str, ...]
    selected_government_station_ids: tuple[str, ...]
    selected_military_station_ids: tuple[str, ...]
    selected_station_ids: tuple[str, ...]
    selection_hash: str | None
    ground_design_hash: str

    def __post_init__(self) -> None:
        if self.ground_design_schema_version != GROUND_DESIGN_SCHEMA_VERSION:
            raise ValueError(
                f"ground_design_schema_version must be '{GROUND_DESIGN_SCHEMA_VERSION}'"
            )
        if type(self.run_id) is not int or self.run_id < 0:
            raise TypeError("run_id must be a nonnegative integer")
        _validate_sha256(self.satellite_config_hash, "satellite_config_hash")
        if type(self.ground_segment_enabled) is not bool:
            raise TypeError("ground_segment_enabled must be a Boolean")
        for field_name in ("civilian_count", "government_count", "military_count"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{field_name} must be a nonnegative integer")
        id_fields = (
            "selected_civilian_station_ids",
            "selected_government_station_ids",
            "selected_military_station_ids",
            "selected_station_ids",
        )
        for field_name in id_fields:
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise TypeError(f"{field_name} must be a tuple")
            if any(
                not isinstance(identifier, str)
                or not STATION_ID_PATTERN.fullmatch(identifier)
                for identifier in values
            ):
                raise ValueError(f"{field_name} contains an invalid station ID")
        combined = (
            self.selected_civilian_station_ids
            + self.selected_government_station_ids
            + self.selected_military_station_ids
        )
        if self.selected_station_ids != combined:
            raise ValueError("selected_station_ids must match canonical class concatenation")
        if len(set(combined)) != len(combined):
            raise ValueError("Ground design contains duplicate selected station IDs")
        if len(self.selected_civilian_station_ids) != self.civilian_count:
            raise ValueError("civilian_count does not match selected civilian IDs")
        if len(self.selected_government_station_ids) != self.government_count:
            raise ValueError("government_count does not match selected government IDs")
        if len(self.selected_military_station_ids) != self.military_count:
            raise ValueError("military_count does not match selected military IDs")
        if self.ground_segment_enabled:
            self._validate_enabled()
        else:
            self._validate_disabled()
        _validate_sha256(self.ground_design_hash, "ground_design_hash")
        expected_hash = _ground_design_hash(
            enabled=self.ground_segment_enabled,
            selection_hash=self.selection_hash,
        )
        if self.ground_design_hash != expected_hash:
            raise ValueError("ground_design_hash does not match canonical ground design")

    def _validate_enabled(self) -> None:
        if self.catalog_hash is None:
            raise ValueError("Enabled ground design requires catalog_hash")
        _validate_sha256(self.catalog_hash, "catalog_hash")
        if self.selection_hash is None:
            raise ValueError("Enabled ground design requires selection_hash")
        _validate_sha256(self.selection_hash, "selection_hash")
        if self.selection_version != GROUND_STATION_SELECTION_VERSION:
            raise ValueError(
                f"selection_version must be '{GROUND_STATION_SELECTION_VERSION}'"
            )
        if type(self.station_selection_seed) is not int:
            raise TypeError("Enabled ground design requires an integer station_selection_seed")
        if not 0 <= self.station_selection_seed <= MAX_SELECTION_SEED:
            raise ValueError(
                f"station_selection_seed must be within [0, {MAX_SELECTION_SEED}]"
            )
        if self.civilian_count + self.government_count + self.military_count <= 0:
            raise ValueError("Enabled ground design must select at least one station")

    def _validate_disabled(self) -> None:
        if self.catalog_hash is not None:
            raise ValueError("Disabled ground design must not define catalog_hash")
        if self.selection_version is not None:
            raise ValueError("Disabled ground design must not define selection_version")
        if self.station_selection_seed is not None:
            raise ValueError("Disabled ground design must not define station_selection_seed")
        if self.selection_hash is not None:
            raise ValueError("Disabled ground design must not define selection_hash")
        if self.civilian_count or self.government_count or self.military_count:
            raise ValueError("Disabled ground design must use zero class counts")
        if self.selected_station_ids:
            raise ValueError("Disabled ground design must use empty station-ID lists")

    def to_manifest_object(self) -> dict[str, Any]:
        result = asdict(self)
        for field_name in (
            "selected_civilian_station_ids",
            "selected_government_station_ids",
            "selected_military_station_ids",
            "selected_station_ids",
        ):
            result[field_name] = list(result[field_name])
        return result


def make_enabled_ground_design_record(
    *,
    run_id: int,
    satellite_config_hash: str,
    selection: GroundStationSelection,
) -> GroundRunDesignRecord:
    if not isinstance(selection, GroundStationSelection):
        raise TypeError("selection must be a GroundStationSelection")
    ground_hash = _ground_design_hash(enabled=True, selection_hash=selection.selection_hash)
    return GroundRunDesignRecord(
        ground_design_schema_version=GROUND_DESIGN_SCHEMA_VERSION,
        run_id=run_id,
        satellite_config_hash=satellite_config_hash,
        ground_segment_enabled=True,
        catalog_hash=selection.catalog_hash,
        selection_version=selection.selection_version,
        station_selection_seed=selection.selection_seed,
        civilian_count=len(selection.civilian_station_ids),
        government_count=len(selection.government_station_ids),
        military_count=len(selection.military_station_ids),
        selected_civilian_station_ids=selection.civilian_station_ids,
        selected_government_station_ids=selection.government_station_ids,
        selected_military_station_ids=selection.military_station_ids,
        selected_station_ids=selection.selected_station_ids,
        selection_hash=selection.selection_hash,
        ground_design_hash=ground_hash,
    )


def make_disabled_ground_design_record(
    *,
    run_id: int,
    satellite_config_hash: str,
) -> GroundRunDesignRecord:
    return GroundRunDesignRecord(
        ground_design_schema_version=GROUND_DESIGN_SCHEMA_VERSION,
        run_id=run_id,
        satellite_config_hash=satellite_config_hash,
        ground_segment_enabled=False,
        catalog_hash=None,
        selection_version=None,
        station_selection_seed=None,
        civilian_count=0,
        government_count=0,
        military_count=0,
        selected_civilian_station_ids=(),
        selected_government_station_ids=(),
        selected_military_station_ids=(),
        selected_station_ids=(),
        selection_hash=None,
        ground_design_hash=_ground_design_hash(enabled=False, selection_hash=None),
    )


def _object_pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key '{key}'")
        result[key] = value
    return result


def _record_from_manifest_object(value: Any, line_number: int) -> GroundRunDesignRecord:
    if not isinstance(value, dict):
        raise ValueError(f"Line {line_number}: manifest record must be a JSON object")
    keys = set(value)
    missing = sorted(GROUND_RUN_DESIGN_FIELDS - keys)
    unknown = sorted(keys - GROUND_RUN_DESIGN_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"Line {line_number}: manifest fields are invalid; missing={missing}, unknown={unknown}"
        )
    if type(value["run_id"]) is not int:
        raise TypeError(f"Line {line_number}: run_id must be an integer")
    if type(value["ground_segment_enabled"]) is not bool:
        raise TypeError(f"Line {line_number}: ground_segment_enabled must be a Boolean")
    for field_name in ("civilian_count", "government_count", "military_count"):
        if type(value[field_name]) is not int:
            raise TypeError(f"Line {line_number}: {field_name} must be an integer")
    if value["station_selection_seed"] is not None and type(
        value["station_selection_seed"]
    ) is not int:
        raise TypeError(f"Line {line_number}: station_selection_seed must be an integer or null")
    id_fields = (
        "selected_civilian_station_ids",
        "selected_government_station_ids",
        "selected_military_station_ids",
        "selected_station_ids",
    )
    for field_name in id_fields:
        if not isinstance(value[field_name], list):
            raise TypeError(f"Line {line_number}: {field_name} must be a JSON array")
    converted = dict(value)
    for field_name in id_fields:
        converted[field_name] = tuple(converted[field_name])
    try:
        return GroundRunDesignRecord(**converted)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"Line {line_number}: {exc}") from exc


def read_ground_design_manifest(path: str | Path) -> tuple[GroundRunDesignRecord, ...]:
    manifest_path = Path(path)
    if manifest_path.suffix != ".jsonl":
        raise ValueError("Canonical ground-design manifest must use the .jsonl extension")
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Unable to read ground-design manifest '{manifest_path}': {exc}") from exc
    if not text:
        raise ValueError("Ground-design manifest is empty")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    if not lines or any(line == "" for line in lines):
        raise ValueError("Ground-design manifest contains an empty line")
    records: list[GroundRunDesignRecord] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line, object_pairs_hook=_object_pairs_without_duplicates)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Line {line_number}: malformed JSON: {exc}") from exc
        records.append(_record_from_manifest_object(value, line_number))
    _validate_unique_run_ids(records)
    return tuple(sorted(records, key=lambda record: record.run_id))


def _validate_unique_run_ids(records: Sequence[GroundRunDesignRecord]) -> None:
    run_ids = [record.run_id for record in records]
    duplicates = sorted(run_id for run_id in set(run_ids) if run_ids.count(run_id) > 1)
    if duplicates:
        raise ValueError(f"Duplicate ground-design manifest run IDs: {duplicates}")


def write_ground_design_manifest(
    records: Sequence[GroundRunDesignRecord],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    manifest_path = Path(path)
    if manifest_path.suffix != ".jsonl":
        raise ValueError("Canonical ground-design manifest must use the .jsonl extension")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a Boolean")
    if not records:
        raise ValueError("Ground-design manifest must contain at least one record")
    if any(not isinstance(record, GroundRunDesignRecord) for record in records):
        raise TypeError("Every manifest item must be a GroundRunDesignRecord")
    _validate_unique_run_ids(records)
    ordered = sorted(records, key=lambda record: record.run_id)
    serialized = [canonical_json(record.to_manifest_object()) for record in ordered]
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Ground-design manifest already exists: {manifest_path}")
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
            raise FileExistsError(f"Ground-design manifest already exists: {manifest_path}")
        os.replace(temporary_path, manifest_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def validate_manifest_against_satellite_runs(
    records: Sequence[GroundRunDesignRecord],
    satellite_run_hashes: Mapping[int, str],
) -> None:
    if not records:
        raise ValueError("Ground-design manifest must contain at least one record")
    _validate_unique_run_ids(records)
    normalized_satellite_runs: dict[int, str] = {}
    for run_id, config_hash in satellite_run_hashes.items():
        if type(run_id) is not int or run_id < 0:
            raise TypeError("Satellite run IDs must be nonnegative integers")
        _validate_sha256(config_hash, f"satellite config hash for run {run_id}")
        normalized_satellite_runs[run_id] = config_hash
    manifest_ids = {record.run_id for record in records}
    satellite_ids = set(normalized_satellite_runs)
    missing = sorted(satellite_ids - manifest_ids)
    orphan = sorted(manifest_ids - satellite_ids)
    if missing or orphan:
        raise ValueError(
            f"Ground-design manifest run IDs do not match satellite runs; "
            f"missing={missing}, orphan={orphan}"
        )
    for record in records:
        expected_hash = normalized_satellite_runs[record.run_id]
        if record.satellite_config_hash != expected_hash:
            raise ValueError(
                f"Run {record.run_id}: satellite_config_hash does not match satellite run"
            )


def reconstruct_ground_selection(
    record: GroundRunDesignRecord,
    catalog: GroundStationCatalog | None = None,
) -> GroundStationSelection | None:
    if not isinstance(record, GroundRunDesignRecord):
        raise TypeError("record must be a GroundRunDesignRecord")
    if not record.ground_segment_enabled:
        if catalog is not None and not isinstance(catalog, GroundStationCatalog):
            raise TypeError("catalog must be a GroundStationCatalog when provided")
        return None
    if not isinstance(catalog, GroundStationCatalog):
        raise ValueError("Enabled ground-design reconstruction requires a validated catalog")
    if catalog.catalog_hash != record.catalog_hash:
        raise ValueError("Supplied catalog hash does not match persisted catalog_hash")
    catalog_by_id = catalog.by_id()
    class_lists = (
        (record.selected_civilian_station_ids, GroundStationClass.CIVILIAN),
        (record.selected_government_station_ids, GroundStationClass.GOVERNMENT),
        (record.selected_military_station_ids, GroundStationClass.MILITARY),
    )
    for identifiers, expected_class in class_lists:
        for identifier in identifiers:
            if identifier not in catalog_by_id:
                raise ValueError(f"Persisted station ID does not exist in catalog: {identifier}")
            station = catalog_by_id[identifier]
            if not station.enabled:
                raise ValueError(f"Persisted station ID is disabled: {identifier}")
            if station.station_class is not expected_class:
                raise ValueError(
                    f"Persisted station ID has wrong class for {expected_class.value}: {identifier}"
                )
    config = GroundSegmentEnabledConfig(
        civilian_count=record.civilian_count,
        government_count=record.government_count,
        military_count=record.military_count,
        station_selection_seed=record.station_selection_seed,
    )
    expected = select_ground_stations(catalog=catalog, config=config)
    comparisons = {
        "selected_civilian_station_ids": (
            record.selected_civilian_station_ids,
            expected.civilian_station_ids,
        ),
        "selected_government_station_ids": (
            record.selected_government_station_ids,
            expected.government_station_ids,
        ),
        "selected_military_station_ids": (
            record.selected_military_station_ids,
            expected.military_station_ids,
        ),
        "selected_station_ids": (record.selected_station_ids, expected.selected_station_ids),
        "selection_hash": (record.selection_hash, expected.selection_hash),
    }
    mismatches = [name for name, (actual, wanted) in comparisons.items() if actual != wanted]
    if mismatches:
        raise ValueError(
            "Persisted ground selection does not match canonical deterministic prefixes: "
            f"{mismatches}"
        )
    return GroundStationSelection(
        catalog_hash=record.catalog_hash,
        selection_version=record.selection_version,
        selection_seed=record.station_selection_seed,
        civilian_station_ids=record.selected_civilian_station_ids,
        government_station_ids=record.selected_government_station_ids,
        military_station_ids=record.selected_military_station_ids,
        selected_station_ids=record.selected_station_ids,
        selection_hash=record.selection_hash,
    )
