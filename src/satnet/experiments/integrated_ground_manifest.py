from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from satnet.ground.canonical import canonical_float_string, canonical_hash, canonical_json
from satnet.ground.catalog import (
    CATALOG_COLUMNS,
    GroundStation,
    GroundStationCatalog,
    GroundStationClass,
    load_ground_station_catalog,
    validate_production_catalog_readiness,
)
from satnet.ground.failure_policy import GroundFailurePolicy
from satnet.ground.service_policy import GroundServicePolicy
from satnet.ground.visibility import GroundVisibilityPolicy
from satnet.simulation.tier1_rollout import (
    DEFAULT_EPOCH_ISO,
    DEFAULT_FAILURE_MODEL,
    Tier1RolloutConfig,
)

PILOT_SCHEMA_VERSION = "1"
PILOT_MASTER_SEED = 20260716
PILOT_DURATION_MINUTES = 10
PILOT_STEP_SECONDS = 60
PILOT_MINIMUM_ELEVATION_DEG = 10.0
PILOT_SPACE_THRESHOLD = 0.8
PILOT_GROUND_THRESHOLD = 0.8
PILOT_EXPECTED_DESIGN_COUNT = 5
PILOT_REALIZATIONS_PER_DESIGN = 5
PILOT_EXPECTED_RUN_COUNT = 25
PILOT_SEED_MODULUS = 1 << 63
PILOT_SEED_DOMAIN = "satnet_integrated_ground_pilot_seed"
PILOT_SEED_VERSION = "1"
PILOT_DESIGN_DOMAIN = "satnet_integrated_ground_pilot_design"
PILOT_DESIGN_VERSION = "1"
PILOT_REALIZATION_DOMAIN = "satnet_integrated_ground_pilot_realization"
PILOT_REALIZATION_VERSION = "1"
PILOT_RUN_MANIFEST_DOMAIN = "satnet_integrated_ground_pilot_run_manifest"
PILOT_RUN_MANIFEST_VERSION = "1"
PILOT_CATALOG_PROVENANCE_DOMAIN = "satnet_integrated_ground_pilot_catalog_provenance"
PILOT_CATALOG_PROVENANCE_VERSION = "1"
PILOT_DESIGN_MANIFEST_DOMAIN = "satnet_integrated_ground_pilot_design_manifest"
PILOT_DESIGN_MANIFEST_VERSION = "1"
SEED_PURPOSES = (
    "satellite_rollout_and_failure",
    "ground_station_selection",
    "ground_failure_realization",
)
DESIGN_ID_PATTERN = re.compile(r"^P0[1-5]$")
REALIZATION_ID_PATTERN = re.compile(r"^R0[1-5]$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class IntegratedPilotDesign:
    design_id: str
    design_name: str
    design_description: str
    num_planes: int
    sats_per_plane: int
    altitude_km: float
    inclination_deg: float
    node_failure_probability: float
    edge_failure_probability: float
    civilian_count: int
    government_count: int
    military_count: int
    ground_failure_probability: float
    duration_minutes: int = PILOT_DURATION_MINUTES
    step_seconds: int = PILOT_STEP_SECONDS
    phasing_factor: int = 1
    max_isl_distance_km: float = 10000.0
    isl_policy: str = "grid_fixed"
    adjacent_search_k: int = 1
    max_inter_plane_links_per_sat: int = 1
    epoch_iso: str = DEFAULT_EPOCH_ISO
    orbital_engine: str = "sgp4"
    satellite_failure_model: str = DEFAULT_FAILURE_MODEL
    minimum_elevation_deg: float = PILOT_MINIMUM_ELEVATION_DEG
    space_gcc_threshold: float = PILOT_SPACE_THRESHOLD
    ground_service_threshold: float = PILOT_GROUND_THRESHOLD
    visibility_policy_hash: str = field(init=False)
    ground_service_policy_hash: str = field(init=False)
    ground_failure_policy_hash: str = field(init=False)
    design_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.design_id, str) or not DESIGN_ID_PATTERN.fullmatch(self.design_id):
            raise ValueError("design_id must be one of P01 through P05")
        for name in ("design_name", "design_description"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in (
            "num_planes",
            "sats_per_plane",
            "civilian_count",
            "government_count",
            "military_count",
            "duration_minutes",
            "step_seconds",
            "phasing_factor",
            "adjacent_search_k",
            "max_inter_plane_links_per_sat",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise TypeError(f"{name} must be a positive integer")
        for name in (
            "altitude_km",
            "inclination_deg",
            "node_failure_probability",
            "edge_failure_probability",
            "ground_failure_probability",
            "max_isl_distance_km",
            "minimum_elevation_deg",
            "space_gcc_threshold",
            "ground_service_threshold",
        ):
            if type(getattr(self, name)) is not float:
                raise TypeError(f"{name} must be a float")
        if self.duration_minutes != PILOT_DURATION_MINUTES or self.step_seconds != PILOT_STEP_SECONDS:
            raise ValueError("Pilot temporal profile is locked to 10 minutes at 60 seconds")
        if self.minimum_elevation_deg != PILOT_MINIMUM_ELEVATION_DEG:
            raise ValueError("Pilot minimum elevation is locked to 10 degrees")
        if (
            self.space_gcc_threshold != PILOT_SPACE_THRESHOLD
            or self.ground_service_threshold != PILOT_GROUND_THRESHOLD
        ):
            raise ValueError("Pilot service thresholds are locked to 0.80")
        visibility_policy = GroundVisibilityPolicy(self.minimum_elevation_deg)
        service_policy = GroundServicePolicy(
            self.space_gcc_threshold,
            self.ground_service_threshold,
        )
        failure_policy = GroundFailurePolicy(self.ground_failure_probability)
        object.__setattr__(
            self, "visibility_policy_hash", visibility_policy.visibility_policy_hash
        )
        object.__setattr__(
            self,
            "ground_service_policy_hash",
            service_policy.ground_service_policy_hash,
        )
        object.__setattr__(
            self,
            "ground_failure_policy_hash",
            failure_policy.ground_failure_policy_hash,
        )
        Tier1RolloutConfig(
            num_planes=self.num_planes,
            sats_per_plane=self.sats_per_plane,
            inclination_deg=self.inclination_deg,
            altitude_km=self.altitude_km,
            phasing_factor=self.phasing_factor,
            duration_minutes=self.duration_minutes,
            step_seconds=self.step_seconds,
            max_isl_distance_km=self.max_isl_distance_km,
            isl_policy=self.isl_policy,
            adjacent_search_k=self.adjacent_search_k,
            max_inter_plane_links_per_sat=self.max_inter_plane_links_per_sat,
            gcc_threshold=self.space_gcc_threshold,
            node_failure_prob=self.node_failure_probability,
            edge_failure_prob=self.edge_failure_probability,
            failure_model=self.satellite_failure_model,
            seed=0,
            epoch_iso=self.epoch_iso,
            orbital_engine=self.orbital_engine,
        )
        object.__setattr__(self, "design_hash", canonical_hash(self.identity_payload()))

    @property
    def configured_satellite_count(self) -> int:
        return self.num_planes * self.sats_per_plane

    @property
    def total_ground_station_count(self) -> int:
        return self.civilian_count + self.government_count + self.military_count

    def identity_payload(self) -> dict[str, object]:
        return {
            "adjacent_search_k": self.adjacent_search_k,
            "altitude_km": canonical_float_string(self.altitude_km),
            "civilian_count": self.civilian_count,
            "design_description": self.design_description,
            "design_id": self.design_id,
            "design_name": self.design_name,
            "duration_minutes": self.duration_minutes,
            "edge_failure_probability": canonical_float_string(
                self.edge_failure_probability
            ),
            "epoch_iso": self.epoch_iso,
            "government_count": self.government_count,
            "ground_failure_policy_hash": self.ground_failure_policy_hash,
            "ground_failure_probability": canonical_float_string(
                self.ground_failure_probability
            ),
            "ground_service_policy_hash": self.ground_service_policy_hash,
            "ground_service_threshold": canonical_float_string(
                self.ground_service_threshold
            ),
            "identity_domain": PILOT_DESIGN_DOMAIN,
            "identity_version": PILOT_DESIGN_VERSION,
            "inclination_deg": canonical_float_string(self.inclination_deg),
            "isl_policy": self.isl_policy,
            "max_inter_plane_links_per_sat": self.max_inter_plane_links_per_sat,
            "max_isl_distance_km": canonical_float_string(self.max_isl_distance_km),
            "military_count": self.military_count,
            "minimum_elevation_deg": canonical_float_string(
                self.minimum_elevation_deg
            ),
            "node_failure_probability": canonical_float_string(
                self.node_failure_probability
            ),
            "num_planes": self.num_planes,
            "orbital_engine": self.orbital_engine,
            "phasing_factor": self.phasing_factor,
            "satellite_failure_model": self.satellite_failure_model,
            "sats_per_plane": self.sats_per_plane,
            "space_gcc_threshold": canonical_float_string(self.space_gcc_threshold),
            "step_seconds": self.step_seconds,
            "visibility_policy_hash": self.visibility_policy_hash,
        }

    def to_manifest_object(self) -> dict[str, object]:
        result = asdict(self)
        for name in (
            "altitude_km",
            "inclination_deg",
            "node_failure_probability",
            "edge_failure_probability",
            "ground_failure_probability",
            "max_isl_distance_km",
            "minimum_elevation_deg",
            "space_gcc_threshold",
            "ground_service_threshold",
        ):
            result[name] = canonical_float_string(result[name])
        return result

    def satellite_config(self, *, satellite_seed: int) -> Tier1RolloutConfig:
        _validate_seed(satellite_seed, "satellite_seed")
        return Tier1RolloutConfig(
            num_planes=self.num_planes,
            sats_per_plane=self.sats_per_plane,
            inclination_deg=self.inclination_deg,
            altitude_km=self.altitude_km,
            phasing_factor=self.phasing_factor,
            duration_minutes=self.duration_minutes,
            step_seconds=self.step_seconds,
            max_isl_distance_km=self.max_isl_distance_km,
            isl_policy=self.isl_policy,
            adjacent_search_k=self.adjacent_search_k,
            max_inter_plane_links_per_sat=self.max_inter_plane_links_per_sat,
            gcc_threshold=self.space_gcc_threshold,
            node_failure_prob=self.node_failure_probability,
            edge_failure_prob=self.edge_failure_probability,
            failure_model=self.satellite_failure_model,
            seed=satellite_seed,
            epoch_iso=self.epoch_iso,
            orbital_engine=self.orbital_engine,
        )


@dataclass(frozen=True)
class IntegratedPilotRun:
    run_id: int
    design_id: str
    realization_id: str
    design_group_id: str
    design_hash: str
    satellite_rollout_seed: int
    ground_station_selection_seed: int
    ground_failure_seed: int
    realization_manifest_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.run_id) is not int or not 0 <= self.run_id < PILOT_EXPECTED_RUN_COUNT:
            raise ValueError("run_id must be within [0, 24]")
        if not isinstance(self.design_id, str) or not DESIGN_ID_PATTERN.fullmatch(self.design_id):
            raise ValueError("design_id must be one of P01 through P05")
        if not isinstance(self.realization_id, str) or not REALIZATION_ID_PATTERN.fullmatch(
            self.realization_id
        ):
            raise ValueError("realization_id must be one of R01 through R05")
        if self.design_group_id != self.design_id:
            raise ValueError("design_group_id must equal design_id")
        _validate_hash(self.design_hash, "design_hash")
        for name in (
            "satellite_rollout_seed",
            "ground_station_selection_seed",
            "ground_failure_seed",
        ):
            _validate_seed(getattr(self, name), name)
        expected_run = (int(self.design_id[1:]) - 1) * 5 + int(self.realization_id[1:]) - 1
        if self.run_id != expected_run:
            raise ValueError("run_id does not match deterministic design-major mapping")
        object.__setattr__(
            self,
            "realization_manifest_hash",
            canonical_hash(self.identity_payload()),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "design_group_id": self.design_group_id,
            "design_hash": self.design_hash,
            "design_id": self.design_id,
            "ground_failure_seed": self.ground_failure_seed,
            "ground_station_selection_seed": self.ground_station_selection_seed,
            "identity_domain": PILOT_REALIZATION_DOMAIN,
            "identity_version": PILOT_REALIZATION_VERSION,
            "realization_id": self.realization_id,
            "run_id": self.run_id,
            "satellite_rollout_seed": self.satellite_rollout_seed,
        }

    def to_manifest_object(self) -> dict[str, object]:
        return asdict(self)


def _validate_hash(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 value")


def _validate_seed(value: object, field_name: str) -> None:
    if type(value) is not int or not 0 <= value < PILOT_SEED_MODULUS:
        raise ValueError(f"{field_name} must be an integer within [0, 2**63 - 1]")


def derive_pilot_seed(
    *,
    design_id: str,
    realization_id: str,
    seed_purpose: str,
    master_seed: int = PILOT_MASTER_SEED,
) -> int:
    if not isinstance(design_id, str) or not DESIGN_ID_PATTERN.fullmatch(design_id):
        raise ValueError("design_id must be one of P01 through P05")
    if not isinstance(realization_id, str) or not REALIZATION_ID_PATTERN.fullmatch(
        realization_id
    ):
        raise ValueError("realization_id must be one of R01 through R05")
    if seed_purpose not in SEED_PURPOSES:
        raise ValueError(f"seed_purpose must be one of {SEED_PURPOSES}")
    if type(master_seed) is not int or not 0 <= master_seed < PILOT_SEED_MODULUS:
        raise ValueError("master_seed must be an integer within [0, 2**63 - 1]")
    payload = {
        "design_id": design_id,
        "identity_domain": PILOT_SEED_DOMAIN,
        "identity_version": PILOT_SEED_VERSION,
        "master_seed": master_seed,
        "realization_id": realization_id,
        "seed_purpose": seed_purpose,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % PILOT_SEED_MODULUS


def build_pilot_designs() -> tuple[IntegratedPilotDesign, ...]:
    values = (
        (
            "P01",
            "Large robust reference",
            "Large high-altitude reference with no configured failures",
            6,
            8,
            1200.0,
            98.0,
            0.0,
            0.0,
            8,
            6,
            6,
            0.0,
        ),
        (
            "P02",
            "Large constellation, light failures",
            "Large constellation with light persistent satellite and ground failures",
            6,
            8,
            800.0,
            60.0,
            0.05,
            0.05,
            12,
            4,
            4,
            0.05,
        ),
        (
            "P03",
            "Balanced middle case",
            "Middle constellation and ground segment with moderate failures",
            5,
            6,
            600.0,
            55.0,
            0.10,
            0.10,
            4,
            3,
            3,
            0.15,
        ),
        (
            "P04",
            "Sparse constellation, elevated failures",
            "Sparse constellation with elevated persistent failures",
            4,
            5,
            500.0,
            45.0,
            0.15,
            0.20,
            2,
            4,
            4,
            0.25,
        ),
        (
            "P05",
            "Stress case",
            "Low-altitude sparse constellation and minimal ground segment stress case",
            4,
            5,
            300.0,
            30.0,
            0.20,
            0.25,
            1,
            1,
            1,
            0.40,
        ),
    )
    return tuple(
        IntegratedPilotDesign(
            design_id=value[0],
            design_name=value[1],
            design_description=value[2],
            num_planes=value[3],
            sats_per_plane=value[4],
            altitude_km=value[5],
            inclination_deg=value[6],
            node_failure_probability=value[7],
            edge_failure_probability=value[8],
            civilian_count=value[9],
            government_count=value[10],
            military_count=value[11],
            ground_failure_probability=value[12],
        )
        for value in values
    )


def build_pilot_runs(
    designs: Sequence[IntegratedPilotDesign] | None = None,
) -> tuple[IntegratedPilotRun, ...]:
    normalized = tuple(build_pilot_designs() if designs is None else designs)
    validate_pilot_designs(normalized)
    result: list[IntegratedPilotRun] = []
    for design_index, design in enumerate(normalized):
        for realization_index in range(PILOT_REALIZATIONS_PER_DESIGN):
            realization_id = f"R{realization_index + 1:02d}"
            result.append(
                IntegratedPilotRun(
                    run_id=design_index * PILOT_REALIZATIONS_PER_DESIGN
                    + realization_index,
                    design_id=design.design_id,
                    realization_id=realization_id,
                    design_group_id=design.design_id,
                    design_hash=design.design_hash,
                    satellite_rollout_seed=derive_pilot_seed(
                        design_id=design.design_id,
                        realization_id=realization_id,
                        seed_purpose="satellite_rollout_and_failure",
                    ),
                    ground_station_selection_seed=derive_pilot_seed(
                        design_id=design.design_id,
                        realization_id=realization_id,
                        seed_purpose="ground_station_selection",
                    ),
                    ground_failure_seed=derive_pilot_seed(
                        design_id=design.design_id,
                        realization_id=realization_id,
                        seed_purpose="ground_failure_realization",
                    ),
                )
            )
    validate_pilot_runs(tuple(result), normalized)
    return tuple(result)


def validate_pilot_designs(designs: Sequence[IntegratedPilotDesign]) -> None:
    normalized = tuple(designs)
    if len(normalized) != PILOT_EXPECTED_DESIGN_COUNT:
        raise ValueError("Pilot requires exactly five designs")
    if any(not isinstance(value, IntegratedPilotDesign) for value in normalized):
        raise TypeError("Pilot designs must be IntegratedPilotDesign values")
    ids = [value.design_id for value in normalized]
    hashes = [value.design_hash for value in normalized]
    if ids != [f"P{index:02d}" for index in range(1, 6)]:
        raise ValueError("Pilot designs must use canonical P01 through P05 order")
    if len(hashes) != len(set(hashes)):
        raise ValueError("Pilot design hashes must be unique")


def validate_pilot_runs(
    runs: Sequence[IntegratedPilotRun],
    designs: Sequence[IntegratedPilotDesign],
) -> None:
    normalized = tuple(runs)
    validate_pilot_designs(designs)
    if len(normalized) != PILOT_EXPECTED_RUN_COUNT:
        raise ValueError("Pilot requires exactly 25 runs")
    if any(not isinstance(value, IntegratedPilotRun) for value in normalized):
        raise TypeError("Pilot runs must be IntegratedPilotRun values")
    if [value.run_id for value in normalized] != list(range(PILOT_EXPECTED_RUN_COUNT)):
        raise ValueError("Pilot runs must use canonical run-ID order 0 through 24")
    pairs = [(value.design_id, value.realization_id) for value in normalized]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Pilot design-realization pairs must be unique")
    design_by_id = {value.design_id: value for value in designs}
    for run in normalized:
        if run.design_id not in design_by_id:
            raise ValueError("Pilot run references an unknown design")
        if run.design_hash != design_by_id[run.design_id].design_hash:
            raise ValueError("Pilot run design hash does not match design manifest")
    counts = {design.design_id: 0 for design in designs}
    for run in normalized:
        counts[run.design_id] += 1
    if set(counts.values()) != {PILOT_REALIZATIONS_PER_DESIGN}:
        raise ValueError("Each pilot design must have exactly five realizations")


def pilot_run_manifest_hash(runs: Sequence[IntegratedPilotRun]) -> str:
    normalized = tuple(runs)
    return canonical_hash(
        {
            "identity_domain": PILOT_RUN_MANIFEST_DOMAIN,
            "identity_version": PILOT_RUN_MANIFEST_VERSION,
            "runs": [value.to_manifest_object() for value in normalized],
        }
    )


def pilot_design_manifest_hash(designs: Sequence[IntegratedPilotDesign]) -> str:
    normalized = tuple(designs)
    return canonical_hash(
        {
            "designs": [value.to_manifest_object() for value in normalized],
            "identity_domain": PILOT_DESIGN_MANIFEST_DOMAIN,
            "identity_version": PILOT_DESIGN_MANIFEST_VERSION,
        }
    )


def build_synthetic_pilot_catalog() -> GroundStationCatalog:
    prefixes = {
        GroundStationClass.CIVILIAN: "CIV",
        GroundStationClass.GOVERNMENT: "GOV",
        GroundStationClass.MILITARY: "MIL",
    }
    stations: list[GroundStation] = []
    for class_index, station_class in enumerate(GroundStationClass):
        for index in range(50):
            latitude = -62.0 + (index % 10) * 13.5 + class_index * 1.25
            longitude = -170.0 + (index // 10) * 72.0 + class_index * 4.0
            stations.append(
                GroundStation(
                    station_id=f"{prefixes[station_class]}_PILOT_{index + 1:03d}",
                    name=f"Synthetic Pilot {station_class.value.title()} {index + 1:03d}",
                    station_class=station_class,
                    latitude_deg=latitude,
                    longitude_deg=longitude,
                    altitude_m=25.0 + (index % 10) * 75.0 + class_index * 10.0,
                    region=f"region_{(index % 10) + 1:02d}",
                    country_code="ZZ",
                    enabled=True,
                )
            )
    catalog = GroundStationCatalog(tuple(stations))
    validate_production_catalog_readiness(catalog)
    return catalog


def catalog_provenance(catalog: GroundStationCatalog) -> dict[str, object]:
    if not isinstance(catalog, GroundStationCatalog):
        raise TypeError("catalog must be a GroundStationCatalog")
    class_counts = {
        station_class.value: len(catalog.eligible(station_class))
        for station_class in GroundStationClass
    }
    return {
        "catalog_hash": catalog.catalog_hash,
        "country_code": "ZZ",
        "enabled_class_counts": class_counts,
        "identity_domain": PILOT_CATALOG_PROVENANCE_DOMAIN,
        "identity_version": PILOT_CATALOG_PROVENANCE_VERSION,
        "pilot_only": True,
        "production_research_catalog_availability": "NOT AVAILABLE",
        "production_research_catalog_scientific_review": "NOT PERFORMED",
        "provenance_description": (
            "Deterministic synthetic geometry generated for integrated engineering validation"
        ),
        "scientifically_reviewed": False,
        "sensitive_locations": False,
        "synthetic": True,
        "total_enabled_station_count": sum(class_counts.values()),
    }


def _atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a Boolean")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Pilot artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"Pilot artifact already exists: {path}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_synthetic_pilot_catalog(
    catalog: GroundStationCatalog,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(catalog, GroundStationCatalog):
        raise TypeError("catalog must be a GroundStationCatalog")
    output = Path(path)
    if output.suffix != ".csv":
        raise ValueError("Pilot catalog must use .csv")
    rows = [",".join(CATALOG_COLUMNS)]
    for station in catalog.stations:
        record = station.canonical_record()
        rows.append(
            ",".join(
                (
                    str(record["station_id"]),
                    str(record["name"]),
                    str(record["station_class"]),
                    str(record["latitude_deg"]),
                    str(record["longitude_deg"]),
                    str(record["altitude_m"]),
                    str(record["region"]),
                    str(record["country_code"]),
                    "true" if record["enabled"] else "false",
                )
            )
        )
    _atomic_write_text(output, "\n".join(rows) + "\n", overwrite=overwrite)
    if load_ground_station_catalog(output) != catalog:
        raise RuntimeError("Persisted pilot catalog did not round-trip exactly")


def write_catalog_provenance(
    catalog: GroundStationCatalog,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    output = Path(path)
    if output.suffix != ".json":
        raise ValueError("Catalog provenance must use .json")
    _atomic_write_text(
        output,
        canonical_json(catalog_provenance(catalog)) + "\n",
        overwrite=overwrite,
    )


def write_pilot_design_manifest(
    designs: Sequence[IntegratedPilotDesign],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    normalized = tuple(designs)
    validate_pilot_designs(normalized)
    output = Path(path)
    if output.suffix != ".json":
        raise ValueError("Pilot design manifest must use .json")
    value = {
        "design_manifest_hash": pilot_design_manifest_hash(normalized),
        "designs": [design.to_manifest_object() for design in normalized],
        "pilot_schema_version": PILOT_SCHEMA_VERSION,
    }
    _atomic_write_text(output, canonical_json(value) + "\n", overwrite=overwrite)


def write_pilot_run_manifest(
    runs: Sequence[IntegratedPilotRun],
    designs: Sequence[IntegratedPilotDesign],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    normalized = tuple(runs)
    validate_pilot_runs(normalized, designs)
    output = Path(path)
    if output.suffix != ".jsonl":
        raise ValueError("Pilot run manifest must use .jsonl")
    _atomic_write_text(
        output,
        "\n".join(canonical_json(value.to_manifest_object()) for value in normalized)
        + "\n",
        overwrite=overwrite,
    )


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key '{key}'")
        result[key] = value
    return result


def _read_json(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    if not text or not text.endswith("\n") or text.count("\n") != 1:
        raise ValueError(f"Pilot JSON artifact must be one nonempty newline-terminated record: {path}")
    try:
        return json.loads(text, object_pairs_hook=_pairs_without_duplicates)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Malformed pilot JSON artifact {path}: {exc}") from exc


def _design_from_object(value: object) -> IntegratedPilotDesign:
    if not isinstance(value, dict):
        raise TypeError("Pilot design must be an object")
    expected = {field.name for field in fields(IntegratedPilotDesign)}
    if set(value) != expected:
        raise ValueError(
            f"Pilot design fields invalid; missing={sorted(expected-set(value))}, "
            f"unknown={sorted(set(value)-expected)}"
        )
    converted = dict(value)
    for name in (
        "altitude_km",
        "inclination_deg",
        "node_failure_probability",
        "edge_failure_probability",
        "ground_failure_probability",
        "max_isl_distance_km",
        "minimum_elevation_deg",
        "space_gcc_threshold",
        "ground_service_threshold",
    ):
        raw = converted[name]
        if not isinstance(raw, str):
            raise TypeError(f"{name} must be a canonical float string")
        parsed = float(raw)
        if canonical_float_string(parsed) != raw:
            raise ValueError(f"{name} is not canonical")
        converted[name] = parsed
    supplied_hashes = {
        name: converted.pop(name)
        for name in (
            "visibility_policy_hash",
            "ground_service_policy_hash",
            "ground_failure_policy_hash",
            "design_hash",
        )
    }
    design = IntegratedPilotDesign(**converted)
    for name, supplied in supplied_hashes.items():
        if getattr(design, name) != supplied:
            raise ValueError(f"Pilot design {name} mismatch")
    return design


def read_pilot_design_manifest(path: str | Path) -> tuple[IntegratedPilotDesign, ...]:
    value = _read_json(Path(path))
    if not isinstance(value, dict) or set(value) != {
        "design_manifest_hash",
        "designs",
        "pilot_schema_version",
    }:
        raise ValueError("Pilot design manifest fields are invalid")
    if value["pilot_schema_version"] != PILOT_SCHEMA_VERSION:
        raise ValueError("Unsupported pilot_schema_version")
    if not isinstance(value["designs"], list):
        raise TypeError("Pilot designs must be an array")
    designs = tuple(_design_from_object(item) for item in value["designs"])
    validate_pilot_designs(designs)
    if value["design_manifest_hash"] != pilot_design_manifest_hash(designs):
        raise ValueError("Pilot design manifest hash mismatch")
    return designs


def _run_from_object(value: object) -> IntegratedPilotRun:
    if not isinstance(value, dict):
        raise TypeError("Pilot run must be an object")
    expected = {field.name for field in fields(IntegratedPilotRun)}
    if set(value) != expected:
        raise ValueError(
            f"Pilot run fields invalid; missing={sorted(expected-set(value))}, "
            f"unknown={sorted(set(value)-expected)}"
        )
    converted = dict(value)
    supplied_hash = converted.pop("realization_manifest_hash")
    run = IntegratedPilotRun(**converted)
    if run.realization_manifest_hash != supplied_hash:
        raise ValueError("Pilot realization manifest hash mismatch")
    return run


def read_pilot_run_manifest(
    path: str | Path,
    designs: Sequence[IntegratedPilotDesign],
) -> tuple[IntegratedPilotRun, ...]:
    output = Path(path)
    if output.suffix != ".jsonl":
        raise ValueError("Pilot run manifest must use .jsonl")
    text = output.read_text(encoding="utf-8")
    if not text or not text.endswith("\n"):
        raise ValueError("Pilot run manifest must be nonempty and newline terminated")
    lines = text.splitlines()
    runs: list[IntegratedPilotRun] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError("Pilot run manifest contains an empty line")
        try:
            value = json.loads(line, object_pairs_hook=_pairs_without_duplicates)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Line {line_number}: malformed pilot run: {exc}") from exc
        runs.append(_run_from_object(value))
    normalized = tuple(runs)
    validate_pilot_runs(normalized, designs)
    return normalized


def materialize_pilot_inputs(
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    root = Path(output_root)
    input_root = root / "inputs"
    catalog = build_synthetic_pilot_catalog()
    designs = build_pilot_designs()
    runs = build_pilot_runs(designs)
    catalog_path = input_root / "pilot_catalog.csv"
    provenance_path = input_root / "pilot_catalog_provenance.json"
    design_path = input_root / "pilot_designs.json"
    run_path = input_root / "pilot_runs.jsonl"
    write_synthetic_pilot_catalog(catalog, catalog_path, overwrite=overwrite)
    write_catalog_provenance(catalog, provenance_path, overwrite=overwrite)
    write_pilot_design_manifest(designs, design_path, overwrite=overwrite)
    write_pilot_run_manifest(runs, designs, run_path, overwrite=overwrite)
    return {
        "catalog_hash": catalog.catalog_hash,
        "catalog_path": str(catalog_path),
        "design_manifest_hash": pilot_design_manifest_hash(designs),
        "design_manifest_path": str(design_path),
        "run_manifest_hash": pilot_run_manifest_hash(runs),
        "run_manifest_path": str(run_path),
    }
