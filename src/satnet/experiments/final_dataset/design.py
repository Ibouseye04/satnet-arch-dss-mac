from __future__ import annotations

from itertools import product
import math
from pathlib import Path
from typing import Any, Iterable

from satnet.experiments.final_dataset.deterministic import (
    allocate_ground_classes,
    deterministic_order,
    derived_seed,
    ground_failure_seed_payload,
    ground_selection_seed_payload,
    satellite_seed_payload,
    select_lhs,
)
from satnet.experiments.final_dataset.specification import (
    CANONICAL_DIMENSION_ORDER,
    CATALOG_HASH,
    PILOT_DESIGN_MANIFEST_HASH,
    build_contract_specification,
)
from satnet.experiments.integrated_ground_manifest import (
    IntegratedPilotDesign,
    pilot_design_manifest_hash,
    read_pilot_design_manifest,
)
from satnet.ground.canonical import canonical_float_string, canonical_hash
from satnet.ground.catalog import GroundStationCatalog, load_ground_station_catalog
from satnet.ground.failure_policy import GroundFailurePolicy
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.simulation.tier1_rollout import Tier1RolloutConfig

DESIGN_RECORD_IDENTITY_DOMAIN = "satnet_final_integrated_dataset_design_record"
DESIGN_RECORD_IDENTITY_VERSION = "1"
RUN_RECORD_IDENTITY_DOMAIN = "satnet_final_integrated_dataset_run_record"
RUN_RECORD_IDENTITY_VERSION = "1"
DESIGN_MANIFEST_IDENTITY_DOMAIN = "satnet_final_integrated_dataset_design_manifest"
RUN_MANIFEST_IDENTITY_DOMAIN = "satnet_final_integrated_dataset_run_manifest"
MANIFEST_IDENTITY_VERSION = "1"


def _c(value: float) -> str:
    return canonical_float_string(value)


def _record_hash(record: dict[str, Any], *, domain: str, version: str, hash_field: str) -> str:
    payload = {
        "identity_domain": domain,
        "identity_version": version,
        **{key: value for key, value in record.items() if key != hash_field},
    }
    return canonical_hash(payload)


def _ground_architecture(
    *,
    catalog: GroundStationCatalog,
    design_id: str,
    civilian_count: int,
    government_count: int,
    military_count: int,
) -> dict[str, Any]:
    selection_seed = derived_seed(ground_selection_seed_payload(design_id))
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(
            civilian_count=civilian_count,
            government_count=government_count,
            military_count=military_count,
            station_selection_seed=selection_seed,
        ),
    )
    template = make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash="0" * 64,
        selection=selection,
    )
    return {
        "ground_selection_seed": selection_seed,
        "civilian_selected_station_ids": list(selection.civilian_station_ids),
        "government_selected_station_ids": list(selection.government_station_ids),
        "military_selected_station_ids": list(selection.military_station_ids),
        "selected_station_ids": list(selection.selected_station_ids),
        "ground_selection_hash": selection.selection_hash,
        "ground_design_hash": template.ground_design_hash,
    }


def _base_design_record(
    *,
    design_index: int,
    doe_stratum: str,
    num_planes: int,
    sats_per_plane: int,
    altitude_km: float,
    inclination_deg: float,
    node_probability: float,
    edge_probability: float,
    civilian_count: int,
    government_count: int,
    military_count: int,
    ground_probability: float,
    catalog: GroundStationCatalog,
    composition_weights: tuple[int, int, int] | None,
    lhs_candidate_id: int | None,
    pilot: IntegratedPilotDesign | None = None,
) -> dict[str, Any]:
    specification = build_contract_specification()
    fixed = specification["fixed_profile"]
    design_id = f"D{design_index:03d}"
    if pilot is None:
        design_name = f"Final integrated {doe_stratum} design {design_id}"
        description = f"Pre-outcome deterministic {doe_stratum} design"
    else:
        design_name = f"Pilot anchor {pilot.design_id}: {pilot.design_name}"
        description = f"Final scientific-parameter anchor sourced from {pilot.design_id}"
    record: dict[str, Any] = {
        "contract_spec_hash": specification["contract_spec_hash"],
        "design_id": design_id,
        "design_index": design_index,
        "design_group_id": design_id,
        "design_name": design_name,
        "design_description": description,
        "doe_stratum": doe_stratum,
        "pilot_design_id": None if pilot is None else pilot.design_id,
        "pilot_design_hash": None if pilot is None else pilot.design_hash,
        "pilot_design_manifest_hash": None if pilot is None else PILOT_DESIGN_MANIFEST_HASH,
        "num_planes": num_planes,
        "sats_per_plane": sats_per_plane,
        "configured_satellite_count": num_planes * sats_per_plane,
        "altitude_km": _c(altitude_km),
        "inclination_deg": _c(inclination_deg),
        "phasing_factor": fixed["phasing_factor"],
        "satellite_node_failure_probability": _c(node_probability),
        "satellite_edge_failure_probability": _c(edge_probability),
        "civilian_count": civilian_count,
        "government_count": government_count,
        "military_count": military_count,
        "total_ground_station_count": civilian_count + government_count + military_count,
        "ground_station_failure_probability": _c(ground_probability),
        "ground_failure_policy_hash": GroundFailurePolicy(
            ground_probability
        ).ground_failure_policy_hash,
        "composition_weights": None if composition_weights is None else list(composition_weights),
        "lhs_candidate_id": lhs_candidate_id,
        "duration_minutes": fixed["duration_minutes"],
        "step_seconds": fixed["step_seconds"],
        "epoch_iso": fixed["epoch_iso"],
        "orbital_engine": fixed["orbital_engine"],
        "max_isl_distance_km": fixed["max_isl_distance_km"],
        "isl_policy": fixed["isl_policy"],
        "adjacent_search_k": fixed["adjacent_search_k"],
        "max_inter_plane_links_per_sat": fixed[
            "max_inter_plane_links_per_sat"
        ],
        "satellite_failure_model": fixed["failure_model"],
        "minimum_elevation_deg": fixed["minimum_elevation_deg"],
        "space_gcc_threshold": fixed["space_gcc_threshold"],
        "ground_service_threshold": fixed["ground_service_threshold"],
        "visibility_policy_hash": fixed["visibility_policy_hash"],
        "ground_service_policy_hash": fixed["ground_service_policy_hash"],
    }
    record.update(
        _ground_architecture(
            catalog=catalog,
            design_id=design_id,
            civilian_count=civilian_count,
            government_count=government_count,
            military_count=military_count,
        )
    )
    record["design_record_hash"] = _record_hash(
        record,
        domain=DESIGN_RECORD_IDENTITY_DOMAIN,
        version=DESIGN_RECORD_IDENTITY_VERSION,
        hash_field="design_record_hash",
    )
    return record


def _anchor_records(
    *, pilot_designs: tuple[IntegratedPilotDesign, ...], catalog: GroundStationCatalog
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for design_index, pilot in enumerate(pilot_designs):
        result.append(
            _base_design_record(
                design_index=design_index,
                doe_stratum="pilot_anchor",
                num_planes=pilot.num_planes,
                sats_per_plane=pilot.sats_per_plane,
                altitude_km=pilot.altitude_km,
                inclination_deg=pilot.inclination_deg,
                node_probability=pilot.node_failure_probability,
                edge_probability=pilot.edge_failure_probability,
                civilian_count=pilot.civilian_count,
                government_count=pilot.government_count,
                military_count=pilot.military_count,
                ground_probability=pilot.ground_failure_probability,
                catalog=catalog,
                composition_weights=None,
                lhs_candidate_id=None,
                pilot=pilot,
            )
        )
    return result


def _transition_records(*, catalog: GroundStationCatalog) -> list[dict[str, Any]]:
    ranges = {
        "altitude_km": (600.0, 800.0),
        "inclination_deg": (55.0, 60.0),
        "satellite_node_failure_probability": (0.05, 0.10),
        "satellite_edge_failure_probability": (0.05, 0.10),
        "ground_station_failure_probability": (0.05, 0.15),
    }
    rows, evidence = select_lhs(row_count=35, stratum_id="transition", ranges=ranges)
    ground_cells = [
        (total, weights, allocate_ground_classes(total, weights))
        for total, weights in product(
            (10, 12, 15, 18, 20),
            (
                (1, 1, 1),
                (3, 1, 1),
                (1, 3, 1),
                (1, 1, 3),
                (2, 2, 1),
                (2, 1, 2),
                (1, 2, 2),
            ),
        )
    ]
    satellite_pairs = [
        pair
        for pair, count in (
            ((5, 6), 6),
            ((5, 7), 6),
            ((5, 8), 6),
            ((6, 6), 6),
            ((6, 7), 6),
            ((6, 8), 5),
        )
        for _ in range(count)
    ]
    paired_rows = deterministic_order(
        rows, stratum_id="transition", schedule_purpose="continuous_rows"
    )
    paired_ground = deterministic_order(
        ground_cells, stratum_id="transition", schedule_purpose="ground_schedule"
    )
    paired_satellites = deterministic_order(
        satellite_pairs,
        stratum_id="transition",
        schedule_purpose="satellite_pair_schedule",
    )
    result: list[dict[str, Any]] = []
    for offset, (row, ground, satellites) in enumerate(
        zip(paired_rows, paired_ground, paired_satellites, strict=True)
    ):
        _, weights, counts = ground
        result.append(
            _base_design_record(
                design_index=5 + offset,
                doe_stratum="transition",
                num_planes=satellites[0],
                sats_per_plane=satellites[1],
                altitude_km=row["altitude_km"],
                inclination_deg=row["inclination_deg"],
                node_probability=row["satellite_node_failure_probability"],
                edge_probability=row["satellite_edge_failure_probability"],
                civilian_count=counts[0],
                government_count=counts[1],
                military_count=counts[2],
                ground_probability=row["ground_station_failure_probability"],
                catalog=catalog,
                composition_weights=weights,
                lhs_candidate_id=evidence["candidate_id"],
            )
        )
    return result


def _global_ground_schedule() -> list[tuple[int, tuple[int, int, int], tuple[int, int, int]]]:
    totals = (6, 10, 15, 20, 25, 30, 35, 40, 45, 50)
    weights = (
        (1, 1, 1),
        (3, 1, 1),
        (1, 3, 1),
        (1, 1, 3),
        (9, 9, 2),
        (9, 2, 9),
        (2, 9, 9),
        (8, 1, 1),
        (1, 8, 1),
        (1, 1, 8),
    )
    result: list[tuple[int, tuple[int, int, int], tuple[int, int, int]]] = []
    for total_index, total in enumerate(totals):
        for repetition in range(6):
            composition = weights[(total_index + repetition) % len(weights)]
            result.append((total, composition, allocate_ground_classes(total, composition)))
    return result


def _global_records(*, catalog: GroundStationCatalog) -> list[dict[str, Any]]:
    ranges = {
        "altitude_km": (300.0, 1200.0),
        "inclination_deg": (30.0, 98.0),
        "satellite_node_failure_probability": (0.0, 0.20),
        "satellite_edge_failure_probability": (0.0, 0.25),
        "ground_station_failure_probability": (0.0, 0.40),
    }
    rows, evidence = select_lhs(row_count=60, stratum_id="global", ranges=ranges)
    satellite_pairs = [
        pair for pair in product((4, 5, 6), (5, 6, 7, 8)) for _ in range(5)
    ]
    paired_rows = deterministic_order(
        rows, stratum_id="global", schedule_purpose="continuous_rows"
    )
    paired_ground = deterministic_order(
        _global_ground_schedule(), stratum_id="global", schedule_purpose="ground_schedule"
    )
    paired_satellites = deterministic_order(
        satellite_pairs,
        stratum_id="global",
        schedule_purpose="satellite_pair_schedule",
    )
    result: list[dict[str, Any]] = []
    for offset, (row, ground, satellites) in enumerate(
        zip(paired_rows, paired_ground, paired_satellites, strict=True)
    ):
        _, weights, counts = ground
        result.append(
            _base_design_record(
                design_index=40 + offset,
                doe_stratum="global",
                num_planes=satellites[0],
                sats_per_plane=satellites[1],
                altitude_km=row["altitude_km"],
                inclination_deg=row["inclination_deg"],
                node_probability=row["satellite_node_failure_probability"],
                edge_probability=row["satellite_edge_failure_probability"],
                civilian_count=counts[0],
                government_count=counts[1],
                military_count=counts[2],
                ground_probability=row["ground_station_failure_probability"],
                catalog=catalog,
                composition_weights=weights,
                lhs_candidate_id=evidence["candidate_id"],
            )
        )
    return result


def build_design_records(
    *, pilot_design_manifest: str | Path, catalog_path: str | Path
) -> tuple[dict[str, Any], ...]:
    pilot_designs = read_pilot_design_manifest(pilot_design_manifest)
    if pilot_design_manifest_hash(pilot_designs) != PILOT_DESIGN_MANIFEST_HASH:
        raise ValueError("Pilot design manifest identity does not match the contract")
    catalog = load_ground_station_catalog(catalog_path)
    if catalog.catalog_hash != CATALOG_HASH:
        raise ValueError("Pilot catalog semantic identity does not match the contract")
    records = (
        _anchor_records(pilot_designs=pilot_designs, catalog=catalog)
        + _transition_records(catalog=catalog)
        + _global_records(catalog=catalog)
    )
    validate_design_records(records, pilot_designs=pilot_designs)
    return tuple(records)


def _assert_equal(actual: object, expected: object, field_name: str) -> None:
    if actual != expected:
        raise ValueError(f"Anchor field mismatch for {field_name}: {actual!r} != {expected!r}")


def validate_design_records(
    records: Iterable[dict[str, Any]], *, pilot_designs: tuple[IntegratedPilotDesign, ...]
) -> None:
    normalized = tuple(records)
    if len(normalized) != 100:
        raise ValueError("Final contract requires exactly 100 designs")
    if [record["design_id"] for record in normalized] != [f"D{index:03d}" for index in range(100)]:
        raise ValueError("Design IDs or ordering are invalid")
    if [record["design_index"] for record in normalized] != list(range(100)):
        raise ValueError("Design indices are invalid")
    if len({record["design_record_hash"] for record in normalized}) != 100:
        raise ValueError("Design-record hashes must be unique")
    for record in normalized:
        expected_hash = _record_hash(
            record,
            domain=DESIGN_RECORD_IDENTITY_DOMAIN,
            version=DESIGN_RECORD_IDENTITY_VERSION,
            hash_field="design_record_hash",
        )
        if record["design_record_hash"] != expected_hash:
            raise ValueError("Design-record hash mismatch")
        selected = (
            record["civilian_selected_station_ids"]
            + record["government_selected_station_ids"]
            + record["military_selected_station_ids"]
        )
        if selected != record["selected_station_ids"]:
            raise ValueError("Selected IDs do not preserve authoritative G1 class order")
        if len(selected) != record["total_ground_station_count"]:
            raise ValueError("Ground count does not match selected IDs")
    counts = {stratum: sum(record["doe_stratum"] == stratum for record in normalized) for stratum in ("pilot_anchor", "transition", "global")}
    if counts != {"pilot_anchor": 5, "transition": 35, "global": 60}:
        raise ValueError("DOE stratum counts are invalid")
    anchor_fields = {
        "num_planes": "num_planes",
        "sats_per_plane": "sats_per_plane",
        "altitude_km": "altitude_km",
        "inclination_deg": "inclination_deg",
        "phasing_factor": "phasing_factor",
        "satellite_node_failure_probability": "node_failure_probability",
        "satellite_edge_failure_probability": "edge_failure_probability",
        "civilian_count": "civilian_count",
        "government_count": "government_count",
        "military_count": "military_count",
        "ground_station_failure_probability": "ground_failure_probability",
        "duration_minutes": "duration_minutes",
        "step_seconds": "step_seconds",
        "isl_policy": "isl_policy",
        "adjacent_search_k": "adjacent_search_k",
        "max_inter_plane_links_per_sat": "max_inter_plane_links_per_sat",
        "orbital_engine": "orbital_engine",
        "satellite_failure_model": "satellite_failure_model",
    }
    float_fields = {
        "altitude_km",
        "inclination_deg",
        "satellite_node_failure_probability",
        "satellite_edge_failure_probability",
        "ground_station_failure_probability",
    }
    for record, pilot in zip(normalized[:5], pilot_designs, strict=True):
        if record["pilot_design_id"] != pilot.design_id or record["pilot_design_hash"] != pilot.design_hash:
            raise ValueError("Pilot anchor provenance mismatch")
        for final_name, pilot_name in anchor_fields.items():
            expected = getattr(pilot, pilot_name)
            if final_name in float_fields:
                expected = _c(expected)
            _assert_equal(record[final_name], expected, final_name)
        _assert_equal(record["max_isl_distance_km"], _c(pilot.max_isl_distance_km), "max_isl_distance_km")
        _assert_equal(record["minimum_elevation_deg"], _c(pilot.minimum_elevation_deg), "minimum_elevation_deg")
        _assert_equal(record["space_gcc_threshold"], _c(pilot.space_gcc_threshold), "space_gcc_threshold")
        _assert_equal(record["ground_service_threshold"], _c(pilot.ground_service_threshold), "ground_service_threshold")
    transition = normalized[5:40]
    transition_cells = {
        (
            record["total_ground_station_count"],
            tuple(record["composition_weights"]),
        )
        for record in transition
    }
    if len(transition_cells) != 35:
        raise ValueError("Transition ground Cartesian coverage is incomplete")
    transition_pair_counts = {
        pair: sum((record["num_planes"], record["sats_per_plane"]) == pair for record in transition)
        for pair in ((5, 6), (5, 7), (5, 8), (6, 6), (6, 7), (6, 8))
    }
    if transition_pair_counts != {(5, 6): 6, (5, 7): 6, (5, 8): 6, (6, 6): 6, (6, 7): 6, (6, 8): 5}:
        raise ValueError("Transition satellite-pair frequencies are invalid")
    global_records = normalized[40:]
    for total in (6, 10, 15, 20, 25, 30, 35, 40, 45, 50):
        if sum(record["total_ground_station_count"] == total for record in global_records) != 6:
            raise ValueError("Global station-total frequencies are invalid")
    global_weights = ((1, 1, 1), (3, 1, 1), (1, 3, 1), (1, 1, 3), (9, 9, 2), (9, 2, 9), (2, 9, 9), (8, 1, 1), (1, 8, 1), (1, 1, 8))
    for weights in global_weights:
        if sum(tuple(record["composition_weights"]) == weights for record in global_records) != 6:
            raise ValueError("Global composition frequencies are invalid")
    for pair in product((4, 5, 6), (5, 6, 7, 8)):
        if sum((record["num_planes"], record["sats_per_plane"]) == pair for record in global_records) != 5:
            raise ValueError("Global satellite-pair frequencies are invalid")


def design_manifest_hash(records: Iterable[dict[str, Any]]) -> str:
    normalized = tuple(records)
    return canonical_hash(
        {
            "designs": list(normalized),
            "identity_domain": DESIGN_MANIFEST_IDENTITY_DOMAIN,
            "identity_version": MANIFEST_IDENTITY_VERSION,
        }
    )


def _satellite_config(record: dict[str, Any], seed: int) -> Tier1RolloutConfig:
    return Tier1RolloutConfig(
        num_planes=record["num_planes"],
        sats_per_plane=record["sats_per_plane"],
        inclination_deg=float(record["inclination_deg"]),
        altitude_km=float(record["altitude_km"]),
        phasing_factor=record["phasing_factor"],
        duration_minutes=record["duration_minutes"],
        step_seconds=record["step_seconds"],
        max_isl_distance_km=float(record["max_isl_distance_km"]),
        isl_policy=record["isl_policy"],
        adjacent_search_k=record["adjacent_search_k"],
        max_inter_plane_links_per_sat=record["max_inter_plane_links_per_sat"],
        gcc_threshold=float(record["space_gcc_threshold"]),
        node_failure_prob=float(record["satellite_node_failure_probability"]),
        edge_failure_prob=float(record["satellite_edge_failure_probability"]),
        failure_model=record["satellite_failure_model"],
        seed=seed,
        epoch_iso=record["epoch_iso"],
        orbital_engine=record["orbital_engine"],
    )


def build_run_records(designs: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    normalized = tuple(designs)
    records: list[dict[str, Any]] = []
    for design in normalized:
        for realization_index in range(1, 6):
            realization_id = f"R{realization_index:02d}"
            satellite_seed = derived_seed(
                satellite_seed_payload(design["design_id"], realization_id)
            )
            ground_failure_seed = derived_seed(
                ground_failure_seed_payload(design["design_id"], realization_id)
            )
            run_index = design["design_index"] * 5 + realization_index - 1
            record: dict[str, Any] = {
                "contract_spec_hash": design["contract_spec_hash"],
                "run_id": f"{design['design_id']}-{realization_id}",
                "run_index": run_index,
                "design_id": design["design_id"],
                "design_group_id": design["design_group_id"],
                "design_record_hash": design["design_record_hash"],
                "realization_id": realization_id,
                "satellite_seed_purpose": "satellite_rollout_and_failure",
                "satellite_seed": satellite_seed,
                "ground_failure_seed_purpose": "ground_failure_realization",
                "ground_failure_seed": ground_failure_seed,
                "ground_selection_seed": design["ground_selection_seed"],
                "ground_selection_hash": design["ground_selection_hash"],
                "ground_design_hash": design["ground_design_hash"],
                "expected_satellite_config_hash": _satellite_config(
                    design, satellite_seed
                ).config_hash(),
            }
            record["run_record_hash"] = _record_hash(
                record,
                domain=RUN_RECORD_IDENTITY_DOMAIN,
                version=RUN_RECORD_IDENTITY_VERSION,
                hash_field="run_record_hash",
            )
            records.append(record)
    validate_run_records(records, designs=normalized)
    return tuple(records)


def validate_run_records(
    records: Iterable[dict[str, Any]], *, designs: Iterable[dict[str, Any]]
) -> None:
    normalized = tuple(records)
    design_by_id = {design["design_id"]: design for design in designs}
    if len(normalized) != 500 or [record["run_index"] for record in normalized] != list(range(500)):
        raise ValueError("Final run manifest must contain ordered run indices 0 through 499")
    if len({record["run_id"] for record in normalized}) != 500:
        raise ValueError("Run IDs must be unique")
    for record in normalized:
        design = design_by_id[record["design_id"]]
        for field in ("ground_selection_seed", "ground_selection_hash", "ground_design_hash"):
            if record[field] != design[field]:
                raise ValueError(f"Run-level {field} differs from its design")
        if record["design_record_hash"] != design["design_record_hash"]:
            raise ValueError("Run references the wrong design-record hash")
        if not 0 <= record["satellite_seed"] < 2**63 or not 0 <= record["ground_failure_seed"] < 2**63:
            raise ValueError("Run seed is outside the production range")
        if record["run_record_hash"] != _record_hash(
            record,
            domain=RUN_RECORD_IDENTITY_DOMAIN,
            version=RUN_RECORD_IDENTITY_VERSION,
            hash_field="run_record_hash",
        ):
            raise ValueError("Run-record hash mismatch")
    for design_id in design_by_id:
        group = [record for record in normalized if record["design_id"] == design_id]
        if len(group) != 5 or len({record["ground_design_hash"] for record in group}) != 1:
            raise ValueError("Every design must have five fixed-ground realizations")


def run_manifest_hash(records: Iterable[dict[str, Any]]) -> str:
    normalized = tuple(records)
    return canonical_hash(
        {
            "identity_domain": RUN_MANIFEST_IDENTITY_DOMAIN,
            "identity_version": MANIFEST_IDENTITY_VERSION,
            "runs": list(normalized),
        }
    )
