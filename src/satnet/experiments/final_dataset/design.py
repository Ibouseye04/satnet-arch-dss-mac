from __future__ import annotations

from itertools import product
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

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
from satnet.experiments.production_profile import (
    HISTORICAL_FIXED_PROFILE,
    ProductionTopologyProfile,
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
RUN_RECORD_IDENTITY_VERSION = "2"
DESIGN_MANIFEST_IDENTITY_DOMAIN = "satnet_final_integrated_dataset_design_manifest"
DESIGN_MANIFEST_IDENTITY_VERSION = "1"
RUN_MANIFEST_IDENTITY_DOMAIN = "satnet_final_integrated_dataset_run_manifest"
RUN_MANIFEST_IDENTITY_VERSION = "2"
REALIZATIONS_PER_DESIGN = 5
FINAL_DESIGN_COUNT = 2000
FINAL_RUN_COUNT = FINAL_DESIGN_COUNT * REALIZATIONS_PER_DESIGN
PILOT_ANCHOR_COUNT = 5
TRANSITION_DESIGN_COUNT = 395
GLOBAL_DESIGN_COUNT = 1600
DESIGN_ID_WIDTH = 4
RUN_RECORD_FIELDS = frozenset(
    {
        "contract_spec_hash",
        "run_id",
        "run_key",
        "design_id",
        "design_index",
        "design_group_id",
        "design_record_hash",
        "realization_id",
        "realization_index",
        "split_assignment",
        "satellite_seed_purpose",
        "satellite_seed",
        "ground_failure_seed_purpose",
        "ground_failure_seed",
        "ground_selection_seed",
        "ground_selection_hash",
        "ground_design_hash",
        "expected_satellite_config_hash",
        "run_record_hash",
    }
)


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
    profile: ProductionTopologyProfile = HISTORICAL_FIXED_PROFILE,
    pilot_design_manifest_hash_value: str = PILOT_DESIGN_MANIFEST_HASH,
) -> dict[str, Any]:
    specification = build_contract_specification(
        profile=profile,
        pilot_design_manifest_hash_value=pilot_design_manifest_hash_value,
    )
    fixed = specification["fixed_profile"]
    design_id = f"D{design_index:0{DESIGN_ID_WIDTH}d}"
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
    *,
    pilot_designs: tuple[IntegratedPilotDesign, ...],
    catalog: GroundStationCatalog,
    profile: ProductionTopologyProfile,
    pilot_design_manifest_hash_value: str,
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
                profile=profile,
                pilot_design_manifest_hash_value=pilot_design_manifest_hash_value,
            )
        )
    return result


def _transition_records(
    *,
    catalog: GroundStationCatalog,
    profile: ProductionTopologyProfile,
    pilot_design_manifest_hash_value: str,
) -> list[dict[str, Any]]:
    ranges = {
        "altitude_km": (600.0, 800.0),
        "inclination_deg": (55.0, 60.0),
        "satellite_node_failure_probability": (0.05, 0.10),
        "satellite_edge_failure_probability": (0.05, 0.10),
        "ground_station_failure_probability": (0.05, 0.15),
    }
    rows, evidence = select_lhs(
        row_count=TRANSITION_DESIGN_COUNT, stratum_id="transition", ranges=ranges
    )
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
        for pair, count in zip(
            ((5, 6), (5, 7), (5, 8), (6, 6), (6, 7), (6, 8)),
            (66, 66, 66, 66, 66, 65),
            strict=True,
        )
        for _ in range(count)
    ]
    ground_cells = ground_cells * 11 + ground_cells[:10]
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
                design_index=PILOT_ANCHOR_COUNT + offset,
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
                profile=profile,
                pilot_design_manifest_hash_value=pilot_design_manifest_hash_value,
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
        for repetition in range(160):
            composition = weights[(total_index + repetition) % len(weights)]
            result.append((total, composition, allocate_ground_classes(total, composition)))
    return result


def _global_records(
    *,
    catalog: GroundStationCatalog,
    profile: ProductionTopologyProfile,
    pilot_design_manifest_hash_value: str,
) -> list[dict[str, Any]]:
    ranges = {
        "altitude_km": (300.0, 1200.0),
        "inclination_deg": (30.0, 98.0),
        "satellite_node_failure_probability": (0.0, 0.20),
        "satellite_edge_failure_probability": (0.0, 0.25),
        "ground_station_failure_probability": (0.0, 0.40),
    }
    rows, evidence = select_lhs(
        row_count=GLOBAL_DESIGN_COUNT, stratum_id="global", ranges=ranges
    )
    satellite_pairs = [
        pair
        for pair, count in zip(
            product((4, 5, 6), (5, 6, 7, 8)),
            (134, 134, 134, 134, 133, 133, 133, 133, 133, 133, 133, 133),
            strict=True,
        )
        for _ in range(count)
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
                design_index=PILOT_ANCHOR_COUNT + TRANSITION_DESIGN_COUNT + offset,
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
                profile=profile,
                pilot_design_manifest_hash_value=pilot_design_manifest_hash_value,
            )
        )
    return result


def build_design_records(
    *,
    pilot_design_manifest: str | Path,
    catalog_path: str | Path,
    profile: ProductionTopologyProfile = HISTORICAL_FIXED_PROFILE,
) -> tuple[dict[str, Any], ...]:
    profile.validate()
    pilot_designs = read_pilot_design_manifest(pilot_design_manifest)
    actual_pilot_hash = pilot_design_manifest_hash(pilot_designs)
    if profile is HISTORICAL_FIXED_PROFILE and actual_pilot_hash != PILOT_DESIGN_MANIFEST_HASH:
        raise ValueError("Pilot design manifest identity does not match the contract")
    if profile is not HISTORICAL_FIXED_PROFILE:
        for pilot in pilot_designs:
            profile.assert_matches(pilot.identity_payload())
            if pilot.satellite_failure_model != profile.failure_model:
                raise ValueError("Adaptive pilot manifest uses the wrong failure model")
    catalog = load_ground_station_catalog(catalog_path)
    if catalog.catalog_hash != CATALOG_HASH:
        raise ValueError("Pilot catalog semantic identity does not match the contract")
    records = (
        _anchor_records(
            pilot_designs=pilot_designs,
            catalog=catalog,
            profile=profile,
            pilot_design_manifest_hash_value=actual_pilot_hash,
        )
        + _transition_records(
            catalog=catalog,
            profile=profile,
            pilot_design_manifest_hash_value=actual_pilot_hash,
        )
        + _global_records(
            catalog=catalog,
            profile=profile,
            pilot_design_manifest_hash_value=actual_pilot_hash,
        )
    )
    validate_design_records(records, pilot_designs=pilot_designs)
    return tuple(records)


def _assert_equal(actual: object, expected: object, field_name: str) -> None:
    if actual != expected:
        raise ValueError(f"Anchor field mismatch for {field_name}: {actual!r} != {expected!r}")


DESIGN_SCIENTIFIC_FIELDS = (
    "doe_stratum",
    "num_planes",
    "sats_per_plane",
    "altitude_km",
    "inclination_deg",
    "satellite_node_failure_probability",
    "satellite_edge_failure_probability",
    "civilian_count",
    "government_count",
    "military_count",
    "ground_station_failure_probability",
)


def design_scientific_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in DESIGN_SCIENTIFIC_FIELDS)


def validate_design_records(
    records: Iterable[dict[str, Any]], *, pilot_designs: tuple[IntegratedPilotDesign, ...]
) -> None:
    normalized = tuple(records)
    if len(normalized) != FINAL_DESIGN_COUNT:
        raise ValueError("Final 10k contract requires exactly 2,000 designs")
    if [record["design_id"] for record in normalized] != [f"D{index:0{DESIGN_ID_WIDTH}d}" for index in range(FINAL_DESIGN_COUNT)]:
        raise ValueError("Design IDs or ordering are invalid")
    if [record["design_index"] for record in normalized] != list(range(FINAL_DESIGN_COUNT)):
        raise ValueError("Design indices are invalid")
    if len({record["design_record_hash"] for record in normalized}) != FINAL_DESIGN_COUNT:
        raise ValueError("Design-record hashes must be unique")
    records_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in normalized:
        records_by_signature.setdefault(design_scientific_signature(record), []).append(record)
    for signature, matching in records_by_signature.items():
        if len(matching) > 1 and not all(record["pilot_design_id"] for record in matching):
            raise ValueError(f"Non-anchor scientific design signature is duplicated: {signature!r}")
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
    if counts != {"pilot_anchor": PILOT_ANCHOR_COUNT, "transition": TRANSITION_DESIGN_COUNT, "global": GLOBAL_DESIGN_COUNT}:
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
    transition = normalized[PILOT_ANCHOR_COUNT : PILOT_ANCHOR_COUNT + TRANSITION_DESIGN_COUNT]
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
    if transition_pair_counts != {(5, 6): 66, (5, 7): 66, (5, 8): 66, (6, 6): 66, (6, 7): 66, (6, 8): 65}:
        raise ValueError("Transition satellite-pair frequencies are invalid")
    global_records = normalized[PILOT_ANCHOR_COUNT + TRANSITION_DESIGN_COUNT :]
    for total in (6, 10, 15, 20, 25, 30, 35, 40, 45, 50):
        if sum(record["total_ground_station_count"] == total for record in global_records) != 160:
            raise ValueError("Global station-total frequencies are invalid")
    global_weights = ((1, 1, 1), (3, 1, 1), (1, 3, 1), (1, 1, 3), (9, 9, 2), (9, 2, 9), (2, 9, 9), (8, 1, 1), (1, 8, 1), (1, 1, 8))
    for weights in global_weights:
        if sum(tuple(record["composition_weights"]) == weights for record in global_records) != 160:
            raise ValueError("Global composition frequencies are invalid")
    for pair in product((4, 5, 6), (5, 6, 7, 8)):
        expected_count = 134 if pair in tuple(product((4,), (5, 6, 7, 8))) else 133
        if sum((record["num_planes"], record["sats_per_plane"]) == pair for record in global_records) != expected_count:
            raise ValueError("Global satellite-pair frequencies are invalid")


def design_manifest_hash(records: Iterable[dict[str, Any]]) -> str:
    normalized = tuple(records)
    return canonical_hash(
        {
            "designs": list(normalized),
            "identity_domain": DESIGN_MANIFEST_IDENTITY_DOMAIN,
            "identity_version": DESIGN_MANIFEST_IDENTITY_VERSION,
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


def build_run_records(
    designs: Iterable[dict[str, Any]], *, design_split: Mapping[str, str]
) -> tuple[dict[str, Any], ...]:
    normalized = tuple(designs)
    if set(design_split) != {design["design_id"] for design in normalized}:
        raise ValueError("Design split mapping does not cover exactly the final designs")
    records: list[dict[str, Any]] = []
    for design in normalized:
        design_index = design["design_index"]
        design_id = design["design_id"]
        for realization_index in range(REALIZATIONS_PER_DESIGN):
            realization_id = f"R{realization_index:02d}"
            run_id = design_index * REALIZATIONS_PER_DESIGN + realization_index
            run_key = f"{design_id}-{realization_id}"
            satellite_seed = derived_seed(satellite_seed_payload(design_id, realization_id))
            ground_failure_seed = derived_seed(
                ground_failure_seed_payload(design_id, realization_id)
            )
            record: dict[str, Any] = {
                "contract_spec_hash": design["contract_spec_hash"],
                "run_id": run_id,
                "run_key": run_key,
                "design_id": design_id,
                "design_index": design_index,
                "design_group_id": design["design_group_id"],
                "design_record_hash": design["design_record_hash"],
                "realization_id": realization_id,
                "realization_index": realization_index,
                "split_assignment": design_split[design_id],
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
    normalized_designs = tuple(designs)
    design_by_id = {design["design_id"]: design for design in normalized_designs}
    if len(design_by_id) != FINAL_DESIGN_COUNT:
        raise ValueError("Final run manifest requires exactly 2,000 unique designs")
    if len(normalized) != FINAL_RUN_COUNT:
        raise ValueError("Final run manifest requires exactly 10,000 runs")
    for record in normalized:
        if set(record) != RUN_RECORD_FIELDS:
            raise ValueError("Run record fields do not match the authoritative schema")
        if type(record.get("run_id")) is not int:
            raise TypeError("run_id must be an exact integer")
        if not 0 <= record["run_id"] < FINAL_RUN_COUNT:
            raise ValueError("run_id must be within 0 through 499")
        if not isinstance(record.get("run_key"), str):
            raise TypeError("run_key must be a string")
        if type(record.get("design_index")) is not int:
            raise TypeError("design_index must be an exact integer")
        if type(record.get("realization_index")) is not int:
            raise TypeError("realization_index must be an exact integer")
    run_ids = [record["run_id"] for record in normalized]
    if run_ids != list(range(FINAL_RUN_COUNT)) or set(run_ids) != set(range(FINAL_RUN_COUNT)):
        raise ValueError("Run IDs must cover ordered integers 0 through 499")
    run_keys = [record["run_key"] for record in normalized]
    if len(run_keys) != len(set(run_keys)):
        raise ValueError("Run keys must be unique")
    design_realizations = [
        (record["design_id"], record["realization_id"]) for record in normalized
    ]
    if len(design_realizations) != len(set(design_realizations)):
        raise ValueError("Design-realization pairs must be unique")
    for record in normalized:
        if record["design_id"] not in design_by_id:
            raise ValueError("Run references an unknown design")
        design = design_by_id[record["design_id"]]
        if record["design_index"] != design["design_index"]:
            raise ValueError("Run design index differs from its design")
        realization_index = record["realization_index"]
        if not 0 <= realization_index < REALIZATIONS_PER_DESIGN:
            raise ValueError("realization_index must be within 0 through 4")
        expected_realization_id = f"R{realization_index:02d}"
        expected_run_id = (
            record["design_index"] * REALIZATIONS_PER_DESIGN + realization_index
        )
        expected_run_key = f"{record['design_id']}-{record['realization_id']}"
        if record["realization_id"] != expected_realization_id:
            raise ValueError("Realization ID differs from its zero-based index")
        if record["run_id"] != expected_run_id:
            raise ValueError("Run ID differs from the authoritative mapping")
        if record["run_key"] != expected_run_key:
            raise ValueError("Run key differs from the authoritative mapping")
        if record["split_assignment"] not in {"train", "validation", "test"}:
            raise ValueError("Run split assignment is invalid")
        if record["contract_spec_hash"] != design["contract_spec_hash"]:
            raise ValueError("Run references the wrong contract specification")
        for field in ("ground_selection_seed", "ground_selection_hash", "ground_design_hash"):
            if record[field] != design[field]:
                raise ValueError(f"Run-level {field} differs from its design")
        if record["design_record_hash"] != design["design_record_hash"]:
            raise ValueError("Run references the wrong design-record hash")
        expected_satellite_seed = derived_seed(
            satellite_seed_payload(record["design_id"], record["realization_id"])
        )
        expected_ground_failure_seed = derived_seed(
            ground_failure_seed_payload(record["design_id"], record["realization_id"])
        )
        if record["satellite_seed_purpose"] != "satellite_rollout_and_failure":
            raise ValueError("Satellite seed purpose is invalid")
        if record["ground_failure_seed_purpose"] != "ground_failure_realization":
            raise ValueError("Ground-failure seed purpose is invalid")
        if record["satellite_seed"] != expected_satellite_seed:
            raise ValueError("Satellite seed differs from canonical derivation")
        if record["ground_failure_seed"] != expected_ground_failure_seed:
            raise ValueError("Ground-failure seed differs from canonical derivation")
        if record["expected_satellite_config_hash"] != _satellite_config(
            design, expected_satellite_seed
        ).config_hash():
            raise ValueError("Expected satellite configuration hash is invalid")
        if record["run_record_hash"] != _record_hash(
            record,
            domain=RUN_RECORD_IDENTITY_DOMAIN,
            version=RUN_RECORD_IDENTITY_VERSION,
            hash_field="run_record_hash",
        ):
            raise ValueError("Run-record hash mismatch")
    for design_id in design_by_id:
        group = [record for record in normalized if record["design_id"] == design_id]
        if (
            len(group) != REALIZATIONS_PER_DESIGN
            or len({record["ground_design_hash"] for record in group}) != 1
            or len({record["split_assignment"] for record in group}) != 1
        ):
            raise ValueError("Every design must have five colocated fixed-ground realizations")


def run_manifest_hash(records: Iterable[dict[str, Any]]) -> str:
    normalized = tuple(records)
    return canonical_hash(
        {
            "identity_domain": RUN_MANIFEST_IDENTITY_DOMAIN,
            "identity_version": RUN_MANIFEST_IDENTITY_VERSION,
            "runs": list(normalized),
        }
    )
