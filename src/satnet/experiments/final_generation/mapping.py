from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from satnet.ground.failure_policy import GroundFailurePolicy
from satnet.ground.selection import GroundSegmentEnabledConfig
from satnet.ground.service_policy import GroundServicePolicy
from satnet.ground.visibility import GroundVisibilityPolicy
from satnet.simulation.tier1_rollout import Tier1RolloutConfig

from .constants import CONTRACT_SPEC_HASH
from .io import parse_canonical_float


@dataclass(frozen=True)
class FinalRunMapping:
    design: dict[str, Any]
    run: dict[str, Any]
    satellite_config: Tier1RolloutConfig
    ground_config: GroundSegmentEnabledConfig
    visibility_policy: GroundVisibilityPolicy
    service_policy: GroundServicePolicy
    failure_policy: GroundFailurePolicy

    @property
    def run_id(self) -> int:
        return self.run["run_id"]

    @property
    def run_key(self) -> str:
        return self.run["run_key"]


def map_frozen_run(design: dict[str, Any], run: dict[str, Any]) -> FinalRunMapping:
    if design.get("contract_spec_hash") != CONTRACT_SPEC_HASH:
        raise ValueError("Design contract specification hash mismatch")
    if run.get("contract_spec_hash") != CONTRACT_SPEC_HASH:
        raise ValueError("Run contract specification hash mismatch")
    if run.get("design_id") != design.get("design_id"):
        raise ValueError("Run does not reference supplied design")
    if run.get("design_record_hash") != design.get("design_record_hash"):
        raise ValueError("Run design-record hash mismatch")
    design_index = design.get("design_index")
    realization_index = run.get("realization_index")
    run_id = run.get("run_id")
    if type(design_index) is not int or type(realization_index) is not int or type(run_id) is not int:
        raise TypeError("Frozen run indices and run_id must be integers")
    if run_id != design_index * 5 + realization_index:
        raise ValueError("Frozen run_id formula mismatch")
    if run.get("run_key") != f"{design['design_id']}-{run['realization_id']}":
        raise ValueError("Frozen run_key mismatch")
    satellite = Tier1RolloutConfig(
        num_planes=design["num_planes"],
        sats_per_plane=design["sats_per_plane"],
        inclination_deg=parse_canonical_float(design["inclination_deg"], "inclination_deg"),
        altitude_km=parse_canonical_float(design["altitude_km"], "altitude_km"),
        phasing_factor=design["phasing_factor"],
        duration_minutes=design["duration_minutes"],
        step_seconds=design["step_seconds"],
        max_isl_distance_km=parse_canonical_float(design["max_isl_distance_km"], "max_isl_distance_km"),
        isl_policy=design["isl_policy"],
        adjacent_search_k=design["adjacent_search_k"],
        max_inter_plane_links_per_sat=design["max_inter_plane_links_per_sat"],
        gcc_threshold=parse_canonical_float(design["space_gcc_threshold"], "space_gcc_threshold"),
        node_failure_prob=parse_canonical_float(
            design["satellite_node_failure_probability"],
            "satellite_node_failure_probability",
        ),
        edge_failure_prob=parse_canonical_float(
            design["satellite_edge_failure_probability"],
            "satellite_edge_failure_probability",
        ),
        failure_model=design["satellite_failure_model"],
        seed=run["satellite_seed"],
        epoch_iso=design["epoch_iso"],
        orbital_engine=design["orbital_engine"],
    )
    if satellite.config_hash() != run["expected_satellite_config_hash"]:
        raise ValueError("Mapped satellite configuration hash mismatch")
    ground = GroundSegmentEnabledConfig(
        civilian_count=design["civilian_count"],
        government_count=design["government_count"],
        military_count=design["military_count"],
        station_selection_seed=run["ground_selection_seed"],
    )
    visibility = GroundVisibilityPolicy(
        parse_canonical_float(design["minimum_elevation_deg"], "minimum_elevation_deg")
    )
    service = GroundServicePolicy(
        parse_canonical_float(design["space_gcc_threshold"], "space_gcc_threshold"),
        parse_canonical_float(design["ground_service_threshold"], "ground_service_threshold"),
    )
    failure = GroundFailurePolicy(
        parse_canonical_float(
            design["ground_station_failure_probability"],
            "ground_station_failure_probability",
        )
    )
    if run["ground_selection_seed"] != design["ground_selection_seed"]:
        raise ValueError("Run substitutes frozen ground-selection seed")
    if visibility.visibility_policy_hash != design["visibility_policy_hash"]:
        raise ValueError("Mapped visibility policy hash mismatch")
    if service.ground_service_policy_hash != design["ground_service_policy_hash"]:
        raise ValueError("Mapped service policy hash mismatch")
    if failure.ground_failure_policy_hash != design["ground_failure_policy_hash"]:
        raise ValueError("Mapped failure policy hash mismatch")
    return FinalRunMapping(
        design=design,
        run=run,
        satellite_config=satellite,
        ground_config=ground,
        visibility_policy=visibility,
        service_policy=service,
        failure_policy=failure,
    )


def map_all_runs(contract: dict[str, Any]) -> tuple[FinalRunMapping, ...]:
    designs = {record["design_id"]: record for record in contract["designs"]}
    mapped = tuple(map_frozen_run(designs[run["design_id"]], run) for run in contract["runs"])
    if tuple(value.run_id for value in mapped) != tuple(range(500)):
        raise ValueError("Mapped run order is not exactly 0 through 499")
    return mapped
