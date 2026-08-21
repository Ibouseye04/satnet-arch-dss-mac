"""Orchestration over authoritative SATNET G1-G5 ground contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from statistics import mean
from typing import Sequence

from satnet.dss.realization import DSSRealizationSeeds
from satnet.dss.schemas import DSSArchitectureRequest
from satnet.dss.scenario_builder import DSSScenario
from satnet.ground.catalog import GroundStationCatalog, load_ground_station_catalog
from satnet.ground.failure_policy import GroundFailurePolicy
from satnet.ground.failure_realization import sample_ground_failure_realization
from satnet.ground.failure_service_metrics import compute_failure_adjusted_ground_service_step
from satnet.ground.integrated_builder import build_integrated_ground_graph
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.ground.service_metrics import compute_ground_service_step
from satnet.ground.service_policy import GroundServicePolicy
from satnet.ground.visibility import GroundVisibilityPolicy, evaluate_ground_design_visibility_sequence

DSS_MINIMUM_ELEVATION_DEG = 10.0


@dataclass(frozen=True)
class GroundScenarioMetrics:
    mean_ground_service_fraction: float
    minimum_ground_service_fraction: float
    mean_overall_service_fraction: float
    minimum_overall_service_fraction: float
    ground_realization_hash: str
    ground_design_hash: str
    catalog_hash: str
    visibility_policy_hash: str
    service_policy_hash: str
    realization_temporal_ground_means: tuple[float, ...]
    realization_temporal_overall_means: tuple[float, ...]


def resolve_catalog_path(catalog_path: str | os.PathLike[str] | None = None) -> Path:
    candidate = catalog_path or os.environ.get("SATNET_DSS_GROUND_CATALOG")
    if not candidate:
        raise RuntimeError(
            "SATNET_DSS_GROUND_CATALOG is required to validate G1 capacity and calculate G4/G5"
        )
    path = Path(candidate)
    if not path.is_file():
        raise RuntimeError(f"SATNET DSS ground-station catalog does not exist: {path}")
    return path


def prepare_ground_design(
    architecture: DSSArchitectureRequest,
    *,
    catalog: GroundStationCatalog,
    station_selection_seed: int,
):
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(
            civilian_count=architecture.civilian_count,
            government_count=architecture.government_count,
            military_count=architecture.military_count,
            station_selection_seed=station_selection_seed,
        ),
    )
    return make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash="0" * 64,
        selection=selection,
    )


def calculate_ground_context(
    architecture: DSSArchitectureRequest,
    scenarios: Sequence[DSSScenario],
    seeds: Sequence[DSSRealizationSeeds],
    *,
    catalog_path: str | os.PathLike[str] | None = None,
) -> GroundScenarioMetrics:
    if len(scenarios) != 5 or len(seeds) != 5:
        raise ValueError("Ground context requires exactly five DSS scenarios and seed sets")
    catalog = load_ground_station_catalog(resolve_catalog_path(catalog_path))
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(
            civilian_count=architecture.civilian_count,
            government_count=architecture.government_count,
            military_count=architecture.military_count,
            station_selection_seed=seeds[0].ground_station_selection_seed,
        ),
    )
    visibility_policy = GroundVisibilityPolicy(DSS_MINIMUM_ELEVATION_DEG)
    service_policy = GroundServicePolicy(
        space_gcc_threshold=architecture.required_minimum_connectivity,
        ground_service_threshold=architecture.required_minimum_connectivity,
    )
    ground_failure_policy = GroundFailurePolicy(
        architecture.ground_station_failure_probability
    )

    all_ground_values: list[float] = []
    all_overall_values: list[float] = []
    realization_ground_means: list[float] = []
    realization_overall_means: list[float] = []
    realization_hashes: list[str] = []
    ground_design_hashes: list[str] = []
    for scenario, seed_set in zip(scenarios, seeds):
        ground_design = make_enabled_ground_design_record(
            run_id=0,
            satellite_config_hash=scenario.rollout_config.config_hash(),
            selection=selection,
        )
        visibility = evaluate_ground_design_visibility_sequence(
            ground_design=ground_design,
            catalog=catalog,
            satellite_sequence=scenario.position_snapshots,
            policy=visibility_policy,
        )
        integrated = tuple(
            build_integrated_ground_graph(
                ground_design=ground_design,
                catalog=catalog,
                satellite_graph_snapshot=graph_snapshot,
                verified_visibility_snapshot=visibility_snapshot,
            )
            for graph_snapshot, visibility_snapshot in zip(
                scenario.graph_snapshots, visibility
            )
        )
        baseline_metrics = tuple(
            compute_ground_service_step(
                integrated_snapshot=snapshot,
                ground_design=ground_design,
                catalog=catalog,
                configured_satellite_count=scenario.rollout_config.total_satellites,
                policy=service_policy,
            )
            for snapshot in integrated
        )
        ground_failure = sample_ground_failure_realization(
            ground_design=ground_design,
            catalog=catalog,
            policy=ground_failure_policy,
            ground_failure_seed=seed_set.ground_failure_seed,
        )
        adjusted_metrics = tuple(
            compute_failure_adjusted_ground_service_step(
                baseline_metrics=baseline,
                ground_design=ground_design,
                catalog=catalog,
                failure_realization=ground_failure,
                ground_service_policy=service_policy,
            )
            for baseline in baseline_metrics
        )
        ground_values = [metric.failure_adjusted_ground_service_fraction for metric in adjusted_metrics]
        overall_values = [metric.failure_adjusted_overall_service_fraction for metric in adjusted_metrics]
        all_ground_values.extend(ground_values)
        all_overall_values.extend(overall_values)
        realization_ground_means.append(mean(ground_values))
        realization_overall_means.append(mean(overall_values))
        realization_hashes.append(ground_failure.ground_failure_realization_hash)
        ground_design_hashes.append(ground_design.ground_design_hash)

    if len(set(realization_hashes)) != len(realization_hashes):
        raise RuntimeError("Ground failure realization identities are not unique")
    return GroundScenarioMetrics(
        mean_ground_service_fraction=mean(realization_ground_means),
        minimum_ground_service_fraction=min(all_ground_values),
        mean_overall_service_fraction=mean(realization_overall_means),
        minimum_overall_service_fraction=min(all_overall_values),
        ground_realization_hash=";".join(realization_hashes),
        ground_design_hash=";".join(ground_design_hashes),
        catalog_hash=catalog.catalog_hash,
        visibility_policy_hash=visibility_policy.visibility_policy_hash,
        service_policy_hash=service_policy.ground_service_policy_hash,
        realization_temporal_ground_means=tuple(realization_ground_means),
        realization_temporal_overall_means=tuple(realization_overall_means),
    )
