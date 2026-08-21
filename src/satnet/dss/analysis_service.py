"""Python-level entry point for the SATNET Phase 1 DSS."""

from __future__ import annotations

from typing import Any

from satnet.dss.domain import (
    DSSAnalysisResult,
    DSSModelMetadata,
    DSSSpaceResilience,
    DSSSystemContext,
)
from satnet.dss.ground_context import calculate_ground_context, resolve_catalog_path
from satnet.dss.realization import derive_all_realization_seeds
from satnet.dss.schemas import DSSArchitectureRequest, DSS_REALIZATION_COUNT
from satnet.dss.scenario_builder import build_scenario, scenario_provenance
from satnet.dss.tgnn_inference import (
    FROZEN_TGNN_CHECKPOINT_SHA256,
    FROZEN_TGNN_TARGET,
    FROZEN_TGNN_TASK,
    load_frozen_tgnn,
    predict_graph_sequence,
)
from satnet.ground.catalog import load_ground_station_catalog
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations


def _coerce_request(request: DSSArchitectureRequest | dict[str, Any]) -> DSSArchitectureRequest:
    if isinstance(request, DSSArchitectureRequest):
        return request
    return DSSArchitectureRequest.from_mapping(request)


def aggregate_space_predictions(
    predictions: list[float], threshold: float
) -> DSSSpaceResilience:
    if len(predictions) != DSS_REALIZATION_COUNT:
        raise ValueError("DSS aggregation requires exactly five predictions")
    expected = sum(predictions) / DSS_REALIZATION_COUNT
    lowest = min(predictions)
    highest = max(predictions)
    meeting_count = sum(prediction >= threshold for prediction in predictions)
    return DSSSpaceResilience(
        expected_minimum_gcc=expected,
        lowest_modeled_gcc=lowest,
        highest_modeled_gcc=highest,
        required_minimum_connectivity=threshold,
        expected_margin=expected - threshold,
        lowest_margin=lowest - threshold,
        expected_assessment=(
            "MEETS_EXPECTED_REQUIREMENT"
            if expected >= threshold
            else "BELOW_EXPECTED_REQUIREMENT"
        ),
        realizations_meeting_requirement=meeting_count,
        realization_count=DSS_REALIZATION_COUNT,
        realization_risk_flag=any(prediction < threshold for prediction in predictions),
    )


def analyze_architecture(
    request: DSSArchitectureRequest | dict[str, Any],
    *,
    checkpoint_path: str | None = None,
    ground_catalog_path: str | None = None,
) -> DSSAnalysisResult:
    """Analyze one architecture through five deterministic Tier 1 realizations."""
    architecture = _coerce_request(request)

    # Validate catalog capacity before expensive physics/model work. This is G1
    # validation; selection is architecture-level and is shared by all realizations.
    catalog = load_ground_station_catalog(resolve_catalog_path(ground_catalog_path))
    seeds = derive_all_realization_seeds(architecture)
    select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(
            civilian_count=architecture.civilian_count,
            government_count=architecture.government_count,
            military_count=architecture.military_count,
            station_selection_seed=seeds[0].ground_station_selection_seed,
        ),
    )

    model, checkpoint_sha256 = load_frozen_tgnn(checkpoint_path)
    scenarios = tuple(build_scenario(architecture, seed_set) for seed_set in seeds)
    predictions: list[float] = []
    feature_shapes: list[list[tuple[int, int, int]]] = []
    graph_hashes: list[list[str]] = []
    for scenario in scenarios:
        prediction, shapes = predict_graph_sequence(
            model,
            scenario.graphs,
            num_planes=architecture.num_planes,
            sats_per_plane=architecture.sats_per_plane,
        )
        predictions.append(prediction)
        feature_shapes.append(shapes)
        graph_hashes.append([snapshot.graph_hash for snapshot in scenario.graph_snapshots])

    if len(predictions) != DSS_REALIZATION_COUNT:
        raise RuntimeError("DSS did not produce exactly five TGNN predictions")
    threshold = architecture.required_minimum_connectivity
    space = aggregate_space_predictions(predictions, threshold)
    expected_minimum_gcc = space.expected_minimum_gcc

    try:
        ground = calculate_ground_context(architecture, scenarios, seeds, catalog_path=ground_catalog_path)
        if expected_minimum_gcc < ground.mean_ground_service_fraction:
            limiting_segment = "SPACE"
        elif expected_minimum_gcc > ground.mean_ground_service_fraction:
            limiting_segment = "GROUND"
        else:
            limiting_segment = "TIE"
        system_context = DSSSystemContext(
            mean_ground_service_fraction=ground.mean_ground_service_fraction,
            minimum_ground_service_fraction=ground.minimum_ground_service_fraction,
            mean_overall_service_fraction=ground.mean_overall_service_fraction,
            minimum_overall_service_fraction=ground.minimum_overall_service_fraction,
            limiting_segment=limiting_segment,
            ground_provenance="SATNET_CALCULATED",
            overall_provenance="SATNET_CALCULATED",
            status="AVAILABLE",
        )
        ground_provenance: dict[str, object] = {
            "status": "AVAILABLE",
            "catalog_hash": ground.catalog_hash,
            "ground_design_hash": ground.ground_design_hash,
            "visibility_policy_hash": ground.visibility_policy_hash,
            "service_policy_hash": ground.service_policy_hash,
            "ground_failure_realization_hashes": ground.ground_realization_hash.split(";"),
        }
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        system_context = DSSSystemContext(
            mean_ground_service_fraction=None,
            minimum_ground_service_fraction=None,
            mean_overall_service_fraction=None,
            minimum_overall_service_fraction=None,
            limiting_segment=None,
            ground_provenance="SATNET_CALCULATED",
            overall_provenance="SATNET_CALCULATED",
            status="BLOCKED",
            blocked_reason=str(exc),
        )
        ground_provenance = {"status": "BLOCKED", "blocked_reason": str(exc)}

    return DSSAnalysisResult(
        architecture=architecture.to_dict(),
        model=DSSModelMetadata(
            family="TGNN",
            task="space_regression",
            target=FROZEN_TGNN_TARGET,
            checkpoint_sha256=checkpoint_sha256,
            realization_count=DSS_REALIZATION_COUNT,
        ),
        space_resilience=space,
        system_context=system_context,
        provenance={
            "operational_model": FROZEN_TGNN_TASK,
            "prediction_provenance": "TGNN_PREDICTION",
            "ground_provenance": ground_provenance,
            "architecture_hash": architecture.architecture_hash,
            "physical_architecture_hash": architecture.physical_architecture_hash,
            "seed_policy": "SHA-256(canonical physical architecture + frozen domain + realization index)",
            "threshold_in_seed_derivation": False,
            "scenario": scenario_provenance(),
            "training_performed": False,
        },
        analysis_details={
            "realization_predictions": predictions,
            "realization_prediction_out_of_physical_interval": [
                prediction < 0.0 or prediction > 1.0 for prediction in predictions
            ],
            "realization_seeds": [
                {
                    "realization_index": seed_set.realization_index,
                    "satellite_failure_seed": seed_set.satellite_failure_seed,
                    "ground_failure_seed": seed_set.ground_failure_seed,
                    "ground_station_selection_seed": seed_set.ground_station_selection_seed,
                }
                for seed_set in seeds
            ],
            "temporal_feature_shapes": feature_shapes,
            "temporal_graph_hashes": graph_hashes,
        },
    )
