from __future__ import annotations

from copy import deepcopy
from typing import Any

from satnet.ground.canonical import canonical_hash

TARGET_SCHEMA_IDENTITY_DOMAIN = "satnet_final_integrated_target_schema"
RF_SCHEMA_IDENTITY_DOMAIN = "satnet_integrated_rf_export_schema"
TGNN_SCHEMA_IDENTITY_DOMAIN = "satnet_integrated_tgnn_adapter_schema"
SCHEMA_IDENTITY_VERSION = "1"


def _with_hash(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    result = deepcopy(payload)
    result[field_name] = canonical_hash(payload)
    return result


def build_target_schema() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "identity_domain": TARGET_SCHEMA_IDENTITY_DOMAIN,
        "identity_version": SCHEMA_IDENTITY_VERSION,
        "schema_name": "final_integrated_target_schema_v1",
        "schema_version": "1",
        "authoritative_source": {
            "class": "FailureAdjustedGroundServiceRunSummary",
            "module": "satnet.ground.failure_service_aggregation",
            "verification_stage": "G5",
        },
        "classification_direction": "1 means at least one threshold breach occurred",
        "targets": [
            {
                "field": "overall_threshold_breach_any",
                "role": "primary_classification",
                "value_type": "boolean",
            },
            {
                "field": "ground_threshold_breach_any",
                "role": "secondary_classification",
                "value_type": "boolean",
            },
            {
                "field": "space_threshold_breach_any",
                "role": "secondary_classification",
                "value_type": "boolean",
            },
            {
                "field": "failure_adjusted_overall_service_fraction_mean",
                "role": "primary_regression",
                "value_type": "canonical_binary64_fraction",
            },
            {
                "field": "failure_adjusted_overall_service_fraction_min",
                "role": "safety_secondary_regression",
                "value_type": "canonical_binary64_fraction",
            },
            {
                "field": "failure_adjusted_ground_service_fraction_min",
                "role": "secondary_regression",
                "value_type": "canonical_binary64_fraction",
            },
            {
                "field": "space_gcc_fraction_original_min",
                "role": "secondary_regression",
                "value_type": "canonical_binary64_fraction",
            },
            {
                "field": "ground_service_loss_due_to_failures_max",
                "role": "diagnostic_only",
                "value_type": "canonical_binary64_fraction",
            },
        ],
        "primary_regression_rationale": (
            "The integrated pilot found greater unique-value and within-design stochastic "
            "variation for the mean than for the minimum."
        ),
    }
    return _with_hash(payload, "target_schema_hash")


def build_rf_schema() -> dict[str, Any]:
    predictors = [
        ("num_planes", "integer", "direct"),
        ("sats_per_plane", "integer", "direct"),
        ("configured_satellite_count", "integer", "derived_redundant"),
        ("altitude_km", "canonical_binary64", "direct"),
        ("inclination_deg", "canonical_binary64", "direct"),
        ("satellite_node_failure_probability", "canonical_binary64_fraction", "direct"),
        ("satellite_edge_failure_probability", "canonical_binary64_fraction", "direct"),
        ("civilian_count", "integer", "direct"),
        ("government_count", "integer", "direct"),
        ("military_count", "integer", "direct"),
        ("total_ground_station_count", "integer", "derived_redundant"),
        ("ground_station_failure_probability", "canonical_binary64_fraction", "direct"),
    ]
    payload: dict[str, Any] = {
        "identity_domain": RF_SCHEMA_IDENTITY_DOMAIN,
        "identity_version": SCHEMA_IDENTITY_VERSION,
        "schema_name": "integrated_rf_export_schema_v1",
        "schema_version": "1",
        "implementation_status": "future_export_schema_not_current_registry",
        "framing": "ex_ante_design_time_prediction",
        "row_granularity": "one_row_per_run",
        "predictors": [
            {"field": field, "value_type": value_type, "derivation": derivation}
            for field, value_type, derivation in predictors
        ],
        "constant_provenance": [
            "minimum_elevation_deg",
            "space_gcc_threshold",
            "ground_service_threshold",
            "duration_minutes",
            "step_seconds",
            "epoch_iso",
            "orbital_engine",
            "failure_model",
            "isl_policy",
            "physics_model_version",
            "link_budget_config",
            "g1_to_g5_versions",
        ],
        "identity_provenance_not_predictors": [
            "run_id",
            "design_id",
            "design_index",
            "design_group_id",
            "realization_id",
            "doe_stratum",
            "contract_spec_hash",
            "design_record_hash",
            "ground_selection_seed",
            "selected_station_ids",
            "ground_selection_hash",
            "ground_design_hash",
            "ground_failure_policy_hash",
        ],
        "forbidden_predictor_categories": [
            "all_random_seeds",
            "selected_station_identities",
            "all_hashes",
            "realized_failure_sets_or_counts",
            "graph_connectivity_metrics",
            "service_metrics",
            "threshold_outcomes",
            "target_fields",
            "generation_or_replay_status",
            "runtime_or_artifact_size",
        ],
        "redundancy_note": (
            "Configured satellite count and total ground-station count are deterministic "
            "functions of component predictors and can dilute impurity-based importance."
        ),
        "current_model_changes_authorized": False,
    }
    return _with_hash(payload, "rf_schema_hash")


def build_tgnn_schema() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "identity_domain": TGNN_SCHEMA_IDENTITY_DOMAIN,
        "identity_version": SCHEMA_IDENTITY_VERSION,
        "schema_name": "integrated_tgnn_adapter_schema_v1",
        "schema_version": "1",
        "implementation_status": "future_adapter_not_implemented",
        "compatibility": {
            "existing_satnet_temporal_dataset": False,
            "current_satellite_gnn": False,
            "current_general_edge_attr_consumption": False,
        },
        "framing": "ex_post_full_sequence_assessment",
        "canonical_sources": [
            "G3 IntegratedGroundGraphRecord sequence",
            "G5 GroundFailureRealizationRecord persistent overlay",
        ],
        "failure_semantics": {
            "satellite": "failed satellites are absent under inherited G3 semantics",
            "ground_station": "selected nodes are retained with a G5 operational indicator",
        },
        "node_identity": {
            "satellite": "IntegratedNodeRef(kind=satellite, satellite_id)",
            "ground_station": "IntegratedNodeRef(kind=ground_station, ground_station_id)",
            "ordering": "canonical G3 node order",
        },
        "node_inputs": {
            "shared": ["node_type_one_hot", "operational_indicator"],
            "satellite": ["plane_index", "satellite_within_plane_index"],
            "ground_station": [
                "station_class_one_hot",
                "latitude_deg",
                "longitude_deg",
                "altitude_m",
                "derived_wgs84_ecef_xyz_km",
            ],
            "satellite_ecef": "omitted_in_v1",
            "fill_rule": "features not applicable to a node type use canonical positive zero",
        },
        "ground_ecef_derivation": {
            "function": "satnet.ground.coordinates.ground_station_to_ecef",
            "source_attributes": ["latitude_deg", "longitude_deg", "altitude_m"],
            "model_version": "wgs84_geodetic_ecef_v1",
        },
        "edge_inputs": {
            "shared": ["edge_type_one_hot"],
            "inter_satellite": [
                "distance_km",
                "margin_db",
                "link_type",
                "link_mode",
            ],
            "satellite_ground": ["elevation_deg", "slant_range_km"],
            "source": "canonical G3 edge attributes only",
        },
        "graph_static_inputs": [
            "design-time satellite architecture",
            "design-time ground class counts",
            "design-time failure probabilities",
            "fixed policy provenance",
        ],
        "deferred_model_contract": [
            "heterogeneous_message_passing",
            "graph_static_encoder_placement",
            "edge_feature_consumption",
            "pooling_changes",
            "prediction_head_changes",
        ],
        "prohibited_inputs": [
            "serviced_station_labels",
            "component_or_gcc_labels",
            "threshold_outcomes",
            "run_level_target_values",
            "post_outcome_summary_metrics",
        ],
        "current_model_changes_authorized": False,
    }
    return _with_hash(payload, "tgnn_adapter_schema_hash")
