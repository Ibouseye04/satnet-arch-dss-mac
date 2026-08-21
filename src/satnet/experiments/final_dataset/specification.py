from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from satnet.experiments.final_dataset.schemas import (
    build_rf_schema,
    build_target_schema,
    build_tgnn_schema,
)
from satnet.ground.canonical import canonical_float_string, canonical_hash, canonical_json
from satnet.ground.coordinates import (
    GROUND_VISIBILITY_FRAME_CONTRACT_VERSION,
    GROUND_VISIBILITY_MODEL_VERSION,
    GROUND_WGS84_MODEL_VERSION,
)
from satnet.ground.failure_policy import (
    GROUND_FAILURE_MODEL_VERSION,
    GROUND_FAILURE_POLICY_VERSION,
    GROUND_FAILURE_SAMPLING_VERSION,
)
from satnet.ground.failure_realization import GROUND_FAILURE_REALIZATION_SCHEMA_VERSION
from satnet.ground.failure_service_aggregation import GROUND_FAILURE_SERVICE_RUN_SCHEMA_VERSION
from satnet.ground.failure_service_metrics import (
    GROUND_FAILURE_SERVICE_MODEL_VERSION,
    GROUND_FAILURE_SERVICE_STEP_SCHEMA_VERSION,
)
from satnet.ground.integrated_graph import (
    INTEGRATED_GRAPH_MODEL_VERSION,
    INTEGRATED_GRAPH_SCHEMA_VERSION,
)
from satnet.ground.persistence import GROUND_DESIGN_SCHEMA_VERSION
from satnet.ground.selection import GROUND_STATION_SELECTION_VERSION
from satnet.ground.service_aggregation import (
    GROUND_SERVICE_RUN_SCHEMA_VERSION,
    GROUND_SERVICE_STEP_SCHEMA_VERSION,
)
from satnet.ground.service_policy import (
    GROUND_SERVICE_MODEL_VERSION,
    GROUND_SERVICE_POLICY_VERSION,
    GroundServicePolicy,
)
from satnet.experiments.production_profile import (
    FINAL_ADAPTIVE_PRODUCTION_PROFILE,
    HISTORICAL_FIXED_PROFILE,
    ProductionTopologyProfile,
)
from satnet.ground.visibility import GroundVisibilityPolicy
from satnet.ground.visibility_persistence import GROUND_VISIBILITY_SCHEMA_VERSION
from satnet.network.hypatia_adapter import LinkBudgetEngine, PHYSICS_MODEL_VERSION
from satnet.simulation.tier1_rollout import (
    DATASET_VERSION,
    DEFAULT_EPOCH_ISO,
    DEFAULT_FAILURE_MODEL,
    SCHEMA_VERSION,
)

CONTRACT_IDENTITY_DOMAIN = "satnet_final_integrated_dataset_contract_specification"
CONTRACT_IDENTITY_VERSION = "1"
CONTRACT_VERSION = "3"
CONTRACT_MASTER_SEED = 20260719
SPLIT_MASTER_SEED = 20260720
SEED_MODULUS = 2**63
CATALOG_HASH = "810c64dfb030b042311c90f2f42f8dee866a48fc63a6a29e362ee328c52eaa6e"
PILOT_DESIGN_MANIFEST_HASH = "514795b9cc17deacd81ffb9308e0d944f8da64b80444047647e919b375ef27bf"
PILOT_REPORT_SHA = "e4475f9bc22a83b30cdc6862d3337a1e2b6dbc3f"
PROTECTED_SCIENCE_BASE_SHA = "62beda9df1576958d9e33d33d2d9eb5489e24b20"
CANONICAL_DIMENSION_ORDER = (
    "altitude_km",
    "inclination_deg",
    "satellite_node_failure_probability",
    "satellite_edge_failure_probability",
    "ground_station_failure_probability",
)
QUALIFICATION_RUN_IDS = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 35, 60,
    3039, 3539, 3735, 4115, 5564, 6099, 6225, 6874, 7195, 7365,
    7369, 8624, 8645, 8980, 9174, 9514, 9999,
)


def _c(value: float) -> str:
    return canonical_float_string(value)


def _canonical_float_mapping(source: dict[str, float]) -> dict[str, str]:
    return {name: _c(source[name]) for name in sorted(source)}


def _schema_hash(schema: dict[str, Any], field_name: str) -> str:
    value = schema.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"{field_name} is missing")
    payload = {key: item for key, item in schema.items() if key != field_name}
    if canonical_hash(payload) != value:
        raise ValueError(f"{field_name} does not match schema payload")
    return value


def build_later_generation_acceptance_gates() -> dict[str, Any]:
    return {
        "expected_run_count": 10000,
        "required_generation_attempt_count": 10000,
        "required_successful_generation_count": 10000,
        "required_authoritative_replay_count": 10000,
        "required_successful_replay_count": 10000,
        "allow_seed_substitution": False,
        "allow_run_omission": False,
        "allow_replacement_runs": False,
        "require_failure_evidence_preservation": True,
        "require_all_numeric_targets_finite": True,
        "require_all_target_fractions_in_unit_interval": True,
        "require_primary_classification_both_classes_per_split": True,
        "require_primary_regression_nonzero_standard_deviation_per_split": True,
        "require_primary_regression_minimum_unique_values_per_split": 5,
        "allow_outcome_driven_split_reshuffle": False,
        "require_frozen_split_manifest": True,
        "require_all_five_realizations_colocated_by_design": True,
        "require_zero_missing_stage_artifacts": True,
        "require_zero_duplicate_run_ids": True,
        "require_zero_duplicate_design_realization_pairs": True,
        "require_exact_g1_g5_replay": True,
        "require_protected_science_diff_empty": True,
        "count_semantics": {
            "expected_run_count": "records_in_frozen_run_manifest",
            "required_generation_attempt_count": "frozen_runs_submitted_to_generation",
            "required_successful_generation_count": "frozen_runs_successfully_generated",
            "required_authoritative_replay_count": "frozen_runs_submitted_to_authoritative_replay",
            "required_successful_replay_count": "frozen_runs_successfully_replayed",
        },
        "failure_behavior": {
            "preserve_failed_run_evidence": True,
            "keep_frozen_run_manifest_unchanged": True,
            "dataset_status_on_any_failure": "incomplete",
            "retry_requires_same_run_id_and_frozen_seeds": True,
        },
        "numeric_target_validity": {
            "reject_nan": True,
            "reject_positive_infinity": True,
            "reject_negative_infinity": True,
            "fraction_interval": ["0", "1"],
            "fraction_interval_closed": True,
        },
        "primary_classification_gate": {
            "target_field": "overall_threshold_breach_any",
            "required_splits": ["train", "validation", "test"],
            "required_values_per_split": [False, True],
            "failure_action": "dataset_acceptance_failed_pending_explicit_contract_version_review",
            "allow_threshold_tuning": False,
        },
        "primary_regression_gate": {
            "target_field": "failure_adjusted_overall_service_fraction_mean",
            "required_splits": ["train", "validation", "test"],
            "standard_deviation_semantics": "population_standard_deviation_over_finite_binary64_values",
            "require_nonzero_standard_deviation": True,
            "unique_value_semantics": "exact_binary64_equality_after_finite_parse",
            "minimum_unique_values_per_split": 5,
        },
        "split_immutability": {
            "selected_candidate_id": 3958,
            "outcomes_or_labels_may_modify_assignments": False,
            "all_realizations_grouped_by_design": True,
        },
        "required_stage_artifacts": [
            "satellite_rollout",
            "G1",
            "G2",
            "G3",
            "G4",
            "G5",
        ],
    }


def build_contract_specification(
    *,
    profile: ProductionTopologyProfile = HISTORICAL_FIXED_PROFILE,
    pilot_design_manifest_hash_value: str = PILOT_DESIGN_MANIFEST_HASH,
) -> dict[str, Any]:
    profile.validate()
    if profile is not HISTORICAL_FIXED_PROFILE and pilot_design_manifest_hash_value == PILOT_DESIGN_MANIFEST_HASH:
        from satnet.experiments.integrated_ground_manifest import (
            build_adaptive_pilot_designs,
            pilot_design_manifest_hash,
        )

        pilot_design_manifest_hash_value = pilot_design_manifest_hash(
            build_adaptive_pilot_designs()
        )
    target_schema = build_target_schema()
    rf_schema = build_rf_schema()
    tgnn_schema = build_tgnn_schema()
    visibility_policy = GroundVisibilityPolicy(10.0)
    service_policy = GroundServicePolicy(0.8, 0.8)
    payload: dict[str, Any] = {
        "identity_domain": CONTRACT_IDENTITY_DOMAIN,
        "identity_version": CONTRACT_IDENTITY_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_scope": {
            "simulation_generation_authorized": False,
            "g1_to_g5_scientific_changes_authorized": False,
            "rf_model_changes_authorized": False,
            "tgnn_model_changes_authorized": False,
            "release_tag_authorized": False,
        },
        "validated_foundation": {
            "pilot_report_sha": PILOT_REPORT_SHA,
            "protected_science_base_sha": PROTECTED_SCIENCE_BASE_SHA,
            "pilot_design_manifest_hash": pilot_design_manifest_hash_value,
            "catalog_hash": CATALOG_HASH,
        },
        "seeds": {
            "contract_master_seed": CONTRACT_MASTER_SEED,
            "split_master_seed": SPLIT_MASTER_SEED,
            "seed_modulus": SEED_MODULUS,
            "derivation": "unsigned_big_endian_sha256_of_canonical_json_modulo_seed_modulus",
            "production_purposes": {
                "ground_selection": "ground_station_selection",
                "satellite": "satellite_rollout_and_failure",
                "ground_failure": "ground_failure_realization",
            },
            "domains": {
                "ground_selection": "satnet_final_dataset_ground_selection_seed",
                "satellite": "satnet_final_dataset_satellite_seed",
                "ground_failure": "satnet_final_dataset_ground_failure_seed",
                "lhs_permutation": "satnet_final_dataset_lhs_permutation",
                "lhs_jitter": "satnet_final_dataset_lhs_jitter",
                "schedule_pairing": "satnet_final_dataset_schedule_pairing",
                "split_candidate": "satnet_final_dataset_split_candidate",
            },
            "identity_version": "1",
        },
        "dataset_cardinality": {
            "design_count": 2000,
            "realizations_per_design": 5,
            "run_count": 10000,
            "design_ids": "D0000_through_D1999",
            "realization_ids": "R00_through_R04",
            "qualification_run_ids": list(QUALIFICATION_RUN_IDS),
        },
        "run_identity": {
            "schema_version": "2",
            "authoritative_field": "run_id",
            "run_id_value_type": "exact_integer_not_boolean",
            "run_id_first": 0,
            "run_id_last": 9999,
            "run_id_formula": "design_index * realizations_per_design + realization_index",
            "run_key_value_type": "string",
            "run_key_formula": "design_id + '-' + realization_id",
            "design_index_first": 0,
            "design_index_last": 99,
            "realization_index_first": 0,
            "realization_index_last": 4,
            "run_index_present": False,
            "authoritative_execution_join_field": "run_id",
            "run_key_role": "human_readable_composite_identity",
            "scientific_identity_fields": [
                "contract_spec_hash",
                "design_record_hash",
                "run_id",
                "run_key",
                "design_id",
                "realization_id",
                "design_index",
                "realization_index",
                "satellite_seed",
                "ground_failure_seed",
                "ground_selection_seed",
                "split_assignment",
            ],
        },
        "doe": {
            "strata": [
                {
                    "stratum_id": "pilot_anchor",
                    "design_index_first": 0,
                    "design_index_last": 4,
                    "design_count": 5,
                    "source_design_ids": ["P01", "P02", "P03", "P04", "P05"],
                },
                {
                    "stratum_id": "transition",
                    "design_index_first": 5,
                    "design_index_last": 399,
                    "design_count": 395,
                },
                {
                    "stratum_id": "global",
                    "design_index_first": 400,
                    "design_index_last": 1999,
                    "design_count": 1600,
                },
            ],
            "transition": {
                "continuous_ranges": {
                    "altitude_km": [_c(600.0), _c(800.0)],
                    "inclination_deg": [_c(55.0), _c(60.0)],
                    "satellite_node_failure_probability": [_c(0.05), _c(0.10)],
                    "satellite_edge_failure_probability": [_c(0.05), _c(0.10)],
                    "ground_station_failure_probability": [_c(0.05), _c(0.15)],
                },
                "satellite_pair_frequencies": [
                    {"num_planes": 5, "sats_per_plane": 6, "count": 66},
                    {"num_planes": 5, "sats_per_plane": 7, "count": 66},
                    {"num_planes": 5, "sats_per_plane": 8, "count": 66},
                    {"num_planes": 6, "sats_per_plane": 6, "count": 66},
                    {"num_planes": 6, "sats_per_plane": 7, "count": 66},
                    {"num_planes": 6, "sats_per_plane": 8, "count": 65},
                ],
                "station_totals": [10, 12, 15, 18, 20],
                "composition_weights": [
                    [1, 1, 1],
                    [3, 1, 1],
                    [1, 3, 1],
                    [1, 1, 3],
                    [2, 2, 1],
                    [2, 1, 2],
                    [1, 2, 2],
                ],
                "ground_schedule": "balanced_repetition_of_exact_cartesian_product",
                "ground_cell_count": 35,
                "ground_cell_repetition_counts": {"base_cells": 11, "remainder_cells": 10},
                "satellite_pair_schedule": "balanced_largest_remainder",
                "satellite_pair_remainder_order": "listed_order",
            },
            "global": {
                "continuous_ranges": {
                    "altitude_km": [_c(300.0), _c(1200.0)],
                    "inclination_deg": [_c(30.0), _c(98.0)],
                    "satellite_node_failure_probability": [_c(0.0), _c(0.20)],
                    "satellite_edge_failure_probability": [_c(0.0), _c(0.25)],
                    "ground_station_failure_probability": [_c(0.0), _c(0.40)],
                },
                "satellite_values": {"num_planes": [4, 5, 6], "sats_per_plane": [5, 6, 7, 8]},
                "satellite_pair_frequency": 133,
                "satellite_pair_remainder": 4,
                "satellite_pair_remainder_order": "product_order",
                "station_total_frequencies": [
                    {"total": total, "count": 160}
                    for total in [6, 10, 15, 20, 25, 30, 35, 40, 45, 50]
                ],
                "composition_weight_frequencies": [
                    {"weights": weights, "count": 160}
                    for weights in [
                        [1, 1, 1],
                        [3, 1, 1],
                        [1, 3, 1],
                        [1, 1, 3],
                        [9, 9, 2],
                        [9, 2, 9],
                        [2, 9, 9],
                        [8, 1, 1],
                        [1, 8, 1],
                        [1, 1, 8],
                    ]
                ],
            },
            "integer_allocation": {
                "minimum_per_class": 1,
                "method": "minimum_first_exact_rational_largest_remainder",
                "tie_order": ["civilian", "government", "military"],
                "catalog_capacity_per_class": 50,
            },
        },
        "lhs": {
            "implementation": "repository_local",
            "candidate_count": 256,
            "scoring_implementation": "numpy_vectorized_pairwise_distances_exact_selection_order",
            "dimension_order": list(CANONICAL_DIMENSION_ORDER),
            "permutation_order": ["digest_bytes_ascending", "row_index_ascending"],
            "rank_semantics": "k[d,r] is the zero-based rank of row r in dimension d",
            "jitter_bits": "53_most_significant_sha256_bits_big_endian",
            "unit_value": "(k[d,r] + jitter[r,d]) / row_count",
            "pair_order": "i_ascending_then_j_ascending_for_i_less_than_j",
            "distance": "sqrt(fsum((x_i_d-x_j_d)**2 in canonical dimension order))",
            "selection_order": [
                "largest_minimum_pairwise_distance",
                "largest_mean_pairwise_distance",
                "smallest_candidate_id",
            ],
            "external_doe_library": False,
        },
        "schedule_pairing": {
            "purposes": ["continuous_rows", "ground_schedule", "satellite_pair_schedule"],
            "ordering": ["digest_bytes_ascending", "record_index_ascending"],
            "pairing": "independently_permuted_records_paired_by_position",
        },
        "fixed_profile": {
            "duration_minutes": profile.duration_minutes,
            "step_seconds": profile.step_seconds,
            "inclusive_timestep_count": profile.inclusive_timestep_count,
            "phasing_factor": profile.phasing_factor,
            "max_isl_distance_km": _c(profile.max_isl_distance_km),
            "isl_policy": profile.isl_policy,
            "adjacent_search_k": profile.adjacent_search_k,
            "max_inter_plane_links_per_sat": profile.max_inter_plane_links_per_sat,
            "orbital_engine": profile.orbital_engine,
            "epoch_iso": profile.epoch_iso,
            "failure_model": profile.failure_model,
            "minimum_elevation_deg": _c(10.0),
            "space_gcc_threshold": _c(0.8),
            "ground_service_threshold": _c(0.8),
            "visibility_policy_hash": visibility_policy.visibility_policy_hash,
            "ground_service_policy_hash": service_policy.ground_service_policy_hash,
            "physics_model_version": PHYSICS_MODEL_VERSION,
            "link_budget_config": _canonical_float_mapping(LinkBudgetEngine().to_config()),
            "versions": {
                "satellite_rollout_schema_version": SCHEMA_VERSION,
                "satellite_dataset_version": DATASET_VERSION,
                "ground_station_selection_version": GROUND_STATION_SELECTION_VERSION,
                "ground_design_schema_version": GROUND_DESIGN_SCHEMA_VERSION,
                "ground_visibility_model_version": GROUND_VISIBILITY_MODEL_VERSION,
                "ground_visibility_frame_contract_version": GROUND_VISIBILITY_FRAME_CONTRACT_VERSION,
                "ground_wgs84_model_version": GROUND_WGS84_MODEL_VERSION,
                "ground_visibility_schema_version": GROUND_VISIBILITY_SCHEMA_VERSION,
                "integrated_graph_model_version": INTEGRATED_GRAPH_MODEL_VERSION,
                "integrated_graph_schema_version": INTEGRATED_GRAPH_SCHEMA_VERSION,
                "ground_service_model_version": GROUND_SERVICE_MODEL_VERSION,
                "ground_service_policy_version": GROUND_SERVICE_POLICY_VERSION,
                "ground_service_step_schema_version": GROUND_SERVICE_STEP_SCHEMA_VERSION,
                "ground_service_run_schema_version": GROUND_SERVICE_RUN_SCHEMA_VERSION,
                "ground_failure_model_version": GROUND_FAILURE_MODEL_VERSION,
                "ground_failure_policy_version": GROUND_FAILURE_POLICY_VERSION,
                "ground_failure_sampling_version": GROUND_FAILURE_SAMPLING_VERSION,
                "ground_failure_realization_schema_version": GROUND_FAILURE_REALIZATION_SCHEMA_VERSION,
                "ground_failure_service_model_version": GROUND_FAILURE_SERVICE_MODEL_VERSION,
                "ground_failure_service_step_schema_version": GROUND_FAILURE_SERVICE_STEP_SCHEMA_VERSION,
                "ground_failure_service_run_schema_version": GROUND_FAILURE_SERVICE_RUN_SCHEMA_VERSION,
            },
        },
        "ground_architecture": {
            "selection_level": "design",
            "constant_across_realizations": [
                "ground_selection_seed",
                "selected_station_ids",
                "ground_selection_hash",
                "ground_design_hash",
            ],
            "g1_selected_id_order": "civilian_then_government_then_military_authoritative_order",
        },
        "schemas": {
            "target_schema_hash": _schema_hash(target_schema, "target_schema_hash"),
            "rf_schema_hash": _schema_hash(rf_schema, "rf_schema_hash"),
            "tgnn_adapter_schema_hash": _schema_hash(
                tgnn_schema, "tgnn_adapter_schema_hash"
            ),
        },
        "split": {
            "strategy": "pre_outcome_grouped_marginal_balance",
            "candidate_count": 4096,
            "design_counts": {"train": 1400, "validation": 300, "test": 300},
            "run_counts": {"train": 7000, "validation": 1500, "test": 1500},
            "candidate_order": ["digest_bytes_ascending", "design_id_ascending"],
            "marginals": [
                "num_planes",
                "sats_per_plane",
                "doe_stratum",
                "total_ground_station_count",
                "reduced_composition_ratio",
                "altitude_bin",
                "inclination_bin",
                "node_failure_bin",
                "edge_failure_bin",
                "ground_failure_bin",
            ],
            "bins": {
                "altitude_km": [[_c(300.0), _c(525.0), False], [_c(525.0), _c(750.0), False], [_c(750.0), _c(975.0), False], [_c(975.0), _c(1200.0), True]],
                "inclination_deg": [[_c(30.0), _c(47.0), False], [_c(47.0), _c(64.0), False], [_c(64.0), _c(81.0), False], [_c(81.0), _c(98.0), True]],
                "satellite_node_failure_probability": [[_c(0.0), _c(0.05), False], [_c(0.05), _c(0.10), False], [_c(0.10), _c(0.15), False], [_c(0.15), _c(0.20), True]],
                "satellite_edge_failure_probability": [[_c(0.0), _c(0.0625), False], [_c(0.0625), _c(0.125), False], [_c(0.125), _c(0.1875), False], [_c(0.1875), _c(0.25), True]],
                "ground_station_failure_probability": [[_c(0.0), _c(0.10), False], [_c(0.10), _c(0.20), False], [_c(0.20), _c(0.30), False], [_c(0.30), _c(0.40), True]],
            },
            "hard_requirements": [
                "every split contains all plane categories",
                "every split contains all satellites-per-plane categories",
                "every split contains all DOE strata",
                "train contains at least three pilot anchors",
                "validation contains at least one pilot anchor",
                "test contains at least one pilot anchor",
            ],
            "score_order": [
                "smallest_exact_fraction_maximum_normalized_deviation",
                "smallest_exact_fraction_sum_squared_normalized_deviation",
                "smallest_exact_fraction_total_absolute_deviation",
                "smallest_candidate_id",
            ],
            "failure_behavior": "fail_without_relaxation_or_extra_candidates",
            "outcome_fields_used": False,
        },
        "targets": {
            "primary_classification": "overall_threshold_breach_any",
            "secondary_classification": [
                "ground_threshold_breach_any",
                "space_threshold_breach_any",
            ],
            "primary_regression": "failure_adjusted_overall_service_fraction_mean",
            "safety_secondary_regression": "failure_adjusted_overall_service_fraction_min",
            "secondary_regression": [
                "failure_adjusted_ground_service_fraction_min",
                "space_gcc_fraction_original_min",
            ],
            "diagnostic_only": "ground_service_loss_due_to_failures_max",
        },
        "identity_graph": {
            "schema_hashes": "independent_self_hash_field_omitted",
            "contract_spec_hash": "hash_of_this_payload_with_contract_spec_hash_omitted",
            "design_manifest_hash": "ordered_design_records_bound_to_contract_spec_hash",
            "run_manifest_hash": "ordered_run_records_bound_to_contract_spec_and_design_hash",
            "split_manifest_hash": "exact_assignments_bound_to_contract_spec_and_design_manifest",
            "contract_bundle_hash": "binds_spec_schema_catalog_design_run_and_split_hashes",
            "generated_records_reference_contract_bundle_hash": False,
        },
        "contract_phase_validation_gates": {
            "manifest_counts": {"designs": 2000, "runs": 10000},
            "split_design_counts": {"train": 1400, "validation": 300, "test": 300},
            "split_run_counts": {"train": 7000, "validation": 1500, "test": 1500},
            "all_realizations_colocated": True,
            "canonical_float_strings": True,
            "outcome_free_design_and_run_manifests": True,
            "protected_diff_empty": True,
            "focused_tests_pass": True,
            "complete_tests_pass": True,
            "simulation_artifacts_required": False,
        },
        "later_generation_acceptance_gates": build_later_generation_acceptance_gates(),
    }
    if profile is not HISTORICAL_FIXED_PROFILE:
        payload["production_profile"] = profile.profile_id
    result = deepcopy(payload)
    result["contract_spec_hash"] = canonical_hash(payload)
    return result


def build_adaptive_contract_specification() -> dict[str, Any]:
    """Build the corrected, non-historical adaptive contract identity."""

    from satnet.experiments.integrated_ground_manifest import (
        build_adaptive_pilot_designs,
        pilot_design_manifest_hash,
    )

    adaptive_pilot_hash = pilot_design_manifest_hash(build_adaptive_pilot_designs())
    specification = build_contract_specification(
        profile=FINAL_ADAPTIVE_PRODUCTION_PROFILE,
        pilot_design_manifest_hash_value=adaptive_pilot_hash,
    )
    validate_adaptive_contract_specification(specification)
    return specification


def validate_adaptive_contract_specification(specification: dict[str, Any]) -> None:
    validate_contract_specification(specification)
    if specification.get("production_profile") != FINAL_ADAPTIVE_PRODUCTION_PROFILE.profile_id:
        raise ValueError("Specification is not the final adaptive production lineage")
    FINAL_ADAPTIVE_PRODUCTION_PROFILE.assert_matches(specification["fixed_profile"])
    if specification["contract_spec_hash"] == build_contract_specification()["contract_spec_hash"]:
        raise ValueError("Adaptive specification reuses the historical fixed-policy hash")


def validate_contract_specification(specification: dict[str, Any]) -> None:
    if not isinstance(specification, dict):
        raise TypeError("specification must be a dictionary")
    value = specification.get("contract_spec_hash")
    if not isinstance(value, str):
        raise ValueError("contract_spec_hash is missing")
    payload = {key: item for key, item in specification.items() if key != "contract_spec_hash"}
    if canonical_hash(payload) != value:
        raise ValueError("contract_spec_hash does not match specification payload")
    if specification.get(
        "later_generation_acceptance_gates"
    ) != build_later_generation_acceptance_gates():
        raise ValueError("Later-generation acceptance gates differ from the frozen contract")


def materialize_adaptive_machine_specification(
    output_root: str | Path, *, overwrite: bool = False
) -> dict[str, str]:
    """Materialize only the corrected adaptive machine contract preflight."""

    root = Path(output_root)
    schemas = {
        "target_schema.json": build_target_schema(),
        "integrated_rf_export_schema.json": build_rf_schema(),
        "integrated_tgnn_adapter_schema.json": build_tgnn_schema(),
    }
    specification = build_adaptive_contract_specification()
    values = {**schemas, "contract_specification.json": specification}
    root.mkdir(parents=True, exist_ok=True)
    for filename, value in values.items():
        path = root / filename
        if path.exists() and not overwrite:
            raise FileExistsError(f"Contract artifact already exists: {path}")
        path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")
    return {
        "contract_spec_hash": specification["contract_spec_hash"],
        "target_schema_hash": schemas["target_schema.json"]["target_schema_hash"],
        "rf_schema_hash": schemas["integrated_rf_export_schema.json"]["rf_schema_hash"],
        "tgnn_adapter_schema_hash": schemas["integrated_tgnn_adapter_schema.json"]["tgnn_adapter_schema_hash"],
    }


def materialize_machine_specification(output_root: str | Path, *, overwrite: bool = False) -> dict[str, str]:
    root = Path(output_root)
    schemas = {
        "target_schema.json": build_target_schema(),
        "integrated_rf_export_schema.json": build_rf_schema(),
        "integrated_tgnn_adapter_schema.json": build_tgnn_schema(),
    }
    specification = build_contract_specification()
    validate_contract_specification(specification)
    values = {**schemas, "contract_specification.json": specification}
    root.mkdir(parents=True, exist_ok=True)
    for filename, value in values.items():
        path = root / filename
        if path.exists() and not overwrite:
            raise FileExistsError(f"Machine contract artifact already exists: {path}")
        path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")
    return {
        "target_schema_hash": schemas["target_schema.json"]["target_schema_hash"],
        "rf_schema_hash": schemas["integrated_rf_export_schema.json"]["rf_schema_hash"],
        "tgnn_adapter_schema_hash": schemas["integrated_tgnn_adapter_schema.json"]["tgnn_adapter_schema_hash"],
        "contract_spec_hash": specification["contract_spec_hash"],
    }
