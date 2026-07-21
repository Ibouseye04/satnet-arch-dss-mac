from __future__ import annotations

from decimal import Decimal

ANALYSIS_SCHEMA_VERSION = "satnet.final_integrated_dataset_class_support_analysis.v1"
AUGMENTATION_CONTRACT_VERSION = "satnet.final_integrated_dataset_augmentation_proposal.v1"
PROPOSAL_LABEL = "PROPOSAL ONLY — NOT FROZEN — DO NOT SIMULATE"
PRODUCTION_TOOLING_SHA = "9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab"
FROZEN_CONTRACT_TAG = "final-integrated-dataset-contract-v1"
FROZEN_CONTRACT_COMMIT = "a1967185e80327e4b00c1831828dc975ab6819fc"
CONTRACT_SPECIFICATION_HASH = "482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b"
GENERATION_LEDGER_SHA256 = "a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1"
REPLAY_LEDGER_SHA256 = "4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd"
FREEZE_ARCHIVE_SHA256 = "375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc"
GENERATION_FILE_COUNT = 8502
GENERATION_BYTE_COUNT = 1336139056
REPLAY_FILE_COUNT = 502
REPLAY_BYTE_COUNT = 1410137
RUN_COUNT = 500
DESIGN_COUNT = 100
REALIZATIONS_PER_DESIGN = 5
SERVICE_THRESHOLD = Decimal("0.80")
EXPECTED_CLASS_COUNTS = {
    "train": {False: 2, True: 348},
    "validation": {False: 0, True: 75},
    "test": {False: 5, True: 70},
}
EXPECTED_SPLIT_RUN_COUNTS = {"train": 350, "validation": 75, "test": 75}
EXPECTED_SPLIT_DESIGN_COUNTS = {"train": 70, "validation": 15, "test": 15}
EXPECTED_NON_BREACH_RUN_IDS = (0, 1, 2, 3, 4, 5, 7)
TARGET_FIELDS = (
    "overall_threshold_breach_any",
    "ground_threshold_breach_any",
    "space_threshold_breach_any",
    "failure_adjusted_overall_service_fraction_mean",
    "failure_adjusted_overall_service_fraction_min",
    "failure_adjusted_ground_service_fraction_min",
    "space_gcc_fraction_original_min",
    "ground_service_loss_due_to_failures_max",
)
REGRESSION_TARGETS = (
    "failure_adjusted_overall_service_fraction_mean",
    "failure_adjusted_overall_service_fraction_min",
    "failure_adjusted_ground_service_fraction_min",
    "space_gcc_fraction_original_min",
)
NUMERIC_DESIGN_PARAMETERS = (
    "num_planes",
    "sats_per_plane",
    "configured_satellite_count",
    "altitude_km",
    "inclination_deg",
    "phasing_factor",
    "satellite_node_failure_probability",
    "satellite_edge_failure_probability",
    "civilian_count",
    "government_count",
    "military_count",
    "total_ground_station_count",
    "ground_station_failure_probability",
)
NEIGHBOR_FEATURES = (
    "num_planes",
    "sats_per_plane",
    "altitude_km",
    "inclination_deg",
    "satellite_node_failure_probability",
    "satellite_edge_failure_probability",
    "total_ground_station_count",
    "ground_station_failure_probability",
    "civilian_fraction",
    "government_fraction",
    "military_fraction",
)
DOE_RANGES = {
    "num_planes": (4.0, 6.0),
    "sats_per_plane": (5.0, 8.0),
    "altitude_km": (300.0, 1200.0),
    "inclination_deg": (30.0, 98.0),
    "satellite_node_failure_probability": (0.0, 0.2),
    "satellite_edge_failure_probability": (0.0, 0.25),
    "total_ground_station_count": (3.0, 50.0),
    "ground_station_failure_probability": (0.0, 0.4),
    "civilian_fraction": (0.0, 1.0),
    "government_fraction": (0.0, 1.0),
    "military_fraction": (0.0, 1.0),
}
BOUNDARY_BANDS = (
    ("less_than_-0.20", None, Decimal("-0.20"), False, False),
    ("[-0.20,-0.10)", Decimal("-0.20"), Decimal("-0.10"), True, False),
    ("[-0.10,-0.05)", Decimal("-0.10"), Decimal("-0.05"), True, False),
    ("[-0.05,-0.025)", Decimal("-0.05"), Decimal("-0.025"), True, False),
    ("[-0.025,-0.01)", Decimal("-0.025"), Decimal("-0.01"), True, False),
    ("[-0.01,0)", Decimal("-0.01"), Decimal("0"), True, False),
    ("[0,0.01]", Decimal("0"), Decimal("0.01"), True, True),
    ("(0.01,0.025]", Decimal("0.01"), Decimal("0.025"), False, True),
    ("(0.025,0.05]", Decimal("0.025"), Decimal("0.05"), False, True),
    ("(0.05,0.10]", Decimal("0.05"), Decimal("0.10"), False, True),
    ("greater_than_0.10", Decimal("0.10"), None, False, False),
)
OUTPUT_SCHEMAS = {
    "run_level_class_support.csv": "satnet.class_support.run.v1",
    "design_level_class_support.csv": "satnet.class_support.design.v1",
    "non_breach_runs.csv": "satnet.class_support.non_breach_runs.v1",
    "non_breach_designs.csv": "satnet.class_support.non_breach_designs.v1",
    "boundary_run_ranking.csv": "satnet.class_support.boundary_run_ranking.v1",
    "boundary_design_ranking.csv": "satnet.class_support.boundary_design_ranking.v1",
    "temporal_breach_summary.csv": "satnet.class_support.temporal.v1",
    "parameter_support_summary.csv": "satnet.class_support.parameter_support.v1",
    "nearest_neighbor_summary.csv": "satnet.class_support.nearest_neighbor.v1",
    "split_class_support_summary.json": "satnet.class_support.split_summary.v1",
    "regression_only_summary.json": "satnet.class_support.regression_summary.v1",
    "boundary_region_summary.json": "satnet.class_support.boundary_summary.v1",
    "augmentation_size_options.json": "satnet.class_support.augmentation_options.v1",
    "recommended_augmentation_design.csv": "satnet.class_support.augmentation_design_proposal.v1",
    "augmentation_contract_proposal.json": "satnet.class_support.augmentation_contract_proposal.v1",
}
PLOT_SCHEMAS = {
    "plots/classification_counts_by_split.svg": "satnet.class_support.plot.class_counts.v1",
    "plots/boundary_margin_histogram_by_split.svg": "satnet.class_support.plot.margin_histogram.v1",
    "plots/boundary_margin_empirical_distribution.svg": "satnet.class_support.plot.margin_ecdf.v1",
    "plots/design_non_breach_realization_counts.svg": "satnet.class_support.plot.design_counts.v1",
    "plots/parameter_vs_boundary_margin.svg": "satnet.class_support.plot.parameter_margin.v1",
    "plots/non_breach_near_boundary_profiles.svg": "satnet.class_support.plot.parameter_profiles.v1",
    "plots/nearest_neighbor_outcome_comparison.svg": "satnet.class_support.plot.neighbors.v1",
    "plots/regression_target_distributions_by_split.svg": "satnet.class_support.plot.regression.v1",
    "plots/proposed_augmentation_allocation.svg": "satnet.class_support.plot.augmentation_allocation.v1",
}
