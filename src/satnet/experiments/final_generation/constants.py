from __future__ import annotations

from pathlib import Path

FROZEN_COMMIT = "a8fbfed18b1673f5fc9c6a291ccc02905f1392d6"
FROZEN_TAG = "final-integrated-dataset-10k-contract-v1"
CONTRACT_SPEC_HASH = "6c7dd365f9e7fb67f5f5e70879a19535ede55468aabfac53d82c2ab35b8307eb"
CONTRACT_BUNDLE_HASH = "059dff74930d1125a46947a06d213dd07c3a93dce226a894558a01805c3ed94c"
DESIGN_MANIFEST_HASH = "d39830c861ae3e7c5222dd05c44ecef6fb66b365a6d7a42e26f335c8820600e0"
RUN_MANIFEST_HASH = "e965d5daea19a958fe6ce20d5c4a75240a42497b6fde33a508d4df0df64888ea"
SPLIT_MANIFEST_HASH = "07a84c324b255f21f2697b88150c2cc3171156405f0750e2db1312ff8204c4aa"
TARGET_SCHEMA_HASH = "9088948d6b03db59877a129179ac3091c3690aeab9d2849a921bfa90f05da528"
RF_SCHEMA_HASH = "56ffd04aa7dc7b98e34ec63c6e0422d2eb772271e9710b981f986a323e7ea384"
TGNN_SCHEMA_HASH = "bb1d5454904266b83a3c7457098df00d46733994e9f673e4a40b94f31c893cfa"
CATALOG_FILE_SHA256 = "e8855d4ded4c242f3e5b35b90610d5f9c4f218bdf717d2d08b2464cac39f1598"
CATALOG_HASH = "810c64dfb030b042311c90f2f42f8dee866a48fc63a6a29e362ee328c52eaa6e"

FROZEN_ARTIFACTS = (
    "contract_specification.json",
    "target_schema.json",
    "integrated_rf_export_schema.json",
    "integrated_tgnn_adapter_schema.json",
    "designs.jsonl",
    "runs.jsonl",
    "split_manifest.json",
    "contract_bundle.json",
    "manifest_inventory.json",
    "golden_vectors.json",
    "doe_evidence.json",
)
SUPPORTED_MODES = frozenset(
    {
        "production",
        "production_replay",
        "qualification",
        "qualification_repeat",
        "qualification_replay",
    }
)
QUALIFICATION_RUN_IDS = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 35, 60,
    3039, 3539, 3735, 4115, 5564, 6099, 6225, 6874, 7195, 7365,
    7369, 8624, 8645, 8980, 9174, 9514, 9999,
)
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
TARGET_BOOLEAN_FIELDS = frozenset(TARGET_FIELDS[:3])
TARGET_FLOAT_FIELDS = frozenset(TARGET_FIELDS[3:])

SATELLITE_ARTIFACT_SCHEMA_VERSION = "1"
SATELLITE_ARTIFACT_DOMAIN = "satnet_final_integrated_dataset_satellite_artifact"
SATELLITE_ARTIFACT_IDENTITY_VERSION = "1"
TARGET_ARTIFACT_SCHEMA_VERSION = "1"
TARGET_ARTIFACT_DOMAIN = "satnet_final_integrated_dataset_target_artifact"
TARGET_ARTIFACT_IDENTITY_VERSION = "1"
SCIENTIFIC_INVENTORY_SCHEMA_VERSION = "1"
SCIENTIFIC_INVENTORY_DOMAIN = "satnet_final_integrated_dataset_scientific_inventory"
SCIENTIFIC_INVENTORY_IDENTITY_VERSION = "1"
RUN_RESULT_SCHEMA_VERSION = "1"
RUN_RESULT_DOMAIN = "satnet_final_integrated_dataset_run_result"
RUN_RESULT_IDENTITY_VERSION = "1"
MODE_MARKER_SCHEMA_VERSION = "1"

RUN_ID_WIDTH = 4
FINAL_DESIGN_COUNT = 2000
REALIZATIONS_PER_DESIGN = 5
FINAL_RUN_COUNT = FINAL_DESIGN_COUNT * REALIZATIONS_PER_DESIGN

RUN_FILES = {
    "satellite": "satellite/satellite_rollout.json",
    "g1": "g1/ground_design.jsonl",
    "g2": "g2/ground_visibility.jsonl",
    "g3": "g3/integrated_graphs.jsonl",
    "g4_steps": "g4/ground_service_steps.jsonl",
    "g4_run": "g4/ground_service_run.jsonl",
    "g5_realization": "g5/ground_failure_realization.jsonl",
    "g5_steps": "g5/ground_failure_service_steps.jsonl",
    "g5_run": "g5/ground_failure_service_run.jsonl",
    "target": "targets/target.json",
    "inventory": "scientific_inventory.json",
    "result": "result.json",
}
SCIENTIFIC_FILE_KEYS = (
    "satellite",
    "g1",
    "g2",
    "g3",
    "g4_steps",
    "g4_run",
    "g5_realization",
    "g5_steps",
    "g5_run",
    "target",
)
ARTIFACT_ROLES = {
    "satellite": "satellite_rollout",
    "g1": "ground_selection",
    "g2": "ground_visibility",
    "g3": "integrated_graph",
    "g4_steps": "ground_service_steps",
    "g4_run": "ground_service_run",
    "g5_realization": "ground_failure_realization",
    "g5_steps": "failure_adjusted_service_steps",
    "g5_run": "failure_adjusted_service_run",
    "target": "canonical_target",
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def contract_root() -> Path:
    return repository_root() / "artifacts" / "final_integrated_dataset_10k_contract"


def catalog_path() -> Path:
    return (
        repository_root()
        / "artifacts"
        / "integrated_ground_pilot_25"
        / "inputs"
        / "pilot_catalog.csv"
    )
