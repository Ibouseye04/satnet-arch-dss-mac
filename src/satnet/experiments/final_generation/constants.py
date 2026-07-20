from __future__ import annotations

from pathlib import Path

FROZEN_COMMIT = "a1967185e80327e4b00c1831828dc975ab6819fc"
FROZEN_TAG = "final-integrated-dataset-contract-v1"
CONTRACT_SPEC_HASH = "482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b"
CONTRACT_BUNDLE_HASH = "3250dcf66e859a7dba151c6564827fcb89ddbea17a5ab5d39540e3087dd2e2ba"
DESIGN_MANIFEST_HASH = "43ffe79701c7e624abc17c45f198397c20fae55ed243502953aa8c898462bcc8"
RUN_MANIFEST_HASH = "2925c3c65cf7e2b6debcc42b6e6414186f3474eb88998486f7571af73dda2a36"
SPLIT_MANIFEST_HASH = "930454b2be6eb5033efebc7ab407c2400c66f0ca998e9283c36890b69ea2e08d"
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
    {"qualification", "qualification_replay", "qualification_repeat", "production"}
)
QUALIFICATION_RUN_IDS = (0, 1, 2, 3, 4, 35, 36, 37, 38, 39, 200, 201, 202, 203, 204)
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
    return repository_root() / "artifacts" / "final_integrated_dataset_contract"


def catalog_path() -> Path:
    return (
        repository_root()
        / "artifacts"
        / "integrated_ground_pilot_25"
        / "inputs"
        / "pilot_catalog.csv"
    )
