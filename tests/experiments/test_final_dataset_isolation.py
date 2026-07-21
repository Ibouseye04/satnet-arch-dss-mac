from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).parents[2]
PILOT_REPORT_SHA = "e4475f9bc22a83b30cdc6862d3337a1e2b6dbc3f"
PROTECTED_PATHS = (
    "src/satnet/ground",
    "src/satnet/network",
    "src/satnet/simulation/tier1_rollout.py",
    "src/satnet/models/gnn_dataset.py",
    "src/satnet/models/gnn_model.py",
    "src/satnet/models/risk_model.py",
    "src/satnet/utils/graph_cache.py",
)
ALLOWED_EXACT_PATHS = frozenset({".gitattributes", "tests/test_stage_a_byte_preservation_policy.py"})
ALLOWED_PREFIXES = (
    "artifacts/final_integrated_dataset_contract/",
    "artifacts/final_integrated_dataset_generation_qualification/",
    "artifacts/final_integrated_dataset_class_support_analysis/",
    "artifacts/final_integrated_dataset_class_support_audit/",
    "artifacts/stage_a_discovery_contract_proposal/",
    "artifacts/stage_a_discovery_contract_freeze_audit/",
    "artifacts/stage_a_discovery_contract_v1/",
    "docs/experiments/final_integrated_dataset_",
    "docs/stage_a_discovery_contract_correction_v1.md",
    "docs/stage_a_near_neighbor_resolution_v1.md",
    "docs/validation/final_integrated_dataset_class_support_audit.md",
    "docs/validation/stage_a_discovery_contract_freeze_readiness_audit.md",
    "scripts/analysis/final_integrated_dataset_class_support.py",
    "scripts/validation/audit_final_dataset_class_support.py",
    "scripts/validation/audit_stage_a_discovery_contract.py",
    "scripts/experiments/build_stage_a_contract_proposal.py",
    "src/satnet/experiments/final_class_support/",
    "src/satnet/experiments/final_class_support_audit/",
    "src/satnet/experiments/stage_a_contract/",
    "src/satnet/experiments/final_dataset/",
    "src/satnet/experiments/final_generation/",
    "tests/experiments/test_final_class_support_analysis.py",
    "tests/experiments/test_final_dataset_",
    "tests/experiments/test_final_generation_",
    "tests/experiments/test_stage_a_contract_",
    "tests/validation/test_final_dataset_class_support_audit.py",
    "tests/validation/test_stage_a_discovery_contract_freeze_audit.py",
)


def _is_approved_path(path: str) -> bool:
    return path in ALLOWED_EXACT_PATHS or path.startswith(ALLOWED_PREFIXES)


def test_protected_science_diff_is_empty() -> None:
    result = subprocess.run(
        ["git", "diff", "--name-only", PILOT_REPORT_SHA, "--", *PROTECTED_PATHS],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_contract_changes_are_confined_to_approved_surfaces() -> None:
    result = subprocess.run(
        ["git", "diff", "--name-only", PILOT_REPORT_SHA],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = tuple(line for line in result.stdout.splitlines() if line)
    assert paths
    assert all(_is_approved_path(path) for path in paths)


def test_contract_change_allowlist_rejects_unauthorized_path() -> None:
    assert not _is_approved_path("src/satnet/network/unauthorized_change.py")
    assert not _is_approved_path("tests/experiments/unrelated_test.py")


def test_contract_artifact_root_contains_no_scientific_run_evidence() -> None:
    root = ROOT / "artifacts" / "final_integrated_dataset_contract"
    expected = {
        "contract_bundle.json",
        "contract_specification.json",
        "designs.jsonl",
        "doe_evidence.json",
        "golden_vectors.json",
        "integrated_rf_export_schema.json",
        "integrated_tgnn_adapter_schema.json",
        "manifest_inventory.json",
        "runs.jsonl",
        "split_manifest.json",
        "target_schema.json",
    }
    assert {path.name for path in root.iterdir()} == expected
    prohibited_fragments = (
        "visibility",
        "integrated_graph",
        "service_step",
        "service_run",
        "failure_realization",
        "satellite_artifact",
    )
    assert all(
        not any(fragment in path.name for fragment in prohibited_fragments)
        for path in root.iterdir()
    )
