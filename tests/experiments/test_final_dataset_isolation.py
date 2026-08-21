from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).parents[2]
PILOT_REPORT_SHA = "a8fbfed18b1673f5fc9c6a291ccc02905f1392d6"
PROTECTED_PATHS = (
    "src/satnet/ground",
    "src/satnet/network",
    "src/satnet/simulation/tier1_rollout.py",
    "src/satnet/models/gnn_dataset.py",
    "src/satnet/models/gnn_model.py",
    "src/satnet/models/risk_model.py",
    "src/satnet/utils/graph_cache.py",
)
ALLOWED_EXACT_PATHS = frozenset({".gitattributes"})
ALLOWED_PREFIXES = (
    "artifacts/final_integrated_dataset_contract/",
    "artifacts/final_integrated_dataset_10k_contract/",
    "artifacts/final_integrated_dataset_generation_qualification/",
    "docs/experiments/final_integrated_dataset_",
    "src/satnet/experiments/final_dataset/",
    "src/satnet/experiments/final_generation/",
    "src/satnet/experiments/final_training/",
    "src/satnet/experiments/integrated_ground_analysis.py",
    "tests/experiments/final_training/",
    "tests/experiments/test_final_dataset_",
    "tests/experiments/test_final_generation_",
    "tests/experiments/test_integrated_ground_analysis.py",
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
    root = ROOT / "artifacts" / "final_integrated_dataset_10k_contract"
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
