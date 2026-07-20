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
    allowed_prefixes = (
        "artifacts/final_integrated_dataset_contract/",
        "docs/experiments/final_integrated_dataset_",
        "src/satnet/experiments/final_dataset/",
        "tests/experiments/test_final_dataset_",
    )
    assert paths
    assert all(path.startswith(allowed_prefixes) for path in paths)


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
