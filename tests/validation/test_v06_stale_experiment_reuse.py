from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_ablation_study import (
    ExperimentSpec,
    collect_results,
    run_specs,
    write_experiment_identity,
    write_run_manifest,
)


def _spec(tmp_path: Path, condition: str, command: list[str]) -> ExperimentSpec:
    output_dir = tmp_path / "tgnn" / "partition_any" / condition
    base = f"satellite_gnn_partition_any_{condition}"
    return ExperimentSpec(
        model="tgnn",
        condition=condition,
        target="partition_any",
        output_dir=output_dir,
        command=command,
        metrics_path=output_dir / f"{base}_metrics.json",
        predictions_path=output_dir / f"{base}_predictions.csv",
        model_path=output_dir / f"{base}.pt",
        config_path=output_dir / f"{base}_config.json",
        split_manifest_path=tmp_path / "splits" / "partition_any_split.json",
    )


@pytest.mark.parametrize(
    ("changed_option", "changed_value"),
    [
        ("--cheb-k", "3"),
        ("--seed", "43"),
        ("--data-dir", "dataset-b"),
        ("--split-manifest", "split-b.json"),
        ("--input-mode", "topology_only"),
        ("--target-name", "gcc_frac_min_original"),
        ("--hidden-dim", "128"),
    ],
)
def test_v06_existing_metrics_skip_changed_scientific_specification(
    tmp_path: Path,
    monkeypatch,
    changed_option: str,
    changed_value: str,
) -> None:
    baseline_command = [
        "python",
        "train_gnn_model.py",
        "--cheb-k",
        "2",
        "--seed",
        "42",
        "--data-dir",
        "dataset-a",
        "--split-manifest",
        "split-a.json",
        "--input-mode",
        "full",
        "--target-name",
        "partition_any",
        "--hidden-dim",
        "64",
    ]
    baseline = _spec(tmp_path, "full", baseline_command)
    baseline.output_dir.mkdir(parents=True)
    baseline.metrics_path.write_text(
        json.dumps({"cheb_k": 2, "seed": 42, "dataset_sha256": "dataset-a"})
    )
    write_experiment_identity(baseline)

    changed_command = list(baseline_command)
    changed_command[changed_command.index(changed_option) + 1] = changed_value
    spec = _spec(tmp_path, "full", changed_command)
    subprocess_calls: list[object] = []
    monkeypatch.setattr(
        "scripts.run_ablation_study.subprocess.run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        run_specs([spec], overwrite=False)

    assert subprocess_calls == []


def test_v06_exact_identity_allows_safe_reuse(tmp_path: Path, monkeypatch) -> None:
    spec = _spec(tmp_path, "full", ["python", "train_gnn_model.py", "--cheb-k", "2"])
    spec.output_dir.mkdir(parents=True)
    spec.metrics_path.write_text(json.dumps({"cheb_k": 2, "seed": 42}))
    write_experiment_identity(spec)
    monkeypatch.setattr(
        "scripts.run_ablation_study.subprocess.run",
        lambda *args, **kwargs: pytest.fail("matching identity should be reused"),
    )

    run_specs([spec], overwrite=False)


def test_v06_manifest_refuses_existing_metrics_without_matching_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "ablation"
    spec = _spec(output_dir, "full", ["python", "train_gnn_model.py", "--cheb-k", "3"])
    spec.output_dir.mkdir(parents=True)
    spec.metrics_path.write_text(json.dumps({"cheb_k": 2, "seed": 42}))
    monkeypatch.setattr(
        "scripts.run_ablation_study.subprocess.run",
        lambda *args, **kwargs: pytest.fail("existing metrics should trigger the observed skip"),
    )
    args = Namespace(seed=43, targets=["partition_any"], smoke=False, cheb_k=3)

    with pytest.raises(ValueError, match="lack experiment identity"):
        write_run_manifest(specs=[spec], args=args, output_dir=output_dir)

    assert not (output_dir / "run_manifest.json").exists()
    assert json.loads(spec.metrics_path.read_text()) == {"cheb_k": 2, "seed": 42}


def test_v06_comparison_collection_rejects_incompatible_metric_provenance(
    tmp_path: Path,
) -> None:
    full = _spec(tmp_path, "full", ["python", "train.py"])
    ablated = _spec(tmp_path, "topology_only", ["python", "train.py"])
    for spec, dataset_hash, split_hash, seed in (
        (full, "dataset-a", "split-a", 42),
        (ablated, "dataset-b", "split-b", 43),
    ):
        spec.output_dir.mkdir(parents=True)
        spec.metrics_path.write_text(
            json.dumps(
                {
                    "task_type": "classification",
                    "accuracy": 0.75,
                    "precision": 0.75,
                    "recall": 0.75,
                    "f1": 0.75,
                    "dataset_sha256": dataset_hash,
                    "split_sha256": split_hash,
                    "seed": seed,
                }
            )
        )
        write_experiment_identity(spec)

    with pytest.raises(ValueError, match="Mixed metric provenance"):
        collect_results([full, ablated], tmp_path)

    assert not (tmp_path / "comparison" / "classification_results.csv").exists()
