from __future__ import annotations

from argparse import Namespace
import json

import pandas as pd
import pytest


def _write_dataset(data_dir, n: int = 12) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        rows.append(
            {
                "run_id": i,
                "config_hash": f"cfg-{i}",
                "num_planes": 4,
                "sats_per_plane": 6,
                "total_satellites": 24,
                "inclination_deg": 53.0,
                "altitude_km": 550.0,
                "node_failure_prob": 0.01,
                "edge_failure_prob": 0.02,
                "duration_minutes": 5,
                "step_seconds": 60,
                "failure_model": "persistent_temporal_union_edges_v1",
                "isl_policy": "grid_adaptive",
                "adjacent_search_k": 1,
                "max_inter_plane_links_per_sat": 1,
                "partition_any": i % 2,
                "gcc_frac_min_original": float(i) / max(n - 1, 1),
            }
        )
    pd.DataFrame(rows).to_csv(data_dir / "tier1_design_runs.csv", index=False)


def _args(tmp_path, **overrides):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "artifacts" / "ablation"
    _write_dataset(data_dir)
    values = dict(
        data_dir=str(data_dir),
        seed=42,
        targets=["partition_any", "gcc_frac_min_original"],
        epochs=1,
        device="cpu",
        smoke=True,
        output_dir=str(output_dir),
        skip_rf=False,
        skip_tgnn=False,
        overwrite=True,
        dry_run=True,
        rf_n_estimators=25,
        hidden_dim=16,
        cheb_k=2,
        test_size=0.25,
        val_size=0.25,
    )
    values.update(overrides)
    return Namespace(**values)


def test_dry_run_builds_all_expected_specs(tmp_path) -> None:
    import scripts.run_ablation_study as runner

    specs = runner.build_experiment_specs(_args(tmp_path))

    assert len(specs) == 12
    assert {(s.model, s.condition) for s in specs if s.model == "rf"} == {
        ("rf", "full"),
        ("rf", "architecture_only"),
        ("rf", "no_geometry"),
    }
    assert {(s.model, s.condition) for s in specs if s.model == "tgnn"} == {
        ("tgnn", "full"),
        ("tgnn", "topology_only"),
        ("tgnn", "node_state_only"),
    }
    assert all(str(s.output_dir).endswith(s.condition) for s in specs)
    assert all("--cheb-k" in s.command for s in specs if s.model == "tgnn")
    assert all(runner.build_experiment_identity(spec)["condition_identity"] for spec in specs)
    assert all(runner.build_experiment_identity(spec)["comparison_identity"] for spec in specs)


def test_failed_condition_raises_nonzero_runner_failure(tmp_path, monkeypatch) -> None:
    import scripts.run_ablation_study as runner

    args = _args(tmp_path, targets=["partition_any"], skip_tgnn=True)
    specs = runner.build_experiment_specs(args)

    class Result:
        returncode = 1
        stdout = "out"
        stderr = "err"

    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: Result())

    with pytest.raises(RuntimeError, match="Ablation condition failed"):
        runner.run_specs(specs[:1], overwrite=True)


def test_comparison_deltas_have_expected_direction(tmp_path) -> None:
    import scripts.run_ablation_study as runner

    args = _args(tmp_path, targets=["partition_any"], skip_tgnn=True)
    specs = runner.build_experiment_specs(args)
    for spec in specs:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "task_type": "classification",
            "target_name": "partition_any",
            "num_samples": 12,
            "train_size": 6,
            "val_size": 3,
            "test_size": 3,
            "accuracy": 0.8 if spec.condition == "full" else 0.6,
            "precision": 0.7,
            "recall": 0.7,
            "f1": 0.75 if spec.condition == "full" else 0.5,
        }
        spec.metrics_path.write_text(json.dumps(metrics))
        runner.write_experiment_identity(spec)

    runner.collect_results(specs, tmp_path / "artifacts" / "ablation")
    df = pd.read_csv(tmp_path / "artifacts" / "ablation" / "comparison" / "classification_results.csv")
    ablated = df[df["condition"] == "architecture_only"].iloc[0]

    assert ablated["delta_accuracy"] == pytest.approx(-0.2)
    assert ablated["delta_f1"] == pytest.approx(-0.25)
