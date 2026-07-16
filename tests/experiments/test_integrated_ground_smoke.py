from __future__ import annotations

from pathlib import Path

from satnet.experiments.integrated_ground_manifest import materialize_pilot_inputs
from satnet.experiments.integrated_ground_replay import replay_pilot_manifest
from satnet.experiments.integrated_ground_runner import (
    artifact_paths,
    run_directory,
    run_pilot_manifest,
)


def test_single_stress_run_generates_and_replays_through_g1_to_g5(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pilot"
    materialize_pilot_inputs(root)
    manifest = root / "inputs" / "pilot_runs.jsonl"
    generated = run_pilot_manifest(
        manifest_path=manifest,
        output_root=root,
        run_ids=(24,),
    )
    assert generated["attempted_run_count"] == 1
    assert generated["successful_generation_run_count"] == 1
    assert generated["failed_generation_run_count"] == 0
    paths = artifact_paths(run_directory(root, 24))
    assert all(path.is_file() for path in paths.values())
    replayed = replay_pilot_manifest(
        manifest_path=manifest,
        output_root=root,
        run_ids=(24,),
    )
    assert replayed["attempted_replay_run_count"] == 1
    assert replayed["successful_replay_run_count"] == 1
    assert replayed["failed_replay_run_count"] == 0
    result = replayed["results"][0]
    assert result["g2_record_count"] == 11
    assert result["g3_record_count"] == 11
    assert result["g4_step_count"] == 11
    assert result["g5_step_count"] == 11
