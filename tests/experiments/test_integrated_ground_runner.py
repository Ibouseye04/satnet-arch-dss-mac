from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from satnet.experiments.integrated_ground_manifest import (
    build_pilot_designs,
    build_pilot_runs,
    materialize_pilot_inputs,
)
from satnet.experiments.integrated_ground_replay import compare_repeat_run
from satnet.experiments.integrated_ground_runner import (
    CANONICAL_ARTIFACT_KEYS,
    PilotSatelliteArtifact,
    artifact_paths,
    make_pilot_satellite_artifact,
    read_pilot_satellite_artifact,
    run_directory,
    run_pilot_manifest,
    select_manifest_runs,
    write_pilot_satellite_artifact,
)
from satnet.simulation.tier1_rollout import (
    DATASET_VERSION,
    SCHEMA_VERSION,
    Tier1FailureRealization,
    Tier1RolloutStep,
    Tier1RolloutSummary,
)


def satellite_artifact() -> PilotSatelliteArtifact:
    design = build_pilot_designs()[-1]
    run = build_pilot_runs()[24]
    config = design.satellite_config(satellite_seed=run.satellite_rollout_seed)
    steps = tuple(
        Tier1RolloutStep(
            t=index,
            num_nodes=20,
            num_edges=20,
            num_components=1,
            gcc_size=20,
            gcc_frac=1.0,
            gcc_frac_original=1.0,
            gcc_frac_surviving=1.0,
            partitioned=0,
        )
        for index in range(config.num_steps)
    )
    summary = Tier1RolloutSummary(
        gcc_frac_min=1.0,
        gcc_frac_mean=1.0,
        gcc_frac_min_original=1.0,
        gcc_frac_mean_original=1.0,
        gcc_frac_min_surviving=1.0,
        gcc_frac_mean_surviving=1.0,
        partition_fraction=0.0,
        partition_any=0,
        max_partition_streak=0,
        max_partition_streak_seconds=0,
        max_partition_streak_fraction=0.0,
        num_steps=config.num_steps,
        num_failed_nodes=0,
        num_failed_edges=0,
        schema_version=SCHEMA_VERSION,
        dataset_version=DATASET_VERSION,
        config_hash=config.config_hash(),
    )
    return make_pilot_satellite_artifact(
        run_id=run.run_id,
        config=config,
        steps=steps,
        summary=summary,
        failure_realization=Tier1FailureRealization(set(), set()),
    )


def test_satellite_artifact_roundtrip_and_corruption(tmp_path: Path) -> None:
    artifact = satellite_artifact()
    path = tmp_path / "satellite.json"
    write_pilot_satellite_artifact(artifact, path)
    assert read_pilot_satellite_artifact(path) == artifact
    value = json.loads(path.read_text(encoding="utf-8"))
    value["scientific_payload"]["failed_nodes"] = [0]
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_pilot_satellite_artifact(path)


def test_satellite_record_identity_binds_run() -> None:
    artifact = satellite_artifact()
    with pytest.raises(ValueError, match="record hash"):
        replace(artifact, run_id=23)


def test_run_subset_selection_is_sorted_and_strict() -> None:
    runs = build_pilot_runs()
    assert [run.run_id for run in select_manifest_runs(runs, (24, 0, 12))] == [0, 12, 24]
    with pytest.raises(ValueError, match="duplicate"):
        select_manifest_runs(runs, (0, 0))
    with pytest.raises(ValueError, match="unknown"):
        select_manifest_runs(runs, (25,))


def test_failed_run_is_reported_without_seed_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "pilot"
    materialize_pilot_inputs(root)

    def fail(**kwargs):
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(
        "satnet.experiments.integrated_ground_runner.generate_pilot_run", fail
    )
    result = run_pilot_manifest(
        manifest_path=root / "inputs" / "pilot_runs.jsonl",
        output_root=root,
        run_ids=(0,),
    )
    assert result["attempted_run_count"] == 1
    assert result["successful_generation_run_count"] == 0
    assert result["failed_generation_run_count"] == 1
    assert result["failures"][0]["run_id"] == 0
    assert result["failures"][0]["error_message"] == "controlled failure"
    assert (root / "runs" / "run_000" / "generation_failure.json").is_file()
    assert not (root / "runs" / "run_001").exists()


def test_resume_validates_existing_evidence_instead_of_trusting_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "pilot"
    materialize_pilot_inputs(root)
    run_root = run_directory(root, 0)
    run_root.mkdir(parents=True)
    (run_root / "sentinel").write_text("evidence", encoding="utf-8")
    calls: list[int] = []

    def validate_existing_run(**kwargs):
        calls.append(kwargs["run"].run_id)
        return {"run_id": kwargs["run"].run_id}

    monkeypatch.setattr(
        "satnet.experiments.integrated_ground_replay.validate_existing_run",
        validate_existing_run,
    )
    result = run_pilot_manifest(
        manifest_path=root / "inputs" / "pilot_runs.jsonl",
        output_root=root,
        run_ids=(0,),
        resume=True,
    )
    assert calls == [0]
    assert result["resumed_run_ids"] == [0]
    assert result["successful_run_ids"] == [0]


def test_repeat_comparison_checks_all_canonical_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    primary = artifact_paths(run_directory(root, 0))
    repeated = artifact_paths(run_directory(root, 0, repeat=True))
    for paths in (primary, repeated):
        paths["summary"].parent.mkdir(parents=True, exist_ok=True)
        for key in CANONICAL_ARTIFACT_KEYS:
            paths[key].write_bytes(f"{key}\n".encode())
        paths["summary"].write_text(
            json.dumps(
                {
                    "run_id": 0,
                    "generation_runtime_seconds": 1.0,
                    "replay_runtime_seconds": None,
                    "replay_status": "pending",
                    "artifact_bytes": 10,
                    "satellite_rollout_seconds": 1.0,
                    "g1_seconds": 0.0,
                    "g2_seconds": 0.0,
                    "g3_seconds": 0.0,
                    "g4_seconds": 0.0,
                    "g5_seconds": 0.0,
                    "persistence_seconds": 0.0,
                    "total_generation_seconds": 1.0,
                    "scientific_hash": "a" * 64,
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
    repeated["summary"].write_text(
        repeated["summary"].read_text(encoding="utf-8").replace("1.0", "2.0"),
        encoding="utf-8",
    )
    result = compare_repeat_run(output_root=root, run_id=0)
    assert result["canonical_artifact_match"] is True
    assert result["scientific_summary_match"] is True
    repeated["g5_run"].write_text("different\n", encoding="utf-8")
    with pytest.raises(ValueError, match="determinism mismatch"):
        compare_repeat_run(output_root=root, run_id=0)
