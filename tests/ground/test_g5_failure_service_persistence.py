from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from satnet.ground.failure_policy import GroundFailurePolicy
from satnet.ground.failure_service_persistence import (
    generate_verified_ground_failure_service_records,
    persist_verified_ground_failure_service_evidence,
    read_ground_failure_realization_manifest,
    read_ground_failure_service_run_manifest,
    read_ground_failure_service_step_manifest,
    replay_ground_failure_service_records,
    write_ground_failure_realization_manifest,
    write_ground_failure_service_run_manifest,
    write_ground_failure_service_step_manifest,
)
from tests.ground.test_g4_service_persistence import verified_context


def generation_context(*, probability: float = 0.4, seed: int = 19):
    values = verified_context()
    (
        config,
        satellite_failures,
        catalog,
        ground_design,
        visibility_policy,
        visibility_records,
        _,
        integrated_records,
        ground_service_policy,
        g4_steps,
        g4_run,
    ) = values
    policy = GroundFailurePolicy(probability)
    kwargs = {
        "satellite_config": config,
        "satellite_failure_realization": satellite_failures,
        "ground_design": ground_design,
        "catalog": catalog,
        "visibility_policy": visibility_policy,
        "visibility_records": visibility_records,
        "integrated_records": integrated_records,
        "ground_service_policy": ground_service_policy,
        "g4_step_records": g4_steps,
        "g4_run_record": g4_run,
        "ground_failure_policy": policy,
        "ground_failure_seed": seed,
    }
    return values, policy, kwargs


def replay_kwargs(*, probability: float = 0.4, seed: int = 19):
    values, policy, generation = generation_context(probability=probability, seed=seed)
    realization, steps, run = generate_verified_ground_failure_service_records(**generation)
    kwargs = dict(generation)
    kwargs.pop("ground_failure_seed")
    kwargs.update(
        {
            "realization_records": (realization,),
            "g5_step_records": steps,
            "g5_run_records": (run,),
        }
    )
    return values, policy, (realization, steps, run), kwargs


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def test_verified_generation_begins_with_g4_replay_and_exact_g5_replay() -> None:
    _, _, expected, kwargs = replay_kwargs()
    assert replay_ground_failure_service_records(**kwargs) == expected


def test_generation_and_replay_modes_are_mutually_exclusive() -> None:
    _, _, generation = generation_context()
    with pytest.raises(ValueError, match="Exactly one"):
        generate_verified_ground_failure_service_records(
            **generation,
            persisted_realization_record=generate_verified_ground_failure_service_records(
                **generation
            )[0],
        )
    generation.pop("ground_failure_seed")
    with pytest.raises(ValueError, match="Exactly one"):
        generate_verified_ground_failure_service_records(**generation)


def test_replay_resamples_and_rejects_wrong_authoritative_policy() -> None:
    _, _, _, kwargs = replay_kwargs(probability=0.4)
    kwargs["ground_failure_policy"] = GroundFailurePolicy(0.5)
    with pytest.raises(ValueError, match="authoritative context"):
        replay_ground_failure_service_records(**kwargs)


def test_three_manifest_roundtrip_and_canonical_newlines(tmp_path: Path) -> None:
    _, _, evidence, _ = replay_kwargs()
    realization, steps, run = evidence
    realization_path = tmp_path / "ground_failure_realizations.jsonl"
    step_path = tmp_path / "ground_failure_service_steps.jsonl"
    run_path = tmp_path / "ground_failure_service_runs.jsonl"
    write_ground_failure_realization_manifest((realization,), realization_path)
    write_ground_failure_service_step_manifest(steps, step_path)
    write_ground_failure_service_run_manifest((run,), run_path)
    assert read_ground_failure_realization_manifest(realization_path) == (realization,)
    assert read_ground_failure_service_step_manifest(step_path) == steps
    assert read_ground_failure_service_run_manifest(run_path) == (run,)
    for path in (realization_path, step_path, run_path):
        raw = path.read_bytes()
        assert raw.endswith(b"\n")
        assert b"\r\n" not in raw
        assert b" " not in raw


def test_verified_persistence_generates_before_writing_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    _, _, generation = generation_context()
    paths = {
        "realization_path": tmp_path / "ground_failure_realizations.jsonl",
        "step_path": tmp_path / "ground_failure_service_steps.jsonl",
        "run_path": tmp_path / "ground_failure_service_runs.jsonl",
    }
    expected = generate_verified_ground_failure_service_records(**generation)
    persisted = persist_verified_ground_failure_service_evidence(
        **generation,
        **paths,
    )
    assert persisted == expected
    with pytest.raises(FileExistsError):
        persist_verified_ground_failure_service_evidence(
            **generation,
            **paths,
        )


def test_manifest_readers_reject_unknown_missing_duplicate_keys_and_empty_lines(
    tmp_path: Path,
) -> None:
    _, _, evidence, _ = replay_kwargs()
    realization = evidence[0]
    path = tmp_path / "manifest.jsonl"
    write_ground_failure_realization_manifest((realization,), path)
    original = json.loads(path.read_text(encoding="utf-8"))
    unknown = deepcopy(original)
    unknown["unknown"] = 1
    write_json(path, unknown)
    with pytest.raises(ValueError, match="fields invalid"):
        read_ground_failure_realization_manifest(path)
    missing = deepcopy(original)
    del missing["record_hash"]
    write_json(path, missing)
    with pytest.raises(ValueError, match="fields invalid"):
        read_ground_failure_realization_manifest(path)
    path.write_text('{"run_id":1,"run_id":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        read_ground_failure_realization_manifest(path)
    path.write_text("{}\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty line"):
        read_ground_failure_realization_manifest(path)


def test_persisted_realization_corruption_fails_intrinsically(tmp_path: Path) -> None:
    _, _, evidence, _ = replay_kwargs()
    realization = evidence[0]
    path = tmp_path / "realization.jsonl"
    write_ground_failure_realization_manifest((realization,), path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["realization"]["failed_ground_station_count"] += 1
    write_json(path, value)
    with pytest.raises(ValueError):
        read_ground_failure_realization_manifest(path)


def test_persisted_step_and_run_corruption_fail_intrinsically(tmp_path: Path) -> None:
    _, _, evidence, _ = replay_kwargs()
    _, steps, run = evidence
    step_path = tmp_path / "steps.jsonl"
    run_path = tmp_path / "runs.jsonl"
    write_ground_failure_service_step_manifest(steps, step_path)
    values = [json.loads(line) for line in step_path.read_text(encoding="utf-8").splitlines()]
    values[0]["metrics"]["failure_adjusted_ground_service_fraction"] = "0.123"
    step_path.write_text(
        "\n".join(json.dumps(value, separators=(",", ":")) for value in values) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        read_ground_failure_service_step_manifest(step_path)
    write_ground_failure_service_run_manifest((run,), run_path)
    value = json.loads(run_path.read_text(encoding="utf-8"))
    value["summary"]["step_sequence_hash"] = "f" * 64
    write_json(run_path, value)
    with pytest.raises(ValueError):
        read_ground_failure_service_run_manifest(run_path)


def test_persisted_negative_zero_fails(tmp_path: Path) -> None:
    _, _, evidence, _ = replay_kwargs(probability=0.0)
    path = tmp_path / "steps.jsonl"
    write_ground_failure_service_step_manifest(evidence[1], path)
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    values[0]["metrics"]["ground_service_loss_due_to_failures"] = "-0"
    path.write_text(
        "\n".join(json.dumps(value, separators=(",", ":")) for value in values) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        read_ground_failure_service_step_manifest(path)


def test_replay_rejects_missing_extra_duplicate_and_reordered_steps() -> None:
    _, _, evidence, kwargs = replay_kwargs()
    _, steps, _ = evidence
    cases = (
        steps[:-1],
        (*steps, steps[-1]),
        (steps[0], steps[0]),
        tuple(reversed(steps)),
    )
    for case in cases:
        corrupted = dict(kwargs)
        corrupted["g5_step_records"] = case
        with pytest.raises(ValueError):
            replay_ground_failure_service_records(**corrupted)


def test_replay_rejects_wrong_realization_and_run_collection_cardinality() -> None:
    _, _, evidence, kwargs = replay_kwargs()
    realization, _, run = evidence
    for realization_records, run_records in (
        ((), (run,)),
        ((realization, realization), (run,)),
        ((realization,), ()),
        ((realization,), (run, run)),
    ):
        corrupted = dict(kwargs)
        corrupted["realization_records"] = realization_records
        corrupted["g5_run_records"] = run_records
        with pytest.raises(ValueError):
            replay_ground_failure_service_records(**corrupted)
