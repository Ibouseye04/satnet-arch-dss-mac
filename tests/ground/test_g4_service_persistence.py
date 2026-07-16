from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from satnet.ground.integrated_graph import create_integrated_graph_snapshot
from satnet.ground.integrated_persistence import make_integrated_graph_record
from satnet.ground.service_aggregation import (
    GROUND_SERVICE_STEP_SCHEMA_VERSION,
    GroundServiceRunSummary,
    GroundServiceStepRecord,
    _step_record_hash,
)
from satnet.ground.service_metrics import compute_ground_service_step
from satnet.ground.service_persistence import (
    GroundServiceRunRecord,
    _run_record_hash,
    generate_verified_ground_service_records,
    persist_verified_ground_service_evidence,
    read_ground_service_run_manifest,
    read_ground_service_step_manifest,
    replay_ground_service_records,
    write_ground_service_run_manifest,
    write_ground_service_step_manifest,
)
from satnet.ground.service_policy import GroundServicePolicy
from tests.ground.test_g3_persistence import context


def verified_context(*, all_failed: bool = False):
    (
        config,
        failures,
        catalog,
        ground_design,
        visibility_policy,
        _,
        visibility_records,
        integrated_snapshots,
        integrated_records,
    ) = context(all_failed=all_failed)
    service_policy = GroundServicePolicy(0.5, 0.5)
    step_records, run_record = generate_verified_ground_service_records(
        satellite_config=config,
        failure_realization=failures,
        ground_design=ground_design,
        catalog=catalog,
        visibility_policy=visibility_policy,
        visibility_records=visibility_records,
        integrated_records=integrated_records,
        service_policy=service_policy,
    )
    return (
        config,
        failures,
        catalog,
        ground_design,
        visibility_policy,
        visibility_records,
        integrated_snapshots,
        integrated_records,
        service_policy,
        step_records,
        run_record,
    )


def replay_kwargs(values: tuple) -> dict[str, object]:
    (
        config,
        failures,
        catalog,
        ground_design,
        visibility_policy,
        visibility_records,
        _,
        integrated_records,
        service_policy,
        step_records,
        run_record,
    ) = values
    return {
        "satellite_config": config,
        "failure_realization": failures,
        "ground_design": ground_design,
        "catalog": catalog,
        "visibility_policy": visibility_policy,
        "visibility_records": visibility_records,
        "integrated_records": integrated_records,
        "service_policy": service_policy,
        "step_records": step_records,
        "run_records": (run_record,),
    }


def write_object(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def test_verified_generation_and_exact_replay_normal_and_all_failed() -> None:
    for all_failed in (False, True):
        values = verified_context(all_failed=all_failed)
        expected_steps = values[-2]
        expected_run = values[-1]
        replayed_steps, replayed_run = replay_ground_service_records(**replay_kwargs(values))
        assert replayed_steps == expected_steps
        assert replayed_run == expected_run
        assert all(
            step.metrics.configured_satellite_count == values[0].total_satellites
            for step in replayed_steps
        )
        if all_failed:
            assert all(step.metrics.operational_satellite_count == 0 for step in replayed_steps)


def test_step_and_run_manifest_round_trip_order_and_path_determinism(tmp_path: Path) -> None:
    *_, step_records, run_record = verified_context()
    alternate_steps = tuple(
        GroundServiceStepRecord(
            ground_service_step_schema_version=GROUND_SERVICE_STEP_SCHEMA_VERSION,
            run_id=9,
            metrics=record.metrics,
            record_hash=_step_record_hash(run_id=9, metrics=record.metrics),
        )
        for record in step_records
    )
    step_path = tmp_path / "a" / "ground_service_steps.jsonl"
    step_path_2 = tmp_path / "b" / "ground_service_steps.jsonl"
    run_path = tmp_path / "a" / "ground_service_runs.jsonl"
    write_ground_service_step_manifest(alternate_steps + tuple(reversed(step_records)), step_path)
    write_ground_service_step_manifest(alternate_steps + tuple(reversed(step_records)), step_path_2)
    write_ground_service_run_manifest((run_record,), run_path)
    assert read_ground_service_step_manifest(step_path) == step_records + alternate_steps
    assert read_ground_service_run_manifest(run_path) == (run_record,)
    assert step_path.read_bytes() == step_path_2.read_bytes()
    assert step_path.read_bytes().endswith(b"\n")
    assert b"\r\n" not in step_path.read_bytes()
    assert all(list(json.loads(line)) == sorted(json.loads(line)) for line in step_path.read_text().splitlines())


def test_verified_persistence_writes_standalone_files(tmp_path: Path) -> None:
    values = verified_context()
    kwargs = replay_kwargs(values)
    kwargs.pop("step_records")
    kwargs.pop("run_records")
    step_path = tmp_path / "ground_service_steps.jsonl"
    run_path = tmp_path / "ground_service_runs.jsonl"
    step_records, run_record = persist_verified_ground_service_evidence(
        **kwargs,
        step_path=step_path,
        run_path=run_path,
    )
    assert read_ground_service_step_manifest(step_path) == step_records
    assert read_ground_service_run_manifest(run_path) == (run_record,)
    before = step_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_ground_service_step_manifest(step_records, step_path)
    assert step_path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("content", ["", "\n", "{}\n\n{}\n"])
def test_readers_reject_empty_files_and_lines(tmp_path: Path, content: str) -> None:
    for reader, name in (
        (read_ground_service_step_manifest, "steps.jsonl"),
        (read_ground_service_run_manifest, "runs.jsonl"),
    ):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            reader(path)


def test_readers_reject_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "steps.jsonl"
    path.write_text('{"run_id":1,"run_id":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        read_ground_service_step_manifest(path)


def test_step_reader_rejects_unknown_missing_noncanonical_and_corrupt_fields(
    tmp_path: Path,
) -> None:
    *_, step_records, _ = verified_context()
    path = tmp_path / "steps.jsonl"
    write_ground_service_step_manifest(step_records, path)
    value = json.loads(path.read_text().splitlines()[0])
    extra = deepcopy(value)
    extra["unexpected"] = 1
    write_object(path, extra)
    with pytest.raises(ValueError, match="unknown=.*unexpected"):
        read_ground_service_step_manifest(path)
    missing = deepcopy(value)
    del missing["metrics"]["integrated_graph_hash"]
    write_object(path, missing)
    with pytest.raises(ValueError, match="missing=.*integrated_graph_hash"):
        read_ground_service_step_manifest(path)
    noncanonical = deepcopy(value)
    noncanonical["metrics"]["ground_service_fraction"] = 1.0
    write_object(path, noncanonical)
    with pytest.raises(TypeError, match="canonical float string"):
        read_ground_service_step_manifest(path)
    corrupt = deepcopy(value)
    corrupt["metrics"]["satellite_gcc_size"] += 1
    write_object(path, corrupt)
    with pytest.raises(ValueError):
        read_ground_service_step_manifest(path)


def test_duplicate_step_and_run_records_fail(tmp_path: Path) -> None:
    *_, step_records, run_record = verified_context()
    with pytest.raises(ValueError, match="Duplicate"):
        write_ground_service_step_manifest(
            (step_records[0], step_records[0]), tmp_path / "steps.jsonl"
        )
    with pytest.raises(ValueError, match="Duplicate"):
        write_ground_service_run_manifest((run_record, run_record), tmp_path / "runs.jsonl")


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_replay_rejects_missing_and_extra_g4_steps(mutation: str) -> None:
    values = verified_context()
    kwargs = replay_kwargs(values)
    if mutation == "missing":
        kwargs["step_records"] = values[-2][:-1]
    else:
        source = values[-2][-1]
        extra_metrics_values = asdict(source.metrics)
        extra_metrics_values["timestep_index"] += 1
        from satnet.ground.service_metrics import GroundServiceStepMetrics, _compute_step_metrics_hash

        extra_metrics_values["step_metrics_hash"] = _compute_step_metrics_hash(extra_metrics_values)
        metrics = GroundServiceStepMetrics(**extra_metrics_values)
        extra = GroundServiceStepRecord(
            ground_service_step_schema_version=GROUND_SERVICE_STEP_SCHEMA_VERSION,
            run_id=source.run_id,
            metrics=metrics,
            record_hash=_step_record_hash(run_id=source.run_id, metrics=metrics),
        )
        kwargs["step_records"] = values[-2] + (extra,)
    with pytest.raises(ValueError, match="keys mismatch"):
        replay_ground_service_records(**kwargs)


def test_replay_rejects_missing_and_duplicate_run_summaries() -> None:
    values = verified_context()
    kwargs = replay_kwargs(values)
    kwargs["run_records"] = ()
    with pytest.raises(ValueError, match="exactly one"):
        replay_ground_service_records(**kwargs)
    kwargs["run_records"] = (values[-1], values[-1])
    with pytest.raises(ValueError, match="exactly one"):
        replay_ground_service_records(**kwargs)


def test_cross_stage_g4_run_mismatches_fail() -> None:
    values = verified_context()
    kwargs = replay_kwargs(values)
    wrong_steps = tuple(
        GroundServiceStepRecord(
            ground_service_step_schema_version=record.ground_service_step_schema_version,
            run_id=6,
            metrics=record.metrics,
            record_hash=_step_record_hash(run_id=6, metrics=record.metrics),
        )
        for record in values[-2]
    )
    kwargs["step_records"] = wrong_steps
    with pytest.raises(ValueError, match="step-record run ID"):
        replay_ground_service_records(**kwargs)
    summary_values = asdict(values[-1].summary)
    summary_values["run_id"] = 6
    wrong_summary = GroundServiceRunSummary(**summary_values)
    wrong_run = GroundServiceRunRecord(
        ground_service_run_schema_version=values[-1].ground_service_run_schema_version,
        run_id=6,
        summary=wrong_summary,
        record_hash=_run_record_hash(run_id=6, summary=wrong_summary),
    )
    kwargs = replay_kwargs(values)
    kwargs["run_records"] = (wrong_run,)
    with pytest.raises(ValueError, match="run-record run ID"):
        replay_ground_service_records(**kwargs)


def test_self_consistent_arbitrary_g3_snapshot_is_rejected_by_verified_replay() -> None:
    values = verified_context()
    snapshot = values[6][0]
    arbitrary = create_integrated_graph_snapshot(
        timestep_index=snapshot.timestep_index,
        timestamp_utc=snapshot.timestamp_utc,
        satellite_config_hash=snapshot.satellite_config_hash,
        ground_design_hash=snapshot.ground_design_hash,
        visibility_policy_hash=snapshot.visibility_policy_hash,
        visibility_snapshot_hash=snapshot.visibility_snapshot_hash,
        graph_attributes=snapshot.canonical_graph_attributes,
        nodes=snapshot.canonical_nodes,
        edges=tuple(edge for edge in snapshot.canonical_edges if edge != snapshot.canonical_edges[0]),
    )
    pure = compute_ground_service_step(
        integrated_snapshot=arbitrary,
        ground_design=values[3],
        catalog=values[2],
        configured_satellite_count=values[0].total_satellites,
        policy=values[8],
    )
    assert pure.integrated_graph_hash == arbitrary.graph_hash
    arbitrary_records = list(values[7])
    arbitrary_records[0] = make_integrated_graph_record(
        ground_design=values[3], snapshot=arbitrary
    )
    kwargs = replay_kwargs(values)
    kwargs["integrated_records"] = tuple(arbitrary_records)
    with pytest.raises(ValueError, match="Integrated graph replay mismatch"):
        replay_ground_service_records(**kwargs)


def test_replay_does_not_mutate_upstream_records_or_hashes() -> None:
    values = verified_context()
    before = (
        values[2].catalog_hash,
        values[3].selection_hash,
        values[3].ground_design_hash,
        tuple(record.snapshot_hash for record in values[5]),
        tuple(record.record_hash for record in values[5]),
        tuple(record.graph_hash for record in values[7]),
        tuple(record.record_hash for record in values[7]),
        values[0].config_hash(),
        values[1].failed_nodes.copy(),
        values[1].failed_edges.copy(),
    )
    replay_ground_service_records(**replay_kwargs(values))
    after = (
        values[2].catalog_hash,
        values[3].selection_hash,
        values[3].ground_design_hash,
        tuple(record.snapshot_hash for record in values[5]),
        tuple(record.record_hash for record in values[5]),
        tuple(record.graph_hash for record in values[7]),
        tuple(record.record_hash for record in values[7]),
        values[0].config_hash(),
        values[1].failed_nodes.copy(),
        values[1].failed_edges.copy(),
    )
    assert before == after
