from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from satnet.ground.catalog import load_ground_station_catalog
from satnet.ground.integrated_builder import build_integrated_ground_graph
from satnet.ground.integrated_persistence import (
    make_integrated_graph_record,
    read_integrated_graph_manifest,
    replay_integrated_graph_records,
    validate_integrated_records_against_sources,
    write_integrated_graph_manifest,
)
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.position_adapter import reconstruct_operational_satellite_position_sequence
from satnet.ground.satellite_graph_adapter import reconstruct_operational_satellite_graph_sequence
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.ground.visibility import (
    GroundVisibilityPolicy,
    GroundVisibilitySnapshot,
    evaluate_ground_design_visibility_sequence,
)
from satnet.ground.visibility_persistence import make_ground_visibility_record
from satnet.simulation.tier1_rollout import Tier1FailureRealization, Tier1RolloutConfig

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "ground_segment"
    / "synthetic_ground_station_catalog.csv"
)


def context(*, all_failed: bool = False):
    config = Tier1RolloutConfig(
        num_planes=2,
        sats_per_plane=10,
        duration_minutes=1,
        step_seconds=60,
        node_failure_prob=0.0,
        edge_failure_prob=0.0,
        seed=404,
    )
    failures = Tier1FailureRealization(
        failed_nodes=set(range(config.total_satellites)) if all_failed else set(),
        failed_edges=set(),
    )
    catalog = load_ground_station_catalog(FIXTURE)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(2, 0, 0, 42),
    )
    ground_design = make_enabled_ground_design_record(
        run_id=5,
        satellite_config_hash=config.config_hash(),
        selection=selection,
    )
    policy = GroundVisibilityPolicy(0.0)
    position_sources = reconstruct_operational_satellite_position_sequence(
        satellite_config=config,
        failure_realization=failures,
    )
    graph_sources = reconstruct_operational_satellite_graph_sequence(
        satellite_config=config,
        failure_realization=failures,
    )
    visibility_snapshots = evaluate_ground_design_visibility_sequence(
        ground_design=ground_design,
        catalog=catalog,
        satellite_sequence=position_sources,
        policy=policy,
    )
    visibility_records = tuple(
        make_ground_visibility_record(run_id=5, snapshot=snapshot)
        for snapshot in visibility_snapshots
    )
    integrated_snapshots = tuple(
        build_integrated_ground_graph(
            ground_design=ground_design,
            catalog=catalog,
            satellite_graph_snapshot=graph_source,
            verified_visibility_snapshot=visibility,
        )
        for graph_source, visibility in zip(
            graph_sources, visibility_snapshots, strict=True
        )
    )
    records = tuple(
        make_integrated_graph_record(ground_design=ground_design, snapshot=snapshot)
        for snapshot in integrated_snapshots
    )
    return (
        config,
        failures,
        catalog,
        ground_design,
        policy,
        graph_sources,
        visibility_records,
        integrated_snapshots,
        records,
    )


def write_object(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def test_integrated_manifest_round_trip_and_order(tmp_path: Path) -> None:
    *_, records = context()
    path = tmp_path / "integrated.jsonl"
    write_integrated_graph_manifest(tuple(reversed(records)), path)
    assert read_integrated_graph_manifest(path) == records
    values = [json.loads(line) for line in path.read_text().splitlines()]
    assert [value["timestep_index"] for value in values] == [0, 1]
    assert all(list(value) == sorted(value) for value in values)
    assert path.read_bytes().endswith(b"\n")


def test_integrated_writer_refuses_overwrite(tmp_path: Path) -> None:
    *_, records = context()
    path = tmp_path / "integrated.jsonl"
    write_integrated_graph_manifest(records, path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_integrated_graph_manifest(records, path)
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("content", ["", "\n", "{}\n\n{}\n"])
def test_integrated_reader_rejects_empty_files_and_lines(tmp_path: Path, content: str) -> None:
    path = tmp_path / "integrated.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_integrated_graph_manifest(path)


def test_duplicate_json_keys_fail(tmp_path: Path) -> None:
    path = tmp_path / "integrated.jsonl"
    path.write_text('{"run_id":1,"run_id":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        read_integrated_graph_manifest(path)


def test_unknown_missing_and_exact_integer_fields_fail(tmp_path: Path) -> None:
    *_, records = context()
    value = records[0].to_manifest_object()
    path = tmp_path / "integrated.jsonl"
    extra = dict(value)
    extra["unexpected"] = 1
    write_object(path, extra)
    with pytest.raises(ValueError, match="unknown=.*unexpected"):
        read_integrated_graph_manifest(path)
    missing = dict(value)
    del missing["graph_hash"]
    write_object(path, missing)
    with pytest.raises(ValueError, match="missing=.*graph_hash"):
        read_integrated_graph_manifest(path)
    invalid = dict(value)
    invalid["run_id"] = True
    write_object(path, invalid)
    with pytest.raises(TypeError, match="run_id"):
        read_integrated_graph_manifest(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_ground_node",
        "missing_satellite_node",
        "missing_edge",
        "modified_satellite_attribute",
        "modified_isl_attribute",
        "wrong_edge_kind",
        "wrong_node_kind",
        "wrong_timestamp",
        "wrong_visibility_hash",
        "wrong_graph_hash",
        "wrong_record_hash",
    ],
)
def test_integrated_corruption_fails_on_read(tmp_path: Path, mutation: str) -> None:
    *_, records = context()
    value = deepcopy(records[0].to_manifest_object())
    if mutation == "missing_ground_node":
        value["canonical_nodes"] = [
            node
            for node in value["canonical_nodes"]
            if node["node_ref"]["kind"] != "ground_station"
        ]
    elif mutation == "missing_satellite_node":
        value["canonical_nodes"] = value["canonical_nodes"][1:]
    elif mutation == "missing_edge":
        value["canonical_edges"] = value["canonical_edges"][1:]
    elif mutation == "modified_satellite_attribute":
        value["canonical_nodes"][0]["attributes"][0]["value"] = "changed"
    elif mutation == "modified_isl_attribute":
        isl = next(
            edge for edge in value["canonical_edges"] if edge["edge_kind"] == "inter_satellite"
        )
        isl["attributes"][0]["value"] = "999"
    elif mutation == "wrong_edge_kind":
        value["canonical_edges"][0]["edge_kind"] = "satellite_ground"
    elif mutation == "wrong_node_kind":
        value["canonical_nodes"][0]["node_ref"]["kind"] = "ground_station"
    elif mutation == "wrong_timestamp":
        value["timestamp_utc"] = "2026-07-16T12:00:01.000000Z"
    elif mutation == "wrong_visibility_hash":
        value["visibility_snapshot_hash"] = "e" * 64
    elif mutation == "wrong_graph_hash":
        value["graph_hash"] = "e" * 64
    elif mutation == "wrong_record_hash":
        value["record_hash"] = "e" * 64
    path = tmp_path / "integrated.jsonl"
    write_object(path, value)
    with pytest.raises((TypeError, ValueError)):
        read_integrated_graph_manifest(path)


def test_collection_completeness_and_run_binding() -> None:
    (
        _,
        _,
        _,
        ground_design,
        _,
        graph_sources,
        visibility_records,
        _,
        records,
    ) = context()
    validate_integrated_records_against_sources(
        records=records,
        source_snapshots=graph_sources,
        ground_design=ground_design,
        visibility_records=visibility_records,
    )
    with pytest.raises(ValueError, match="missing"):
        validate_integrated_records_against_sources(
            records=records[:1],
            source_snapshots=graph_sources,
            ground_design=ground_design,
            visibility_records=visibility_records,
        )
    wrong_design = replace(ground_design, run_id=6)
    wrong_run = tuple(
        make_integrated_graph_record(ground_design=wrong_design, snapshot=record.to_snapshot())
        for record in records
    )
    with pytest.raises(ValueError, match="run ID"):
        validate_integrated_records_against_sources(
            records=wrong_run,
            source_snapshots=graph_sources,
            ground_design=ground_design,
            visibility_records=visibility_records,
        )


def test_exact_integrated_replay_and_all_failed_replay() -> None:
    for all_failed in (False, True):
        (
            config,
            failures,
            catalog,
            ground_design,
            policy,
            _,
            visibility_records,
            snapshots,
            records,
        ) = context(all_failed=all_failed)
        replayed = replay_integrated_graph_records(
            records=records,
            satellite_config=config,
            failure_realization=failures,
            ground_design=ground_design,
            catalog=catalog,
            visibility_policy=policy,
            visibility_records=visibility_records,
        )
        assert replayed == snapshots
        if all_failed:
            assert all(snapshot.satellite_node_count == 0 for snapshot in replayed)
            assert all(snapshot.ground_node_count == 2 for snapshot in replayed)


def test_wrong_g2_record_run_id_fails_canonical_replay() -> None:
    (
        config,
        failures,
        catalog,
        ground_design,
        policy,
        _,
        visibility_records,
        _,
        records,
    ) = context()
    wrong_visibility = tuple(
        make_ground_visibility_record(
            run_id=6,
            snapshot=GroundVisibilitySnapshot(
                timestep_index=record.timestep_index,
                timestamp_utc=record.timestamp_utc,
                satellite_config_hash=record.satellite_config_hash,
                ground_design_hash=record.ground_design_hash,
                visibility_policy_hash=record.visibility_policy_hash,
                visibility_model_version=record.visibility_model_version,
                frame_contract_version=record.frame_contract_version,
                wgs84_model_version=record.wgs84_model_version,
                link_observations=record.link_observations,
                visible_links=tuple(
                    observation
                    for observation in record.link_observations
                    if observation.is_visible
                ),
                visible_satellite_ids_by_station=record.visible_satellite_ids_by_station,
                snapshot_hash=record.snapshot_hash,
            ),
        )
        for record in visibility_records
    )
    with pytest.raises(ValueError):
        replay_integrated_graph_records(
            records=records,
            satellite_config=config,
            failure_realization=failures,
            ground_design=ground_design,
            catalog=catalog,
            visibility_policy=policy,
            visibility_records=wrong_visibility,
        )
