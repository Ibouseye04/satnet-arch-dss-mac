from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from satnet.experiments.final_generation.artifacts import (
    make_run_result,
    make_satellite_artifact,
    make_scientific_inventory,
    make_target_artifact,
    read_satellite_artifact,
    validate_run_result,
    validate_scientific_inventory,
    validate_target_artifact,
    write_satellite_artifact,
)
from satnet.experiments.final_generation.constants import RUN_FILES, SCIENTIFIC_FILE_KEYS, TARGET_FIELDS
from satnet.experiments.final_generation.contract import validate_frozen_contract
from satnet.experiments.final_generation.io import atomic_write_json
from satnet.experiments.final_generation.mapping import map_all_runs
from satnet.simulation.tier1_rollout import (
    Tier1FailureRealization,
    Tier1RolloutStep,
    Tier1RolloutSummary,
)


def _mapping():
    return map_all_runs(validate_frozen_contract(compare_tag_blobs=False))[0]


def _satellite_values(mapping):
    config = mapping.satellite_config
    step = Tier1RolloutStep(
        t=0, num_nodes=config.total_satellites, num_edges=0, num_components=1,
        gcc_size=config.total_satellites, gcc_frac=1.0, gcc_frac_original=1.0,
        gcc_frac_surviving=1.0, partitioned=0,
    )
    steps = [Tier1RolloutStep(**{**step.__dict__, "t": index}) for index in range(config.num_steps)]
    summary = Tier1RolloutSummary(
        gcc_frac_min=1.0, gcc_frac_mean=1.0, gcc_frac_min_original=1.0,
        gcc_frac_mean_original=1.0, gcc_frac_min_surviving=1.0,
        gcc_frac_mean_surviving=1.0, partition_fraction=0.0, partition_any=0,
        max_partition_streak=0, max_partition_streak_seconds=0,
        max_partition_streak_fraction=0.0, num_steps=config.num_steps,
        config_hash=config.config_hash(), failure_model=config.failure_model,
    )
    return config, steps, summary, Tier1FailureRealization(set(), set())


def _summary() -> SimpleNamespace:
    return SimpleNamespace(
        overall_threshold_breach_any=False,
        ground_threshold_breach_any=True,
        space_threshold_breach_any=False,
        failure_adjusted_overall_service_fraction_mean=0.75,
        failure_adjusted_overall_service_fraction_min=0.5,
        failure_adjusted_ground_service_fraction_min=0.5,
        space_gcc_fraction_original_min=1.0,
        ground_service_loss_due_to_failures_max=0.25,
    )


def test_final_satellite_artifact_round_trip_and_identity(tmp_path: Path) -> None:
    mapping = _mapping()
    config, steps, summary, failures = _satellite_values(mapping)
    artifact = make_satellite_artifact(
        design=mapping.design, run=mapping.run, config=config, steps=steps,
        summary=summary, failure_realization=failures,
    )
    assert artifact["identity_domain"] == "satnet_final_integrated_dataset_satellite_artifact"
    assert "pilot" not in artifact["identity_domain"]
    assert artifact["expected_configuration_hash"] == artifact["actual_configuration_hash"]
    assert artifact["failed_nodes"] == []
    assert artifact["failed_edges"] == []
    assert "satellite_artifact_hash" not in {
        key for key in artifact if key != "satellite_artifact_hash"
    }
    path = tmp_path / "satellite.json"
    write_satellite_artifact(path, artifact)
    persisted, read_config, read_steps, read_summary, read_failures = read_satellite_artifact(path)
    assert persisted == artifact
    assert read_config == config
    assert read_steps == tuple(steps)
    assert read_summary == summary
    assert read_failures == failures


def test_target_exact_fields_types_and_bundle_exclusion() -> None:
    mapping = _mapping()
    target = make_target_artifact(
        design=mapping.design, run=mapping.run, g5_summary=_summary()
    )
    validate_target_artifact(target)
    assert all(name in target for name in TARGET_FIELDS)
    assert "contract_bundle_hash" not in target
    assert type(target["overall_threshold_breach_any"]) is bool
    invalid = dict(target)
    invalid["overall_threshold_breach_any"] = 0
    with pytest.raises(TypeError):
        validate_target_artifact(invalid)
    invalid = dict(target)
    invalid["contract_bundle_hash"] = "0" * 64
    with pytest.raises(ValueError):
        validate_target_artifact(invalid)


def test_inventory_includes_target_and_excludes_itself_result_and_operational(tmp_path: Path) -> None:
    for key in SCIENTIFIC_FILE_KEYS:
        path = tmp_path / RUN_FILES[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{key}\n".encode())
    (tmp_path / RUN_FILES["result"]).write_text("result\n", encoding="utf-8")
    (tmp_path / "operational").mkdir()
    (tmp_path / "operational" / "attempt.json").write_text("attempt\n", encoding="utf-8")
    inventory = make_scientific_inventory(tmp_path)
    validate_scientific_inventory(tmp_path, inventory)
    paths = {record["path"] for record in inventory["artifacts"]}
    assert RUN_FILES["target"] in paths
    assert RUN_FILES["inventory"] not in paths
    assert RUN_FILES["result"] not in paths
    assert not any(path.startswith("operational/") for path in paths)
    modified = dict(inventory)
    modified["scientific_inventory_hash"] = "0" * 64
    with pytest.raises(ValueError, match="inventory mismatch"):
        validate_scientific_inventory(tmp_path, modified)


def test_result_identity_and_bundle_hash_rejected() -> None:
    mapping = _mapping()
    target = make_target_artifact(design=mapping.design, run=mapping.run, g5_summary=_summary())
    inventory = {"scientific_inventory_hash": "1" * 64}
    result = make_run_result(
        design=mapping.design, run=mapping.run, inventory=inventory, target=target
    )
    validate_run_result(result)
    invalid = dict(result)
    invalid["contract_bundle_hash"] = "2" * 64
    with pytest.raises(ValueError):
        validate_run_result(invalid)
    invalid = dict(result)
    invalid["run_result_hash"] = "3" * 64
    with pytest.raises(ValueError, match="self-hash"):
        validate_run_result(invalid)
