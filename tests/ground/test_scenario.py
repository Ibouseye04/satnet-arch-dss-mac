from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from satnet.ground.catalog import load_ground_station_catalog
from satnet.ground.persistence import (
    make_disabled_ground_design_record,
    make_enabled_ground_design_record,
)
from satnet.ground.scenario import ScenarioDesign
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.simulation.tier1_rollout import Tier1RolloutConfig

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "ground_segment"
    / "synthetic_ground_station_catalog.csv"
)


def satellite_config() -> Tier1RolloutConfig:
    return Tier1RolloutConfig(
        num_planes=3,
        sats_per_plane=4,
        duration_minutes=1,
        step_seconds=60,
        seed=77,
    )


def test_disabled_scenario_preserves_exact_satellite_object_and_hash() -> None:
    config = satellite_config()
    original_hash = config.config_hash()
    ground = make_disabled_ground_design_record(
        run_id=0,
        satellite_config_hash=original_hash,
    )
    scenario = ScenarioDesign(config, ground)
    assert scenario.satellite_config is config
    assert scenario.satellite_config.config_hash() == original_hash
    assert len(scenario.scenario_design_hash) == 64


def test_enabled_scenario_preserves_exact_satellite_object_and_hash() -> None:
    config = satellite_config()
    original_hash = config.config_hash()
    catalog = load_ground_station_catalog(FIXTURE)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(6, 3, 3, 42),
    )
    ground = make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash=original_hash,
        selection=selection,
    )
    scenario = ScenarioDesign(config, ground)
    assert scenario.satellite_config is config
    assert scenario.ground_design is ground
    assert scenario.satellite_config.config_hash() == original_hash


def test_scenario_rejects_satellite_hash_mismatch() -> None:
    config = satellite_config()
    ground = make_disabled_ground_design_record(
        run_id=0,
        satellite_config_hash="a" * 64,
    )
    with pytest.raises(ValueError, match="does not match"):
        ScenarioDesign(config, ground)


def test_scenario_hash_changes_with_ground_design_without_changing_satellite_hash() -> None:
    config = satellite_config()
    satellite_hash = config.config_hash()
    disabled = make_disabled_ground_design_record(
        run_id=0,
        satellite_config_hash=satellite_hash,
    )
    catalog = load_ground_station_catalog(FIXTURE)
    selection = select_ground_stations(
        catalog=catalog,
        config=GroundSegmentEnabledConfig(1, 0, 0, 42),
    )
    enabled = make_enabled_ground_design_record(
        run_id=0,
        satellite_config_hash=satellite_hash,
        selection=selection,
    )
    disabled_scenario = ScenarioDesign(config, disabled)
    enabled_scenario = ScenarioDesign(config, enabled)
    assert disabled_scenario.scenario_design_hash != enabled_scenario.scenario_design_hash
    assert config.config_hash() == satellite_hash


def test_scenario_hash_is_run_id_independent() -> None:
    config = satellite_config()
    first_ground = make_disabled_ground_design_record(
        run_id=0,
        satellite_config_hash=config.config_hash(),
    )
    second_ground = replace(first_ground, run_id=10)
    assert ScenarioDesign(config, first_ground).scenario_design_hash == ScenarioDesign(
        config, second_ground
    ).scenario_design_hash
