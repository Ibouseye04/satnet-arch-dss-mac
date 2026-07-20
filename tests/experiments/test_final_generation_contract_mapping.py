from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from satnet.experiments.final_generation.constants import (
    CATALOG_HASH,
    CONTRACT_BUNDLE_HASH,
    CONTRACT_SPEC_HASH,
    FROZEN_ARTIFACTS,
    QUALIFICATION_RUN_IDS,
)
from satnet.experiments.final_generation.contract import (
    ensure_mode_root,
    mode_marker,
    paths_intersect,
    validate_catalog,
    validate_frozen_contract,
    validate_output_root,
    validate_science_dependencies,
)
from satnet.experiments.final_generation.io import (
    atomic_write_json,
    canonical_relative_path,
    parse_canonical_float,
    read_canonical_json,
)
from satnet.experiments.final_generation.mapping import map_all_runs, map_frozen_run
from satnet.simulation.tier1_rollout import Tier1RolloutConfig


def test_frozen_contract_catalog_science_and_all_500_mappings_validate() -> None:
    contract = validate_frozen_contract(compare_tag_blobs=False)
    catalog = validate_catalog()
    validate_science_dependencies(contract["specification"])
    mappings = map_all_runs(contract)
    assert contract["contract_spec_hash"] == CONTRACT_SPEC_HASH
    assert contract["contract_bundle_hash"] == CONTRACT_BUNDLE_HASH
    assert catalog.catalog_hash == CATALOG_HASH
    assert len(FROZEN_ARTIFACTS) == 11
    assert len(mappings) == 500
    assert [mapping.run_id for mapping in mappings] == list(range(500))
    assert [mappings[index].run_key for index in (0, 4, 5, 35, 200, 499)] == [
        "D000-R00", "D000-R04", "D001-R00", "D007-R00", "D040-R00", "D099-R04"
    ]
    assert not any(field.name == "run_index" for field in fields(Tier1RolloutConfig))


def test_mapping_rejects_seed_and_identity_substitution() -> None:
    contract = validate_frozen_contract(compare_tag_blobs=False)
    design = dict(contract["designs"][0])
    run = dict(contract["runs"][0])
    run["ground_selection_seed"] += 1
    with pytest.raises(ValueError, match="ground-selection seed"):
        map_frozen_run(design, run)
    run = dict(contract["runs"][0])
    run["run_key"] = "D000-R99"
    with pytest.raises(ValueError, match="run_key"):
        map_frozen_run(design, run)


def test_qualification_set_is_exact() -> None:
    assert QUALIFICATION_RUN_IDS == (
        0, 1, 2, 3, 4, 35, 36, 37, 38, 39, 200, 201, 202, 203, 204
    )


def test_canonical_float_reader_rejects_noncanonical_value() -> None:
    assert parse_canonical_float("0.5", "value") == 0.5
    with pytest.raises(ValueError, match="not a canonical"):
        parse_canonical_float("0.500", "value")
    with pytest.raises(TypeError):
        parse_canonical_float(0.5, "value")


@pytest.mark.parametrize("value", ["/absolute", "a\\b", "a/../b", "a/./b", ""])
def test_inventory_path_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(ValueError):
        canonical_relative_path(value)


def test_mode_marker_roles_and_noncanonical_marker_rejected(tmp_path: Path) -> None:
    root = tmp_path / "qualification"
    ensure_mode_root(root, "qualification", create=True)
    assert read_canonical_json(root / "execution_mode.json") == mode_marker("qualification")
    with pytest.raises(ValueError, match="mode marker mismatch"):
        ensure_mode_root(root, "qualification_replay", create=False)
    marker = mode_marker("qualification")
    atomic_write_json(root / "execution_mode.json", marker, overwrite=True)
    (root / "execution_mode.json").write_text("{\n}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ensure_mode_root(root, "qualification", create=False)


def test_equal_and_nested_output_roots_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = first / "nested"
    assert paths_intersect(first.resolve(), second.resolve())
    with pytest.raises(ValueError, match="intersect"):
        validate_output_root(first, other_roots=(first,))
    with pytest.raises(ValueError, match="intersect"):
        validate_output_root(second, other_roots=(first,))
