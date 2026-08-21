from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from satnet.ground.catalog import (
    GroundStation,
    GroundStationCatalog,
    GroundStationClass,
    load_ground_station_catalog,
)
from satnet.ground.selection import (
    GROUND_STATION_SELECTION_VERSION,
    MAX_SELECTION_SEED,
    GroundSegmentDisabledConfig,
    GroundSegmentEnabledConfig,
    build_region_balanced_order,
    select_ground_stations,
)

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "ground_segment"
    / "synthetic_ground_station_catalog.csv"
)


def config(civilian: int = 3, government: int = 0, military: int = 0, seed: int = 42):
    return GroundSegmentEnabledConfig(
        civilian_count=civilian,
        government_count=government,
        military_count=military,
        station_selection_seed=seed,
    )


def station(
    station_id: str,
    station_class: GroundStationClass = GroundStationClass.CIVILIAN,
    region: str = "region_alpha",
    enabled: bool = True,
) -> GroundStation:
    return GroundStation(
        station_id=station_id,
        name=f"Synthetic {station_id}",
        station_class=station_class,
        latitude_deg=0.0,
        longitude_deg=0.0,
        altitude_m=0.0,
        region=region,
        country_code="ZZ",
        enabled=enabled,
    )


def test_disabled_config_is_explicit() -> None:
    disabled = GroundSegmentDisabledConfig()
    assert disabled.enabled is False
    assert disabled.total_ground_station_count == 0


@pytest.mark.parametrize("field_name", ["civilian_count", "government_count", "military_count"])
@pytest.mark.parametrize("value", [True, False, "1", 1.0, None])
def test_enabled_config_counts_require_exact_integers(field_name: str, value: object) -> None:
    values = {
        "civilian_count": 1,
        "government_count": 0,
        "military_count": 0,
        "station_selection_seed": 42,
    }
    values[field_name] = value
    with pytest.raises(TypeError, match=field_name):
        GroundSegmentEnabledConfig(**values)


@pytest.mark.parametrize("field_name", ["civilian_count", "government_count", "military_count"])
def test_enabled_config_rejects_negative_counts(field_name: str) -> None:
    values = {
        "civilian_count": 1,
        "government_count": 0,
        "military_count": 0,
        "station_selection_seed": 42,
    }
    values[field_name] = -1
    with pytest.raises(ValueError, match=field_name):
        GroundSegmentEnabledConfig(**values)


def test_enabled_config_rejects_zero_total() -> None:
    with pytest.raises(ValueError, match="at least one"):
        config(civilian=0, government=0, military=0)


@pytest.mark.parametrize("seed", [True, False, "42", 42.0, None])
def test_selection_seed_requires_exact_integer(seed: object) -> None:
    with pytest.raises(TypeError, match="station_selection_seed"):
        config(seed=seed)


@pytest.mark.parametrize("seed", [-1, MAX_SELECTION_SEED + 1])
def test_selection_seed_range_enforced(seed: int) -> None:
    with pytest.raises(ValueError, match="station_selection_seed"):
        config(seed=seed)


def test_config_total_count_and_class_counts() -> None:
    enabled = config(civilian=6, government=3, military=3)
    assert enabled.total_ground_station_count == 12
    assert enabled.count_for(GroundStationClass.CIVILIAN) == 6
    assert enabled.count_for(GroundStationClass.GOVERNMENT) == 3
    assert enabled.count_for(GroundStationClass.MILITARY) == 3


def test_same_catalog_and_config_are_deterministic() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    first = select_ground_stations(catalog=catalog, config=config(6, 3, 3))
    second = select_ground_stations(catalog=catalog, config=config(6, 3, 3))
    assert first == second
    assert first.selection_version == GROUND_STATION_SELECTION_VERSION


def test_catalog_input_order_does_not_affect_selection() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    reversed_catalog = GroundStationCatalog(tuple(reversed(catalog.stations)))
    first = select_ground_stations(catalog=catalog, config=config(6, 3, 3))
    second = select_ground_stations(catalog=reversed_catalog, config=config(6, 3, 3))
    assert first == second


def test_nested_civilian_prefixes() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    selections = {
        count: select_ground_stations(catalog=catalog, config=config(civilian=count))
        for count in (3, 10, 12, 20)
    }
    assert selections[10].civilian_station_ids[:3] == selections[3].civilian_station_ids
    assert selections[12].civilian_station_ids[:10] == selections[10].civilian_station_ids
    assert selections[20].civilian_station_ids[:12] == selections[12].civilian_station_ids


def test_mixed_composition_has_exact_canonical_counts() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    result = select_ground_stations(catalog=catalog, config=config(6, 3, 3))
    assert len(result.civilian_station_ids) == 6
    assert len(result.government_station_ids) == 3
    assert len(result.military_station_ids) == 3
    assert len(result.selected_station_ids) == 12
    assert len(set(result.selected_station_ids)) == 12
    assert result.selected_station_ids == (
        result.civilian_station_ids
        + result.government_station_ids
        + result.military_station_ids
    )


def test_selected_stations_are_enabled_and_have_expected_classes() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    records = catalog.by_id()
    result = select_ground_stations(catalog=catalog, config=config(6, 3, 3))
    for identifier in result.civilian_station_ids:
        assert records[identifier].enabled
        assert records[identifier].station_class is GroundStationClass.CIVILIAN
    for identifier in result.government_station_ids:
        assert records[identifier].enabled
        assert records[identifier].station_class is GroundStationClass.GOVERNMENT
    for identifier in result.military_station_ids:
        assert records[identifier].enabled
        assert records[identifier].station_class is GroundStationClass.MILITARY


def test_excess_requested_count_fails_without_reduction() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    with pytest.raises(ValueError, match="only 6 enabled"):
        select_ground_stations(catalog=catalog, config=config(government=7))


def test_different_seeds_change_complete_order_for_fixture() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    first = build_region_balanced_order(
        catalog.stations,
        station_class=GroundStationClass.CIVILIAN,
        selection_seed=1,
    )
    second = build_region_balanced_order(
        catalog.stations,
        station_class=GroundStationClass.CIVILIAN,
        selection_seed=2,
    )
    assert tuple(item.station_id for item in first) != tuple(item.station_id for item in second)
    assert {item.station_id for item in first} == {item.station_id for item in second}


def test_different_seeds_with_same_ids_have_different_selection_hashes() -> None:
    catalog = GroundStationCatalog((station("CIV_ONLY_001"),))
    first = select_ground_stations(catalog=catalog, config=config(civilian=1, seed=1))
    second = select_ground_stations(catalog=catalog, config=config(civilian=1, seed=2))
    assert first.selected_station_ids == second.selected_station_ids
    assert first.selection_hash != second.selection_hash


def test_disabled_catalog_addition_preserves_order_and_ids_but_changes_hashes() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    disabled = station("CIV_EXTRA_DISABLED", enabled=False)
    changed_catalog = GroundStationCatalog(catalog.stations + (disabled,))
    original_order = build_region_balanced_order(
        catalog.stations,
        station_class=GroundStationClass.CIVILIAN,
        selection_seed=42,
    )
    changed_order = build_region_balanced_order(
        changed_catalog.stations,
        station_class=GroundStationClass.CIVILIAN,
        selection_seed=42,
    )
    original = select_ground_stations(catalog=catalog, config=config(civilian=10))
    changed = select_ground_stations(catalog=changed_catalog, config=config(civilian=10))
    assert tuple(item.station_id for item in original_order) == tuple(
        item.station_id for item in changed_order
    )
    assert original.selected_station_ids == changed.selected_station_ids
    assert original.catalog_hash != changed.catalog_hash
    assert original.selection_hash != changed.selection_hash


def test_enabled_catalog_addition_changes_complete_eligible_order() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    changed_catalog = GroundStationCatalog(
        catalog.stations + (station("CIV_EXTRA_ENABLED", region="region_beta"),)
    )
    original = build_region_balanced_order(
        catalog.stations,
        station_class=GroundStationClass.CIVILIAN,
        selection_seed=42,
    )
    changed = build_region_balanced_order(
        changed_catalog.stations,
        station_class=GroundStationClass.CIVILIAN,
        selection_seed=42,
    )
    assert tuple(item.station_id for item in original) != tuple(item.station_id for item in changed)


def test_modifying_or_removing_eligible_station_may_change_prefixes() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    original = select_ground_stations(catalog=catalog, config=config(civilian=10))
    removed_id = original.civilian_station_ids[0]
    changed_catalog = GroundStationCatalog(
        tuple(item for item in catalog.stations if item.station_id != removed_id)
    )
    changed = select_ground_stations(catalog=changed_catalog, config=config(civilian=10))
    assert changed.civilian_station_ids != original.civilian_station_ids


def test_region_balance_with_uneven_pool_sizes() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    order = build_region_balanced_order(
        catalog.stations,
        station_class=GroundStationClass.CIVILIAN,
        selection_seed=42,
    )
    pool_sizes = Counter(item.region for item in order)
    selected_counts: Counter[str] = Counter()
    for item in order:
        selected_counts[item.region] += 1
        nonexhausted = [
            region for region, size in pool_sizes.items() if selected_counts[region] < size
        ]
        if nonexhausted:
            active_counts = [selected_counts[region] for region in nonexhausted]
            assert max(active_counts) - min(active_counts) <= 1
    assert sorted(pool_sizes.values()) == [3, 4, 5, 8]


def test_first_prefix_represents_distinct_regions() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    order = build_region_balanced_order(
        catalog.stations,
        station_class=GroundStationClass.CIVILIAN,
        selection_seed=42,
    )
    assert len({item.region for item in order[:4]}) == 4


def test_selection_hash_changes_with_requested_count() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    first = select_ground_stations(catalog=catalog, config=config(civilian=3))
    second = select_ground_stations(catalog=catalog, config=config(civilian=10))
    assert first.selection_hash != second.selection_hash


def test_selection_has_no_duplicate_ids() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    selected = select_ground_stations(catalog=catalog, config=config(20, 6, 6))
    assert len(selected.selected_station_ids) == len(set(selected.selected_station_ids))
