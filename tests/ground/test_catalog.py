from __future__ import annotations

import csv
from dataclasses import replace
import math
from pathlib import Path

import pytest

from satnet.ground.canonical import canonical_float_string
from satnet.ground.catalog import (
    CATALOG_COLUMNS,
    GroundStation,
    GroundStationCatalog,
    GroundStationClass,
    load_ground_station_catalog,
    validate_production_catalog_readiness,
)

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "ground_segment"
    / "synthetic_ground_station_catalog.csv"
)


def station(**overrides) -> GroundStation:
    values = {
        "station_id": "CIV_TEST_001",
        "name": "Synthetic Station",
        "station_class": GroundStationClass.CIVILIAN,
        "latitude_deg": 10.0,
        "longitude_deg": 20.0,
        "altitude_m": 100.0,
        "region": "region_alpha",
        "country_code": "ZZ",
        "enabled": True,
    }
    values.update(overrides)
    return GroundStation(**values)


def write_rows(path: Path, rows: list[dict[str, str]], headers=CATALOG_COLUMNS) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def valid_row(**overrides) -> dict[str, str]:
    values = {
        "station_id": "CIV_TEST_001",
        "name": "Synthetic Station",
        "station_class": "civilian",
        "latitude_deg": "10.0",
        "longitude_deg": "20.0",
        "altitude_m": "100.0",
        "region": "region_alpha",
        "country_code": "ZZ",
        "enabled": "true",
    }
    values.update(overrides)
    return values


def test_valid_fixture_loads_in_canonical_order() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    identifiers = [record.station_id for record in catalog.stations]
    assert identifiers == sorted(identifiers)
    assert len(catalog.catalog_hash) == 64
    assert len(catalog.eligible(GroundStationClass.CIVILIAN)) == 20
    assert len(catalog.eligible(GroundStationClass.GOVERNMENT)) == 6
    assert len(catalog.eligible(GroundStationClass.MILITARY)) == 6


@pytest.mark.parametrize(
    "station_id",
    ["", "ab", "civilian_1", "1CIV", "CIV-TEST", "A" * 65],
)
def test_invalid_station_id_rejected(station_id: str) -> None:
    with pytest.raises(ValueError, match="station_id"):
        station(station_id=station_id)


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("latitude_deg", -90.1, ValueError),
        ("latitude_deg", 90.1, ValueError),
        ("longitude_deg", -180.1, ValueError),
        ("longitude_deg", 180.1, ValueError),
        ("altitude_m", -500.1, ValueError),
        ("altitude_m", 9000.1, ValueError),
        ("latitude_deg", math.nan, ValueError),
        ("longitude_deg", math.inf, ValueError),
        ("altitude_m", -math.inf, ValueError),
        ("latitude_deg", True, TypeError),
        ("longitude_deg", False, TypeError),
        ("altitude_m", "1.0", TypeError),
    ],
)
def test_invalid_numeric_station_fields_rejected(
    field_name: str, value: object, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type, match=field_name):
        station(**{field_name: value})


@pytest.mark.parametrize("region", ["", "A", "North_America", "north-america", "a" * 65])
def test_invalid_region_rejected(region: str) -> None:
    with pytest.raises(ValueError, match="region"):
        station(region=region)


@pytest.mark.parametrize("country_code", ["", "Z", "ZZZ", "zz", "Z1"])
def test_invalid_country_code_rejected(country_code: str) -> None:
    with pytest.raises(ValueError, match="country_code"):
        station(country_code=country_code)


def test_unknown_station_class_rejected() -> None:
    with pytest.raises(TypeError, match="station_class"):
        station(station_class="civilian")


@pytest.mark.parametrize("enabled", [1, 0, "true", None])
def test_non_boolean_enabled_rejected(enabled: object) -> None:
    with pytest.raises(TypeError, match="enabled"):
        station(enabled=enabled)


def test_station_name_is_trimmed_and_nfc_normalized() -> None:
    decomposed = "  Cafe\u0301 Research  "
    composed = "Caf\u00e9 Research"
    assert station(name=decomposed).name == composed
    assert station(name=decomposed).canonical_record() == station(name=composed).canonical_record()


@pytest.mark.parametrize("name", ["", " \t\n", "A" * 129])
def test_invalid_station_name_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        station(name=name)


def test_internal_station_name_whitespace_is_preserved() -> None:
    assert station(name="  Synthetic   Research  ").name == "Synthetic   Research"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "0"), (-0.0, "0"), (1.25, "1.25"), (-1.25, "-1.25"), (1e-100, "1e-100")],
)
def test_canonical_float_strings(value: float, expected: str) -> None:
    assert canonical_float_string(value) == expected
    assert float(canonical_float_string(value)) == value


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_canonical_float_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_float_string(value)


def test_binary64_canonical_round_trip_stability() -> None:
    values = [1.2345678901234567, -987654321.1234567, 5e-324, 1.7976931348623157e308]
    for value in values:
        encoded = canonical_float_string(value)
        assert float(encoded) == value
        assert canonical_float_string(float(encoded)) == encoded


def test_negative_zero_station_hash_matches_positive_zero() -> None:
    positive = GroundStationCatalog((station(latitude_deg=0.0),))
    negative = GroundStationCatalog((station(latitude_deg=-0.0),))
    assert positive.catalog_hash == negative.catalog_hash
    assert positive.stations[0].latitude_deg == 0.0


def test_duplicate_station_ids_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        GroundStationCatalog((station(), station(name="Different Name")))


def test_empty_catalog_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        GroundStationCatalog(())


def test_catalog_hash_is_row_order_independent() -> None:
    first = station(station_id="CIV_TEST_001")
    second = station(station_id="CIV_TEST_002")
    assert GroundStationCatalog((first, second)).catalog_hash == GroundStationCatalog(
        (second, first)
    ).catalog_hash


def test_catalog_hash_is_path_independent(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_bytes(FIXTURE.read_bytes())
    second.write_bytes(FIXTURE.read_bytes())
    assert load_ground_station_catalog(first).catalog_hash == load_ground_station_catalog(
        second
    ).catalog_hash


@pytest.mark.parametrize(
    "changed",
    [
        {"latitude_deg": 11.0},
        {"name": "Changed Synthetic Name"},
        {"station_class": GroundStationClass.GOVERNMENT},
        {"enabled": False},
        {"region": "region_beta"},
    ],
)
def test_station_content_changes_catalog_hash(changed: dict[str, object]) -> None:
    original = station()
    modified = replace(original, **changed)
    assert GroundStationCatalog((original,)).catalog_hash != GroundStationCatalog(
        (modified,)
    ).catalog_hash


def test_generic_fixture_fails_production_readiness() -> None:
    with pytest.raises(ValueError, match="not production-ready"):
        validate_production_catalog_readiness(load_ground_station_catalog(FIXTURE))


def test_generated_fifty_per_class_catalog_is_production_ready() -> None:
    records = []
    for station_class in GroundStationClass:
        prefix = station_class.value[:3].upper()
        for index in range(50):
            records.append(
                station(
                    station_id=f"{prefix}_READY_{index:03d}",
                    name=f"Synthetic {station_class.value} {index}",
                    station_class=station_class,
                    region=f"region_{index % 5:02d}",
                )
            )
    validate_production_catalog_readiness(GroundStationCatalog(tuple(records)))


def test_readiness_minimum_must_be_positive_integer() -> None:
    catalog = load_ground_station_catalog(FIXTURE)
    for value in [0, -1, True, 1.0]:
        with pytest.raises(ValueError, match="positive integer"):
            validate_production_catalog_readiness(catalog, minimum_enabled_per_class=value)


def test_loader_rejects_invalid_boolean_spelling(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    for value in ["TRUE", "FALSE", "1", "0", "yes", "no"]:
        write_rows(path, [valid_row(enabled=value)])
        with pytest.raises(ValueError, match="enabled"):
            load_ground_station_catalog(path)


def test_loader_rejects_unknown_class(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    write_rows(path, [valid_row(station_class="commercial")])
    with pytest.raises(ValueError, match="unknown station_class"):
        load_ground_station_catalog(path)


def test_loader_rejects_duplicate_headers(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(
        "station_id,station_id,name,station_class,latitude_deg,longitude_deg,altitude_m,region,country_code,enabled\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate headers"):
        load_ground_station_catalog(path)


def test_loader_rejects_missing_and_unexpected_headers(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    headers = tuple(column for column in CATALOG_COLUMNS if column != "region") + ("extra",)
    write_rows(path, [], headers=headers)
    with pytest.raises(ValueError, match="exactly match"):
        load_ground_station_catalog(path)


def test_loader_rejects_empty_and_header_only_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_ground_station_catalog(empty)
    header_only = tmp_path / "header.csv"
    header_only.write_text(",".join(CATALOG_COLUMNS) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        load_ground_station_catalog(header_only)


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.csv"
    write_rows(path, [valid_row(), valid_row(name="Duplicate")])
    with pytest.raises(ValueError, match="Duplicate"):
        load_ground_station_catalog(path)
