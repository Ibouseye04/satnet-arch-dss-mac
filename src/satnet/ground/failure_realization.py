from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Mapping

from satnet.ground.canonical import canonical_hash, canonical_json
from satnet.ground.catalog import GroundStationCatalog, STATION_ID_PATTERN
from satnet.ground.failure_policy import (
    GROUND_FAILURE_MODEL_VERSION,
    GROUND_FAILURE_SAMPLING_VERSION,
    GroundFailurePolicy,
    validate_ground_failure_seed,
)
from satnet.ground.persistence import GroundRunDesignRecord, reconstruct_ground_selection

GROUND_FAILURE_TRIAL_IDENTITY_DOMAIN = "satnet_ground_failure_trial"
GROUND_FAILURE_TRIAL_IDENTITY_VERSION = "1"
GROUND_FAILURE_REALIZATION_IDENTITY_DOMAIN = "satnet_ground_failure_realization"
GROUND_FAILURE_REALIZATION_IDENTITY_VERSION = "1"
GROUND_FAILURE_REALIZATION_SCHEMA_VERSION = "1"
GROUND_FAILURE_REALIZATION_RECORD_IDENTITY_DOMAIN = (
    "satnet_ground_failure_realization_record"
)
GROUND_FAILURE_REALIZATION_RECORD_IDENTITY_VERSION = "1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _validate_hash(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be exactly 64 lowercase hexadecimal characters")


def _validate_count(value: object, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise TypeError(f"{field_name} must be a nonnegative integer")


def _validate_station_ids(values: object, field_name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if any(
        not isinstance(value, str) or not STATION_ID_PATTERN.fullmatch(value)
        for value in values
    ):
        raise ValueError(f"{field_name} contains an invalid station ID")
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{field_name} must use unique ascending station IDs")


def ground_failure_trial_payload(
    *, station_id: str, ground_failure_seed: int
) -> dict[str, object]:
    validate_ground_failure_seed(ground_failure_seed)
    if not isinstance(station_id, str) or not STATION_ID_PATTERN.fullmatch(station_id):
        raise ValueError("station_id must be a canonical ground-station ID")
    return {
        "identity_domain": GROUND_FAILURE_TRIAL_IDENTITY_DOMAIN,
        "identity_version": GROUND_FAILURE_TRIAL_IDENTITY_VERSION,
        "ground_failure_model_version": GROUND_FAILURE_MODEL_VERSION,
        "ground_failure_sampling_version": GROUND_FAILURE_SAMPLING_VERSION,
        "ground_failure_seed": ground_failure_seed,
        "station_id": station_id,
    }


def ground_failure_trial_bytes(*, station_id: str, ground_failure_seed: int) -> bytes:
    return canonical_json(
        ground_failure_trial_payload(
            station_id=station_id,
            ground_failure_seed=ground_failure_seed,
        )
    ).encode("utf-8")


def ground_failure_trial_digest(*, station_id: str, ground_failure_seed: int) -> bytes:
    return hashlib.sha256(
        ground_failure_trial_bytes(
            station_id=station_id,
            ground_failure_seed=ground_failure_seed,
        )
    ).digest()


def ground_station_fails(
    *, station_id: str, ground_failure_seed: int, policy: GroundFailurePolicy
) -> bool:
    if not isinstance(policy, GroundFailurePolicy):
        raise TypeError("policy must be a GroundFailurePolicy")
    digest_int = int.from_bytes(
        ground_failure_trial_digest(
            station_id=station_id,
            ground_failure_seed=ground_failure_seed,
        ),
        byteorder="big",
        signed=False,
    )
    numerator, denominator = policy.ground_station_failure_probability.as_integer_ratio()
    return digest_int * denominator < numerator * (1 << 256)


def _realization_payload(
    source: GroundFailureRealization | Mapping[str, object],
) -> dict[str, object]:
    def value(name: str) -> object:
        return source[name] if isinstance(source, Mapping) else getattr(source, name)

    return {
        "failed_ground_station_count": value("failed_ground_station_count"),
        "failed_station_ids": list(value("failed_station_ids")),
        "ground_design_hash": value("ground_design_hash"),
        "ground_failure_model_version": value("ground_failure_model_version"),
        "ground_failure_policy_hash": value("ground_failure_policy_hash"),
        "ground_failure_sampling_version": value("ground_failure_sampling_version"),
        "ground_failure_seed": value("ground_failure_seed"),
        "identity_domain": GROUND_FAILURE_REALIZATION_IDENTITY_DOMAIN,
        "identity_version": GROUND_FAILURE_REALIZATION_IDENTITY_VERSION,
        "operational_ground_station_count": value(
            "operational_ground_station_count"
        ),
        "operational_station_ids": list(value("operational_station_ids")),
        "selected_station_ids": list(value("selected_station_ids")),
        "total_ground_station_count": value("total_ground_station_count"),
    }


def _compute_realization_hash(source: Mapping[str, object]) -> str:
    return canonical_hash(_realization_payload(source))


@dataclass(frozen=True)
class GroundFailureRealization:
    ground_failure_model_version: str
    ground_failure_sampling_version: str
    ground_design_hash: str
    ground_failure_policy_hash: str
    ground_failure_seed: int
    selected_station_ids: tuple[str, ...]
    failed_station_ids: tuple[str, ...]
    operational_station_ids: tuple[str, ...]
    total_ground_station_count: int
    failed_ground_station_count: int
    operational_ground_station_count: int
    ground_failure_realization_hash: str

    def __post_init__(self) -> None:
        if self.ground_failure_model_version != GROUND_FAILURE_MODEL_VERSION:
            raise ValueError("Unsupported ground_failure_model_version")
        if self.ground_failure_sampling_version != GROUND_FAILURE_SAMPLING_VERSION:
            raise ValueError("Unsupported ground_failure_sampling_version")
        _validate_hash(self.ground_design_hash, "ground_design_hash")
        _validate_hash(self.ground_failure_policy_hash, "ground_failure_policy_hash")
        _validate_hash(
            self.ground_failure_realization_hash,
            "ground_failure_realization_hash",
        )
        validate_ground_failure_seed(self.ground_failure_seed)
        for field_name in (
            "selected_station_ids",
            "failed_station_ids",
            "operational_station_ids",
        ):
            _validate_station_ids(getattr(self, field_name), field_name)
        for field_name in (
            "total_ground_station_count",
            "failed_ground_station_count",
            "operational_ground_station_count",
        ):
            _validate_count(getattr(self, field_name), field_name)
        if self.total_ground_station_count <= 0:
            raise ValueError("A ground-failure realization requires selected stations")
        selected = set(self.selected_station_ids)
        failed = set(self.failed_station_ids)
        operational = set(self.operational_station_ids)
        if not failed <= selected or not operational <= selected:
            raise ValueError("Failed and operational station IDs must be selected")
        if failed & operational:
            raise ValueError("Failed and operational station IDs must be disjoint")
        if failed | operational != selected:
            raise ValueError("Failed and operational station IDs must partition selection")
        if self.total_ground_station_count != len(self.selected_station_ids):
            raise ValueError("total_ground_station_count does not match selected IDs")
        if self.failed_ground_station_count != len(self.failed_station_ids):
            raise ValueError("failed_ground_station_count does not match failed IDs")
        if self.operational_ground_station_count != len(self.operational_station_ids):
            raise ValueError(
                "operational_ground_station_count does not match operational IDs"
            )
        if (
            self.failed_ground_station_count
            + self.operational_ground_station_count
            != self.total_ground_station_count
        ):
            raise ValueError("Failure counts do not sum to total ground count")
        if self.ground_failure_realization_hash != canonical_hash(
            _realization_payload(self)
        ):
            raise ValueError(
                "ground_failure_realization_hash does not match canonical realization"
            )

    def scientific_manifest_object(self) -> dict[str, object]:
        return _realization_payload(self)


def _require_enabled_selection(
    ground_design: GroundRunDesignRecord, catalog: GroundStationCatalog
) -> tuple[str, ...]:
    if not isinstance(ground_design, GroundRunDesignRecord):
        raise TypeError("ground_design must be a GroundRunDesignRecord")
    if not isinstance(catalog, GroundStationCatalog):
        raise TypeError("catalog must be a GroundStationCatalog")
    if not ground_design.ground_segment_enabled:
        raise ValueError("G5 requires an enabled ground design")
    selection = reconstruct_ground_selection(ground_design, catalog)
    if selection is None or not selection.selected_station_ids:
        raise ValueError("G5 requires a nonempty ground selection")
    return tuple(sorted(selection.selected_station_ids))


def _sample_values(
    *,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    policy: GroundFailurePolicy,
    ground_failure_seed: int,
) -> dict[str, object]:
    if not isinstance(policy, GroundFailurePolicy):
        raise TypeError("policy must be a GroundFailurePolicy")
    validate_ground_failure_seed(ground_failure_seed)
    selected = _require_enabled_selection(ground_design, catalog)
    failed = tuple(
        station_id
        for station_id in selected
        if ground_station_fails(
            station_id=station_id,
            ground_failure_seed=ground_failure_seed,
            policy=policy,
        )
    )
    failed_set = set(failed)
    operational = tuple(
        station_id for station_id in selected if station_id not in failed_set
    )
    return {
        "ground_failure_model_version": GROUND_FAILURE_MODEL_VERSION,
        "ground_failure_sampling_version": GROUND_FAILURE_SAMPLING_VERSION,
        "ground_design_hash": ground_design.ground_design_hash,
        "ground_failure_policy_hash": policy.ground_failure_policy_hash,
        "ground_failure_seed": ground_failure_seed,
        "selected_station_ids": selected,
        "failed_station_ids": failed,
        "operational_station_ids": operational,
        "total_ground_station_count": len(selected),
        "failed_ground_station_count": len(failed),
        "operational_ground_station_count": len(operational),
    }


def sample_ground_failure_realization(
    *,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    policy: GroundFailurePolicy,
    ground_failure_seed: int,
) -> GroundFailureRealization:
    values = _sample_values(
        ground_design=ground_design,
        catalog=catalog,
        policy=policy,
        ground_failure_seed=ground_failure_seed,
    )
    values["ground_failure_realization_hash"] = _compute_realization_hash(values)
    return GroundFailureRealization(**values)


def validate_ground_failure_realization_context(
    *,
    realization: GroundFailureRealization,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    policy: GroundFailurePolicy,
) -> None:
    if not isinstance(realization, GroundFailureRealization):
        raise TypeError("realization must be a GroundFailureRealization")
    expected = sample_ground_failure_realization(
        ground_design=ground_design,
        catalog=catalog,
        policy=policy,
        ground_failure_seed=realization.ground_failure_seed,
    )
    if realization != expected:
        raise ValueError("Ground-failure realization does not match authoritative context")


def _realization_record_hash(*, run_id: int, realization: GroundFailureRealization) -> str:
    return canonical_hash(
        {
            "ground_failure_realization_hash": realization.ground_failure_realization_hash,
            "ground_failure_realization_schema_version": GROUND_FAILURE_REALIZATION_SCHEMA_VERSION,
            "identity_domain": GROUND_FAILURE_REALIZATION_RECORD_IDENTITY_DOMAIN,
            "identity_version": GROUND_FAILURE_REALIZATION_RECORD_IDENTITY_VERSION,
            "run_id": run_id,
        }
    )


@dataclass(frozen=True)
class GroundFailureRealizationRecord:
    ground_failure_realization_schema_version: str
    run_id: int
    realization: GroundFailureRealization
    record_hash: str

    def __post_init__(self) -> None:
        if (
            self.ground_failure_realization_schema_version
            != GROUND_FAILURE_REALIZATION_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported ground_failure_realization_schema_version")
        if type(self.run_id) is not int or self.run_id < 0:
            raise TypeError("run_id must be a nonnegative integer")
        if not isinstance(self.realization, GroundFailureRealization):
            raise TypeError("realization must be a GroundFailureRealization")
        if self.record_hash != _realization_record_hash(
            run_id=self.run_id, realization=self.realization
        ):
            raise ValueError("record_hash does not match canonical realization record")


def create_ground_failure_realization_record(
    *,
    run_id: int,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    policy: GroundFailurePolicy,
    ground_failure_seed: int,
) -> GroundFailureRealizationRecord:
    if type(run_id) is not int or run_id < 0:
        raise TypeError("run_id must be a nonnegative integer")
    if not isinstance(ground_design, GroundRunDesignRecord):
        raise TypeError("ground_design must be a GroundRunDesignRecord")
    if run_id != ground_design.run_id:
        raise ValueError("Realization-record run ID does not match G1 ground design")
    realization = sample_ground_failure_realization(
        ground_design=ground_design,
        catalog=catalog,
        policy=policy,
        ground_failure_seed=ground_failure_seed,
    )
    validate_ground_failure_realization_context(
        realization=realization,
        ground_design=ground_design,
        catalog=catalog,
        policy=policy,
    )
    return GroundFailureRealizationRecord(
        ground_failure_realization_schema_version=GROUND_FAILURE_REALIZATION_SCHEMA_VERSION,
        run_id=run_id,
        realization=realization,
        record_hash=_realization_record_hash(run_id=run_id, realization=realization),
    )


def validate_ground_failure_realization_record_context(
    *,
    record: GroundFailureRealizationRecord,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    policy: GroundFailurePolicy,
) -> None:
    if not isinstance(record, GroundFailureRealizationRecord):
        raise TypeError("record must be a GroundFailureRealizationRecord")
    if record.run_id != ground_design.run_id:
        raise ValueError("Realization-record run ID does not match G1 ground design")
    validate_ground_failure_realization_context(
        realization=record.realization,
        ground_design=ground_design,
        catalog=catalog,
        policy=policy,
    )
