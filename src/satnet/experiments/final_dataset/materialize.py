from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import os
from typing import Any, Iterable

from satnet.experiments.final_dataset.design import (
    build_design_records,
    build_run_records,
    design_manifest_hash,
    run_manifest_hash,
    validate_design_records,
    validate_run_records,
)
from satnet.experiments.final_dataset.deterministic import (
    canonical_digest,
    derived_seed,
    ground_failure_seed_payload,
    ground_selection_seed_payload,
    lhs_jitter_payload,
    lhs_permutation_payload,
    satellite_seed_payload,
    schedule_pairing_payload,
    split_candidate_payload,
)
from satnet.experiments.final_dataset.specification import (
    CATALOG_HASH,
    PILOT_DESIGN_MANIFEST_HASH,
    build_contract_specification,
    validate_contract_specification,
)
from satnet.experiments.final_dataset.split import (
    build_split_manifest,
    validate_split_manifest,
)
from satnet.experiments.integrated_ground_manifest import read_pilot_design_manifest
from satnet.ground.canonical import canonical_hash, canonical_json

BUNDLE_IDENTITY_DOMAIN = "satnet_final_integrated_dataset_contract_bundle"
BUNDLE_IDENTITY_VERSION = "1"
GOLDEN_VECTOR_IDENTITY_DOMAIN = "satnet_final_integrated_dataset_golden_vectors"
GOLDEN_VECTOR_IDENTITY_VERSION = "1"
DOE_EVIDENCE_IDENTITY_DOMAIN = "satnet_final_integrated_dataset_doe_evidence"
DOE_EVIDENCE_IDENTITY_VERSION = "1"


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _atomic_write(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Contract artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: dict[str, Any], *, overwrite: bool) -> None:
    _atomic_write(path, canonical_json(value) + "\n", overwrite=overwrite)


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]], *, overwrite: bool) -> None:
    serialized = [canonical_json(value) for value in values]
    if not serialized:
        raise ValueError("JSONL artifact cannot be empty")
    _atomic_write(path, "\n".join(serialized) + "\n", overwrite=overwrite)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs_without_duplicates
    )
    if not isinstance(value, dict):
        raise ValueError("Canonical JSON artifact must contain one object")
    return value


def read_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    text = Path(path).read_text(encoding="utf-8")
    if not text or not text.endswith("\n"):
        raise ValueError("Canonical JSONL must be nonempty and end with a newline")
    lines = text[:-1].split("\n")
    if any(not line for line in lines):
        raise ValueError("Canonical JSONL contains an empty line")
    result: list[dict[str, Any]] = []
    for line in lines:
        value = json.loads(line, object_pairs_hook=_pairs_without_duplicates)
        if not isinstance(value, dict):
            raise ValueError("Every canonical JSONL line must contain an object")
        if canonical_json(value) != line:
            raise ValueError("JSONL line is not canonical JSON")
        result.append(value)
    return tuple(result)


def _golden_vector(name: str, payload: dict[str, Any], *, include_seed: bool) -> dict[str, Any]:
    digest = canonical_digest(payload)
    result: dict[str, Any] = {
        "name": name,
        "payload": payload,
        "canonical_json": canonical_json(payload),
        "sha256": digest.hex(),
    }
    if include_seed:
        result["derived_seed"] = derived_seed(payload)
    return result


def build_golden_vectors() -> dict[str, Any]:
    vectors = [
        _golden_vector(
            "ground_selection_seed_D000",
            ground_selection_seed_payload("D000"),
            include_seed=True,
        ),
        _golden_vector(
            "satellite_seed_D000_R01",
            satellite_seed_payload("D000", "R01"),
            include_seed=True,
        ),
        _golden_vector(
            "ground_failure_seed_D000_R01",
            ground_failure_seed_payload("D000", "R01"),
            include_seed=True,
        ),
        _golden_vector(
            "lhs_permutation_transition_candidate_0_altitude_row_0",
            lhs_permutation_payload(
                stratum_id="transition",
                candidate_id=0,
                dimension_name="altitude_km",
                row_index=0,
            ),
            include_seed=False,
        ),
        _golden_vector(
            "lhs_jitter_transition_candidate_0_altitude_row_0",
            lhs_jitter_payload(
                stratum_id="transition",
                candidate_id=0,
                dimension_name="altitude_km",
                row_index=0,
            ),
            include_seed=False,
        ),
        _golden_vector(
            "schedule_pairing_transition_continuous_row_0",
            schedule_pairing_payload(
                stratum_id="transition",
                schedule_purpose="continuous_rows",
                record_index=0,
            ),
            include_seed=False,
        ),
        _golden_vector(
            "split_candidate_0_D000",
            split_candidate_payload(candidate_id=0, design_id="D000"),
            include_seed=False,
        ),
    ]
    payload: dict[str, Any] = {
        "identity_domain": GOLDEN_VECTOR_IDENTITY_DOMAIN,
        "identity_version": GOLDEN_VECTOR_IDENTITY_VERSION,
        "vectors": vectors,
    }
    result = dict(payload)
    result["golden_vectors_hash"] = canonical_hash(payload)
    return result


def build_doe_evidence(designs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    normalized = tuple(designs)
    transition = normalized[5:40]
    global_records = normalized[40:]
    payload: dict[str, Any] = {
        "identity_domain": DOE_EVIDENCE_IDENTITY_DOMAIN,
        "identity_version": DOE_EVIDENCE_IDENTITY_VERSION,
        "design_counts": {
            "pilot_anchor": 5,
            "transition": 35,
            "global": 60,
            "total": 100,
        },
        "lhs_selected_candidate_ids": {
            "transition": transition[0]["lhs_candidate_id"],
            "global": global_records[0]["lhs_candidate_id"],
        },
        "transition_satellite_pair_frequencies": [
            {
                "num_planes": planes,
                "sats_per_plane": sats,
                "count": sum(
                    (record["num_planes"], record["sats_per_plane"]) == (planes, sats)
                    for record in transition
                ),
            }
            for planes, sats in ((5, 6), (5, 7), (5, 8), (6, 6), (6, 7), (6, 8))
        ],
        "transition_ground_cell_count": len(
            {
                (
                    record["total_ground_station_count"],
                    tuple(record["composition_weights"]),
                )
                for record in transition
            }
        ),
        "global_satellite_pair_frequencies": [
            {
                "num_planes": planes,
                "sats_per_plane": sats,
                "count": sum(
                    (record["num_planes"], record["sats_per_plane"]) == (planes, sats)
                    for record in global_records
                ),
            }
            for planes in (4, 5, 6)
            for sats in (5, 6, 7, 8)
        ],
        "global_station_total_frequencies": [
            {
                "total": total,
                "count": sum(
                    record["total_ground_station_count"] == total for record in global_records
                ),
            }
            for total in (6, 10, 15, 20, 25, 30, 35, 40, 45, 50)
        ],
    }
    result = dict(payload)
    result["doe_evidence_hash"] = canonical_hash(payload)
    return result


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_final_contract_manifests(
    *,
    output_root: str | Path,
    pilot_design_manifest: str | Path,
    catalog_path: str | Path,
    overwrite: bool = False,
) -> dict[str, str]:
    root = Path(output_root)
    specification = build_contract_specification()
    validate_contract_specification(specification)
    persisted_specification = read_json(root / "contract_specification.json")
    if persisted_specification != specification:
        raise ValueError("Persisted machine specification differs from production reconstruction")
    designs = build_design_records(
        pilot_design_manifest=pilot_design_manifest,
        catalog_path=catalog_path,
    )
    pilot_designs = read_pilot_design_manifest(pilot_design_manifest)
    validate_design_records(designs, pilot_designs=pilot_designs)
    runs = build_run_records(designs)
    validate_run_records(runs, designs=designs)
    design_hash = design_manifest_hash(designs)
    run_hash = run_manifest_hash(runs)
    split = build_split_manifest(
        designs=designs,
        runs=runs,
        contract_spec_hash=specification["contract_spec_hash"],
        design_manifest_hash=design_hash,
    )
    validate_split_manifest(split, designs=designs, runs=runs)
    schema_hashes = specification["schemas"]
    bundle_payload: dict[str, Any] = {
        "identity_domain": BUNDLE_IDENTITY_DOMAIN,
        "identity_version": BUNDLE_IDENTITY_VERSION,
        "contract_spec_hash": specification["contract_spec_hash"],
        "target_schema_hash": schema_hashes["target_schema_hash"],
        "rf_schema_hash": schema_hashes["rf_schema_hash"],
        "tgnn_adapter_schema_hash": schema_hashes["tgnn_adapter_schema_hash"],
        "catalog_hash": CATALOG_HASH,
        "design_manifest_hash": design_hash,
        "run_manifest_hash": run_hash,
        "split_manifest_hash": split["split_manifest_hash"],
    }
    bundle = dict(bundle_payload)
    bundle["contract_bundle_hash"] = canonical_hash(bundle_payload)
    inventory_payload: dict[str, Any] = {
        "identity_domain": "satnet_final_integrated_dataset_manifest_inventory",
        "identity_version": "1",
        "contract_spec_hash": specification["contract_spec_hash"],
        "contract_bundle_hash": bundle["contract_bundle_hash"],
        "target_schema_hash": schema_hashes["target_schema_hash"],
        "rf_schema_hash": schema_hashes["rf_schema_hash"],
        "tgnn_adapter_schema_hash": schema_hashes["tgnn_adapter_schema_hash"],
        "catalog_hash": CATALOG_HASH,
        "pilot_catalog_file_sha256": _file_sha256(Path(catalog_path)),
        "pilot_design_manifest_hash": PILOT_DESIGN_MANIFEST_HASH,
        "design_manifest_hash": design_hash,
        "run_manifest_hash": run_hash,
        "split_manifest_hash": split["split_manifest_hash"],
        "design_count": len(designs),
        "run_count": len(runs),
        "split_design_counts": {
            name: len(split["design_assignments"][name])
            for name in ("train", "validation", "test")
        },
    }
    inventory = dict(inventory_payload)
    inventory["manifest_inventory_hash"] = canonical_hash(inventory_payload)
    _write_jsonl(root / "designs.jsonl", designs, overwrite=overwrite)
    _write_jsonl(root / "runs.jsonl", runs, overwrite=overwrite)
    _write_json(root / "split_manifest.json", split, overwrite=overwrite)
    _write_json(root / "contract_bundle.json", bundle, overwrite=overwrite)
    _write_json(root / "golden_vectors.json", build_golden_vectors(), overwrite=overwrite)
    _write_json(root / "doe_evidence.json", build_doe_evidence(designs), overwrite=overwrite)
    _write_json(root / "manifest_inventory.json", inventory, overwrite=overwrite)
    return {
        "contract_spec_hash": specification["contract_spec_hash"],
        "contract_bundle_hash": bundle["contract_bundle_hash"],
        "target_schema_hash": schema_hashes["target_schema_hash"],
        "rf_schema_hash": schema_hashes["rf_schema_hash"],
        "tgnn_adapter_schema_hash": schema_hashes["tgnn_adapter_schema_hash"],
        "catalog_hash": CATALOG_HASH,
        "design_manifest_hash": design_hash,
        "run_manifest_hash": run_hash,
        "split_manifest_hash": split["split_manifest_hash"],
    }


def validate_materialized_contract(output_root: str | Path) -> dict[str, str]:
    root = Path(output_root)
    specification = read_json(root / "contract_specification.json")
    validate_contract_specification(specification)
    designs = read_jsonl(root / "designs.jsonl")
    runs = read_jsonl(root / "runs.jsonl")
    pilot_designs = read_pilot_design_manifest(
        Path(__file__).parents[4]
        / "artifacts"
        / "integrated_ground_pilot_25"
        / "inputs"
        / "pilot_designs.json"
    )
    validate_design_records(designs, pilot_designs=pilot_designs)
    validate_run_records(runs, designs=designs)
    design_hash = design_manifest_hash(designs)
    run_hash = run_manifest_hash(runs)
    split = read_json(root / "split_manifest.json")
    validate_split_manifest(split, designs=designs, runs=runs)
    bundle = read_json(root / "contract_bundle.json")
    bundle_payload = {key: value for key, value in bundle.items() if key != "contract_bundle_hash"}
    if canonical_hash(bundle_payload) != bundle.get("contract_bundle_hash"):
        raise ValueError("Contract-bundle hash mismatch")
    if bundle["design_manifest_hash"] != design_hash or bundle["run_manifest_hash"] != run_hash:
        raise ValueError("Bundle manifest identities do not match materialized records")
    if bundle["split_manifest_hash"] != split["split_manifest_hash"]:
        raise ValueError("Bundle split identity does not match materialized split")
    return {
        "contract_spec_hash": specification["contract_spec_hash"],
        "contract_bundle_hash": bundle["contract_bundle_hash"],
        "target_schema_hash": bundle["target_schema_hash"],
        "rf_schema_hash": bundle["rf_schema_hash"],
        "tgnn_adapter_schema_hash": bundle["tgnn_adapter_schema_hash"],
        "catalog_hash": bundle["catalog_hash"],
        "design_manifest_hash": design_hash,
        "run_manifest_hash": run_hash,
        "split_manifest_hash": split["split_manifest_hash"],
    }
