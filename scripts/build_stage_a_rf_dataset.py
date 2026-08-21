from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

EXPECTED_FREEZE_MANIFEST_SHA256 = "b2d1fbd9510d3d828fe051b4da04f6747088ae00d78d65e0a9851dc85559844e"
EXPECTED_CONTRACT_SPEC_HASH = "fc13a33a0a1af435189990e54b6c68efa56bbc207c0cae6e27e12bf031605930"
EXPECTED_SPLIT_FILE_SHA256 = "047b7cc99d83a003add8bca4bb037bce9d642774443eeef1a3df377f855db90d"
EXPECTED_RUN_COUNT = 500
EXPECTED_DESIGN_COUNT = 100
EXPECTED_REALIZATIONS_PER_DESIGN = 5
EXPECTED_SPLIT_DESIGNS = {"train": 70, "validation": 15, "test": 15}
EXPECTED_SPLIT_ROWS = {"train": 350, "validation": 75, "test": 75}
OUTPUT_ARTIFACT_NAMES = frozenset({
    "rf_train.csv",
    "rf_validation.csv",
    "rf_test_sealed_index.csv",
    "rf_feature_schema.json",
    "rf_dataset_manifest.json",
    "rf_provenance_manifest.jsonl",
    "rf_leakage_exclusion_report.json",
    "rf_construction_report.json",
    "artifact_inventory.json",
    "README.md",
})

PREDICTOR_FIELDS = (
    "num_planes",
    "sats_per_plane",
    "configured_satellite_count",
    "altitude_km",
    "inclination_deg",
    "satellite_node_failure_probability",
    "satellite_edge_failure_probability",
    "civilian_count",
    "government_count",
    "military_count",
    "total_ground_station_count",
    "ground_station_failure_probability",
)
TARGET_FIELDS = ("partition_any", "gcc_frac_min")
TARGET_SOURCE_FIELDS = {
    "partition_any": "overall_threshold_breach_any",
    "gcc_frac_min": "space_gcc_fraction_original_min",
}
TRACEABILITY_FIELDS = (
    "design_id",
    "run_id",
    "run_key",
    "design_index",
    "realization_id",
    "realization_index",
    "split",
    "contract_spec_hash",
    "source_result_path",
    "source_result_sha256",
    "source_scientific_inventory_path",
    "source_scientific_inventory_sha256",
    "source_design_record_hash",
    "source_run_record_hash",
    "source_target_artifact_hash",
)
TRAIN_VALIDATION_FIELDS = TRACEABILITY_FIELDS + PREDICTOR_FIELDS + TARGET_FIELDS
TEST_INDEX_FIELDS = TRACEABILITY_FIELDS


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line in {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            records.append(value)
    return records


def semantic_hash(value: dict[str, Any], hash_field: str) -> str:
    payload = {key: item for key, item in value.items() if key != hash_field}
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_file(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> str:
    require(path.is_file(), f"Missing frozen evidence file: {path}")
    if expected_bytes is not None:
        require(path.stat().st_size == expected_bytes, f"Byte length mismatch: {path}")
    actual = sha256_file(path)
    require(actual == expected_sha256, f"SHA-256 mismatch for {path}: {actual} != {expected_sha256}")
    return actual


def validate_freeze(
    freeze_bundle: Path, production_root: Path, contract_root: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    freeze_manifest_path = freeze_bundle / "accepted_production_evidence_freeze_manifest.json"
    verify_file(freeze_manifest_path, EXPECTED_FREEZE_MANIFEST_SHA256)
    freeze_manifest = read_json(freeze_manifest_path)
    require(freeze_manifest["contract"]["contract_spec_hash"] == EXPECTED_CONTRACT_SPEC_HASH, "Freeze contract hash mismatch")
    require(Path(freeze_manifest["generation"]["root"]).resolve() == production_root.resolve(), "Production root differs from frozen root")
    require(Path(freeze_manifest["contract"]["contract_root"]).resolve() == contract_root.resolve(), "Contract root differs from frozen root")
    require(freeze_manifest["scope"]["production_data_modified"] is False, "Frozen scope says production data was modified")
    require(freeze_manifest["scope"]["production_data_regenerated"] is False, "Frozen scope says production data was regenerated")
    require(freeze_manifest["scope"]["model_ready_samples_created"] is False, "Freeze bundle already contains model-ready samples")

    freeze_inventory = read_json(freeze_bundle / "artifact_inventory.json")
    for artifact in freeze_inventory["artifacts"]:
        path = freeze_bundle.parent.parent / artifact["path"]
        verify_file(path, artifact["sha256"], artifact["bytes"])

    contract_spec_path = contract_root / "contract_specification.json"
    contract_spec = read_json(contract_spec_path)
    require(contract_spec.get("contract_spec_hash") == EXPECTED_CONTRACT_SPEC_HASH, "Contract specification field mismatch")
    require(semantic_hash(contract_spec, "contract_spec_hash") == EXPECTED_CONTRACT_SPEC_HASH, "Contract specification semantic hash mismatch")
    designs_path = contract_root / "designs.jsonl"
    runs_path = contract_root / "runs.jsonl"
    split_path = contract_root / "split_manifest.json"
    verify_file(designs_path, freeze_manifest["contract"]["design_manifest"]["sha256"], freeze_manifest["contract"]["design_manifest"]["bytes"])
    verify_file(runs_path, freeze_manifest["contract"]["run_manifest"]["sha256"], freeze_manifest["contract"]["run_manifest"]["bytes"])
    verify_file(split_path, EXPECTED_SPLIT_FILE_SHA256, freeze_manifest["split_manifest"]["bytes"])
    designs = read_jsonl(designs_path)
    runs = read_jsonl(runs_path)
    split_manifest = read_json(split_path)
    require(len(designs) == EXPECTED_DESIGN_COUNT, "Frozen design count mismatch")
    require(len(runs) == EXPECTED_RUN_COUNT, "Frozen run count mismatch")
    require(split_manifest["contract_spec_hash"] == EXPECTED_CONTRACT_SPEC_HASH, "Split contract hash mismatch")
    require(split_manifest["outcome_fields_used"] is False, "Frozen split used outcome fields")
    require(freeze_manifest["split_manifest"]["test_split_sealed"] is True, "Frozen test split is not sealed")
    require({split: len(split_manifest["run_assignments"][split]) for split in EXPECTED_SPLIT_ROWS} == EXPECTED_SPLIT_ROWS, "Frozen split row counts mismatch")
    require({split: len(split_manifest["design_assignments"][split]) for split in EXPECTED_SPLIT_DESIGNS} == EXPECTED_SPLIT_DESIGNS, "Frozen split design counts mismatch")

    design_by_id = {record["design_id"]: record for record in designs}
    require(len(design_by_id) == EXPECTED_DESIGN_COUNT, "Duplicate frozen design IDs")
    split_by_design = {
        design_id: split
        for split, design_ids in split_manifest["design_assignments"].items()
        for design_id in design_ids
    }
    require(len(split_by_design) == EXPECTED_DESIGN_COUNT, "Frozen split does not assign every design exactly once")
    require(set(split_by_design) == set(design_by_id), "Frozen split design IDs differ from contract designs")
    require(set(split_by_design.values()) == set(EXPECTED_SPLIT_DESIGNS), "Frozen split names differ")

    run_by_id = {record["run_id"]: record for record in runs}
    require(len(run_by_id) == EXPECTED_RUN_COUNT and set(run_by_id) == set(range(EXPECTED_RUN_COUNT)), "Frozen run IDs are not 0 through 499")
    for run in runs:
        require(run["split_assignment"] == split_by_design[run["design_id"]], f"Run split mismatch: {run['run_key']}")
        require(run["contract_spec_hash"] == EXPECTED_CONTRACT_SPEC_HASH, f"Run contract hash mismatch: {run['run_key']}")
    for split, run_ids in split_manifest["run_assignments"].items():
        require(set(run_ids) == {run["run_id"] for run in runs if run["split_assignment"] == split}, f"Frozen run assignment mismatch: {split}")
    ledger_path = production_root / "operational" / "generation_ledger.json"
    ledger_expected = freeze_manifest["generation"]["ledger"]
    verify_file(ledger_path, ledger_expected["sha256"], ledger_expected["bytes"])
    ledger = read_json(ledger_path)
    ledger_by_id = {record["run_id"]: record for record in ledger["records"]}
    require(len(ledger_by_id) == EXPECTED_RUN_COUNT, "Production ledger run count mismatch")
    return freeze_manifest, contract_spec, designs, runs, {
        "freeze_manifest_sha256": EXPECTED_FREEZE_MANIFEST_SHA256,
        "contract_spec_sha256": sha256_file(contract_spec_path),
        "design_manifest_sha256": sha256_file(designs_path),
        "run_manifest_sha256": sha256_file(runs_path),
        "split_manifest_sha256": EXPECTED_SPLIT_FILE_SHA256,
        "generation_ledger_sha256": ledger_expected["sha256"],
    }


def validate_source_run(
    production_root: Path,
    design: dict[str, Any],
    run: dict[str, Any],
    ledger_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, str, Path, str]:
    run_directory = production_root / f"run_{run['run_id']:03d}"
    result_path = run_directory / "result.json"
    inventory_path = run_directory / "scientific_inventory.json"
    target_path = run_directory / "targets" / "target.json"
    require(result_path.is_file() and inventory_path.is_file() and target_path.is_file(), f"Missing source artifacts for {run['run_key']}")
    result_sha256 = sha256_file(result_path)
    inventory_sha256 = sha256_file(inventory_path)
    result = read_json(result_path)
    inventory = read_json(inventory_path)
    target = read_json(target_path)
    require(result["run_id"] == run["run_id"] and result["run_key"] == run["run_key"], f"Result identity mismatch: {run['run_key']}")
    require(result["run_record_hash"] == run["run_record_hash"], f"Result run hash mismatch: {run['run_key']}")
    require(result["design_record_hash"] == design["design_record_hash"], f"Result design hash mismatch: {run['run_key']}")
    require(result["contract_spec_hash"] == EXPECTED_CONTRACT_SPEC_HASH, f"Result contract hash mismatch: {run['run_key']}")
    require(result["run_result_hash"] == semantic_hash(result, "run_result_hash"), f"Result semantic hash mismatch: {run['run_key']}")
    require(inventory["scientific_inventory_hash"] == semantic_hash(inventory, "scientific_inventory_hash"), f"Scientific inventory semantic hash mismatch: {run['run_key']}")
    require(inventory["scientific_inventory_hash"] == result["scientific_inventory_hash"], f"Result inventory link mismatch: {run['run_key']}")
    require(target["target_artifact_hash"] == semantic_hash(target, "target_artifact_hash"), f"Target semantic hash mismatch: {run['run_key']}")
    require(target["target_artifact_hash"] == result["target_artifact_hash"], f"Result target link mismatch: {run['run_key']}")
    require(target["run_id"] == run["run_id"] and target["run_key"] == run["run_key"], f"Target identity mismatch: {run['run_key']}")
    require(target["design_record_hash"] == design["design_record_hash"], f"Target design hash mismatch: {run['run_key']}")
    require(ledger_record["state"] == "succeeded", f"Production run not succeeded: {run['run_key']}")
    require(ledger_record["published_result_hash"] == result["run_result_hash"], f"Ledger result link mismatch: {run['run_key']}")
    require(ledger_record["scientific_inventory_hash"] == result["scientific_inventory_hash"], f"Ledger inventory link mismatch: {run['run_key']}")
    require(ledger_record["run_record_hash"] == run["run_record_hash"], f"Ledger run hash mismatch: {run['run_key']}")
    require(ledger_record["design_id"] == run["design_id"] and ledger_record["realization_id"] == run["realization_id"], f"Ledger identity mismatch: {run['run_key']}")
    require(target.get("target_schema_hash") == "9088948d6b03db59877a129179ac3091c3690aeab9d2849a921bfa90f05da528", f"Target schema hash mismatch: {run['run_key']}")
    require(inventory_sha256 == sha256_file(inventory_path), "Inventory hash calculation failed")
    return result, target, result_path, result_sha256, inventory_path, inventory_sha256


def make_row(
    design: dict[str, Any],
    run: dict[str, Any],
    split: str,
    target: dict[str, Any] | None,
    result_path: Path,
    result_sha256: str,
    inventory_path: Path,
    inventory_sha256: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "design_id": design["design_id"],
        "run_id": run["run_id"],
        "run_key": run["run_key"],
        "design_index": design["design_index"],
        "realization_id": run["realization_id"],
        "realization_index": run["realization_index"],
        "split": split,
        "contract_spec_hash": EXPECTED_CONTRACT_SPEC_HASH,
        "source_result_path": str(result_path.resolve()),
        "source_result_sha256": result_sha256,
        "source_scientific_inventory_path": str(inventory_path.resolve()),
        "source_scientific_inventory_sha256": inventory_sha256,
        "source_design_record_hash": design["design_record_hash"],
        "source_run_record_hash": run["run_record_hash"],
        "source_target_artifact_hash": None if target is None else target["target_artifact_hash"],
    }
    for field in PREDICTOR_FIELDS:
        row[field] = design[field]
    if target is not None:
        row["partition_any"] = int(target["overall_threshold_breach_any"])
        row["gcc_frac_min"] = target["space_gcc_fraction_original_min"]
    return row


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row[field] for field in fields})


def count_missing(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, int]:
    return {field: sum(row.get(field) is None or row.get(field) == "" for row in rows) for field in fields}


def duplicate_count(rows: list[dict[str, Any]], field: str) -> int:
    values = [row[field] for row in rows]
    return len(values) - len(set(values))


def field_specs(designs: list[dict[str, Any]], runs: list[dict[str, Any]], targets: list[dict[str, Any]], results: list[dict[str, Any]], inventories: list[dict[str, Any]]) -> dict[str, list[str]]:
    design_fields = sorted(designs[0])
    run_fields = sorted(runs[0])
    target_fields = sorted(targets[0])
    result_fields = sorted(results[0])
    inventory_fields = sorted(inventories[0])
    return {
        "design_fields_not_exported": [field for field in design_fields if field not in PREDICTOR_FIELDS],
        "run_manifest_fields_not_exported": run_fields,
        "target_artifact_fields_not_exported_as_predictors": target_fields,
        "result_manifest_fields_not_exported_as_predictors": result_fields,
        "scientific_inventory_fields_not_exported_as_predictors": inventory_fields,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct Stage A RF dataset v1 from frozen accepted production evidence")
    parser.add_argument("--freeze-bundle", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite-existing-output", action="store_true")
    args = parser.parse_args()
    freeze_bundle = args.freeze_bundle.resolve()
    production_root = args.production_root.resolve()
    contract_root = args.contract_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        require(args.overwrite_existing_output, f"Refusing to overwrite existing output directory: {output_dir}")
        require(output_dir.is_dir(), f"Output path is not a directory: {output_dir}")
        require({path.name for path in output_dir.iterdir()} <= OUTPUT_ARTIFACT_NAMES, "Output directory contains unrelated files")
    require(production_root.is_dir() and contract_root.is_dir() and freeze_bundle.is_dir(), "Required input directory is missing")

    freeze_manifest, contract_spec, designs, runs, frozen_hashes = validate_freeze(freeze_bundle, production_root, contract_root)
    design_by_id = {record["design_id"]: record for record in designs}
    split_manifest = read_json(contract_root / "split_manifest.json")
    split_by_design = {
        design_id: split
        for split, design_ids in split_manifest["design_assignments"].items()
        for design_id in design_ids
    }
    ledger = read_json(production_root / "operational" / "generation_ledger.json")
    ledger_by_id = {record["run_id"]: record for record in ledger["records"]}

    source_results: list[dict[str, Any]] = []
    source_targets: list[dict[str, Any]] = []
    source_inventories: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for run in runs:
        design = design_by_id[run["design_id"]]
        split = split_by_design[design["design_id"]]
        result, target, result_path, result_sha256, inventory_path, inventory_sha256 = validate_source_run(
            production_root, design, run, ledger_by_id[run["run_id"]]
        )
        inventory = read_json(inventory_path)
        source_results.append(result)
        source_targets.append(target)
        source_inventories.append(inventory)
        row = make_row(design, run, split, target if split != "test" else None, result_path, result_sha256, inventory_path, inventory_sha256)
        row["source_target_artifact_hash"] = target["target_artifact_hash"]
        all_rows.append(row)

    require(len(all_rows) == EXPECTED_RUN_COUNT, "Constructed row count mismatch")
    require(len({row["run_key"] for row in all_rows}) == EXPECTED_RUN_COUNT, "Duplicate run keys")
    require(all(row["source_result_sha256"] == sha256_file(Path(row["source_result_path"])) for row in all_rows), "Source result hash verification failed")
    require(all(row["source_scientific_inventory_sha256"] == sha256_file(Path(row["source_scientific_inventory_path"])) for row in all_rows), "Source inventory hash verification failed")
    for split, expected_rows in EXPECTED_SPLIT_ROWS.items():
        rows = [row for row in all_rows if row["split"] == split]
        require(len(rows) == expected_rows, f"{split} row count mismatch")
        require(len({row["design_id"] for row in rows}) == EXPECTED_SPLIT_DESIGNS[split], f"{split} design count mismatch")
        require(all(sum(item["design_id"] == design_id for item in rows) == EXPECTED_REALIZATIONS_PER_DESIGN for design_id in {item["design_id"] for item in rows}), f"{split} realization count mismatch")
    split_sets = {split: {row["design_id"] for row in all_rows if row["split"] == split} for split in EXPECTED_SPLIT_DESIGNS}
    require(not (split_sets["train"] & split_sets["validation"] or split_sets["train"] & split_sets["test"] or split_sets["validation"] & split_sets["test"]), "Design overlap across splits")
    labeled_rows = [row for row in all_rows if row["split"] != "test"]
    require(not any(row["partition_any"] not in (0, 1) for row in labeled_rows), "Invalid partition_any target")
    require(all(math.isfinite(float(row["gcc_frac_min"])) and 0.0 <= float(row["gcc_frac_min"]) <= 1.0 for row in labeled_rows), "Invalid gcc_frac_min target")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [row for row in all_rows if row["split"] == "train"]
    validation_rows = [row for row in all_rows if row["split"] == "validation"]
    test_rows = [row for row in all_rows if row["split"] == "test"]
    write_csv(output_dir / "rf_train.csv", train_rows, TRAIN_VALIDATION_FIELDS)
    write_csv(output_dir / "rf_validation.csv", validation_rows, TRAIN_VALIDATION_FIELDS)
    write_csv(output_dir / "rf_test_sealed_index.csv", test_rows, TEST_INDEX_FIELDS)

    feature_schema_payload: dict[str, Any] = {
        "schema_name": "satnet_stage_a_rf_dataset_v1_feature_schema",
        "schema_version": "1",
        "row_granularity": "one_complete_simulation_run",
        "design_realization_structure": {"design_count": EXPECTED_DESIGN_COUNT, "realizations_per_design": EXPECTED_REALIZATIONS_PER_DESIGN, "run_count": EXPECTED_RUN_COUNT},
        "predictor_count": len(PREDICTOR_FIELDS),
        "predictors": [{"field": field, "source_field": field, "included": True, "is_predictor": True, "availability": "design_time_or_simulation_configuration", "leakage_status": "allowed"} for field in PREDICTOR_FIELDS],
        "targets": [{"field": field, "source_field": TARGET_SOURCE_FIELDS[field], "included_in": ["rf_train.csv", "rf_validation.csv"], "is_predictor": False, "role": "target"} for field in TARGET_FIELDS],
        "traceability_and_provenance": [{"field": field, "included": True, "is_predictor": False, "role": "traceability_or_provenance"} for field in TRACEABILITY_FIELDS],
        "sealed_test_index": {"path": "rf_test_sealed_index.csv", "contains_target_values": False, "outcome_fields_used": False, "columns": list(TEST_INDEX_FIELDS)},
        "source_contract_spec_hash": EXPECTED_CONTRACT_SPEC_HASH,
    }
    feature_schema_payload["feature_schema_hash"] = semantic_hash(feature_schema_payload, "feature_schema_hash")
    write_json(output_dir / "rf_feature_schema.json", feature_schema_payload)

    source_field_documentation = field_specs(designs, runs, source_targets, source_results, source_inventories)
    leakage_report_payload: dict[str, Any] = {
        "report_name": "satnet_stage_a_rf_dataset_v1_leakage_exclusion_report",
        "report_version": "1",
        "predictor_policy": "Ex ante design-time predictors only; identifiers, provenance, targets, outcome summaries, status, seeds, hashes, and split labels are non-predictors.",
        "included_predictors": list(PREDICTOR_FIELDS),
        "excluded_target_fields": [{"dataset_field": field, "source_field": TARGET_SOURCE_FIELDS[field], "reason": "required target; never a predictor"} for field in TARGET_FIELDS] + [{"source_field": field, "reason": "source target artifact field not selected for Stage A RF"} for field in source_field_documentation["target_artifact_fields_not_exported_as_predictors"] if field not in TARGET_SOURCE_FIELDS.values()],
        "excluded_leakage_categories": [
            {"category": "identifiers_and_split", "fields": TRACEABILITY_FIELDS, "reason": "traceability/provenance only; IDs and split labels are forbidden predictors"},
            {"category": "random_seeds_and_realized_failures", "fields": [field for field in runs[0] if "seed" in field or "failure" in field and field not in PREDICTOR_FIELDS], "reason": "stochastic realization identity or post-design outcome information"},
            {"category": "hashes_and_paths", "fields": [field for field in TRACEABILITY_FIELDS if "hash" in field or "path" in field], "reason": "provenance only"},
            {"category": "post_outcome_summaries_and_thresholds", "fields": source_field_documentation["target_artifact_fields_not_exported_as_predictors"], "reason": "target, direct target derivative, service/connectivity outcome, or threshold outcome"},
            {"category": "generation_replay_acceptance_status", "fields": ["state", "attempt_count", "completed_stages", "published_result_hash", "production_acceptance", "replay_status"], "reason": "operational/replay/acceptance status is not a scientific predictor"},
            {"category": "scientific_artifact_inventory", "fields": source_field_documentation["scientific_inventory_fields_not_exported_as_predictors"], "reason": "artifact metadata and hashes are provenance only"},
        ],
        "source_field_audit": source_field_documentation,
        "explicitly_not_exported_design_fields": source_field_documentation["design_fields_not_exported"],
    }
    leakage_report_payload["leakage_report_hash"] = semantic_hash(leakage_report_payload, "leakage_report_hash")
    write_json(output_dir / "rf_leakage_exclusion_report.json", leakage_report_payload)

    provenance_lines = []
    for row in all_rows:
        provenance_lines.append({field: row[field] for field in TRACEABILITY_FIELDS})
    with (output_dir / "rf_provenance_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for value in provenance_lines:
            handle.write(canonical_json(value) + "\n")

    dataset_manifest_payload: dict[str, Any] = {
        "manifest_name": "satnet_stage_a_rf_dataset_v1_manifest",
        "manifest_version": "1",
        "dataset_version": "stage_a_rf_dataset_v1",
        "construction_scope": "RF dataset construction only; no model training, tuning, TGNN construction, or test evaluation",
        "freeze": {"freeze_bundle": str(freeze_bundle), "freeze_manifest_sha256": EXPECTED_FREEZE_MANIFEST_SHA256, "contract_spec_hash": EXPECTED_CONTRACT_SPEC_HASH, "production_root": str(production_root), "contract_root": str(contract_root), "contract_spec_sha256": frozen_hashes["contract_spec_sha256"], "design_manifest_sha256": frozen_hashes["design_manifest_sha256"], "run_manifest_sha256": frozen_hashes["run_manifest_sha256"], "split_manifest_exact_file_sha256": EXPECTED_SPLIT_FILE_SHA256, "generation_ledger_sha256": frozen_hashes["generation_ledger_sha256"]},
        "required_targets": [{"field": field, "source_field": TARGET_SOURCE_FIELDS[field]} for field in TARGET_FIELDS],
        "predictor_count": len(PREDICTOR_FIELDS),
        "predictors": list(PREDICTOR_FIELDS),
        "row_and_design_counts": {split: {"rows": len([row for row in all_rows if row["split"] == split]), "designs": len(split_sets[split]), "realizations_per_design": EXPECTED_REALIZATIONS_PER_DESIGN} for split in EXPECTED_SPLIT_DESIGNS},
        "source_run_count": len(all_rows),
        "test_seal": {"sealed": True, "target_values_materialized": False, "outcome_fields_used": False, "test_index_path": "rf_test_sealed_index.csv"},
        "implementation": "scripts/build_stage_a_rf_dataset.py",
    }
    dataset_manifest_payload["dataset_manifest_hash"] = semantic_hash(dataset_manifest_payload, "dataset_manifest_hash")
    write_json(output_dir / "rf_dataset_manifest.json", dataset_manifest_payload)

    missing_values = {"train": count_missing(train_rows, TRAIN_VALIDATION_FIELDS), "validation": count_missing(validation_rows, TRAIN_VALIDATION_FIELDS), "test_sealed_index": count_missing(test_rows, TEST_INDEX_FIELDS)}
    construction_report_payload: dict[str, Any] = {
        "report_name": "satnet_stage_a_rf_dataset_v1_construction_report",
        "report_version": "1",
        "verdict": "STAGE A RF DATASET V1 CONSTRUCTED — READY FOR NARROW INDEPENDENT DATASET GATE",
        "construction_checks": {"train_rows": len(train_rows) == 350, "validation_rows": len(validation_rows) == 75, "sealed_test_index_rows": len(test_rows) == 75, "train_designs": len(split_sets["train"]) == 70, "validation_designs": len(split_sets["validation"]) == 15, "test_designs": len(split_sets["test"]) == 15, "five_realizations_per_design": all(sum(row["design_id"] == design_id for row in all_rows) == 5 for design_id in design_by_id), "zero_design_overlap": not (split_sets["train"] & split_sets["validation"] or split_sets["train"] & split_sets["test"] or split_sets["validation"] & split_sets["test"]), "zero_run_overlap": len({row["run_id"] for row in all_rows}) == 500, "no_duplicate_run_keys": duplicate_count(all_rows, "run_key") == 0, "no_missing_train_validation_targets": all(value == 0 for field_counts in missing_values.values() for field, value in field_counts.items() if field in TARGET_FIELDS), "no_target_or_provenance_predictors": not set(PREDICTOR_FIELDS) & (set(TARGET_FIELDS) | set(TRACEABILITY_FIELDS))},
        "missing_value_counts": missing_values,
        "duplicate_counts": {"run_key_all": duplicate_count(all_rows, "run_key"), "run_id_all": duplicate_count(all_rows, "run_id"), "design_realization_pair_all": len(all_rows) - len({(row["design_id"], row["realization_id"]) for row in all_rows})},
        "source_hash_verification": {"freeze_manifest": True, "contract_specification": True, "contract_design_manifest": True, "contract_run_manifest": True, "split_manifest_exact_file": True, "generation_ledger": True, "source_result_files": len(all_rows), "source_scientific_inventory_files": len(all_rows), "source_result_links_to_frozen_ledger": True, "source_target_artifact_links_verified": True},
        "evidence_integrity": {"production_artifacts_modified": False, "replay_artifacts_modified": False, "acceptance_artifacts_modified": False, "production_root_used_read_only": True},
        "test_seal_status": {"sealed": True, "test_target_values_used": False, "test_outcome_fields_used": False, "test_index_contains_identifiers_and_provenance_only": True},
        "target_identities": {"classification": "partition_any", "regression": "gcc_frac_min", "classification_source_field": TARGET_SOURCE_FIELDS["partition_any"], "regression_source_field": TARGET_SOURCE_FIELDS["gcc_frac_min"]},
        "predictor_count": len(PREDICTOR_FIELDS),
        "excluded_leakage_report": "rf_leakage_exclusion_report.json",
        "independent_acceptance_gate_run": False,
    }
    construction_report_payload["construction_report_hash"] = semantic_hash(construction_report_payload, "construction_report_hash")
    write_json(output_dir / "rf_construction_report.json", construction_report_payload)

    artifact_names = ["rf_train.csv", "rf_validation.csv", "rf_test_sealed_index.csv", "rf_feature_schema.json", "rf_dataset_manifest.json", "rf_provenance_manifest.jsonl", "rf_leakage_exclusion_report.json", "rf_construction_report.json", "README.md"]
    readme = """# Stage A Random Forest Model-Ready Dataset v1\n\nThis artifact is constructed only from the frozen accepted Stage A production evidence. It contains one row per complete simulation run, with five realization rows per design. No model was trained or tuned, no TGNN dataset was built, and no test target values were materialized.\n\n## Targets\n\n- `partition_any`: source field `overall_threshold_breach_any`.\n- `gcc_frac_min`: source field `space_gcc_fraction_original_min`.\n\n## Predictors\n\nThe 12 predictors are the pre-outcome design/configuration fields documented in `rf_feature_schema.json`: `num_planes`, `sats_per_plane`, `configured_satellite_count`, `altitude_km`, `inclination_deg`, `satellite_node_failure_probability`, `satellite_edge_failure_probability`, `civilian_count`, `government_count`, `military_count`, `total_ground_station_count`, and `ground_station_failure_probability`.\n\nIdentifiers and provenance are retained in every artifact but are explicitly non-feature. The complete included/excluded field audit is in `rf_leakage_exclusion_report.json`. Target fields, target derivatives, graph/service summaries, threshold outcomes, realized failures, seeds, hashes, paths, status, acceptance/replay fields, and split labels are not predictors.\n\n## Splits\n\n- `rf_train.csv`: 70 designs / 350 runs, with targets.\n- `rf_validation.csv`: 15 designs / 75 runs, with targets.\n- `rf_test_sealed_index.csv`: 15 designs / 75 runs, identifiers and provenance only; target values are sealed.\n\nThe frozen split manifest is verified by exact-file SHA-256 `047b7cc99d83a003add8bca4bb037bce9d642774443eeef1a3df377f855db90d`. The freeze manifest SHA-256 is `b2d1fbd9510d3d828fe051b4da04f6747088ae00d78d65e0a9851dc85559844e`.\n\n## Verification\n\n`rf_construction_report.json` records row/design cardinalities, duplicate and missing-value checks, source hash verification, and test-seal status. The independent RF dataset acceptance gate is intentionally not run by this construction task.\n"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    inventory = {"schema": "satnet.stage_a_rf_dataset.artifact_inventory.v1", "inventory_policy": "self_excluding", "artifact_count_excluding_inventory": len(artifact_names), "artifacts": [{"path": f"artifacts/stage_a_rf_dataset_v1/{name}", "bytes": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)} for name in artifact_names]}
    write_json(output_dir / "artifact_inventory.json", inventory)
    print(canonical_json({"output_dir": str(output_dir), "artifact_inventory": inventory, "predictor_count": len(PREDICTOR_FIELDS), "row_counts": EXPECTED_SPLIT_ROWS, "design_counts": EXPECTED_SPLIT_DESIGNS, "targets": TARGET_FIELDS, "test_sealed": True}))


if __name__ == "__main__":
    main()
