from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from statistics import fmean, pstdev
import subprocess
from typing import Any, Iterable, Mapping, Sequence

ANALYSIS_COMMIT = "340af3f22f2bc7facc4f0e749b10490f06e80cde"
ANALYSIS_INVENTORY_SHA256 = "5ac295ba0e08797b63a2ce3f062a1995dc7aa850f11a2bf5d83a0580731afd85"
PRODUCTION_TOOLING_SHA = "9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab"
FROZEN_CONTRACT_TAG = "final-integrated-dataset-contract-v1"
FROZEN_CONTRACT_COMMIT = "a1967185e80327e4b00c1831828dc975ab6819fc"
CONTRACT_SPECIFICATION_HASH = "482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b"
GENERATION_LEDGER_SHA256 = "a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1"
REPLAY_LEDGER_SHA256 = "4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd"
FREEZE_ARCHIVE_SHA256 = "375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc"
GENERATION_FILE_COUNT = 8502
GENERATION_BYTE_COUNT = 1_336_139_056
REPLAY_FILE_COUNT = 502
REPLAY_BYTE_COUNT = 1_410_137
SERVICE_THRESHOLD = 0.80
PROPOSAL_LABEL = "PROPOSAL ONLY — NOT FROZEN — DO NOT SIMULATE"
EXPECTED_CLASS_COUNTS = {
    "train": {False: 2, True: 348},
    "validation": {False: 0, True: 75},
    "test": {False: 5, True: 70},
}
EXPECTED_NON_BREACH_RUN_IDS = (0, 1, 2, 3, 4, 5, 7)
EXPECTED_SPLIT_RUN_COUNTS = {"train": 350, "validation": 75, "test": 75}
EXPECTED_SPLIT_DESIGN_COUNTS = {"train": 70, "validation": 15, "test": 15}
REGRESSION_TARGETS = (
    "failure_adjusted_overall_service_fraction_mean",
    "failure_adjusted_overall_service_fraction_min",
    "failure_adjusted_ground_service_fraction_min",
    "space_gcc_fraction_original_min",
)
PARAMETER_FIELDS = (
    "num_planes",
    "sats_per_plane",
    "configured_satellite_count",
    "altitude_km",
    "inclination_deg",
    "phasing_factor",
    "satellite_node_failure_probability",
    "satellite_edge_failure_probability",
    "civilian_count",
    "government_count",
    "military_count",
    "total_ground_station_count",
    "ground_station_failure_probability",
)
NEIGHBOR_FEATURES = (
    "num_planes",
    "sats_per_plane",
    "altitude_km",
    "inclination_deg",
    "satellite_node_failure_probability",
    "satellite_edge_failure_probability",
    "total_ground_station_count",
    "ground_station_failure_probability",
    "civilian_fraction",
    "government_fraction",
    "military_fraction",
)
DOE_RANGES = {
    "num_planes": (4.0, 6.0),
    "sats_per_plane": (5.0, 8.0),
    "altitude_km": (300.0, 1200.0),
    "inclination_deg": (30.0, 98.0),
    "satellite_node_failure_probability": (0.0, 0.2),
    "satellite_edge_failure_probability": (0.0, 0.25),
    "total_ground_station_count": (3.0, 50.0),
    "ground_station_failure_probability": (0.0, 0.4),
    "civilian_fraction": (0.0, 1.0),
    "government_fraction": (0.0, 1.0),
    "military_fraction": (0.0, 1.0),
}
SCIENTIFIC_SIGNATURE_FIELDS = (
    "num_planes",
    "sats_per_plane",
    "altitude_km",
    "inclination_deg",
    "phasing_factor",
    "satellite_node_failure_probability",
    "satellite_edge_failure_probability",
    "civilian_count",
    "government_count",
    "military_count",
    "total_ground_station_count",
    "ground_station_failure_probability",
    "duration_minutes",
    "step_seconds",
    "epoch_iso",
    "orbital_engine",
    "max_isl_distance_km",
    "isl_policy",
    "adjacent_search_k",
    "max_inter_plane_links_per_sat",
    "satellite_failure_model",
    "minimum_elevation_deg",
    "space_gcc_threshold",
    "ground_service_threshold",
)
CLASS_GATES = {
    "minimum_non_breach_designs": {"train": 12, "validation": 4, "test": 4},
    "minimum_non_breach_runs": {"train": 50, "validation": 14, "test": 14},
    "minimum_breach_designs": {"train": 40, "validation": 10, "test": 10},
    "minimum_breach_runs": {"train": 200, "validation": 60, "test": 60},
    "maximum_majority_to_minority_run_ratio": {"train": 12.0, "validation": 10.0, "test": 10.0},
    "minimum_boundary_region_designs": {"train": 10, "validation": 3, "test": 3},
    "minimum_distinct_run_margin_values": {"train": 30, "validation": 12, "test": 12},
}
ORIGINAL_SPLIT_DESIGNS = {"train": 70, "validation": 15, "test": 15}
STAGE_A_SPLIT_DESIGNS = {"train": 20, "validation": 5, "test": 5}
STAGE_A_BOUNDARY_DESIGNS = {"train": 8, "validation": 2, "test": 2}
STAGE_B_SPLIT_DESIGNS = {"train": 60, "validation": 15, "test": 15}
STAGE_B_BOUNDARY_DESIGNS = {"train": 24, "validation": 6, "test": 6}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    source = Path(path).read_text(encoding="utf-8")
    if not source.endswith("\n"):
        raise ValueError(f"Canonical JSONL newline required: {path}")
    values = tuple(json.loads(line) for line in source[:-1].split("\n"))
    if not values or any(not isinstance(value, dict) for value in values):
        raise ValueError(f"JSONL object records required: {path}")
    return values


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Git command failed")
    return result.stdout.strip()


def _is_read_only(path: Path) -> bool:
    attributes = getattr(path.stat(), "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_READONLY", 1)
    return bool(attributes & marker)


def _is_within(path: Path, root: Path) -> bool:
    candidate = path.resolve(strict=False)
    parent = root.resolve(strict=False)
    return candidate == parent or candidate.is_relative_to(parent)


def validate_output_root(output_root: str | Path, protected_roots: Sequence[str | Path]) -> Path:
    candidate = Path(output_root)
    for protected in protected_roots:
        if _is_within(candidate, Path(protected)):
            raise ValueError(f"Audit output root is inside protected evidence: {protected}")
    return candidate


def verify_manifest(
    root: str | Path,
    manifest_path: str | Path,
    expected_count: int,
    expected_bytes: int,
) -> dict[str, Any]:
    evidence_root = Path(root)
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != expected_count:
        raise ValueError(f"Manifest count mismatch: {manifest_path}")
    files = sorted(path for path in evidence_root.rglob("*") if path.is_file())
    actual_paths = {path.relative_to(evidence_root).as_posix(): path for path in files}
    expected_paths = {row["relative_path"] for row in manifest}
    if set(actual_paths) != expected_paths:
        raise ValueError(f"Manifest path set mismatch: {manifest_path}")
    byte_count = sum(path.stat().st_size for path in files)
    if len(files) != expected_count or byte_count != expected_bytes:
        raise ValueError(f"Evidence cardinality mismatch: {evidence_root}")
    for row in manifest:
        path = actual_paths[row["relative_path"]]
        if path.stat().st_size != int(row["length_bytes"]):
            raise ValueError(f"Evidence length mismatch: {path}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Evidence SHA-256 mismatch: {path}")
    return {
        "file_count": len(files),
        "byte_count": byte_count,
        "verified_sha256_count": len(manifest),
        "read_only_file_count": sum(_is_read_only(path) for path in files),
        "all_files_read_only": all(_is_read_only(path) for path in files),
    }


def verify_frozen_evidence(
    *,
    production_tooling_root: str | Path,
    generation_root: str | Path,
    replay_root: str | Path,
    freeze_root: str | Path,
    freeze_archive: str | Path,
    freeze_archive_hash_file: str | Path,
) -> dict[str, Any]:
    tooling = Path(production_tooling_root)
    generation = Path(generation_root)
    replay = Path(replay_root)
    freeze = Path(freeze_root)
    archive = Path(freeze_archive)
    archive_hash_file = Path(freeze_archive_hash_file)
    for path in (tooling, generation, replay, freeze, archive, archive_hash_file):
        if not path.exists():
            raise FileNotFoundError(path)
    head = _git(tooling, "rev-parse", "HEAD")
    status = _git(tooling, "status", "--short")
    tag = _git(tooling, "rev-list", "-n", "1", FROZEN_CONTRACT_TAG)
    if head != PRODUCTION_TOOLING_SHA or status or tag != FROZEN_CONTRACT_COMMIT:
        raise ValueError("Production tooling identity mismatch")
    contract = read_json(tooling / "artifacts/final_integrated_dataset_contract/contract_specification.json")
    if contract.get("contract_spec_hash") != CONTRACT_SPECIFICATION_HASH:
        raise ValueError("Contract specification identity mismatch")
    generation_ledger = generation / "operational/generation_ledger.json"
    replay_ledger = replay / "replay_ledger.json"
    if sha256_file(generation_ledger) != GENERATION_LEDGER_SHA256:
        raise ValueError("Generation ledger SHA-256 mismatch")
    if sha256_file(replay_ledger) != REPLAY_LEDGER_SHA256:
        raise ValueError("Replay ledger SHA-256 mismatch")
    archive_hash = sha256_file(archive)
    if archive_hash != FREEZE_ARCHIVE_SHA256:
        raise ValueError("Freeze archive SHA-256 mismatch")
    expected_hash_record = f"{archive_hash}  {archive.name}"
    if archive_hash_file.read_text(encoding="utf-8").strip() != expected_hash_record:
        raise ValueError("Freeze archive hash record mismatch")
    generation_result = verify_manifest(
        generation,
        freeze / "generation_file_manifest_sha256.csv",
        GENERATION_FILE_COUNT,
        GENERATION_BYTE_COUNT,
    )
    replay_result = verify_manifest(
        replay,
        freeze / "replay_file_manifest_sha256.csv",
        REPLAY_FILE_COUNT,
        REPLAY_BYTE_COUNT,
    )
    sums = {}
    for line in (freeze / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    for name, digest in sums.items():
        if sha256_file(freeze / name) != digest:
            raise ValueError(f"Freeze metadata hash mismatch: {name}")
    freeze_files = [path for path in freeze.rglob("*") if path.is_file()]
    preserved = freeze_files + [archive, archive_hash_file]
    if not all(_is_read_only(path) for path in preserved):
        raise ValueError("Freeze metadata or archive is not read-only")
    return {
        "verification_status": "passed",
        "production_tooling_sha": head,
        "production_tooling_clean": True,
        "frozen_contract_tag": FROZEN_CONTRACT_TAG,
        "frozen_contract_commit": tag,
        "contract_specification_hash": CONTRACT_SPECIFICATION_HASH,
        "generation_ledger_sha256": GENERATION_LEDGER_SHA256,
        "replay_ledger_sha256": REPLAY_LEDGER_SHA256,
        "freeze_archive_sha256": archive_hash,
        "archive_hash_record_verified": True,
        "generation": generation_result,
        "replay": replay_result,
        "combined": {
            "file_count": generation_result["file_count"] + replay_result["file_count"],
            "byte_count": generation_result["byte_count"] + replay_result["byte_count"],
            "verified_sha256_count": generation_result["verified_sha256_count"]
            + replay_result["verified_sha256_count"],
        },
        "freeze_metadata_file_count": len(freeze_files),
        "freeze_metadata_all_read_only": True,
        "archive_read_only": _is_read_only(archive),
        "archive_hash_record_read_only": _is_read_only(archive_hash_file),
    }


def verify_analysis_outputs(
    analysis_output_root: str | Path,
    tracked_analysis_root: str | Path,
) -> dict[str, Any]:
    external = Path(analysis_output_root)
    tracked = Path(tracked_analysis_root)
    inventory_path = external / "analysis_inventory.json"
    if sha256_file(inventory_path) != ANALYSIS_INVENTORY_SHA256:
        raise ValueError("Analysis inventory SHA-256 mismatch")
    inventory = read_json(inventory_path)
    verified = 0
    for record in inventory["outputs"]:
        path = external / record["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["byte_length"]):
            raise ValueError(f"Analysis output length mismatch: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Analysis output SHA-256 mismatch: {path}")
        verified += 1
    tracked_files = {
        path.relative_to(tracked).as_posix(): path
        for path in tracked.rglob("*")
        if path.is_file() and path.name != "analysis_inventory.json"
    }
    semantic_mismatches: list[str] = []
    for relative, tracked_path in tracked_files.items():
        external_path = external / relative
        if not external_path.is_file():
            semantic_mismatches.append(relative)
            continue
        if tracked_path.suffix == ".json":
            equal = json.loads(tracked_path.read_text(encoding="utf-8")) == json.loads(
                external_path.read_text(encoding="utf-8")
            )
        elif tracked_path.suffix == ".csv":
            with tracked_path.open("r", encoding="utf-8", newline="") as handle:
                left = list(csv.DictReader(handle))
            with external_path.open("r", encoding="utf-8", newline="") as handle:
                right = list(csv.DictReader(handle))
            equal = left == right
        else:
            equal = tracked_path.read_text(encoding="utf-8").splitlines() == external_path.read_text(
                encoding="utf-8"
            ).splitlines()
        if not equal:
            semantic_mismatches.append(relative)
    external_relatives = {record["relative_path"] for record in inventory["outputs"]}
    external_only = sorted(external_relatives - set(tracked_files))
    expected_external_only = sorted(
        {
            "run_level_class_support.csv",
            "design_level_class_support.csv",
            "temporal_breach_summary.csv",
        }
    )
    if semantic_mismatches or external_only != expected_external_only:
        raise ValueError(
            f"Tracked/external analysis reconciliation failed: {semantic_mismatches}, {external_only}"
        )
    return {
        "verification_status": "passed",
        "analysis_inventory_sha256": ANALYSIS_INVENTORY_SHA256,
        "inventory_output_count": verified,
        "tracked_semantic_mismatch_count": 0,
        "canonical_external_only_outputs": external_only,
        "tracked_checkout_inventory_sha256": sha256_file(tracked / "analysis_inventory.json"),
        "line_ending_note": "Canonical external bytes are authoritative; tracked text files are semantically equal after checkout line-ending conversion.",
    }


def _numeric(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(f"Numeric field invalid: {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite field: {field}")
    return result


def _longest_streak(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _recovery_count(values: Sequence[bool]) -> int:
    return sum(previous and not current for previous, current in zip(values, values[1:]))


def _run_paths(root: Path, run_id: int) -> dict[str, Path]:
    run_root = root / f"run_{run_id:03d}"
    return {
        "design": run_root / "input/design_record.json",
        "run": run_root / "input/run_record.json",
        "target": run_root / "targets/target.json",
        "steps": run_root / "g5/ground_failure_service_steps.jsonl",
        "summary": run_root / "g5/ground_failure_service_run.jsonl",
    }


def extract_run(generation_root: str | Path, run_id: int) -> dict[str, Any]:
    paths = _run_paths(Path(generation_root), run_id)
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"Authoritative files missing for run {run_id}")
    design = read_json(paths["design"])
    run = read_json(paths["run"])
    target = read_json(paths["target"])
    steps = read_jsonl(paths["steps"])
    summaries = read_jsonl(paths["summary"])
    if len(steps) != 11 or len(summaries) != 1:
        raise ValueError(f"Run {run_id} temporal cardinality mismatch")
    if not (
        run["run_id"] == target["run_id"] == run_id
        and run["run_key"] == target["run_key"]
        and run["design_id"] == design["design_id"]
        and run["design_index"] == design["design_index"]
        and run_id == int(run["design_index"]) * 5 + int(run["realization_index"])
    ):
        raise ValueError(f"Run {run_id} identity mismatch")
    metrics = [step["metrics"] for step in steps]
    indices = [int(value["timestep_index"]) for value in metrics]
    if indices != list(range(11)):
        raise ValueError(f"Run {run_id} temporal ordering mismatch")
    timestamps = [datetime.fromisoformat(str(value["timestamp_utc"])) for value in metrics]
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"Run {run_id} timestamps are not strictly increasing")
    overall = [_numeric(value["failure_adjusted_overall_service_fraction"], "overall") for value in metrics]
    ground = [_numeric(value["failure_adjusted_ground_service_fraction"], "ground") for value in metrics]
    space = [_numeric(value["space_gcc_fraction_original"], "space") for value in metrics]
    breaches = [value < SERVICE_THRESHOLD for value in overall]
    recorded_met = [bool(value["overall_threshold_met"]) for value in metrics]
    if recorded_met != [not value for value in breaches]:
        raise ValueError(f"Run {run_id} exact threshold behavior mismatch")
    overall_minimum = _numeric(
        target["failure_adjusted_overall_service_fraction_min"],
        "failure_adjusted_overall_service_fraction_min",
    )
    overall_mean = _numeric(
        target["failure_adjusted_overall_service_fraction_mean"],
        "failure_adjusted_overall_service_fraction_mean",
    )
    margin = overall_minimum - SERVICE_THRESHOLD
    target_breach = bool(target["overall_threshold_breach_any"])
    if overall_minimum != min(overall) or overall_mean != math.fsum(overall) / len(overall):
        raise ValueError(f"Run {run_id} target aggregate mismatch")
    if target_breach != (margin < 0.0) or target_breach != any(breaches):
        raise ValueError(f"Run {run_id} target boundary mismatch")
    summary = summaries[0]["summary"]
    if (
        bool(summary["overall_threshold_breach_any"]) != target_breach
        or int(summary["overall_threshold_breach_timestep_count"]) != sum(breaches)
        or _numeric(summary["failure_adjusted_overall_service_fraction_min"], "summary_min")
        != overall_minimum
    ):
        raise ValueError(f"Run {run_id} G5 summary mismatch")
    first_breach = next((index for index, value in enumerate(breaches) if value), None)
    last_breach = next(
        (index for index in range(len(breaches) - 1, -1, -1) if breaches[index]),
        None,
    )
    minimum_timestep = min(range(len(overall)), key=lambda index: (overall[index], index))
    row: dict[str, Any] = {
        "run_id": run_id,
        "run_key": run["run_key"],
        "design_id": design["design_id"],
        "design_index": int(design["design_index"]),
        "realization_id": run["realization_id"],
        "realization_index": int(run["realization_index"]),
        "split": run["split_assignment"],
        "doe_stratum": design["doe_stratum"],
        "satellite_seed": int(run["satellite_seed"]),
        "ground_failure_seed": int(run["ground_failure_seed"]),
        "ground_selection_seed": int(run["ground_selection_seed"]),
        "overall_threshold_breach_any": target_breach,
        "failure_adjusted_overall_service_fraction_mean": overall_mean,
        "failure_adjusted_overall_service_fraction_min": overall_minimum,
        "failure_adjusted_ground_service_fraction_min": _numeric(
            target["failure_adjusted_ground_service_fraction_min"], "ground_min"
        ),
        "space_gcc_fraction_original_min": _numeric(
            target["space_gcc_fraction_original_min"], "space_min"
        ),
        "overall_boundary_margin": margin,
        "sampled_state_count": len(overall),
        "temporal_breach_count": sum(breaches),
        "temporal_breach_fraction": sum(breaches) / len(breaches),
        "first_overall_breach_timestep": first_breach,
        "last_overall_breach_timestep": last_breach,
        "longest_overall_breach_streak": _longest_streak(breaches),
        "number_of_recoveries_above_threshold": _recovery_count(breaches),
        "minimum_overall_service_timestep": minimum_timestep,
        "overall_service_sequence": overall,
        "ground_service_sequence": ground,
        "space_gcc_sequence": space,
    }
    numeric_design_fields = {
        "altitude_km",
        "inclination_deg",
        "satellite_node_failure_probability",
        "satellite_edge_failure_probability",
        "ground_station_failure_probability",
        "max_isl_distance_km",
        "minimum_elevation_deg",
        "space_gcc_threshold",
        "ground_service_threshold",
    }
    for field, value in design.items():
        if field not in row:
            row[field] = _numeric(value, field) if field in numeric_design_fields else value
    return row


def extract_corpus(generation_root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs = [extract_run(generation_root, run_id) for run_id in range(500)]
    if [row["run_id"] for row in runs] != list(range(500)):
        raise ValueError("Run ID sequence mismatch")
    if len({row["run_key"] for row in runs}) != 500:
        raise ValueError("Run key uniqueness mismatch")
    if len({(row["design_id"], row["realization_id"]) for row in runs}) != 500:
        raise ValueError("Design-realization uniqueness mismatch")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        groups[row["design_id"]].append(row)
    if sorted(groups) != [f"D{index:03d}" for index in range(100)]:
        raise ValueError("Design identity sequence mismatch")
    designs: list[dict[str, Any]] = []
    for design_id in sorted(groups):
        group = sorted(groups[design_id], key=lambda row: row["realization_index"])
        if len(group) != 5 or {row["realization_index"] for row in group} != set(range(5)):
            raise ValueError(f"Realization cardinality mismatch: {design_id}")
        if len({row["split"] for row in group}) != 1:
            raise ValueError(f"Split colocation mismatch: {design_id}")
        first = group[0]
        expected_stratum = (
            "pilot_anchor"
            if first["design_index"] <= 4
            else "transition"
            if first["design_index"] <= 39
            else "global"
        )
        if first["doe_stratum"] != expected_stratum:
            raise ValueError(f"DOE stratum mismatch: {design_id}")
        margins = [float(row["overall_boundary_margin"]) for row in group]
        designs.append(
            {
                **{
                    field: first[field]
                    for field in first
                    if field
                    not in {
                        "run_id",
                        "run_key",
                        "realization_id",
                        "realization_index",
                        "satellite_seed",
                        "ground_failure_seed",
                        "overall_threshold_breach_any",
                        "failure_adjusted_overall_service_fraction_mean",
                        "failure_adjusted_overall_service_fraction_min",
                        "failure_adjusted_ground_service_fraction_min",
                        "space_gcc_fraction_original_min",
                        "overall_boundary_margin",
                        "sampled_state_count",
                        "temporal_breach_count",
                        "temporal_breach_fraction",
                        "first_overall_breach_timestep",
                        "last_overall_breach_timestep",
                        "longest_overall_breach_streak",
                        "number_of_recoveries_above_threshold",
                        "minimum_overall_service_timestep",
                        "overall_service_sequence",
                        "ground_service_sequence",
                        "space_gcc_sequence",
                    }
                },
                "split": first["split"],
                "non_breach_realization_count": sum(
                    not row["overall_threshold_breach_any"] for row in group
                ),
                "breach_realization_count": sum(
                    row["overall_threshold_breach_any"] for row in group
                ),
                "minimum_boundary_margin": min(margins),
                "mean_boundary_margin": fmean(margins),
                "maximum_boundary_margin": max(margins),
                "mean_temporal_breach_fraction": fmean(
                    row["temporal_breach_fraction"] for row in group
                ),
                "run_ids": [row["run_id"] for row in group],
            }
        )
    return runs, designs


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = math.fsum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_scale = math.fsum((value - left_mean) ** 2 for value in left)
    right_scale = math.fsum((value - right_mean) ** 2 for value in right)
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / math.sqrt(left_scale * right_scale)


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_average_ranks(left), _average_ranks(right))


def neighbor_vector(row: Mapping[str, Any]) -> tuple[float, ...]:
    total = float(row["total_ground_station_count"])
    values = {
        key: float(row[key])
        for key in NEIGHBOR_FEATURES
        if not key.endswith("_fraction")
    }
    values.update(
        civilian_fraction=float(row["civilian_count"]) / total,
        government_fraction=float(row["government_count"]) / total,
        military_fraction=float(row["military_count"]) / total,
    )
    return tuple(
        (values[key] - DOE_RANGES[key][0]) / (DOE_RANGES[key][1] - DOE_RANGES[key][0])
        for key in NEIGHBOR_FEATURES
    )


def normalized_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return math.dist(neighbor_vector(left), neighbor_vector(right))


def scientific_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    numeric_fields = {
        "num_planes",
        "sats_per_plane",
        "altitude_km",
        "inclination_deg",
        "phasing_factor",
        "satellite_node_failure_probability",
        "satellite_edge_failure_probability",
        "civilian_count",
        "government_count",
        "military_count",
        "total_ground_station_count",
        "ground_station_failure_probability",
        "duration_minutes",
        "step_seconds",
        "max_isl_distance_km",
        "adjacent_search_k",
        "max_inter_plane_links_per_sat",
        "minimum_elevation_deg",
        "space_gcc_threshold",
        "ground_service_threshold",
    }
    return tuple(float(row[field]) if field in numeric_fields else str(row[field]) for field in SCIENTIFIC_SIGNATURE_FIELDS)


def reproduce_corpus(generation_root: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    runs, designs = extract_corpus(generation_root)
    split_runs = Counter(row["split"] for row in runs)
    split_designs = Counter(row["split"] for row in designs)
    class_counts = {
        split: {
            False: sum(
                row["split"] == split and not row["overall_threshold_breach_any"] for row in runs
            ),
            True: sum(
                row["split"] == split and row["overall_threshold_breach_any"] for row in runs
            ),
        }
        for split in ("train", "validation", "test")
    }
    non_breach_ids = tuple(
        row["run_id"] for row in runs if not row["overall_threshold_breach_any"]
    )
    non_breach_designs = [
        row["design_id"] for row in designs if row["non_breach_realization_count"] > 0
    ]
    if dict(split_runs) != EXPECTED_SPLIT_RUN_COUNTS:
        raise ValueError("Run split counts differ")
    if dict(split_designs) != EXPECTED_SPLIT_DESIGN_COUNTS:
        raise ValueError("Design split counts differ")
    if class_counts != EXPECTED_CLASS_COUNTS:
        raise ValueError("Classification counts differ")
    if non_breach_ids != EXPECTED_NON_BREACH_RUN_IDS:
        raise ValueError("Non-breach run identities differ")
    if non_breach_designs != ["D000", "D001"]:
        raise ValueError("Non-breach design identities differ")
    stratum_counts = Counter(row["doe_stratum"] for row in designs)
    if stratum_counts != {"pilot_anchor": 5, "transition": 35, "global": 60}:
        raise ValueError("DOE stratum counts differ")
    if any(
        not row["overall_threshold_breach_any"] and row["doe_stratum"] != "pilot_anchor"
        for row in runs
    ):
        raise ValueError("Transition or global non-breach outcome found")
    summary = {
        "verification_status": "independently_reproduced",
        "run_count": len(runs),
        "design_count": len(designs),
        "realizations_per_design": 5,
        "split_run_counts": dict(split_runs),
        "split_design_counts": dict(split_designs),
        "classification_counts": {
            split: {"false": values[False], "true": values[True]}
            for split, values in class_counts.items()
        },
        "non_breach_run_ids": list(non_breach_ids),
        "non_breach_design_ids": non_breach_designs,
        "zero_of_five_non_breach_design_count": sum(
            row["non_breach_realization_count"] == 0 for row in designs
        ),
        "doe_stratum_design_counts": dict(stratum_counts),
        "run_ids_exact_0_through_499": True,
        "unique_run_key_count": len({row["run_key"] for row in runs}),
        "unique_design_realization_pair_count": len(
            {(row["design_id"], row["realization_id"]) for row in runs}
        ),
        "all_realizations_colocated": True,
        "transition_and_global_non_breach_count": 0,
    }
    return summary, runs, designs


def boundary_audit(runs: Sequence[dict[str, Any]], designs: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    margins = [float(row["overall_boundary_margin"]) for row in runs]
    design_margins = [float(row["mean_boundary_margin"]) for row in designs]
    windows = {
        format(width, ".3f"): sum(abs(value) <= width for value in margins)
        for width in (0.01, 0.025, 0.05, 0.10)
    }
    below = sum(value < -0.20 for value in margins)
    design_below = sum(value < -0.20 for value in design_margins)
    nearest_breach = min(
        (row for row in runs if row["overall_threshold_breach_any"]),
        key=lambda row: (abs(float(row["overall_boundary_margin"])), int(row["run_id"])),
    )
    if below != 480 or design_below != 96 or windows != {
        "0.010": 2,
        "0.025": 2,
        "0.050": 2,
        "0.100": 3,
    }:
        raise ValueError("Boundary counts differ")
    if nearest_breach["run_id"] != 6 or nearest_breach["design_id"] != "D001":
        raise ValueError("Nearest breach identity differs")
    rows = [
        {
            "run_id": row["run_id"],
            "run_key": row["run_key"],
            "design_id": row["design_id"],
            "split": row["split"],
            "overall_minimum": row["failure_adjusted_overall_service_fraction_min"],
            "threshold": SERVICE_THRESHOLD,
            "signed_margin": row["overall_boundary_margin"],
            "absolute_margin": abs(float(row["overall_boundary_margin"])),
            "recorded_breach": row["overall_threshold_breach_any"],
            "reproduced_breach": float(row["overall_boundary_margin"]) < 0.0,
            "polarity_match": row["overall_threshold_breach_any"]
            == (float(row["overall_boundary_margin"]) < 0.0),
            "exact_threshold": float(row["overall_boundary_margin"]) == 0.0,
        }
        for row in runs
    ]
    summary = {
        "status": "passed",
        "threshold": SERVICE_THRESHOLD,
        "canonical_comparison": "binary64 service value >= binary64 0.80 is threshold-met; equality is non-breach",
        "all_500_polarities_match": all(row["polarity_match"] for row in rows),
        "exact_threshold_run_ids": [row["run_id"] for row in rows if row["exact_threshold"]],
        "runs_below_negative_0_20_exclusive": below,
        "design_means_below_negative_0_20_exclusive": design_below,
        "symmetric_window_counts_inclusive": windows,
        "nearest_breach": {
            "run_id": nearest_breach["run_id"],
            "design_id": nearest_breach["design_id"],
            "margin": nearest_breach["overall_boundary_margin"],
        },
        "d000_margins": [
            row["overall_boundary_margin"] for row in runs if row["design_id"] == "D000"
        ],
        "d001_margins": [
            row["overall_boundary_margin"] for row in runs if row["design_id"] == "D001"
        ],
        "endpoint_treatment": {
            "below_negative_0_20": "exclusive",
            "symmetric_windows": "inclusive at both endpoints",
        },
    }
    return summary, rows


def temporal_audit(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_stratum = {
        stratum: fmean(
            row["temporal_breach_fraction"]
            for row in runs
            if row["doe_stratum"] == stratum
        )
        for stratum in ("pilot_anchor", "transition", "global")
    }
    d001 = [
        {
            "run_id": row["run_id"],
            "breach_count": row["temporal_breach_count"],
            "breach_fraction": row["temporal_breach_fraction"],
            "first_breach_timestep": row["first_overall_breach_timestep"],
            "last_breach_timestep": row["last_overall_breach_timestep"],
            "longest_breach_streak": row["longest_overall_breach_streak"],
            "recovery_count": row["number_of_recoveries_above_threshold"],
            "minimum_service_timestep": row["minimum_overall_service_timestep"],
        }
        for row in runs
        if row["design_id"] == "D001"
    ]
    return {
        "status": "passed",
        "authoritative_source": "persisted generation run G5 ground_failure_service_steps.jsonl",
        "sampled_state_count_per_run": 11,
        "total_sampled_state_count": sum(row["sampled_state_count"] for row in runs),
        "temporal_ordering": "persisted order exactly timestep_index 0 through 10 with strictly increasing timestamps",
        "breach_comparison": "failure_adjusted_overall_service_fraction < 0.80",
        "mean_temporal_breach_fraction_by_stratum": by_stratum,
        "d001": d001,
    }


def parameter_audit(runs: Sequence[dict[str, Any]], designs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(row["overall_boundary_margin"]) for row in runs]
    correlations = {
        field: spearman([float(row[field]) for row in runs], margins)
        for field in PARAMETER_FIELDS
    }
    non_breach = [row for row in runs if not row["overall_threshold_breach_any"]]
    support = {
        field: sorted({float(row[field]) for row in non_breach})
        for field in (
            "num_planes",
            "sats_per_plane",
            "configured_satellite_count",
            "altitude_km",
            "inclination_deg",
            "satellite_node_failure_probability",
            "satellite_edge_failure_probability",
            "ground_station_failure_probability",
        )
    }
    design_support = [
        {
            "design_id": row["design_id"],
            "non_breach_realization_count": row["non_breach_realization_count"],
            **{field: row[field] for field in support},
        }
        for row in designs
        if row["non_breach_realization_count"] > 0
    ]
    return {
        "status": "passed",
        "method": "run-level Spearman rank correlation using average ranks for ties and Pearson correlation of ranks",
        "normalization": "none",
        "missing_value_handling": "fail closed; no missing values observed",
        "run_count": len(runs),
        "correlations": correlations,
        "non_breach_run_support": support,
        "non_breach_design_support": design_support,
        "interpretation": "Observed support consists of seven realizations from only D000 and D001. Numeric intervals between their values are DOE-containing ranges, not independently supported resilient regions.",
    }


def nearest_neighbor_audit(designs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source_id in ("D000", "D001"):
        source = next(row for row in designs if row["design_id"] == source_id)
        neighbors = sorted(
            (
                (normalized_distance(source, candidate), candidate)
                for candidate in designs
                if candidate["design_id"] != source_id
            ),
            key=lambda value: (value[0], value[1]["design_index"]),
        )
        result[source_id] = [
            {
                "rank": rank,
                "design_id": row["design_id"],
                "distance": distance,
                "non_breach_realization_count": row["non_breach_realization_count"],
                "mean_boundary_margin": row["mean_boundary_margin"],
            }
            for rank, (distance, row) in enumerate(neighbors[:8], start=1)
        ]
    if result["D000"][0]["design_id"] != "D001":
        raise ValueError("D000 nearest neighbor differs")
    if result["D001"][0]["design_id"] != "D022":
        raise ValueError("D001 nearest neighbor differs")
    if any(row["non_breach_realization_count"] for row in result["D001"]):
        raise ValueError("D001 neighbor class support differs")
    return {
        "status": "passed",
        "features": list(NEIGHBOR_FEATURES),
        "normalization": "predeclared full DOE min-max ranges",
        "zero_range_handling": "not applicable; all 11 declared ranges are nonzero",
        "distance_metric": "Euclidean",
        "categorical_variables": [],
        "fixed_parameters_excluded": True,
        "failure_probabilities_included": True,
        "tie_handling": "ascending design_index",
        "neighbors": result,
        "cluster_conclusion": "No resilient local cluster is observed under this declared metric; this is descriptive and metric-dependent.",
    }


def regression_audit(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        by_split[split] = {}
        for target in REGRESSION_TARGETS:
            values = [float(row[target]) for row in runs if row["split"] == split]
            by_split[split][target] = {
                "count": len(values),
                "mean": fmean(values),
                "population_standard_deviation": pstdev(values),
                "unique_value_count": len(set(values)),
                "nonzero_spread": pstdev(values) > 0.0,
            }
    return {
        "status": "passed",
        "primary_regression_target": "failure_adjusted_overall_service_fraction_mean",
        "by_split": by_split,
        "all_four_targets_have_nonzero_spread_in_every_split": all(
            values["nonzero_spread"]
            for split_values in by_split.values()
            for values in split_values.values()
        ),
        "assessment": "Supports review of a separately predeclared regression-only contract; does not approve regression use.",
    }


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _proposal_split(row: Mapping[str, Any]) -> str:
    return str(row.get("intended_split", row.get("split", "")))


def _proposal_id(row: Mapping[str, Any]) -> str:
    return str(row.get("augmentation_design_id", row.get("design_id", "")))


def proposal_audit(
    *,
    analysis_output_root: str | Path,
    runs: Sequence[dict[str, Any]],
    designs: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(analysis_output_root)
    proposal_rows = _load_csv(root / "recommended_augmentation_design.csv")
    contract = read_json(root / "augmentation_contract_proposal.json")
    if len(proposal_rows) != 90:
        raise ValueError("Stage B proposal row count differs")
    expected_ids = [f"AUGV1-D{index:03d}" for index in range(90)]
    observed_ids = [row["augmentation_design_id"] for row in proposal_rows]
    original_ids = {row["design_id"] for row in designs}
    original_run_ids = {row["run_id"] for row in runs}
    if observed_ids != expected_ids or set(observed_ids) & original_ids:
        raise ValueError("Stage B design identity collision")
    if any(
        row["proposal_label"] != PROPOSAL_LABEL
        or row["proposal_status"] != "NOT_FROZEN"
        or row["simulation_authorized"].lower() != "false"
        or row["seed_status"] != "NOT_ISSUED"
        for row in proposal_rows
    ):
        raise ValueError("Stage B proposal authorization state differs")
    if contract["proposal_status"] != "NOT_FROZEN" or contract["simulation_authorized"] is not False:
        raise ValueError("Contract proposal authorization state differs")
    signatures = [scientific_signature(row) for row in proposal_rows]
    original_signatures = {scientific_signature(row) for row in designs}
    duplicate_new = len(signatures) - len(set(signatures))
    duplicate_original = sum(signature in original_signatures for signature in signatures)
    if duplicate_new or duplicate_original:
        raise ValueError("Stage B scientific design duplicate detected")
    for row in proposal_rows:
        if int(row["configured_satellite_count"]) != int(row["num_planes"]) * int(
            row["sats_per_plane"]
        ):
            raise ValueError(f"Configured satellite count mismatch: {_proposal_id(row)}")
        if int(row["civilian_count"]) + int(row["government_count"]) + int(
            row["military_count"]
        ) != int(row["total_ground_station_count"]):
            raise ValueError(f"Station composition mismatch: {_proposal_id(row)}")
        for field, (lower, upper) in DOE_RANGES.items():
            if field.endswith("_fraction"):
                total = float(row["total_ground_station_count"])
                value = float(row[field.removesuffix("_fraction") + "_count"]) / total
            else:
                value = float(row[field])
            if not lower <= value <= upper:
                raise ValueError(f"DOE range violation: {_proposal_id(row)} {field}")
    region_split = Counter(
        (row["intended_region"], row["intended_split"]) for row in proposal_rows
    )
    expected_region_split = {
        ("resilient_core", "train"): 24,
        ("resilient_core", "validation"): 6,
        ("resilient_core", "test"): 6,
        ("boundary", "train"): 24,
        ("boundary", "validation"): 6,
        ("boundary", "test"): 6,
        ("global_control", "train"): 12,
        ("global_control", "validation"): 3,
        ("global_control", "test"): 3,
    }
    if dict(region_split) != expected_region_split:
        raise ValueError("Stage B region/split arithmetic differs")
    all_designs = [*designs, *proposal_rows]
    pair_rows: list[dict[str, Any]] = []
    nearest_candidates: dict[tuple[str, str], tuple[float, Mapping[str, Any], Mapping[str, Any]]] = {}
    cross_split_pairs_within_radius = 0
    minimum_new_new_cross_split = math.inf
    minimum_new_original_cross_split = math.inf
    for left_index, left in enumerate(all_designs):
        for right in all_designs[left_index + 1 :]:
            left_new = "augmentation_design_id" in left
            right_new = "augmentation_design_id" in right
            if not (left_new or right_new):
                continue
            distance = normalized_distance(left, right)
            left_split = _proposal_split(left)
            right_split = _proposal_split(right)
            relation = "new_new" if left_new and right_new else "new_original"
            cross_split = left_split != right_split
            if cross_split and distance <= 0.10:
                cross_split_pairs_within_radius += 1
            if cross_split and relation == "new_new":
                minimum_new_new_cross_split = min(minimum_new_new_cross_split, distance)
            if cross_split and relation == "new_original":
                minimum_new_original_cross_split = min(minimum_new_original_cross_split, distance)
            if left_new:
                key = (_proposal_id(left), relation)
                current = nearest_candidates.get(key)
                if current is None or distance < current[0]:
                    nearest_candidates[key] = (distance, left, right)
            if right_new:
                key = (_proposal_id(right), relation)
                current = nearest_candidates.get(key)
                if current is None or distance < current[0]:
                    nearest_candidates[key] = (distance, right, left)
    for (source_id, relation), (distance, source, neighbor) in sorted(nearest_candidates.items()):
        pair_rows.append(
            {
                "record_type": "nearest_by_relation",
                "relation": relation,
                "source_design_id": source_id,
                "source_split": _proposal_split(source),
                "neighbor_design_id": _proposal_id(neighbor),
                "neighbor_split": _proposal_split(neighbor),
                "normalized_euclidean_distance": distance,
                "cross_split": _proposal_split(source) != _proposal_split(neighbor),
                "within_predeclared_radius_0_10": distance <= 0.10,
            }
        )
    collision_rows = [
        {
            "check": "original_design_id_collision",
            "collision_count": len(set(observed_ids) & original_ids),
            "status": "passed",
            "detail": "AUGV1-D000..D089 is disjoint from D000..D099",
        },
        {
            "check": "original_integer_run_id_collision",
            "collision_count": sum(
                template in {str(value) for value in original_run_ids}
                for template in (row["run_identity_template"] for row in proposal_rows)
            ),
            "status": "passed",
            "detail": "AUGV1-Dxxx-R00..R04 does not reuse frozen integer run IDs 0..499",
        },
        {
            "check": "duplicate_new_scientific_parameter_vector",
            "collision_count": duplicate_new,
            "status": "passed",
            "detail": "Full comparable scientific signatures are unique",
        },
        {
            "check": "duplicate_original_scientific_parameter_vector",
            "collision_count": duplicate_original,
            "status": "passed",
            "detail": "No proposed signature exactly duplicates an original design",
        },
        {
            "check": "stage_a_namespace_defined",
            "collision_count": 0,
            "status": "failed",
            "detail": "No Stage A design/run namespace or exact Stage A table is defined; Stage B already occupies AUGV1-D000..D089",
        },
    ]
    stage_a = contract["stage_a"]
    stage_b = contract["stage_b"]
    stage_a_arithmetic = {
        "design_count": stage_a["design_count"],
        "run_count": stage_a["run_count"],
        "realizations_per_design": stage_a["realization_count_per_design"],
        "region_total": sum(stage_a["region_allocation"].values()),
        "split_total": sum(stage_a["split_allocation"].values()),
        "region_split_total": sum(
            sum(values.values()) for values in stage_a["region_split_allocation"].values()
        ),
    }
    stage_b_arithmetic = {
        "design_count": stage_b["design_count"],
        "run_count": stage_b["run_count"],
        "realizations_per_design": stage_b["realizations_per_design"],
        "region_total": sum(sum(values.values()) for values in stage_b["region_split_allocation"].values()),
        "split_design_counts": {
            split: sum(values[split] for values in stage_b["region_split_allocation"].values())
            for split in ("train", "validation", "test")
        },
    }
    stage_a_review = {
        "status": "not_freeze_ready",
        "arithmetic": stage_a_arithmetic,
        "arithmetic_passed": stage_a_arithmetic
        == {
            "design_count": 30,
            "run_count": 150,
            "realizations_per_design": 5,
            "region_total": 30,
            "split_total": 30,
            "region_split_total": 30,
        },
        "discovery_phase_assessment": "Scientifically appropriate as discovery only if outcomes inform Stage B solely through a new contract.",
        "selected_partition_approach": "B. Discovery plus internal holdout",
        "partition_tradeoff": "A five-design internal holdout provides an outcome-independent check but lowers development coverage. It must not influence Stage B and must not be called a final evaluation set before a final combined-corpus contract declares its role.",
        "holdout_policy": "Stage A train/development and validation may inform Stage B; the five-design internal holdout remains sealed until the predeclared final acceptance point.",
        "exact_stage_a_design_table_present": False,
        "stage_a_namespace_present": False,
        "stage_a_seed_manifest_present": False,
        "stage_a_output_roots_present": False,
        "existing_discovery_criteria_assessment": "Operational completion and core support are useful, but 'four boundary designs collectively sample both signs' does not require any design to exhibit mixed realizations and does not localize a usable transition.",
        "proposed_stop_go_criteria": {
            "operational": "Exactly 30 unique designs and 150 fixed-identity runs generate and replay with no substitutions, retries under new identities, or evidence mutations.",
            "core_reproduction": "Among development plus validation, at least two distinct resilient-core designs each produce at least three of five non-breach realizations.",
            "boundary_mixing": "Among development plus validation, at least three distinct boundary designs are individually mixed, each producing at least one breach and one non-breach realization.",
            "transition_localization": "At least four distinct boundary designs have at least one run within inclusive ±0.10 margin, with both margin signs represented across at least two distinct designs per sign.",
            "independent_class_support": "Development plus validation contain at least four non-breach-supporting designs and at least four breach-supporting designs; mixed designs may support both counts but must be reported explicitly.",
            "realization_stability": "Report each design's 0..5 non-breach count and margin spread; no pooled run count may substitute for distinct-design support.",
            "control_consistency": "At least four of the five development-plus-validation global controls remain 0/5 non-breach; otherwise investigate population shift before Stage B design.",
            "go": "All operational, core, boundary, transition, independent-support, and control criteria pass using only development plus validation.",
            "revise": "Core support exists but boundary localization or control consistency fails; create a new Stage B proposal and contract without consulting the sealed holdout.",
            "stop": "Core reproduction or independent class support fails; do not freeze Stage B.",
        },
        "binding_corrections": [
            "Create an exact 30-design Stage A table with admissible ranges and preassigned development/validation/internal-holdout blocks.",
            "Define disjoint Stage A design and run namespaces before any seed manifest is created.",
            "Replace the collective-sign boundary criterion with design-level mixed-realization and near-boundary criteria.",
            "Declare whether Stage A evidence will later join the final combined corpus or remain discovery-only.",
            "Freeze domain-separated deterministic seed derivation and explicit new output roots in the future Stage A contract.",
        ],
    }
    stage_b_review = {
        "status": "proposal_only_not_freeze_ready",
        "arithmetic": stage_b_arithmetic,
        "arithmetic_passed": stage_b_arithmetic
        == {
            "design_count": 90,
            "run_count": 450,
            "realizations_per_design": 5,
            "region_total": 90,
            "split_design_counts": {"train": 60, "validation": 15, "test": 15},
        },
        "proposal_status": contract["proposal_status"],
        "simulation_authorized": contract["simulation_authorized"],
        "unique_design_count": len(set(observed_ids)),
        "exact_duplicate_new_count": duplicate_new,
        "exact_duplicate_original_count": duplicate_original,
        "minimum_new_new_cross_split_distance": minimum_new_new_cross_split,
        "minimum_new_original_cross_split_distance": minimum_new_original_cross_split,
        "cross_split_pairs_within_0_10": cross_split_pairs_within_radius,
        "freeze_readiness": "Stage B must remain NOT_FROZEN until Stage A development/validation findings are reviewed under a new proposal and independently audited.",
    }
    return stage_a_review, stage_b_review, collision_rows, pair_rows


def _scope_counts() -> dict[str, dict[str, dict[str, int]]]:
    scopes: dict[str, dict[str, dict[str, int]]] = {}
    for name, design_parts, boundary_parts in (
        ("stage_a_augmentation_only", [STAGE_A_SPLIT_DESIGNS], [STAGE_A_BOUNDARY_DESIGNS]),
        ("stage_b_augmentation_only", [STAGE_B_SPLIT_DESIGNS], [STAGE_B_BOUNDARY_DESIGNS]),
        (
            "original_plus_stage_b",
            [ORIGINAL_SPLIT_DESIGNS, STAGE_B_SPLIT_DESIGNS],
            [STAGE_B_BOUNDARY_DESIGNS],
        ),
        (
            "original_plus_stage_a_plus_stage_b",
            [ORIGINAL_SPLIT_DESIGNS, STAGE_A_SPLIT_DESIGNS, STAGE_B_SPLIT_DESIGNS],
            [STAGE_A_BOUNDARY_DESIGNS, STAGE_B_BOUNDARY_DESIGNS],
        ),
    ):
        scopes[name] = {}
        for split in ("train", "validation", "test"):
            designs = sum(part[split] for part in design_parts)
            scopes[name][split] = {
                "designs": designs,
                "runs": designs * 5,
                "boundary_designs": sum(part[split] for part in boundary_parts),
            }
    return scopes


def _joint_class_gate_feasible(total: int, non_breach: int, breach: int) -> bool:
    return non_breach + breach <= total


def gate_feasibility_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, by_split in _scope_counts().items():
        for split, available in by_split.items():
            requirements = {
                "minimum_non_breach_designs": (
                    CLASS_GATES["minimum_non_breach_designs"][split],
                    available["designs"],
                ),
                "minimum_breach_designs": (
                    CLASS_GATES["minimum_breach_designs"][split],
                    available["designs"],
                ),
                "minimum_non_breach_runs": (
                    CLASS_GATES["minimum_non_breach_runs"][split],
                    available["runs"],
                ),
                "minimum_breach_runs": (
                    CLASS_GATES["minimum_breach_runs"][split],
                    available["runs"],
                ),
                "minimum_boundary_region_designs": (
                    CLASS_GATES["minimum_boundary_region_designs"][split],
                    available["boundary_designs"],
                ),
                "minimum_distinct_run_margin_values": (
                    CLASS_GATES["minimum_distinct_run_margin_values"][split],
                    available["runs"],
                ),
            }
            for gate, (required, maximum) in requirements.items():
                rows.append(
                    {
                        "scope": scope,
                        "split": split,
                        "gate": gate,
                        "required": required,
                        "maximum_physically_available": maximum,
                        "feasible": required <= maximum,
                        "applicability": "not_applicable_discovery_only"
                        if scope == "stage_a_augmentation_only"
                        else "candidate_final_gate_scope",
                    }
                )
            joint_design = _joint_class_gate_feasible(
                available["designs"],
                CLASS_GATES["minimum_non_breach_designs"][split],
                CLASS_GATES["minimum_breach_designs"][split],
            )
            joint_runs = _joint_class_gate_feasible(
                available["runs"],
                CLASS_GATES["minimum_non_breach_runs"][split],
                CLASS_GATES["minimum_breach_runs"][split],
            )
            ratio = CLASS_GATES["maximum_majority_to_minority_run_ratio"][split]
            minimum_non = CLASS_GATES["minimum_non_breach_runs"][split]
            minimum_breach = CLASS_GATES["minimum_breach_runs"][split]
            ratio_feasible = (
                joint_runs
                and max(minimum_non, minimum_breach) / min(minimum_non, minimum_breach) <= ratio
            )
            rows.extend(
                [
                    {
                        "scope": scope,
                        "split": split,
                        "gate": "joint_minimum_class_design_counts",
                        "required": CLASS_GATES["minimum_non_breach_designs"][split]
                        + CLASS_GATES["minimum_breach_designs"][split],
                        "maximum_physically_available": available["designs"],
                        "feasible": joint_design,
                        "applicability": "not_applicable_discovery_only"
                        if scope == "stage_a_augmentation_only"
                        else "candidate_final_gate_scope",
                    },
                    {
                        "scope": scope,
                        "split": split,
                        "gate": "joint_minimum_class_run_counts",
                        "required": minimum_non + minimum_breach,
                        "maximum_physically_available": available["runs"],
                        "feasible": joint_runs,
                        "applicability": "not_applicable_discovery_only"
                        if scope == "stage_a_augmentation_only"
                        else "candidate_final_gate_scope",
                    },
                    {
                        "scope": scope,
                        "split": split,
                        "gate": "majority_to_minority_ratio_at_minimum_counts",
                        "required": ratio,
                        "maximum_physically_available": max(minimum_non, minimum_breach)
                        / min(minimum_non, minimum_breach),
                        "feasible": ratio_feasible,
                        "applicability": "not_applicable_discovery_only"
                        if scope == "stage_a_augmentation_only"
                        else "candidate_final_gate_scope",
                    },
                ]
            )
    return rows


def _format_csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return format(value, ".17g")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV output requires rows: {path}")
    columns = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_csv_value(row.get(column)) for column in columns})


def _copy_outputs(source: Path, destination: Path, names: Sequence[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        payload = (source / name).read_bytes()
        (destination / name).write_bytes(payload)


def run_audit(
    *,
    audit_worktree: str | Path,
    production_tooling_root: str | Path,
    generation_root: str | Path,
    replay_root: str | Path,
    freeze_root: str | Path,
    freeze_archive: str | Path,
    freeze_archive_hash_file: str | Path,
    analysis_output_root: str | Path,
    tracked_analysis_root: str | Path,
    output_root: str | Path,
    tracked_output_root: str | Path | None = None,
) -> dict[str, Any]:
    protected = (
        generation_root,
        replay_root,
        freeze_root,
        freeze_archive,
        freeze_archive_hash_file,
        analysis_output_root,
        production_tooling_root,
    )
    destination = validate_output_root(output_root, protected)
    if destination.exists():
        raise FileExistsError(destination)
    worktree = Path(audit_worktree)
    if _git(worktree, "merge-base", "HEAD", ANALYSIS_COMMIT) != ANALYSIS_COMMIT:
        raise ValueError("Analysis commit under audit differs")
    if _git(worktree, "branch", "--show-current") != "audit/final-dataset-class-support-v1":
        raise ValueError("Audit branch differs")
    evidence = verify_frozen_evidence(
        production_tooling_root=production_tooling_root,
        generation_root=generation_root,
        replay_root=replay_root,
        freeze_root=freeze_root,
        freeze_archive=freeze_archive,
        freeze_archive_hash_file=freeze_archive_hash_file,
    )
    analysis_outputs = verify_analysis_outputs(analysis_output_root, tracked_analysis_root)
    reproduction, runs, designs = reproduce_corpus(generation_root)
    boundary, boundary_rows = boundary_audit(runs, designs)
    temporal = temporal_audit(runs)
    parameters = parameter_audit(runs, designs)
    neighbors = nearest_neighbor_audit(designs)
    regression = regression_audit(runs)
    stage_a, stage_b, collisions, near_neighbors = proposal_audit(
        analysis_output_root=analysis_output_root,
        runs=runs,
        designs=designs,
    )
    gate_rows = gate_feasibility_rows()
    combined_gate_scopes_feasible = all(
        row["feasible"]
        for row in gate_rows
        if row["scope"] in {"original_plus_stage_b", "original_plus_stage_a_plus_stage_b"}
    )
    findings = {
        "schema_identifier": "satnet.final_dataset_class_support_audit_findings.v1",
        "final_verdict": "NOT APPROVED FOR STAGE A CONTRACT FREEZE",
        "binding": [
            {
                "id": "B-001",
                "title": "Stage A scientific design contract is incomplete",
                "finding": "The proposal defines only 30-design regional/split arithmetic. It provides no exact Stage A design table, admissible Stage A parameter bounds, frozen identity manifest, or output-root manifest, so it cannot become a frozen contract as written.",
                "required_correction": "Create and independently audit an exact 30-design discovery table and manifests before freeze.",
            },
            {
                "id": "B-002",
                "title": "Stage A identity namespace is undefined",
                "finding": "Stage B already occupies AUGV1-D000 through AUGV1-D089, while Stage A has no separate design or run namespace. Current artifacts do not collide because Stage A IDs do not exist, but freezing Stage A without a disjoint namespace would create material collision risk.",
                "required_correction": "Declare stable Stage A design/run IDs disjoint from D000-D099, integer runs 0-499, and AUGV1-D000-D089.",
            },
            {
                "id": "B-003",
                "title": "Final combined-corpus membership is ambiguous",
                "finding": "The gates say fixed-original-plus-augmentation, but the proposal does not state whether Stage A becomes part of the final corpus alongside Stage B or remains discovery-only. Both mathematically feasible totals are different.",
                "required_correction": "Declare one final membership and split-appending rule before gate freeze.",
            },
            {
                "id": "B-004",
                "title": "Stage A boundary discovery criterion is not design-level",
                "finding": "The existing criterion that four boundary designs collectively sample both signs can pass with no individually mixed design and does not establish a usable realization-sensitive transition region.",
                "required_correction": "Predeclare distinct-design mixed-realization and near-boundary stop/go criteria.",
            },
            {
                "id": "B-005",
                "title": "Final classification gate definitions are incomplete",
                "finding": "The proposal does not define whether a boundary design means a preassigned DOE region or an observed margin condition, how distinct binary64 margins are compared, or whether a mixed design may satisfy both breach-design and non-breach-design minima.",
                "required_correction": "Define every acceptance-gate counting rule and numeric equality rule in the future contract and its executable validator.",
            },
        ],
        "nonbinding": [
            {
                "id": "N-001",
                "title": "Stage A internal holdout terminology",
                "finding": "The sealed five-design Stage A test is scientifically acceptable under Approach B, but should be called an internal holdout until a final combined-corpus contract defines its evaluation role.",
            },
            {
                "id": "N-002",
                "title": "Single-design regional wording",
                "finding": "Phrases such as 'well inside the observed resilient region' should be narrowed to an observed anchor or candidate region because D000 is the only consistently non-breach design.",
            },
            {
                "id": "N-003",
                "title": "Near-neighbor radius sensitivity",
                "finding": "The 0.10 normalized radius passes the current Stage B table, but its scientific sensitivity should be reviewed before freeze because the conclusion is metric- and radius-dependent.",
            },
        ],
        "observations": [
            {
                "id": "O-001",
                "title": "Scientific reproduction passed",
                "finding": "All binding corpus, class, boundary, temporal, parameter, nearest-neighbor, and regression findings independently reproduced.",
            },
            {
                "id": "O-002",
                "title": "Proposed final gates are arithmetically feasible",
                "finding": "Every proposed final gate is physically feasible under both original-plus-Stage-B and original-plus-Stage-A-plus-Stage-B totals. The gates are not applicable to Stage A discovery alone.",
            },
            {
                "id": "O-003",
                "title": "Canonical analysis outputs are external",
                "finding": "The external inventory and all 24 bound outputs verify. Three large tables are external-only; tracked summaries are semantically equal despite Windows line-ending conversion.",
            },
        ],
    }
    non_breach_rows = [
        {
            "run_id": row["run_id"],
            "run_key": row["run_key"],
            "design_id": row["design_id"],
            "split": row["split"],
            "overall_minimum": row["failure_adjusted_overall_service_fraction_min"],
            "signed_margin": row["overall_boundary_margin"],
            "temporal_breach_count": row["temporal_breach_count"],
            "non_breach": not row["overall_threshold_breach_any"],
        }
        for row in runs
        if not row["overall_threshold_breach_any"]
    ]
    reproduction_summary = {
        "schema_identifier": "satnet.final_dataset_class_support_audit_reproduction.v1",
        "analysis_commit_audited": ANALYSIS_COMMIT,
        "evidence": evidence,
        "analysis_outputs": analysis_outputs,
        "corpus": reproduction,
        "boundary": boundary,
        "temporal": temporal,
        "parameter_support": parameters,
        "nearest_neighbors": neighbors,
        "regression_only": regression,
        "classification_gate_feasibility": {
            "combined_candidate_scopes_feasible": combined_gate_scopes_feasible,
            "scope_ambiguity_requires_correction": True,
            "operational_definitions_require_correction": {
                "boundary_design": "Declare preassigned region membership versus observed inclusive margin window.",
                "distinct_margin": "Declare canonical binary64 equality or a predeclared tolerance/quantization rule.",
                "mixed_design_class_counting": "Declare whether one mixed design may support both design-level class minima.",
            },
        },
    }
    destination.mkdir(parents=True)
    write_json(destination / "audit_findings.json", findings)
    write_json(destination / "audit_reproduction_summary.json", reproduction_summary)
    write_csv(destination / "audit_boundary_reproduction.csv", boundary_rows)
    write_csv(destination / "audit_non_breach_reproduction.csv", non_breach_rows)
    write_json(destination / "audit_stage_a_review.json", stage_a)
    write_json(destination / "audit_stage_b_review.json", stage_b)
    write_csv(destination / "audit_gate_feasibility.csv", gate_rows)
    write_csv(destination / "audit_identity_collision_report.csv", collisions)
    write_csv(destination / "audit_near_neighbor_leakage.csv", near_neighbors)
    output_names = [
        "audit_findings.json",
        "audit_reproduction_summary.json",
        "audit_boundary_reproduction.csv",
        "audit_non_breach_reproduction.csv",
        "audit_stage_a_review.json",
        "audit_stage_b_review.json",
        "audit_gate_feasibility.csv",
        "audit_identity_collision_report.csv",
        "audit_near_neighbor_leakage.csv",
    ]
    inventory = {
        "schema_identifier": "satnet.final_dataset_class_support_audit_inventory.v1",
        "analysis_commit_audited": ANALYSIS_COMMIT,
        "self_exclusion": "audit_inventory.json is excluded to avoid a self-referential hash cycle.",
        "outputs": [
            {
                "relative_path": name,
                "byte_length": (destination / name).stat().st_size,
                "sha256": sha256_file(destination / name),
            }
            for name in output_names
        ],
    }
    write_json(destination / "audit_inventory.json", inventory)
    output_names.append("audit_inventory.json")
    if tracked_output_root is not None:
        tracked_destination = validate_output_root(tracked_output_root, protected)
        _copy_outputs(destination, tracked_destination, output_names)
    return {
        "audit_status": "completed",
        "final_verdict": findings["final_verdict"],
        "binding_finding_count": len(findings["binding"]),
        "nonbinding_finding_count": len(findings["nonbinding"]),
        "analysis_commit_audited": ANALYSIS_COMMIT,
        "run_count": reproduction["run_count"],
        "design_count": reproduction["design_count"],
        "non_breach_run_ids": reproduction["non_breach_run_ids"],
        "non_breach_design_ids": reproduction["non_breach_design_ids"],
        "audit_inventory_sha256": sha256_file(destination / "audit_inventory.json"),
    }
