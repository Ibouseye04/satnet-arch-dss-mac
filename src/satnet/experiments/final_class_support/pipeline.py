from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from satnet.experiments.final_class_support.analysis import (
    analysis_summary,
    augmentation_contract,
    augmentation_options,
    boundary_rankings,
    boundary_region_summary,
    nearest_neighbors,
    parameter_support,
    proposed_augmentation_designs,
    regression_summary,
    split_summary,
)
from satnet.experiments.final_class_support.constants import (
    ANALYSIS_SCHEMA_VERSION,
    CONTRACT_SPECIFICATION_HASH,
    FREEZE_ARCHIVE_SHA256,
    GENERATION_LEDGER_SHA256,
    OUTPUT_SCHEMAS,
    PLOT_SCHEMAS,
    PRODUCTION_TOOLING_SHA,
    PROPOSAL_LABEL,
    REPLAY_LEDGER_SHA256,
)
from satnet.experiments.final_class_support.extraction import extract_corpus
from satnet.experiments.final_class_support.io import (
    sha256_file,
    validate_output_root,
    verify_frozen_evidence,
    write_csv,
    write_json,
)
from satnet.experiments.final_class_support.plots import create_plots


def _columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        raise ValueError("At least one output record is required")
    preferred = [
        "proposal_label",
        "run_id",
        "run_key",
        "design_id",
        "augmentation_design_id",
        "realization_id",
        "design_index",
        "augmentation_design_index",
        "realization_index",
        "split",
        "intended_split",
        "doe_stratum",
        "intended_region",
    ]
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def _temporal_rows(run_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "run_id",
        "run_key",
        "design_id",
        "realization_id",
        "split",
        "sampled_state_count",
        "overall_service_sequence",
        "ground_service_sequence",
        "space_gcc_sequence",
        "temporal_breach_count",
        "temporal_breach_fraction",
        "first_overall_breach_timestep",
        "last_overall_breach_timestep",
        "longest_overall_breach_streak",
        "minimum_overall_service_timestep",
        "number_of_recoveries_above_threshold",
    )
    return [{field: row[field] for field in fields} for row in run_rows]


def _non_breach_rows(run_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "run_id",
        "run_key",
        "design_id",
        "realization_id",
        "split",
        "doe_stratum",
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
        "satellite_seed",
        "ground_failure_seed",
        "ground_selection_seed",
        "failure_adjusted_overall_service_fraction_min",
        "failure_adjusted_overall_service_fraction_mean",
        "failure_adjusted_ground_service_fraction_min",
        "space_gcc_fraction_original_min",
        "overall_boundary_margin",
        "temporal_breach_count",
        "failed_ground_station_count",
        "failed_ground_station_ids",
        "failed_satellite_node_count",
        "failed_satellite_nodes",
        "failed_satellite_edge_count",
        "failed_satellite_edges",
    )
    return [
        {field: row[field] for field in fields}
        for row in run_rows
        if not row["overall_threshold_breach_any"]
    ]


def _inventory_record(path: Path, output_root: Path, schema: str) -> dict[str, Any]:
    record_count: int | None = None
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            record_count = sum(1 for _ in csv.reader(handle)) - 1
    return {
        "relative_path": path.relative_to(output_root).as_posix(),
        "byte_length": path.stat().st_size,
        "sha256": sha256_file(path),
        "record_count": record_count,
        "schema_identifier": schema,
    }


def _create_inventory(output_root: Path) -> dict[str, Any]:
    schemas = {**OUTPUT_SCHEMAS, **PLOT_SCHEMAS}
    records = []
    for relative_path, schema in sorted(schemas.items()):
        path = output_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required final output missing: {path}")
        records.append(_inventory_record(path, output_root, schema))
    inventory = {
        "schema_identifier": "satnet.class_support.analysis_inventory.v1",
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "self_exclusion": "analysis_inventory.json is excluded to avoid a self-referential hash cycle.",
        "input_identities": {
            "tooling_sha": PRODUCTION_TOOLING_SHA,
            "contract_specification_hash": CONTRACT_SPECIFICATION_HASH,
            "generation_ledger_sha256": GENERATION_LEDGER_SHA256,
            "replay_ledger_sha256": REPLAY_LEDGER_SHA256,
            "freeze_archive_sha256": FREEZE_ARCHIVE_SHA256,
        },
        "outputs": records,
    }
    write_json(output_root / "analysis_inventory.json", inventory)
    return inventory


def run_analysis(
    *,
    production_tooling_root: str | Path,
    generation_root: str | Path,
    replay_root: str | Path,
    freeze_root: str | Path,
    freeze_archive: str | Path,
    freeze_archive_hash_file: str | Path,
    output_root: str | Path,
    verify_all_hashes: bool = True,
) -> dict[str, Any]:
    protected = (generation_root, replay_root, freeze_root, freeze_archive, freeze_archive_hash_file)
    destination = validate_output_root(output_root, protected)
    if destination.exists():
        raise FileExistsError(f"Analysis output root already exists: {destination}")
    if verify_all_hashes:
        evidence_verification = verify_frozen_evidence(
            production_tooling_root=production_tooling_root,
            generation_root=generation_root,
            replay_root=replay_root,
            freeze_root=freeze_root,
            freeze_archive=freeze_archive,
            freeze_archive_hash_file=freeze_archive_hash_file,
        )
    else:
        evidence_verification = {"verification_status": "explicitly_skipped_for_test_only"}
    run_rows, design_rows = extract_corpus(generation_root)
    boundary = boundary_region_summary(run_rows, design_rows)
    run_ranking, design_ranking = boundary_rankings(run_rows, design_rows)
    parameters = parameter_support(run_rows, design_rows)
    neighbors = nearest_neighbors(design_rows)
    proposal_rows = proposed_augmentation_designs()
    split = split_summary(run_rows, design_rows)
    regression = regression_summary(run_rows)
    options = augmentation_options()
    contract = augmentation_contract(proposal_rows)
    summary = analysis_summary(run_rows, design_rows, neighbors, boundary)
    destination.mkdir(parents=True)
    output_rows = {
        "run_level_class_support.csv": run_rows,
        "design_level_class_support.csv": design_rows,
        "non_breach_runs.csv": _non_breach_rows(run_rows),
        "non_breach_designs.csv": [
            row for row in design_rows if row["non_breach_realization_count"] > 0
        ],
        "boundary_run_ranking.csv": run_ranking,
        "boundary_design_ranking.csv": design_ranking,
        "temporal_breach_summary.csv": _temporal_rows(run_rows),
        "parameter_support_summary.csv": parameters,
        "nearest_neighbor_summary.csv": neighbors,
        "recommended_augmentation_design.csv": proposal_rows,
    }
    for name, rows in output_rows.items():
        write_csv(destination / name, rows, _columns(rows))
    output_json = {
        "split_class_support_summary.json": split,
        "regression_only_summary.json": regression,
        "boundary_region_summary.json": boundary,
        "augmentation_size_options.json": options,
        "augmentation_contract_proposal.json": contract,
    }
    for name, value in output_json.items():
        write_json(destination / name, value)
    create_plots(destination, run_rows, design_rows, neighbors, proposal_rows)
    inventory = _create_inventory(destination)
    handoff = {
        "analysis_status": "completed",
        "proposal_label": PROPOSAL_LABEL,
        "simulation_authorized": False,
        "evidence_verification": evidence_verification,
        "run_count": len(run_rows),
        "design_count": len(design_rows),
        "summary": summary,
        "analysis_inventory_sha256": sha256_file(destination / "analysis_inventory.json"),
        "inventory_output_count": len(inventory["outputs"]),
    }
    return handoff
