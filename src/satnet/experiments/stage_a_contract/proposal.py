from __future__ import annotations

from collections import Counter, defaultdict
import csv
from decimal import Decimal
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from satnet.experiments.stage_a_contract.designs import (
    REGION_BOUNDS,
    build_design_rows,
    minimum_distances,
    scientific_signature,
    validate_design_rows,
)
from satnet.experiments.stage_a_contract.semantics import (
    ANALYSIS_COMMIT,
    ANALYSIS_INVENTORY_SHA256,
    AUDIT_COMMIT,
    AUDIT_IMPLEMENTATION_COMMIT,
    AUDIT_INVENTORY_SHA256,
    CONTRACT_SPECIFICATION_HASH,
    CORPUS_NAMESPACE,
    DOE_RANGES,
    FINAL_GATES,
    FIXED_PROFILE,
    FREEZE_ARCHIVE_SHA256,
    FROZEN_CONTRACT_COMMIT,
    FROZEN_ROOTS,
    GENERATION_LEDGER_SHA256,
    MARGIN_QUANTUM,
    NEIGHBOR_FEATURES,
    OUTPUT_ROOTS,
    PRODUCTION_TOOLING_SHA,
    PROPOSAL_SCHEMA,
    PROPOSAL_STATUS,
    PROPOSAL_VERSION,
    REPLAY_LEDGER_SHA256,
    SEED_DOMAIN,
    SEED_MODULUS,
    SEED_POLICY_VERSION,
    SERVICE_THRESHOLD,
    SIMULATION_AUTHORIZED,
    canonical_json_bytes,
    canonical_margin,
    canonical_payload_hash,
    derive_seed,
    design_outcome,
    normalized_distance,
    observed_boundary_design,
    paths_overlap,
    sha256_bytes,
)

ARTIFACT_SCHEMAS = {
    "stage_a_design_manifest.csv": "satnet.stage_a.design_manifest.v1",
    "stage_a_run_manifest.csv": "satnet.stage_a.run_manifest.v1",
    "stage_a_partition_manifest.json": "satnet.stage_a.partition_manifest.v1",
    "stage_a_seed_policy.json": "satnet.stage_a.seed_policy.v1",
    "stage_a_seed_manifest.csv": "satnet.stage_a.seed_manifest.v1",
    "stage_a_region_bounds.json": "satnet.stage_a.region_bounds.v1",
    "stage_a_output_root_manifest.json": "satnet.stage_a.output_roots.v1",
    "stage_a_discovery_criteria.json": "satnet.stage_a.discovery_criteria.v1",
    "stage_a_near_neighbor_policy.json": "satnet.stage_a.near_neighbor_policy.v1",
    "stage_a_contract_proposal.json": PROPOSAL_SCHEMA,
}


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("CSV serialization requires at least one row")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def load_original_designs(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 100 or [row["design_id"] for row in rows] != [f"D{index:03d}" for index in range(100)]:
        raise ValueError("Original design manifest identity differs")
    return rows


def build_run_rows(designs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for design in designs:
        for realization_index in range(5):
            realization_id = f"R{realization_index:02d}"
            global_run_id = 500 + int(design["design_index"]) * 5 + realization_index
            row = {
                "corpus_namespace": CORPUS_NAMESPACE,
                "global_run_id": global_run_id,
                "run_key": f"{design['design_id']}-{realization_id}",
                "design_id": design["design_id"],
                "design_index": design["design_index"],
                "realization_id": realization_id,
                "realization_index": realization_index,
                "region": design["region"],
                "partition": design["partition"],
                "sealed": design["sealed"],
                "design_record_hash": design["design_record_hash"],
                "proposal_status": PROPOSAL_STATUS,
                "simulation_authorized": SIMULATION_AUTHORIZED,
            }
            row["run_record_hash"] = canonical_payload_hash(row, domain="satnet_stage_a_run_record")
            rows.append(row)
    validate_run_rows(rows, designs)
    return rows


def validate_run_rows(rows: Sequence[Mapping[str, Any]], designs: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 150:
        raise ValueError("Stage A requires exactly 150 runs")
    if [int(row["global_run_id"]) for row in rows] != list(range(500, 650)):
        raise ValueError("Stage A global run IDs must cover 500 through 649 exactly")
    keys = [str(row["run_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Stage A run keys must be unique")
    pairs = [(row["design_id"], row["realization_id"]) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Stage A design-realization pairs must be unique")
    by_design = {row["design_id"]: row for row in designs}
    for row in rows:
        design = by_design.get(row["design_id"])
        if design is None:
            raise ValueError("Run references an unknown Stage A design")
        expected_id = 500 + int(design["design_index"]) * 5 + int(row["realization_index"])
        expected_realization = f"R{int(row['realization_index']):02d}"
        if row["global_run_id"] != expected_id or row["realization_id"] != expected_realization:
            raise ValueError("Stage A run identity formula mismatch")
        if row["run_key"] != f"{design['design_id']}-{expected_realization}":
            raise ValueError("Stage A run key mismatch")
        for field in ("region", "partition", "sealed", "design_record_hash"):
            if row[field] != design[field]:
                raise ValueError(f"Run-level {field} differs from design")
        if row["proposal_status"] != PROPOSAL_STATUS or row["simulation_authorized"] is not False:
            raise ValueError("Stage A run must remain unauthorized and NOT_FROZEN")
        expected_hash = canonical_payload_hash({key: value for key, value in row.items() if key != "run_record_hash"}, domain="satnet_stage_a_run_record")
        if row["run_record_hash"] != expected_hash:
            raise ValueError("Stage A run-record hash mismatch")


def build_seed_policy() -> dict[str, Any]:
    return {
        "schema_identifier": ARTIFACT_SCHEMAS["stage_a_seed_policy.json"],
        "seed_policy_version": SEED_POLICY_VERSION,
        "proposal_status": PROPOSAL_STATUS,
        "simulation_authorized": SIMULATION_AUTHORIZED,
        "hash_algorithm": "SHA-256",
        "input_encoding": "UTF-8 canonical JSON with lexicographically sorted object keys and compact separators",
        "domain_separation_string": SEED_DOMAIN,
        "bound_identity_fields": ["corpus_namespace", "proposal_version", "design_id", "realization_id when applicable", "seed_purpose"],
        "byte_extraction": "all 32 SHA-256 digest bytes",
        "integer_conversion": "unsigned big-endian",
        "range_reduction": f"modulo {SEED_MODULUS}",
        "supported_integer_range": {"minimum_inclusive": 0, "maximum_inclusive": SEED_MODULUS - 1},
        "purposes": {
            "design_construction": {"realization_bound": False},
            "ground_station_selection": {"realization_bound": False},
            "satellite_rollout_and_failure": {"realization_bound": True},
            "ground_failure_realization": {"realization_bound": True},
        },
        "retry_policy": "Same-run retry retains identical identity and seeds; substitution and outcome-driven seed changes are prohibited.",
        "authority": "Proposed only; seeds become authoritative only after future freeze and independent audit.",
    }


def build_seed_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        design_id = str(run["design_id"])
        realization_id = str(run["realization_id"])
        rows.append(
            {
                "corpus_namespace": CORPUS_NAMESPACE,
                "global_run_id": run["global_run_id"],
                "run_key": run["run_key"],
                "design_id": design_id,
                "realization_id": realization_id,
                "design_construction_seed": derive_seed(purpose="design_construction", design_id=design_id),
                "ground_selection_seed": derive_seed(purpose="ground_station_selection", design_id=design_id),
                "satellite_failure_seed": derive_seed(purpose="satellite_rollout_and_failure", design_id=design_id, realization_id=realization_id),
                "ground_failure_seed": derive_seed(purpose="ground_failure_realization", design_id=design_id, realization_id=realization_id),
                "seed_policy_version": SEED_POLICY_VERSION,
                "proposal_status": PROPOSAL_STATUS,
                "simulation_authorized": SIMULATION_AUTHORIZED,
            }
        )
    validate_seed_rows(rows)
    return rows


def validate_seed_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 150:
        raise ValueError("Stage A seed manifest requires 150 rows")
    for row in rows:
        expected = {
            "design_construction_seed": derive_seed(purpose="design_construction", design_id=str(row["design_id"])),
            "ground_selection_seed": derive_seed(purpose="ground_station_selection", design_id=str(row["design_id"])),
            "satellite_failure_seed": derive_seed(purpose="satellite_rollout_and_failure", design_id=str(row["design_id"]), realization_id=str(row["realization_id"])),
            "ground_failure_seed": derive_seed(purpose="ground_failure_realization", design_id=str(row["design_id"]), realization_id=str(row["realization_id"])),
        }
        if any(row[field] != value for field, value in expected.items()):
            raise ValueError("Stage A seed substitution or derivation mismatch")
        if any(type(row[field]) is not int or not 0 <= row[field] < SEED_MODULUS for field in expected):
            raise ValueError("Stage A seed lies outside supported range")
        if row["proposal_status"] != PROPOSAL_STATUS or row["simulation_authorized"] is not False:
            raise ValueError("Stage A proposed seeds must remain unauthorized and NOT_FROZEN")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["design_id"])].append(row)
    for group in groups.values():
        if len(group) != 5 or len({row["ground_selection_seed"] for row in group}) != 1:
            raise ValueError("Ground-selection seed must be fixed within each design")
        for field in ("satellite_failure_seed", "ground_failure_seed"):
            if len({row[field] for row in group}) != 5:
                raise ValueError(f"{field} must vary by realization")


def build_partition_manifest(designs: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    partitions: dict[str, Any] = {}
    for partition in ("development", "validation", "sealed_holdout"):
        design_rows = [row for row in designs if row["partition"] == partition]
        run_rows = [row for row in runs if row["partition"] == partition]
        partitions[partition] = {
            "design_ids": [row["design_id"] for row in design_rows],
            "global_run_ids": [row["global_run_id"] for row in run_rows],
            "region_counts": dict(sorted(Counter(row["region"] for row in design_rows).items())),
            "design_count": len(design_rows),
            "run_count": len(run_rows),
            "sealed": partition == "sealed_holdout",
        }
    partitions["development"]["permitted_uses"] = ["Stage A discovery analysis", "May inform a separately proposed Stage B contract"]
    partitions["validation"]["permitted_uses"] = ["Assess whether development findings reproduce", "May inform a separately proposed Stage B contract"]
    partitions["sealed_holdout"]["permitted_uses"] = ["After lawful unsealing, decide whether an already-frozen and independently audited Stage B contract is executed"]
    partitions["sealed_holdout"]["prohibited_uses"] = ["Select Stage B parameter bounds", "Select Stage B regions", "Select Stage B design density", "Select Stage B sample size", "Select Stage B split allocation", "Select Stage B seeds", "Select Stage B acceptance gates", "Modify a frozen Stage B contract"]
    return {
        "schema_identifier": ARTIFACT_SCHEMAS["stage_a_partition_manifest.json"],
        "corpus_namespace": CORPUS_NAMESPACE,
        "proposal_status": PROPOSAL_STATUS,
        "simulation_authorized": SIMULATION_AUTHORIZED,
        "assignment_timing": "Pre-simulation and immutable with respect to outcomes",
        "partitions": partitions,
        "unsealing_conditions": [
            "Stage B contract completely specified",
            "Stage B design manifest generated",
            "Stage B run and seed manifests generated",
            "Stage B split frozen",
            "Stage B contract passed independent audit",
            "Stage B contract hash recorded",
        ],
        "holdout_not_final_test_set": True,
        "holdout_change_rule": "If holdout results require design changes, reject Stage B v1 and create, freeze, and audit Stage B v2.",
    }


def validate_holdout_policy(manifest: Mapping[str, Any], *, stage_b_contract_hash: str | None = None) -> None:
    holdout = manifest["partitions"]["sealed_holdout"]
    if holdout["sealed"] is not True or holdout["design_count"] != 5 or holdout["run_count"] != 25:
        raise ValueError("Stage A holdout is not sealed with exact cardinality")
    required = {"Select Stage B parameter bounds", "Select Stage B regions", "Select Stage B design density", "Select Stage B sample size", "Select Stage B split allocation", "Select Stage B seeds", "Select Stage B acceptance gates", "Modify a frozen Stage B contract"}
    if not required.issubset(set(holdout.get("prohibited_uses", []))):
        raise ValueError("Stage A holdout leakage prohibition is incomplete")
    if stage_b_contract_hash is None:
        raise ValueError("Stage A holdout cannot be unsealed before a Stage B contract hash exists")


def build_region_bounds() -> dict[str, Any]:
    rationale = {
        "resilient_core": "Controlled variation around independently observed resilient anchors D000 and D001; assignment is a sampling hypothesis, not validated regional resilience.",
        "boundary": "Controlled transition probes spanning D001 toward D022 and nearby breached evidence; assignment does not imply an observed boundary outcome.",
        "global_control": "Broad comparison coverage across the original admissible DOE rather than concentration near D000 or D001.",
    }
    source = {
        "resilient_core": ["D000", "D001", "audit parameter-support findings"],
        "boundary": ["D001", "D022", "D001-to-D022 normalized distance 0.3932311802307627", "audit boundary findings"],
        "global_control": ["frozen original global DOE ranges"],
    }
    return {
        "schema_identifier": ARTIFACT_SCHEMAS["stage_a_region_bounds.json"],
        "proposal_status": PROPOSAL_STATUS,
        "simulation_authorized": SIMULATION_AUTHORIZED,
        "base_admissible_bounds": DOE_RANGES,
        "fixed_values": FIXED_PROFILE,
        "excluded_values": {"all_regions": ["non-finite numeric values", "negative class counts", "zero total ground-station count", "parameters outside frozen base DOE"]},
        "regions": {region: {"allowed_parameter_bounds": bounds, "fixed_values": FIXED_PROFILE, "excluded_values": [], "construction_rationale": rationale[region], "source_evidence": source[region]} for region, bounds in REGION_BOUNDS.items()},
    }


def build_output_root_manifest() -> dict[str, Any]:
    validate_output_roots(OUTPUT_ROOTS, require_absent=True)
    return {
        "schema_identifier": ARTIFACT_SCHEMAS["stage_a_output_root_manifest.json"],
        "proposal_status": PROPOSAL_STATUS,
        "simulation_authorized": SIMULATION_AUTHORIZED,
        "proposed_resolved_paths": OUTPUT_ROOTS,
        "all_external_to_git_worktrees": True,
        "all_outside_frozen_evidence": True,
        "mutually_non_overlapping": True,
        "current_existence": {name: False for name in OUTPUT_ROOTS},
        "creation_authorized": False,
        "isolation_policy": "Future roots may be created only after contract freeze, independent audit, and explicit execution authorization.",
    }


def validate_output_roots(roots: Mapping[str, str], *, require_absent: bool) -> None:
    if set(roots) != set(OUTPUT_ROOTS):
        raise ValueError("Stage A output-root roles are incomplete")
    values = list(roots.values())
    for index, first in enumerate(values):
        for second in values[index + 1 :]:
            if paths_overlap(first, second):
                raise ValueError("Stage A output roots overlap")
        if any(paths_overlap(first, frozen) for frozen in FROZEN_ROOTS):
            raise ValueError("Stage A output root overlaps frozen evidence")
        if require_absent and Path(first).exists():
            raise ValueError(f"Proposed Stage A output root already exists: {first}")


def _criterion(identifier: str, description: str, partition: str, metric: str, operator: str, threshold: Any, rationale: str, consequence: str, influence: bool) -> dict[str, Any]:
    return {"criterion_id": identifier, "description": description, "input_partition": partition, "metric": metric, "operator": operator, "threshold": threshold, "scientific_rationale": rationale, "failure_consequence": consequence, "can_influence_stage_b": influence}


def build_discovery_criteria() -> dict[str, Any]:
    return {
        "schema_identifier": ARTIFACT_SCHEMAS["stage_a_discovery_criteria.json"],
        "proposal_status": PROPOSAL_STATUS,
        "simulation_authorized": SIMULATION_AUTHORIZED,
        "final_classification_gates_apply": False,
        "development_criteria": [
            _criterion("DEV-001", "Complete fixed development block", "development", "successful fixed-identity designs/runs", "equals", {"designs": 20, "runs": 100}, "Operational completeness", "STAGE_A_DISCOVERY_INSUFFICIENT", True),
            _criterion("DEV-002", "Reproduce resilient-core support", "development", "resilient-core non-breach-majority designs", "greater_than_or_equal", 2, "Independent design support cannot be replaced by pooled runs", "STAGE_A_DISCOVERY_INSUFFICIENT", True),
            _criterion("DEV-003", "Locate realization-sensitive boundary probes", "development", "individually mixed preassigned-boundary designs", "greater_than_or_equal", 2, "Requires within-design class variation", "Continue only to predeclared validation; no Stage B freeze", True),
            _criterion("DEV-004", "Retain broad-control consistency", "development", "global controls with zero of five non-breach", "greater_than_or_equal", 3, "Detect targeted-region population shift", "Investigate before Stage B proposal", True),
            _criterion("DEV-005", "Report realization stability", "development", "designs with reported non-breach count and margin spread", "equals", 20, "Pooled run counts cannot replace design support", "STAGE_A_DISCOVERY_INSUFFICIENT", True),
        ],
        "validation_criteria": [
            _criterion("VAL-001", "Complete fixed validation block", "validation", "successful fixed-identity designs/runs", "equals", {"designs": 5, "runs": 25}, "Outcome-independent reproduction", "STAGE_A_VALIDATION_FAILED", True),
            _criterion("VAL-002", "Reproduce core support independently", "validation", "resilient-core non-breach-majority designs", "greater_than_or_equal", 1, "Development support must reproduce", "STAGE_A_VALIDATION_FAILED", True),
            _criterion("VAL-003", "Reproduce boundary mixing independently", "validation", "individually mixed preassigned-boundary designs", "greater_than_or_equal", 1, "Boundary localization must not be development-only", "STAGE_A_VALIDATION_FAILED", True),
            _criterion("VAL-004", "Retain validation control consistency", "validation", "global controls with zero of five non-breach", "equals", 1, "Detect population shift", "STAGE_A_VALIDATION_FAILED", True),
            _criterion("VAL-005", "Contain both majority classes", "validation", "majority-class design counts", "greater_than_or_equal", {"non_breach_majority": 1, "breach_majority": 1}, "Independent class support", "STAGE_A_VALIDATION_FAILED", True),
        ],
        "pre_holdout_stage_b_proposal_criteria": [
            _criterion("PRE-001", "Core reproduction", "development+validation", "distinct resilient-core non-breach-majority designs", "greater_than_or_equal", 2, "Audit minimum", "STAGE_A_DISCOVERY_INSUFFICIENT", True),
            _criterion("PRE-002", "Boundary mixing", "development+validation", "distinct individually mixed preassigned-boundary designs", "greater_than_or_equal", 3, "Audit minimum and design-level correction", "STAGE_A_DISCOVERY_INSUFFICIENT", True),
            _criterion("PRE-003", "Near-boundary localization", "development+validation", "distinct preassigned-boundary designs with at least one absolute margin <= 0.10", "greater_than_or_equal", 4, "Audit minimum", "STAGE_A_DISCOVERY_INSUFFICIENT", True),
            _criterion("PRE-004", "Both transition signs across independent designs", "development+validation", "distinct near-boundary designs by sign", "greater_than_or_equal", {"negative": 2, "nonnegative": 2}, "Audit minimum; zero is non-breach", "STAGE_A_DISCOVERY_INSUFFICIENT", True),
            _criterion("PRE-005", "Independent majority-class support", "development+validation", "majority-class design counts", "greater_than_or_equal", {"non_breach_majority": 4, "breach_majority": 4}, "Mixed designs count once by majority class", "STAGE_A_DISCOVERY_INSUFFICIENT", True),
            _criterion("PRE-006", "Global-control consistency", "development+validation", "global controls with zero of five non-breach", "greater_than_or_equal", 4, "Audit minimum", "Investigate population shift before Stage B proposal", True),
            _criterion("PRE-007", "No holdout consultation", "sealed_holdout", "holdout outcomes accessed", "equals", 0, "Protect outcome-independent confirmation", "Reject Stage B proposal process and create a new holdout", False),
        ],
        "sealed_holdout_confirmation_criteria": [
            _criterion("HOLD-001", "Lawful unsealing prerequisites", "sealed_holdout", "completed unsealing conditions", "equals", 6, "Holdout cannot shape Stage B", "STAGE_A_HOLDOUT_NOT_CONFIRMED", False),
            _criterion("HOLD-002", "Complete fixed holdout block", "sealed_holdout", "successful fixed-identity designs/runs", "equals", {"designs": 5, "runs": 25}, "Operational confirmation", "STAGE_A_HOLDOUT_NOT_CONFIRMED", False),
            _criterion("HOLD-003", "Confirm core support", "sealed_holdout", "resilient-core non-breach-majority designs", "greater_than_or_equal", 1, "Independent confirmation", "STAGE_A_HOLDOUT_NOT_CONFIRMED", False),
            _criterion("HOLD-004", "Confirm boundary localization", "sealed_holdout", "observed boundary designs", "greater_than_or_equal", 1, "Uses outcome-defined boundary rule", "STAGE_A_HOLDOUT_NOT_CONFIRMED", False),
            _criterion("HOLD-005", "Confirm global control", "sealed_holdout", "global controls with zero of five non-breach", "equals", 1, "Population-shift check", "STAGE_A_HOLDOUT_NOT_CONFIRMED", False),
        ],
        "decision_states": ["STAGE_A_DISCOVERY_INSUFFICIENT", "STAGE_A_READY_FOR_STAGE_B_PROPOSAL", "STAGE_A_VALIDATION_FAILED", "STAGE_A_STAGE_B_CONTRACT_FROZEN", "STAGE_A_HOLDOUT_CONFIRMED", "STAGE_A_HOLDOUT_NOT_CONFIRMED"],
        "workflow": ["Run development", "Evaluate development criteria", "Expose and evaluate validation", "Draft Stage B", "Freeze and independently audit Stage B", "Record Stage B contract hash", "Unseal Stage A holdout", "Evaluate holdout confirmation", "Execute unchanged frozen Stage B or reject it"],
    }


def _nearest(source: Mapping[str, Any], candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    distance, candidate = min((normalized_distance(source, value), value) for value in candidates)
    return {"source_design_id": source["design_id"], "neighbor_design_id": candidate["design_id"], "neighbor_partition": candidate.get("partition", candidate.get("split_assignment")), "normalized_distance": distance}


def build_near_neighbor_policy(designs: Sequence[Mapping[str, Any]], original: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stage_pairs = [(first, second, normalized_distance(first, second)) for index, first in enumerate(designs) for second in designs[index + 1 :]]
    cross = [(first, second, distance) for first, second, distance in stage_pairs if first["partition"] != second["partition"]]
    sealed = [(first, second, distance) for first, second, distance in cross if "sealed_holdout" in {first["partition"], second["partition"]}]
    threshold = 0.10
    cross_review = [{"first_design_id": first["design_id"], "first_partition": first["partition"], "second_design_id": second["design_id"], "second_partition": second["partition"], "normalized_distance": distance, "review_status": "REQUIRES_EXPLICIT_PRE_FREEZE_REVIEW"} for first, second, distance in cross if distance <= threshold]
    sealed_review = [row for row in cross_review if "sealed_holdout" in {row["first_partition"], row["second_partition"]}]
    if sealed_review:
        raise ValueError("Sealed holdout has an unapproved neighbor within normalized distance 0.10")
    within_reports = {}
    for partition in ("development", "validation", "sealed_holdout"):
        group = [row for row in designs if row["partition"] == partition]
        within_reports[partition] = [_nearest(row, [candidate for candidate in group if candidate["design_id"] != row["design_id"]]) for row in group]
    return {
        "schema_identifier": ARTIFACT_SCHEMAS["stage_a_near_neighbor_policy.json"],
        "proposal_status": PROPOSAL_STATUS,
        "simulation_authorized": SIMULATION_AUTHORIZED,
        "distance_method": {"metric": "Euclidean", "features": list(NEIGHBOR_FEATURES), "normalization": "Frozen full-DOE min-max ranges", "near_neighbor_radius_inclusive": threshold},
        "duplicate_policy": {"identity_duplicates": "PROHIBITED", "parameter_vector_duplicates": "PROHIBITED", "exact_original_duplicates": "PROHIBITED"},
        "cross_partition_policy": "Pairs at distance <= 0.10 require explicit pre-freeze review; scientifically important boundary probes are not removed automatically.",
        "sealed_holdout_policy": "No development or validation neighbor at distance <= 0.10 without scientific justification and explicit approval before freeze.",
        "reports": {
            "stage_a_to_original_nearest": [_nearest(row, original) for row in designs],
            "stage_a_within_partition_nearest": within_reports,
            "stage_a_cross_partition": {"minimum_distance": min(distance for _, _, distance in cross), "pairs_within_0_10": cross_review},
            "stage_a_sealed_holdout": {"minimum_distance_to_unsealed": min(distance for _, _, distance in sealed), "pairs_within_0_10": sealed_review, "nearest_by_design": [_nearest(row, [candidate for candidate in designs if candidate["partition"] != "sealed_holdout"]) for row in designs if row["partition"] == "sealed_holdout"]},
        },
        "justified_exceptions": [],
        "pre_freeze_requirement": "Repeat exact duplicate and distance audit, including radius sensitivity, on frozen candidate bytes.",
    }


def _load_original_outcomes(path: Path) -> dict[str, dict[str, Any]]:
    by_design: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_design[row["design_id"]].append(row)
    result: dict[str, dict[str, Any]] = {}
    for design_id, rows in by_design.items():
        margins = [Decimal(row["signed_margin"]) for row in rows]
        outcome = design_outcome(margins)
        result[design_id] = {**outcome, "split": rows[0]["split"], "margins": margins}
    if len(result) != 100:
        raise ValueError("Original outcome evidence must contain 100 designs")
    return result


def build_final_gate_feasibility(boundary_path: Path) -> dict[str, Any]:
    outcomes = _load_original_outcomes(boundary_path)
    stage_b_capacity = {"train": 60, "validation": 15, "test": 15}
    evidence: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        split_outcomes = [value for value in outcomes.values() if value["split"] == split]
        original_designs = len(split_outcomes)
        original_non_designs = sum(value["majority_class"] == "non_breach_majority" for value in split_outcomes)
        original_breach_designs = original_designs - original_non_designs
        original_boundary = sum(value["observed_boundary_design"] for value in split_outcomes)
        margins = [margin for value in split_outcomes for margin in value["margins"]]
        original_non_runs = sum(margin >= 0 for margin in margins)
        original_breach_runs = len(margins) - original_non_runs
        original_distinct = len({canonical_margin(margin) for margin in margins})
        new_designs = stage_b_capacity[split]
        new_runs = new_designs * 5
        design_feasible_values = [value for value in range(new_designs + 1) if original_non_designs + value >= FINAL_GATES["minimum_non_breach_majority_designs"][split] and original_breach_designs + new_designs - value >= FINAL_GATES["minimum_breach_majority_designs"][split]]
        run_feasible_values = []
        for value in range(new_runs + 1):
            non_breach = original_non_runs + value
            breach = original_breach_runs + new_runs - value
            minority = min(non_breach, breach)
            ratio = float("inf") if minority == 0 else max(non_breach, breach) / minority
            if non_breach >= FINAL_GATES["minimum_non_breach_runs"][split] and breach >= FINAL_GATES["minimum_breach_runs"][split] and ratio <= FINAL_GATES["maximum_majority_to_minority_run_ratio"][split]:
                run_feasible_values.append(value)
        boundary_required = max(0, FINAL_GATES["minimum_observed_boundary_designs"][split] - original_boundary)
        distinct_required = max(0, FINAL_GATES["minimum_distinct_canonical_margins"][split] - original_distinct)
        feasible = bool(design_feasible_values) and bool(run_feasible_values) and boundary_required <= new_designs and distinct_required <= new_runs
        evidence[split] = {
            "original": {"designs": original_designs, "runs": len(margins), "non_breach_majority_designs": original_non_designs, "breach_majority_designs": original_breach_designs, "non_breach_runs": original_non_runs, "breach_runs": original_breach_runs, "observed_boundary_designs": original_boundary, "distinct_canonical_margins": original_distinct},
            "stage_b_capacity": {"designs": new_designs, "runs": new_runs},
            "combined_capacity": {"designs": original_designs + new_designs, "runs": len(margins) + new_runs},
            "exact_feasible_stage_b_non_breach_majority_design_interval": {"minimum": min(design_feasible_values) if design_feasible_values else None, "maximum": max(design_feasible_values) if design_feasible_values else None},
            "exact_feasible_stage_b_non_breach_run_interval": {"minimum": min(run_feasible_values) if run_feasible_values else None, "maximum": max(run_feasible_values) if run_feasible_values else None},
            "minimum_new_observed_boundary_designs_required": boundary_required,
            "maximum_new_observed_boundary_designs_available": new_designs,
            "minimum_new_distinct_canonical_margins_required": distinct_required,
            "maximum_new_distinct_canonical_margins_available": new_runs,
            "all_gates_mathematically_feasible": feasible,
        }
    result = {
        "scope": "primary combined classification corpus: original frozen corpus + future frozen Stage B corpus",
        "stage_a_excluded": True,
        "expected_size_if_stage_b_remains_90_designs": {"designs": 190, "runs": 950},
        "gate_definitions": {
            "design_class": "Five-realization majority; a mixed design counts once according to majority and never toward both class minima.",
            "observed_boundary_design": "minimum margin < 0 and maximum margin >= 0, or at least two absolute margins <= 0.05",
            "distinct_margin": f"Margin quantized to {MARGIN_QUANTUM} with round-half-even; count unique canonical values.",
            "run_ratio": "max(breach runs, non-breach runs) / min(breach runs, non-breach runs)",
            "threshold": "margin = failure_adjusted_overall_service_fraction_min - 0.80; zero is non-breach",
        },
        "gates": FINAL_GATES,
        "by_split": evidence,
        "overall_feasible": all(value["all_gates_mathematically_feasible"] for value in evidence.values()),
    }
    if not result["overall_feasible"]:
        raise ValueError("Corrected final classification gates are infeasible")
    return result


def _artifact_reference(name: str, payload: bytes) -> dict[str, Any]:
    return {"relative_path": name, "sha256": sha256_bytes(payload), "byte_length": len(payload)}


def build_artifact_payloads(repo_root: Path) -> dict[str, bytes]:
    original = load_original_designs(repo_root / "artifacts/final_integrated_dataset_contract/designs.jsonl")
    designs = build_design_rows()
    validate_design_rows(designs, original)
    distances = minimum_distances(designs)
    if distances["minimum_sealed_to_unsealed_distance"] <= 0.10:
        raise ValueError("Stage A sealed holdout violates near-neighbor isolation")
    runs = build_run_rows(designs)
    seeds = build_seed_rows(runs)
    partition = build_partition_manifest(designs, runs)
    seed_policy = build_seed_policy()
    region_bounds = build_region_bounds()
    output_roots = build_output_root_manifest()
    criteria = build_discovery_criteria()
    neighbors = build_near_neighbor_policy(designs, original)
    feasibility = build_final_gate_feasibility(repo_root / "artifacts/final_integrated_dataset_class_support_audit/audit_boundary_reproduction.csv")
    payloads: dict[str, bytes] = {
        "stage_a_design_manifest.csv": _csv_bytes(designs),
        "stage_a_run_manifest.csv": _csv_bytes(runs),
        "stage_a_partition_manifest.json": canonical_json_bytes(partition),
        "stage_a_seed_policy.json": canonical_json_bytes(seed_policy),
        "stage_a_seed_manifest.csv": _csv_bytes(seeds),
        "stage_a_region_bounds.json": canonical_json_bytes(region_bounds),
        "stage_a_output_root_manifest.json": canonical_json_bytes(output_roots),
        "stage_a_discovery_criteria.json": canonical_json_bytes(criteria),
        "stage_a_near_neighbor_policy.json": canonical_json_bytes(neighbors),
    }
    refs = {name.removesuffix(".json").removesuffix(".csv"): _artifact_reference(name, payload) for name, payload in payloads.items()}
    contract = {
        "proposal_schema": PROPOSAL_SCHEMA,
        "proposal_version": PROPOSAL_VERSION,
        "proposal_status": PROPOSAL_STATUS,
        "simulation_authorized": SIMULATION_AUTHORIZED,
        "base_corpus_identities": {"production_tooling_sha": PRODUCTION_TOOLING_SHA, "frozen_contract_commit": FROZEN_CONTRACT_COMMIT, "contract_specification_hash": CONTRACT_SPECIFICATION_HASH, "generation_ledger_sha256": GENERATION_LEDGER_SHA256, "replay_ledger_sha256": REPLAY_LEDGER_SHA256, "freeze_archive_sha256": FREEZE_ARCHIVE_SHA256},
        "audit_identities": {"audit_commit_corrected": AUDIT_COMMIT, "audit_implementation_commit": AUDIT_IMPLEMENTATION_COMMIT, "analysis_commit_audited": ANALYSIS_COMMIT, "analysis_inventory_sha256": ANALYSIS_INVENTORY_SHA256, "audit_inventory_sha256": AUDIT_INVENTORY_SHA256},
        "corpus_namespace": CORPUS_NAMESPACE,
        "scientific_objective": "Outcome-independent Stage A discovery of reproducible resilient-core support and design-level threshold transitions without altering frozen science.",
        "artifact_references": refs,
        "target_schema_reference": "Frozen final integrated eight-target schema; no target or threshold change",
        "threshold_definition": {"value": "0.80", "margin": "failure_adjusted_overall_service_fraction_min - 0.80", "non_breach": "margin >= 0", "breach": "margin < 0"},
        "boundary_definitions": {"preassigned_boundary_region": "Pre-simulation DOE assignment only", "observed_boundary_design": "Five margins straddle zero or at least two have absolute margin <= 0.05", "endpoint_behavior": "Absolute margin 0.05 qualifies; margin 0 is non-breach"},
        "class_support_definitions": {"non_breach_majority": "at least 3 of 5 non-breach", "breach_majority": "at least 3 of 5 breach", "mixed_design": "at least one of each class; counts once by majority and may separately count as observed boundary"},
        "distinct_margin_definition": {"quantization_increment": "0.000001", "rounding_mode": "ROUND_HALF_EVEN", "canonical_margin": "signed boundary margin quantized to six decimal places"},
        "holdout_policy": partition["partitions"]["sealed_holdout"] | {"unsealing_conditions": partition["unsealing_conditions"], "not_final_model_test_set": True},
        "final_corpus_membership": {"primary_final_classification_corpus": ["original frozen 500-run corpus", "future frozen Stage B corpus"], "stage_a_development": "EXCLUDED", "stage_a_validation": "EXCLUDED", "stage_a_sealed_holdout": "EXCLUDED", "expected_if_stage_b_remains_90_designs": {"designs": 190, "runs": 950}, "future_training_inclusion": "Requires a separately proposed and audited corpus contract"},
        "stage_b_adaptation_boundary": {"proposal_status": PROPOSAL_STATUS, "simulation_authorized": False, "planning_baseline": {"designs": 90, "runs": 450, "resilient_core": 36, "boundary": 36, "global_control": 18}, "adaptation_sources": ["Stage A development", "Stage A validation"], "prohibited_source": "Stage A sealed holdout", "required_process": ["new Stage B proposal", "new design/run/seed manifests", "frozen split", "independent audit", "recorded contract hash"]},
        "acceptance_definitions": {"stage_a": "Discovery criteria only; final classification gates do not apply", "final_classification": feasibility},
        "original_split_preservation": {"train_designs": 70, "validation_designs": 15, "test_designs": 15, "rule": "Future Stage B appends designs; every original design remains in its frozen split"},
        "required_future_freeze_steps": ["Independent freeze-readiness audit", "Resolve any reviewed near-neighbor exception", "Freeze exact artifact bytes and hashes", "Record frozen contract hash", "Explicitly authorize simulation in a later task"],
        "required_independent_audit_steps": ["Verify every proposal artifact and inventory hash", "Recompute exact duplicates and distance reports", "Recompute seeds and record hashes", "Verify holdout and final-corpus semantics", "Verify final-gate feasibility", "Verify output roots remain absent and isolated"],
    }
    if contract["proposal_status"] != PROPOSAL_STATUS or contract["simulation_authorized"] is not False:
        raise ValueError("Stage A contract proposal authorization boundary violated")
    payloads["stage_a_contract_proposal.json"] = canonical_json_bytes(contract)
    return payloads


def build_inventory(payloads: Mapping[str, bytes]) -> dict[str, Any]:
    entries = []
    for name in sorted(payloads):
        payload = payloads[name]
        record_count = None
        if name.endswith(".csv"):
            record_count = max(0, payload.decode("utf-8").count("\n") - 1)
        entries.append({"relative_path": name, "byte_length": len(payload), "sha256": sha256_bytes(payload), "record_count": record_count, "schema_identifier": ARTIFACT_SCHEMAS[name]})
    return {
        "schema_identifier": "satnet.stage_a.proposal_inventory.v1",
        "proposal_status": PROPOSAL_STATUS,
        "simulation_authorized": SIMULATION_AUTHORIZED,
        "ordering": "relative_path ascending",
        "encoding": "UTF-8",
        "self_reference_policy": "This inventory intentionally excludes its own bytes to avoid a self-referential hash cycle.",
        "artifact_count_excluding_inventory": len(entries),
        "artifacts": entries,
    }


def validate_contract_authorization(contract: Mapping[str, Any]) -> None:
    if contract.get("proposal_status") != PROPOSAL_STATUS:
        raise ValueError("Stage A proposal status must remain NOT_FROZEN")
    if contract.get("simulation_authorized") is not False:
        raise ValueError("Stage A simulation authorization must remain false")


def write_proposal_artifacts(repo_root: Path, output_root: Path) -> dict[str, Any]:
    if any(paths_overlap(output_root, frozen) for frozen in FROZEN_ROOTS):
        raise ValueError("Proposal artifacts cannot be written beneath frozen evidence roots")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Proposal output root must be absent or empty: {output_root}")
    payloads = build_artifact_payloads(repo_root)
    inventory = build_inventory(payloads)
    payloads["stage_a_proposal_inventory.json"] = canonical_json_bytes(inventory)
    output_root.mkdir(parents=True, exist_ok=True)
    for name in sorted(payloads):
        (output_root / name).write_bytes(payloads[name])
    return {"output_count": len(payloads), "proposal_inventory_sha256": sha256_bytes(payloads["stage_a_proposal_inventory.json"]), "proposal_status": PROPOSAL_STATUS, "simulation_authorized": SIMULATION_AUTHORIZED}
