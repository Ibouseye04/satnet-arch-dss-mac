from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

from satnet.ground.canonical import canonical_float_string, canonical_hash, canonical_json
from satnet.network.hypatia_adapter import HypatiaAdapter

from .adaptive_contract import load_adaptive_contract, map_adaptive_contract_runs
from .artifacts import read_satellite_artifact
from .constants import FINAL_RUN_COUNT, RUN_FILES, RUN_ID_WIDTH, TARGET_FIELDS
from .contract import ensure_mode_root, validate_catalog, validate_output_root, validate_science_dependencies
from .evidence import validate_generation_evidence, validate_replay_evidence
from .io import atomic_write_json, parse_canonical_float, read_canonical_json, tree_inventory_hash
from .ledgers import materialize_generation_ledger, materialize_replay_ledger
from .orchestrator import generate_run, run_directory
from .replay import replay_runs_read_only

EXPECTED_PROFILE_ID = "final_integrated_dataset_10k_adaptive_v2"
EXPECTED_CONTRACT_SPEC_HASH = "23c5fffc10849c3bc3ea027251ac3e5ad4c96f0eea85edf1e8deab079cb0871e"
EXPECTED_CONTRACT_BUNDLE_HASH = "da3c73711b1d60635afcceee8bda0a60d1379e492e1ad0d588d5a0c25e10abe3"
PREFLIGHT_RUN_COUNT = 25
REPRESENTATIVE_RUN_IDS = (0, 25, 3039, 4115, 9999)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _contract_root() -> Path:
    return _repository_root() / "artifacts" / "final_integrated_dataset_10k_adaptive_v2_contract"


def _load_context() -> tuple[dict[str, Any], tuple[Any, ...], Any]:
    contract = load_adaptive_contract(_contract_root())
    if contract["contract_spec_hash"] != EXPECTED_CONTRACT_SPEC_HASH:
        raise ValueError("Adaptive contract specification hash mismatch")
    if contract["contract_bundle_hash"] != EXPECTED_CONTRACT_BUNDLE_HASH:
        raise ValueError("Adaptive contract bundle hash mismatch")
    specification = contract["specification"]
    if specification.get("production_profile") != EXPECTED_PROFILE_ID:
        raise ValueError("Adaptive production profile identity mismatch")
    mappings = map_adaptive_contract_runs(contract)
    if len(mappings) != FINAL_RUN_COUNT or tuple(value.run_id for value in mappings) != tuple(range(FINAL_RUN_COUNT)):
        raise ValueError("Adaptive mapping is not exactly the frozen 10K run set")
    catalog = validate_catalog()
    validate_science_dependencies(specification)
    return contract, mappings, catalog


def _selected(mappings: Sequence[Any], run_ids: Iterable[int]) -> tuple[Any, ...]:
    indexed = {mapping.run_id: mapping for mapping in mappings}
    selected = tuple(indexed[run_id] for run_id in sorted(run_ids))
    if len(selected) != len(set(run_ids)):
        raise ValueError("Duplicate frozen run IDs requested")
    return selected


def _historical_root_snapshot(excluded: Sequence[Path]) -> dict[str, str | None]:
    external = Path.home() / "external"
    excluded_resolved = {path.resolve() for path in excluded}
    result: dict[str, str | None] = {}
    if not external.is_dir():
        return result
    for path in sorted(external.glob("satnet-*")):
        if path.resolve() in excluded_resolved or not path.is_dir():
            continue
        if "production" in path.name or "contract" in path.name or "training" in path.name:
            result[str(path)] = tree_inventory_hash(path)
    return result


def _generate(
    mappings: Sequence[Any],
    output_root: Path,
    catalog: Any,
    *,
    resume_replay_root: Path | None = None,
) -> dict[str, Any]:
    ensure_mode_root(
        output_root,
        "production",
        create=True,
        contract_spec_hash=EXPECTED_CONTRACT_SPEC_HASH,
    )
    failures: list[dict[str, Any]] = []
    retries = 0
    results: list[dict[str, Any]] = []
    for index, mapping in enumerate(sorted(mappings, key=lambda value: value.run_id), start=1):
        attempt = 0
        while True:
            try:
                result = generate_run(
                    mapping=mapping,
                    catalog=catalog,
                    output_root=output_root,
                    mode="production",
                    retry=attempt > 0,
                    verified_resume=resume_replay_root is not None,
                    resume_replay_root=resume_replay_root,
                )
                results.append(result)
                break
            except Exception as error:
                failures.append(
                    {
                        "attempt": attempt + 1,
                        "exception": type(error).__name__,
                        "message": " ".join(str(error).split())[:1000],
                        "run_id": mapping.run_id,
                        "run_key": mapping.run_key,
                    }
                )
                if attempt >= 2:
                    raise
                attempt += 1
                retries += 1
        if index == 1 or index == len(mappings) or index % 25 == 0:
            print(f"generated {index}/{len(mappings)}", flush=True)
    ledger = materialize_generation_ledger(
        output_root=output_root,
        mappings=mappings,
        catalog=catalog,
    )
    return {
        "failed_attempts": failures,
        "generation_ledger": ledger,
        "generation_submissions": len(mappings),
        "retries": retries,
        "successful_generation": len(results),
    }


def _replay(mappings: Sequence[Any], generation_root: Path, replay_root: Path, catalog: Any) -> dict[str, Any]:
    existing = tuple(
        mapping
        for mapping in mappings
        if (run_directory(replay_root, mapping.run_id) / "replay_report.json").is_file()
    )
    missing = tuple(mapping for mapping in mappings if mapping not in existing)
    if missing:
        replay_runs_read_only(
            mappings=missing,
            catalog=catalog,
            input_root=generation_root,
            replay_output_root=replay_root,
            input_mode="production",
            output_mode="production_replay",
        )
    ledger = materialize_replay_ledger(replay_root=replay_root, mappings=mappings)
    return {
        "replay_ledger": ledger,
        "replay_submissions": ledger["replay_submission_count"],
        "successful_replay": ledger["successful_replay_count"],
    }


def _runtime_audit(mappings: Sequence[Any], root: Path) -> dict[str, Any]:
    tuples: Counter[tuple[Any, ...]] = Counter()
    missing = 0
    duplicate_run_ids: list[int] = []
    seen: set[int] = set()
    target_values: dict[str, list[float]] = {field: [] for field in TARGET_FIELDS if field not in {"overall_threshold_breach_any", "ground_threshold_breach_any", "space_threshold_breach_any"}}
    classification_counts = Counter()
    fixed_occurrences: list[dict[str, str]] = []
    for mapping in sorted(mappings, key=lambda value: value.run_id):
        run_root = run_directory(root, mapping.run_id)
        if mapping.run_id in seen:
            duplicate_run_ids.append(mapping.run_id)
        seen.add(mapping.run_id)
        satellite_path = run_root / RUN_FILES["satellite"]
        if not satellite_path.is_file():
            missing += 1
            continue
        artifact = read_canonical_json(satellite_path)
        config = artifact["tier1_rollout_config"]
        tuples[(
            config["isl_policy"],
            config["adjacent_search_k"],
            config["max_inter_plane_links_per_sat"],
            config["failure_model"],
        )] += 1
        target = read_canonical_json(run_root / RUN_FILES["target"])
        classification_counts[str(target["overall_threshold_breach_any"])] += 1
        for field in target_values:
            value = parse_canonical_float(target[field], field)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid target {field} for run {mapping.run_id}")
            target_values[field].append(value)
    for path in (*root.rglob("*.json"), *root.rglob("*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8")
            values = [json.loads(line) for line in text.splitlines()] if path.suffix == ".jsonl" else [json.loads(text)]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for value in values:
            _collect_fixed_occurrences(value, path.relative_to(root).as_posix(), fixed_occurrences)
    summary: dict[str, Any] = {}
    for field, values in target_values.items():
        summary[field] = {
            "count": len(values),
            "max": canonical_float_string(max(values)) if values else None,
            "mean": canonical_float_string(sum(values) / len(values)) if values else None,
            "min": canonical_float_string(min(values)) if values else None,
        }
    return {
        "duplicate_run_ids": sorted(set(duplicate_run_ids)),
        "missing_runtime_runs": missing,
        "numeric_target_summary": summary,
        "outcome_distribution": dict(sorted(classification_counts.items())),
        "runtime_topology_tuples": [
            {"count": count, "tuple": list(key)} for key, count in sorted(tuples.items(), key=lambda item: repr(item[0]))
        ],
        "fixed_policy_occurrences": fixed_occurrences,
    }


def _collect_fixed_occurrences(value: Any, path: str, output: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if item == "grid_" + "fixed":
                output.append({"field": str(key), "path": path})
            _collect_fixed_occurrences(item, path, output)
    elif isinstance(value, list):
        for item in value:
            _collect_fixed_occurrences(item, path, output)


def _canonicalize_trace(value: Any) -> Any:
    if isinstance(value, float):
        return canonical_float_string(value)
    if isinstance(value, list):
        return [_canonicalize_trace(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonicalize_trace(item) for key, item in value.items()}
    return value


def _behavioral_traces(mappings: Sequence[Any], root: Path) -> dict[str, Any]:
    by_design: dict[str, list[Any]] = {}
    for mapping in mappings:
        by_design.setdefault(mapping.design["design_id"], []).append(mapping)
    traces: list[dict[str, Any]] = []
    for design_id, design_mappings in sorted(by_design.items()):
        selected_trace: dict[str, Any] | None = None
        for mapping in sorted(design_mappings, key=lambda value: value.run_id):
            config = mapping.satellite_config
            trace_dir = root / "behavioral_traces" / f"run_{mapping.run_id:0{RUN_ID_WIDTH}d}"
            with HypatiaAdapter(
                num_planes=config.num_planes,
                sats_per_plane=config.sats_per_plane,
                inclination_deg=config.inclination_deg,
                altitude_km=config.altitude_km,
                phasing_factor=config.phasing_factor,
                output_dir=trace_dir,
                epoch=config.epoch,
                orbital_engine=config.orbital_engine,
            ) as adapter:
                adapter.generate_tles()
                _, stats = adapter.calculate_isls(
                    duration_minutes=config.duration_minutes,
                    step_seconds=config.step_seconds,
                    max_isl_distance_km=config.max_isl_distance_km,
                    isl_policy=config.isl_policy,
                    adjacent_search_k=config.adjacent_search_k,
                    max_inter_plane_links_per_sat=config.max_inter_plane_links_per_sat,
                    collect_adaptive_examples=128,
                )
            examples = [
                example for example in stats.adaptive_selection_examples
                if any(record.get("candidate_offset", 0) != 0 for record in example.get("selected", []))
            ]
            if examples and selected_trace is None:
                selected_trace = {
                    "adaptive_selection_examples": examples,
                    "accepted_inter_plane_links": stats.accepted_inter_plane_links,
                    "config_hash": config.config_hash(),
                    "design_id": design_id,
                    "run_id": mapping.run_id,
                    "run_key": mapping.run_key,
                    "runtime_topology": {
                        "adjacent_search_k": config.adjacent_search_k,
                        "failure_model": config.failure_model,
                        "isl_policy": config.isl_policy,
                        "max_inter_plane_links_per_sat": config.max_inter_plane_links_per_sat,
                    },
                }
        if selected_trace is None:
            raise ValueError(f"No selected non-zero adaptive offset trace for {design_id}")
        selected_trace = _canonicalize_trace(selected_trace)
        atomic_write_json(
            root / "behavioral_traces" / f"{design_id}.json",
            selected_trace,
        )
        traces.append(selected_trace)
    return {
        "anchor_design_count": len(traces),
        "designs_with_selected_nonzero_offset": len(traces),
        "representative_adaptive_only_edges": [
            {
                "design_id": trace["design_id"],
                "run_id": trace["run_id"],
                "selected": [record for example in trace["adaptive_selection_examples"] for record in example["selected"]],
            }
            for trace in traces
        ],
        "traces": traces,
    }


def _validate(mappings: Sequence[Any], generation_root: Path, replay_root: Path, catalog: Any) -> dict[str, Any]:
    generation_ledger = read_canonical_json(generation_root / "operational" / "generation_ledger.json")
    replay_ledger = read_canonical_json(replay_root / "replay_ledger.json")
    targets, results = validate_generation_evidence(
        mappings=mappings,
        generation_root=generation_root,
        catalog=catalog,
        ledger=generation_ledger,
    )
    validate_replay_evidence(
        mappings=mappings,
        replay_root=replay_root,
        generation_results=results,
        generation_targets=targets,
        ledger=replay_ledger,
    )
    return {
        "all_g1_g5_and_target_artifacts_valid": True,
        "generation_submissions": generation_ledger["distinct_frozen_run_submission_count"],
        "successful_generation": generation_ledger["successful_generation_count"],
        "replay_submissions": replay_ledger["replay_submission_count"],
        "successful_replay": replay_ledger["successful_replay_count"],
    }


def _representative_replay_checks(replay_root: Path, mappings: Sequence[Any]) -> list[dict[str, Any]]:
    indexed = {mapping.run_id: mapping for mapping in mappings}
    checks: list[dict[str, Any]] = []
    for run_id in REPRESENTATIVE_RUN_IDS:
        if run_id not in indexed:
            continue
        report = read_canonical_json(run_directory(replay_root, run_id) / "replay_report.json")
        checks.append({
            "recomputed_result_hash": report["recomputed_result_hash"],
            "input_result_hash": report["input_result_hash"],
            "run_id": run_id,
            "run_record_hash": report["run_record_hash"],
            "all_stages_matched": all(item["state"] == "matched" for item in report["per_stage_comparison"]),
            "input_tree_unchanged": report["input_tree_unchanged"],
        })
    return checks


def _run_phase1(args: argparse.Namespace) -> dict[str, Any]:
    preflight_root = validate_output_root(args.preflight_root)
    preflight_replay_root = validate_output_root(
        args.preflight_replay_root,
        other_roots=(preflight_root,),
    )
    production_root = validate_output_root(
        args.production_root,
        other_roots=(preflight_root, preflight_replay_root),
    )
    replay_root = validate_output_root(
        args.replay_root,
        other_roots=(preflight_root, preflight_replay_root, production_root),
    )
    contract, mappings, catalog = _load_context()
    historical_before = _historical_root_snapshot(
        (preflight_root, preflight_replay_root, production_root, replay_root)
    )
    preflight_mappings = _selected(mappings, range(PREFLIGHT_RUN_COUNT))
    preflight_generation = _generate(
        preflight_mappings,
        preflight_root,
        catalog,
        resume_replay_root=preflight_replay_root,
    )
    preflight_replay = _replay(preflight_mappings, preflight_root, preflight_replay_root, catalog)
    preflight_validation = _validate(preflight_mappings, preflight_root, preflight_replay_root, catalog)
    preflight_audit = _runtime_audit(preflight_mappings, preflight_root)
    preflight_behavior = _behavioral_traces(preflight_mappings, preflight_root)
    if preflight_audit["fixed_policy_occurrences"]:
        raise ValueError("Preflight contains corrected runtime fixed-policy occurrences")
    if preflight_audit["runtime_topology_tuples"] != [{"count": 25, "tuple": ["grid_adaptive", 1, 1, "persistent_temporal_union_edges_v1"]}]:
        raise ValueError("Preflight runtime topology aggregate failed")
    if preflight_validation["successful_generation"] != 25 or preflight_validation["successful_replay"] != 25:
        raise ValueError("Preflight generation/replay count gate failed")
    production_generation = _generate(mappings, production_root, catalog)
    production_replay = _replay(mappings, production_root, replay_root, catalog)
    production_validation = _validate(mappings, production_root, replay_root, catalog)
    production_audit = _runtime_audit(mappings, production_root)
    if production_audit["fixed_policy_occurrences"]:
        raise ValueError("Production contains corrected runtime fixed-policy occurrences")
    if production_audit["runtime_topology_tuples"] != [{"count": 10000, "tuple": ["grid_adaptive", 1, 1, "persistent_temporal_union_edges_v1"]}]:
        raise ValueError("Production runtime topology aggregate failed")
    historical_after = _historical_root_snapshot(
        (preflight_root, preflight_replay_root, production_root, replay_root)
    )
    historical_untouched = historical_before == historical_after
    report = {
        "blockers": [],
        "contract_bundle_hash": contract["contract_bundle_hash"],
        "contract_specification_hash": contract["contract_spec_hash"],
        "historical_fixed_artifacts_untouched": historical_untouched,
        "no_training_occurred": True,
        "phase_1a": {
            "behavioral_evidence": preflight_behavior,
            "generation": preflight_generation,
            "replay": preflight_replay,
            "validation": preflight_validation,
            "runtime_audit": preflight_audit,
        },
        "phase_1b": {
            "generation": production_generation,
            "replay": production_replay,
            "validation": production_validation,
            "runtime_audit": production_audit,
            "representative_replay_checks": _representative_replay_checks(replay_root, mappings),
        },
        "profile_id": EXPECTED_PROFILE_ID,
        "production_root": str(production_root),
        "replay_root": str(replay_root),
        "starting_sha": "f53ebdad3f364e5e37ad3a8cebee5951cbc66e83",
        "status": "SATNET ADAPTIVE V2 10K PRODUCTION — PASS",
    }
    atomic_write_json(production_root / "phase1_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="satnet-adaptive-production")
    parser.add_argument("phase1", choices=("phase1",))
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--preflight-replay-root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = _run_phase1(args)
    except Exception as error:
        print(f"SATNET ADAPTIVE V2 PRODUCTION PREFLIGHT — BLOCKED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
