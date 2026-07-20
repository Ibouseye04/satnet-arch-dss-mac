from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .constants import CONTRACT_SPEC_HASH
from .io import atomic_write_json, read_canonical_json
from .mapping import FinalRunMapping
from .orchestrator import artifact_paths, run_directory, validate_completed_run


def _attempt_records(root: Path, run_id: int) -> tuple[dict[str, Any], ...]:
    attempt_root = root / "operational" / "attempts" / f"run_{run_id:03d}"
    if not attempt_root.exists():
        return ()
    return tuple(read_canonical_json(path) for path in sorted(attempt_root.glob("attempt_*.json")))


def materialize_generation_ledger(
    *, output_root: str | Path, mappings: Sequence[FinalRunMapping]
) -> dict[str, Any]:
    root = Path(output_root)
    records: list[dict[str, Any]] = []
    distinct_submissions = 0
    successful = 0
    attempt_events = 0
    for mapping in sorted(mappings, key=lambda item: item.run_id):
        attempts = _attempt_records(root, mapping.run_id)
        attempt_events += len(attempts)
        if attempts:
            distinct_submissions += 1
        result_hash: str | None = None
        state = "not_started"
        final_root = run_directory(root, mapping.run_id)
        if final_root.exists():
            result = validate_completed_run(mapping=mapping, run_root=final_root)
            result_hash = result["run_result_hash"]
            state = "succeeded"
            successful += 1
        elif attempts:
            state = attempts[-1]["state"]
        records.append(
            {
                "attempt_count": len(attempts),
                "design_id": mapping.design["design_id"],
                "failure_evidence_hashes": [
                    record.get("failure_evidence_hash")
                    for record in attempts
                    if record["state"] == "failed" and record.get("failure_evidence_hash")
                ],
                "published_result_hash": result_hash,
                "realization_id": mapping.run["realization_id"],
                "run_id": mapping.run_id,
                "run_key": mapping.run_key,
                "run_record_hash": mapping.run["run_record_hash"],
                "split": mapping.run["split_assignment"],
                "state": state,
            }
        )
    ledger = {
        "contract_spec_hash": CONTRACT_SPEC_HASH,
        "distinct_frozen_run_submission_count": distinct_submissions,
        "generation_ledger_schema_version": "1",
        "operational_attempt_event_count": attempt_events,
        "records": records,
        "successful_generation_count": successful,
    }
    atomic_write_json(root / "operational" / "generation_ledger.json", ledger, overwrite=True)
    return ledger


def materialize_replay_ledger(
    *, replay_root: str | Path, mappings: Sequence[FinalRunMapping]
) -> dict[str, Any]:
    root = Path(replay_root)
    records: list[dict[str, Any]] = []
    for mapping in sorted(mappings, key=lambda item: item.run_id):
        path = run_directory(root, mapping.run_id) / "replay_report.json"
        if path.is_file():
            records.append(read_canonical_json(path))
    run_ids = [record["run_id"] for record in records]
    if run_ids != sorted(set(run_ids)):
        raise ValueError("Replay ledger contains duplicate or unordered run IDs")
    ledger = {
        "contract_spec_hash": CONTRACT_SPEC_HASH,
        "records": records,
        "replay_ledger_schema_version": "1",
        "replay_submission_count": len(records),
        "successful_replay_count": sum(
            record["replay_state"] == "succeeded" for record in records
        ),
    }
    atomic_write_json(root / "replay_ledger.json", ledger, overwrite=True)
    return ledger
