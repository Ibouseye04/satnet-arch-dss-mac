from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any

from .common import atomic_write_json
from .generate import SimulationAdapter
from .integrity import artifact_inventory, inventory_hash, verify_artifact_inventory
from .ledger import ExclusiveLock, build_ledger, read_ledger, transition, write_ledger
from .paths import validate_relative_artifact_path

REPLAY_SCHEMA = "satnet.stage_a.replay_report.v1"


def _scientific(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if "/operational/" not in f"/{record['relative_path']}/"]


def execute_replay(
    *, plan: dict[str, Any], authorization_hash: str, generation_root: Path,
    replay_root: Path, adapter: SimulationAdapter,
) -> dict[str, Any]:
    with ExclusiveLock(replay_root.with_name(replay_root.name + ".lock"), plan["plan_hash"]):
        replay_root.mkdir(parents=False, exist_ok=False)
        ledger_path = replay_root / "replay_ledger.json"
        ledger = build_ledger(plan, authorization_hash)
        ledger["schema_identifier"] = "satnet.stage_a.execution_ledger.v1"
        from .ledger import ledger_hash
        ledger["ledger_hash"] = ledger_hash(ledger)
        write_ledger(ledger_path, ledger, overwrite=False)
        for plan_run in plan["runs"]:
            run_id = plan_run["global_run_id"]
            relative = validate_relative_artifact_path(plan_run["expected_output_relative_path"])
            source = generation_root / relative
            generation_ledger = read_ledger(generation_root / "execution_ledger.json")
            source_record = next(row for row in generation_ledger["records"] if row["global_run_id"] == run_id)
            if source_record["state"] != "SUCCEEDED":
                raise ValueError("Replay source run is not successful")
            verify_artifact_inventory(source, source_record["artifacts"])
            transition(ledger_path, global_run_id=run_id, new_state="STARTING")
            temporary = Path(tempfile.mkdtemp(dir=replay_root, prefix=f".{plan_run['run_key']}.replay."))
            try:
                transition(ledger_path, global_run_id=run_id, new_state="RUNNING")
                adapter(plan_run, temporary)
                expected = _scientific(source_record["artifacts"])
                observed = _scientific(artifact_inventory(temporary))
                matched = expected == observed
                report = {
                    "schema_identifier": REPLAY_SCHEMA,
                    "contract_hash": plan["contract_hash"],
                    "plan_hash": plan["plan_hash"],
                    "global_run_id": run_id,
                    "run_key": plan_run["run_key"],
                    "run_record_hash": plan_run["run_record_hash"],
                    "generation_artifact_inventory_hash": inventory_hash(expected),
                    "replay_artifact_inventory_hash": inventory_hash(observed),
                    "generation_replay_equal": matched,
                    "replay_state": "SUCCEEDED" if matched else "FAILED",
                }
                atomic_write_json(temporary / "replay_report.json", report)
                if not matched:
                    raise ValueError("Generation/replay canonical output mismatch")
                records = artifact_inventory(temporary)
                final_root = replay_root / relative
                final_root.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(final_root)
                transition(ledger_path, global_run_id=run_id, new_state="SUCCEEDED", artifacts=records, artifact_inventory_hash=inventory_hash(records))
            except Exception as error:
                transition(ledger_path, global_run_id=run_id, new_state="FAILED", failure=type(error).__name__)
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        return read_ledger(ledger_path)
