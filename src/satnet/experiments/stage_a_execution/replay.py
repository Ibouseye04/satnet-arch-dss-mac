from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any

from satnet.experiments.final_generation.contract import validate_catalog

from .common import atomic_write_json
from .artifact_contract import bind_plan_run, validate_run_output, validate_science_completion_result
from .contract import FrozenStageAContract
from .generate import ScienceCompletionValidator, SimulationAdapter, _make_production_adapter, _make_production_science_validator
from .integrity import artifact_inventory, inventory_hash, verify_artifact_inventory
from .ledger import build_ledger, read_bound_ledger, read_ledger, transition, write_ledger
from .locking import campaign_lock, per_run_lock
from .paths import validate_output_roots, validate_relative_artifact_path
from .preflight import PreflightCertificate, require_preflight
from .resume import validate_ledger_binding

REPLAY_SCHEMA = "satnet.stage_a.replay_report.v1"


def _failure_diagnostic(error: BaseException) -> str:
    message = " ".join(str(error).split())[:1000]
    return f"{type(error).__name__}: {message or type(error).__name__}"


def _scientific(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record for record in records
        if "/operational/" not in f"/{record['relative_path']}/"
        and record["relative_path"] != "stage_a_binding.json"
    ]


def _execute_replay(
    *, repo_root: Path, plan: dict[str, Any], authorization_hash: str,
    preflight: PreflightCertificate, generation_root: Path, replay_root: Path,
    adapter: SimulationAdapter, science_validator: ScienceCompletionValidator,
) -> dict[str, Any]:
    require_preflight(preflight, plan=plan, authorization_hash=authorization_hash)
    roots = validate_output_roots(
        repo_root=repo_root,
        generation_root=Path(plan["output_roots"]["generation"]),
        replay_root=Path(plan["output_roots"]["replay"]),
        acceptance_root=Path(plan["output_roots"]["acceptance"]),
        require_absent=False,
    )
    if roots != plan["output_roots"] or str(generation_root.resolve(strict=False)) != roots["generation"] or str(replay_root.resolve(strict=False)) != roots["replay"]:
        raise PermissionError("Replay roots differ from preflight-bound plan")
    generation_ledger, source_identity = read_bound_ledger(
        generation_root,
        relative_path=plan["source_generation_ledger_relative_path"],
        byte_length=plan["source_generation_ledger_byte_length"],
        sha256=plan["source_generation_ledger_sha256"],
    )
    validate_ledger_binding(generation_ledger, plan, operation="GENERATE")
    with campaign_lock(replay_root, plan, authorization_hash):
        replay_root.mkdir(parents=False, exist_ok=False)
        ledger_path = replay_root / "replay_ledger.json"
        write_ledger(ledger_path, build_ledger(plan, authorization_hash), overwrite=False)
        for plan_run in plan["runs"]:
            run_id = plan_run["global_run_id"]
            relative = validate_relative_artifact_path(plan_run["expected_output_relative_path"])
            source = generation_root / relative
            source_record = next(row for row in generation_ledger["records"] if row["global_run_id"] == run_id)
            if source_record["state"] != "SUCCEEDED":
                raise ValueError("Replay source run is not successful")
            verify_artifact_inventory(source, source_record["artifacts"])
            with per_run_lock(replay_root, plan, authorization_hash, plan_run):
                transition(ledger_path, global_run_id=run_id, new_state="STARTING")
                temporary = Path(tempfile.mkdtemp(dir=replay_root, prefix=f".{plan_run['run_key']}.replay."))
                try:
                    transition(ledger_path, global_run_id=run_id, new_state="RUNNING")
                    bound_run = bind_plan_run(plan, plan_run)
                    adapter_result = adapter(bound_run, temporary)
                    validated_records = validate_run_output(bound_run, temporary, adapter_result)
                    science_completion = science_validator(bound_run, temporary, adapter_result)
                    validate_science_completion_result(science_completion)
                    expected = _scientific(source_record["artifacts"])
                    observed = _scientific(validated_records)
                    matched = expected == observed
                    report = {
                        "schema_identifier": REPLAY_SCHEMA,
                        "contract_hash": plan["contract_hash"],
                        "plan_hash": plan["plan_hash"],
                        "authorization_hash": authorization_hash,
                        "generation_authorization_hash": generation_ledger["authorization_hash"],
                        "generation_plan_hash": generation_ledger["plan_hash"],
                        "generation_campaign_manifest_hash": generation_ledger["campaign_manifest_hash"],
                        "source_generation_ledger_relative_path": source_identity["relative_path"],
                        "source_generation_ledger_byte_length": source_identity["byte_length"],
                        "source_generation_ledger_sha256": source_identity["sha256"],
                        "stable_executable_commit": plan["stable_executable_commit"],
                        "executable_inventory_hash": plan["executable_inventory_hash"],
                        "tooling_proposal_hash": plan["tooling_proposal_hash"],
                        "artifact_contract_hash": plan["artifact_contract_hash"],
                        "global_run_id": run_id,
                        "run_key": plan_run["run_key"],
                        "run_record_hash": plan_run["run_record_hash"],
                        "design_construction_seed": plan_run["design_construction_seed"],
                        "ground_selection_seed": plan_run["ground_selection_seed"],
                        "satellite_failure_seed": plan_run["satellite_failure_seed"],
                        "ground_failure_seed": plan_run["ground_failure_seed"],
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
                    transition(
                        ledger_path, global_run_id=run_id, new_state="SUCCEEDED",
                        artifacts=records, artifact_inventory_hash=inventory_hash(records),
                        adapter_result=adapter_result.as_dict(),
                        science_completion=science_completion.as_dict(),
                    )
                except Exception as error:
                    transition(ledger_path, global_run_id=run_id, new_state="FAILED", failure=_failure_diagnostic(error))
                    shutil.rmtree(temporary, ignore_errors=True)
                    raise
        return read_ledger(ledger_path)


def execute_replay(
    *, repo_root: Path, contract: FrozenStageAContract, plan: dict[str, Any], authorization_hash: str,
    preflight: PreflightCertificate, generation_root: Path, replay_root: Path,
) -> dict[str, Any]:
    catalog = validate_catalog()
    return _execute_replay(
        repo_root=repo_root, plan=plan, authorization_hash=authorization_hash,
        preflight=preflight, generation_root=generation_root, replay_root=replay_root,
        adapter=_make_production_adapter(contract, catalog),
        science_validator=_make_production_science_validator(contract, catalog),
    )
