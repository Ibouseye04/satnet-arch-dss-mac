from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping

from satnet.experiments.final_generation.constants import CONTRACT_SPEC_HASH as BASE_CONTRACT_SPEC_HASH
from satnet.experiments.final_generation.contract import validate_catalog
from satnet.experiments.final_generation.mapping import FinalRunMapping
from satnet.experiments.final_generation.orchestrator import generate_run
from satnet.experiments.final_generation.run_validation import validate_run_authoritatively
from satnet.ground.catalog import GroundStationCatalog
from satnet.ground.failure_policy import GroundFailurePolicy
from satnet.ground.persistence import make_enabled_ground_design_record
from satnet.ground.selection import GroundSegmentEnabledConfig, select_ground_stations
from satnet.ground.service_policy import GroundServicePolicy
from satnet.ground.visibility import GroundVisibilityPolicy
from satnet.simulation.tier1_rollout import Tier1RolloutConfig

from .artifact_contract import (
    PRODUCTION_ARTIFACT_CONTRACT,
    ScienceCompletionResult,
    SimulationAdapterResult,
    bind_plan_run,
    make_adapter_result,
    science_completion_hash,
    validate_run_output,
    validate_science_completion_result,
)
from .common import atomic_write_json, read_json_object
from .contract import FrozenStageAContract
from .integrity import inventory_hash
from .ledger import build_ledger, read_ledger, transition, write_ledger
from .locking import campaign_lock, per_run_lock
from .paths import validate_output_roots, validate_relative_artifact_path
from .preflight import PreflightCertificate, require_preflight
from .resume import resumable_run_ids, validate_resume_identity

SimulationAdapter = Callable[[Mapping[str, Any], Path], SimulationAdapterResult]
ScienceCompletionValidator = Callable[[Mapping[str, Any], Path, SimulationAdapterResult], ScienceCompletionResult]


def _failure_diagnostic(error: BaseException) -> str:
    message = " ".join(str(error).split())[:1000]
    return f"{type(error).__name__}: {message or type(error).__name__}"


def build_scientific_arguments(contract: FrozenStageAContract, global_run_id: int) -> dict[str, Any]:
    run = next((row for row in contract.runs if row["global_run_id"] == global_run_id), None)
    if run is None:
        raise ValueError("Unknown frozen Stage A run")
    design = next(row for row in contract.designs if row["design_id"] == run["design_id"])
    seed = next(row for row in contract.seeds if row["run_key"] == run["run_key"])
    required_design = (
        "num_planes", "sats_per_plane", "inclination_deg", "altitude_km", "phasing_factor",
        "duration_minutes", "step_seconds", "max_isl_distance_km", "isl_policy",
        "adjacent_search_k", "max_inter_plane_links_per_sat", "space_gcc_threshold",
        "satellite_node_failure_probability", "satellite_edge_failure_probability",
        "satellite_failure_model", "epoch_iso", "orbital_engine", "civilian_count",
        "government_count", "military_count", "minimum_elevation_deg",
        "ground_service_threshold", "ground_station_failure_probability",
    )
    if any(field not in design or design[field] == "" for field in required_design):
        raise ValueError("Frozen design omits a required scientific value")
    return {
        "contract_hash": contract.contract_hash,
        "global_run_id": run["global_run_id"],
        "pipeline_local_run_id": run["global_run_id"] - 500,
        "run_key": run["run_key"],
        "run_record_hash": run["run_record_hash"],
        "design_record_hash": design["design_record_hash"],
        "design": {field: design[field] for field in required_design},
        "design_construction_seed": seed["design_construction_seed"],
        "ground_selection_seed": seed["ground_selection_seed"],
        "satellite_failure_seed": seed["satellite_failure_seed"],
        "ground_failure_seed": seed["ground_failure_seed"],
    }


def _float(value: object) -> float:
    if isinstance(value, str):
        return float(value)
    raise TypeError("Frozen scientific floating-point value must be string encoded")


def _build_production_mapping(
    contract: FrozenStageAContract, catalog: GroundStationCatalog, plan_run: Mapping[str, Any],
) -> FinalRunMapping:
    arguments = build_scientific_arguments(contract, int(plan_run["global_run_id"]))
    if arguments["run_record_hash"] != plan_run["run_record_hash"]:
        raise ValueError("Adapter run-record hash mismatch")
    design_source = next(row for row in contract.designs if row["design_id"] == plan_run["design_id"])
    source = arguments["design"]
    satellite = Tier1RolloutConfig(
        num_planes=source["num_planes"], sats_per_plane=source["sats_per_plane"],
        inclination_deg=_float(source["inclination_deg"]), altitude_km=_float(source["altitude_km"]),
        phasing_factor=source["phasing_factor"], duration_minutes=source["duration_minutes"],
        step_seconds=source["step_seconds"], max_isl_distance_km=_float(source["max_isl_distance_km"]),
        isl_policy=source["isl_policy"], adjacent_search_k=source["adjacent_search_k"],
        max_inter_plane_links_per_sat=source["max_inter_plane_links_per_sat"],
        gcc_threshold=_float(source["space_gcc_threshold"]),
        node_failure_prob=_float(source["satellite_node_failure_probability"]),
        edge_failure_prob=_float(source["satellite_edge_failure_probability"]),
        failure_model=source["satellite_failure_model"], seed=arguments["satellite_failure_seed"],
        epoch_iso=source["epoch_iso"], orbital_engine=source["orbital_engine"],
    )
    ground = GroundSegmentEnabledConfig(
        civilian_count=source["civilian_count"], government_count=source["government_count"],
        military_count=source["military_count"], station_selection_seed=arguments["ground_selection_seed"],
    )
    visibility = GroundVisibilityPolicy(_float(source["minimum_elevation_deg"]))
    service = GroundServicePolicy(_float(source["space_gcc_threshold"]), _float(source["ground_service_threshold"]))
    failure = GroundFailurePolicy(_float(source["ground_station_failure_probability"]))
    selection = select_ground_stations(catalog=catalog, config=ground)
    local_id = arguments["pipeline_local_run_id"]
    ground_design = make_enabled_ground_design_record(
        run_id=local_id, satellite_config_hash=satellite.config_hash(), selection=selection,
    )
    design = dict(design_source)
    design.update({
        "contract_spec_hash": BASE_CONTRACT_SPEC_HASH,
        "selected_station_ids": list(selection.selected_station_ids),
        "ground_selection_hash": selection.selection_hash,
        "ground_design_hash": ground_design.ground_design_hash,
        "ground_failure_policy_hash": failure.ground_failure_policy_hash,
    })
    run = {
        "contract_spec_hash": BASE_CONTRACT_SPEC_HASH,
        "run_id": local_id,
        "global_run_id": arguments["global_run_id"],
        "run_key": arguments["run_key"],
        "run_record_hash": arguments["run_record_hash"],
        "design_id": design["design_id"],
        "design_record_hash": design["design_record_hash"],
        "realization_id": next(row["realization_id"] for row in contract.runs if row["global_run_id"] == arguments["global_run_id"]),
        "realization_index": local_id % 5,
        "design_construction_seed": arguments["design_construction_seed"],
        "ground_selection_seed": arguments["ground_selection_seed"],
        "satellite_seed": arguments["satellite_failure_seed"],
        "ground_failure_seed": arguments["ground_failure_seed"],
        "expected_satellite_config_hash": satellite.config_hash(),
        "split_assignment": plan_run["partition"],
    }
    return FinalRunMapping(design, run, satellite, ground, visibility, service, failure)


def _make_production_adapter(contract: FrozenStageAContract, catalog: GroundStationCatalog) -> SimulationAdapter:
    def adapter(plan_run: Mapping[str, Any], output_root: Path) -> SimulationAdapterResult:
        mapping = _build_production_mapping(contract, catalog, plan_run)
        local_id = mapping.run_id
        pipeline_root = output_root / "validated_pipeline"
        result = generate_run(mapping=mapping, catalog=catalog, output_root=pipeline_root, mode="production")
        atomic_write_json(output_root / "stage_a_binding.json", {
            "schema_identifier": "satnet.stage_a.run_binding.v2",
            "base_production_contract_specification_hash": BASE_CONTRACT_SPEC_HASH,
            "contract_hash": contract.contract_hash,
            "plan_hash": plan_run["plan_hash"],
            "stable_executable_commit": plan_run["stable_executable_commit"],
            "executable_inventory_hash": plan_run["executable_inventory_hash"],
            "tooling_proposal_hash": plan_run["tooling_proposal_hash"],
            "artifact_contract_hash": plan_run["artifact_contract_hash"],
            "partition": plan_run["partition"],
            "design_id": plan_run["design_id"],
            "design_record_hash": plan_run["design_record_hash"],
            "global_run_id": plan_run["global_run_id"],
            "pipeline_local_run_id": local_id,
            "realization_id": plan_run["realization_id"],
            "realization_index": plan_run["realization_index"],
            "run_key": plan_run["run_key"],
            "run_record_hash": plan_run["run_record_hash"],
            "design_construction_seed": plan_run["design_construction_seed"],
            "ground_selection_seed": plan_run["ground_selection_seed"],
            "satellite_failure_seed": plan_run["satellite_failure_seed"],
            "ground_failure_seed": plan_run["ground_failure_seed"],
            "validated_pipeline_result_hash": result["run_result_hash"],
        })
        return make_adapter_result(plan_run, output_root, PRODUCTION_ARTIFACT_CONTRACT)
    return adapter


def _make_production_science_validator(
    contract: FrozenStageAContract, catalog: GroundStationCatalog,
) -> ScienceCompletionValidator:
    def validator(
        plan_run: Mapping[str, Any], output_root: Path, adapter_result: SimulationAdapterResult,
    ) -> ScienceCompletionResult:
        if adapter_result.artifact_contract != PRODUCTION_ARTIFACT_CONTRACT:
            raise ValueError("Production completion requires the production artifact contract")
        mapping = _build_production_mapping(contract, catalog, plan_run)
        run_root = output_root / "validated_pipeline" / f"run_{mapping.run_id:03d}"
        authoritative = validate_run_authoritatively(mapping=mapping, catalog=catalog, run_root=run_root)
        expected_stages = ("satellite", "g1", "g2", "g3", "g4", "g5", "target", "inventory", "result")
        observed_stages = tuple(record.get("stage") for record in authoritative.get("stages", ()))
        if observed_stages != expected_stages or any(record.get("state") != "matched" for record in authoritative["stages"]):
            raise ValueError("Authoritative satellite and G1-G5 completion is incomplete")
        binding = read_json_object(output_root / "stage_a_binding.json")
        result_hash = authoritative["result"].get("run_result_hash")
        if not isinstance(result_hash, str) or binding.get("validated_pipeline_result_hash") != result_hash:
            raise ValueError("Authoritative result and Stage A binding are inconsistent")
        payload = {
            "validation_status": "PASSED",
            "validation_kind": "authoritative_satellite_g1_g5_replay_v1",
            "verified_stages": list(expected_stages),
            "authoritative_result_hash": result_hash,
        }
        return ScienceCompletionResult(
            validation_status=payload["validation_status"],
            validation_kind=payload["validation_kind"],
            verified_stages=expected_stages,
            authoritative_result_hash=result_hash,
            completion_identity=science_completion_hash(payload),
        )
    return validator


def initialize_campaign(root: Path, plan: dict[str, Any], authorization_hash: str) -> Path:
    root.mkdir(parents=False, exist_ok=False)
    ledger_path = root / "execution_ledger.json"
    write_ledger(ledger_path, build_ledger(plan, authorization_hash), overwrite=False)
    atomic_write_json(root / "campaign_identity.json", {
        "authorization_hash": authorization_hash,
        "contract_hash": plan["contract_hash"],
        "plan_hash": plan["plan_hash"],
        "stable_executable_commit": plan["stable_executable_commit"],
        "executable_inventory_hash": plan["executable_inventory_hash"],
        "tooling_proposal_hash": plan["tooling_proposal_hash"],
        "artifact_contract_hash": plan["artifact_contract_hash"],
    })
    return ledger_path


def _execute_generation(
    *, repo_root: Path, contract: FrozenStageAContract, plan: dict[str, Any], authorization_hash: str,
    preflight: PreflightCertificate, campaign_root: Path, adapter: SimulationAdapter,
    science_validator: ScienceCompletionValidator, resume: bool = False,
    retry_failed: bool = False,
) -> dict[str, Any]:
    require_preflight(preflight, plan=plan, authorization_hash=authorization_hash)
    roots = validate_output_roots(
        repo_root=repo_root,
        generation_root=Path(plan["output_roots"]["generation"]),
        replay_root=Path(plan["output_roots"]["replay"]),
        acceptance_root=Path(plan["output_roots"]["acceptance"]),
        require_absent=False,
    )
    if roots != plan["output_roots"] or str(campaign_root.resolve(strict=False)) != roots["generation"]:
        raise PermissionError("Generation root differs from preflight-bound plan")
    with campaign_lock(campaign_root, plan, authorization_hash):
        ledger_path = campaign_root / "execution_ledger.json"
        if resume:
            ledger = read_ledger(ledger_path)
            validate_resume_identity(ledger, plan, authorization_hash)
            run_ids = set(resumable_run_ids(ledger, campaign_root, retry_failed=retry_failed))
        else:
            ledger_path = initialize_campaign(campaign_root, plan, authorization_hash)
            run_ids = {row["global_run_id"] for row in plan["runs"]}
        for plan_run in plan["runs"]:
            run_id = plan_run["global_run_id"]
            if run_id not in run_ids:
                continue
            relative = validate_relative_artifact_path(plan_run["expected_output_relative_path"])
            final_root = campaign_root / relative
            if final_root.exists():
                raise FileExistsError("Changed or partial run artifact cannot be replaced")
            with per_run_lock(campaign_root, plan, authorization_hash, plan_run):
                transition(ledger_path, global_run_id=run_id, new_state="STARTING")
                temporary = Path(tempfile.mkdtemp(dir=campaign_root, prefix=f".{plan_run['run_key']}.in_progress."))
                try:
                    transition(ledger_path, global_run_id=run_id, new_state="RUNNING")
                    bound_run = bind_plan_run(plan, plan_run)
                    adapter_result = adapter(bound_run, temporary)
                    records = validate_run_output(bound_run, temporary, adapter_result)
                    science_completion = science_validator(bound_run, temporary, adapter_result)
                    validate_science_completion_result(science_completion)
                    final_root.parent.mkdir(parents=True, exist_ok=True)
                    temporary.replace(final_root)
                    transition(
                        ledger_path, global_run_id=run_id, new_state="SUCCEEDED",
                        artifacts=records, artifact_inventory_hash=inventory_hash(records),
                        adapter_result=adapter_result.as_dict(),
                        science_completion=science_completion.as_dict(),
                    )
                except KeyboardInterrupt:
                    transition(ledger_path, global_run_id=run_id, new_state="INTERRUPTED", failure="KeyboardInterrupt")
                    shutil.rmtree(temporary, ignore_errors=True)
                    raise
                except Exception as error:
                    transition(ledger_path, global_run_id=run_id, new_state="FAILED", failure=_failure_diagnostic(error))
                    failed_record = next(
                        record for record in read_ledger(ledger_path)["records"]
                        if record["global_run_id"] == run_id
                    )
                    failed = campaign_root / "failed_attempts" / f"{plan_run['run_key']}-{failed_record['attempt_count']:03d}"
                    failed.parent.mkdir(parents=True, exist_ok=True)
                    temporary.replace(failed)
                    raise
        return read_ledger(ledger_path)


def execute_generation(
    *, repo_root: Path, contract: FrozenStageAContract, plan: dict[str, Any], authorization_hash: str,
    preflight: PreflightCertificate, campaign_root: Path, resume: bool = False,
    retry_failed: bool = False,
) -> dict[str, Any]:
    catalog = validate_catalog()
    return _execute_generation(
        repo_root=repo_root, contract=contract, plan=plan, authorization_hash=authorization_hash,
        preflight=preflight, campaign_root=campaign_root,
        adapter=_make_production_adapter(contract, catalog),
        science_validator=_make_production_science_validator(contract, catalog),
        resume=resume, retry_failed=retry_failed,
    )
