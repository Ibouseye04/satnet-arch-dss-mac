"""Read-only access to the prepared adaptive final contract.

This module deliberately exposes contract qualification and mapping only. It
contains no production-generation entry point, so preparing the contract
cannot launch the 10K simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from satnet.experiments.final_dataset.materialize import (
    read_json,
    read_jsonl,
    validate_materialized_contract,
)
from satnet.experiments.final_dataset.specification import (
    validate_adaptive_contract_specification,
)
from satnet.experiments.production_profile import FINAL_ADAPTIVE_PRODUCTION_PROFILE

from .mapping import FinalRunMapping, map_adaptive_run

ADAPTIVE_CONTRACT_ROOT_NAME = "final_integrated_dataset_10k_adaptive_v2_contract"


def adaptive_contract_root(repository_root: str | Path) -> Path:
    return Path(repository_root) / "artifacts" / ADAPTIVE_CONTRACT_ROOT_NAME


def load_adaptive_contract(root: str | Path) -> dict[str, Any]:
    """Validate and load the prepared adaptive contract without simulation."""

    contract_root = Path(root)
    pilot_manifest = contract_root / "pilot_inputs" / "inputs" / "pilot_designs.json"
    identities = validate_materialized_contract(
        contract_root,
        pilot_design_manifest=pilot_manifest,
        profile=FINAL_ADAPTIVE_PRODUCTION_PROFILE,
    )
    specification = read_json(contract_root / "contract_specification.json")
    validate_adaptive_contract_specification(specification)
    return {
        "specification": specification,
        "designs": read_jsonl(contract_root / "designs.jsonl"),
        "runs": read_jsonl(contract_root / "runs.jsonl"),
        **identities,
    }


def map_adaptive_contract_runs(contract: dict[str, Any]) -> tuple[FinalRunMapping, ...]:
    """Map every prepared run and assert the adaptive topology identity."""

    contract_hash = contract["contract_spec_hash"]
    designs = {record["design_id"]: record for record in contract["designs"]}
    mappings = tuple(
        map_adaptive_run(designs[run["design_id"]], run, contract_spec_hash=contract_hash)
        for run in contract["runs"]
    )
    if len(mappings) != 10_000 or tuple(mapping.run_id for mapping in mappings) != tuple(range(10_000)):
        raise ValueError("Adaptive run mapping is not the ordered 10K contract")
    if any(
        mapping.satellite_config.isl_policy != FINAL_ADAPTIVE_PRODUCTION_PROFILE.isl_policy
        or mapping.satellite_config.adjacent_search_k
        != FINAL_ADAPTIVE_PRODUCTION_PROFILE.adjacent_search_k
        or mapping.satellite_config.max_inter_plane_links_per_sat
        != FINAL_ADAPTIVE_PRODUCTION_PROFILE.max_inter_plane_links_per_sat
        for mapping in mappings
    ):
        raise ValueError("Adaptive run mapping contains topology drift")
    return mappings
