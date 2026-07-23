from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from satnet.experiments.stage_a_execution.authorization import Authorization, authorization_digest
from satnet.experiments.stage_a_execution.contract import FrozenStageAContract
from satnet.experiments.stage_a_execution.ledger import read_ledger
from satnet.experiments.stage_a_execution.resume import LedgerProvenance, ledger_provenance_from_mapping

STABLE_EXECUTABLE_COMMIT = "1" * 40
EXECUTABLE_INVENTORY = "2" * 64
TOOLING_PROPOSAL = "3" * 64
ARTIFACT_CONTRACT = "4" * 64
TOOLING_COMMIT = STABLE_EXECUTABLE_COMMIT
TOOLING_INVENTORY = EXECUTABLE_INVENTORY
LEDGER_PROVENANCE_FIELDS = (
    "contract_hash",
    "plan_hash",
    "campaign_manifest_hash",
    "authorization_hash",
    "stable_executable_commit",
    "executable_inventory_hash",
    "tooling_proposal_hash",
    "artifact_contract_hash",
    "operation",
    "partition",
    "expected_run_count",
    "output_root_identity",
    "source_generation_ledger_relative_path",
    "source_generation_ledger_byte_length",
    "source_generation_ledger_sha256",
    "source_replay_ledger_relative_path",
    "source_replay_ledger_byte_length",
    "source_replay_ledger_sha256",
)


def source_provenance_from_ledger(path: Path) -> LedgerProvenance:
    ledger = read_ledger(path)
    return ledger_provenance_from_mapping({field: ledger[field] for field in LEDGER_PROVENANCE_FIELDS})


@pytest.fixture
def synthetic_contract(tmp_path: Path) -> FrozenStageAContract:
    designs = (
        {"design_id": "SYN-D000", "design_index": 0, "region": "synthetic", "partition": "development", "sealed": False, "design_record_hash": "3" * 64},
        {"design_id": "SYN-D001", "design_index": 1, "region": "synthetic", "partition": "validation", "sealed": False, "design_record_hash": "4" * 64},
        {"design_id": "SYN-D002", "design_index": 2, "region": "synthetic", "partition": "sealed_holdout", "sealed": True, "design_record_hash": "5" * 64},
    )
    runs = (
        {"global_run_id": 1, "run_key": "SYN-D000-R00", "design_id": "SYN-D000", "design_index": 0, "realization_id": "R00", "realization_index": 0, "region": "synthetic", "partition": "development", "sealed": False, "design_record_hash": "3" * 64, "run_record_hash": "6" * 64},
        {"global_run_id": 2, "run_key": "SYN-D000-R01", "design_id": "SYN-D000", "design_index": 0, "realization_id": "R01", "realization_index": 1, "region": "synthetic", "partition": "development", "sealed": False, "design_record_hash": "3" * 64, "run_record_hash": "7" * 64},
        {"global_run_id": 3, "run_key": "SYN-D001-R00", "design_id": "SYN-D001", "design_index": 1, "realization_id": "R00", "realization_index": 0, "region": "synthetic", "partition": "validation", "sealed": False, "design_record_hash": "4" * 64, "run_record_hash": "8" * 64},
        {"global_run_id": 4, "run_key": "SYN-D002-R00", "design_id": "SYN-D002", "design_index": 2, "realization_id": "R00", "realization_index": 0, "region": "synthetic", "partition": "sealed_holdout", "sealed": True, "design_record_hash": "5" * 64, "run_record_hash": "9" * 64},
    )
    seeds = tuple(
        {
            "global_run_id": row["global_run_id"], "run_key": row["run_key"], "design_id": row["design_id"],
            "realization_id": row["realization_id"], "design_construction_seed": 1_000 + row["design_index"],
            "ground_selection_seed": row["global_run_id"] * 10 + 1,
            "satellite_failure_seed": row["global_run_id"] * 10 + 2, "ground_failure_seed": row["global_run_id"] * 10 + 3,
        }
        for row in runs
    )
    return FrozenStageAContract(
        repo_root=tmp_path / "repo", contract_root=tmp_path / "contract",
        specification={}, declaration={}, designs=designs, runs=runs, seeds=seeds,
        partitions={
            "development": {"global_run_ids": [1, 2], "design_count": 1, "run_count": 2, "sealed": False},
            "validation": {"global_run_ids": [3], "design_count": 1, "run_count": 1, "sealed": False},
            "sealed_holdout": {"global_run_ids": [4], "design_count": 1, "run_count": 1, "sealed": True},
        },
        output_roots={}, contract_identity="a" * 64, frozen_tag="synthetic-contract-v1", frozen_commit="b" * 40,
    )


def make_authorization(
    contract: FrozenStageAContract, *, operation: str, partition: str, run_ids: list[int],
    generation_root: Path, replay_root: Path, acceptance_root: Path,
    source_generation_ledger: tuple[str, int, str] | None = None,
    source_replay_ledger: tuple[str, int, str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Authorization:
    document: dict[str, Any] = {
        "schema_identifier": "satnet.stage_a.execution_authorization.v1",
        "authorization_version": "1",
        "authorization_id": f"SYNTHETIC-{operation}-{partition}",
        "authorization_status": "AUTHORIZED",
        "authorized_contract_hash": contract.contract_hash,
        "authorized_contract_tag": contract.frozen_tag,
        "authorized_frozen_commit": contract.frozen_commit,
        "authorized_stable_executable_commit": STABLE_EXECUTABLE_COMMIT,
        "authorized_executable_inventory_hash": EXECUTABLE_INVENTORY,
        "authorized_tooling_proposal_hash": TOOLING_PROPOSAL,
        "authorized_artifact_contract_hash": ARTIFACT_CONTRACT,
        "authorized_partition": partition,
        "authorized_run_ids": run_ids,
        "authorized_run_count": len(run_ids),
        "authorized_operation": operation,
        "authorized_generation_root": str(generation_root.resolve()),
        "authorized_replay_root": str(replay_root.resolve()),
        "authorized_acceptance_root": str(acceptance_root.resolve()),
        "source_generation_ledger_relative_path": None if source_generation_ledger is None else source_generation_ledger[0],
        "source_generation_ledger_byte_length": None if source_generation_ledger is None else source_generation_ledger[1],
        "source_generation_ledger_sha256": None if source_generation_ledger is None else source_generation_ledger[2],
        "source_replay_ledger_relative_path": None if source_replay_ledger is None else source_replay_ledger[0],
        "source_replay_ledger_byte_length": None if source_replay_ledger is None else source_replay_ledger[1],
        "source_replay_ledger_sha256": None if source_replay_ledger is None else source_replay_ledger[2],
        "authorization_date": "2099-01-01",
        "authorizing_decision_reference": "SYNTHETIC-TEST-ONLY",
        "independently_approved": True,
    }
    if overrides:
        document.update(overrides)
    document["authorization_sha256"] = authorization_digest(document)
    return Authorization(document=document, sha256=authorization_digest(document))
