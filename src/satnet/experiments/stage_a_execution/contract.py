from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

from satnet.experiments.stage_a_contract import proposal
from satnet.experiments.stage_a_contract import freeze

from .common import sha256_file

FROZEN_CONTRACT_HASH = "e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a"
FROZEN_SPECIFICATION_HASH = "c1822db61182e6ff6436c767ac39a35065ff84741119bd7306b261dc2a1f7373"
FROZEN_DECLARATION_HASH = "e0987858e1eca4d04e7008a468de232ef26eaa867f9d2a750c83fc8c89717bd4"
FROZEN_README_HASH = "d4d9d8b796b194ef6bebe7ad1a4520cc38cee4f3fb40889fc922cf3705928b0c"
FROZEN_COMMIT = "301d8a224daa070b15ecc6447f503d42d5d1e70a"
FROZEN_TAG = "stage-a-discovery-contract-v1"
FROZEN_AUDIT_COMMIT = "c350d693bf47d007a0cb2a8c3b5cfb2259d2da48"
FROZEN_AUDIT_INVENTORY = "2fe074c018f69da95f29ffaca06ae45c50e7098988d7b181fe9756452db41df9"
CONTRACT_RELATIVE_ROOT = Path("artifacts/stage_a_discovery_contract_v1")

_INTEGER_DESIGN_FIELDS = {
    "design_index", "num_planes", "sats_per_plane", "configured_satellite_count", "phasing_factor",
    "civilian_count", "government_count", "military_count", "total_ground_station_count",
    "duration_minutes", "step_seconds", "adjacent_search_k", "max_inter_plane_links_per_sat",
    "design_construction_seed", "ground_selection_seed",
}
_INTEGER_RUN_FIELDS = {"global_run_id", "design_index", "realization_index"}
_INTEGER_SEED_FIELDS = {
    "global_run_id", "design_construction_seed", "ground_selection_seed", "satellite_failure_seed", "ground_failure_seed"
}


@dataclass(frozen=True)
class FrozenStageAContract:
    repo_root: Path
    contract_root: Path
    specification: dict[str, Any]
    declaration: dict[str, Any]
    designs: tuple[dict[str, Any], ...]
    runs: tuple[dict[str, Any], ...]
    seeds: tuple[dict[str, Any], ...]
    partitions: dict[str, Any]
    output_roots: dict[str, str]
    contract_identity: str = FROZEN_CONTRACT_HASH
    frozen_tag: str = FROZEN_TAG
    frozen_commit: str = FROZEN_COMMIT

    @property
    def contract_hash(self) -> str:
        return self.contract_identity


def _git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=repo_root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "Git identity verification failed")
    return result.stdout.strip()


def _csv(path: Path, integer_fields: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            row: dict[str, Any] = dict(source)
            for field in integer_fields:
                row[field] = int(row[field])
            for field in ("sealed", "simulation_authorized"):
                if field in row:
                    if row[field] not in {"True", "False"}:
                        raise ValueError(f"Invalid Boolean encoding: {field}")
                    row[field] = row[field] == "True"
            rows.append(row)
    return rows


def verify_tag(repo_root: Path) -> None:
    if _git(repo_root, "rev-list", "-n", "1", FROZEN_TAG) != FROZEN_COMMIT:
        raise ValueError("Frozen Stage A tag target mismatch")
    if _git(repo_root, "cat-file", "-t", FROZEN_TAG) != "tag":
        raise ValueError("Frozen Stage A tag is not annotated")


def frozen_contract_output_root_states(operation: str | None) -> dict[str, bool]:
    states = {name: False for name in freeze.OUTPUT_ROOTS}
    required_sources = {
        None: (),
        "PLAN": (),
        "GENERATE": (),
        "REPLAY": ("production_generation",),
        "ACCEPT": ("production_generation", "production_replay"),
    }
    if operation not in required_sources:
        raise ValueError(f"Unsupported Stage A operation: {operation}")
    for name in required_sources[operation]:
        states[name] = True
    return states


def load_frozen_contract(
    repo_root: Path,
    contract_root: Path | None = None,
    operation: str | None = None,
) -> FrozenStageAContract:
    root = (contract_root or repo_root / CONTRACT_RELATIVE_ROOT).resolve(strict=True)
    verify_tag(repo_root)
    result = freeze.validate_frozen_contract(
        repo_root=repo_root,
        contract_root=root,
        expected_proposal_commit=freeze.APPROVED_PROPOSAL_COMMIT,
        expected_proposal_inventory=freeze.APPROVED_PROPOSAL_INVENTORY,
        expected_audit_commit=freeze.AUDIT_COMMIT,
        expected_audit_inventory=freeze.AUDIT_INVENTORY,
        expected_output_root_states=frozen_contract_output_root_states(operation),
    )
    if result["contract_hash"] != FROZEN_CONTRACT_HASH:
        raise ValueError("Frozen contract hash mismatch")
    expected_hashes = {
        freeze.INVENTORY_NAME: FROZEN_CONTRACT_HASH,
        freeze.SPECIFICATION_NAME: FROZEN_SPECIFICATION_HASH,
        freeze.DECLARATION_NAME: FROZEN_DECLARATION_HASH,
        freeze.README_NAME: FROZEN_README_HASH,
    }
    for name, expected in expected_hashes.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Frozen contract artifact hash mismatch: {name}")
    audit_inventory = repo_root / "artifacts/stage_a_discovery_contract_frozen_audit/audit_inventory.json"
    if sha256_file(audit_inventory) != FROZEN_AUDIT_INVENTORY:
        raise ValueError("Frozen-contract audit inventory mismatch")
    source = root / "source_bundle"
    designs = _csv(source / "stage_a_design_manifest.csv", _INTEGER_DESIGN_FIELDS)
    runs = _csv(source / "stage_a_run_manifest.csv", _INTEGER_RUN_FIELDS)
    seeds = _csv(source / "stage_a_seed_manifest.csv", _INTEGER_SEED_FIELDS)
    proposal.validate_design_rows(designs)
    proposal.validate_run_rows(runs, designs)
    proposal.validate_seed_rows(seeds)
    if [row["run_key"] for row in runs] != [row["run_key"] for row in seeds]:
        raise ValueError("Run and seed manifest ordering differs")
    for run, seed in zip(runs, seeds, strict=True):
        for field in ("global_run_id", "run_key", "design_id", "realization_id"):
            if run[field] != seed[field]:
                raise ValueError(f"Run/seed identity mismatch: {field}")
    specification = freeze.read_json(root / freeze.SPECIFICATION_NAME)
    declaration = freeze.read_json(root / freeze.DECLARATION_NAME)
    if any(declaration[field] is not False for field in ("simulation_authorized", "production_authorized", "execution_authorized")):
        raise ValueError("Frozen declaration unexpectedly authorizes execution")
    partition_manifest = freeze.read_json(source / "stage_a_partition_manifest.json")
    roots_manifest = freeze.read_json(source / "stage_a_output_root_manifest.json")
    expected = {
        "development": (20, 100, False),
        "validation": (5, 25, False),
        "sealed_holdout": (5, 25, True),
    }
    for name, (design_count, run_count, sealed) in expected.items():
        value = partition_manifest["partitions"][name]
        if (value["design_count"], value["run_count"], value["sealed"]) != (design_count, run_count, sealed):
            raise ValueError(f"Frozen partition mismatch: {name}")
    return FrozenStageAContract(
        repo_root=repo_root.resolve(), contract_root=root, specification=specification,
        declaration=declaration, designs=tuple(designs), runs=tuple(runs), seeds=tuple(seeds),
        partitions=partition_manifest["partitions"], output_roots=roots_manifest["proposed_resolved_paths"],
    )
