from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable

from satnet.experiments.final_dataset.materialize import (
    build_doe_evidence,
    build_golden_vectors,
    read_json,
    read_jsonl,
    validate_materialized_contract,
)
from satnet.experiments.final_dataset.schemas import (
    build_rf_schema,
    build_target_schema,
    build_tgnn_schema,
)
from satnet.experiments.final_dataset.specification import validate_contract_specification
from satnet.ground.canonical import canonical_float_string, canonical_hash
from satnet.ground.catalog import GroundStationCatalog, load_ground_station_catalog
from satnet.ground.coordinates import (
    GROUND_VISIBILITY_FRAME_CONTRACT_VERSION,
    GROUND_VISIBILITY_MODEL_VERSION,
    GROUND_WGS84_MODEL_VERSION,
)
from satnet.ground.failure_policy import (
    GROUND_FAILURE_MODEL_VERSION,
    GROUND_FAILURE_POLICY_VERSION,
    GROUND_FAILURE_SAMPLING_VERSION,
)
from satnet.ground.failure_realization import GROUND_FAILURE_REALIZATION_SCHEMA_VERSION
from satnet.ground.failure_service_aggregation import GROUND_FAILURE_SERVICE_RUN_SCHEMA_VERSION
from satnet.ground.failure_service_metrics import (
    GROUND_FAILURE_SERVICE_MODEL_VERSION,
    GROUND_FAILURE_SERVICE_STEP_SCHEMA_VERSION,
)
from satnet.ground.integrated_graph import (
    INTEGRATED_GRAPH_MODEL_VERSION,
    INTEGRATED_GRAPH_SCHEMA_VERSION,
)
from satnet.ground.persistence import GROUND_DESIGN_SCHEMA_VERSION
from satnet.ground.selection import GROUND_STATION_SELECTION_VERSION
from satnet.ground.service_aggregation import (
    GROUND_SERVICE_RUN_SCHEMA_VERSION,
    GROUND_SERVICE_STEP_SCHEMA_VERSION,
)
from satnet.ground.service_policy import GROUND_SERVICE_MODEL_VERSION, GROUND_SERVICE_POLICY_VERSION
from satnet.ground.visibility_persistence import GROUND_VISIBILITY_SCHEMA_VERSION
from satnet.network.hypatia_adapter import LinkBudgetEngine, PHYSICS_MODEL_VERSION
from satnet.simulation.tier1_rollout import DATASET_VERSION, SCHEMA_VERSION

from .constants import (
    CATALOG_FILE_SHA256,
    CATALOG_HASH,
    CONTRACT_BUNDLE_HASH,
    CONTRACT_SPEC_HASH,
    DESIGN_MANIFEST_HASH,
    FROZEN_ARTIFACTS,
    FROZEN_COMMIT,
    FROZEN_TAG,
    RF_SCHEMA_HASH,
    RUN_MANIFEST_HASH,
    SPLIT_MANIFEST_HASH,
    SUPPORTED_MODES,
    TARGET_SCHEMA_HASH,
    TGNN_SCHEMA_HASH,
    catalog_path,
    contract_root,
    repository_root,
)
from .io import atomic_write_json, read_canonical_json

_EXPECTED_UNTRACKED_ENTRIES = frozenset(
    {
        "data/",
        "docs/refactor_plans/2026-07-15_tier1_validity_remediation_atomic_gameplan.md",
        "docs/validation/tier1_defect_verification.md",
    }
)
_PROTECTED_SCIENCE_PATHS = (
    "src/satnet/ground",
    "src/satnet/network",
    "src/satnet/simulation/tier1_rollout.py",
    "src/satnet/models/gnn_dataset.py",
    "src/satnet/models/gnn_model.py",
    "src/satnet/models/risk_model.py",
    "src/satnet/utils/graph_cache.py",
)


def _git(*arguments: str, repo_root: Path | None = None) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root or repository_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Git command failed: {message}")
    return result.stdout


def _validate_self_hash(path: Path, field_name: str) -> None:
    value = read_canonical_json(path)
    persisted = value.pop(field_name, None)
    if persisted != canonical_hash(value):
        raise ValueError(f"{field_name} mismatch")


def validate_frozen_contract(*, compare_tag_blobs: bool = True) -> dict[str, Any]:
    root = contract_root()
    for name in FROZEN_ARTIFACTS:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen contract artifact: {name}")
        if compare_tag_blobs:
            tag_bytes = _git("show", f"{FROZEN_TAG}:artifacts/final_integrated_dataset_10k_contract/{name}")
            working_bytes = path.read_bytes()
            if len(tag_bytes) != len(working_bytes) or tag_bytes != working_bytes:
                raise ValueError(f"Frozen contract artifact differs from tag blob: {name}")
    specification = read_json(root / "contract_specification.json")
    validate_contract_specification(specification)
    if specification["contract_spec_hash"] != CONTRACT_SPEC_HASH:
        raise ValueError("Frozen contract specification hash mismatch")
    identities = validate_materialized_contract(root)
    expected = {
        "contract_spec_hash": CONTRACT_SPEC_HASH,
        "contract_bundle_hash": CONTRACT_BUNDLE_HASH,
        "design_manifest_hash": DESIGN_MANIFEST_HASH,
        "run_manifest_hash": RUN_MANIFEST_HASH,
        "split_manifest_hash": SPLIT_MANIFEST_HASH,
        "target_schema_hash": TARGET_SCHEMA_HASH,
        "rf_schema_hash": RF_SCHEMA_HASH,
        "tgnn_adapter_schema_hash": TGNN_SCHEMA_HASH,
        "catalog_hash": CATALOG_HASH,
    }
    if identities != expected:
        raise ValueError("Frozen contract semantic identities mismatch")
    schemas = (
        ("target_schema.json", build_target_schema(), "target_schema_hash", TARGET_SCHEMA_HASH),
        ("integrated_rf_export_schema.json", build_rf_schema(), "rf_schema_hash", RF_SCHEMA_HASH),
        ("integrated_tgnn_adapter_schema.json", build_tgnn_schema(), "tgnn_adapter_schema_hash", TGNN_SCHEMA_HASH),
    )
    for name, reconstructed, hash_field, expected_hash in schemas:
        if read_json(root / name) != reconstructed or reconstructed[hash_field] != expected_hash:
            raise ValueError(f"Frozen schema mismatch: {name}")
    _validate_self_hash(root / "golden_vectors.json", "golden_vectors_hash")
    _validate_self_hash(root / "doe_evidence.json", "doe_evidence_hash")
    _validate_self_hash(root / "manifest_inventory.json", "manifest_inventory_hash")
    inventory = read_json(root / "manifest_inventory.json")
    expected_inventory = {
        "catalog_hash": CATALOG_HASH,
        "contract_bundle_hash": CONTRACT_BUNDLE_HASH,
        "contract_spec_hash": CONTRACT_SPEC_HASH,
        "design_count": 2000,
        "design_manifest_hash": DESIGN_MANIFEST_HASH,
        "pilot_catalog_file_sha256": CATALOG_FILE_SHA256,
        "rf_schema_hash": RF_SCHEMA_HASH,
        "run_count": 10000,
        "run_manifest_hash": RUN_MANIFEST_HASH,
        "split_design_counts": {"test": 300, "train": 1400, "validation": 300},
        "split_manifest_hash": SPLIT_MANIFEST_HASH,
        "target_schema_hash": TARGET_SCHEMA_HASH,
        "tgnn_adapter_schema_hash": TGNN_SCHEMA_HASH,
    }
    for name, expected_value in expected_inventory.items():
        if inventory.get(name) != expected_value:
            raise ValueError(f"Frozen manifest inventory mismatch: {name}")
    designs = read_jsonl(root / "designs.jsonl")
    runs = read_jsonl(root / "runs.jsonl")
    if read_json(root / "golden_vectors.json") != build_golden_vectors():
        raise ValueError("Frozen golden-vector identities mismatch")
    if read_json(root / "doe_evidence.json") != build_doe_evidence(designs):
        raise ValueError("Frozen DOE evidence identities mismatch")
    if len(designs) != 2000 or len(runs) != 10000:
        raise ValueError("Frozen 10k manifest cardinality mismatch")
    if tuple(record["run_id"] for record in runs) != tuple(range(10000)):
        raise ValueError("Frozen run IDs are not contiguous integers 0 through 9999")
    return {"designs": designs, "runs": runs, "specification": specification, **identities}


def validate_catalog() -> GroundStationCatalog:
    path = catalog_path()
    if hashlib.sha256(path.read_bytes()).hexdigest() != CATALOG_FILE_SHA256:
        raise ValueError("Ground catalog raw SHA-256 mismatch")
    catalog = load_ground_station_catalog(path)
    if catalog.catalog_hash != CATALOG_HASH:
        raise ValueError("Ground catalog semantic identity mismatch")
    return catalog


def validate_science_dependencies(specification: dict[str, Any]) -> None:
    fixed = specification["fixed_profile"]
    if PHYSICS_MODEL_VERSION != fixed["physics_model_version"]:
        raise ValueError("Physics model version mismatch")
    link_budget = {
        name: canonical_float_string(value)
        for name, value in sorted(LinkBudgetEngine().to_config().items())
    }
    if link_budget != fixed["link_budget_config"]:
        raise ValueError("Link-budget configuration mismatch")
    versions = {
        "satellite_rollout_schema_version": SCHEMA_VERSION,
        "satellite_dataset_version": DATASET_VERSION,
        "ground_station_selection_version": GROUND_STATION_SELECTION_VERSION,
        "ground_design_schema_version": GROUND_DESIGN_SCHEMA_VERSION,
        "ground_visibility_model_version": GROUND_VISIBILITY_MODEL_VERSION,
        "ground_visibility_frame_contract_version": GROUND_VISIBILITY_FRAME_CONTRACT_VERSION,
        "ground_wgs84_model_version": GROUND_WGS84_MODEL_VERSION,
        "ground_visibility_schema_version": GROUND_VISIBILITY_SCHEMA_VERSION,
        "integrated_graph_model_version": INTEGRATED_GRAPH_MODEL_VERSION,
        "integrated_graph_schema_version": INTEGRATED_GRAPH_SCHEMA_VERSION,
        "ground_service_model_version": GROUND_SERVICE_MODEL_VERSION,
        "ground_service_policy_version": GROUND_SERVICE_POLICY_VERSION,
        "ground_service_step_schema_version": GROUND_SERVICE_STEP_SCHEMA_VERSION,
        "ground_service_run_schema_version": GROUND_SERVICE_RUN_SCHEMA_VERSION,
        "ground_failure_model_version": GROUND_FAILURE_MODEL_VERSION,
        "ground_failure_policy_version": GROUND_FAILURE_POLICY_VERSION,
        "ground_failure_sampling_version": GROUND_FAILURE_SAMPLING_VERSION,
        "ground_failure_realization_schema_version": GROUND_FAILURE_REALIZATION_SCHEMA_VERSION,
        "ground_failure_service_model_version": GROUND_FAILURE_SERVICE_MODEL_VERSION,
        "ground_failure_service_step_schema_version": GROUND_FAILURE_SERVICE_STEP_SCHEMA_VERSION,
        "ground_failure_service_run_schema_version": GROUND_FAILURE_SERVICE_RUN_SCHEMA_VERSION,
    }
    if versions != fixed["versions"]:
        raise ValueError("Authoritative G1-G5 version dependency mismatch")


def runtime_preflight(expected_tooling_sha: str) -> dict[str, Any]:
    if not isinstance(expected_tooling_sha, str) or len(expected_tooling_sha) != 40:
        raise ValueError("expected_tooling_sha must be a full Git SHA")
    head = _git("rev-parse", "HEAD").decode().strip()
    if head != expected_tooling_sha:
        raise ValueError("Current HEAD does not equal expected tooling SHA")
    tag = _git("rev-list", "-n", "1", FROZEN_TAG).decode().strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", tag, head],
        cwd=repository_root(),
        check=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", FROZEN_COMMIT, head],
        cwd=repository_root(),
        check=True,
    )
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Runtime preflight requires a clean tracked worktree")
    contract = validate_frozen_contract(compare_tag_blobs=True)
    validate_catalog()
    validate_science_dependencies(contract["specification"])
    return {"head": head, "frozen_commit": tag, "contract_spec_hash": CONTRACT_SPEC_HASH}


def protected_science_isolation_passes() -> bool:
    committed = _git("diff", "--name-only", FROZEN_COMMIT, "HEAD", "--", *_PROTECTED_SCIENCE_PATHS)
    status = _git("status", "--porcelain", "--", *_PROTECTED_SCIENCE_PATHS)
    return not committed and not status


def validate_initial_untracked_entries(entries: Iterable[str]) -> None:
    normalized: set[str] = set()
    for entry in entries:
        value = entry.replace("\\", "/")
        normalized.add("data/" if value == "data" or value.startswith("data/") else value)
    if normalized != _EXPECTED_UNTRACKED_ENTRIES:
        missing = sorted(_EXPECTED_UNTRACKED_ENTRIES - normalized)
        extra = sorted(normalized - _EXPECTED_UNTRACKED_ENTRIES)
        raise ValueError(f"Initial untracked entries mismatch; missing={missing}, extra={extra}")


def resolved_path(value: str | Path) -> Path:
    expanded = Path(value).expanduser().absolute()
    return Path(os.path.realpath(expanded))


def _normalized_path(value: str | Path) -> str:
    return os.path.normcase(str(resolved_path(value)))


def paths_intersect(left: Path, right: Path) -> bool:
    left_value = _normalized_path(left)
    right_value = _normalized_path(right)
    try:
        common = os.path.commonpath((left_value, right_value))
    except ValueError:
        return False
    return common == left_value or common == right_value


def repository_worktree_roots() -> tuple[Path, ...]:
    output = _git("worktree", "list", "--porcelain").decode("utf-8")
    roots = [
        resolved_path(line.removeprefix("worktree "))
        for line in output.splitlines()
        if line.startswith("worktree ")
    ]
    current = resolved_path(repository_root())
    if current not in roots:
        roots.append(current)
    return tuple(sorted(set(roots), key=lambda path: _normalized_path(path)))


def git_common_directory() -> Path:
    value = _git("rev-parse", "--path-format=absolute", "--git-common-dir").decode().strip()
    return resolved_path(value)


def repository_family_protected_paths() -> tuple[Path, ...]:
    relatives = (
        "artifacts/final_integrated_dataset_contract",
        "artifacts/final_integrated_dataset_10k_contract",
        "data",
        "docs/refactor_plans/2026-07-15_tier1_validity_remediation_atomic_gameplan.md",
        "docs/validation/tier1_defect_verification.md",
        "src/satnet/ground",
        "src/satnet/network",
        "src/satnet/simulation/tier1_rollout.py",
        "src/satnet/models/gnn_dataset.py",
        "src/satnet/models/gnn_model.py",
        "src/satnet/models/risk_model.py",
        "src/satnet/utils/graph_cache.py",
    )
    protected: set[Path] = {git_common_directory()}
    for worktree in repository_worktree_roots():
        protected.add(worktree)
        protected.update(resolved_path(worktree / relative) for relative in relatives)
    return tuple(sorted(protected, key=lambda path: _normalized_path(path)))


def validate_output_root(root: str | Path, *, other_roots: Iterable[str | Path] = ()) -> Path:
    resolved = resolved_path(root)
    for prohibited in repository_family_protected_paths():
        if paths_intersect(resolved, prohibited):
            raise ValueError(f"Execution root intersects protected repository-family path: {prohibited}")
    for other in other_roots:
        if paths_intersect(resolved, resolved_path(other)):
            raise ValueError("Execution roots intersect")
    return resolved


def mode_marker(
    mode: str, *, contract_spec_hash: str = CONTRACT_SPEC_HASH
) -> dict[str, str]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported execution mode: {mode}")
    return {
        "contract_spec_hash": contract_spec_hash,
        "execution_mode": mode,
        "mode_marker_schema_version": "1",
    }


def ensure_mode_root(
    root: str | Path,
    mode: str,
    *,
    create: bool,
    contract_spec_hash: str = CONTRACT_SPEC_HASH,
) -> Path:
    resolved = validate_output_root(root)
    marker_path = resolved / "execution_mode.json"
    expected = mode_marker(mode, contract_spec_hash=contract_spec_hash)
    if not resolved.exists():
        if not create:
            raise FileNotFoundError(f"Execution root does not exist: {resolved}")
        resolved.mkdir(parents=True)
        atomic_write_json(marker_path, expected)
        return resolved
    entries = list(resolved.iterdir())
    if not entries:
        if not create:
            raise ValueError("Execution root is empty and unmarked")
        atomic_write_json(marker_path, expected)
        return resolved
    if not marker_path.is_file() or read_canonical_json(marker_path) != expected:
        raise ValueError("Execution root mode marker mismatch")
    return resolved
