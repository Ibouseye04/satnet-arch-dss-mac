from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from .common import ensure_hex, sha256_file

EXECUTABLE_INVENTORY_SCHEMA = "satnet.stage_a.executable_source_inventory.v2"
STABLE_IDENTITY_SCHEMA = "satnet.stage_a.stable_executable_identity.v2"
EXECUTABLE_INVENTORY_RELATIVE = Path("artifacts/stage_a_execution_tooling_v1_proposal/stage_a_executable_source_inventory.json")
STABLE_IDENTITY_RELATIVE = Path("artifacts/stage_a_execution_tooling_v1_proposal/stage_a_stable_executable_identity.json")
TOOLING_INVENTORY_RELATIVE = Path("artifacts/stage_a_execution_tooling_v1_proposal/stage_a_execution_tooling_inventory.json")
EXECUTION_PACKAGE_RELATIVE = Path("src/satnet/experiments/stage_a_execution")


def _git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=repo_root, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"Git command failed: {' '.join(arguments)}")
    return result.stdout


def executable_source_paths(repo_root: Path) -> tuple[str, ...]:
    fixed_files = (
        "src/satnet/experiments/final_class_support_audit/audit.py",
        "src/satnet/simulation/tier1_rollout.py",
        "src/satnet/models/gnn_dataset.py",
        "src/satnet/models/gnn_model.py",
        "src/satnet/models/risk_model.py",
        "src/satnet/utils/graph_cache.py",
    )
    roots = (
        "src/satnet/experiments/stage_a_execution",
        "src/satnet/experiments/stage_a_contract",
        "src/satnet/experiments/final_generation",
        "src/satnet/ground",
        "src/satnet/network",
    )
    paths = set(fixed_files)
    for relative in roots:
        paths.update(path.relative_to(repo_root).as_posix() for path in (repo_root / relative).rglob("*.py"))
    missing = [relative for relative in paths if not (repo_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Executable source missing: {missing}")
    return tuple(sorted(paths))


def source_role(relative_path: str) -> str:
    if "/stage_a_execution/" in relative_path:
        return "stage_a_execution_control"
    if "/stage_a_contract/" in relative_path:
        return "frozen_contract_validation"
    if "/final_generation/" in relative_path:
        return "validated_generation_pipeline"
    if "/final_class_support_audit/" in relative_path:
        return "frozen_production_evidence_verification"
    if "/ground/" in relative_path or "/network/" in relative_path or "/simulation/" in relative_path:
        return "protected_simulation_science"
    return "protected_scientific_adapter"


def make_executable_inventory(repo_root: Path, stable_commit: str) -> dict[str, Any]:
    ensure_hex(stable_commit, length=40, field="stable_executable_commit")
    records = [
        {
            "byte_length": (repo_root / relative).stat().st_size,
            "relative_path": relative,
            "sha256": sha256_file(repo_root / relative),
            "source_role": source_role(relative),
        }
        for relative in executable_source_paths(repo_root)
    ]
    return {
        "schema_identifier": EXECUTABLE_INVENTORY_SCHEMA,
        "stable_executable_commit": stable_commit,
        "artifact_count": len(records),
        "artifacts": records,
        "ordering": "relative_path ordinal lexical ascending",
        "path_policy": "repository-relative forward-slash paths",
        "hash_algorithm": "SHA-256",
        "self_reference_policy": "inventory contains executable source only and excludes its own bytes",
    }


def read_executable_inventory(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or value.get("schema_identifier") != EXECUTABLE_INVENTORY_SCHEMA:
        raise ValueError("Executable inventory schema mismatch")
    records = value.get("artifacts")
    if not isinstance(records, list) or len(records) != value.get("artifact_count"):
        raise ValueError("Executable inventory count mismatch")
    paths = [record.get("relative_path") for record in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Executable inventory ordering or uniqueness mismatch")
    for record in records:
        if set(record) != {"byte_length", "relative_path", "sha256", "source_role"}:
            raise ValueError("Executable inventory record fields mismatch")
        relative = record["relative_path"]
        if not isinstance(relative, str) or "\\" in relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("Executable inventory path is not canonical")
        ensure_hex(record["sha256"], length=64, field="executable_sha256")
    ensure_hex(value.get("stable_executable_commit"), length=40, field="stable_executable_commit")
    return value


def _expected_imports(records: Iterable[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    imports: list[tuple[str, str]] = []
    for record in records:
        relative = str(record["relative_path"])
        if relative.startswith("src/") and relative.endswith(".py") and "/__init__.py" not in relative:
            imports.append((relative[4:-3].replace("/", "."), relative))
    return tuple(imports)


def verify_executable_identity(
    repo_root: Path,
    inventory_path: Path | None = None,
    stable_identity_path: Path | None = None,
) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    inventory_file = inventory_path or root / EXECUTABLE_INVENTORY_RELATIVE
    stable_file = stable_identity_path or root / STABLE_IDENTITY_RELATIVE
    inventory = read_executable_inventory(inventory_file)
    stable = json.loads(stable_file.read_bytes())
    if not isinstance(stable, dict) or stable.get("schema_identifier") != STABLE_IDENTITY_SCHEMA:
        raise ValueError("Stable executable identity schema mismatch")
    stable_commit = ensure_hex(stable.get("stable_executable_commit"), length=40, field="stable_executable_commit")
    inventory_sha256 = sha256_file(inventory_file)
    ensure_hex(stable.get("executable_inventory_sha256"), length=64, field="executable_inventory_sha256")
    if stable["executable_inventory_sha256"] != inventory_sha256:
        raise ValueError("Stable executable identity does not bind the executable inventory bytes")
    if stable.get("executable_file_count") != inventory["artifact_count"]:
        raise ValueError("Stable executable identity source count mismatch")
    if inventory["stable_executable_commit"] != stable_commit:
        raise ValueError("Stable executable commit differs from executable inventory")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError("Repository is not clean for executable preflight")
    if _git(root, "diff") or _git(root, "diff", "--cached"):
        raise ValueError("Repository tracked state is dirty")
    _git(root, "merge-base", "--is-ancestor", stable_commit, "HEAD")
    expected_paths = tuple(record["relative_path"] for record in inventory["artifacts"])
    actual_paths = executable_source_paths(root)
    if expected_paths != actual_paths:
        missing = sorted(set(expected_paths) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected_paths))
        raise ValueError(f"Executable source membership mismatch: missing={missing}, extra={extra}")
    post_stable = _git(root, "diff", "--name-only", f"{stable_commit}..HEAD", "--", *expected_paths).splitlines()
    if post_stable:
        raise ValueError(f"Executable source changed after stable commit: {post_stable}")
    verified = 0
    for record in inventory["artifacts"]:
        path = root / record["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record["byte_length"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Executable byte identity mismatch: {record['relative_path']}")
        verified += 1
    for module_name, relative in _expected_imports(inventory["artifacts"]):
        specification = importlib.util.find_spec(module_name)
        if specification is None or specification.origin is None:
            raise RuntimeError(f"Executable import is unavailable: {module_name}")
        if Path(specification.origin).resolve(strict=True) != (root / relative).resolve(strict=True):
            raise RuntimeError(f"Executable import resolves outside expected repository file: {module_name}")
    return {
        "stable_executable_commit": stable_commit,
        "executable_inventory_sha256": inventory_sha256,
        "executable_file_count": verified,
        "repository_clean": True,
        "stable_commit_is_ancestor": True,
        "post_stable_executable_diff": [],
        "import_resolution": "EXPECTED_REPOSITORY_FILES",
        "verification": "PASSED",
    }


def tooling_identity(repo_root: Path) -> tuple[str, str, str]:
    result = verify_executable_identity(repo_root)
    proposal_hash = sha256_file(repo_root / TOOLING_INVENTORY_RELATIVE)
    return result["stable_executable_commit"], result["executable_inventory_sha256"], proposal_hash
