from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
from typing import Any

from .authorization import Authorization, validate_authorization
from .common import sha256_file
from .contract import FrozenStageAContract, load_frozen_contract
from .paths import available_bytes, probe_parent, validate_output_roots
from .plan import validate_plan

MINIMUM_FREE_BYTES = 1_000_000_000


def tooling_identity(repo_root: Path, inventory_path: Path) -> tuple[str, str]:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True)
    return result.stdout.strip(), sha256_file(inventory_path)


def run_preflight(
    *, repo_root: Path, contract: FrozenStageAContract, plan: dict[str, Any],
    authorization: Authorization, tooling_commit: str, tooling_inventory_hash: str,
    generation_root: Path, replay_root: Path, acceptance_root: Path,
    check_write_probe: bool = True, minimum_free_bytes: int = MINIMUM_FREE_BYTES,
) -> dict[str, Any]:
    reloaded = load_frozen_contract(repo_root, contract.contract_root)
    validate_plan(plan)
    validate_authorization(
        authorization, contract=reloaded, tooling_commit=tooling_commit,
        tooling_inventory_hash=tooling_inventory_hash, operation=plan["operation"],
        partition=plan["partition"], run_ids=[row["global_run_id"] for row in plan["runs"]],
        generation_root=generation_root, replay_root=replay_root, acceptance_root=acceptance_root,
    )
    roots = validate_output_roots(
        repo_root=repo_root, generation_root=generation_root, replay_root=replay_root,
        acceptance_root=acceptance_root, require_absent=plan["operation"] in {"GENERATE", "REPLAY", "ACCEPT"},
    )
    if check_write_probe:
        for root in (generation_root, replay_root, acceptance_root):
            probe_parent(root)
    free = {name: available_bytes(Path(path)) for name, path in roots.items()}
    if any(value < minimum_free_bytes for value in free.values()):
        raise OSError("Insufficient disk space for authorized execution")
    required_modules = (
        "satnet.experiments.final_generation.orchestrator",
        "satnet.simulation.tier1_rollout",
        "satnet.ground.failure_service_persistence",
    )
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Required runtime modules unavailable: {missing}")
    lock_path = Path(roots[{"GENERATE": "generation", "REPLAY": "replay", "ACCEPT": "acceptance", "PLAN": "generation"}[plan["operation"]]])
    if lock_path.with_name(lock_path.name + ".lock").exists():
        raise RuntimeError("Conflicting execution lock exists")
    return {
        "authorization": "VALID",
        "contract_hash": contract.contract_hash,
        "dependencies": "AVAILABLE",
        "disk_free_bytes": free,
        "output_roots": roots,
        "plan_hash": plan["plan_hash"],
        "preflight": "PASSED",
        "simulation_executed": False,
    }
