from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import tempfile
from typing import Iterable

FROZEN_EVIDENCE_PATHS = (
    Path(r"C:\Users\johns\satnet-final-production-20260720"),
    Path(r"C:\Users\johns\satnet-final-production-replay-20260720"),
    Path(r"C:\Users\johns\satnet-final-production-v1-freeze-20260721"),
    Path(r"C:\Users\johns\satnet-final-production-v1-freeze-20260721.zip"),
)


def resolved(path: Path) -> Path:
    return Path(os.path.realpath(path.expanduser().absolute()))


def overlaps(first: Path, second: Path) -> bool:
    left = os.path.normcase(str(resolved(first)))
    right = os.path.normcase(str(resolved(second)))
    try:
        common = os.path.commonpath((left, right))
    except ValueError:
        return False
    return common in {left, right}


def repository_worktrees(repo_root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"], cwd=repo_root,
        capture_output=True, text=True, check=True,
    )
    return tuple(
        resolved(Path(line.removeprefix("worktree ")))
        for line in result.stdout.splitlines() if line.startswith("worktree ")
    )


def validate_output_roots(
    *, repo_root: Path, generation_root: Path, replay_root: Path, acceptance_root: Path,
    require_absent: bool,
) -> dict[str, str]:
    roots = {
        "generation": resolved(generation_root),
        "replay": resolved(replay_root),
        "acceptance": resolved(acceptance_root),
    }
    values = tuple(roots.values())
    protected = (*repository_worktrees(repo_root), *map(resolved, FROZEN_EVIDENCE_PATHS))
    for name, root in roots.items():
        if root == root.anchor or root.parent == root:
            raise ValueError(f"Unsafe execution root: {name}")
        if require_absent and root.exists():
            raise FileExistsError(f"Authorized output root already exists: {name}")
        for other_name, other in roots.items():
            if name != other_name and overlaps(root, other):
                raise ValueError("Execution roots overlap")
        if any(overlaps(root, item) for item in protected):
            raise ValueError(f"Execution root overlaps protected path: {name}")
        ancestor = root.parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if ancestor.is_symlink() or resolved(ancestor) != ancestor.absolute():
            raise ValueError(f"Execution root parent escapes through link or junction: {name}")
    return {name: str(path) for name, path in roots.items()}


def probe_parent(path: Path) -> None:
    parent = resolved(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    probe = Path(tempfile.mkdtemp(dir=parent, prefix=".satnet_stage_a_preflight."))
    try:
        marker = probe / "probe"
        marker.write_bytes(b"probe")
        if marker.read_bytes() != b"probe":
            raise OSError("Parent write probe mismatch")
    finally:
        shutil.rmtree(probe)


def available_bytes(path: Path) -> int:
    ancestor = resolved(path).parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    return shutil.disk_usage(ancestor).free


def validate_relative_artifact_path(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Artifact path must be nonempty relative POSIX")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.drive or windows.root or value.startswith("/") or any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError("Artifact path traversal, drive, or absolute path denied")
    return Path(*posix.parts)


def validate_no_links(root: Path, files: Iterable[Path]) -> None:
    base = resolved(root)
    for path in files:
        if path.is_symlink() or not resolved(path).is_relative_to(base):
            raise ValueError("Artifact link or junction escape detected")
