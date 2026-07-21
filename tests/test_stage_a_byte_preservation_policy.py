from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).parents[1]
AUDIT_COMMIT = "a1514a654fe76518db16001b98e899f773eb9d1e"
AUDIT_INVENTORY_SHA256 = "16ce1a1b138567a144cbf9b1d715c74b30f05e1d8262841d9580240080339d10"
AUDIT_ROOT = "artifacts/stage_a_discovery_contract_freeze_audit"
AUDIT_INVENTORY_PATH = f"{AUDIT_ROOT}/audit_inventory.json"


def _git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], cwd=repo, text=text)


def _attribute(repo: Path, path: str) -> str:
    output = str(_git(repo, "check-attr", "text", "--", path)).strip()
    return output.rsplit(": ", 1)[1]


def _blob(repo: Path, commit: str, path: str) -> bytes:
    return bytes(_git(repo, "cat-file", "blob", f"{commit}:{path}", text=False))


def is_repository_wide_negative_text_rule(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    tokens = stripped.split()
    return len(tokens) >= 2 and tokens[0] == "*" and "-text" in tokens[1:]


@pytest.mark.parametrize(
    "path",
    [
        AUDIT_INVENTORY_PATH,
        "artifacts/stage_a_discovery_contract_v1/stage_a_frozen_contract_inventory.json",
        "artifacts/stage_a_discovery_contract_proposal/stage_a_proposal_inventory.json",
    ],
)
def test_stage_a_identity_artifacts_are_checkout_byte_preserved(path: str) -> None:
    assert _attribute(ROOT, path) == "unset"


@pytest.mark.parametrize(
    "line",
    [
        "artifacts/stage_a_discovery_contract_freeze_audit/** -text",
        "artifacts/stage_a_discovery_contract_v1/** -text",
        "artifacts/stage_a_discovery_contract_proposal/** -text",
        "docs/example/** -text",
        "",
        "   ",
        "# * -text",
        "  # repository policy",
    ],
)
def test_repository_wide_negative_text_rule_allows_narrow_or_ignored_lines(
    line: str,
) -> None:
    assert is_repository_wide_negative_text_rule(line) is False


@pytest.mark.parametrize("line", ["* -text", "*    -text", "* text -text"])
def test_repository_wide_negative_text_rule_rejects_exact_global_pattern(
    line: str,
) -> None:
    assert is_repository_wide_negative_text_rule(line) is True


def test_unrelated_text_is_not_broadly_forced_to_binary_safe_checkout() -> None:
    assert _attribute(ROOT, "README.md") == "unspecified"
    lines = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert not any(is_repository_wide_negative_text_rule(line) for line in lines)
    assert not any(line.split(maxsplit=1)[0] == "*.json" for line in lines if line.strip())
    assert not any(line.split(maxsplit=1)[0] == "*.csv" for line in lines if line.strip())


def test_approved_audit_inventory_git_blob_has_binding_identity() -> None:
    payload = _blob(ROOT, AUDIT_COMMIT, AUDIT_INVENTORY_PATH)
    assert len(payload) == 3691
    assert hashlib.sha256(payload).hexdigest() == AUDIT_INVENTORY_SHA256
    assert payload.count(b"\n") == 107
    assert payload.count(b"\r\n") == 0


def test_every_approved_audit_git_blob_matches_inventory() -> None:
    inventory_payload = _blob(ROOT, AUDIT_COMMIT, AUDIT_INVENTORY_PATH)
    inventory = json.loads(inventory_payload)
    records = {record["relative_path"]: record for record in inventory["artifacts"]}
    tracked = str(
        _git(ROOT, "ls-tree", "-r", "--name-only", AUDIT_COMMIT, "--", AUDIT_ROOT)
    ).splitlines()
    expected = {f"{AUDIT_ROOT}/{name}" for name in records} | {AUDIT_INVENTORY_PATH}
    assert set(tracked) == expected
    for name, record in records.items():
        payload = _blob(ROOT, AUDIT_COMMIT, f"{AUDIT_ROOT}/{name}")
        assert len(payload) == record["byte_length"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]


def test_fresh_autocrlf_checkout_retains_approved_audit_inventory(
    tmp_path: Path,
) -> None:
    clone = tmp_path / "fresh-autocrlf-checkout"
    head = str(_git(ROOT, "rev-parse", "HEAD")).strip()
    subprocess.run(
        ["git", "clone", "--local", "--no-checkout", str(ROOT), str(clone)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "core.autocrlf", "true"],
        cwd=clone,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", head],
        cwd=clone,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = clone.joinpath(AUDIT_INVENTORY_PATH).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == AUDIT_INVENTORY_SHA256
    assert len(payload) == 3691
    assert payload.count(b"\n") == 107
    assert payload.count(b"\r\n") == 0
    assert _attribute(clone, AUDIT_INVENTORY_PATH) == "unset"
    assert (
        _attribute(
            clone,
            "artifacts/stage_a_discovery_contract_v1/stage_a_frozen_contract_inventory.json",
        )
        == "unset"
    )
    assert str(_git(clone, "status", "--short")).strip() == ""
