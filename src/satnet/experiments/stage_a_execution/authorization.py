from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .common import canonical_json_bytes, ensure_hex, read_json_object, sha256_bytes
from .contract import FrozenStageAContract

AUTHORIZATION_SCHEMA = "satnet.stage_a.execution_authorization.v1"
OPERATIONS = frozenset({"PLAN", "GENERATE", "REPLAY", "ACCEPT"})
EXECUTABLE_PARTITIONS = frozenset({"development", "validation"})


@dataclass(frozen=True)
class Authorization:
    document: dict[str, Any]
    sha256: str

    @property
    def operation(self) -> str:
        return str(self.document["authorized_operation"])

    @property
    def partition(self) -> str:
        return str(self.document["authorized_partition"])

    @property
    def run_ids(self) -> tuple[int, ...]:
        return tuple(self.document["authorized_run_ids"])


def authorization_digest(document: dict[str, Any]) -> str:
    payload = dict(document)
    claimed = payload.pop("authorization_sha256", None)
    if claimed is not None and not isinstance(claimed, str):
        raise ValueError("authorization_sha256 must be a string")
    return sha256_bytes(canonical_json_bytes(payload))


def load_authorization(path: Path) -> Authorization:
    document = read_json_object(path)
    return Authorization(document=document, sha256=authorization_digest(document))


def validate_authorization(
    authorization: Authorization,
    *,
    contract: FrozenStageAContract,
    stable_executable_commit: str,
    executable_inventory_hash: str,
    tooling_proposal_hash: str,
    artifact_contract_hash: str,
    operation: str,
    partition: str,
    run_ids: Iterable[int],
    generation_root: Path,
    replay_root: Path,
    acceptance_root: Path,
) -> Authorization:
    value = authorization.document
    required = {
        "schema_identifier", "authorization_version", "authorization_id", "authorization_status",
        "authorized_contract_hash", "authorized_contract_tag", "authorized_frozen_commit",
        "authorized_stable_executable_commit", "authorized_executable_inventory_hash",
        "authorized_tooling_proposal_hash", "authorized_artifact_contract_hash",
        "authorized_partition", "authorized_run_ids", "authorized_run_count", "authorized_operation",
        "authorized_generation_root", "authorized_replay_root", "authorized_acceptance_root",
        "source_generation_ledger_relative_path", "source_generation_ledger_byte_length",
        "source_generation_ledger_sha256", "source_replay_ledger_relative_path",
        "source_replay_ledger_byte_length", "source_replay_ledger_sha256",
        "authorization_date", "authorizing_decision_reference", "independently_approved",
        "authorization_sha256",
    }
    if set(value) != required:
        raise ValueError("Authorization field set mismatch")
    if value["schema_identifier"] != AUTHORIZATION_SCHEMA or value["authorization_version"] != "1":
        raise ValueError("Authorization schema/version mismatch")
    if value["authorization_status"] != "AUTHORIZED" or value["independently_approved"] is not True:
        raise PermissionError("Authorization is not independently approved and AUTHORIZED")
    if operation not in OPERATIONS or value["authorized_operation"] != operation:
        raise PermissionError("Authorization operation mismatch")
    if partition not in EXECUTABLE_PARTITIONS or value["authorized_partition"] != partition:
        raise PermissionError("Authorization partition mismatch or sealed holdout denied")
    expected_ids = tuple(sorted(run_ids))
    authorized_ids = value["authorized_run_ids"]
    if not isinstance(authorized_ids, list) or any(type(item) is not int for item in authorized_ids):
        raise ValueError("Authorized run IDs must be integers")
    if tuple(authorized_ids) != expected_ids or len(set(authorized_ids)) != len(authorized_ids):
        raise PermissionError("Authorization run set mismatch")
    if value["authorized_run_count"] != len(expected_ids):
        raise PermissionError("Authorization run count mismatch")
    frozen_partition_ids = tuple(contract.partitions[partition]["global_run_ids"])
    if expected_ids != frozen_partition_ids:
        raise PermissionError("Authorization must bind one complete frozen partition")
    if any(run_id in contract.partitions["sealed_holdout"]["global_run_ids"] for run_id in expected_ids):
        raise PermissionError("Sealed-holdout authorization is prohibited")
    ledger_fields = {
        "source_generation_ledger_relative_path": value["source_generation_ledger_relative_path"],
        "source_generation_ledger_byte_length": value["source_generation_ledger_byte_length"],
        "source_generation_ledger_sha256": value["source_generation_ledger_sha256"],
        "source_replay_ledger_relative_path": value["source_replay_ledger_relative_path"],
        "source_replay_ledger_byte_length": value["source_replay_ledger_byte_length"],
        "source_replay_ledger_sha256": value["source_replay_ledger_sha256"],
    }
    required_generation = operation in {"REPLAY", "ACCEPT"}
    required_replay = operation == "ACCEPT"
    expected_paths = {
        "source_generation_ledger_relative_path": "execution_ledger.json" if required_generation else None,
        "source_replay_ledger_relative_path": "replay_ledger.json" if required_replay else None,
    }
    for field, expected in expected_paths.items():
        if ledger_fields[field] != expected:
            raise PermissionError(f"Authorization ledger path mismatch: {field}")
    for prefix, required_binding in (("source_generation_ledger", required_generation), ("source_replay_ledger", required_replay)):
        length = ledger_fields[f"{prefix}_byte_length"]
        digest = ledger_fields[f"{prefix}_sha256"]
        if required_binding:
            if type(length) is not int or length <= 0:
                raise ValueError(f"{prefix}_byte_length must be a positive integer")
            ensure_hex(digest, length=64, field=f"{prefix}_sha256")
        elif length is not None or digest is not None:
            raise PermissionError(f"Authorization includes an inapplicable ledger binding: {prefix}")
    expected_identity = {
        "authorized_contract_hash": contract.contract_hash,
        "authorized_contract_tag": contract.frozen_tag,
        "authorized_frozen_commit": contract.frozen_commit,
        "authorized_stable_executable_commit": stable_executable_commit,
        "authorized_executable_inventory_hash": executable_inventory_hash,
        "authorized_tooling_proposal_hash": tooling_proposal_hash,
        "authorized_artifact_contract_hash": artifact_contract_hash,
        "authorized_generation_root": str(generation_root.resolve(strict=False)),
        "authorized_replay_root": str(replay_root.resolve(strict=False)),
        "authorized_acceptance_root": str(acceptance_root.resolve(strict=False)),
    }
    for field, expected in expected_identity.items():
        if value[field] != expected:
            raise PermissionError(f"Authorization identity mismatch: {field}")
    ensure_hex(stable_executable_commit, length=40, field="stable_executable_commit")
    ensure_hex(executable_inventory_hash, length=64, field="executable_inventory_hash")
    ensure_hex(tooling_proposal_hash, length=64, field="tooling_proposal_hash")
    ensure_hex(artifact_contract_hash, length=64, field="artifact_contract_hash")
    digest = authorization_digest(value)
    if value["authorization_sha256"] != digest or authorization.sha256 != digest:
        raise PermissionError("Authorization SHA-256 mismatch")
    if not str(value["authorization_id"]).strip() or not str(value["authorizing_decision_reference"]).strip():
        raise ValueError("Authorization governance identity is empty")
    return authorization
