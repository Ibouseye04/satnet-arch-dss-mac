from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from satnet.experiments.final_class_support_audit.audit import verify_frozen_evidence

PRODUCTION_TOOLING_SHA = "9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab"
CONTRACT_SPECIFICATION_HASH = "482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b"
GENERATION_LEDGER_SHA256 = "a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1"
REPLAY_LEDGER_SHA256 = "4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd"
FREEZE_ARCHIVE_SHA256 = "375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc"
EXPECTED_FILE_COUNT = 9_004
EXPECTED_BYTE_COUNT = 1_337_549_193


@dataclass(frozen=True)
class FrozenEvidencePaths:
    production_tooling_root: Path
    generation_root: Path
    replay_root: Path
    freeze_root: Path
    freeze_archive: Path
    freeze_archive_hash_file: Path


DEFAULT_FROZEN_EVIDENCE_PATHS = FrozenEvidencePaths(
    production_tooling_root=Path(r"C:\Users\johns\satnet-production-tooling-20260720"),
    generation_root=Path(r"C:\Users\johns\satnet-final-production-20260720"),
    replay_root=Path(r"C:\Users\johns\satnet-final-production-replay-20260720"),
    freeze_root=Path(r"C:\Users\johns\satnet-final-production-v1-freeze-20260721"),
    freeze_archive=Path(r"C:\Users\johns\satnet-final-production-v1-freeze-20260721.zip"),
    freeze_archive_hash_file=Path(r"C:\Users\johns\satnet-final-production-v1-freeze-20260721.zip.sha256"),
)


def validate_frozen_evidence_result(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "production_tooling_sha": PRODUCTION_TOOLING_SHA,
        "contract_specification_hash": CONTRACT_SPECIFICATION_HASH,
        "generation_ledger_sha256": GENERATION_LEDGER_SHA256,
        "replay_ledger_sha256": REPLAY_LEDGER_SHA256,
        "freeze_archive_sha256": FREEZE_ARCHIVE_SHA256,
    }
    for field, identity in expected.items():
        if value.get(field) != identity:
            raise ValueError(f"Frozen production evidence identity mismatch: {field}")
    combined = value.get("combined")
    if not isinstance(combined, dict):
        raise ValueError("Frozen production evidence combined result missing")
    if combined.get("file_count") != EXPECTED_FILE_COUNT or combined.get("byte_count") != EXPECTED_BYTE_COUNT:
        raise ValueError("Frozen production evidence cardinality mismatch")
    if combined.get("verified_sha256_count") != EXPECTED_FILE_COUNT:
        raise ValueError("Frozen production evidence was not fully hash verified")
    generation = value.get("generation")
    replay = value.get("replay")
    if not isinstance(generation, dict) or not isinstance(replay, dict):
        raise ValueError("Frozen production evidence partition results missing")
    if generation.get("all_files_read_only") is not True or replay.get("all_files_read_only") is not True:
        raise ValueError("Frozen production evidence files are not read-only")
    if value.get("archive_read_only") is not True or value.get("freeze_metadata_all_read_only") is not True:
        raise ValueError("Frozen production archive or metadata is not read-only")
    if value.get("verification_status") != "passed":
        raise ValueError("Frozen production evidence verification did not pass")
    return value


def verify_frozen_production_evidence(
    paths: FrozenEvidencePaths = DEFAULT_FROZEN_EVIDENCE_PATHS,
    verifier: Callable[..., dict[str, Any]] = verify_frozen_evidence,
) -> dict[str, Any]:
    result = verifier(
        production_tooling_root=paths.production_tooling_root,
        generation_root=paths.generation_root,
        replay_root=paths.replay_root,
        freeze_root=paths.freeze_root,
        freeze_archive=paths.freeze_archive,
        freeze_archive_hash_file=paths.freeze_archive_hash_file,
    )
    return validate_frozen_evidence_result(result)
