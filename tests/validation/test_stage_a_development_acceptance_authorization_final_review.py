from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).parents[2]
PROPOSAL_COMMIT = "9cc2e6664756958539dbdee215e1543e5cab363b"
PROPOSAL_PARENT = "1b48b5735a192db84ef191c75c75d3872b9d7d32"
PROPOSAL_ROOT = ROOT / "artifacts/stage_a_development_acceptance_authorization_v1_proposal"
GENERATION_ROOT = Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-production")
REPLAY_ROOT = Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-replay")
ACCEPTANCE_ROOT = Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-acceptance")
GENERATION_LEDGER = GENERATION_ROOT / "execution_ledger.json"
REPLAY_LEDGER = REPLAY_ROOT / "replay_ledger.json"
EXPECTED_GENERATION_LEDGER_LENGTH = 1_060_132
EXPECTED_GENERATION_LEDGER_SHA256 = "9df1448612d4719d164cf014a2ab60d932d82ab07a90e05c50a0c1650b097328"
EXPECTED_REPLAY_LEDGER_LENGTH = 1_079_014
EXPECTED_REPLAY_LEDGER_SHA256 = "140bda39b3f519d7baeae09d222cd3f6c80791ff03215c11c48dd66fb6e8b9bb"
EXPECTED_SEMANTIC_SHA256 = "d6d1d6b4cf1679f341a934372de07bc5a6e1f28cab647cfd10b373f482f9a00d"
EXPECTED_ORDERED_RUN_IDS_SHA256 = "94a3fa9098c01b348ced330b0d12f69fcf3d36003d47aa19294fe4637a27c732"
EXPECTED_PROPOSAL_FILES = {
    "stage_a_development_acceptance_authorization_proposal.json": (
        7_340,
        "68fe8890e1c4eabc283ea3954f966388c6a92897b138d51666ae6a9bdef186f4",
    ),
    "stage_a_development_authorized_acceptance_run_manifest.csv": (
        25_122,
        "7f6013872824552bac32ebb76a5d26eda85f645f339b0eaa0a2fdd8577018642",
    ),
    "stage_a_development_acceptance_authorization_inventory.json": (
        1_518,
        "b908d7fe1aedd33b2476d4f562d3b442054ff93d0119aa53faf46ad3e8255dd7",
    ),
    "README.md": (
        4_272,
        "4af03938ff0adf97b58d5d16644a0972497291c8ea8e9b29a6138ad703ba2e57",
    ),
}
EXPECTED_ROOTS = {
    GENERATION_ROOT: (1_902, 366_369_907, "1be2fe3523d40a900a0283f41b55f8f7fbdeba1ea9c394325c0003d914169e23"),
    REPLAY_ROOT: (2_001, 366_571_000, "e5dcd6d51d55f072b2531e7cbd684b712e605f56be6f9332d051ebb9cdba06a3"),
}
EXPECTED_ARTIFACTS = {
    GENERATION_ROOT: (1_900, 365_309_146),
    REPLAY_ROOT: (2_000, 365_491_986),
}
SEED_FIELDS = (
    "design_construction_seed",
    "ground_selection_seed",
    "satellite_failure_seed",
    "ground_failure_seed",
)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _root_file_identities(root: Path) -> dict[str, tuple[int, str]]:
    identities: dict[str, tuple[int, str]] = {}
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative_path = path.relative_to(root).as_posix()
        identities[relative_path] = (path.stat().st_size, _sha256(path))
    return identities


def _tree_sha256(identities: dict[str, tuple[int, str]]) -> str:
    hasher = hashlib.sha256()
    for relative_path, (byte_length, digest) in identities.items():
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(byte_length).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(digest.encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def test_fresh_windows_checkout_has_exact_proposal_bytes() -> None:
    checkout = Path(os.environ["SATNET_ACCEPTANCE_AUTHORIZATION_FRESH_CHECKOUT"])
    assert _git(checkout, "rev-parse", "HEAD").stdout.strip() == PROPOSAL_COMMIT
    assert _git(checkout, "rev-parse", "HEAD^").stdout.strip() == PROPOSAL_PARENT
    assert _git(checkout, "config", "--get", "core.autocrlf").stdout.strip() == "true"
    assert _git(checkout, "status", "--porcelain=v1", "--untracked-files=all").stdout == ""
    attributes = (checkout / ".gitattributes").read_text(encoding="utf-8").splitlines()
    rule = "artifacts/stage_a_development_acceptance_authorization_v1_proposal/** -text"
    assert rule in attributes
    for name, (expected_length, expected_hash) in EXPECTED_PROPOSAL_FILES.items():
        relative = f"artifacts/stage_a_development_acceptance_authorization_v1_proposal/{name}"
        checkout_path = checkout / relative
        audit_path = ROOT / relative
        expected_blob = _git(checkout, "rev-parse", f"{PROPOSAL_COMMIT}:{relative}").stdout.strip()
        checkout_blob = _git(checkout, "hash-object", "--no-filters", "--", relative).stdout.strip()
        audit_blob = _git(ROOT, "hash-object", "--no-filters", "--", relative).stdout.strip()
        assert checkout_blob == expected_blob
        assert audit_blob == expected_blob
        assert checkout_path.read_bytes() == audit_path.read_bytes()
        assert checkout_path.stat().st_size == expected_length
        assert _sha256(checkout_path) == expected_hash
        assert _git(checkout, "check-attr", "text", "--", relative).stdout.strip().endswith(
            "text: unset"
        )


def test_inventory_semantic_hash_and_inactive_acceptance_scope_are_exact() -> None:
    proposal_path = PROPOSAL_ROOT / "stage_a_development_acceptance_authorization_proposal.json"
    manifest_path = PROPOSAL_ROOT / "stage_a_development_authorized_acceptance_run_manifest.csv"
    inventory_path = PROPOSAL_ROOT / "stage_a_development_acceptance_authorization_inventory.json"
    readme_path = PROPOSAL_ROOT / "README.md"
    proposal = _json(proposal_path)
    inventory = _json(inventory_path)
    assert inventory["schema_identifier"] == (
        "satnet.stage_a.development_acceptance_authorization_proposal_inventory.v1"
    )
    assert inventory["artifact_count_excluding_inventory"] == 3
    entries = {entry["relative_path"]: entry for entry in inventory["artifacts"]}
    assert set(entries) == {proposal_path.name, manifest_path.name, readme_path.name}
    for path in (proposal_path, manifest_path, readme_path):
        assert entries[path.name]["byte_length"] == path.stat().st_size
        assert entries[path.name]["sha256"] == _sha256(path)
    assert inventory_path.stat().st_size == EXPECTED_PROPOSAL_FILES[inventory_path.name][0]
    assert _sha256(inventory_path) == EXPECTED_PROPOSAL_FILES[inventory_path.name][1]
    semantic = dict(proposal)
    stored_semantic_hash = semantic.pop("proposal_semantic_sha256")
    canonical = (
        json.dumps(semantic, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    assert stored_semantic_hash == EXPECTED_SEMANTIC_SHA256
    assert hashlib.sha256(canonical).hexdigest() == EXPECTED_SEMANTIC_SHA256
    for document in (proposal, inventory):
        assert document["authorization_status"] == "PROPOSED"
        assert document["authorization_active"] is False
        assert document["execution_authorized"] is False
        assert document["simulation_authorized"] is False
        assert document["production_authorized"] is False
    assert proposal["independently_approved"] is False
    assert proposal["acceptance_executed"] is False
    assert proposal["authorization_operation"] == "ACCEPT"
    assert proposal["authorized_partition"] == "development"
    assert proposal["authorized_design_count"] == 20
    assert proposal["authorized_run_count"] == 100
    assert proposal["first_authorized_run"] == "SA-D000-R00"
    assert proposal["last_authorized_run"] == "SA-D027-R04"
    assert proposal["duplicate_run_count"] == 0
    assert proposal["missing_run_count"] == 0
    assert proposal["extra_run_count"] == 0
    assert proposal["validation_run_count"] == 0
    assert proposal["sealed_holdout_run_count"] == 0
    assert proposal["seed_mismatch_count"] == 0
    assert proposal["artifact_completeness_result"] == "PASSED"
    assert set(proposal["explicitly_not_authorized"]) == {
        "GENERATE",
        "REPLAY",
        "validation partition",
        "sealed_holdout",
        "Stage B",
        "RF training",
        "TGNN training",
    }
    readme = readme_path.read_text(encoding="utf-8")
    assert "inactive authorization proposal for operation `ACCEPT`" in readme
    assert "This proposal does not authorize `GENERATE`, `REPLAY`, validation" in readme


def test_manifest_and_both_ledgers_have_exact_membership_ordering_ids_and_seeds() -> None:
    manifest = _csv(PROPOSAL_ROOT / "stage_a_development_authorized_acceptance_run_manifest.csv")
    generation = _json(GENERATION_LEDGER)
    replay = _json(REPLAY_LEDGER)
    assert GENERATION_LEDGER.stat().st_size == EXPECTED_GENERATION_LEDGER_LENGTH
    assert _sha256(GENERATION_LEDGER) == EXPECTED_GENERATION_LEDGER_SHA256
    assert REPLAY_LEDGER.stat().st_size == EXPECTED_REPLAY_LEDGER_LENGTH
    assert _sha256(REPLAY_LEDGER) == EXPECTED_REPLAY_LEDGER_SHA256
    assert generation["schema_identifier"] == "satnet.stage_a.execution_ledger.v2"
    assert replay["schema_identifier"] == "satnet.stage_a.execution_ledger.v2"
    assert generation["operation"] == "GENERATE"
    assert replay["operation"] == "REPLAY"
    assert generation["partition"] == replay["partition"] == "development"
    assert generation["expected_run_count"] == replay["expected_run_count"] == 100
    generation_records = generation["records"]
    replay_records = replay["records"]
    assert len(manifest) == len(generation_records) == len(replay_records) == 100
    assert all(record["state"] == "SUCCEEDED" for record in generation_records + replay_records)
    assert all(
        record["science_completion"]["validation_status"] == "PASSED"
        for record in generation_records + replay_records
    )
    assert all(
        record["adapter_result"]["validation_status"] == "PASSED"
        for record in generation_records + replay_records
    )
    manifest_ids = [int(row["global_run_id"]) for row in manifest]
    generation_ids = [int(record["global_run_id"]) for record in generation_records]
    replay_ids = [int(record["global_run_id"]) for record in replay_records]
    assert manifest_ids == generation_ids == replay_ids == sorted(manifest_ids)
    ordered_payload = json.dumps(manifest_ids, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(ordered_payload).hexdigest() == EXPECTED_ORDERED_RUN_IDS_SHA256
    assert len(set(manifest_ids)) == 100
    assert len({row["run_key"] for row in manifest}) == 100
    assert len({(row["design_id"], row["realization_id"]) for row in manifest}) == 100
    assert len({row["design_id"] for row in manifest}) == 20
    assert all(row["partition"] == "development" for row in manifest)
    assert sum(row["partition"] == "validation" for row in manifest) == 0
    assert sum(row["partition"] == "sealed_holdout" for row in manifest) == 0
    assert manifest[0]["run_key"] == generation_records[0]["run_key"] == replay_records[0]["run_key"] == "SA-D000-R00"
    assert manifest[-1]["run_key"] == generation_records[-1]["run_key"] == replay_records[-1]["run_key"] == "SA-D027-R04"
    manifest_by_key = {row["run_key"]: row for row in manifest}
    generation_by_key = {record["run_key"]: record for record in generation_records}
    replay_by_key = {record["run_key"]: record for record in replay_records}
    assert len(manifest_by_key) == len(generation_by_key) == len(replay_by_key) == 100
    assert set(manifest_by_key) == set(generation_by_key) == set(replay_by_key)
    for run_key, expected in manifest_by_key.items():
        generation_record = generation_by_key[run_key]
        replay_record = replay_by_key[run_key]
        for record in (generation_record, replay_record):
            assert int(record["global_run_id"]) == int(expected["global_run_id"])
            assert record["run_key"] == expected["run_key"]
            assert record["run_record_hash"] == expected["run_record_hash"]
            assert record["adapter_result"]["global_run_id"] == int(expected["global_run_id"])
            assert record["adapter_result"]["run_key"] == expected["run_key"]
            assert record["adapter_result"]["design_id"] == expected["design_id"]
            assert record["adapter_result"]["realization_id"] == expected["realization_id"]
            for field in SEED_FIELDS:
                assert int(record[field]) == int(expected[field])
                assert int(record["adapter_result"][field]) == int(expected[field])
        for field in ("global_run_id", "run_key", "run_record_hash", *SEED_FIELDS):
            assert generation_record[field] == replay_record[field]
        for field in ("global_run_id", "run_key", "design_id", "realization_id", *SEED_FIELDS):
            assert generation_record["adapter_result"][field] == replay_record["adapter_result"][field]


def test_replay_ledger_exactly_binds_generation_ledger_and_has_100_reports() -> None:
    replay = _json(REPLAY_LEDGER)
    proposal = _json(PROPOSAL_ROOT / "stage_a_development_acceptance_authorization_proposal.json")
    assert replay["source_generation_ledger_relative_path"] == "execution_ledger.json"
    assert replay["source_generation_ledger_byte_length"] == EXPECTED_GENERATION_LEDGER_LENGTH
    assert replay["source_generation_ledger_sha256"] == EXPECTED_GENERATION_LEDGER_SHA256
    assert proposal["source_generation_ledger_absolute_path"] == str(GENERATION_LEDGER)
    assert proposal["source_replay_ledger_absolute_path"] == str(REPLAY_LEDGER)
    assert proposal["source_generation_ledger_byte_length"] == EXPECTED_GENERATION_LEDGER_LENGTH
    assert proposal["source_generation_ledger_sha256"] == EXPECTED_GENERATION_LEDGER_SHA256
    assert proposal["source_replay_ledger_byte_length"] == EXPECTED_REPLAY_LEDGER_LENGTH
    assert proposal["source_replay_ledger_sha256"] == EXPECTED_REPLAY_LEDGER_SHA256
    reports = list(REPLAY_ROOT.rglob("replay_report.json"))
    assert len(reports) == 100
    assert all(path.is_file() for path in reports)
    assert all(
        sum(entry["relative_path"] == "replay_report.json" for entry in record["artifacts"]) == 1
        for record in replay["records"]
    )


def test_all_artifacts_and_immutable_source_tree_identities_rehash_exactly() -> None:
    for root, ledger_path in ((GENERATION_ROOT, GENERATION_LEDGER), (REPLAY_ROOT, REPLAY_LEDGER)):
        ledger = _json(ledger_path)
        identities = _root_file_identities(root)
        expected_file_count, expected_byte_count, expected_tree_hash = EXPECTED_ROOTS[root]
        assert len(identities) == expected_file_count
        assert sum(identity[0] for identity in identities.values()) == expected_byte_count
        assert _tree_sha256(identities) == expected_tree_hash
        artifact_paths: list[str] = []
        artifact_bytes = 0
        for record in ledger["records"]:
            adapter_artifacts = record["adapter_result"]["artifact_manifest"]
            if root == GENERATION_ROOT:
                assert record["artifacts"] == adapter_artifacts
            else:
                assert [
                    entry for entry in record["artifacts"]
                    if entry["relative_path"] != "replay_report.json"
                ] == adapter_artifacts
            assert len({entry["relative_path"] for entry in record["artifacts"]}) == len(record["artifacts"])
            for entry in record["artifacts"]:
                relative = Path(entry["relative_path"])
                assert not relative.is_absolute()
                assert ".." not in relative.parts
                rooted_relative = (Path(record["output_relative_path"]) / relative).as_posix()
                assert identities[rooted_relative] == (entry["byte_length"], entry["sha256"])
                artifact_paths.append(rooted_relative)
                artifact_bytes += entry["byte_length"]
        expected_artifact_count, expected_artifact_bytes = EXPECTED_ARTIFACTS[root]
        assert len(artifact_paths) == len(set(artifact_paths)) == expected_artifact_count
        assert artifact_bytes == expected_artifact_bytes
        metadata = {"execution_ledger.json", "campaign_identity.json"} if root == GENERATION_ROOT else {"replay_ledger.json"}
        assert set(identities) == set(artifact_paths) | metadata


def test_roots_are_canonical_distinct_non_overlapping_and_lock_free() -> None:
    proposal = _json(PROPOSAL_ROOT / "stage_a_development_acceptance_authorization_proposal.json")
    roots = (GENERATION_ROOT, REPLAY_ROOT, ACCEPTANCE_ROOT)
    assert GENERATION_ROOT.is_dir()
    assert REPLAY_ROOT.is_dir()
    assert not ACCEPTANCE_ROOT.exists()
    assert proposal["generation_root"] == str(GENERATION_ROOT)
    assert proposal["replay_root"] == str(REPLAY_ROOT)
    assert proposal["reserved_acceptance_root"] == str(ACCEPTANCE_ROOT)
    resolved = [root.resolve(strict=root != ACCEPTANCE_ROOT) for root in roots]
    assert len(set(resolved)) == 3
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            assert not left.is_relative_to(right)
            assert not right.is_relative_to(left)
    source_directories = [path for root in (GENERATION_ROOT, REPLAY_ROOT) for path in root.rglob("*") if path.is_dir()]
    assert not [path for path in source_directories if "in_progress" in path.name.lower()]
    assert not [path for path in source_directories if "failed" in path.name.lower()]
    lock_paths = [path for root in (GENERATION_ROOT, REPLAY_ROOT) for path in root.rglob("*.lock")]
    lock_paths.extend(root.with_name(root.name + ".lock") for root in roots if root.with_name(root.name + ".lock").exists())
    assert lock_paths == []
    assert proposal["in_progress_directory_count"] == 0
    assert proposal["failed_directory_count"] == 0
    assert proposal["lock_count"] == 0
