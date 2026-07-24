from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from satnet.experiments.stage_a_execution.authorization import Authorization, authorization_digest
from satnet.experiments.stage_a_execution.common import read_json_object, sha256_file
from satnet.experiments.stage_a_execution.contract import load_frozen_contract
from satnet.experiments.stage_a_execution.identity import (
    STABLE_IDENTITY_RELATIVE,
    TOOLING_INVENTORY_RELATIVE,
)
from satnet.experiments.stage_a_execution.plan import build_plan


ROOT = Path(__file__).parents[2]
PROPOSAL_COMMIT = "fc9250aa702ddf8da1814da32fe06977af4afc44"
PROPOSAL_PARENT = "9c85fc614949b1b99eb57473751e46cc0b598665"
V1_PROPOSAL_COMMIT = "9cc2e6664756958539dbdee215e1543e5cab363b"
V1_ROOT = ROOT / "artifacts/stage_a_development_acceptance_authorization_v1_proposal"
V2_ROOT = ROOT / "artifacts/stage_a_development_acceptance_authorization_v2_proposal"
GENERATION_ROOT = Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-production")
REPLAY_ROOT = Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-replay")
ACCEPTANCE_ROOT = Path(r"C:\Users\johns\satnet-stage-a-discovery-v1-acceptance")
GENERATION_LEDGER = GENERATION_ROOT / "execution_ledger.json"
REPLAY_LEDGER = REPLAY_ROOT / "replay_ledger.json"
GENERATION_LEDGER_IDENTITY = (1_060_132, "9df1448612d4719d164cf014a2ab60d932d82ab07a90e05c50a0c1650b097328")
REPLAY_LEDGER_IDENTITY = (1_079_014, "140bda39b3f519d7baeae09d222cd3f6c80791ff03215c11c48dd66fb6e8b9bb")
EXPECTED_SEMANTIC_SHA256 = "0fc3b4eb0be9e8cb046f4198876402f386de014ed29422bf8395981150210d7b"
EXPECTED_ORDERED_RUN_IDS_SHA256 = "94a3fa9098c01b348ced330b0d12f69fcf3d36003d47aa19294fe4637a27c732"
EXPECTED_PLANNING_AUTHORIZATION_SHA256 = "2e1942e8d1ab1f6b3e8fd62eef25b72dcdf2eb3a141d8725085631c0345becb8"
EXPECTED_PLAN_HASH = "4d90038ed8f95a76285b73e9ea32ccedc4536964d2a73a4e5420530642e74dea"
EXPECTED_ACCEPT_CAMPAIGN = "c7020e85471d12a5cdb1ff33df614bbd3d04ff58f422ad5c4d60d87328bc3bf6"
EXPECTED_REPLAY_CAMPAIGNS = {
    "3a8a62a5e73c81503e5b75d0c44a820eb545596968d837aa72e7e1719a7e3a50",
    "cc50220189cc849431319fc9ce80b486e76beb6de3c25164ec24f74ff024cb44",
}
EXPECTED_V2_FILES = {
    "README.md": (6_696, "b421fef9af5686cabb53339fde257cb374885013d1bd1b1d9ac7845826b9df67"),
    "stage_a_development_acceptance_authorization_inventory.json": (2_489, "ee6f180d6a3e628c50a9a8d2b1bfe35b975981dbc2149aa959f5c4703d7b3d72"),
    "stage_a_development_acceptance_authorization_proposal.json": (9_888, "5b101b34d0a9f7edf3a13f8394c8e8f2275cc628ba4accbde038febf9801afea"),
    "stage_a_development_acceptance_authorization_supersession.json": (5_845, "b476d446bc84a5ff4f74f8522abc23d0b108082de1b348229a06f1b328edef54"),
    "stage_a_development_authorized_acceptance_run_manifest.csv": (25_122, "7f6013872824552bac32ebb76a5d26eda85f645f339b0eaa0a2fdd8577018642"),
}
EXPECTED_V1_FILES = {
    "README.md": (4_272, "4af03938ff0adf97b58d5d16644a0972497291c8ea8e9b29a6138ad703ba2e57"),
    "stage_a_development_acceptance_authorization_inventory.json": (1_518, "b908d7fe1aedd33b2476d4f562d3b442054ff93d0119aa53faf46ad3e8255dd7"),
    "stage_a_development_acceptance_authorization_proposal.json": (7_340, "68fe8890e1c4eabc283ea3954f966388c6a92897b138d51666ae6a9bdef186f4"),
    "stage_a_development_authorized_acceptance_run_manifest.csv": (25_122, "7f6013872824552bac32ebb76a5d26eda85f645f339b0eaa0a2fdd8577018642"),
}
EXPECTED_ROOTS = {
    GENERATION_ROOT: (1_902, 366_369_907, "1be2fe3523d40a900a0283f41b55f8f7fbdeba1ea9c394325c0003d914169e23"),
    REPLAY_ROOT: (2_001, 366_571_000, "e5dcd6d51d55f072b2531e7cbd684b712e605f56be6f9332d051ebb9cdba06a3"),
}
EXPECTED_ARTIFACTS = {GENERATION_ROOT: (1_900, 365_309_146), REPLAY_ROOT: (2_000, 365_491_986)}
SEED_FIELDS = (
    "design_construction_seed",
    "ground_selection_seed",
    "satellite_failure_seed",
    "ground_failure_seed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _root_identities(root: Path) -> dict[str, tuple[int, str]]:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    return {path.relative_to(root).as_posix(): (path.stat().st_size, _sha256(path)) for path in files}


def _tree_sha256(identities: dict[str, tuple[int, str]]) -> str:
    digest = hashlib.sha256()
    for relative_path, (byte_length, sha256) in identities.items():
        digest.update(f"{relative_path}\0{byte_length}\0{sha256}\n".encode())
    return digest.hexdigest()


def _planning_plan() -> tuple[Authorization, dict[str, Any]]:
    contract = load_frozen_contract(ROOT, operation="ACCEPT")
    stable = read_json_object(ROOT / STABLE_IDENTITY_RELATIVE)
    proposal = _json(V2_ROOT / "stage_a_development_acceptance_authorization_proposal.json")
    document: dict[str, Any] = {
        "schema_identifier": "satnet.stage_a.execution_authorization.v1",
        "authorization_version": "1",
        "authorization_id": "TEST-ONLY-OPERATION-BOUND-ACCEPT",
        "authorization_status": "AUTHORIZED",
        "authorized_contract_hash": contract.contract_hash,
        "authorized_contract_tag": contract.frozen_tag,
        "authorized_frozen_commit": contract.frozen_commit,
        "authorized_stable_executable_commit": stable["stable_executable_commit"],
        "authorized_executable_inventory_hash": stable["executable_inventory_sha256"],
        "authorized_tooling_proposal_hash": sha256_file(ROOT / TOOLING_INVENTORY_RELATIVE),
        "authorized_artifact_contract_hash": proposal["artifact_contract_sha256"],
        "authorized_partition": "development",
        "authorized_run_ids": list(contract.partitions["development"]["global_run_ids"]),
        "authorized_run_count": contract.partitions["development"]["run_count"],
        "authorized_operation": "ACCEPT",
        "authorized_generation_root": str(GENERATION_ROOT.resolve()),
        "authorized_replay_root": str(REPLAY_ROOT.resolve()),
        "authorized_acceptance_root": str(ACCEPTANCE_ROOT.resolve()),
        "source_generation_ledger_relative_path": "execution_ledger.json",
        "source_generation_ledger_byte_length": GENERATION_LEDGER_IDENTITY[0],
        "source_generation_ledger_sha256": GENERATION_LEDGER_IDENTITY[1],
        "source_replay_ledger_relative_path": "replay_ledger.json",
        "source_replay_ledger_byte_length": REPLAY_LEDGER_IDENTITY[0],
        "source_replay_ledger_sha256": REPLAY_LEDGER_IDENTITY[1],
        "authorization_date": "2099-01-01",
        "authorizing_decision_reference": "TEST-ONLY-NONPERSISTED-PREFLIGHT",
        "independently_approved": True,
    }
    document["authorization_sha256"] = authorization_digest(document)
    authorization = Authorization(document=document, sha256=document["authorization_sha256"])
    plan = build_plan(
        contract,
        partition="development",
        operation="ACCEPT",
        stable_executable_commit=document["authorized_stable_executable_commit"],
        executable_inventory_hash=document["authorized_executable_inventory_hash"],
        tooling_proposal_hash=document["authorized_tooling_proposal_hash"],
        artifact_contract_hash=document["authorized_artifact_contract_hash"],
        generation_root=GENERATION_ROOT,
        replay_root=REPLAY_ROOT,
        acceptance_root=ACCEPTANCE_ROOT,
        authorization=authorization,
    )
    return authorization, plan


def test_proposal_bytes_inventory_and_v1_supersession_are_exact() -> None:
    assert _git(ROOT, "rev-parse", PROPOSAL_COMMIT).strip() == PROPOSAL_COMMIT
    assert _git(ROOT, "rev-parse", f"{PROPOSAL_COMMIT}^").strip() == PROPOSAL_PARENT
    assert set(path.name for path in V2_ROOT.iterdir()) == set(EXPECTED_V2_FILES)
    for name, identity in EXPECTED_V2_FILES.items():
        path = V2_ROOT / name
        relative = path.relative_to(ROOT).as_posix()
        assert (path.stat().st_size, _sha256(path)) == identity
        assert _git(ROOT, "hash-object", "--no-filters", "--", relative).strip() == _git(
            ROOT, "rev-parse", f"{PROPOSAL_COMMIT}:{relative}"
        ).strip()
    proposal = _json(V2_ROOT / "stage_a_development_acceptance_authorization_proposal.json")
    semantic = dict(proposal)
    assert semantic.pop("proposal_semantic_sha256") == EXPECTED_SEMANTIC_SHA256
    canonical_semantic = (json.dumps(semantic, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    assert hashlib.sha256(canonical_semantic).hexdigest() == EXPECTED_SEMANTIC_SHA256
    inventory_path = V2_ROOT / "stage_a_development_acceptance_authorization_inventory.json"
    inventory = _json(inventory_path)
    regenerated = dict(inventory)
    regenerated["artifacts"] = [
        {"byte_length": (V2_ROOT / name).stat().st_size, "relative_path": name, "sha256": _sha256(V2_ROOT / name)}
        for name in sorted(set(EXPECTED_V2_FILES) - {inventory_path.name})
    ]
    assert (json.dumps(regenerated, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode() == inventory_path.read_bytes()
    assert _git(ROOT, "diff", "--name-only", V1_PROPOSAL_COMMIT, PROPOSAL_COMMIT, "--", V1_ROOT.relative_to(ROOT).as_posix()) == ""
    for name, identity in EXPECTED_V1_FILES.items():
        path = V1_ROOT / name
        relative = path.relative_to(ROOT).as_posix()
        assert (path.stat().st_size, _sha256(path)) == identity
        assert _git(ROOT, "hash-object", "--no-filters", "--", relative).strip() == _git(
            ROOT, "rev-parse", f"{V1_PROPOSAL_COMMIT}:{relative}"
        ).strip()
    v1_proposal = _json(V1_ROOT / "stage_a_development_acceptance_authorization_proposal.json")
    supersession = _json(V2_ROOT / "stage_a_development_acceptance_authorization_supersession.json")
    assert v1_proposal["authorization_status"] == "PROPOSED"
    assert all(b"SUPERSEDED" not in path.read_bytes() for path in V1_ROOT.iterdir() if path.is_file())
    assert supersession["superseded_proposal"]["authorization_status"] == "SUPERSEDED"
    assert supersession["superseded_proposal"]["proposal_files_modified"] is False
    assert supersession["superseded_proposal"]["inventory_modified"] is False


def test_inactive_scope_planning_and_campaign_identities_are_exact() -> None:
    proposal = _json(V2_ROOT / "stage_a_development_acceptance_authorization_proposal.json")
    inventory = _json(V2_ROOT / "stage_a_development_acceptance_authorization_inventory.json")
    supersession = _json(V2_ROOT / "stage_a_development_acceptance_authorization_supersession.json")
    for document in (proposal, inventory, supersession):
        assert document["authorization_status"] == "PROPOSED"
        assert document["authorization_active"] is False
        assert document["execution_authorized"] is False
        assert document["simulation_authorized"] is False
        assert document["production_authorized"] is False
    assert proposal["authorization_operation"] == "ACCEPT"
    assert proposal["authorized_partition"] == "development"
    assert proposal["explicitly_not_authorized"] == [
        "GENERATE", "REPLAY", "validation partition", "sealed_holdout", "Stage B", "RF training", "TGNN training"
    ]
    authorization, plan = _planning_plan()
    assert authorization.sha256 == proposal["current_accept_planning_authorization_sha256"] == EXPECTED_PLANNING_AUTHORIZATION_SHA256
    assert plan["plan_hash"] == proposal["current_accept_plan_hash"] == EXPECTED_PLAN_HASH
    assert plan["campaign_manifest_version"] == proposal["current_accept_campaign_manifest_version"] == "2"
    assert plan["campaign_manifest_algorithm"] == proposal["current_accept_campaign_manifest_algorithm"] == "satnet.stage_a.campaign_manifest.operation_bound.v2"
    assert plan["campaign_manifest_operation"] == "ACCEPT"
    assert plan["campaign_manifest_hash"] == proposal["current_accept_campaign_manifest_hash"] == EXPECTED_ACCEPT_CAMPAIGN
    assert EXPECTED_ACCEPT_CAMPAIGN not in EXPECTED_REPLAY_CAMPAIGNS
    assert proposal["current_accept_plan_identity_policy"] == "AUDIT_COMMIT_DETERMINISTIC_PLANNING_REFERENCE_ONLY_NOT_RUNTIME_AUTHORIZATION"
    readme = (V2_ROOT / "README.md").read_text(encoding="utf-8")
    assert "It is a nonauthorizing planning identity only" in readme
    assert "Activation must create a separate runtime authorization" in readme
    assert "activated runtime plan identity must be derived and verified from that exact authorization" in readme
    stable_path = ROOT / STABLE_IDENTITY_RELATIVE
    stable = _json(stable_path)
    assert stable["stable_executable_commit"] == proposal["current_stable_executable_commit"] == "b1da92c3184ee86997079f1eb458fae95d994c22"
    assert stable["executable_inventory_sha256"] == proposal["current_executable_inventory_sha256"] == "2ebac3bb615ed3204edc1359a78f78e4354495364fa1f76dfd1648b417df7b56"
    assert _sha256(stable_path) == proposal["current_stable_identity_sha256"] == "1902a91a38e87faef6fe99c2e1f7aa4c3aa8dd6febf38e6a5b43a272b190c7f9"


def test_manifest_and_ledgers_have_exact_membership_ordering_and_cross_binding() -> None:
    manifest = _csv(V2_ROOT / "stage_a_development_authorized_acceptance_run_manifest.csv")
    generation = _json(GENERATION_LEDGER)
    replay = _json(REPLAY_LEDGER)
    assert (GENERATION_LEDGER.stat().st_size, _sha256(GENERATION_LEDGER)) == GENERATION_LEDGER_IDENTITY
    assert (REPLAY_LEDGER.stat().st_size, _sha256(REPLAY_LEDGER)) == REPLAY_LEDGER_IDENTITY
    assert generation["campaign_manifest_hash"] == "af76d64bbabdb055d413a1c3e2f28809749710d67f7819465d45c0ef0769e164"
    assert replay["campaign_manifest_hash"] == "3a8a62a5e73c81503e5b75d0c44a820eb545596968d837aa72e7e1719a7e3a50"
    assert replay["source_generation_ledger_relative_path"] == "execution_ledger.json"
    assert replay["source_generation_ledger_byte_length"] == GENERATION_LEDGER_IDENTITY[0]
    assert replay["source_generation_ledger_sha256"] == GENERATION_LEDGER_IDENTITY[1]
    generation_records = generation["records"]
    replay_records = replay["records"]
    assert len(manifest) == len(generation_records) == len(replay_records) == 100
    assert all(record["state"] == "SUCCEEDED" for record in generation_records + replay_records)
    assert all(record["science_completion"]["validation_status"] == "PASSED" for record in generation_records + replay_records)
    assert all(record["adapter_result"]["validation_status"] == "PASSED" for record in generation_records + replay_records)
    manifest_ids = [int(row["global_run_id"]) for row in manifest]
    assert manifest_ids == [record["global_run_id"] for record in generation_records] == [record["global_run_id"] for record in replay_records]
    assert hashlib.sha256(json.dumps(manifest_ids, separators=(",", ":")).encode()).hexdigest() == EXPECTED_ORDERED_RUN_IDS_SHA256
    assert len(set(manifest_ids)) == len({row["run_key"] for row in manifest}) == 100
    assert len({(row["design_id"], row["realization_id"]) for row in manifest}) == 100
    assert len({row["design_id"] for row in manifest}) == 20
    assert {row["partition"] for row in manifest} == {"development"}
    assert manifest[0]["run_key"] == generation_records[0]["run_key"] == replay_records[0]["run_key"] == "SA-D000-R00"
    assert manifest[-1]["run_key"] == generation_records[-1]["run_key"] == replay_records[-1]["run_key"] == "SA-D027-R04"
    for expected, generation_record, replay_record in zip(manifest, generation_records, replay_records, strict=True):
        for record in (generation_record, replay_record):
            assert record["run_key"] == expected["run_key"]
            assert record["global_run_id"] == int(expected["global_run_id"])
            assert record["run_record_hash"] == expected["run_record_hash"]
            assert record["adapter_result"]["design_id"] == expected["design_id"]
            assert record["adapter_result"]["realization_id"] == expected["realization_id"]
            for field in SEED_FIELDS:
                assert record[field] == int(expected[field])
                assert record["adapter_result"][field] == int(expected[field])
        for field in ("run_key", "global_run_id", "run_record_hash", *SEED_FIELDS):
            assert generation_record[field] == replay_record[field]
        for field in ("design_id", "realization_id", *SEED_FIELDS):
            assert generation_record["adapter_result"][field] == replay_record["adapter_result"][field]
    reports = list(REPLAY_ROOT.rglob("replay_report.json"))
    assert len(reports) == 100
    assert all(sum(entry["relative_path"] == "replay_report.json" for entry in record["artifacts"]) == 1 for record in replay_records)


def test_all_source_artifacts_and_trees_rehash_exactly() -> None:
    for root, ledger_path in ((GENERATION_ROOT, GENERATION_LEDGER), (REPLAY_ROOT, REPLAY_LEDGER)):
        ledger = _json(ledger_path)
        identities = _root_identities(root)
        expected_files, expected_bytes, expected_hash = EXPECTED_ROOTS[root]
        assert (len(identities), sum(value[0] for value in identities.values()), _tree_sha256(identities)) == (
            expected_files, expected_bytes, expected_hash
        )
        paths: list[str] = []
        byte_count = 0
        for record in ledger["records"]:
            for entry in record["artifacts"]:
                relative = Path(entry["relative_path"])
                assert not relative.is_absolute() and ".." not in relative.parts
                rooted = (Path(record["output_relative_path"]) / relative).as_posix()
                assert identities[rooted] == (entry["byte_length"], entry["sha256"])
                paths.append(rooted)
                byte_count += entry["byte_length"]
        expected_artifact_count, expected_artifact_bytes = EXPECTED_ARTIFACTS[root]
        assert (len(paths), len(set(paths)), byte_count) == (
            expected_artifact_count,
            expected_artifact_count,
            expected_artifact_bytes,
        )
        metadata = {"execution_ledger.json", "campaign_identity.json"} if root == GENERATION_ROOT else {"replay_ledger.json"}
        assert set(identities) == set(paths) | metadata


def test_roots_locks_and_nonexecution_state_are_exact() -> None:
    roots = (GENERATION_ROOT, REPLAY_ROOT, ACCEPTANCE_ROOT)
    assert GENERATION_ROOT.is_dir() and REPLAY_ROOT.is_dir() and not ACCEPTANCE_ROOT.exists()
    resolved = [root.resolve(strict=root != ACCEPTANCE_ROOT) for root in roots]
    assert len(set(resolved)) == 3
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            assert not left.is_relative_to(right) and not right.is_relative_to(left)
    directories = [path for root in roots[:2] for path in root.rglob("*") if path.is_dir()]
    assert not [path for path in directories if "failed" in path.name.lower() or "in_progress" in path.name.lower()]
    locks = [path for root in roots[:2] for path in root.rglob("*.lock")]
    locks.extend(root.with_name(root.name + ".lock") for root in roots if root.with_name(root.name + ".lock").exists())
    assert locks == []
    assert not list(GENERATION_ROOT.rglob("acceptance_report.json"))
    assert not list(REPLAY_ROOT.rglob("acceptance_report.json"))


def test_fresh_windows_checkout_is_clean_with_exact_unfiltered_bytes() -> None:
    checkout = Path(os.environ["SATNET_ACCEPTANCE_AUTHORIZATION_V2_FRESH_CHECKOUT"])
    assert _git(checkout, "rev-parse", "HEAD").strip() == PROPOSAL_COMMIT
    assert _git(checkout, "rev-parse", "HEAD^").strip() == PROPOSAL_PARENT
    assert _git(checkout, "config", "--get", "core.autocrlf").strip() == "true"
    assert _git(checkout, "status", "--porcelain=v1", "--untracked-files=all") == ""
    for name, identity in EXPECTED_V2_FILES.items():
        relative = f"artifacts/stage_a_development_acceptance_authorization_v2_proposal/{name}"
        path = checkout / relative
        assert (path.stat().st_size, _sha256(path)) == identity
        expected_blob = _git(checkout, "rev-parse", f"{PROPOSAL_COMMIT}:{relative}").strip()
        assert _git(checkout, "hash-object", "--no-filters", "--", relative).strip() == expected_blob
        assert _git(checkout, "check-attr", "text", "--", relative).strip().endswith("text: unset")
