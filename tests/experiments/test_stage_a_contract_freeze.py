from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import pytest

import satnet.experiments.stage_a_contract.freeze as freeze

ROOT = Path(__file__).parents[2]
CONTRACT_ROOT = ROOT / "artifacts/stage_a_discovery_contract_v1"
PROPOSAL_ROOT = ROOT / "artifacts/stage_a_discovery_contract_proposal"
AUDIT_ROOT = ROOT / "artifacts/stage_a_discovery_contract_freeze_audit"


def _validate(contract_root: Path, **kwargs: Any) -> dict[str, Any]:
    arguments = {
        "repo_root": ROOT,
        "contract_root": contract_root,
        "expected_proposal_commit": freeze.APPROVED_PROPOSAL_COMMIT,
        "expected_proposal_inventory": freeze.APPROVED_PROPOSAL_INVENTORY,
        "expected_audit_commit": freeze.AUDIT_COMMIT,
        "expected_audit_inventory": freeze.AUDIT_INVENTORY,
        **kwargs,
    }
    return freeze.validate_frozen_contract(**arguments)


def _contract_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "contract"
    shutil.copytree(CONTRACT_ROOT, destination)
    return destination


def _proposal_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "proposal"
    shutil.copytree(PROPOSAL_ROOT, destination)
    return destination


def _rewrite_json(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    value = json.loads(path.read_bytes())
    mutate(value)
    path.write_bytes(freeze.canonical_json_bytes(value))


def _rewrite_csv(path: Path, mutate: Callable[[list[dict[str, str]]], None]) -> None:
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    mutate(rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(stream.getvalue().encode("utf-8"))


def test_fresh_worktree_approved_audit_bundle_identity() -> None:
    rows = freeze.verify_approved_audit_bundle(ROOT)
    assert len(rows) == 17
    assert all(row["inventory_match"] is True for row in rows)
    inventory = next(row for row in rows if row["relative_path"] == "audit_inventory.json")
    assert inventory == {
        "relative_path": "audit_inventory.json",
        "byte_length": 3691,
        "sha256": freeze.AUDIT_INVENTORY,
        "lf_count": 107,
        "crlf_count": 0,
        "inventory_match": True,
    }
    freeze.verify_byte_preservation_attributes(ROOT)


def test_approved_proposal_and_frozen_source_copy_are_byte_exact() -> None:
    reproduced = freeze.reproduce_approved_proposal(ROOT)
    assert len(reproduced) == 11
    source_root = CONTRACT_ROOT / "source_bundle"
    assert tuple(sorted(reproduced)) == freeze.SOURCE_ARTIFACTS
    for name, payload in reproduced.items():
        assert PROPOSAL_ROOT.joinpath(name).read_bytes() == payload
        assert source_root.joinpath(name).read_bytes() == payload
    assert freeze.sha256_file(PROPOSAL_ROOT / "stage_a_proposal_inventory.json") == freeze.APPROVED_PROPOSAL_INVENTORY
    assert freeze.sha256_file(PROPOSAL_ROOT / "stage_a_seed_manifest.csv") == freeze.APPROVED_SEED_MANIFEST


def test_frozen_inventory_binds_exactly_twelve_artifacts() -> None:
    inventory_path = CONTRACT_ROOT / freeze.INVENTORY_NAME
    inventory_payload = inventory_path.read_bytes()
    inventory = json.loads(inventory_payload)
    records = inventory["artifacts"]
    assert inventory["contract_bound_artifact_count"] == 12
    assert len(records) == 12
    assert [row["relative_path"] for row in records] == sorted(row["relative_path"] for row in records)
    assert all("\\" not in row["relative_path"] for row in records)
    assert all(row["sha256"] == row["sha256"].lower() and len(row["sha256"]) == 64 for row in records)
    assert all(set(row) == {"relative_path", "byte_length", "sha256", "schema_identifier", "record_count", "source_classification"} for row in records)
    assert freeze.sha256_bytes(inventory_payload) == "e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a"
    assert CONTRACT_ROOT.joinpath(freeze.HASH_NAME).read_bytes() == (
        f"{freeze.sha256_bytes(inventory_payload)}  {freeze.INVENTORY_NAME}\n".encode("utf-8")
    )


def test_frozen_specification_counts_allocations_and_identities() -> None:
    specification = freeze.read_json(CONTRACT_ROOT / freeze.SPECIFICATION_NAME)
    assert specification["cardinality"] == {
        "design_count": 30,
        "run_count": 150,
        "seed_record_count": 150,
        "realizations_per_design": 5,
    }
    assert specification["region_allocation"] == {
        "resilient_core": 12,
        "boundary": 12,
        "global_control": 6,
    }
    assert specification["partition_allocation"] == {
        "development": {"design_count": 20, "run_count": 100},
        "validation": {"design_count": 5, "run_count": 25},
        "sealed_holdout": {"design_count": 5, "run_count": 25},
    }
    identity = specification["identity_namespace"]
    assert identity["design_ids"] == "SA-D000 through SA-D029"
    assert identity["global_run_ids"] == "500 through 649"
    assert identity["run_keys"] == "SA-D000-R00 through SA-D029-R04"


def test_frozen_governance_and_near_neighbor_state() -> None:
    specification = freeze.read_json(CONTRACT_ROOT / freeze.SPECIFICATION_NAME)
    assert specification["threshold_definition"] == {
        "value": "0.80",
        "margin": "failure_adjusted_overall_service_fraction_min - 0.80",
        "non_breach": "margin >= 0",
        "breach": "margin < 0",
    }
    assert specification["canonical_margin"] == {
        "quantization_increment": "0.000001",
        "rounding": "ROUND_HALF_EVEN",
        "distinct_margin_count": "number of unique quantized margins",
    }
    assert specification["boundary_definitions"]["observed_boundary_design"]["combination"] == "condition_a OR condition_b"
    assert "exactly one majority-class" in specification["design_level_class_rules"]["counting_rule"]
    assert specification["sealed_holdout_policy"]["may_confirm_or_reject_only"] is True
    assert specification["sealed_holdout_policy"]["may_alter_frozen_stage_b_contract"] is False
    membership = specification["final_corpus_membership"]
    assert membership["primary_final_classification_corpus"] == [
        "original frozen 500-run corpus",
        "future frozen Stage B corpus",
    ]
    assert {membership[name] for name in ("stage_a_development", "stage_a_validation", "stage_a_sealed_holdout")} == {"EXCLUDED"}
    neighbor = specification["near_neighbor_policy"]
    assert neighbor["changed_design"] == "SA-D020"
    assert neighbor["original_value"] == "0.075"
    assert neighbor["corrected_value"] == "0.100"
    assert neighbor["original_sa_d013_sa_d020_distance"] == pytest.approx(0.07681919236933395, abs=1e-15)
    assert neighbor["corrected_sa_d013_sa_d020_distance"] == pytest.approx(0.12039492645571381, abs=1e-15)
    assert neighbor["minimum_development_validation"] == pytest.approx(0.12039492645571381, abs=1e-15)
    assert neighbor["minimum_development_holdout"] == pytest.approx(0.1205683310788027, abs=1e-15)
    assert neighbor["minimum_validation_holdout"] == pytest.approx(0.1500946005884104, abs=1e-15)
    assert neighbor["remaining_exceptions"] == []
    assert neighbor["pending_scientific_reviews"] == []


def test_specification_inventory_hash_and_declaration_are_deterministic() -> None:
    source_root = CONTRACT_ROOT / "source_bundle"
    first = freeze._expected_bundle_payloads(ROOT, source_root)
    second = freeze._expected_bundle_payloads(ROOT, source_root)
    assert first == second
    for name, payload in first.items():
        assert CONTRACT_ROOT.joinpath(name).read_bytes() == payload
    for name in (freeze.SPECIFICATION_NAME, freeze.INVENTORY_NAME, freeze.DECLARATION_NAME):
        payload = first[name]
        assert not payload.startswith(b"\xef\xbb\xbf")
        assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
        assert b"\r\n" not in payload
    assert freeze.sha256_bytes(first[freeze.INVENTORY_NAME]) == "e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a"


def test_freeze_state_is_frozen_but_fully_unauthorized() -> None:
    result = _validate(CONTRACT_ROOT)
    assert result["validation_verdict"] == "PASSED"
    assert result["source_artifact_count"] == 11
    assert result["contract_artifact_count"] == 12
    assert set(result["authorization_state"].values()) == {False}
    specification = freeze.read_json(CONTRACT_ROOT / freeze.SPECIFICATION_NAME)
    declaration = freeze.read_json(CONTRACT_ROOT / freeze.DECLARATION_NAME)
    for value in (specification, declaration):
        assert value["contract_frozen"] is True
        assert value["freeze_status"] == freeze.FREEZE_STATUS
        assert value["simulation_authorized"] is False
        assert value["production_authorized"] is False
        assert value["execution_authorized"] is False
    readme = CONTRACT_ROOT.joinpath(freeze.README_NAME).read_text(encoding="utf-8")
    assert readme.startswith(
        "STAGE A CONTRACT FROZEN\nSIMULATION NOT AUTHORIZED\nINDEPENDENT FROZEN-CONTRACT AUDIT REQUIRED\n"
    )
    assert freeze.REQUIRED_NEXT_TASK in readme


def test_reserved_roots_absent_and_protected_boundaries_clean() -> None:
    assert freeze.verify_output_roots(ROOT) == {
        "evidence_freeze": False,
        "production_acceptance": False,
        "production_generation": False,
        "production_replay": False,
    }
    freeze.verify_repository_boundaries(ROOT)
    freeze.verify_repository_identity(
        ROOT,
        freeze.APPROVED_PROPOSAL_COMMIT,
        freeze.AUDIT_COMMIT,
        freeze.AUDIT_INVENTORY,
        freeze.BYTE_POLICY_COMMIT,
    )


def test_freeze_source_has_no_simulation_replay_or_training_entrypoint() -> None:
    source = Path(freeze.__file__).read_text(encoding="utf-8")
    prohibited = (
        "run_tier1_rollout(",
        "generate_run(",
        "generate_runs(",
        "replay_runs_read_only(",
        "train_rf_model(",
        "SatelliteGNN(",
    )
    assert not any(value in source for value in prohibited)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_proposal_commit", "0" * 40, "proposal commit"),
        ("expected_proposal_inventory", "0" * 64, "proposal inventory"),
        ("expected_audit_commit", "0" * 40, "audit"),
        ("expected_audit_inventory", "0" * 64, "audit"),
        ("expected_byte_policy_commit", "0" * 40, "byte-policy"),
    ],
)
def test_rejects_wrong_approved_identity(field: str, value: str, message: str) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=message):
        _validate(CONTRACT_ROOT, **kwargs)


@pytest.mark.parametrize("name", freeze.SOURCE_ARTIFACTS)
def test_rejects_every_source_artifact_mutation(tmp_path: Path, name: str) -> None:
    contract = _contract_copy(tmp_path)
    path = contract / "source_bundle" / name
    path.write_bytes(path.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="source-copy|artifact"):
        _validate(contract, enforce_repository=False, enforce_output_roots=False)


def test_rejects_missing_and_extra_source_artifacts(tmp_path: Path) -> None:
    missing = _contract_copy(tmp_path / "missing")
    (missing / "source_bundle" / freeze.SOURCE_ARTIFACTS[0]).unlink()
    with pytest.raises(ValueError, match="source-bundle"):
        _validate(missing, enforce_repository=False, enforce_output_roots=False)
    extra = _contract_copy(tmp_path / "extra")
    (extra / "source_bundle" / "extra.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="source-bundle"):
        _validate(extra, enforce_repository=False, enforce_output_roots=False)


@pytest.mark.parametrize(
    "name",
    [
        freeze.SPECIFICATION_NAME,
        freeze.INVENTORY_NAME,
        freeze.HASH_NAME,
        freeze.DECLARATION_NAME,
        freeze.README_NAME,
    ],
)
def test_rejects_every_derived_artifact_mutation(tmp_path: Path, name: str) -> None:
    contract = _contract_copy(tmp_path)
    path = contract / name
    path.write_bytes(path.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="derived artifact|hash record|bundle"):
        _validate(contract, enforce_repository=False, enforce_output_roots=False)


def test_rejects_extra_contract_artifact_and_directory(tmp_path: Path) -> None:
    extra_file = _contract_copy(tmp_path / "file")
    (extra_file / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or extra"):
        _validate(extra_file, enforce_repository=False, enforce_output_roots=False)
    extra_directory = _contract_copy(tmp_path / "directory")
    (extra_directory / "extra").mkdir()
    with pytest.raises(ValueError, match="missing or extra"):
        _validate(extra_directory, enforce_repository=False, enforce_output_roots=False)


@pytest.mark.parametrize("field", ["simulation_authorized", "production_authorized", "execution_authorized"])
def test_rejects_enabled_authorization(field: str) -> None:
    specification = freeze.read_json(CONTRACT_ROOT / freeze.SPECIFICATION_NAME)
    declaration = freeze.read_json(CONTRACT_ROOT / freeze.DECLARATION_NAME)
    specification[field] = True
    with pytest.raises(ValueError, match="authorization enabled"):
        freeze._validate_authorization(specification, declaration)
    specification[field] = False
    declaration[field] = True
    with pytest.raises(ValueError, match="authorization enabled"):
        freeze._validate_authorization(specification, declaration)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("contract_frozen", False, "marked unfrozen"),
        ("freeze_status", "FROZEN", "status mismatch"),
    ],
)
def test_rejects_wrong_freeze_state(field: str, value: object, message: str) -> None:
    specification = freeze.read_json(CONTRACT_ROOT / freeze.SPECIFICATION_NAME)
    declaration = freeze.read_json(CONTRACT_ROOT / freeze.DECLARATION_NAME)
    specification[field] = value
    with pytest.raises(ValueError, match=message):
        freeze._validate_authorization(specification, declaration)


def test_rejects_audit_inventory_crlf_conversion_and_bound_file_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = tmp_path / "audit"
    shutil.copytree(AUDIT_ROOT, audit)
    monkeypatch.setattr(freeze, "AUDIT_ROOT_RELATIVE", audit)
    inventory = audit / "audit_inventory.json"
    inventory.write_bytes(inventory.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="byte length|SHA-256|CRLF"):
        freeze.verify_approved_audit_bundle(ROOT)
    shutil.rmtree(audit)
    shutil.copytree(AUDIT_ROOT, audit)
    bound = audit / "audit_findings.json"
    bound.write_bytes(bound.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="artifact byte mismatch"):
        freeze.verify_approved_audit_bundle(ROOT)


def test_rejects_missing_or_extra_audit_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit = tmp_path / "audit"
    shutil.copytree(AUDIT_ROOT, audit)
    monkeypatch.setattr(freeze, "AUDIT_ROOT_RELATIVE", audit)
    (audit / "audit_findings.json").unlink()
    with pytest.raises(ValueError, match="missing or extra"):
        freeze.verify_approved_audit_bundle(ROOT)
    shutil.rmtree(audit)
    shutil.copytree(AUDIT_ROOT, audit)
    (audit / "extra.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="missing or extra"):
        freeze.verify_approved_audit_bundle(ROOT)


@pytest.mark.parametrize(
    ("artifact", "mutation", "message"),
    [
        (
            "stage_a_contract_proposal.json",
            lambda value: value["threshold_definition"].update({"value": "0.81"}),
            "threshold or margin",
        ),
        (
            "stage_a_contract_proposal.json",
            lambda value: value["boundary_definitions"].update({"endpoint_behavior": "changed"}),
            "observed-boundary",
        ),
        (
            "stage_a_contract_proposal.json",
            lambda value: value["distinct_margin_definition"].update({"rounding_mode": "ROUND_UP"}),
            "quantization",
        ),
        (
            "stage_a_contract_proposal.json",
            lambda value: value["final_corpus_membership"].update({"stage_a_validation": "INCLUDED"}),
            "final-corpus",
        ),
        (
            "stage_a_contract_proposal.json",
            lambda value: value["stage_b_adaptation_boundary"].update({"prohibited_source": "none"}),
            "adaptation boundary",
        ),
        (
            "stage_a_partition_manifest.json",
            lambda value: value["partitions"]["sealed_holdout"]["prohibited_uses"].remove("Select Stage B seeds"),
            "leakage",
        ),
        (
            "stage_a_near_neighbor_policy.json",
            lambda value: value["pending_scientific_reviews"].append("pending"),
            "review or exception",
        ),
    ],
)
def test_rejects_governance_mutation(
    tmp_path: Path,
    artifact: str,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    proposal = _proposal_copy(tmp_path)
    _rewrite_json(proposal / artifact, mutation)
    with pytest.raises(ValueError, match=message):
        freeze.build_frozen_specification(ROOT, proposal)


@pytest.mark.parametrize(
    ("artifact", "mutation", "message"),
    [
        (
            "stage_a_design_manifest.csv",
            lambda rows: rows.__setitem__(0, {**rows[0], "region": "boundary"}),
            "region allocation",
        ),
        (
            "stage_a_design_manifest.csv",
            lambda rows: rows.__setitem__(0, {**rows[0], "partition": "validation"}),
            "partition allocation",
        ),
        (
            "stage_a_run_manifest.csv",
            lambda rows: rows.__setitem__(0, {**rows[0], "global_run_id": "499"}),
            "global run identity",
        ),
        (
            "stage_a_run_manifest.csv",
            lambda rows: rows.pop(),
            "cardinality",
        ),
        (
            "stage_a_seed_manifest.csv",
            lambda rows: rows.__setitem__(0, {**rows[0], "run_key": rows[1]["run_key"]}),
            "seed identity",
        ),
    ],
)
def test_rejects_count_allocation_and_identity_mutation(
    tmp_path: Path,
    artifact: str,
    mutation: Callable[[list[dict[str, str]]], None],
    message: str,
) -> None:
    proposal = _proposal_copy(tmp_path)
    _rewrite_csv(proposal / artifact, mutation)
    with pytest.raises(ValueError, match=message):
        freeze.build_frozen_specification(ROOT, proposal)


def test_rejects_reserved_root_existence_and_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setattr(
        freeze,
        "OUTPUT_ROOTS",
        {
            "production_generation": str(existing),
            "production_replay": str(tmp_path / "replay"),
            "production_acceptance": str(tmp_path / "acceptance"),
            "evidence_freeze": str(tmp_path / "freeze"),
        },
    )
    with pytest.raises(ValueError, match="exists"):
        freeze.verify_output_roots(ROOT)
    existing.rmdir()
    monkeypatch.setattr(
        freeze,
        "OUTPUT_ROOTS",
        {
            "production_generation": str(tmp_path / "output"),
            "production_replay": str(tmp_path / "output" / "child"),
            "production_acceptance": str(tmp_path / "acceptance"),
            "evidence_freeze": str(tmp_path / "freeze"),
        },
    )
    with pytest.raises(ValueError, match="overlaps"):
        freeze.verify_output_roots(ROOT)


def test_rejects_existing_contract_root_and_conflicting_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    original_git = freeze.git

    def qualified_branch(repo_root: Path, *args: str, check: bool = True) -> str:
        if args == ("branch", "--show-current"):
            return freeze.FREEZE_BRANCH
        return original_git(repo_root, *args, check=check)

    monkeypatch.setattr(freeze, "git", qualified_branch)
    with pytest.raises(FileExistsError, match="already exists"):
        freeze.create_frozen_contract(ROOT, existing)

    def conflicting_tag(repo_root: Path, *args: str, check: bool = True) -> str:
        if args == ("tag", "--list", freeze.TAG_NAME):
            return freeze.TAG_NAME
        return qualified_branch(repo_root, *args, check=check)

    monkeypatch.setattr(freeze, "git", conflicting_tag)
    with pytest.raises(ValueError, match="tag already exists"):
        freeze.create_frozen_contract(ROOT, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()
