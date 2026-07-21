from __future__ import annotations

from copy import deepcopy
import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/validation/audit_stage_a_frozen_contract_v1.py"
SPEC = importlib.util.spec_from_file_location("stage_a_frozen_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
CONTRACT_ROOT = ROOT / audit.CONTRACT_ROOT_RELATIVE
PROPOSAL_ROOT = ROOT / audit.PROPOSAL_ROOT_RELATIVE
WINDOWS_ROOT = Path(r"C:\Users\johns\satnet-stage-a-frozen-contract-windows-audit-20260721")


def contract_copy(tmp_path: Path) -> Path:
    path = tmp_path / "contract"
    shutil.copytree(CONTRACT_ROOT, path)
    return path


def rewrite_json(path: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    value = json.loads(path.read_bytes())
    mutation(value)
    path.write_bytes(audit.canonical_json_bytes(value))


def rewrite_csv(path: Path, mutation: Callable[[list[dict[str, str]]], None]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mutation(rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(stream.getvalue().encode("utf-8"))


def test_all_independent_positive_validators() -> None:
    assert audit.validate_tag(ROOT)["annotation_result"] == "PASS"
    assert audit.validate_byte_policy(ROOT)["pass"] is True
    assert audit.validate_audit_bundle(ROOT)["tracked_file_count"] == 17
    assert len(audit.validate_source_bundle(ROOT)) == 11
    inventory, contract_hash = audit.validate_frozen_inventory(ROOT)
    assert len(inventory) == 12
    assert contract_hash["authoritative_contract_hash"] == audit.CONTRACT_HASH
    assert audit.validate_specification_and_declaration(ROOT)[0]["pass"] is True
    assert audit.validate_readme(ROOT)["pass"] is True
    designs, runs, seeds, summary = audit.validate_manifests(ROOT)
    assert (len(designs), len(runs), len(seeds)) == (30, 150, 150)
    assert summary["duplicate_count"] == summary["original_collision_count"] == 0
    assert audit.validate_governance(ROOT)["criterion_count"] == 22
    assert audit.validate_neighbors(ROOT)["cross_partition_pairs_below_0_10"] == 0
    assert audit.validate_authorization(ROOT)["execution_authorized"] is False
    assert audit.validate_freeze_tooling(ROOT)["can_execute_simulations"] is False
    assert audit.validate_protected_science(ROOT)["pass"] is True


def test_exact_freeze_tag_and_windows_checkout_identities() -> None:
    tag = audit.validate_tag(ROOT)
    windows = audit.validate_windows_checkout(ROOT, WINDOWS_ROOT)
    assert tag["tag_type"] == "tag"
    assert tag["tag_target"] == audit.FREEZE_HEAD
    assert tag["conflicting_tags"] == []
    assert windows["head"] == audit.FREEZE_HEAD
    assert windows["core_autocrlf"] is True
    assert windows["clean"] is True
    assert windows["audit_inventory_byte_length"] == 3691
    assert windows["audit_inventory_lf"] == 107
    assert windows["audit_inventory_crlf"] == 0
    assert windows["frozen_contract_hash"] == audit.CONTRACT_HASH
    assert len(windows["source_bundle"]) == 11


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", "0.000000"), ("-0", "0.000000"),
        ("0.0000005", "0.000000"), ("0.0000015", "0.000002"),
        ("0.000000499999", "0.000000"), ("0.000000500001", "0.000001"),
        ("-0.0000005", "0.000000"), ("-0.0000015", "-0.000002"),
        ("-0.000000499999", "0.000000"), ("-0.000000500001", "-0.000001"),
    ],
)
def test_margin_quantization_vectors(value: str, expected: str) -> None:
    assert audit.canonical_margin(value) == expected


@pytest.mark.parametrize("non_breach", range(6))
def test_every_majority_and_mixed_design_count(non_breach: int) -> None:
    result = audit.classify_design(["0"] * non_breach + ["-0.1"] * (5 - non_breach))
    assert result["majority_class"] == ("non_breach_majority" if non_breach >= 3 else "breach_majority")
    assert result["mixed_design"] is (0 < non_breach < 5)
    assert result["non_breach_realization_count"] == non_breach


def test_observed_boundary_endpoint_and_region_separation() -> None:
    assert audit.classify_design(["-0.1", "0", "0.2", "0.3", "0.4"])["observed_boundary_design"] is True
    assert audit.classify_design(["0.05", "0.05", "0.2", "0.3", "0.4"])["observed_boundary_design"] is True
    assert audit.classify_design(["0.051", "0.051", "0.2", "0.3", "0.4"])["observed_boundary_design"] is False
    governance = audit.validate_governance(ROOT)
    assert governance["boundary_definition"] == "PASS"
    assert governance["final_corpus_membership"] == "PASS"
    assert governance["stage_b_adaptation_boundary"] == "PASS"


def test_rejects_crlf_and_noncanonical_json(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')
    with pytest.raises(ValueError, match="Noncanonical JSON"):
        audit.read_json_bytes(path)
    path.write_bytes(b'{"value":1}\n')
    with pytest.raises(ValueError, match="Noncanonical JSON"):
        audit.read_json_bytes(path)


@pytest.mark.parametrize(
    ("name", "mutation", "message"),
    [
        ("stage_a_frozen_contract_inventory.json", lambda value: value["artifacts"].append(deepcopy(value["artifacts"][0])), "Authoritative frozen contract hash|cardinality|ordering"),
        ("stage_a_frozen_contract_inventory.json", lambda value: value["artifacts"][0].update({"sha256": "0" * 64}), "Authoritative frozen contract hash"),
    ],
)
def test_rejects_inventory_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, mutation: Callable[[dict[str, Any]], None], message: str) -> None:
    contract = contract_copy(tmp_path)
    rewrite_json(contract / name, mutation)
    monkeypatch.setattr(audit, "CONTRACT_ROOT_RELATIVE", contract)
    with pytest.raises(ValueError, match=message):
        audit.validate_frozen_inventory(ROOT)


@pytest.mark.parametrize("name", audit.SOURCE_ARTIFACTS)
def test_rejects_each_mutated_frozen_source_copy(tmp_path: Path, name: str) -> None:
    contract = contract_copy(tmp_path)
    source = contract / "source_bundle" / name
    source.write_bytes(source.read_bytes() + b"mutation")
    proposal = PROPOSAL_ROOT / name
    with pytest.raises(ValueError, match="Byte identity mismatch"):
        audit.require_byte_identical(name, proposal.read_bytes(), source.read_bytes())


def test_rejects_missing_source_and_proposal_mutation(tmp_path: Path) -> None:
    contract = contract_copy(tmp_path)
    missing = contract / "source_bundle" / audit.SOURCE_ARTIFACTS[0]
    missing.unlink()
    assert not missing.exists()
    proposal = tmp_path / "proposal.json"
    shutil.copyfile(PROPOSAL_ROOT / "stage_a_contract_proposal.json", proposal)
    proposal.write_bytes(proposal.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="Byte identity mismatch"):
        audit.require_byte_identical("proposal", proposal.read_bytes(), (CONTRACT_ROOT / "source_bundle/stage_a_contract_proposal.json").read_bytes())


@pytest.mark.parametrize(
    ("artifact", "mutation", "message"),
    [
        ("stage_a_design_manifest.csv", lambda rows: rows.pop(), "design identity or cardinality"),
        ("stage_a_design_manifest.csv", lambda rows: rows.__setitem__(1, dict(rows[0])), "design identity or cardinality|duplicate"),
        ("stage_a_design_manifest.csv", lambda rows: rows[0].update({"region": "boundary"}), "region or partition"),
        ("stage_a_design_manifest.csv", lambda rows: rows[0].update({"partition": "validation"}), "region or partition"),
        ("stage_a_design_manifest.csv", lambda rows: rows[0].update({"altitude_km": "9999"}), "design validation"),
        ("stage_a_run_manifest.csv", lambda rows: rows.pop(), "run cardinality"),
        ("stage_a_run_manifest.csv", lambda rows: rows[1].update({"run_key": rows[0]["run_key"]}), "duplicate run"),
        ("stage_a_run_manifest.csv", lambda rows: rows[0].update({"partition": "validation"}), "run validation"),
        ("stage_a_seed_manifest.csv", lambda rows: rows.pop(), "seed identity cardinality"),
        ("stage_a_seed_manifest.csv", lambda rows: rows[0].update({"satellite_failure_seed": "1"}), "seed reproduction"),
    ],
)
def test_rejects_manifest_mutations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str, mutation: Callable[[list[dict[str, str]]], None], message: str) -> None:
    contract = contract_copy(tmp_path)
    rewrite_csv(contract / "source_bundle" / artifact, mutation)
    monkeypatch.setattr(audit, "CONTRACT_ROOT_RELATIVE", contract)
    with pytest.raises(ValueError, match=message):
        audit.validate_manifests(ROOT)


@pytest.mark.parametrize(
    ("artifact", "mutation", "message"),
    [
        ("stage_a_contract_proposal.json", lambda value: value["threshold_definition"].update({"value": "0.81"}), "threshold or margin"),
        ("stage_a_contract_proposal.json", lambda value: value["boundary_definitions"].update({"endpoint_behavior": "changed"}), "boundary definition"),
        ("stage_a_contract_proposal.json", lambda value: value["class_support_definitions"].update({"mixed_design": "counts twice"}), "majority or mixed-design|final-corpus|adaptation|discovery|holdout|boundary"),
        ("stage_a_contract_proposal.json", lambda value: value["distinct_margin_definition"].update({"rounding_mode": "ROUND_UP"}), "quantization"),
        ("stage_a_contract_proposal.json", lambda value: value["final_corpus_membership"].update({"stage_a_validation": "INCLUDED"}), "final-corpus"),
        ("stage_a_contract_proposal.json", lambda value: value["stage_b_adaptation_boundary"].update({"prohibited_source": "none"}), "adaptation boundary"),
        ("stage_a_partition_manifest.json", lambda value: value["partitions"]["sealed_holdout"]["prohibited_uses"].remove("Select Stage B seeds"), "sealed-holdout"),
        ("stage_a_discovery_criteria.json", lambda value: value["decision_states"].pop(), "criteria or decision-state"),
        ("stage_a_discovery_criteria.json", lambda value: value["sealed_holdout_confirmation_criteria"][0].update({"can_influence_stage_b": True}), "criteria or decision-state"),
    ],
)
def test_rejects_scientific_and_governance_mutations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str, mutation: Callable[[dict[str, Any]], None], message: str) -> None:
    contract = contract_copy(tmp_path)
    rewrite_json(contract / "source_bundle" / artifact, mutation)
    monkeypatch.setattr(audit, "CONTRACT_ROOT_RELATIVE", contract)
    with pytest.raises(ValueError, match=message):
        audit.validate_governance(ROOT)


@pytest.mark.parametrize("field", ["simulation_authorized", "production_authorized", "execution_authorized"])
def test_rejects_enabled_authorizations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    contract = contract_copy(tmp_path)
    spec = contract / "stage_a_frozen_contract_specification.json"
    rewrite_json(spec, lambda value: value.update({field: True}))
    monkeypatch.setattr(audit, "CONTRACT_ROOT_RELATIVE", contract)
    with pytest.raises(ValueError, match="authorization"):
        audit.validate_authorization(ROOT)


@pytest.mark.parametrize(("field", "value"), [("contract_frozen", False), ("freeze_status", "FROZEN")])
def test_rejects_wrong_freeze_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object) -> None:
    contract = contract_copy(tmp_path)
    spec = contract / "stage_a_frozen_contract_specification.json"
    rewrite_json(spec, lambda item: item.update({field: value}))
    monkeypatch.setattr(audit, "CONTRACT_ROOT_RELATIVE", contract)
    with pytest.raises(ValueError, match="envelope"):
        audit.validate_authorization(ROOT)


def test_rejects_existing_and_overlapping_reserved_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setattr(audit, "RESERVED_ROOTS", {"a": existing, "b": tmp_path / "b", "c": tmp_path / "c", "d": tmp_path / "d"})
    monkeypatch.setattr(audit, "FROZEN_ROOTS", ())
    with pytest.raises(ValueError, match="output-root"):
        audit.validate_output_roots(ROOT, [])
    existing.rmdir()
    monkeypatch.setattr(audit, "RESERVED_ROOTS", {"a": tmp_path / "x", "b": tmp_path / "x/child", "c": tmp_path / "c", "d": tmp_path / "d"})
    with pytest.raises(ValueError, match="output-root"):
        audit.validate_output_roots(ROOT, [])


@pytest.mark.parametrize("mismatch", ["target", "annotation", "type"])
def test_rejects_tag_mismatches(monkeypatch: pytest.MonkeyPatch, mismatch: str) -> None:
    real_git = audit.git
    def fake_git(repo_root: Path, *arguments: str, binary: bool = False, check: bool = True) -> str | bytes:
        if mismatch == "type" and arguments == ("cat-file", "-t", audit.TAG_NAME):
            return "commit"
        if mismatch == "target" and arguments == ("rev-list", "-n", "1", audit.TAG_NAME):
            return "0" * 40
        if mismatch == "annotation" and arguments == ("cat-file", "-p", audit.TAG_NAME):
            return b"object x\ntype commit\ntag x\n\nwrong\n"
        return real_git(repo_root, *arguments, binary=binary, check=check)
    monkeypatch.setattr(audit, "git", fake_git)
    with pytest.raises(ValueError, match="tag"):
        audit.validate_tag(ROOT)


def test_rejects_freeze_tooling_simulation_call(tmp_path: Path) -> None:
    path = tmp_path / "src/satnet/experiments/stage_a_contract"
    path.mkdir(parents=True)
    source = (ROOT / "src/satnet/experiments/stage_a_contract/freeze.py").read_text(encoding="utf-8")
    (path / "freeze.py").write_text(source + "\nrun_tier1_rollout()\n", encoding="utf-8")
    with pytest.raises(ValueError, match="simulation"):
        audit.validate_freeze_tooling(tmp_path)


def test_rejects_windows_byte_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clone = tmp_path / "windows"
    shutil.copytree(WINDOWS_ROOT, clone)
    inventory = clone / audit.READINESS_ROOT_RELATIVE / "audit_inventory.json"
    inventory.write_bytes(inventory.read_bytes().replace(b"\n", b"\r\n"))
    real_git = audit.git
    def fake_git(repo_root: Path, *arguments: str, binary: bool = False, check: bool = True) -> str | bytes:
        if repo_root == clone:
            if arguments == ("rev-parse", "HEAD"): return audit.FREEZE_HEAD
            if arguments == ("config", "--get", "core.autocrlf"): return "true"
            if arguments == ("status", "--short"): return ""
        return real_git(repo_root, *arguments, binary=binary, check=check)
    monkeypatch.setattr(audit, "git", fake_git)
    with pytest.raises(ValueError, match="Windows checkout changed"):
        audit.validate_windows_checkout(ROOT, clone)


def test_output_inventory_is_deterministic_and_self_excluding(tmp_path: Path) -> None:
    outputs = {"audit_findings.json": {"pass": True}, "audit_rows.csv": [{"id": 1, "pass": True}]}
    first = audit.write_outputs(tmp_path, outputs)
    first_bytes = (tmp_path / "audit_inventory.json").read_bytes()
    second = audit.write_outputs(tmp_path, outputs)
    assert first == second
    assert first_bytes == (tmp_path / "audit_inventory.json").read_bytes()
    assert all(record["relative_path"] != "audit_inventory.json" for record in first["artifacts"])


def test_full_fail_closed_surface_is_covered() -> None:
    required = {
        "crlf", "proposal source", "frozen source", "missing source", "inventory", "proposal inventory",
        "seed", "audit inventory", "proposal commit", "audit commit", "design", "run", "region",
        "partition", "holdout", "near-neighbor", "threshold", "margin", "boundary", "mixed-design",
        "quantization", "reserved", "authorization", "contract_frozen", "freeze status", "tag",
        "windows", "protected science", "simulation evidence", "freeze tooling",
    }
    suite_text = Path(__file__).read_text(encoding="utf-8").lower() + (ROOT / "tests/experiments/test_stage_a_contract_freeze.py").read_text(encoding="utf-8").lower() + (ROOT / "tests/experiments/test_stage_a_contract_fail_closed.py").read_text(encoding="utf-8").lower()
    assert all(term in suite_text for term in required)
