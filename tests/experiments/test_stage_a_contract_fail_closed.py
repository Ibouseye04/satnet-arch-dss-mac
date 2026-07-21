from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from satnet.experiments.stage_a_contract.designs import build_design_rows, validate_design_rows
from satnet.experiments.stage_a_contract.proposal import (
    build_final_gate_feasibility,
    build_partition_manifest,
    build_run_rows,
    build_seed_rows,
    validate_contract_authorization,
    validate_holdout_policy,
    validate_output_roots,
    validate_run_rows,
    validate_seed_rows,
    write_proposal_artifacts,
)
from satnet.experiments.stage_a_contract.semantics import FINAL_GATES, OUTPUT_ROOTS

ROOT = Path(__file__).parents[2]


def test_rejects_duplicate_design_and_identity_collision() -> None:
    rows = build_design_rows()
    duplicate = deepcopy(rows)
    duplicate[1] = deepcopy(duplicate[0])
    duplicate[1]["design_id"] = "SA-D001"
    duplicate[1]["design_index"] = 1
    with pytest.raises(ValueError, match="duplicate parameter vector"):
        validate_design_rows(duplicate)
    collision = deepcopy(rows)
    collision[0]["design_id"] = "D000"
    with pytest.raises(ValueError, match="SA-D000 through SA-D029"):
        validate_design_rows(collision)


def test_rejects_incorrect_partition_and_region_counts() -> None:
    rows = build_design_rows()
    bad_partition = deepcopy(rows)
    bad_partition[0]["partition"] = "validation"
    with pytest.raises(ValueError, match="partition allocation"):
        validate_design_rows(bad_partition)
    bad_region = deepcopy(rows)
    bad_region[0]["region"] = "boundary"
    with pytest.raises(ValueError, match="region allocation"):
        validate_design_rows(bad_region)


def test_rejects_duplicate_run_and_run_identity_collision() -> None:
    designs = build_design_rows()
    rows = build_run_rows(designs)
    duplicate = deepcopy(rows)
    duplicate[1]["run_key"] = duplicate[0]["run_key"]
    with pytest.raises(ValueError, match="run keys"):
        validate_run_rows(duplicate, designs)
    collision = deepcopy(rows)
    collision[0]["global_run_id"] = 499
    with pytest.raises(ValueError, match="500 through 649"):
        validate_run_rows(collision, designs)


def test_rejects_seed_substitution() -> None:
    rows = build_seed_rows(build_run_rows(build_design_rows()))
    rows[0]["satellite_failure_seed"] += 1
    with pytest.raises(ValueError, match="substitution"):
        validate_seed_rows(rows)


def test_rejects_holdout_leakage_and_early_unsealing() -> None:
    designs = build_design_rows()
    partition = build_partition_manifest(designs, build_run_rows(designs))
    leaked = deepcopy(partition)
    leaked["partitions"]["sealed_holdout"]["prohibited_uses"].remove("Select Stage B seeds")
    with pytest.raises(ValueError, match="leakage"):
        validate_holdout_policy(leaked, stage_b_contract_hash="a" * 64)
    with pytest.raises(ValueError, match="contract hash"):
        validate_holdout_policy(partition)


def test_rejects_output_root_overlap_and_frozen_evidence_write(tmp_path: Path) -> None:
    roots = dict(OUTPUT_ROOTS)
    roots["production_replay"] = roots["production_generation"] + "\\child"
    with pytest.raises(ValueError, match="overlap"):
        validate_output_roots(roots, require_absent=False)
    with pytest.raises(ValueError, match="frozen evidence"):
        write_proposal_artifacts(ROOT, Path("C:/Users/johns/satnet-final-production-20260720/correction"))


def test_rejects_frozen_or_authorized_proposal() -> None:
    with pytest.raises(ValueError, match="NOT_FROZEN"):
        validate_contract_authorization({"proposal_status": "FROZEN", "simulation_authorized": False})
    with pytest.raises(ValueError, match="authorization"):
        validate_contract_authorization({"proposal_status": "NOT_FROZEN", "simulation_authorized": True})


def test_rejects_infeasible_final_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(FINAL_GATES["minimum_non_breach_runs"], "validation", 151)
    with pytest.raises(ValueError, match="infeasible"):
        build_final_gate_feasibility(ROOT / "artifacts/final_integrated_dataset_class_support_audit/audit_boundary_reproduction.csv")
