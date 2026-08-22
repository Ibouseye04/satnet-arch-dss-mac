from __future__ import annotations

from pathlib import Path

import pytest

from satnet.experiments.production_profile import FINAL_ADAPTIVE_PRODUCTION_PROFILE

import sys

ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from export_final_ml_datasets import (  # noqa: E402
    ADAPTIVE_V2_EXPORT_PROFILE,
    CONTRACT_SPEC_HASH,
    HISTORICAL_FIXED_EXPORT_PROFILE,
    ML_CONTRACT_BUNDLE_HASH,
    PRODUCTION_CONTRACT_BUNDLE_HASH,
    Exporter,
    RunEvidence,
    read_json,
    sha256_file,
)


EXPECTED_SOURCE_SHA = "346b3ff1670237645acf4836283adbbdc359093a"
EXPECTED_SPEC_HASH = "23c5fffc10849c3bc3ea027251ac3e5ad4c96f0eea85edf1e8deab079cb0871e"
EXPECTED_BUNDLE_HASH = "da3c73711b1d60635afcceee8bda0a60d1379e492e1ad0d588d5a0c25e10abe3"


def _exporter(tmp_path: Path, profile) -> Exporter:
    return Exporter(
        production_root=profile.source_root,
        replay_root=profile.replay_root,
        acceptance_root=Path("."),
        audit_root=Path("."),
        ml_contract_root=profile.schema_root,
        contract_root=profile.contract_root,
        output_root=tmp_path / profile.profile_id,
        profile=profile,
    )


def test_adaptive_binding_is_distinct_and_exact() -> None:
    assert ADAPTIVE_V2_EXPORT_PROFILE.adaptive is True
    assert ADAPTIVE_V2_EXPORT_PROFILE.source_lineage_sha == EXPECTED_SOURCE_SHA
    assert ADAPTIVE_V2_EXPORT_PROFILE.source_contract_spec_hash == EXPECTED_SPEC_HASH
    assert ADAPTIVE_V2_EXPORT_PROFILE.source_contract_bundle_hash == EXPECTED_BUNDLE_HASH
    assert ADAPTIVE_V2_EXPORT_PROFILE.source_root.name == "satnet-10k-final-production-v2-adaptive"
    assert ADAPTIVE_V2_EXPORT_PROFILE.replay_root.name == "satnet-10k-final-production-v2-adaptive-replay"
    assert FINAL_ADAPTIVE_PRODUCTION_PROFILE.isl_policy == "grid_adaptive"
    assert FINAL_ADAPTIVE_PRODUCTION_PROFILE.adjacent_search_k == 1
    assert FINAL_ADAPTIVE_PRODUCTION_PROFILE.max_inter_plane_links_per_sat == 1


def test_adaptive_binding_rejects_fixed_source_before_materialization(tmp_path: Path) -> None:
    exporter = _exporter(tmp_path, ADAPTIVE_V2_EXPORT_PROFILE)
    exporter.production_root = HISTORICAL_FIXED_EXPORT_PROFILE.source_root.resolve()
    with pytest.raises(ValueError, match="Adaptive source root"):
        exporter.verify_frozen_contract()
    assert not exporter.output_root.exists()


def test_fixed_binding_constants_and_default_profile_remain_historical(tmp_path: Path) -> None:
    exporter = _exporter(tmp_path, HISTORICAL_FIXED_EXPORT_PROFILE)
    assert exporter.profile is HISTORICAL_FIXED_EXPORT_PROFILE
    assert CONTRACT_SPEC_HASH == HISTORICAL_FIXED_EXPORT_PROFILE.source_contract_spec_hash
    assert ML_CONTRACT_BUNDLE_HASH == HISTORICAL_FIXED_EXPORT_PROFILE.export_contract_bundle_hash
    assert PRODUCTION_CONTRACT_BUNDLE_HASH == HISTORICAL_FIXED_EXPORT_PROFILE.source_contract_bundle_hash
    assert exporter._export_provenance() == {
        "tooling_sha": "d0515088cf3fca06a6aa2d47059269089dcb10a7",
        "contract_spec_hash": CONTRACT_SPEC_HASH,
        "ml_contract_bundle_hash": ML_CONTRACT_BUNDLE_HASH,
        "split_candidate": 3958,
    }


def test_fixed_transformation_matches_historical_first_row_and_graph_sequence(tmp_path: Path) -> None:
    source = Path(r"C:\\Users\\johns\\external\\satnet-10k-production-generation") / "run_0000"
    run = read_json(source / "input" / "run_record.json")
    design = read_json(source / "input" / "design_record.json")
    target = read_json(source / "targets" / "target.json")
    item = RunEvidence(
        run_id=0,
        run=run,
        design=design,
        target=target,
        result_sha256="",
        inventory_sha256="",
        target_sha256=sha256_file(source / "targets" / "target.json"),
        graph_sha256=sha256_file(source / "g3" / "integrated_graphs.jsonl"),
        graph_relative_path="g3/integrated_graphs.jsonl",
        satellite_sha256="",
        source_run_dir=source,
    )
    exporter = _exporter(tmp_path, HISTORICAL_FIXED_EXPORT_PROFILE)
    assert exporter._rf_row(item, ("num_planes", "sats_per_plane", "altitude_km", "inclination_deg", "satellite_node_failure_probability", "satellite_edge_failure_probability"), ("space_threshold_breach_any",)) == {
        "run_id": "0", "run_key": "D0000-R00", "design_id": "D0000", "realization_id": "R00", "split": "train",
        "num_planes": "6", "sats_per_plane": "8", "altitude_km": "1200", "inclination_deg": "98",
        "satellite_node_failure_probability": "0", "satellite_edge_failure_probability": "0", "space_threshold_breach_any": "false",
    }
    header, arrays, metadata = exporter._sequence_for_run(item)
    assert metadata["source_graph_path"] == "production_generation/run_0000/g3/integrated_graphs.jsonl"
    assert header["sequence"]["contract_spec_hash"] == CONTRACT_SPEC_HASH
    assert header["sequence_length"] == 11
    assert arrays["node_features"].shape[1] == 3
    assert arrays["edge_attr"].shape[1] == 4
    assert metadata["directed_edge_counts"] == [184, 184, 184, 184, 184, 184, 188, 188, 188, 188, 188]


def test_fixed_binding_rejects_adaptive_source_before_materialization(tmp_path: Path) -> None:
    exporter = _exporter(tmp_path, HISTORICAL_FIXED_EXPORT_PROFILE)
    exporter.production_root = ADAPTIVE_V2_EXPORT_PROFILE.source_root.resolve()
    with pytest.raises(ValueError, match="Historical fixed source root"):
        exporter.verify_frozen_contract()
    assert not exporter.output_root.exists()


def test_adaptive_schema_contract_preserves_scientific_feature_meanings() -> None:
    schema_root = ADAPTIVE_V2_EXPORT_PROFILE.schema_root
    assert (schema_root / "rf_space_classification_schema.json").read_text(encoding="utf-8").find('"field":"space_threshold_breach_any"') >= 0
    assert (schema_root / "rf_space_regression_schema.json").read_text(encoding="utf-8").find('"field":"space_gcc_fraction_original_min"') >= 0
    assert (schema_root / "tgnn_space_classification_schema.json").read_text(encoding="utf-8").find('"name":"plane_idx_normalized"') >= 0
    assert (schema_root / "tgnn_space_regression_schema.json").read_text(encoding="utf-8").find('"name":"link_mode_binary"') >= 0
