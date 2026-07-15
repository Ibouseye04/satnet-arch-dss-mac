from __future__ import annotations

import json

import pandas as pd
import pytest


def _write_dataset(path, n: int = 12) -> None:
    rows = []
    for i in range(n):
        rows.append(
            {
                "run_id": i,
                "config_hash": f"cfg-{i}",
                "partition_any": i % 2,
                "gcc_frac_min_original": float(i) / max(n - 1, 1),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_split_manifest_disjoint_union_and_rerun_stability(tmp_path) -> None:
    from satnet.utils.split_manifest import build_split_manifest, validate_manifest_for_dataset

    csv_path = tmp_path / "tier1_design_runs.csv"
    _write_dataset(csv_path)

    manifest_a = build_split_manifest(
        csv_path=csv_path,
        target_name="partition_any",
        task_type="classification",
        seed=42,
        test_size=0.25,
        val_size=0.25,
    )
    manifest_b = build_split_manifest(
        csv_path=csv_path,
        target_name="partition_any",
        task_type="classification",
        seed=42,
        test_size=0.25,
        val_size=0.25,
    )

    assert manifest_a["train_indices"] == manifest_b["train_indices"]
    splits = validate_manifest_for_dataset(
        manifest_a,
        csv_path=csv_path,
        target_name="partition_any",
    )
    all_indices = splits["train"] + splits["val"] + splits["test"]
    assert len(all_indices) == len(set(all_indices))
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])
    assert set(all_indices) == set(manifest_a["active_indices"])


def test_split_manifest_rejects_dataset_hash_mismatch(tmp_path) -> None:
    from satnet.utils.split_manifest import build_split_manifest, validate_manifest_for_dataset

    csv_path = tmp_path / "tier1_design_runs.csv"
    _write_dataset(csv_path)
    manifest = build_split_manifest(
        csv_path=csv_path,
        target_name="gcc_frac_min_original",
        task_type="regression",
        seed=42,
        test_size=0.25,
        val_size=0.25,
    )
    df = pd.read_csv(csv_path)
    df.loc[0, "gcc_frac_min_original"] = 0.12345
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="dataset hash"):
        validate_manifest_for_dataset(
            manifest,
            csv_path=csv_path,
            target_name="gcc_frac_min_original",
        )


def test_split_manifest_write_and_read_roundtrip(tmp_path) -> None:
    from satnet.utils.split_manifest import build_split_manifest, read_split_manifest, write_split_manifest

    csv_path = tmp_path / "tier1_design_runs.csv"
    _write_dataset(csv_path)
    manifest = build_split_manifest(
        csv_path=csv_path,
        target_name="partition_any",
        task_type="classification",
        seed=7,
        test_size=0.25,
        val_size=0.25,
    )
    path = tmp_path / "split.json"
    write_split_manifest(manifest, path, overwrite=True)

    assert read_split_manifest(path)["train_indices"] == manifest["train_indices"]
    assert json.loads(path.read_text())["target_name"] == "partition_any"
