from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path


def test_candidate_mode_emits_stable_result_paths(monkeypatch, capsys, tmp_path) -> None:
    import scripts.run_autoresearch_rf as run_script

    experiments_root = tmp_path / "experiments" / "autoresearch"
    result = {
        "status": "succeeded",
        "experiment_id": "rf-20260311T000000Z-abcd1234",
        "promotion_decision": "promoted",
        "artifact_paths": {
            "summary": str(experiments_root / "rf-20260311T000000Z-abcd1234" / "summary.json"),
            "incumbent": str(experiments_root / "incumbent.json"),
        },
    }
    args = Namespace(
        mode="candidate",
        config=None,
        dataset_path=None,
        target_name="gcc_frac_min",
        feature_mode="tier1_full",
        custom_feature_columns="",
        n_estimators=300,
        max_depth=None,
        seed=42,
        fidelity_tier="screening",
        experiments_root=str(experiments_root),
        max_candidates=None,
        notes="",
        no_save_model=True,
        candidate_path=str(tmp_path / "candidate.json"),
        policy_path=str(tmp_path / "mutation_policy.json"),
        source_experiment_id=None,
    )

    monkeypatch.setattr(run_script, "parse_args", lambda: args)
    monkeypatch.setattr(run_script, "run_agent_candidate", lambda **_: result)

    run_script.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "succeeded"
    assert payload["experiment_id"] == result["experiment_id"]
    assert payload["summary_path"] == result["artifact_paths"]["summary"]
    assert payload["incumbent_path"] == result["artifact_paths"]["incumbent"]
    assert payload["last_run_path"] == str(Path(args.experiments_root) / "last_run.json")


def test_confirmatory_mode_requires_source_experiment_id(monkeypatch, tmp_path) -> None:
    import scripts.run_autoresearch_rf as run_script

    args = Namespace(
        mode="confirmatory",
        config=None,
        dataset_path=None,
        target_name="gcc_frac_min",
        feature_mode="tier1_full",
        custom_feature_columns="",
        n_estimators=300,
        max_depth=None,
        seed=42,
        fidelity_tier="screening",
        experiments_root=str(tmp_path / "experiments" / "autoresearch"),
        max_candidates=None,
        notes="",
        no_save_model=True,
        candidate_path=str(tmp_path / "candidate.json"),
        policy_path=str(tmp_path / "mutation_policy.json"),
        source_experiment_id=None,
    )

    monkeypatch.setattr(run_script, "parse_args", lambda: args)

    try:
        run_script.main()
    except SystemExit as exc:
        assert "--source-experiment-id is required" in str(exc)
    else:
        raise AssertionError("Expected SystemExit when confirmatory mode has no source experiment id")

    last_run_path = Path(args.experiments_root) / "last_run.json"
    payload = json.loads(last_run_path.read_text())

    assert payload["schema_version"] == "satnet_rf_autoresearch_last_run_v1"
    assert payload["status"] == "failed"
    assert payload["mode"] == "confirmatory"
    assert payload["experiment_id"] is None
    assert payload["summary_path"] is None
    assert payload["artifact_paths"]["summary"] is None
    assert payload["fidelity_tier"] == "confirmatory"
    assert payload["error_type"] == "SystemExit"
