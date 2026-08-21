from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from satnet.dss import analysis_service
from satnet.dss.schemas import DSSArchitectureRequest
from tests.dss.test_schemas_and_policy import valid_request

FIXTURE = Path(__file__).parents[1] / "fixtures" / "ground_segment" / "synthetic_ground_station_catalog.csv"


def test_threshold_independence_and_exact_five_orchestration(monkeypatch) -> None:
    calls = []
    prediction_calls = []
    predictions = [0.84, 0.81, 0.76, 0.83, 0.86]

    def fake_build(architecture, seed_set):
        calls.append(seed_set)
        return SimpleNamespace(
            seeds=seed_set,
            graphs=[],
            graph_snapshots=[SimpleNamespace(graph_hash=f"graph-{seed_set.realization_index}")],
            rollout_config=SimpleNamespace(config_hash=lambda: f"config-{seed_set.realization_index}"),
        )

    def fake_predict(model, graphs, *, num_planes, sats_per_plane):
        index = len(prediction_calls)
        prediction_calls.append(index)
        return predictions[index], [(3, 4, 2)] * 11

    fake_ground = SimpleNamespace(
        mean_ground_service_fraction=0.75,
        minimum_ground_service_fraction=0.50,
        mean_overall_service_fraction=0.70,
        minimum_overall_service_fraction=0.40,
        catalog_hash="catalog",
        ground_design_hash="design",
        visibility_policy_hash="visibility",
        service_policy_hash="service",
        ground_realization_hash="g0;g1;g2;g3;g4",
    )
    monkeypatch.setattr(analysis_service, "load_frozen_tgnn", lambda _: (object(), "c" * 64))
    monkeypatch.setattr(analysis_service, "build_scenario", fake_build)
    monkeypatch.setattr(analysis_service, "predict_graph_sequence", fake_predict)
    monkeypatch.setattr(analysis_service, "calculate_ground_context", lambda *args, **kwargs: fake_ground)

    first = analysis_service.analyze_architecture(
        valid_request(required_minimum_connectivity=0.80),
        checkpoint_path="ignored",
        ground_catalog_path=FIXTURE,
    )
    first_calls = tuple(calls)
    calls.clear()
    prediction_calls.clear()
    second = analysis_service.analyze_architecture(
        valid_request(required_minimum_connectivity=0.70),
        checkpoint_path="ignored",
        ground_catalog_path=FIXTURE,
    )

    assert len(first_calls) == len(calls) == 5
    assert first.analysis_details["realization_seeds"] == second.analysis_details["realization_seeds"]
    assert first.analysis_details["realization_predictions"] == second.analysis_details["realization_predictions"]
    assert first.analysis_details["temporal_graph_hashes"] == second.analysis_details["temporal_graph_hashes"]
    assert first.space_resilience.expected_minimum_gcc == second.space_resilience.expected_minimum_gcc
    assert first.system_context.mean_overall_service_fraction == second.system_context.mean_overall_service_fraction
    assert first.system_context.ground_provenance == "SATNET_CALCULATED"
    assert first.system_context.overall_provenance == "SATNET_CALCULATED"
    assert first.space_resilience.required_minimum_connectivity == 0.80
    assert second.space_resilience.required_minimum_connectivity == 0.70
    assert first.space_resilience.realizations_meeting_requirement == 4
    assert second.space_resilience.realizations_meeting_requirement == 5
    assert first.space_resilience.realization_risk_flag is True
    assert second.space_resilience.realization_risk_flag is False
