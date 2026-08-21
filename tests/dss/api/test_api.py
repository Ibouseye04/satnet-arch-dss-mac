from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from satnet.dss import analysis_service
from satnet.dss.api.app import app as exported_app
import importlib

app_module = importlib.import_module("satnet.dss.api.app")
from satnet.dss.api.settings import DSSApiSettings
from satnet.dss.domain import (
    DSSAnalysisResult,
    DSSModelMetadata,
    DSSSpaceResilience,
    DSSSystemContext,
)
from satnet.dss.schemas import DSSArchitectureRequest
from satnet.dss.tgnn_inference import DSSCheckpointError, FROZEN_TGNN_CHECKPOINT_SHA256


CATALOG = Path(__file__).parents[2] / "fixtures" / "ground_segment" / "synthetic_ground_station_catalog.csv"


def settings(**overrides: object) -> DSSApiSettings:
    values: dict[str, object] = {
        "checkpoint_path": "operational.pt",
        "ground_catalog_path": str(CATALOG),
        "cors_origins": ("http://localhost:5173",),
    }
    values.update(overrides)
    return DSSApiSettings(**values)


def payload(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "num_planes": 5,
        "sats_per_plane": 7,
        "altitude_km": 550,
        "inclination_deg": 53,
        "satellite_node_failure_probability": 0.10,
        "satellite_edge_failure_probability": 0.12,
        "civilian_count": 1,
        "government_count": 1,
        "military_count": 1,
        "ground_station_failure_probability": 0.08,
        "required_minimum_connectivity": 0.80,
    }
    result.update(overrides)
    return result


def result_for(request: DSSArchitectureRequest) -> DSSAnalysisResult:
    predictions = [0.84, 0.81, 0.76, 0.83, 1.04]
    return DSSAnalysisResult(
        architecture=request.to_dict(),
        model=DSSModelMetadata(
            family="TGNN",
            task="space_regression",
            target="space_gcc_fraction_original_min",
            checkpoint_sha256=FROZEN_TGNN_CHECKPOINT_SHA256,
            realization_count=5,
        ),
        space_resilience=DSSSpaceResilience(
            expected_minimum_gcc=sum(predictions) / 5,
            lowest_modeled_gcc=min(predictions),
            highest_modeled_gcc=max(predictions),
            required_minimum_connectivity=request.required_minimum_connectivity,
            expected_margin=sum(predictions) / 5 - request.required_minimum_connectivity,
            lowest_margin=min(predictions) - request.required_minimum_connectivity,
            expected_assessment="MEETS_EXPECTED_REQUIREMENT",
            realizations_meeting_requirement=sum(
                prediction >= request.required_minimum_connectivity for prediction in predictions
            ),
            realization_count=5,
            realization_risk_flag=True,
        ),
        system_context=DSSSystemContext(
            mean_ground_service_fraction=0.72,
            minimum_ground_service_fraction=0.51,
            mean_overall_service_fraction=0.69,
            minimum_overall_service_fraction=0.40,
            limiting_segment="GROUND",
            ground_provenance="SATNET_CALCULATED",
            overall_provenance="SATNET_CALCULATED",
            status="AVAILABLE",
        ),
        provenance={
            "prediction_provenance": "TGNN_PREDICTION",
            "ground_provenance": "SATNET_CALCULATED",
            "training_performed": False,
        },
        analysis_details={
            "realization_predictions": predictions,
            "temporal_graph_hashes": [[f"graph-{index}"] for index in range(5)],
        },
    )


def test_health_does_not_require_operational_artifacts() -> None:
    client = TestClient(exported_app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "satnet-dss", "api_version": "v1"}


def test_readiness_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "resolve_checkpoint_path", lambda path: Path(path))
    client = TestClient(app_module.create_app(settings()))
    response = client.get("/api/v1/readiness")
    assert response.status_code == 200
    assert response.json()["status"] == "READY"


def test_readiness_missing_checkpoint() -> None:
    client = TestClient(app_module.create_app(settings(checkpoint_path=None)))
    response = client.get("/api/v1/readiness")
    assert response.status_code == 503
    assert response.json()["status"] == "NOT_READY"
    assert "checkpoint" in response.json()["reasons"][0].lower()


def test_readiness_checkpoint_hash_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def mismatch(_: str) -> Path:
        raise DSSCheckpointError("internal mismatch details")

    monkeypatch.setattr(app_module, "resolve_checkpoint_path", mismatch)
    client = TestClient(app_module.create_app(settings()))
    response = client.get("/api/v1/readiness")
    assert response.status_code == 503
    assert response.json()["reasons"] == [
        "The configured TGNN checkpoint is missing, unreadable, or fails the frozen SHA-256 check."
    ]
    assert "internal mismatch" not in response.text


def test_readiness_missing_catalog() -> None:
    client = TestClient(app_module.create_app(settings(ground_catalog_path=None)))
    response = client.get("/api/v1/readiness")
    assert response.status_code == 503
    assert "catalog" in response.text.lower()


def test_config_exposes_authoritative_ranges_and_default() -> None:
    client = TestClient(app_module.create_app(settings()))
    response = client.get("/api/v1/config")
    body = response.json()
    assert response.status_code == 200
    assert body["realization_count"] == 5
    assert body["default_required_minimum_connectivity"] == 0.80
    assert body["space_domain"]["num_planes"] == {"min": 4, "max": 6}
    assert body["space_domain"]["altitude_km"] == {"min": 300, "max": 1200}
    assert body["ground_domain"]["ground_station_failure_probability"] == {
        "min": 0.0,
        "max": 0.40,
    }


def test_analyze_accepts_valid_request_and_preserves_result_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[DSSArchitectureRequest] = []

    def fake_analyze(request: DSSArchitectureRequest, **_: object) -> DSSAnalysisResult:
        calls.append(request)
        return result_for(request)

    monkeypatch.setattr(analysis_service, "analyze_architecture", fake_analyze)
    client = TestClient(app_module.create_app(settings()))
    response = client.post("/api/v1/analyze", json=payload())
    body = response.json()
    assert response.status_code == 200
    assert calls[0].required_minimum_connectivity == 0.80
    assert body["space_resilience"]["expected_minimum_gcc"] == pytest.approx(0.856)
    assert body["space_resilience"]["realization_count"] == 5
    assert body["system_context"]["mean_ground_service_fraction"] == 0.72
    assert body["provenance"]["prediction_provenance"] == "TGNN_PREDICTION"
    assert body["system_context"]["ground_provenance"] == "SATNET_CALCULATED"
    assert body["analysis_details"]["realization_predictions"][-1] == 1.04
    assert "probability" not in body["space_resilience"]
    assert "likelihood" not in body["space_resilience"]


def test_analyze_defaults_threshold_when_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[DSSArchitectureRequest] = []

    def fake_analyze(request: DSSArchitectureRequest, **_: object) -> DSSAnalysisResult:
        observed.append(request)
        return result_for(request)

    monkeypatch.setattr(analysis_service, "analyze_architecture", fake_analyze)
    client = TestClient(app_module.create_app(settings()))
    response = client.post("/api/v1/analyze", json=payload(required_minimum_connectivity=None))
    assert response.status_code == 422

    response = client.post(
        "/api/v1/analyze",
        json={key: value for key, value in payload().items() if key != "required_minimum_connectivity"},
    )
    assert response.status_code == 200
    assert observed[0].required_minimum_connectivity == 0.80


def test_analyze_rejects_invalid_domain_with_structured_error() -> None:
    client = TestClient(app_module.create_app(settings()))
    response = client.post("/api/v1/analyze", json=payload(altitude_km=1200.001))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "ARCHITECTURE_OUTSIDE_VALIDATED_DOMAIN"
    assert response.json()["error"]["field"] == "altitude_km"
    assert "traceback" not in response.text.lower()


def test_malformed_json_is_structured() -> None:
    client = TestClient(app_module.create_app(settings()))
    response = client.post("/api/v1/analyze", content="{not-json", headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert response.json() == {
        "error": {"code": "MALFORMED_JSON", "message": "Request body is not valid JSON."}
    }


def test_unexpected_failure_does_not_leak_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object, **__: object) -> None:
        raise RuntimeError("unexpected internal secret")

    monkeypatch.setattr(analysis_service, "analyze_architecture", fail)
    client = TestClient(app_module.create_app(settings()), raise_server_exceptions=False)
    response = client.post("/api/v1/analyze", json=payload())
    assert response.status_code == 500
    assert "unexpected internal secret" not in response.text
    assert "traceback" not in response.text.lower()


def test_http_and_python_service_share_identical_scientific_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_analyze(request: DSSArchitectureRequest, **_: object) -> DSSAnalysisResult:
        return result_for(request)

    monkeypatch.setattr(analysis_service, "analyze_architecture", fake_analyze)
    client = TestClient(app_module.create_app(settings()))
    domain_request = DSSArchitectureRequest.from_mapping(payload())
    python_result = analysis_service.analyze_architecture(domain_request)
    http_result = client.post("/api/v1/analyze", json=payload()).json()
    for field in ("architecture", "model", "space_resilience", "system_context", "provenance", "analysis_details"):
        assert http_result[field] == python_result.to_dict()[field]


def test_threshold_changes_decision_fields_but_not_prediction_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_analyze(request: DSSArchitectureRequest, **_: object) -> DSSAnalysisResult:
        return result_for(request)

    monkeypatch.setattr(analysis_service, "analyze_architecture", fake_analyze)
    client = TestClient(app_module.create_app(settings()))
    low = client.post("/api/v1/analyze", json=payload(required_minimum_connectivity=0.70)).json()
    high = client.post("/api/v1/analyze", json=payload(required_minimum_connectivity=0.90)).json()
    assert low["analysis_details"]["realization_predictions"] == high["analysis_details"]["realization_predictions"]
    assert low["space_resilience"]["expected_minimum_gcc"] == high["space_resilience"]["expected_minimum_gcc"]
    assert low["space_resilience"]["required_minimum_connectivity"] == 0.70
    assert high["space_resilience"]["required_minimum_connectivity"] == 0.90


def test_cors_allows_configured_origin_and_rejects_unconfigured_origin() -> None:
    client = TestClient(
        app_module.create_app(settings(cors_origins=("http://ui.example",)))
    )
    allowed = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://ui.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    denied = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.headers.get("access-control-allow-origin") == "http://ui.example"
    assert "access-control-allow-origin" not in denied.headers


@pytest.mark.skipif(
    not __import__("os").environ.get("SATNET_DSS_TGNN_CHECKPOINT")
    or not Path(__import__("os").environ["SATNET_DSS_TGNN_CHECKPOINT"]).is_file()
    or not __import__("os").environ.get("SATNET_DSS_GROUND_CATALOG"),
    reason="Operational checkpoint and catalog are supplied by the deployment environment",
)
def test_real_phase_one_acceptance_path() -> None:
    client = TestClient(exported_app)
    readiness = client.get("/api/v1/readiness")
    assert readiness.status_code == 200
    response = client.post("/api/v1/analyze", json=payload())
    assert response.status_code == 200
    body = response.json()
    assert body["space_resilience"]["realization_count"] == 5
    assert "expected_minimum_gcc" in body["space_resilience"]
