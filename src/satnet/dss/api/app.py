"""FastAPI transport layer for the qualified SATNET Phase 1 DSS service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from satnet.dss import analysis_service
from satnet.dss.api.errors import DSSApiError
from satnet.dss.api.models import DSSArchitectureRequestDTO
from satnet.dss.api.settings import DSSApiSettings
from satnet.dss.ground_context import resolve_catalog_path
from satnet.dss.schemas import DSS_DEFAULT_THRESHOLD, DSS_REALIZATION_COUNT, DSSValidationError
from satnet.dss.tgnn_inference import DSSCheckpointError, resolve_checkpoint_path
from satnet.ground.catalog import GroundStationClass, load_ground_station_catalog

API_VERSION = "v1"
SERVICE_NAME = "satnet-dss"


class _ValidatedDomain:
    space_domain = {
        "num_planes": {"min": 4, "max": 6},
        "sats_per_plane": {"min": 5, "max": 8},
        "altitude_km": {"min": 300, "max": 1200},
        "inclination_deg": {"min": 30, "max": 98},
        "satellite_node_failure_probability": {"min": 0.0, "max": 0.20},
        "satellite_edge_failure_probability": {"min": 0.0, "max": 0.25},
    }
    ground_domain = {
        "ground_station_failure_probability": {"min": 0.0, "max": 0.40},
    }


def _error_payload(error: DSSApiError) -> dict[str, dict[str, str]]:
    body: dict[str, str] = {"code": error.code, "message": error.message}
    if error.field is not None:
        body["field"] = error.field
    return {"error": body}


def _field_from_message(message: str) -> str | None:
    candidate = message.split(" ", 1)[0]
    return candidate if candidate.isidentifier() else None


def _request_validation_error(exc: RequestValidationError) -> DSSApiError:
    details = exc.errors()
    json_invalid = any(detail.get("type") == "json_invalid" for detail in details)
    if json_invalid:
        return DSSApiError(400, "MALFORMED_JSON", "Request body is not valid JSON.")
    detail = details[0] if details else {}
    location = detail.get("loc", ())
    field = next((str(part) for part in reversed(location) if part != "body"), None)
    return DSSApiError(
        422,
        "INVALID_ARCHITECTURE_REQUEST",
        "Request body does not match the DSS architecture input contract.",
        field,
    )


def _readiness_reasons(settings: DSSApiSettings) -> list[str]:
    reasons: list[str] = []
    if not settings.checkpoint_path:
        reasons.append("SATNET_DSS_TGNN_CHECKPOINT is not configured.")
    else:
        try:
            resolve_checkpoint_path(settings.checkpoint_path)
        except DSSCheckpointError:
            reasons.append("The configured TGNN checkpoint is missing, unreadable, or fails the frozen SHA-256 check.")

    if not settings.ground_catalog_path:
        reasons.append("SATNET_DSS_GROUND_CATALOG is not configured.")
    else:
        try:
            catalog_path = resolve_catalog_path(settings.ground_catalog_path)
            load_ground_station_catalog(catalog_path)
        except (OSError, RuntimeError, TypeError, ValueError):
            reasons.append("The configured ground-station catalog is missing, unreadable, or invalid.")
    return reasons


def _catalog_capacity(settings: DSSApiSettings) -> dict[str, int] | None:
    if not settings.ground_catalog_path:
        return None
    try:
        catalog = load_ground_station_catalog(resolve_catalog_path(settings.ground_catalog_path))
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return {
        station_class.value: len(catalog.eligible(station_class))
        for station_class in GroundStationClass
    }


def _config_payload(settings: DSSApiSettings) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "realization_count": DSS_REALIZATION_COUNT,
        "default_required_minimum_connectivity": DSS_DEFAULT_THRESHOLD,
        "space_domain": _ValidatedDomain.space_domain,
        "ground_domain": _ValidatedDomain.ground_domain,
        "model": {
            "family": "TGNN",
            "target": "space_gcc_fraction_original_min",
        },
    }
    capacity = _catalog_capacity(settings)
    if capacity is not None:
        payload["ground_catalog_capacity"] = capacity
    return payload


def create_app(settings: DSSApiSettings | None = None) -> FastAPI:
    service_settings = settings or DSSApiSettings.from_environment()

    def current_settings() -> DSSApiSettings:
        return service_settings if settings is not None else DSSApiSettings.from_environment()

    application = FastAPI(
        title="SATNET DSS API",
        version=API_VERSION,
        description="Thin HTTP layer around the qualified SATNET Phase 1 DSS analysis service.",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(service_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @application.exception_handler(DSSApiError)
    async def handle_dss_api_error(_: Request, exc: DSSApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=_error_payload(exc))

    @application.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        error = _request_validation_error(exc)
        return JSONResponse(status_code=error.status_code, content=_error_payload(error))

    @application.get(f"/api/{API_VERSION}/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": SERVICE_NAME, "api_version": API_VERSION}

    @application.get(f"/api/{API_VERSION}/readiness")
    async def readiness() -> JSONResponse:
        reasons = _readiness_reasons(current_settings())
        if reasons:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "NOT_READY",
                    "service": SERVICE_NAME,
                    "api_version": API_VERSION,
                    "reasons": reasons,
                },
            )
        return JSONResponse(
            status_code=200,
            content={"status": "READY", "service": SERVICE_NAME, "api_version": API_VERSION},
        )

    @application.get(f"/api/{API_VERSION}/config")
    async def config() -> dict[str, Any]:
        return _config_payload(current_settings())

    @application.post(f"/api/{API_VERSION}/analyze")
    async def analyze(request: DSSArchitectureRequestDTO) -> dict[str, Any]:
        try:
            architecture = request.to_domain()
        except DSSValidationError as exc:
            raise DSSApiError(
                422,
                "ARCHITECTURE_OUTSIDE_VALIDATED_DOMAIN",
                str(exc),
                _field_from_message(str(exc)),
            ) from exc
        except (TypeError, ValueError) as exc:
            raise DSSApiError(
                422,
                "INVALID_ARCHITECTURE_REQUEST",
                "Architecture values do not match the DSS input contract.",
                _field_from_message(str(exc)),
            ) from exc

        runtime_settings = current_settings()
        try:
            result = analysis_service.analyze_architecture(
                architecture,
                checkpoint_path=runtime_settings.checkpoint_path,
                ground_catalog_path=runtime_settings.ground_catalog_path,
            )
        except DSSCheckpointError as exc:
            raise DSSApiError(
                503,
                "DSS_CHECKPOINT_UNAVAILABLE",
                "The frozen TGNN checkpoint is unavailable or invalid.",
                "SATNET_DSS_TGNN_CHECKPOINT",
            ) from exc
        except OSError as exc:
            raise DSSApiError(
                503,
                "DSS_CATALOG_UNAVAILABLE",
                "The configured ground-station catalog is unavailable or invalid.",
                "SATNET_DSS_GROUND_CATALOG",
            ) from exc
        except RuntimeError as exc:
            message = str(exc).lower()
            if "catalog" in message or "satnet_dss_ground_catalog" in message:
                raise DSSApiError(
                    503,
                    "DSS_CATALOG_UNAVAILABLE",
                    "The configured ground-station catalog is unavailable or invalid.",
                    "SATNET_DSS_GROUND_CATALOG",
                ) from exc
            raise DSSApiError(500, "INTERNAL_ANALYSIS_ERROR", "The DSS analysis failed internally.") from exc
        except ValueError as exc:
            message = str(exc)
            lowered_message = message.lower()
            if "eligible" in lowered_message:
                raise DSSApiError(
                    422,
                    "ARCHITECTURE_OUTSIDE_VALIDATED_DOMAIN",
                    "Requested ground-station counts exceed the configured catalog capacity.",
                ) from exc
            if "catalog" in lowered_message:
                raise DSSApiError(
                    503,
                    "DSS_CATALOG_UNAVAILABLE",
                    "The configured ground-station catalog is unavailable or invalid.",
                    "SATNET_DSS_GROUND_CATALOG",
                ) from exc
            raise DSSApiError(500, "INTERNAL_ANALYSIS_ERROR", "The DSS analysis failed internally.") from exc
        except Exception as exc:
            raise DSSApiError(500, "INTERNAL_ANALYSIS_ERROR", "The DSS analysis failed internally.") from exc
        return result.to_dict()

    return application


app = create_app()
