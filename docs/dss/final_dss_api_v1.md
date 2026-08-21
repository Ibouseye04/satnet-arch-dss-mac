# SATNET Final DSS — Phase 2 API v1

## Scope

Phase 2 adds a small, stateless FastAPI transport layer around the qualified
Phase 1 Python service:

```python
from satnet.dss import analyze_architecture
```

The API does not duplicate orbital propagation, graph construction, failure
sampling, ground service calculations, TGNN inference, or result aggregation.
It converts the HTTP request into `DSSArchitectureRequest`, calls
`analyze_architecture(...)`, and serializes `DSSAnalysisResult`.

The frontend API base is:

```text
http://localhost:8000/api/v1
```

## Optional installation and startup

FastAPI and Uvicorn are optional dependencies and are not part of the core
SATNET dependency list:

```powershell
python -m pip install -e ".[dss]"
python -m uvicorn satnet.dss.api.app:app --host 127.0.0.1 --port 8000
```

The service requires Python 3.11 or newer. No database, authentication,
cloud SDK, background job, or persistence service is used.

## Environment variables

| Variable | Required for | Meaning |
| --- | --- | --- |
| `SATNET_DSS_TGNN_CHECKPOINT` | readiness and analysis | Path to the operational TGNN checkpoint. The file must have the frozen SHA-256 `22cabef076428ba5c118b10fa230c5930af1b1c2da53bdd5c51b118dc0c9960a`. |
| `SATNET_DSS_GROUND_CATALOG` | readiness and analysis | Path to the qualified ground-station catalog. |
| `SATNET_DSS_CORS_ORIGINS` | optional | Comma-separated allowed browser origins. The default is `http://localhost:5173`; wildcard origins are not enabled by default. |

Only variable names are included in `.env.example`. Checkpoints, catalogs,
credentials, and local absolute paths are not committed.

## Endpoints

### `GET /api/v1/health`

Process/liveness check only. It does not load the catalog, inspect the
checkpoint, run SATNET, or run TGNN.

```json
{
  "status": "ok",
  "service": "satnet-dss",
  "api_version": "v1"
}
```

### `GET /api/v1/readiness`

Scientific-serving readiness check. It does not train or run inference. It
checks that:

1. `SATNET_DSS_TGNN_CHECKPOINT` is configured.
2. The checkpoint exists and has the exact frozen SHA-256.
3. `SATNET_DSS_GROUND_CATALOG` is configured.
4. The catalog exists and can be loaded using the authoritative catalog loader.

A ready service returns HTTP 200:

```json
{
  "status": "READY",
  "service": "satnet-dss",
  "api_version": "v1"
}
```

A failed check returns HTTP 503 with safe reasons. Local pathnames and
observed hashes are not returned:

```json
{
  "status": "NOT_READY",
  "service": "satnet-dss",
  "api_version": "v1",
  "reasons": ["The configured TGNN checkpoint is missing, unreadable, or fails the frozen SHA-256 check."]
}
```

### `GET /api/v1/config`

Metadata only. This endpoint does not run inference. It returns the
authoritative input contract for the UI, including the five-realization count,
default threshold, validated domain ranges, and TGNN target. When a readable
catalog is configured, `ground_catalog_capacity` reports enabled station
capacity by class.

The API does not use this metadata as a second validation path. Domain
acceptance remains authoritative in `DSSArchitectureRequest`.

### `POST /api/v1/analyze`

One independent architecture evaluation. The service supports one scientific
analysis per request and does not create history records.

Request:

```json
{
  "num_planes": 5,
  "sats_per_plane": 7,
  "altitude_km": 550,
  "inclination_deg": 53,
  "satellite_node_failure_probability": 0.10,
  "satellite_edge_failure_probability": 0.12,
  "civilian_count": 10,
  "government_count": 8,
  "military_count": 6,
  "ground_station_failure_probability": 0.08,
  "required_minimum_connectivity": 0.80
}
```

`required_minimum_connectivity` is optional and defaults to `0.80`. The HTTP
layer passes the resulting `DSSArchitectureRequest` unchanged to
`analyze_architecture(...)`.

The response preserves the Phase 1 result sections:

- `architecture`: canonical accepted inputs.
- `model`: operational model identity, target, checkpoint hash, and realization count.
- `space_resilience`: expected, lowest, and highest raw modeled GCC values; threshold; margins; assessment; meeting count; and risk flag.
- `system_context`: ground and overall service metrics, limiting segment, status, and SATNET-calculated provenance.
- `provenance`: model and calculation provenance, seed policy, graph/scenario identity evidence, and `training_performed: false`.
- `analysis_details`: individual realization predictions, seeds, temporal feature shapes, graph identities, and out-of-physical-interval flags.

`expected_minimum_gcc` is the arithmetic mean of five raw TGNN-predicted
minimum GCC values. Predictions are not clipped. The count is descriptive:
`realizations_meeting_requirement` means “4 of 5 modeled realizations met the
requirement,” not a probability, likelihood, or confidence estimate. The API
has no probability-of-success field.

Ground and overall metrics are marked `SATNET_CALCULATED`; space predictions
are marked `TGNN_PREDICTION`.

## Threshold semantics

The threshold is a post-inference engineering requirement. Changing only the
threshold changes margins, assessment, meeting count, and risk flag. It does
not change physical scenario construction, realization seeds, temporal graph
identities, TGNN feature tensors, or raw predictions. No model is selected,
retrained, tuned, or modified by the API.

## Validation and errors

The transport model handles JSON shape and scalar types, then delegates domain
validation to `DSSArchitectureRequest`. Space values outside the validated
TGNN domain are rejected with HTTP 422. Catalog capacity remains an
authoritative Phase 1 selection check.

All API errors use this shape and never expose Python tracebacks:

```json
{
  "error": {
    "code": "ARCHITECTURE_OUTSIDE_VALIDATED_DOMAIN",
    "message": "altitude_km must be finite and within [300.0, 1200.0]",
    "field": "altitude_km"
  }
}
```

Common statuses:

- `400 MALFORMED_JSON`: request body is not valid JSON.
- `422 INVALID_ARCHITECTURE_REQUEST`: request shape or scalar types are invalid.
- `422 ARCHITECTURE_OUTSIDE_VALIDATED_DOMAIN`: domain validation rejects an architecture.
- `503 DSS_CHECKPOINT_UNAVAILABLE`: checkpoint is missing or fails the frozen operational check.
- `503 DSS_CATALOG_UNAVAILABLE`: catalog is unavailable, invalid, or cannot serve the requested catalog-backed counts.
- `500 INTERNAL_ANALYSIS_ERROR`: unexpected failure with a safe public message.

## CORS

The default allowed browser origin is `http://localhost:5173`. Set
`SATNET_DSS_CORS_ORIGINS` to a comma-separated allowlist for another local
Vite origin or deployment. Wildcard origins are not enabled by default and
credentials are not allowed.

## Operational policies

- **No persistence:** every analysis is independent; comparison is client-side.
- **No training:** the API only loads the frozen operational checkpoint and performs inference.
- **No frontend:** React/Vite is a later phase.
- **No authentication:** authentication is outside Phase 2 scope.
- **No model binaries or private catalogs:** operational artifacts remain deployment inputs and are not committed.
