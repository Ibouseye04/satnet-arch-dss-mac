# SATNET Final DSS — Phase 3 UI v1

## Scope

Phase 3 adds the first engineering-manager interface for the frozen Final DSS
API v1. The React client is a browser-only decision surface: it does not
reimplement orbital simulation, temporal graph construction, failure semantics,
TGNN inference, ground calculations, aggregation, or threshold semantics.

## Frontend stack

- React 19
- TypeScript 5.8
- Vite 7
- Material UI 7
- Vitest and React Testing Library
- Ordinary React state; no Redux, database, authentication, SSR, cloud SDK, or frontend ML library

The application lives under `ui/`. `node_modules/`, `ui/dist/`, and local
`.env` files are not committed.

## Local setup

From the repository root:

```powershell
cd ui
npm install
npm run dev
```

The development server is normally available at `http://localhost:5173`.
Run the API separately from the repository environment:

```powershell
python -m pip install -e ".[dss]"
$env:SATNET_DSS_TGNN_CHECKPOINT = "<local operational checkpoint>"
$env:SATNET_DSS_GROUND_CATALOG = "<local qualified ground catalog>"
python -m uvicorn satnet.dss.api.app:app --host 127.0.0.1 --port 8000
```

The API must allow the Vite origin through `SATNET_DSS_CORS_ORIGINS` when a
non-default origin is used. The UI does not require the checkpoint or catalog
files locally; they remain deployment inputs to the API.

## API configuration and startup

Set `VITE_SATNET_API_BASE_URL` in `ui/.env.local` to configure the browser API
origin. The development default is:

```text
http://localhost:8000/api/v1
```

On startup the client calls `GET /config` and `GET /readiness`. The config
response is authoritative for validated input ranges, default requirement,
realization count, catalog capacities, and model target metadata. A `READY`
readiness response enables analysis. A `NOT_READY` response shows a safe banner
and disables `ANALYZE ARCHITECTURE`; it does not expose local paths,
checkpoint internals, or stack traces.

## UI layout

The primary desktop layout is a two-column workspace:

- **Left — Architecture Configuration:** space segment, ground segment, and mission requirement.
- **Right — Resilience Assessment:** expected minimum connectivity, requirement, margin, assessment, resilience bar, realization context, and analysis details.
- **Bottom — System Context:** ground and integrated service metrics after analysis.

At narrower widths the columns stack. The top-level screen emphasizes
architecture, resilience, requirement, margin, bottleneck, and comparison;
model details are in the collapsed `Analysis Details` area or Methodology
view.

## Input semantics

The editable example starts with:

```text
5 planes × 7 satellites/plane
550 km altitude, 53° inclination
10% satellite node failure, 12% satellite edge failure
10 civilian, 10 government, 10 military ground stations
8% ground station failure
80% required minimum connectivity
```

These are example inputs, not an optimal architecture. Users see failure
probabilities and connectivity requirements as percentages. The JSON request
converts them to API fractions (`10%` → `0.10`, `80%` → `0.80`). Discrete
counts use numeric controls. The client validates against `/config` ranges and
catalog capacities before submission; the API remains authoritative.

## Result semantics

`Expected Minimum Connectivity` maps directly to
`space_resilience.expected_minimum_gcc` and is rendered as a percentage. It is
the arithmetic mean of the five raw TGNN-predicted minimum GCC values. The
result card also shows the API threshold, expected margin in percentage points,
and either `MEETS EXPECTED REQUIREMENT` or `BELOW EXPECTED REQUIREMENT`.

The horizontal resilience visual distinguishes the predicted expected minimum
GCC from the user requirement. The supporting realization message uses
`realization_count - realizations_meeting_requirement`; it intentionally never
labels the count as a probability, likelihood, confidence, or success rate.

Ground and overall values are shown only when `system_context.status` is
available. A blocked context displays the safe unavailability message without
inventing metrics. Provenance labels distinguish `TGNN Prediction` for space
resilience from `SATNET Calculated` for ground/system values.

The collapsed `Analysis Details` section includes raw realization predictions,
lowest/highest modeled GCC, seeds when supplied, realization count, model family,
target, abbreviated checkpoint identity, and provenance labels.

## Comparison behavior

After a successful analysis, the user may edit a label and select `SAVE TO
COMPARISON`. Up to four architecture/result summaries are retained in client
state only. The table retains each design's own threshold and displays
satellite count, expected minimum GCC, margin, ground/overall service,
limiting segment, and assessment. Designs can be removed individually or
cleared in one action. Saving never reruns an analysis and does not persist to
the backend.

## Methodology / About view

The Methodology dialog concisely explains the five evaluation stages, expected
minimum connectivity, descriptive meeting counts, metric provenance, the
operational TGNN versus Random Forest narrative, and model limitations:
validated ranges, minimum original-denominator GCC, an 11-timestep temporal
sequence, deterministic scenarios rather than Monte Carlo probability,
external Starlink orbital observations transformed through SATNET methodology
rather than proprietary telemetry, and calculated ground/system values.

## Testing and build

```powershell
cd ui
npm test
npm run build
```

The Vitest suite covers config/readiness startup, default threshold, percentage
conversion, API-derived ranges, readiness gating, request payloads, loading,
result/margin/assessment rendering, realization wording, system context and
blocked context, details disclosure, comparison save/remove/limit semantics,
per-design thresholds, API errors, and responsive layout smoke behavior.

## Development timing

When running in development mode, a completed analysis logs an informational
browser timing entry (`SATNET analysis completed in ... ms`). This is for
operator visibility only; the UI establishes no performance threshold.
