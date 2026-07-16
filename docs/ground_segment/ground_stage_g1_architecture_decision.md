# Ground Segment Stage G1 Architecture Decision

**Status:** Approved

**Decision date:** 2026-07-16

**Validated satellite code SHA:** `0229a9366756cf8e922589253e681f2fa1d50989`

**Corrected-pilot documentation SHA:** `0229a9366756cf8e922589253e681f2fa1d50989`

**Ground branch base SHA:** `0229a9366756cf8e922589253e681f2fa1d50989`

## Decision

Stage G1 authorizes metadata-only ground-station catalog, classification, deterministic region-balanced selection, identity, companion-manifest persistence, exact reconstruction, and passive scenario composition.

Stage G1 does not authorize visibility, Earth rotation, coordinate propagation, ground graph elements, ground failures, service metrics, combined datasets, satellite schema migration, cache migration, or model changes.

## Satellite boundary

The following satellite contracts remain unchanged:

- `Tier1RolloutConfig` and its configuration hash.
- `run_tier1_rollout`.
- Satellite run and step rows.
- Dataset schema and version.
- TGNN reconstruction requirements.
- Satellite graph-cache identity.
- Satellite propagation, graph construction, failures, and labels.

Ground metadata has no execution path into the satellite rollout. A passive scenario object may hold the existing satellite configuration and a ground-design record, validate their hashes, and calculate an outer identity.

## Domain layers

The ground subsystem consists of four layers:

1. A satellite-independent catalog domain.
2. A satellite-independent deterministic selection domain.
3. A satellite-independent companion-manifest persistence domain.
4. A passive scenario-composition module that may import `Tier1RolloutConfig` but no rollout runner, graph builder, model loader, cache function, or metric.

## Station classes

The only G1 station classes are civilian, government, and military. These values are categorical research-scenario metadata. They do not imply different antennas, RF performance, bandwidth, scheduling priority, routing priority, security policy, failure behavior, service weighting, or operational policy.

## Catalog contract

Canonical station records contain a stable ID, canonical research name, class, geodetic latitude and longitude, terrestrial altitude, normalized region slug, uppercase two-letter country code, and enabled state.

The supported terrestrial altitude bounds are named constants with values `-500.0` and `9000.0` metres. They support Earth-surface research scenarios ranging from below-sea-level locations through high-altitude terrestrial sites. They are input-validation limits, not satellite-ground propagation physics. A future incompatible interpretation of altitude or coordinate reference system requires a ground-domain or schema-version decision.

Catalog identity is the SHA-256 hash of canonical station content under an explicit catalog identity domain and version. It is independent of path and row order. All station-defining fields, including disabled records, participate. Raw CSV numeric text is never identity.

Generic validity and production readiness are separate. Generic catalogs may be small. Production readiness requires at least 50 enabled records in each class. A generated test catalog proves software capacity only, not production catalog availability or scientific validity.

Actual catalog content must be preserved in immutable external or versioned artifact storage under its content-derived hash. Machine-local paths are not identity and are not persisted in canonical manifests.

## Selection contract

Selection uses a versioned deterministic region-balanced complete ordering for each class. Regions and stations are ranked with SHA-256 using the selection version, explicit seed, class, region, and stable station ID. Explicit identifiers break digest ties. The full catalog hash is excluded from ranking, but included in selection identity.

Selection takes a prefix of each complete class ordering. Increasing a requested count preserves prior prefixes only while eligible catalog content, selection version, class, and seed remain unchanged. Catalog mutation may produce new prefixes. Adding a disabled record leaves eligible ordering and selected IDs unchanged while changing catalog and selection hashes.

Region balance is relative to the supplied region taxonomy. It does not establish great-circle optimality, hemisphere balance, latitude or longitude balance, physical separation, or globally optimal placement.

## Identity boundaries

- `catalog_hash` identifies complete canonical station content.
- `selection_hash` identifies catalog identity, algorithm version, seed, requested class counts, and canonical selected IDs.
- `ground_design_hash` identifies enabled or disabled G1 ground design under a ground-design schema version.
- Optional `scenario_design_hash` composes unchanged satellite and ground identities under a scenario identity version.

Every canonical hash payload contains an explicit identity domain and version. Ground identity never enters the satellite configuration hash or satellite graph-cache key.

## Companion manifest

Canonical manifests use strict UTF-8 JSON Lines with one compact, sorted-key object per line and records sorted by integer `run_id`. Every satellite run has exactly one enabled or disabled companion record. Duplicate, orphan, and missing run IDs fail.

Enabled records persist explicit class-grouped and combined selected station IDs. Reconstruction receives a validated catalog explicitly, verifies catalog content, verifies deterministic prefixes, and recomputes selection and ground-design hashes. Persisted IDs are authoritative and reordered IDs fail. Seed-only reconstruction is prohibited.

Disabled records use zero counts, empty ID lists, and no catalog or selection metadata. Their ground-design identity is canonical and independent of satellite identity.

Canonical writes validate before modifying the destination, use a temporary sibling, flush and synchronize where supported, close, and atomically replace. Existing manifests are not overwritten by default.

## Validation boundary

Portable tests cover domain validation, canonicalization, catalog identity, production-readiness capacity, deterministic selection, nested prefixes, region balance, manifests, reconstruction, passive composition, direct static import boundaries, and satellite scientific invariance.

Git history and protected-path diff checks are release-validation procedures, not unit tests. Final release validation requires an empty diff from the recorded ground branch base across protected satellite modules.

## Research-catalog qualification

Subsystem production readiness, production catalog availability, and production catalog scientific validity are independent states. Software readiness does not certify that a 50-per-class research catalog exists or that its sources, license, taxonomy, completeness, and classifications have been reviewed.

The catalog represents research scenarios and must not be described as a complete or authoritative inventory of civilian, government, or military ground infrastructure.
