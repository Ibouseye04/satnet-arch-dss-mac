# Ground Segment Stage G1 Catalog and Selection Contract

## Scope

Stage G1 provides metadata-only ground-station catalog, classification, deterministic region-balanced selection, scientific identity, canonical companion-manifest persistence, exact reconstruction, and passive scenario composition.

Stage G1 does not implement satellite-to-ground visibility, Earth rotation, coordinate propagation, ground graph elements, ground failures, service metrics, combined resilience labels, satellite dataset migration, graph-cache migration, or model changes.

The existing `Tier1RolloutConfig`, satellite configuration hash, rollout runner, run and step tables, schema and dataset versions, TGNN reconstruction contract, satellite graph-cache identity, propagation, ISL construction, satellite failures, and satellite labels remain unchanged.

## Station classes

The supported classes are:

- `civilian`
- `government`
- `military`

Station classes are categorical metadata used for scenario construction and later class-specific resilience reporting. They do not currently imply different antennas, RF performance, bandwidth, scheduling priority, failure behavior, or operational policy.

The catalog represents research scenarios and must not be described as a complete or authoritative inventory of civilian, government, or military ground infrastructure.

## Catalog schema

The canonical CSV columns are:

| Column | Contract |
|---|---|
| `station_id` | Uppercase stable identifier matching `^[A-Z][A-Z0-9_]{2,63}$` |
| `name` | Surrounding Unicode whitespace removed, NFC-normalized, nonempty, at most 128 code points |
| `station_class` | Exactly `civilian`, `government`, or `military` |
| `latitude_deg` | Finite binary64 value in `[-90, 90]` |
| `longitude_deg` | Finite binary64 value in `[-180, 180]` |
| `altitude_m` | Finite binary64 value in `[-500, 9000]` |
| `region` | Lowercase slug matching `^[a-z][a-z0-9_]{1,63}$` |
| `country_code` | Exactly two uppercase ASCII letters; synthetic records use `ZZ` |
| `enabled` | Exactly lowercase `true` or `false` in canonical CSV |

Headers must match the exact set. Missing, unexpected, or duplicate headers fail. Invalid rows, duplicate station IDs, empty catalogs, invalid numeric values, and noncanonical Boolean spellings fail.

The altitude bounds support terrestrial Earth-surface research scenarios from below-sea-level locations through high-altitude terrestrial sites. They are input-validation limits, not propagation physics.

## Canonicalization and catalog identity

CSV numeric tokens are parsed as Python binary64 floats. Nonfinite values are rejected. Negative zero is normalized to zero. Canonical float values use `format(value, ".17g")` and are encoded as JSON strings in hash payloads.

Canonical JSON uses UTF-8, sorted dictionary keys, compact separators, and no raw floating-point values. Station records are sorted by station ID before catalog hashing.

Catalog hash payloads include:

```text
identity_domain = satnet_ground_catalog
identity_version = 1
all canonical station records and all station-defining fields
```

The catalog hash is independent of source path and row order. Changes to names, classes, coordinates, region, country code, enabled state, record membership, or station ID change catalog identity.

The complete catalog hash does not participate in ranking. This prevents unrelated catalog classes and disabled records from reranking eligible stations. It remains part of selection identity so catalog provenance is fail-closed.

## Generic and production-ready catalogs

Generic catalog validity requires at least one valid station and successful canonical hashing. Synthetic fixtures may be small.

Production-readiness validation separately requires at least:

- 50 enabled civilian stations.
- 50 enabled government stations.
- 50 enabled military stations.

Disabled stations do not count. More than 150 total records are permitted.

Passing the generated 50-per-class capacity test does not prove that an actual research catalog exists or that its sources, license, taxonomy, classifications, or completeness have been scientifically reviewed.

## Ground configuration

Enabled configuration stores exact nonnegative integer counts for each class and a selection seed in `[0, 2^63 - 1]`. Booleans, strings, integral floats, and negative values fail. Enabled mode requires a positive total count and fails if any requested count exceeds eligible capacity.

Disabled mode is a distinct type with zero selected stations. It is not represented by an ambiguous enabled configuration containing three zero counts.

Ground configuration contains no constellation, visibility, elevation, failure, RF, service, or model parameters.

## Deterministic region-balanced selection

Selection algorithm version is `1`.

For each station class, enabled stations are grouped by region. Regions are sorted by:

```text
SHA-256(selection version, class, seed, region), region ID
```

Stations within each region are sorted by:

```text
SHA-256(selection version, class, seed, region, station ID), station ID
```

The explicit identifier is the deterministic digest-collision tie-breaker. The ranked region queues are traversed round-robin until every eligible station is emitted.

Before an eligible region is exhausted, selected counts among eligible regions differ by no more than one. After exhaustion, exhausted regions retain their available maximum while nonexhausted regions continue to differ by no more than one.

This is region balance relative to catalog taxonomy. It is not great-circle optimization, hemisphere balance, latitude or longitude balance, physical separation, or globally optimal placement.

## Nested-prefix guarantee

Each class has one complete ordering for a fixed:

- Eligible catalog content.
- Selection algorithm version.
- Station class.
- Selection seed.

Requested designs select prefixes. Increasing a count preserves and does not reorder the prior prefix while those inputs remain unchanged.

Catalog mutation may change ordering and prefixes. Adding a disabled record does not change eligible ordering or selected IDs, but changes both catalog and selection hashes because selection identity includes catalog identity.

## Selection output and identity

Canonical combined output order is:

1. Civilian IDs in civilian prefix order.
2. Government IDs in government prefix order.
3. Military IDs in military prefix order.

Selection output validates exact class counts, total count, uniqueness, existence, enablement, class membership, class-list order, and combined-list concatenation.

`selection_hash` identifies the full selection specification and persisted realization. Its payload contains:

```text
identity_domain = satnet_ground_selection
identity_version = 1
catalog_hash
selection version
selection seed
requested per-class counts
canonical per-class selected IDs
canonical combined selected IDs
```

Different seeds remain different selection identities even if they happen to produce identical selected IDs.

## Ground-design companion manifest

Canonical manifests use `.jsonl` and contain exactly one enabled or disabled ground-design record for every satellite run in the associated run collection.

Each record stores:

- Ground-design schema version.
- Integer satellite run ID.
- Existing satellite configuration hash.
- Enabled or disabled mode.
- Catalog and selection identity when enabled.
- Selection seed and exact class counts when enabled.
- Explicit per-class and combined station-ID lists.
- Ground-design hash.

Manifest and satellite run-ID sets must be equal. Duplicate, orphan, and missing run IDs fail. Satellite hash mismatches fail.

The canonical writer:

- Validates and serializes all records before touching the destination.
- Sorts records by numeric run ID.
- Emits one compact sorted-key UTF-8 JSON object per line.
- Uses Unix newlines.
- Writes to a temporary sibling.
- Flushes and synchronizes the temporary file.
- Atomically replaces the destination.
- Removes temporary files on failure.
- Refuses overwrite by default.

The reader rejects empty files, empty lines, duplicate JSON keys, malformed JSON, unknown or missing fields, duplicate run IDs, invalid exact types, invalid mode unions, and invalid hashes.

## Exact reconstruction

Enabled reconstruction receives an explicitly supplied validated catalog. It does not load a path from the manifest and never downloads or guesses replacement content.

Reconstruction verifies:

1. Catalog content hash.
2. Explicit persisted IDs.
3. ID existence and enablement.
4. Class membership and counts.
5. Canonical class concatenation.
6. Deterministic prefix order rebuilt from the stored seed.
7. Selection hash.
8. Ground-design hash.

Reordered, unknown, disabled, wrong-class, wrong-count, non-prefix, missing, or hash-inconsistent IDs fail. Explicit IDs are authoritative evidence; seed-only reconstruction is prohibited.

Disabled reconstruction requires no catalog and returns no selection.

## Ground and scenario identities

Ground-design schema version is `1`. Ground-design identity contains an explicit `satnet_ground_design` domain/version, enabled mode, and selection hash. Disabled mode has one canonical ground-design identity.

The passive scenario identity contains:

```text
identity_domain = satnet_scenario_design
identity_version = 1
existing satellite config hash
ground design hash
```

`ScenarioDesign` holds the exact existing satellite configuration object and ground record, verifies their satellite hashes, and computes only the outer identity. It does not run simulation or import rollout, graph, model, metric, or cache functions.

## Satellite invariance

Ground metadata remains outside satellite execution. Automated tests verify unchanged:

- Satellite configuration object and hash.
- SGP4 positions.
- Satellite node identities.
- ISL edge identities and attributes.
- Satellite failure realization.
- Run and step metrics.
- Satellite labels.
- TGNN reconstructed node features, edge indices, edge attributes, timesteps, and targets.
- Satellite graph-cache identity.

AST tests enforce direct static import boundaries. Protected satellite modules do not import the ground subsystem. Runtime tests and review additionally check for callback, dynamic-import, registration, monkey-patching, and side-effect integration.

## Catalog retention and provenance

Every canonical experiment using enabled ground design must preserve the exact catalog artifact in immutable external or versioned artifact storage under its content-derived catalog hash. File paths are not scientific identity and are not stored in canonical manifests.

Optional nonidentity provenance may record catalog identifier, version, designation, source description, license, region-taxonomy description, and review status. Provenance metadata does not change ranking unless canonical station records change.

## Deferred Stage G2 work

Stage G2 design may address coordinate reference systems, Earth rotation, ECEF positions, visibility, elevation policies, and physical ground-link semantics. Ground failures, service metrics, graph integration, combined labels, cache domains, dataset schemas, and model changes require later explicit contracts and versions.
