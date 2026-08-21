# Ground Segment Stage G1 Validation Report

**Validation date:** 2026-07-16

**Branch:** `feature/ground-segment`

**Validated satellite code SHA:** `0229a9366756cf8e922589253e681f2fa1d50989`

**Corrected-pilot documentation SHA:** `0229a9366756cf8e922589253e681f2fa1d50989`

**Ground branch base SHA:** `0229a9366756cf8e922589253e681f2fa1d50989`

**Validated implementation SHA:** `753b28be767b7bcee213e239bbfd086b26b1944b`

## Scope result

Stage G1 implemented the metadata-only ground catalog, classification, deterministic region-balanced selection, identity, canonical JSONL companion manifest, exact reconstruction, and passive scenario composition contracts.

No visibility, Earth rotation, ground coordinate propagation, ground graph elements, ground failures, service metrics, combined datasets, model training, or satellite scientific changes were implemented.

## Test summary

| Gate | Result |
|---|---:|
| Baseline complete suite | 318 passed |
| Final Stage G1 focused suite | 153 passed |
| Final complete suite | 471 passed |
| Changed Python module compilation | Passed |
| `git diff --check` | Passed |
| Baseline ancestry check | Passed |
| Protected satellite-path diff | Empty |

An unrelated pre-existing Windows clock-resolution flake was observed during intermediate full-suite runs: `time.monotonic()` sometimes reported zero elapsed time around a 10 ms sleep. The isolated test passed, repeated direct measurements reproduced the platform behavior without any G1 import, and the final complete release run passed all 471 tests. No unrelated utility or timing test was modified.

## Production modules added

- Ground package exports.
- Shared canonical serialization and hash utility.
- Ground-station catalog domain and strict CSV loader.
- Deterministic region-balanced selection domain.
- Ground-design JSONL persistence and exact reconstruction.
- Passive scenario composition.

Existing satellite production modules modified: **none**.

## Catalog schema and validation

The exact canonical CSV schema contains station ID, canonical name, station class, latitude, longitude, altitude, region, country code, and enabled state.

Validation covers stable identifier syntax, NFC names, binary64 coordinates, terrestrial altitude limits, region slugs, country-code syntax, strict lowercase CSV booleans, exact headers, duplicate headers, duplicate IDs, empty catalogs, invalid values, and canonical ordering.

The synthetic fixture contains all classes, multiple regions, unequal regional pool sizes, and disabled stations. It is explicitly nonoperational.

## Canonicalization and catalog hash evidence

Canonical floats are finite binary64 values serialized with `.17g` as JSON strings. Negative zero maps to `0`. Names are stripped, NFC-normalized, and length-validated. Hash payloads use UTF-8 compact sorted-key JSON and explicit identity domains and versions.

Synthetic fixture catalog hash:

```text
0087139cdf9bbc4ea279788258b3d07fd5d13aeaacbea1364ba32ce43783b7f9
```

Tests prove path and row-order independence. Coordinate, name, class, enabled-state, region, addition, and removal changes alter catalog identity.

## Production-readiness validation

Generic validity and production readiness are separate. The small fixture loads generically and fails 50-per-class readiness as expected. A generated test catalog containing 50 enabled stations in each class passes readiness validation.

This proves software capacity only. No actual production research catalog was supplied.

## Selection algorithm

**Selection algorithm version:** `1`

Regions and stations use SHA-256 seeded ranks with explicit identifier collision tie-breakers. Round-robin traversal provides deterministic region balance relative to supplied taxonomy. The full catalog hash is excluded from ranking but included in selection identity.

Unequal civilian region pools of 8, 5, 4, and 3 stations were exercised. Before exhaustion, regional counts differ by no more than one; after exhaustion, remaining active regions retain the same bound.

## Nested-prefix evidence

For seed `42`, civilian prefixes were:

```text
3:  CIV_TEST_014, CIV_TEST_018, CIV_TEST_011
10: CIV_TEST_014, CIV_TEST_018, CIV_TEST_011, CIV_TEST_005,
    CIV_TEST_016, CIV_TEST_020, CIV_TEST_010, CIV_TEST_006,
    CIV_TEST_015, CIV_TEST_019
12: the 10-prefix followed by CIV_TEST_013, CIV_TEST_008
20: the 12-prefix followed by CIV_TEST_017, CIV_TEST_012,
    CIV_TEST_002, CIV_TEST_009, CIV_TEST_001, CIV_TEST_007,
    CIV_TEST_003, CIV_TEST_004
```

Tests prove count-only growth preserves prefixes and row reordering does not alter them. Eligible catalog mutation may change prefixes. Adding a disabled station leaves ordering and IDs unchanged while changing catalog and selection hashes.

## Mixed-composition evidence

The `6/3/3` design returned exactly 12 unique IDs in civilian, government, military concatenation order:

```text
CIV_TEST_014, CIV_TEST_018, CIV_TEST_011, CIV_TEST_005,
CIV_TEST_016, CIV_TEST_020,
GOV_TEST_003, GOV_TEST_002, GOV_TEST_004,
MIL_TEST_003, MIL_TEST_004, MIL_TEST_005
```

Every selected station exists, is enabled, and has the claimed class.

## Selection identity evidence

The synthetic mixed selection hash is:

```text
978fc00110c6149d0cf1adfa36b70007a447669ff7500071998aa2e60004dd2f
```

Selection identity includes catalog hash, selection version, seed, requested counts, class-grouped IDs, and combined IDs under the `satnet_ground_selection` identity domain/version. Different seeds produce different hashes even when a one-station catalog yields the same selected ID.

## Ground-design identity evidence

Ground-design schema version is `1`.

```text
disabled: 4f7aeba1327b846ee2001e9800856e4af31d664ef33868478e67c865529164aa
enabled:  96f4789d82bf5af74ca53830f91800104d5cd65f8fc3fa455f371c5b5327065e
```

Ground-design identity is independent of run ID and satellite configuration. Scenario identity composes but does not redefine satellite or ground identity.

## Manifest evidence

Canonical `.jsonl` tests prove:

- Exact enabled and disabled round trips.
- Numeric run-ID sorting.
- Sorted compact JSON keys and Unix newlines.
- UTF-8 writing.
- Atomic temporary-sibling replacement.
- No overwrite by default and explicit overwrite support.
- Empty file and empty-line rejection.
- Duplicate JSON-key detection.
- Unknown and missing field rejection.
- Exact Boolean and integer typing.
- Duplicate, missing, and orphan run-ID rejection.
- Exact satellite-hash matching.
- Complete one-to-one satellite/ground run coverage.

## Fail-closed reconstruction evidence

Reconstruction requires an explicitly supplied validated catalog for enabled mode and verifies its content hash. Tests reject missing catalogs, catalog mismatches, unknown IDs, disabled IDs, wrong-class IDs, wrong counts, reordered IDs, non-prefix IDs, selection-hash mismatches, ground-design-hash mismatches, invalid schema versions, and invalid SHA-256 values.

Explicit persisted IDs remain authoritative. The stored seed verifies deterministic prefixes and never silently replaces persisted IDs.

## Passive scenario evidence

`ScenarioDesign` retains the exact satellite configuration object, validates the existing satellite configuration hash against the ground record, and computes a versioned outer scenario hash.

```text
disabled scenario: cbb92267ac62419d5b890364cce380703c872ba8449cbb509be93b0838ac2e94
enabled scenario:  5cc73141c0fb2a4a5209ecffc4b2d8c94b59862936a3c46e8e5469395f540ab6
```

It imports only `Tier1RolloutConfig` from satellite simulation and has no rollout, graph, metric, model, or cache execution path.

## Satellite-invariance evidence

Enabled and disabled scenario designs were compared against direct satellite-only execution using the exact same `Tier1RolloutConfig` object.

Tests prove identical:

- Satellite configuration hash.
- SGP4 positions at every tested timestep.
- Satellite node identities.
- ISL edge identities and all attributes.
- Persistent satellite failure realization.
- Run-level and step-level metrics.
- Satellite labels.
- TGNN timesteps, node features, edge indices, edge attributes, and target tensors.
- Satellite graph-cache key.

AST guards prove no prohibited direct static imports in catalog, selection, or persistence modules and no ground imports in protected satellite modules.

## Protected satellite diff

Release command:

```text
git diff 0229a9366756cf8e922589253e681f2fa1d50989..HEAD -- src/satnet/network src/satnet/simulation src/satnet/models/gnn_dataset.py src/satnet/utils/graph_cache.py
```

Result: **empty**.

## Catalog retention and provenance

Canonical experiments must retain exact catalog content in immutable external or versioned artifact storage under its content-derived hash. Manifests contain content identity, not machine-local paths.

Optional source, license, taxonomy, designation, and review metadata is nonidentity provenance unless canonical station content changes.

## Known limitations

- No production 50-per-class research catalog has been supplied.
- Catalog sources, license, region taxonomy, classification rationale, completeness, and scientific suitability have not been reviewed.
- Region balance is not physical-distance or hemisphere optimization.
- No station coordinate propagation or visibility exists.
- No ground-link, failure, service, resilience, dataset, cache, or model semantics exist.
- The observed Windows 10 ms timer-resolution flake is unrelated to G1 and remains outside this task.

## Deferred Stage G2 work

Stage G2 may design coordinate-reference semantics, Earth rotation, ECEF conversion, elevation masks, and satellite-to-ground visibility. It must not infer physical or operational behavior from station class and must introduce explicit versions for any new physical, graph, failure, cache, dataset, or label semantics.

READY FOR GROUND VISIBILITY DESIGN certifies the catalog and selection software contract. It does not certify that a production research catalog has been supplied or scientifically approved.

Production research catalog availability: NOT AVAILABLE
Production research catalog scientific review: NOT PERFORMED

READY FOR GROUND VISIBILITY DESIGN
