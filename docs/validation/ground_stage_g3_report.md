# Ground Segment Stage G3 Validation Report

**Validation date:** 2026-07-16

**Branch:** `feature/integrated-ground-graph`

**Validated G2 implementation SHA:** `5cc9e71399cc6587162634958e0173985c4ee228`

**G2 documentation/tagged HEAD:** `ddfae1d271be304e95e4168f1256d74aef240543`

**G3 branch base SHA:** `ddfae1d271be304e95e4168f1256d74aef240543`

**Validated G3 implementation SHA:** `8f63531c9bedac97c1c28d76e44f0a8b24ca909a`

## Scope result

Stage G3 implemented immutable typed integrated graph records, production operational graph reconstruction, deterministic construction, exact satellite projection, visible-edge equality, standalone persistence, verified G2-first replay, diagnostics, and upstream isolation.

No ground failures, service fractions, thresholds, resilience metrics, labels, dataset migration, model integration, training, routing, scheduling, capacity, or link budgets were implemented.

## Test summary

| Gate | Result |
|---|---:|
| G3 baseline complete suite | 582 passed |
| Final focused G3 suite | 60 passed |
| Final complete suite | 642 passed |
| Changed Python module compilation | Passed |
| Diagnostic harness | Passed |
| `git diff --check` | Passed |
| G2 base ancestry | Passed |
| Protected upstream diff | Empty |

The known Windows 10 ms timer-resolution test flaked during an intermediate baseline/full run but passed in the final complete release suite. No unrelated timing code was modified.

## Production modules

G3 added:

- Type-tagged canonical graph attributes.
- Typed integrated and operational graph records.
- Immutable operational and integrated snapshots.
- Operational satellite graph reconstruction adapter.
- Integrated graph construction and invariant validators.
- Integrated graph JSONL persistence and replay.
- Structural diagnostic harness.

Protected upstream satellite modules modified: **none**.

## Typed node contract

Satellite nodes use `IntegratedNodeRef` with exact nonnegative integer IDs. Ground nodes use valid G1 station IDs. Invalid mixed, empty, Boolean, negative, string, or float variants fail.

Canonical order is numeric satellites followed by lexicographic ground IDs.

Satellite attribute dictionaries remain exact source attributes with no injected G3 metadata. Ground nodes retain selected-station metadata and ground-design identity.

## Typed edge contract

ISLs connect two typed satellites with ascending numeric endpoints and exact source attributes.

Satellite-ground edges use satellite first and ground station second. Only visible G2 observations produce edges. Elevation and range are copied exactly.

Edges bind visibility model, frame, WGS84, policy, and snapshot identity.

## Canonical attribute evidence

Supported exact scalar types are none, Boolean, integer, finite float, and string. Tests prove:

- Boolean and integer remain distinct.
- Float and numeric-looking string remain distinct.
- `None` round-trips.
- Floats use canonical binary64 strings.
- Attributes are name-sorted and duplicate names fail.
- Nested values, bytes, objects, NaN, and infinity fail.

## Operational source reconstruction

The source adapter uses `HypatiaAdapter`, existing TLE/ISL generation, and the supplied failure realization. It produces every configured timestep, removes persistent failed nodes and edges, preserves source graph metadata/attributes, samples no failures, and supports empty all-failed graphs.

G3 accepts exactly undirected non-multigraph `nx.Graph` inputs. Directed and multigraph variants fail.

## Immutable graph evidence

Canonical records are authoritative. Tests mutate and clear derived NetworkX graphs and prove:

- Canonical nodes and edges remain unchanged.
- Graph hash remains unchanged.
- A subsequent `to_networkx()` call returns an independent complete graph.
- Adapter-owned source graphs are never exposed.

## Satellite projection evidence

Typed integrated satellite nodes project back to original integer IDs. Permanent validation compares exact graph type, metadata, node IDs, node attributes, ISL endpoints, and ISL attributes.

No comparison relies solely on counts or fingerprints. Projection passed for synthetic and production-reconstructed graph cases.

## G1/G2 binding evidence

Construction reconstructs the exact G1 station selection and validates the full Cartesian product of selected stations and operational satellites represented by G2 observations.

Tests reject union-equal but station-incomplete observations. All-failed Cartesian products validate with empty observations and complete empty station mappings.

G1 ground-design, G2 visibility, and G3 record run IDs must match.

## Visibility-edge equality

Every visible observation has exactly one satellite-ground edge. Nonvisible observations produce none. Validation compares canonical `(satellite_id, station_id)` pairs and exact attributes.

Tests cover multiple visible satellites, shared satellites, no-access stations, zero visible links, and all satellites failed.

## Deterministic identity evidence

Diagnostic cases produced:

| Case | Satellites | Ground | ISLs | Sat-ground | Graph hash |
|---|---:|---:|---:|---:|---|
| One visible pair | 1 | 1 | 0 | 1 | `6bd92ee76c4d3e8e141eb33ec4ae7ac986e6896fccd47ae9964b820fc901cf63` |
| One nonvisible pair | 1 | 1 | 0 | 0 | `65cc21ab407db45817f014dc4165b207f5e625c03e0faf86063e3b311a214122` |
| Multiple stations/satellites | 2 | 2 | 1 | 4 | `73a05c2cbe08ed91afdb15eac832b77a143b8dbe292545e0709a14c85354607b` |
| Station with no access | 1 | 2 | 0 | 0 | `a041a09a539fbb88ceae8ca3a5fa25fc47c87cfe189a15e77dd41d839848c5be` |
| All satellites failed | 0 | 2 | 0 | 0 | `a054fac0b3157d8f9719e838f7631322f758bedc02e1e7ddbb01179502213416` |
| Satellite graph, zero ground links | 2 | 1 | 1 | 0 | `d4f57481104d9902aa9ae8a63f3e54cecee1dcd4254edcf04dbab605472cc5db` |

Tests prove source insertion order and output path do not affect identity. Changing a visible link or exact edge geometry changes graph identity. Changing only nonvisible geometry leaves structure unchanged but changes the bound G2 snapshot and G3 graph hash.

## Persistence and replay

Canonical `.jsonl` tests prove exact round trip, sorted records/keys, UTF-8 Unix newlines, atomic synchronized replacement, no overwrite by default, duplicate-key rejection, empty-line rejection, strict types, and hash verification.

Corruption tests reject missing/extra nodes and edges, modified satellite/ISL attributes, wrong kinds/endpoints, timestamps, visibility identities, graph hashes, record hashes, duplicate keys, and missing timesteps.

Canonical replay begins with verified G2 records, reconstructs positions and operational graphs, replays G2, rebuilds G3, and compares canonical records and both hashes. Normal and all-failed temporal replays completed without mismatch.

## Upstream isolation

Existing G1 catalog, selection, and ground-design identities remain unchanged. Existing G2 policy, snapshot, and record identities remain unchanged. Satellite configuration, topology, failures, labels, TGNN tensors, and graph-cache identity remain protected by the complete upstream suite.

Pure G3 modules have no model, metric, experiment, or cache imports. The graph adapter imports only the existing canonical satellite configuration, failure, and topology boundary.

## Protected upstream diff

Release command:

```text
git diff ddfae1d271be304e95e4168f1256d74aef240543..HEAD -- src/satnet/network src/satnet/simulation/tier1_rollout.py src/satnet/models/gnn_dataset.py src/satnet/utils/graph_cache.py
```

Result: **empty**.

## Known limitations

- Satellite-ground edges indicate G2 geometric visibility only.
- No RF, optical, atmosphere, terrain, interference, capacity, scheduling, routing, or service semantics exist.
- No ground failures exist.
- NetworkX views are mutable derived objects; immutable canonical records are authoritative.
- No integrated graph cache, dataset schema, metric, label, or model feature exists.

## Deferred G4 work

G4 may define service and resilience semantics using validated G3 records. Space-GCC attachment, class-specific service, thresholds, labels, datasets, caches, and model changes require separate contracts and scientific versions.

READY FOR GROUND SERVICE METRIC DESIGN
