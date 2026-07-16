# Ground Segment Stage G3 Integrated Graph

## Scope

Stage G3 creates a separate deterministic integrated representation containing operational satellite nodes, selected G1 ground-station nodes, operational ISLs, and geometrically visible G2 satellite-ground links.

The integrated graph is a separate derived representation. It does not replace or mutate the validated satellite-only graph.

G3 defines no ground failures, service fractions, attachment thresholds, resilience metrics, labels, dataset migration, model inputs, training, bandwidth, routing, scheduling, traffic, or link budgets.

## Authoritative immutable state

NetworkX graphs are mutable and are not authoritative G3 state. G3 stores immutable canonical graph attributes, node records, and edge records. `to_networkx()` returns a fresh derived `nx.Graph` on every call.

Mutating a derived graph cannot alter canonical records, graph identity, or subsequent views.

## Canonical attributes

Every graph, node, and edge attribute uses an explicit type tag:

- `none`
- `boolean`
- `integer`
- `float`
- `string`

Booleans and integers remain distinct. Finite floats use G1 canonical binary64 strings and restore as Python floats. Strings remain exact, including numeric-looking strings. `None` round-trips exactly.

Nested structures, bytes, arbitrary objects, unnormalized NumPy scalars, NaN, and infinity fail in model version 1. Attribute names are unique nonempty strings sorted by name.

## Operational source graph

The read-only source adapter constructs `HypatiaAdapter` from the existing rollout configuration, generates TLEs and ISLs through production code, copies each graph, applies the supplied persistent failed nodes and edges, and canonicalizes immediately.

No ISLs or failures are generated independently by G3. One source snapshot is produced for every configured timestep.

G3 version 1 accepts exactly an undirected, non-multigraph `nx.Graph`. Directed and multigraph variants fail.

Operational source snapshots contain:

- Timestep and canonical UTC timestamp.
- Satellite configuration hash.
- Canonical graph metadata.
- Numeric-ordered satellite node records.
- Canonical ascending ISL records.
- Operational graph hash.

All-satellites-failed timesteps remain valid with zero nodes and zero edges.

## Typed node identity

`IntegratedNodeRef` is a strict union:

- Satellite: exact nonnegative integer satellite ID and no ground ID.
- Ground station: valid G1 station ID and no satellite ID.

Canonical order is all satellites by numeric ID followed by ground stations by station ID. This is serialization policy only and implies no operational priority.

Satellite node attributes are copied exactly from the source graph. No G3 metadata is injected into their attribute dictionaries.

Ground nodes contain:

- Namespaced integrated node kind.
- Ground station ID.
- Station class.
- Latitude, longitude, and altitude.
- Region and country code.
- Enabled state.
- Ground-design hash.

Every selected station remains present, including stations with no visible satellites.

## Typed edge identity

`IntegratedEdgeKind` distinguishes:

- `inter_satellite`
- `satellite_ground`

ISL endpoints are satellite references ordered by ascending numeric ID. Their source attributes remain exact.

Satellite-ground edges always use satellite first and ground station second. They are undirected topological access edges. This does not imply equal uplink/downlink budgets or communications viability.

Each satellite-ground edge contains exact G2 values and identities:

- Station ID.
- Integer satellite ID.
- Elevation.
- Slant range.
- Visibility model version.
- Frame contract version.
- WGS84 model version.
- Visibility policy hash.
- Visibility snapshot hash.

Only visible observations create edges. Geometry is copied exactly and is not recomputed or rounded in G3.

## G1 selection binding

Construction reconstructs the exact G1 ground selection from the supplied catalog and ground-design record. Catalog, class, count, order, selection hash, ground-design hash, and satellite configuration identity must validate.

Disabled ground designs fail because an integrated ground graph requires selected stations.

## G2 visibility binding

G3 validates the complete Cartesian product:

```text
selected G1 station IDs × operational source satellite IDs
```

against all G2 observations. A union-only satellite check is insufficient and is not used.

Required equality covers:

- Every expected station-satellite pair exactly once.
- No missing or extra pair.
- Every selected station in the station mapping.
- Every mapped satellite operational in the source graph.
- Exact visible subset.
- Matching timestamp and satellite, ground, policy, model, frame, WGS84, and snapshot identities.

For all-failed timesteps, observations and links are empty while every selected station remains mapped to an empty tuple.

## Integrated construction

Canonical construction:

1. Reconstructs the G1 selection.
2. Validates source graph and G2 identities.
3. Validates the full observation Cartesian product.
4. Wraps source satellite IDs in typed references.
5. Preserves source graph, node, and ISL attributes.
6. Adds all selected ground nodes.
7. Adds one edge for every visible G2 pair.
8. Detects duplicate nodes and edges.
9. Computes counts from canonical records.
10. Computes deterministic graph identity.
11. Validates satellite projection and visible-edge equality.

Inputs are never mutated.

## Projected satellite invariant

Because integrated satellite keys are typed, exact preservation is tested through projection.

`project_satellite_subgraph()` selects typed satellite nodes and ISLs, maps nodes back to integer IDs, restores exact source attributes and graph metadata, and returns a fresh ordinary `nx.Graph`.

`validate_satellite_projection()` verifies exact graph type, directedness, multigraph status, graph metadata, integer node set, node attributes, ISL endpoints, and ISL attributes. It never relies only on counts or hashes.

## Visible-edge invariant

The canonical integrated satellite-ground edge set must equal the exact G2 visible observation set using `(satellite_id, station_id)` pairs.

Validation rejects missing, extra, duplicate, nonvisible, wrong-endpoint, or attribute-inconsistent edges.

## Graph identity

Graph identity domain is `satnet_integrated_ground_graph`, version `1`.

The hash contains:

- Model version.
- Timestep and canonical UTC timestamp.
- Satellite configuration hash.
- Ground-design hash.
- Visibility policy hash.
- Complete visibility snapshot hash.
- Canonical graph metadata.
- Every canonical typed node and attribute.
- Every canonical typed edge and attribute.

It excludes iteration order, object representations, memory addresses, file paths, output directories, hosts, and wall-clock generation time.

A changed nonvisible G2 observation leaves graph structure unchanged but changes the G2 snapshot hash and therefore the G3 graph hash.

## Persistence

Standalone `.jsonl` records contain:

- Schema, model, and record identity versions.
- Run and timestep IDs.
- Canonical UTC timestamp.
- Upstream scientific hashes.
- Node and edge counts.
- Canonical graph attributes, nodes, and edges.
- Graph and record hashes.

The writer validates before writing, uses compact sorted-key UTF-8 JSON, numeric run/timestep order, Unix newlines, a synchronized temporary sibling, atomic replacement, cleanup on failure, and no overwrite by default.

The reader rejects duplicate JSON keys, empty content, unknown/missing fields, wrong exact types, unsupported attributes, invalid unions, invalid counts, and hash mismatches.

## Run binding and completeness

G1 ground-design, G2 visibility, and G3 integrated records must share the same integer run ID. Hash equality alone is insufficient because ground-design identity is intentionally run-independent.

For every run, G3 record keys exactly equal operational source timestep keys. Missing, extra, orphan, duplicate, all-failed, and zero-ground-link records are handled fail-closed.

## Canonical replay

Replay begins with G2 visibility records, not an arbitrary snapshot.

It:

1. Reconstructs operational satellite position and graph sequences from configuration and failures.
2. Verifies and replays every G2 record through G2 production code.
3. Reconstructs the exact G1 selection.
4. Rebuilds each G3 graph.
5. Compares canonical graph attributes, nodes, and edges.
6. Validates satellite projection and visible-edge equality.
7. Recomputes graph and record hashes.

Persisted graph structure is never trusted without reconstruction.

## Diagnostic harness

Synthetic diagnostics cover:

- One station and one visible satellite.
- One station and one nonvisible satellite.
- Multiple stations and satellites.
- A station with no access.
- All satellites failed.
- A complete satellite graph with zero ground links.

The harness reports structural counts, visible station mappings, both invariant results, and graph hash. It computes no resilience metrics.

## Scientific limitations

A satellite-ground edge represents geometric visibility above the configured minimum-elevation threshold. It does not prove RF viability, bandwidth availability, scheduling, routing, or service delivery.

Ground-station class remains categorical scenario metadata and does not alter graph physics or edge behavior in G3.

## Deferred G4 work

G4 may define ground service, attachment, resilience, and labeling semantics using validated G3 graphs. Those contracts must preserve the independent satellite-only metrics and require separate scientific versions.
