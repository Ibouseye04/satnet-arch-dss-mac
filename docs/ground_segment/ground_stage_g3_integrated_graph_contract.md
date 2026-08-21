# Ground Segment Stage G3 Integrated Graph Contract

**Status:** Approved

**Decision date:** 2026-07-16

**Validated G2 implementation SHA:** `5cc9e71399cc6587162634958e0173985c4ee228`

**G2 documentation/tagged HEAD and G3 base:** `ddfae1d271be304e95e4168f1256d74aef240543`

## Scope

Stage G3 creates a separate deterministic integrated satellite-ground graph from the validated operational satellite graph, exact G1 station selection, and verified G2 visibility record.

G3 adds no failures, metrics, thresholds, labels, services, dataset fields, caches, model features, training, traffic, routing, capacity, or communications link budget.

## Source satellite graph

The canonical source graph is reconstructed through `HypatiaAdapter`, the existing topology policy, and the supplied persistent failure realization. G3 samples no failures and creates no ISLs independently.

G3 version 1 accepts exactly an undirected non-multigraph `networkx.Graph`. Source graph metadata, integer satellite IDs, node attributes, ISL endpoints, and ISL attributes are canonicalized immediately into immutable records. Adapter-owned graph objects are never authoritative or exposed.

One source snapshot exists for every configured timestep, including an empty all-satellites-failed graph.

## Typed identity and projected preservation

Integrated nodes use typed references. Satellite keys contain exact integer IDs; ground keys contain exact G1 station IDs. Integrated satellite keys are not literally equal to source integer keys.

Exact satellite preservation is established by deterministic projection. Projecting typed satellite nodes and ISL edges back to integer IDs must exactly restore the source graph type, metadata, node IDs, node attributes, edge endpoints, and edge attributes.

Satellite attribute dictionaries receive no G3 metadata. Typed node references contain integrated node kind and source identity. Ground nodes contain only contracted ground metadata.

## Canonical attributes

All graph, node, and edge attributes use explicit type tags: none, Boolean, integer, float, or string.

Booleans and integers are distinct. Finite floats use the G1 canonical binary64 string. Strings remain exact. Reconstruction restores exact scalar Python types.

Nested structures, bytes, arbitrary objects, NumPy scalars without explicit normalization, NaN, and infinity fail in G3 version 1. Attribute names are unique nonempty strings and records are sorted by name.

## Immutable authority

Immutable canonical graph attributes, node records, and edge records are authoritative. NetworkX graphs are fresh derived views returned by `to_networkx()`.

Mutating a returned graph cannot alter canonical records, graph identity, or later derived graph views. Graph hashes never consume mutable NetworkX iteration order or object representations.

## Ordering

Canonical node order is numeric satellites followed by lexicographic ground station IDs.

ISL endpoints use smaller integer satellite ID first. Satellite-ground endpoints use satellite first and ground station second. Edge order is edge kind followed by explicit canonical endpoint keys. Duplicate canonical nodes and edges fail before graph materialization.

## G1 and G2 binding

G3 reconstructs the exact G1 selection and verifies catalog, class, count, ordering, selection, and ground-design identity.

Before adding edges, G3 validates the complete Cartesian product of selected station IDs and operational satellite IDs against every G2 observation. The visible station mapping must contain every selected station. All-failed timesteps retain all ground mappings with empty satellite tuples.

Only visible G2 observations produce satellite-ground edges. Elevation and slant range are copied exactly without geometry recomputation or rounding. Edge attributes bind G2 visibility model, frame, WGS84, policy, and snapshot identity.

Canonical G3 replay starts from a verified G2 visibility record, replays G2 through production code, then builds G3. G1, G2, and G3 run IDs must match.

## Graph identity and persistence

Graph identity includes explicit domain/version, model version, timestep, canonical UTC, satellite, ground, policy, and visibility snapshot hashes, and every canonical graph attribute, node, and edge.

Standalone JSONL records persist immutable canonical records, graph hash, and a run/timestep record hash. Source keys and record keys must match exactly, including empty and zero-link timesteps.

## Scientific interpretation

The integrated graph is a separate derived representation. It does not replace or mutate the validated satellite-only graph.

A satellite-ground edge represents geometric visibility above the configured minimum-elevation threshold. It does not prove RF viability, bandwidth availability, scheduling, routing, or service delivery.

Ground-station class remains categorical scenario metadata and does not alter graph physics or edge behavior in G3.
