# Stage A TGNN Dataset v1 Independent Gate

Verdict: **STAGE A TGNN DATASET V1 ACCEPTED — READY FOR TRAINING-PLAN FREEZE**

This is an independent, read-only dataset acceptance gate. It did not rebuild the TGNN dataset, rebuild or modify the accepted RF dataset, train or tune a model, inspect test targets, or modify production, replay, acceptance, freeze, or unrelated worktree files.

- Independent gate implementation HEAD: `f9cd85bbf762dce673a5877e080e675bf8721290`
- Dataset construction implementation SHA-256: `acd493b71e24ba28a5696e50d0467cb6ab91ad0c49a252139956cec1a1aeed3b`
- Dataset root: `C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_tgnn_dataset_v1`
- Authoritative freeze manifest SHA-256: `b2d1fbd9510d3d828fe051b4da04f6747088ae00d78d65e0a9851dc85559844e`
- Contract specification SHA-256: `fc13a33a0a1af435189990e54b6c68efa56bbc207c0cae6e27e12bf031605930`

## Split counts

- Train: 350 sequences / 70 designs
- Validation: 75 sequences / 15 designs
- Sealed test: 75 sequence indexes / 15 designs
- Five realizations per design; zero design overlap; zero run overlap; zero duplicate run keys; zero duplicate design/realization pairs

## Graph scope

Independent source tracing accepts **Scope B: combined satellite-ground heterogeneous temporal graph**. Frozen G3 records contain satellite and selected ground-station nodes with inter-satellite and satellite-ground edges. Failed satellites are absent from inherited G3 operational graphs; failed selected ground stations remain present and are encoded with `operational_indicator=0`. Variable node populations are represented by offsets and identity indexes; no padding or masking is required.

## Results

- Independent serialization parser: PASS for 500/500 sequences; 27 zero-edge snapshots were checked with exact `[2, 0]` / `[0, 10]` shapes.
- Node schema: PASS; exactly 14 ordered float64 features.
- Edge schema: PASS; exactly 10 ordered float64 attributes, directed reverse pairs, no self-loops.
- Temporal integrity: PASS for 500 sequences and 5,500 snapshots.
- Leakage review: PASS; test target sources were not read.
- Provenance: PASS for 500/500 result, scientific-inventory, G3, and G5 hashes.
- Test seal: PASS; 75 test sequence indexes, no materialized targets, outcome fields unused.

Detailed evidence is in `tgnn_serialization_verification.json`, `tgnn_graph_scope_verification.json`, `tgnn_schema_verification.json`, `tgnn_temporal_integrity_report.json`, and `tgnn_dataset_gate_report.json`. `artifact_inventory.json` is self-excluding and records the SHA-256 of the six other gate artifacts.
