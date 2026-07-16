# SATNET Tier 1 Corrected Dataset Pilot Validation

**Validation date:** 2026-07-16

**Scope:** First canonical dataset pilot after V-01 through V-06 remediation

**Restriction:** No RF or TGNN model was trained. No historical dataset, model, prediction, split manifest, comparison table, or cache was modified or reused.

## Provenance

| Item | Value |
|---|---|
| Git SHA | `eb3def1ae2c6f70efd98a2a6448a0a667cdd35c6` |
| Exact Git tag | `tier1-validity-remediated-v1` |
| Branch | `main` |
| Tracked working tree at generation | Clean |
| Permitted untracked files at generation | Atomic gameplan and original defect-verification report |
| Python | `3.11.9` |
| NetworkX | `3.6.1` |
| NumPy | `1.26.4` |
| pandas | `2.3.3` |
| SGP4 | `2.26` |
| PyTorch | `2.12.1` |
| PyTorch Geometric | `2.8.0` |
| PyTorch Geometric Temporal | `0.56.2` |
| pytest | `9.1.1` |
| Dataset path | `data/tier1_validity_pilot_v2/` |
| Runs SHA-256 | `d02dd1cb58e6c56c07b863a3b9dd7f3f520c1509fee14c1d99a6d422d358112a` |
| Steps SHA-256 | `798015bf86dcaf5d9d15f07225bc3c50306dd2463199d5d22b4d6c209ae64bc2` |
| Combined dataset SHA-256 | `80737a60a2cee4021239be371af25ac2aaab5f49016ae0428104295b2f869b3b` |
| Schema version | `2` |
| Dataset version | `tier1_temporal_connectivity_v2` |
| Physics/topology version | `tier1_space_segment_physics_v2` |
| Orbital engine | `sgp4` |
| Pilot generation seed | `20260716` |

The complete dependency inventory, 25 configuration hashes, file hashes, source state, and generator settings are retained in `dataset_metadata.json` and `generation_config.json` inside the isolated pilot directory.

## Configuration-selection strategy

The first 25 historical configurations could not be reused exactly without guessing. Historical schema-v1 rows omit three graph-defining fields required by the repaired canonical contract:

- `phasing_factor`
- `max_isl_distance_km`
- `orbital_engine`

The pilot therefore used the authorized fresh deterministic strategy with seed `20260716`.

The design-space ranges were taken from the immutable former 500-run table rather than inferred from incomplete documentation:

| Parameter | Pilot setting |
|---|---|
| Runs | 25 |
| Orbital planes | 3–8 |
| Satellites per plane | 4–12 |
| Inclination | 30–98 degrees |
| Altitude | 300–1,200 km |
| Duration | 30 minutes |
| Step interval | 60 seconds |
| Epoch | J2000 default, exported per row |
| Walker phasing factor | 1 |
| Maximum ISL distance | 10,000 km, inclusive |
| ISL policy | `grid_adaptive` |
| Adjacent search K | 1 |
| Total incident inter-plane capacity | 1 |
| Node-failure probability | 0–0.2 |
| Edge-failure probability | 0–0.3 |
| Failure model | `persistent_temporal_union_edges_v1` |
| GCC partition threshold | 0.8 |

## Generation summary

The pilot was produced through `scripts/export_design_dataset.py`, which invokes `generate_tier1_temporal_dataset`, `run_tier1_rollout`, `HypatiaAdapter`, and the canonical schema-validating CSV writer.

| Item | Result |
|---|---:|
| Requested runs | 25 |
| Successful runs | 25 |
| Failed runs | 0 |
| Steps per run | 31 |
| Total timesteps | 775 |
| Simulation runtime reported by generator | Approximately 1.63 seconds |

Generated pilot files:

- `tier1_design_runs.csv`
- `tier1_design_steps.csv`
- `dataset_metadata.json`
- `generation_config.json`
- `validation_results.json`
- `validation_report.md`

The output directory did not exist before generation. The complete dataset remained in memory until all 25 simulations completed; a propagation or configuration exception would have prevented canonical CSV writing.

## Validation matrix

| Validation | Status | Evidence |
|---|---|---|
| Dataset structure | PASS | 25 unique runs, 775 unique run-step keys, 31 contiguous steps/run, no orphans, no nonfinite numeric values, valid probabilities, all 25 hashes recomputed |
| SGP4 validity | PASS | 31,775 positions checked; zero invalid/origin positions and zero accepted propagation errors |
| Maximum ISL distance | PASS | Maximum accepted distance 7,835.755 km versus 10,000 km limit; zero over-limit accepted links |
| Adaptive capacity | PASS | Maximum total incident inter-plane degree 1 at capacity 1; zero violating nodes and timesteps |
| Exact temporal replay | PASS | 25 runs and 775 timesteps replayed through the TGNN loader; zero structural, feature, or label mismatches |
| Failure realization consistency | PASS | 24 runs had nonempty failures; all 24 differed from nominal and replayed exactly; missing metadata was rejected |
| Cache identity | PASS | Valid temporary write/hit; all required mutations changed identity; malformed and stale entries rejected |
| Experiment identity | PASS | Compatible reuse accepted; seven incompatible scientific changes rejected without training |
| Target distribution reviewed | PASS | Both classes present and regression target has nonzero variance |
| Historical comparison | LIMITED | Exact v1 pairing is impossible without guessing three omitted fields; only unpaired aggregate comparison is reported |

## Dataset structure

- Exactly 25 rows and 25 unique `run_id` values exist.
- Exactly 775 step rows exist.
- Every run contains steps `0..30` with no gaps.
- No duplicate `(run_id, t)` keys exist.
- No orphan step rows exist.
- All schema-v2 run and step fields are present.
- All numeric values expected to be finite are finite.
- Node and edge failure probabilities are within `[0, 1]`.
- Every run records schema `2` and dataset version `tier1_temporal_connectivity_v2`.
- Every run records `sgp4`, the adaptive policy, K=1, capacity=1, range=10,000 km, phasing=1, and an explicit epoch.

Timestamps were validated as `epoch_iso + t * step_seconds` for all 775 rows. Schema-v2 step rows store the authoritative integer step rather than a redundant timestamp column.

All 25 configuration hashes recomputed exactly from the raw CSV serialization using `csv.DictReader` and Python `float()`. pandas' C parser can shift low-order floating-point bits, so parsed pandas floats were not used for byte-exact hash reconstruction. This does not change graph replay: the independently parsed TGNN path still produced zero mismatches.

## SGP4 validity

Every satellite in every run was propagated again at every timestep through the canonical SGP4 path.

| Metric | Result |
|---|---:|
| Positions checked | 31,775 |
| Recorded nonzero SGP4 errors | 0 |
| Nonfinite positions | 0 |
| Origin-position fallbacks | 0 |
| Minimum position norm | 6,682.922 km |
| Maximum position norm | 7,552.112 km |
| Minimum altitude | 311.922 km |
| Maximum altitude | 1,181.112 km |

Every position norm exceeded the configured Earth radius. Because the repaired propagator raises on every nonzero SGP4 result, no errored propagation could be written as a successful run.

## Maximum ISL distance

All accepted satellite-to-satellite edges were checked against their run's hard inclusive maximum.

| Metric | Result |
|---|---:|
| Configured maximum | 10,000 km |
| Maximum accepted ISL distance | 7,835.755 km |
| Accepted links above maximum | 0 |
| Candidate links rejected by distance | 15,853 |
| Candidate links rejected by LOS | 48,289 |
| Candidate links rejected by link budget | 0 |

Rejection counts are aggregated `ISLComputationStats` returned by the production adapter during independent replay of all 25 configurations. Distance rejection occurred before LOS and link-budget evaluation as required.

## Adaptive inter-plane capacity

For every base graph timestep, intra-plane ring links were excluded and total incident inter-plane degree was calculated at both endpoints.

| Metric | Result |
|---|---:|
| Configured capacity | 1 |
| Maximum observed incident inter-plane degree | 1 |
| Violating nodes | 0 |
| Violating timesteps | 0 |
| Node-timestep observations with degree 0 | 9,661 |
| Node-timestep observations with degree 1 | 22,114 |
| Seam links across all timesteps | 2,058 |
| Non-seam inter-plane links across all timesteps | 8,999 |

Both seam and non-seam adaptive selection were exercised. The required result of zero capacity violations was achieved.

## Exact TGNN graph replay

All 25 sequences were reconstructed through `SatNetTemporalDataset` for both `partition_any` and `gcc_frac_min_original` targets. The independently generated production graph was failure-adjusted from the exported realization and compared against TGNN reconstruction at each timestep.

Compared values included:

- Original satellite node identities.
- Contiguous canonical node mapping.
- Undirected edge identities.
- Persistent failed nodes and edges.
- Derived timestamp index.
- TGNN node features.
- TGNN edge attributes at absolute tolerance `1e-7`.
- Node and edge counts.
- Connected-component count.
- Largest connected-component size.
- Original-denominator GCC fraction.
- Step partition label.
- Run-level classification label.
- Run-level regression label.

| Replay result | Count |
|---|---:|
| Runs replayed | 25 |
| Timesteps replayed | 775 |
| Structural mismatches | 0 |
| Feature or floating-attribute mismatches | 0 |
| Classification or regression label mismatches | 0 |

## Failure realization consistency

| Metric | Result |
|---|---:|
| Runs with nonempty failure realization | 24 |
| Failed runs that differed from nominal topology | 24 |
| Exported count/JSON mismatches | 0 |
| Failed graph reconstructed as nominal | 0 |
| Missing failure metadata rejected | Yes |

The pilot therefore exercises both node/edge failure export and fail-closed reconstruction behavior. No production failure probability or sampling logic was changed to obtain this coverage.

## Graph-cache identity

Cache validation used an OS temporary directory; no repository cache file remains.

An actual schema-v5 graph sequence was written through `SatNetTemporalDataset`, loaded, structurally validated, and reused by an identical request without rebuilding the adapter. Independent mutations to every requested identity dimension produced a different key:

- Orbital engine.
- Physics/topology version.
- Maximum ISL distance.
- Adaptive capacity.
- ISL policy.
- Epoch.
- Phasing factor.
- Link-budget configuration.
- Failure realization.
- Realization seed.

A malformed payload and a schema-v4 metadata mutation were both rejected.

## Experiment identity smoke check

No model process was started. Temporary completed metrics and split files were used only to exercise identity guards.

- An exact compatible condition reused its completed output without subprocess execution.
- Changes to dataset content identity were rejected.
- Changes to split content identity were rejected.
- Changes to target were rejected.
- Changes to TGNN input mode were rejected.
- Changes to seed were rejected.
- Changes to Chebyshev K were rejected.
- Changes to hidden dimension were rejected.

All temporary identity artifacts were removed after the check.

## Target and topology distributions

### Targets

| Metric | Result |
|---|---:|
| `partition_any = 0` | 6 runs (24.0%) |
| `partition_any = 1` | 19 runs (76.0%) |
| `gcc_frac_min` minimum | 0.027778 |
| `gcc_frac_min` maximum | 0.916667 |
| `gcc_frac_min` mean | 0.371035 |
| `gcc_frac_min` median | 0.133333 |
| `gcc_frac_min` population standard deviation | 0.346667 |

Both classification classes are represented. The regression target spans 0.889 and has meaningful nonzero variance. These values establish pilot suitability for canonical dataset design, not model performance.

### Topology

| Metric | Result |
|---|---:|
| Satellites per run, minimum | 15 |
| Satellites per run, maximum | 80 |
| Satellites per run, mean | 41.0 |
| Effective edges/timestep, minimum | 0 |
| Effective edges/timestep, maximum | 93 |
| Effective edges/timestep, mean | 28.215 |
| Effective edges/timestep, median | 8 |
| Accepted intra-plane edges across base timesteps | 21,266 |
| Accepted inter-plane edges across base timesteps | 11,057 |
| Effective components, minimum | 1 |
| Effective components, maximum | 35 |
| Effective components, mean | 12.272 |
| Minimum GCC fraction | 0.027778 |

## Historical comparison

Exact paired topology comparison was not scientifically admissible. Historical schema-v1 rows do not record phasing, maximum range, or orbital engine, so constructing their graphs under repaired code would require silent assumptions. The pilot deliberately used a fresh seed and does not claim per-run or per-edge causal deltas.

The only legitimate comparison is an unpaired descriptive comparison over the same sampled design-space ranges:

| Metric | Historical v1, 500 runs | Corrected v2 pilot, 25 runs |
|---|---:|---:|
| `partition_any = 0` | 171 (34.2%) | 6 (24.0%) |
| `partition_any = 1` | 329 (65.8%) | 19 (76.0%) |
| Mean `gcc_frac_min` | 0.506238 | 0.371035 |

The 10.2 percentage-point difference in positive class rate and lower pilot mean GCC cannot be attributed solely to corrected capacity because the samples are not paired and the pilot is small. Runs with changed edge sets, changed timesteps, edges removed by corrected capacity, edges removed by range, and paired label deltas remain **not estimable** from historical artifacts.

## Scientific observations

- Correct adaptive capacity is directly observed: total incident inter-plane degree never exceeds 1 across 31,775 node-timestep observations.
- The repaired range gate is active, rejecting 15,853 candidates; no accepted edge exceeds 10,000 km.
- Both seam and non-seam inter-plane links remain present under endpoint-aware capacity enforcement.
- All 25 configurations completed. No sampled altitude, inclination, plane-count, or density region systematically failed generation in this pilot.
- SGP4 propagation remained physically valid throughout the 775 temporal states.
- Twenty-four runs contain nonempty persistent failures, and every such reconstruction differs from its nominal graph.
- Corrected labels have usable pilot variation: both classes occur and continuous GCC minima have substantial spread.
- The exact amount by which V-03 changed historical topology cannot be recovered without archived v1 edge sequences or complete v1 graph identity. No speculative number is reported.
- Corrected graph behavior matches the approved SGP4, hard inclusive range, total incident capacity, fail-closed reconstruction, cache identity, and experiment identity contracts.

## Release recommendation

All structural validations pass. There are zero adaptive-capacity violations, zero over-distance accepted links, zero replay mismatches, exact failure replay, no silently accepted SGP4 error, successful cache and experiment identity guards, and usable target variation.

This recommendation authorizes canonical dataset **design**, not immediate generation of 500 runs, split creation, model training, or ground-segment work.

READY FOR CANONICAL DATASET DESIGN
