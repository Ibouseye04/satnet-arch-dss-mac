# SATNET Adaptive Remediation Phase 0 Qualification

## Authorization boundary

This qualification prepared and validated the adaptive contract and ran only
small deterministic anchors. It did not generate the 10K simulation dataset,
train models, or alter fixed-policy evidence.

## Corrected lineage

- Branch: `experiment/final-integrated-dataset-10k-adaptive-v2`
- Base SHA: `f599fbf457f44ec8318bbc2dacd2dd2d94bb6389`
- Contract identity: `final_integrated_dataset_10k_adaptive_v2`
- Prepared contract root: `artifacts/final_integrated_dataset_10k_adaptive_v2_contract`
- Contract specification hash: `23c5fffc10849c3bc3ea027251ac3e5ad4c96f0eea85edf1e8deab079cb0871e`
- Contract bundle hash: `da3c73711b1d60635afcceee8bda0a60d1379e492e1ad0d588d5a0c25e10abe3`

The prepared root contains only contract/manifests and adaptive pilot input
identity; it contains no satellite, G1-G5, target, RF, TGNN, or model outputs.

## Runtime profile

`grid_adaptive`, `adjacent_search_k=1`, and total incident inter-plane
capacity `1`; 10 minutes at 60-second steps (11 inclusive timesteps), J2000,
SGP4, phasing factor 1, 10,000 km inclusive ISL range, and
`persistent_temporal_union_edges_v1`.

## Qualification results

The five existing architecture anchors P01-P05 were evaluated with the
adaptive profile over all 11 timesteps. Zero-failure diagnostics reported:

| Anchor | GCC minimum | Max components | Edge range | Accepted intra/inter links |
|---|---:|---:|---:|---:|
| P01 | 0.3333 | 3 | 71-72 | 528 / 263 |
| P02 | 1.0000 | 1 | 68 | 528 / 220 |
| P03 | 0.0667 | 22 | 8 | 0 / 88 |
| P04 | 0.1000 | 14 | 6 | 0 / 66 |
| P05 | 0.1000 | 16 | 4-5 | 0 / 46 |

P03-P05 are sparse anchors and remain fragmented under both fixed and
adaptive construction; adaptive construction increases accepted inter-plane
links and reduces component counts. No fixed-policy-only fragmentation was
observed.

One adaptive P03 run was generated in an external temporary qualification root
and replayed through all G1-G5 stages. Replay succeeded with 11 G2, G3, G4,
and G5 records. A second in-memory satellite rollout produced identical
summary, failure realization, and 11-step graph reconstruction identities.

Prepared adaptive run mapping validated all 10,000 manifest rows: every row
preserved adaptive policy, K=1, capacity=1, 11 timesteps, and a distinct
adaptive-bound configuration identity.

## Failure semantics

The temporal-union implementation remains active: node failures are sampled
once, edge failures are sampled from the union of accepted undirected pairs
across all timesteps, and failed pairs are removed whenever active. Determinism
was confirmed for the adaptive P03 seed and replay.

## Execution plan and cost basis

Proposed new roots (not created or authorized in Phase 0):

- `C:\Users\johns\external\satnet-10k-final-ml-datasets-v2-adaptive`
- `C:\Users\johns\external\satnet-10k-final-ml-datasets-v2-adaptive-replay`
- `C:\Users\johns\external\satnet-10k-final-models-v2-adaptive`
- `C:\Users\johns\external\satnet-10k-heldout-v2-adaptive`
- `C:\Users\johns\external\satnet-external-validation-v2-adaptive`
- `C:\Users\johns\external\satnet-dss-v2-adaptive`

The production plan is 2,000 designs x 5 realizations = 10,000 generation
runs and 10,000 authoritative replays. Using the accepted 25-run pilot
summary as a linear planning basis gives approximately 9,439 seconds (2.62 h)
for generation and 12,193 seconds (3.39 h) for replay, before parallelism.
The same evidence projects approximately 20,162,537,600 bytes (18.78 GiB)
of G1-G5 generation artifacts for 10,000 runs; ML exports and model storage
are excluded.

Satellite/G3 outputs and targets must be regenerated; RF/TGNN exports must be
regenerated and models retrained; held-out and external results must be
reevaluated; DSS must be rebound and requalified. Physics, G1-G5 algorithms,
DOE coordinates, split strategy, leakage protections, targets' definitions,
and ground composition design are reusable unchanged. Existing fixed-policy
contract, training, held-out, external, and DSS artifacts remain historical
only.
