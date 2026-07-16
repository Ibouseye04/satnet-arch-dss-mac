# Tier 1 Preliminary Evidence Inventory

**Inventory date:** 2026-07-16  
**Status:** Read-only inventory for Atomic Step 0.1  
**Repository:** `satnet-arch-dss-mac`  
**Inventory baseline branch:** `main`  
**Inventory baseline HEAD:** `c353d2543a8071fc7bd951904ed506ef09cf0b09`  

## 1. Evidence status and preservation rule

The evidence listed here is preliminary, historical, and immutable. It may support development history and topology-sensitivity findings, but it is not canonical dissertation evidence. Do not delete, overwrite, resume in place, or regenerate into any listed directory.

All post-remediation experiments must use new versioned output directories that are distinct from every path in this inventory. Historical metrics, checkpoints, predictions, manifests, logs, datasets, and comparison tables remain unchanged.

## 2. Dataset identity

Hashes are SHA-256 over exact file bytes.

| Dataset file | Size (bytes) | SHA-256 |
|---|---:|---|
| `data/tier1_design_runs.csv` | 275,556 | `9e7cf202082c363b755404e48f315724e009a52d6d9e47df121339d3b32ab0cb` |
| `data/tier1_design_steps.csv` | 960,137 | `f082bcd5e747d099e0caf5588102f9c76816875bfeae43681b4f7499d30d7453` |

All six inventoried split manifests record the run-table content hash `9e7cf202082c363b755404e48f315724e009a52d6d9e47df121339d3b32ab0cb` and a 500-row source dataset.

## 3. Immutable evidence roots

| Evidence root | Historical interpretation | Run-manifest SHA-256 |
|---|---|---|
| `artifacts/ablation` | Historical full RF/TGNN ablation. Its TGNN evidence is pre-fix K=1: `cheb_k` is absent or null in the manifest and TGNN metrics. | `4bb012c86d57c9f8d3e6c78caebcaee3cca10745e743a379bf33ee7529a8063a` |
| `artifacts/validation/tgnn_k2_ablation_smoke` | Preliminary K=2 smoke topology-sensitivity evidence; `cheb_k=2`, seed 42, smoke mode. | `8819d52574a139547d2a472725efef97a0433a4cdb1101b80525807ac6a903d9` |
| `artifacts/validation/space_segment_k2_500` | Preliminary K=2 500-run topology-sensitivity evidence; `cheb_k=2`, seed 42, non-smoke mode. | `37c53eb88f4b4efcd1042fcf513a39674d9650399512a99fbb9887d32dba047c` |

## 4. Split-manifest file identities

Hashes are SHA-256 over exact manifest bytes. Identical hashes show byte-identical manifests.

| Evidence root | Target | Split-manifest SHA-256 |
|---|---|---|
| `artifacts/ablation` | `partition_any` | `1f731c2bd5ceb297aabe2fa23e7c8d51ef801eb560e895292eb51dcc03e6d813` |
| `artifacts/ablation` | `gcc_frac_min_original` | `e5d054f48a02bd206c37166a9bc250b0d312d451af131f6f5539c18c59db122d` |
| `artifacts/validation/tgnn_k2_ablation_smoke` | `partition_any` | `e586590ac9bf30a3af448b1a92109472d94c22b1a05da918f6977633b053b683` |
| `artifacts/validation/tgnn_k2_ablation_smoke` | `gcc_frac_min_original` | `7bbf56d75984a2ea2b675ef0ad4f3e4e8b19e4af71ba329d947e90ce0feba0f7` |
| `artifacts/validation/space_segment_k2_500` | `partition_any` | `1f731c2bd5ceb297aabe2fa23e7c8d51ef801eb560e895292eb51dcc03e6d813` |
| `artifacts/validation/space_segment_k2_500` | `gcc_frac_min_original` | `e5d054f48a02bd206c37166a9bc250b0d312d451af131f6f5539c18c59db122d` |

## 5. Recorded source provenance and limitation

The inventoried TGNN experiment logs record:

- `git_sha = 859141b57eb613619855d69aee4d0c6d896b8378`

This SHA does not fully identify the executed K=2 source. The full validity review found that the K=2 run used behavior committed later in `c353d2543a8071fc7bd951904ed506ef09cf0b09`. The experiment logs do not record dirty-worktree state or a tracked-diff hash. The K=2 evidence is therefore not fully source-reproducible and remains preliminary.

The inventory itself was created at HEAD `c353d2543a8071fc7bd951904ed506ef09cf0b09` while the working tree contained pre-existing untracked review, plan, validation-report, and diagnostic-test work. No historical artifact was modified to add this inventory.

## 6. Read-only artifact snapshot

Each snapshot digest is SHA-256 over sorted records containing relative path, byte size, nanosecond modification time, and file SHA-256. These values establish the pre-edit state used to verify that Atomic Step 0.1 did not alter artifact content or modification timestamps.

| Evidence root | File count | Snapshot digest |
|---|---:|---|
| `artifacts/ablation` | 110 | `d2d04d77b1f107e44af3929559836fd0a637b6a63bad0e2d67525b1d9fe84c90` |
| `artifacts/validation/tgnn_k2_ablation_smoke` | 55 | `f197b0e158ed3da7ceb2859ff0fbaf0b8029f52dbdc161f0ed2f0b28a93d9254` |
| `artifacts/validation/space_segment_k2_500` | 55 | `f5c72a272b3b50974301bb737050c96cff6605eb2c07b79e57dbe9b64735958e` |

## 7. Permitted use

These artifacts may be cited only with their limitations:

- K=1 artifacts document the historical edge-insensitive model behavior.
- K=2 artifacts provide preliminary evidence that the corrected model is topology-sensitive.
- Neither set establishes canonical post-remediation performance.
- RF and TGNN headline metrics do not establish direct model superiority because the models receive different information.
- Any future example, smoke run, pilot, or canonical experiment must target a new versioned directory and carry a complete scientific identity.
