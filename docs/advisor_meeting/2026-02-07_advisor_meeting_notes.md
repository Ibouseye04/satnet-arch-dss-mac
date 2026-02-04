# Advisor Meeting Notes — Feb 7, 2026

**Purpose:** Focused progress on highest-ROI external datasets + review Alex’s process flowchart.

---

## Phase 1 Constraints → Phase 2 Expansion (Callout)

**Phase 1 (validated baseline — space segment only):**
- **Sat-to-sat only** (no ground stations yet)
- **Temporal graphs** evaluated at every timestep (t=0..T), not snapshots
- **Physics-based ISLs** via SGP4 + link budgets (no toy topology)
- **Deterministic** runs (seed + config hash)
- **Non-leaky labels** computed from graph state only

**Phase 2 (next step after Phase 1 validation):**
- Add **ground stations / gateways** and SAT–GS visibility windows
- Incorporate **Earth rotation + elevation masks + weather margins**
- Extend labels to **service availability** (end-to-end connectivity)
- Introduce **traffic/routing metrics** (latency, throughput, congestion)
- Expand ML models with **ground + service-level features**

## 1) External Dataset Progress (ROI-focused)

**Executive summary:** Only **LENS** is recommended as a high‑value **Track B validation** dataset. All other datasets are synthetic or algorithm outputs that would violate the Tier‑1 first‑principles rule if used as truth or training sources.

### Approved (Track B validation only)
1. **LENS: A LEO Satellite Network Measurement Dataset**
   - **Why:** Empirical Starlink measurements (latency/RTT, PoP shifts) from real terminals.
   - **Use:** **Phase 2 validation target** — compare Hypatia‑predicted SAT–GS latency trends against LENS; **does not feed labels**.
   - **Action:** Add to `external_datasets_manifest.md` as priority validation source; document fields + snapshot cadence.

### Optional (only as non‑training sanity check)
- **Physical‑Layer ISL features (IEEE DataPort)** — can be used to spot‑check PHY formulas (delay/Doppler/range). **Do not** use as labels or training data.

### Rejected / Deferred (do not integrate into Tier‑1 truth pipeline)
- **Synthetic Dynamic STIN dataset (IEEE DataPort)** — synthetic simulation outputs; unknown topology assumptions.
- **Multi‑Scale Planning / Flow Allocation (IEEE DataPort)** — algorithm‑specific outputs; only useful for Phase 2 benchmarking (if at all).
- **Network congestion prediction (IEEE DataPort)** — generic dataset; superseded by higher‑fidelity LEO congestion logs in plan.
- **Satellite Constellation Dataset (Kaggle)** — static TLE snapshot; we use fresh TLEs/archived epochs via Hypatia.

---

## 2) Minimal Deliverable Checklist (By Sat)
- [ ] **Dataset manifest draft** (LENS priority + access/license notes)
- [ ] **1–2 page EDA note** for LENS (schema + 2–3 plots)
- [ ] **One slide**: Phase 1 vs Phase 2 dataset mapping (LENS as Phase 2 validation)
- [ ] **Optional**: ISL PHY sanity‑check note (if advisor wants a physics cross‑check)

---

## 3) Alex’s Flowchart Review (Corrections/Notes)

Overall structure is good. Suggested fixes to align with Tier‑1 constraints:

1. **“NetworkX Tool” box**
   - Rename to **“Graph Metrics (NetworkX)”** or **“Metric Extraction (NetworkX)”**.
   - Clarify that **NetworkX is used for metrics only**, not for generating topology.

2. **Temporal loop emphasis**
   - Add explicit note: **metrics computed for every timestep (t=0..T)**, not just at t=0.
   - Suggest adding a small label: “Temporal evaluation loop (t=0..T)”.

3. **Non‑leaky labels**
   - In the **Rule-Based Risk Logic (binning.py)** box, add note:
     - “Uses only graph metrics; does NOT use failure parameters.”

4. **Determinism / Reproducibility**
   - In **Scenario/Experiment Configuration**, add: **seed + config_hash** for reproducibility.

5. **Phase 1 scope boundary**
   - Add a small callout: **“Phase 1 = sat‑to‑sat only (no ground stations yet).”**

6. **ML Model clarity**
   - Under **ML Risk Model**, note:
     - **RF baseline** uses design parameters only.
     - **Temporal GNN** uses time‑series graph snapshots.

---

## 4) Advisor-Safe Talking Points (Short)
- “We prioritize **physics‑based ISL data** for Phase 1 validation; ground‑segment datasets are staged for Phase 2.”
- “External datasets are **supporting/validation tracks**, not the source of Phase‑1 truth labels.”
- “The flowchart reflects a **temporal, deterministic, non‑leaky** pipeline.”
