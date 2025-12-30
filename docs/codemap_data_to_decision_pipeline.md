# Codemap: Data-to-Decision Pipeline

> **Purpose**: Visualize the complete data flow from raw simulation through metrics calculation to actionable risk tiers.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           DATA-TO-DECISION PIPELINE                                  │
│                                                                                      │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│   │  INGESTION   │───▶│  SIMULATION  │───▶│   METRICS    │───▶│ DECISION SUPPORT │  │
│   │  (Raw Data)  │    │ (The Truth)  │    │ (Analysis)   │    │  (Output Layer)  │  │
│   └──────────────┘    └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                                      │
│   TLE/Config Files    Physics Engine      Pure Math          Risk Tiers + Actions   │
│   Design Parameters   Graph Snapshots     GCC Fractions      Healthy/Watchlist/Crit │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: INGESTION (Raw Data Handling)

**Purpose**: Load constellation configurations and orbital parameters.

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION LAYER                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   External Sources                    Internal Config        │
│   ─────────────────                   ───────────────        │
│   ├── Hypatia TLE files               ├── num_planes         │
│   ├── ISL distance matrices           ├── sats_per_plane     │
│   └── Ground station coords           ├── inclination_deg    │
│                                       └── altitude_km        │
│                                                              │
│   Key Files:                                                 │
│   ├── ../../hypatia/                  (External dependency)  │
│   └── data/tier1_design_runs.csv      (Generated datasets)   │
│                                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
```

---

## Stage 2: SIMULATION (The Truth)

**Purpose**: Generate physics-accurate temporal graph snapshots via SGP4 propagation.

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION LAYER                          │
│                     "The Truth"                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   src/satnet/network/hypatia_adapter.py                      │
│   ─────────────────────────────────────                      │
│   │                                                          │
│   ├── HypatiaAdapter                                         │
│   │   ├── load_constellation()     # Parse TLE/config        │
│   │   ├── get_graph_at_step(t)     # Temporal snapshot       │
│   │   └── get_isl_distance_matrix()# Link budget inputs      │
│   │                                                          │
│   └── Output: nx.Graph per timestep                          │
│       ├── Nodes: Satellite IDs                               │
│       └── Edges: Active ISL links (1550nm optical)           │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   src/satnet/simulation/                                     │
│   ──────────────────────                                     │
│   │                                                          │
│   ├── engine.py          # Temporal loop orchestration       │
│   ├── failures.py        # Stochastic failure injection      │
│   ├── tier1_rollout.py   # Tier1 dataset generation          │
│   └── monte_carlo.py     # (Legacy - being refactored)       │
│                                                              │
│   Output: Time-series of graph states G(t=0), G(t=1), ...    │
│                                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │  For each timestep t:
                           │  ├── G(t) = nx.Graph
                           │  └── failure_mask applied
                           ▼
```

---

## Stage 3: METRICS CALCULATION (Core Analysis)

**Purpose**: Compute pure, stateless connectivity metrics from graph state.

```
┌─────────────────────────────────────────────────────────────┐
│                     METRICS LAYER                            │
│                    (Pure Functions)                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   src/satnet/metrics/labels.py                               │
│   ────────────────────────────                               │
│   │                                                          │
│   ├── compute_num_components(G)    → int                     │
│   │   "How many disconnected pieces?"                        │
│   │                                                          │
│   ├── compute_gcc_size(G)          → int                     │
│   │   "Size of largest connected component"                  │
│   │                                                          │
│   ├── compute_gcc_frac(G)          → float [0.0, 1.0]        │
│   │   "Fraction of satellites in GCC"                        │
│   │   ★ PRIMARY RELIABILITY SCORE ★                          │
│   │                                                          │
│   ├── compute_partitioned(gcc_frac, threshold) → 0|1         │
│   │   "Is network partitioned?"                              │
│   │                                                          │
│   └── aggregate_partition_streaks(list[int]) → int           │
│       "Max consecutive partition duration"                   │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Data Flow Example:                                         │
│   ──────────────────                                         │
│                                                              │
│   G(t=5) ──▶ compute_gcc_frac(G) ──▶ 0.73                    │
│                                       │                      │
│                                       │  (reliability_score) │
│                                       ▼                      │
│                              DataFrame row:                  │
│                              {                               │
│                                "timestep": 5,                │
│                                "gcc_frac": 0.73,             │
│                                "partitioned": 0              │
│                              }                               │
│                                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │  Aggregated reliability_score
                           │  (e.g., min GCC frac over time)
                           ▼
```

---

## Stage 4: DECISION SUPPORT (Output Layer) ★ NEW ★

**Purpose**: Convert continuous reliability scores into actionable risk tiers.

```
┌─────────────────────────────────────────────────────────────┐
│                  DECISION SUPPORT LAYER                      │
│                    ★ OUTPUT LAYER ★                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   src/satnet/metrics/risk_binning.py    ◀── NEW FILE         │
│   ──────────────────────────────────                         │
│   │                                                          │
│   ├── bin_satellite_risk(df, score_column, ...)              │
│   │   │                                                      │
│   │   │  Input:  DataFrame with reliability_score [0.0, 1.0] │
│   │   │  Output: DataFrame + risk_tier + risk_label + action │
│   │   │                                                      │
│   │   └── Applies 3-Tier Classification:                     │
│   │                                                          │
│   │       ┌─────────────────────────────────────────────┐    │
│   │       │  TIER 1: HEALTHY      (score > 0.8)         │    │
│   │       │  ───────────────────────────────────────    │    │
│   │       │  Label:  "Healthy"                          │    │
│   │       │  Action: "No Action"                        │    │
│   │       │  Color:  🟢 Green                           │    │
│   │       └─────────────────────────────────────────────┘    │
│   │                                                          │
│   │       ┌─────────────────────────────────────────────┐    │
│   │       │  TIER 2: WATCHLIST    (0.5 ≤ score ≤ 0.8)   │    │
│   │       │  ───────────────────────────────────────    │    │
│   │       │  Label:  "Watchlist"                        │    │
│   │       │  Action: "Schedule Diagnostics"             │    │
│   │       │  Color:  🟡 Yellow                          │    │
│   │       └─────────────────────────────────────────────┘    │
│   │                                                          │
│   │       ┌─────────────────────────────────────────────┐    │
│   │       │  TIER 3: CRITICAL     (score < 0.5)         │    │
│   │       │  ───────────────────────────────────────    │    │
│   │       │  Label:  "Critical"                         │    │
│   │       │  Action: "Immediate Maneuver"               │    │
│   │       │  Color:  🔴 Red                             │    │
│   │       └─────────────────────────────────────────────┘    │
│   │                                                          │
│   ├── compute_tier(score)          → 1, 2, or 3              │
│   ├── get_tier_label(tier)         → "Healthy" | ...         │
│   ├── get_tier_action(tier)        → "No Action" | ...       │
│   └── summarize_tier_distribution(df) → dict                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow: Simulation → Metric → Risk Bin

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│  ┌────────────────────┐                                                              │
│  │   HYPATIA ADAPTER  │                                                              │
│  │   (Physics Truth)  │                                                              │
│  └─────────┬──────────┘                                                              │
│            │                                                                         │
│            │  get_graph_at_step(t)                                                   │
│            ▼                                                                         │
│  ┌────────────────────┐                                                              │
│  │   nx.Graph G(t)    │   Nodes: 1584 satellites                                     │
│  │   (Temporal State) │   Edges: Active ISL links                                    │
│  └─────────┬──────────┘                                                              │
│            │                                                                         │
│            │  compute_gcc_frac(G)                                                    │
│            ▼                                                                         │
│  ┌────────────────────┐                                                              │
│  │  reliability_score │   Example: 0.73                                              │
│  │     float [0,1]    │   (73% of satellites in GCC)                                 │
│  └─────────┬──────────┘                                                              │
│            │                                                                         │
│            │  bin_satellite_risk(df)                                                 │
│            ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │                           DECISION OUTPUT                                       │  │
│  ├────────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                                 │  │
│  │   {                                                                             │  │
│  │     "reliability_score": 0.73,                                                  │  │
│  │     "risk_tier": 2,                    ◀── Numeric tier                         │  │
│  │     "risk_label": "Watchlist",         ◀── Human-readable                       │  │
│  │     "recommended_action": "Schedule Diagnostics"  ◀── Actionable               │  │
│  │   }                                                                             │  │
│  │                                                                                 │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
                    ┌─────────────────────────────────┐
                    │         EXTERNAL                │
                    │    ../../hypatia/ (SGP4/TLE)    │
                    └───────────────┬─────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                              src/satnet/                                           │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│   network/                                                                         │
│   ├── hypatia_adapter.py ─────────────────────────────────┐                        │
│   │   (HypatiaAdapter)                                    │                        │
│   │                                                       │                        │
│   │                                                       ▼                        │
│   simulation/                                    ┌─────────────────┐               │
│   ├── engine.py ─────────────────────────────────│   nx.Graph(t)   │               │
│   ├── failures.py                                └────────┬────────┘               │
│   └── tier1_rollout.py                                    │                        │
│                                                           │                        │
│                                                           ▼                        │
│   metrics/                                       ┌─────────────────┐               │
│   ├── labels.py ─────────────────────────────────│  gcc_frac: 0.73 │               │
│   │   (compute_gcc_frac)                         └────────┬────────┘               │
│   │                                                       │                        │
│   │                                                       ▼                        │
│   └── risk_binning.py ★ NEW ★ ───────────────────┌─────────────────┐               │
│       (bin_satellite_risk)                       │  Tier 2:        │               │
│                                                  │  "Watchlist"    │               │
│                                                  │  "Schedule      │               │
│                                                  │   Diagnostics"  │               │
│                                                  └─────────────────┘               │
│                                                                                    │
│   models/                                                                          │
│   ├── risk_model.py      (ML training, consumes metrics)                           │
│   ├── gnn_model.py       (Graph neural network)                                    │
│   └── gnn_dataset.py     (Dataset preparation)                                     │
│                                                                                    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## Usage Example: End-to-End Pipeline

```python
# 1. INGESTION: Load constellation config
from satnet.network.hypatia_adapter import HypatiaAdapter
adapter = HypatiaAdapter(constellation="starlink_550")

# 2. SIMULATION: Get temporal graph snapshot
G = adapter.get_graph_at_step(t=100)  # Graph at timestep 100

# 3. METRICS: Compute reliability score
from satnet.metrics import compute_gcc_frac
reliability_score = compute_gcc_frac(G)  # e.g., 0.73

# 4. DECISION SUPPORT: Bin into risk tier
import pandas as pd
from satnet.metrics import bin_satellite_risk

df = pd.DataFrame({"reliability_score": [reliability_score]})
result = bin_satellite_risk(df)

print(result)
#    reliability_score  risk_tier risk_label   recommended_action
# 0               0.73          2  Watchlist  Schedule Diagnostics
```

---

## File Reference

| Stage | Module | Key Functions |
|-------|--------|---------------|
| **Ingestion** | `network/hypatia_adapter.py` | `HypatiaAdapter.load_constellation()` |
| **Simulation** | `simulation/engine.py` | Temporal loop orchestration |
| **Simulation** | `simulation/failures.py` | `apply_random_failures()` |
| **Metrics** | `metrics/labels.py` | `compute_gcc_frac()`, `compute_partitioned()` |
| **Decision** | `metrics/risk_binning.py` ★ | `bin_satellite_risk()`, `compute_tier()` |

---

*Codemap generated for thesis "Data Binning Scheme" requirement.*
*Last updated: December 2024*
