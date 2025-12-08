# satnet-arch-dss

**Satellite Network Architecture Decision Support System (DSS)**

This repo implements a simulation + machine learning pipeline that helps a
satellite network architect estimate **partition risk** for LEO constellations
under stochastic failures.

The system:

1. Generates parameterized satellite network topologies.
2. Samples node/link failures according to environment assumptions.
3. Computes network connectivity / partition outcomes.
4. Builds two ML models:
   - **Outcome-aware model (Model A)** – oracle upper bound using post-failure metrics.
   - **Design-time model (Model B)** – thesis model that predicts partition risk
     *only from design-time parameters* (architecture + failure probabilities).

---

## Project Structure

```text
satnet-arch-dss/
├── src/
│   └── satnet/
│       ├── network/
│       │   └── topology.py       # Parametric constellation generator
│       ├── simulation/
│       │   ├── failures.py       # Failure sampling + impact computation
│       │   └── monte_carlo.py    # Monte Carlo + dataset generators
│       └── models/
│           └── risk_model.py     # ML models + training helpers
├── scripts/
│   ├── simulate.py               # Simple one-off simulation run
│   ├── failure_sweep.py          # Summary stats over many runs
│   ├── export_failure_dataset.py # Outcome-aware dataset (Model A)
│   ├── export_design_dataset.py  # Design-time dataset (Model B)
│   ├── train_risk_model.py       # Train outcome-aware model
│   └── train_design_risk_model.py# Train design-time model
├── Data/
│   ├── failure_dataset.csv       # (generated) outcome-aware samples
│   └── design_failure_dataset.csv# (generated) design-time samples
├── models/
│   ├── risk_model.joblib         # (generated) Model A
│   └── design_risk_model.joblib  # (generated) Model B
└── README.md
Installation
Requires Python 3.11+ (you’re using 3.14 locally).

bash
Copy code
git clone https://github.com/Alexsabatino/satnet-arch-dss.git
cd satnet-arch-dss

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install networkx pandas scikit-learn joblib
(If you add a requirements.txt later, this becomes pip install -r requirements.txt.)

Quickstart: Run a Simulation
From the project root:

bash
Copy code
python scripts/simulate.py
Expected console output (example):

text
Copy code
=== Simulation Started ===
Nodes: 24
Edges: 40
Bottlenecks detected: 0
=== Simulation Complete ===
This verifies that:

topology generation works

failure engine runs

basic connectivity metrics are computed

Generate Datasets
1. Outcome-aware dataset (Model A – upper bound)
bash
Copy code
python scripts/export_failure_dataset.py
This writes:

Data/failure_dataset.csv

Each row includes:

pre-failure graph metrics

realized failures (failed nodes/edges)

post-failure largest component size

binary label partitioned

Used only for the oracle / sanity-check model.

2. Design-time dataset (Model B – thesis model)
bash
Copy code
python scripts/export_design_dataset.py
This:

Sweeps over a grid of architectures and failure assumptions:

num_satellites ∈ {16, 24, 32}

num_ground_stations ∈ {2, 4, 6}

isl_degree ∈ {2, 4, 6}

node_failure_prob ∈ {0.01, 0.03, 0.05}

edge_failure_prob ∈ {0.01, 0.03, 0.05}

Runs multiple Monte Carlo realizations per scenario.

Writes Data/design_failure_dataset.csv.

Each row has:

Architecture features: num_nodes_0, num_edges_0, avg_degree_0,
num_satellites, num_ground_stations, isl_degree

Failure assumptions: node_failure_prob, edge_failure_prob

Label: partitioned (1 if the network partitions under that realization, else 0)

This dataset is pure design-time: no post-failure features are used as inputs.

Train the Models
Model A – Outcome-aware (sanity check / upper bound)
bash
Copy code
python scripts/train_risk_model.py
Example output:

text
Copy code
Loading dataset from Data/failure_dataset.csv ...
Saved model to models/risk_model.joblib
=== Evaluation Metrics (test split) ===
Accuracy: 1.000
ROC AUC: 1.000
...
This uses post-failure metrics (e.g., largest component ratio) and is expected to
be near-oracle performance. It’s mainly used to validate that the simulation and
labeling logic are consistent.

Model B – Design-time (thesis model)
bash
Copy code
python scripts/train_design_risk_model.py
Example output (what we observed):

text
Copy code
Loading design dataset from Data/design_failure_dataset.csv ...
Saved design-time model to models/design_risk_model.joblib
=== Design-Time Model Metrics (test split) ===
Accuracy: 0.685
ROC AUC: 0.733
Confusion matrix [ [TN, FP], [FN, TP] ]:
[[719, 378], [388, 945]]

Top design feature importances:
  node_failure_prob: 0.681
  edge_failure_prob: 0.100
  num_nodes_0: 0.069
  num_satellites: 0.050
  num_edges_0: 0.039
  avg_degree_0: 0.033
  isl_degree: 0.016
  num_ground_stations: 0.013
Interpretation:

AUC ~0.73 → nontrivial predictive signal from design-time features only.

Dominant drivers: node_failure_prob, edge_failure_prob.

Architectural knobs (sat count, connectivity, ISLs, GSs) have smaller but
non-zero influence, matching engineering intuition.

What the Design-Time Model Does
The design-time model estimates:

P(partition | architecture, failure environment)

Given:

num_satellites

num_ground_stations

isl_degree

Derived graph metrics (num_nodes_0, num_edges_0, avg_degree_0)

Assumed node_failure_prob, edge_failure_prob

it predicts the probability that the constellation will partition under those
conditions, without seeing any post-failure state.

This is intended as a decision support tool for a satellite network
architect / engineering manager:

Compare candidate architectures under the same environment.

Understand how risk shifts as failure rates or ISL degree change.

Use model outputs to prioritize detailed simulations on the riskiest designs.

How This Fits a Praxis / Thesis
This repo supports a praxis / thesis framed roughly as:

“An AI-assisted decision support tool for satellite network architecture risk.
Given parametric constellation designs and probabilistic failure assumptions,
we use Monte Carlo simulation to label partition outcomes and train a
design-time classifier that predicts network partition risk from
architecture-level features alone.”

Key contributions:

Simulation-backed labeling
Partition labels are derived from graph connectivity metrics after sampled
failures, not assumed.

Outcome-aware vs design-time models

Outcome-aware model shows an upper bound and validates the pipeline.

Design-time model demonstrates that useful risk structure is learnable
from high-level design parameters.

Explainable drivers of risk
Feature importances quantify how much environment vs architecture drive
partition risk, grounding the ML model in engineering terms.

Reproducibility
To reproduce main results:

bash
Copy code
# 1. Create virtual env and install deps
python -m venv .venv
.venv\Scripts\activate
pip install networkx pandas scikit-learn joblib

# 2. Generate datasets
python scripts/export_failure_dataset.py
python scripts/export_design_dataset.py

# 3. Train models
python scripts/train_risk_model.py
python scripts/train_design_risk_model.py
Metrics and feature importances are written to:

models/risk_model_metrics.json

models/design_risk_model_metrics.json

Next Extensions (Ideas)
Add richer graph features (clustering coefficient, algebraic connectivity).

Calibrate probabilities (Platt / isotonic) for risk thresholds.

Integrate simple web UI where an engineer can slide num_satellites,
isl_degree, node_failure_prob, etc., and see risk scores live.

yaml
Copy code

---

## 2. Commit + push the README

In the repo root:

```powershell
git add README.md
git commit -m "Add README with simulation + design-time model details"
git push
