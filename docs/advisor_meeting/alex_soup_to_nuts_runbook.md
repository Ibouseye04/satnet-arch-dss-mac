# Alex Runbook: Data Generation to RMSE

This is the start-to-finish operating guide for tomorrow's advisor discussion.
It is written for Alex to run the pipeline, explain what each command does, and
interpret the output without needing to read the whole codebase during the
meeting.

The practical goal is not to claim final model quality. The goal is to show
that the research pipeline is technically coherent, reproducible, and producing
valid evaluation artifacts.

---

## 0. What the Pipeline Does

The repository turns a satellite constellation design into ML-ready resilience
labels:

```text
Walker Delta design
  -> HypatiaAdapter
  -> SGP4 orbital propagation
  -> LOS + optical inter-satellite-link viability
  -> dynamic NetworkX graph sequence over time
  -> node/link failure injection
  -> connectivity metrics per timestep
  -> run-level labels for Random Forest
  -> graph-sequence samples for Temporal GNN
  -> evaluation metrics and prediction artifacts
```

### Where HypatiaAdapter Comes In

`HypatiaAdapter` is the physics graph provider. It is not the ML model and it is
not the failure simulator. It sits between the sampled constellation design and
the resilience metrics:

```text
scripts/export_design_dataset.py
  -> generate_tier1_temporal_dataset(...)
  -> run_tier1_rollout(...)
  -> HypatiaAdapter(...)
  -> adapter.generate_tles()
  -> adapter.calculate_isls(...)
  -> adapter.iter_graphs()
  -> failure injection + GCC metrics
  -> CSV labels used by RF/GNN
```

In code:

- `scripts/export_design_dataset.py` calls `generate_tier1_temporal_dataset`.
- `src/satnet/simulation/monte_carlo.py` calls `run_tier1_rollout` once per
  sampled design/failure run.
- `src/satnet/simulation/tier1_rollout.py` constructs `HypatiaAdapter`, calls
  `generate_tles`, calls `calculate_isls`, then iterates `adapter.iter_graphs()`.
- `src/satnet/network/hypatia_adapter.py` generates the Walker Delta TLEs,
  propagates satellites, computes viable ISLs, and returns a NetworkX graph for
  each timestep.
- `src/satnet/models/gnn_dataset.py` also uses `HypatiaAdapter` to reconstruct
  the same graph sequence from the run CSV for Temporal GNN training.

Alex's short answer:

> "HypatiaAdapter is the Tier 1 physics layer. It converts a constellation
> design into a time sequence of satellite network graphs using TLE generation,
> SGP4 propagation, LOS checks, and link-budget filtering. The failure simulator
> then modifies those graphs, and the metrics layer turns them into labels."

### How Failures Are Injected

There are two different concepts that Alex should keep separate:

1. Physical link viability: `HypatiaAdapter` decides whether an ISL can exist at
   each timestep based on geometry, LOS, and link budget.
2. Stochastic failures: the simulation layer removes some satellites and links
   from those already-physical graphs.

Failure injection happens inside `src/satnet/simulation/tier1_rollout.py`, after
`HypatiaAdapter` has already generated the baseline graph sequence.

For each Monte Carlo run:

```text
1. Sample constellation design parameters.
2. Sample node failure probability pn from the configured range.
3. Sample edge failure probability pe from the configured range.
4. Build the physical graph sequence with HypatiaAdapter.
5. Get G0, the graph at timestep 0.
6. For every satellite node in G0:
     fail the node with probability pn.
7. For every edge in G0:
     fail the edge with probability pe.
8. For every timestep Gt:
     remove the failed nodes.
     remove failed edges if they exist at that timestep.
     compute GCC/connectivity metrics on the remaining graph.
```

The implementation uses a seeded random number generator, so a given run is
reproducible. During dataset generation, the run seed is `base_seed + run_id`.

Important v1 semantics:

- Node failures are persistent. If a satellite fails, it is removed at every
  timestep.
- Edge failures are persistent. If a link is marked failed, it is removed
  whenever that edge would otherwise exist.
- Edge failures are sampled only from edges present in `G0`.
- Edges that are not visible at `t=0` but appear later are not eligible for
  failure in this v1 model.
- Failures are satellite-to-satellite only. There are no ground stations in
  this phase.

The exact failure realization is exported into the run CSV:

```text
failed_nodes_json
failed_edges_json
```

That matters for the GNN. The Temporal GNN dataset loader reads those JSON
columns and reapplies the same failed nodes and failed edges when reconstructing
the graph sequence. That keeps the graph input consistent with the labels that
were computed during dataset generation.

Alex's short answer:

> "First the physics layer tells us which links are physically viable. Then the
> simulation layer samples persistent node and edge failures using the run seed.
> Nodes are removed from every timestep. Edges sampled from the t=0 graph are
> removed whenever they appear. Labels are computed only from the resulting graph
> state, not directly from the failure probabilities."

What not to overclaim:

- This is not yet a correlated failure model.
- This is not yet a time-varying outage model.
- This does not yet include ground stations.
- Edge failure sampling from only `G0` is a documented v1 simplification.

The Random Forest sees one row per simulation run. The Temporal GNN is intended
to see one full graph sequence per simulation run:

```text
G0 -> G1 -> G2 -> ... -> GT
```

Alex should emphasize that the ML split is by complete run, not by individual
timestep. That avoids timestep leakage.

### Is This Generating Two Datasets?

Short answer: it generates one simulation campaign with two CSV tables.

```text
data/tier1_design_runs.csv
data/tier1_design_steps.csv
```

Those two tables serve different purposes:

| File | Grain | Purpose |
|---|---|---|
| `tier1_design_runs.csv` | One row per Monte Carlo run | Main Random Forest training table. Contains design inputs, failure probabilities, seed/config metadata, failure realization JSON, and aggregate labels such as `gcc_frac_min`, `gcc_frac_mean`, `partition_fraction`, `partition_any`, and `max_partition_streak`. |
| `tier1_design_steps.csv` | One row per run per timestep | Temporal diagnostic table. Contains per-timestep connectivity metrics such as number of nodes, number of edges, connected components, GCC size, GCC fraction, and partition flag. |

Conceptually, the project has two ML views:

1. Random Forest dataset: the run-level CSV, `tier1_design_runs.csv`.
2. Temporal GNN dataset: full graph sequences reconstructed per run.

The GNN graph-sequence dataset is not primarily exported as a normal CSV. The
GNN loader reads `tier1_design_runs.csv`, uses the saved design parameters,
epoch, seed, and failure realization, then calls `HypatiaAdapter` to reconstruct
the graph sequence:

```text
run row -> HypatiaAdapter -> G0, G1, ..., GT -> PyTorch Geometric Data sequence
```

If graph caching is enabled, those graph sequences can also be cached under the
artifact cache directory, but they still come from the same simulation run
records. The key point is that the RF and GNN are not trained from unrelated
random datasets. They are two representations of the same underlying simulation
runs.

Alex's short answer:

> "The export script writes two CSV tables: a run-level table and a timestep
> table. The Random Forest trains on the run-level table. The GNN reconstructs
> graph sequences from those same run records, so it is a second ML view of the
> same simulation campaign, not an unrelated dataset."

---

## 1. Environment Setup

Use Python 3.11 or 3.12. During the readiness audit, Python 3.14 caused package
conflicts with the current scientific stack, so do not use Python 3.14 for the
demo.

### macOS verified setup

From the repository root:

```bash
cd /Users/ibrahimseye/Developer/satnet-arch-dss-mac

/opt/homebrew/bin/python3.12 -m venv /tmp/satnet-arch-dss-py312
source /tmp/satnet-arch-dss-py312/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]" scikit-learn scipy joblib matplotlib tqdm
```

Sanity check:

```bash
python --version
python -c "import numpy, pandas, sklearn, scipy, networkx, sgp4; print('deps ok')"
```

Expected: Python 3.11 or 3.12, then `deps ok`.

### Windows equivalent

```powershell
cd path\to\satnet-arch-dss
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]" scikit-learn scipy joblib matplotlib tqdm
```

---

## 2. Fast End-to-End Demo

Use this path tonight. It proves the pipeline works without waiting for a large
experiment.

### One copy-paste smoke run

After activating the Python 3.11/3.12 environment, Alex can run this block:

```bash
python scripts/export_design_dataset.py --smoke --seed 42

python tools/analyze_dataset_targets.py data/tier1_design_runs.csv \
  --output artifacts/smoke/target_summary.csv \
  --output-corr artifacts/smoke/target_correlations.csv

python scripts/train_design_risk_model.py \
  --smoke \
  --target-name partition_any \
  --seed 42

python scripts/train_design_risk_model.py \
  --smoke \
  --target-name gcc_frac_min \
  --seed 42
```

This is the recommended demo path unless the GNN dependency stack has been
fixed locally. Avoid `make smoke` for the advisor demo if
`torch_geometric_temporal` is not installed, because that target also attempts
the GNN smoke run.

### Windows PowerShell copy-paste smoke run

PowerShell does not use Bash's `\` line continuation. If Alex is using
PowerShell, use one-line commands:

```powershell
python scripts/export_design_dataset.py --smoke --seed 42

python tools/analyze_dataset_targets.py data/tier1_design_runs.csv --output artifacts/smoke/target_summary.csv --output-corr artifacts/smoke/target_correlations.csv

python scripts/train_design_risk_model.py --smoke --target-name partition_any --seed 42

python scripts/train_design_risk_model.py --smoke --target-name gcc_frac_min --seed 42
```

PowerShell multiline form uses a backtick, not a backslash. There must be no
spaces after the backtick:

```powershell
python tools/analyze_dataset_targets.py data/tier1_design_runs.csv `
  --output artifacts/smoke/target_summary.csv `
  --output-corr artifacts/smoke/target_correlations.csv

python scripts/train_design_risk_model.py `
  --smoke `
  --target-name partition_any `
  --seed 42
```

If Alex is using Git Bash instead of PowerShell, the Bash block with `\` is
valid.

### How to Read the Smoke Run Output

When Alex runs the smoke block successfully, the output should be read in four
stages.

Stage 1: dataset generation

```text
Generated 24 runs, 48 step records
Written 24 run rows to ...\data\tier1_design_runs.csv
Written 48 step rows to ...\data\tier1_design_steps.csv
Partition probability: 0.792
Mean GCC fraction: 0.318
```

What this means:

- `24 runs`: 24 independent Monte Carlo simulation runs.
- `48 step records`: smoke mode uses 1 minute at 60-second steps, with inclusive
  `t=0` and `t=1`, so there are 2 timestep rows per run.
- `Partition probability: 0.792`: 19 of 24 runs partitioned at least once.
- `Mean GCC fraction: 0.318`: on average, the largest connected component
  contained about 31.8% of active satellites in this tiny smoke sample.

Advisor phrasing:

> "This confirms the physics and simulation layers generated both run-level and
> timestep-level data. The smoke dataset is intentionally small, but it has both
> partitioned and non-partitioned examples, so the classifier has a usable target."

Stage 2: target audit

```text
partition_any: zeros=5, ones=19
partition_fraction: zeros=5, ones=19
No exact-signature duplicates found.
```

What this means:

- The binary classification target is imbalanced but not degenerate.
- There are 5 non-partitioned runs and 19 partitioned runs.
- No duplicate design signatures were found in the smoke sample.
- The negative Spearman correlation between `gcc_frac_min` and
  `partition_any` is expected: lower GCC fraction means higher partition risk.

In the smoke run, `gcc_frac_mean` and `gcc_frac_min` may match because the run is
only two timesteps long. In longer runs, they should be interpreted separately:
mean GCC is average connectivity, while min GCC is worst-case connectivity.

Stage 3: Random Forest classification

The classification command writes:

```text
artifacts\smoke\rf\design_risk_model_tier1_metrics.json
artifacts\smoke\rf\design_risk_model_tier1_predictions.csv
artifacts\smoke\rf\design_risk_model_tier1_feature_importance.csv
artifacts\smoke\rf\design_risk_model_tier1_confusion_matrix.png
```

If the terminal line wraps or appears truncated after
`feature_importance...`, that is usually just terminal display width. Check the
actual files with `Get-ChildItem`.

Advisor phrasing:

> "The classifier predicts `partition_any`, meaning whether a run ever
> partitioned. I inspect accuracy, precision, recall, F1, and the confusion
> matrix, but because this is only a 24-run smoke dataset, I treat the metrics as
> a pipeline sanity check."

Stage 4: Random Forest regression

The regression command writes:

```text
artifacts\smoke\rf\rf_gcc_frac_min_metrics.json
artifacts\smoke\rf\rf_gcc_frac_min_predictions.csv
artifacts\smoke\rf\rf_gcc_frac_min_feature_importance.csv
artifacts\smoke\rf\rf_gcc_frac_min_prediction_vs_actual.png
```

Advisor phrasing:

> "The regressor predicts `gcc_frac_min`, the worst-case Giant Connected
> Component fraction. RMSE tells us the typical error scale in GCC-fraction
> units, with larger mistakes penalized more strongly."

### Why the Smoke Run Splits the Data

Yes, the Random Forest smoke run creates train, validation, and test splits.

For the 24-run smoke dataset, the expected split is:

```text
train: 12 runs
validation: 6 runs
test: 6 runs
```

The split is recorded in the metrics JSON:

```text
num_samples: 24
train_size: 12
val_size: 6
test_size: 6
split_strategy: run_id_grouped
```

Why split a tiny smoke run at all?

- It verifies that the evaluation pipeline works end to end.
- It proves the code can produce train, validation, and test metrics.
- It verifies that predictions are exported with `split=train`, `split=val`, and
  `split=test`.
- It checks that splitting happens by `run_id`, not by timestep.
- It prevents Alex from accidentally reporting training accuracy as if it were
  generalization performance.

What each split means:

| Split | Purpose |
|---|---|
| Train | Rows used to fit the Random Forest. |
| Validation | Rows reserved for model-selection/tuning checks. In this RF smoke run, it is mostly a sanity holdout because no hyperparameter search is being performed. |
| Test | Final held-out rows used for the top-level reported metrics. |

Important caveat:

The smoke split is intentionally small. Six validation rows and six test rows
are enough to prove the machinery works, but not enough for stable scientific
claims. Larger runs should use the same split logic with many more simulations.

Alex's short answer:

> "Yes, it splits the smoke dataset into train, validation, and test. The purpose
> is not to get final performance from 24 samples. It is to verify the leakage-safe
> evaluation pipeline, especially that complete runs stay grouped and that final
> metrics come from held-out data."

### PowerShell Commands to Inspect the Run

Use these after the smoke block finishes.

Recommended script-based checks:

```powershell
python tools/inspect_smoke_run.py files
python tools/inspect_smoke_run.py dataset
python tools/inspect_smoke_run.py targets
python tools/inspect_smoke_run.py classification
python tools/inspect_smoke_run.py regression
python tools/inspect_smoke_run.py features
python tools/inspect_smoke_run.py predictions
```

Run everything at once:

```powershell
python tools/inspect_smoke_run.py all
```

What each investigation answers:

| Command | Question it answers |
|---|---|
| `files` | Did the smoke workflow create every expected artifact? |
| `dataset` | How many run rows and timestep rows were created? What do the first rows look like? |
| `targets` | Are the labels usable? Are there both partitioned and non-partitioned cases? |
| `classification` | What are the RF classification metrics and confusion matrix? |
| `regression` | What are MAE, RMSE, R2, and ranking metrics for `gcc_frac_min`? |
| `features` | Which inputs did the Random Forest use most heavily? |
| `predictions` | What did the held-out test predictions look like? |

Manual PowerShell checks, if Alex wants to inspect the raw files directly:

Check files were created:

```powershell
Get-ChildItem data\tier1_design_*.csv
Get-ChildItem artifacts\smoke\rf
```

Look at the first run-level rows:

```powershell
Import-Csv data\tier1_design_runs.csv |
  Select-Object -First 8 run_id,num_planes,sats_per_plane,total_satellites,node_failure_prob,edge_failure_prob,partition_any,gcc_frac_min,gcc_frac_mean |
  Format-Table -AutoSize
```

Look at the first timestep rows:

```powershell
Import-Csv data\tier1_design_steps.csv |
  Select-Object -First 12 run_id,t,num_nodes,num_edges,num_components,gcc_size,gcc_frac,partitioned |
  Format-Table -AutoSize
```

Open the target summary:

```powershell
Import-Csv artifacts\smoke\target_summary.csv | Format-Table -AutoSize
```

Inspect classification metrics:

```powershell
$cls = Get-Content artifacts\smoke\rf\design_risk_model_tier1_metrics.json | ConvertFrom-Json
$cls | Select-Object num_samples,train_size,val_size,test_size,split_strategy,accuracy,precision,recall,f1,roc_auc | Format-List
$cls.confusion_matrix | ForEach-Object { $_ -join " " }
```

Read the classification confusion matrix as:

```text
TN FP
FN TP
```

For example, `1 0` and `1 4` means:

- 1 true negative
- 0 false positives
- 1 false negative
- 4 true positives

Inspect regression metrics:

```powershell
$reg = Get-Content artifacts\smoke\rf\rf_gcc_frac_min_metrics.json | ConvertFrom-Json
$reg | Select-Object num_samples,train_size,val_size,test_size,split_strategy,test_mae,test_rmse,test_r2,test_spearman_rho,test_kendall_tau | Format-List
```

Inspect feature importances:

```powershell
Import-Csv artifacts\smoke\rf\design_risk_model_tier1_feature_importance.csv |
  Sort-Object { [double]$_.importance } -Descending |
  Select-Object -First 10 |
  Format-Table -AutoSize

Import-Csv artifacts\smoke\rf\rf_gcc_frac_min_feature_importance.csv |
  Sort-Object { [double]$_.importance } -Descending |
  Select-Object -First 10 |
  Format-Table -AutoSize
```

Inspect held-out test predictions:

```powershell
Import-Csv artifacts\smoke\rf\design_risk_model_tier1_predictions.csv |
  Where-Object { $_.split -eq "test" } |
  Select-Object sample_idx,run_id,y_true,y_pred |
  Format-Table -AutoSize

Import-Csv artifacts\smoke\rf\rf_gcc_frac_min_predictions.csv |
  Where-Object { $_.split -eq "test" } |
  Select-Object sample_idx,run_id,y_true,y_pred |
  Format-Table -AutoSize
```

Open the plots:

```powershell
Invoke-Item artifacts\smoke\rf\design_risk_model_tier1_confusion_matrix.png
Invoke-Item artifacts\smoke\rf\rf_gcc_frac_min_prediction_vs_actual.png
```

### Step 1: Generate the smoke dataset

```bash
python scripts/export_design_dataset.py --smoke --seed 42
```

What it does:

- Samples 24 small Tier 1 constellation/failure configurations.
- Builds Walker Delta constellations.
- Propagates orbits through the Tier 1 Hypatia adapter.
- Computes satellite-to-satellite links.
- Injects node and edge failures.
- Measures connectivity over time.
- Writes one run-level CSV and one timestep-level CSV.

Expected outputs:

```text
data/tier1_design_runs.csv
data/tier1_design_steps.csv
```

Quick checks:

```bash
head -5 data/tier1_design_runs.csv
head -5 data/tier1_design_steps.csv
wc -l data/tier1_design_runs.csv data/tier1_design_steps.csv
```

Advisor talking point:

> "The runs file is one row per simulation run. The steps file keeps the
> temporal trace. We train the Random Forest on run-level labels and keep the
> temporal graph sequence available for the GNN."

### Step 2: Audit the targets

```bash
python tools/analyze_dataset_targets.py data/tier1_design_runs.csv \
  --output artifacts/smoke/target_summary.csv \
  --output-corr artifacts/smoke/target_correlations.csv
```

Expected outputs:

```text
artifacts/smoke/target_summary.csv
artifacts/smoke/target_correlations.csv
```

Why this matters:

- Confirms target distributions before training.
- Helps detect all-one/all-zero classification labels.
- Reports whether design signatures are duplicated.

Advisor talking point:

> "Before training, I check whether the target has enough variation. If all
> examples are partitioned or all examples are healthy, classification metrics
> would be meaningless."

### Step 3: Train Random Forest classification

```bash
python scripts/train_design_risk_model.py \
  --smoke \
  --target-name partition_any \
  --seed 42
```

This predicts whether a run ever partitions.

Expected outputs:

```text
artifacts/smoke/rf/design_risk_model_tier1.joblib
artifacts/smoke/rf/design_risk_model_tier1_metrics.json
artifacts/smoke/rf/design_risk_model_tier1_predictions.csv
artifacts/smoke/rf/design_risk_model_tier1_feature_importance.csv
artifacts/smoke/rf/design_risk_model_tier1_confusion_matrix.png
artifacts/smoke/rf/design_risk_model_tier1_prediction_vs_actual.png
```

Inspect the metrics:

```bash
cat artifacts/smoke/rf/design_risk_model_tier1_metrics.json
head -10 artifacts/smoke/rf/design_risk_model_tier1_predictions.csv
cat artifacts/smoke/rf/design_risk_model_tier1_feature_importance.csv
```

During the readiness audit, this smoke run produced approximately:

```text
Accuracy: 0.8333
Precision: 1.0000
Recall: 0.8000
F1: 0.8889
```

Do not present those as final research results. They are a smoke-test sanity
check on a tiny dataset.

### Step 4: Train Random Forest regression

```bash
python scripts/train_design_risk_model.py \
  --smoke \
  --target-name gcc_frac_min \
  --seed 42
```

This predicts the worst observed Giant Connected Component fraction for a run.
Lower `gcc_frac_min` means the network became more fragmented.

Expected outputs:

```text
artifacts/smoke/rf/rf_gcc_frac_min.joblib
artifacts/smoke/rf/rf_gcc_frac_min_metrics.json
artifacts/smoke/rf/rf_gcc_frac_min_predictions.csv
artifacts/smoke/rf/rf_gcc_frac_min_feature_importance.csv
artifacts/smoke/rf/rf_gcc_frac_min_prediction_vs_actual.png
```

Inspect RMSE and related metrics:

```bash
cat artifacts/smoke/rf/rf_gcc_frac_min_metrics.json
head -10 artifacts/smoke/rf/rf_gcc_frac_min_predictions.csv
```

During the readiness audit, this smoke run produced approximately:

```text
MAE:  0.2335
RMSE: 0.2579
R2:   0.3417
```

Again, this is only a smoke-test value. The number demonstrates that evaluation
is wired correctly; it is not enough data for a dissertation result.

---

## 3. What RMSE Means

`gcc_frac_min` is a fraction between 0 and 1.

If RMSE is `0.26`, that means the model's typical squared-error scale is about
0.26 GCC-fraction units. Because RMSE squares errors before averaging, one very
bad prediction hurts more than several small misses.

Plain-language version:

> "RMSE tells us how far the regression prediction is from the true resilience
> label, in the same units as the target. For `gcc_frac_min`, lower RMSE is
> better, and large mistakes are penalized heavily."

Use RMSE with MAE and R2:

| Metric | What Alex should say |
|---|---|
| MAE | Average absolute miss. Easy to interpret in target units. |
| RMSE | Like MAE, but punishes large misses more strongly. |
| R2 | Fraction of target variance explained versus a mean-only baseline. |
| Spearman | Whether the model ranks designs in the right order. |
| Kendall | Pairwise ranking agreement between predicted and true scores. |

What good looks like:

- MAE and RMSE should decrease as training data improves.
- R2 should move above 0 and ideally toward 1.
- Ranking metrics matter because the project is also about comparing designs,
  not only predicting exact labels.

---

## 4. What the Classification Metrics Mean

`partition_any` is a binary label:

- `0`: the run never fell below the partition/GCC threshold.
- `1`: the run partitioned at least once.

| Metric | What Alex should say |
|---|---|
| Accuracy | Fraction of correct predictions. Can be misleading if classes are imbalanced. |
| Precision | Of the designs predicted risky, how many were actually risky? |
| Recall | Of the actually risky designs, how many did the model catch? |
| F1 | Balance between precision and recall. Useful when both false alarms and misses matter. |
| Confusion matrix | Counts true negatives, false positives, false negatives, and true positives. |

For this project, recall is important because missing a fragile constellation is
worse than flagging one for extra review.

---

## 5. Full Random Forest Run

Use this after the smoke path works.

```bash
python scripts/export_design_dataset.py \
  --num-runs 500 \
  --seed 42 \
  --duration 10 \
  --step-seconds 60 \
  --output-dir data/

python tools/analyze_dataset_targets.py data/tier1_design_runs.csv \
  --output artifacts/target_summary.csv \
  --output-corr artifacts/target_correlations.csv

python scripts/train_design_risk_model.py \
  --target-name partition_any \
  --seed 42

python scripts/train_design_risk_model.py \
  --target-name gcc_frac_min \
  --seed 42
```

Default full-run outputs go to `models/`:

```text
models/design_risk_model_tier1.joblib
models/design_risk_model_tier1_metrics.json
models/design_risk_model_tier1_predictions.csv
models/design_risk_model_tier1_feature_importance.csv

models/rf_gcc_frac_min.joblib
models/rf_gcc_frac_min_metrics.json
models/rf_gcc_frac_min_predictions.csv
models/rf_gcc_frac_min_feature_importance.csv
```

Advisor talking point:

> "The Random Forest is the baseline. It is fast, reproducible, and gives
> feature importances. The GNN is the thesis model, but the RF baseline is what
> tells us whether a more complex temporal model is worth the cost."

---

## 6. Temporal GNN Status

The intended GNN command is:

```bash
python scripts/train_gnn_model.py \
  --target-name partition_any \
  --epochs 20 \
  --hidden-dim 64 \
  --data-dir data/ \
  --output-model models/satellite_gnn.pt \
  --device auto
```

Smoke command:

```bash
python scripts/train_gnn_model.py \
  --smoke \
  --target-name partition_any \
  --device cpu
```

Current audit finding:

- The GNN code now splits by complete simulation run into train, validation,
  and test sets.
- It saves a best-validation checkpoint and metrics JSON.
- On the audited machine, the run is blocked by the optional
  `torch_geometric_temporal` dependency, specifically the `torch-sparse` build.

What Alex should say if asked:

> "The Temporal GNN design is implemented around complete graph sequences, not
> individual timesteps, so the split avoids timestep leakage. The remaining
> blocker is environment packaging for the optional PyTorch Geometric Temporal
> stack. I can demonstrate the RF pipeline tonight and explain the GNN path and
> its dependency blocker honestly."

Do not claim GNN training metrics until that dependency is installed and the
smoke command completes.

---

## 7. Data Leakage Explanation

The main leakage risk in this project is splitting timesteps from the same
simulation run across train, validation, and test sets.

Correct behavior:

```text
Run 001: G0, G1, G2, ..., GT -> train only
Run 002: G0, G1, G2, ..., GT -> validation only
Run 003: G0, G1, G2, ..., GT -> test only
```

Incorrect behavior:

```text
Run 001 timestep 0 -> train
Run 001 timestep 1 -> validation
Run 001 timestep 2 -> test
```

Why incorrect behavior is bad:

- Adjacent timesteps from the same run are highly correlated.
- The model could effectively see the same scenario during training and test.
- Metrics would look better than real generalization.

What was fixed/audited:

- Random Forest training groups by `run_id` when splitting.
- Temporal GNN split indices are complete run indices.
- Validation is used for model selection; test is held out for final evaluation.

Advisor talking point:

> "I split by simulation run because the scientific unit of observation is the
> full run. This prevents timestep leakage and gives a more honest estimate of
> generalization to unseen constellation/failure scenarios."

---

## 8. Quick Artifact Checklist

After a successful smoke run, verify:

```bash
ls -la data/tier1_design_runs.csv data/tier1_design_steps.csv
ls -la artifacts/smoke/rf/
```

Must-have files:

- `tier1_design_runs.csv`: one row per simulation run.
- `tier1_design_steps.csv`: per-timestep connectivity trace.
- `*_metrics.json`: evaluation metrics.
- `*_predictions.csv`: true/predicted labels with split labels.
- `*_feature_importance.csv`: RF interpretability artifact.
- `*_confusion_matrix.png`: classification diagnostic.
- `*_prediction_vs_actual.png`: regression/classification sanity plot.
- `*.joblib`: trained Random Forest checkpoint.

---

## 9. Troubleshooting

### Dataset not found

Error:

```text
ERROR: Dataset not found at .../data/tier1_design_runs.csv
```

Fix:

```bash
python scripts/export_design_dataset.py --smoke --seed 42
```

### Python/package conflict

Symptom:

```text
AttributeError: module 'numpy' has no attribute 'long'
```

Likely cause: incompatible NumPy/SciPy/scikit-learn combination under Python
3.14.

Fix: create a clean Python 3.11 or 3.12 environment and reinstall dependencies.

### Classification target has one class

Symptom: model training fails or metrics are meaningless because all rows have
`partition_any=0` or all rows have `partition_any=1`.

Fixes:

- Generate more runs.
- Broaden node/edge failure probabilities.
- Use a regression target such as `gcc_frac_min` for the demo.

### GNN missing dependency

Symptom:

```text
Temporal GNN dependency unavailable: No module named 'torch_geometric_temporal'
```

Fix: install the PyTorch Geometric Temporal stack for the exact Python and Torch
version being used. If that cannot be resolved before the meeting, do not block
the demo. Show the RF pipeline and document the GNN environment issue.

---

## 10. Alex's Short Demo Script

1. "I start by generating a Tier 1 temporal dataset from sampled Walker Delta
   constellations."
2. "Each run propagates orbits, computes viable ISLs, injects failures, and
   records connectivity over time."
3. "The run-level CSV is for Random Forest. The timestep/graph sequence is for
   the Temporal GNN."
4. "I audit target distributions before training so I know the labels are
   usable."
5. "The Random Forest baseline predicts `partition_any` for classification and
   `gcc_frac_min` for regression."
6. "For classification I inspect accuracy, precision, recall, F1, and the
   confusion matrix."
7. "For regression I inspect MAE, RMSE, R2, and ranking correlations."
8. "RMSE is in the same unit as the target and penalizes large errors."
9. "Splits are by run_id, not timestep, to prevent leakage."
10. "The next engineering step is stabilizing the GNN dependency stack and then
    running larger experiments with the same reproducible commands."

---

## 11. What Not to Claim Tomorrow

Do not claim:

- The smoke metrics prove final model performance.
- The GNN has final results if the optional dependency is still blocked.
- The current dataset is large enough for dissertation-scale conclusions.
- Feature importance proves causality.

Safe claims:

- The Tier 1 pipeline produces reproducible run-level and timestep-level data.
- The Random Forest baseline trains and evaluates end to end.
- Evaluation artifacts include metrics, predictions, feature importances, and
  plots.
- The leakage risk has been addressed by splitting complete runs.
- Larger experiments can use the same commands with larger `--num-runs`.
