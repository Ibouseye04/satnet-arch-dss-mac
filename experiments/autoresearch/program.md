# SATNET RF Autoresearch Program

## Scope
This program is only for SATNET RandomForest autoresearch on an existing `data/tier1_design_runs.csv` dataset.

Do not modify simulation generation.
Do not modify rollout physics.
Do not modify target semantics.
Do not modify GNN code.
Do not modify dataset export code.
Do not redesign `src/satnet/models/risk_model.py`.

## Files You May Edit
- `experiments/autoresearch/candidate.json`
- `experiments/autoresearch/program.md` only if you are explicitly asked to update the runbook

## Files You May Read
- `experiments/autoresearch/mutation_policy.json`
- `experiments/autoresearch/candidate.json`
- `experiments/autoresearch/incumbent.json`
- `experiments/autoresearch/last_run.json`
- `experiments/autoresearch/results.jsonl`
- `experiments/autoresearch/<experiment_id>/summary.json`
- `experiments/autoresearch/<experiment_id>/config.json`
- `experiments/autoresearch/<experiment_id>/metrics.json`

## Files You Must Not Edit
- `src/satnet/simulation/**`
- `src/satnet/network/**`
- `src/satnet/models/gnn*`
- `scripts/export_*`
- `scripts/train_gnn_model.py`
- `src/satnet/models/risk_model.py`
- `experiments/autoresearch/incumbent.json`
- `experiments/autoresearch/results.jsonl`
- `experiments/autoresearch/last_run.json`
- any per-run `summary.json`
- any per-run `config.json`
- any per-run `metrics.json`

## Allowed Mutation Surface
Edit only these fields in `experiments/autoresearch/candidate.json`:
- `target_name` in `{gcc_frac_min, partition_fraction}`
- `feature_mode` in `{tier1_full, design_only}`
- `n_estimators` in `{100, 300, 600}`
- `max_depth` in `{null, 12, 20}`

Do not edit any other field.

## Screening Command
Run exactly:

```bash
python scripts/run_autoresearch_rf.py --mode candidate
```

This command reads `experiments/autoresearch/candidate.json`, validates it against `experiments/autoresearch/mutation_policy.json`, executes one screening run, writes per-run `summary.json`, updates `experiments/autoresearch/last_run.json`, and may update `experiments/autoresearch/incumbent.json`.

## Confirmatory Command
Run exactly:

```bash
python scripts/run_autoresearch_rf.py --mode confirmatory --source-experiment-id <SCREENING_EXPERIMENT_ID>
```

Use confirmatory only for a screening experiment that already achieved `promotion_decision = promoted`.
In this RF-only v1 slice, confirmatory reuses the same RandomForest evaluation path with a separate `fidelity_tier` label.
Treat it as workflow confirmation within the bounded loop, not as independent scientific confirmation from a new split, new seed, or stronger evaluator.

## Source of Truth
Never use raw stdout or stderr as the scientific result.
Read only JSON files.
Primary read order:
1. `experiments/autoresearch/last_run.json`
2. the per-run `summary.json` path named inside `last_run.json`, only when `summary_path` is not `null`
3. `experiments/autoresearch/incumbent.json`

## Improvement Rule
A screening candidate is better only if the run summary says:
- `status = succeeded`
- `promotion_decision = promoted`

Promotion is already machine-computed by SATNET using:
- primary: lower `test_rmse`
- secondary tiebreak: higher `test_spearman_rho`
- tertiary tiebreak: higher `test_r2`

Do not recompute promotion yourself.
Do not manually edit the incumbent file.

## Incumbent Update Rule
Incumbent updates happen only through the approved command path.
If a run is promoted, the command updates `experiments/autoresearch/incumbent.json` for the matching `(dataset_path, target_name, fidelity_tier)` scope.
Screening and confirmatory incumbents are separate scopes.

## Failure Handling
If `last_run.json` shows:
- `status = failed`
- or `promotion_decision = failed`

then do not edit any history file.
Read `promotion_reason`, `error_type`, and `error_message` from `last_run.json`.
Only read per-run `metrics.json` when `summary_path` is present and points to a real experiment directory.
Make at most one legal mutation in `candidate.json` and rerun.
Stop after 2 consecutive failed runs.

## Stop Conditions
Stop when any of these is true:
- a confirmatory run is promoted
- 5 consecutive screening runs are rejected
- 2 consecutive runs fail
- there is no unused legal mutation left in the allowed mutation surface
- you are asked to stop

## Required Loop
1. Read `mutation_policy.json`, `candidate.json`, and `incumbent.json`.
2. Make one legal mutation in `candidate.json`.
3. Run the screening command.
4. Read `last_run.json`, then read the per-run `summary.json` only if `summary_path` is not `null`.
5. If screening was promoted and confirmatory is warranted, run the confirmatory command using that screening experiment ID.
6. Read `last_run.json` again.
7. Stop or continue using the stop rules above.
