# Candidate Feature Inventory

## Future random-forest inputs

### Satellite design variables

- Number of planes.
- Satellites per plane.
- Configured satellite count.
- Altitude.
- Inclination.
- Walker phasing factor.
- ISL policy and fixed ISL search parameters.
- Fixed temporal profile.

### Ground design variables

- Selected civilian count.
- Selected government count.
- Selected military count.
- Total selected ground count.
- Synthetic catalog identity when comparing catalog designs.

### Failure probabilities

- Persistent satellite-node failure probability.
- Persistent accepted-edge failure probability.
- Persistent ground-station failure probability.
- Versioned satellite and ground failure-model identities.

### Visibility-policy variables

- Minimum elevation threshold.
- Visibility-model and frame-contract versions.

### Service-policy variables

- Original-denominator space GCC threshold.
- Ground-service threshold.

### Aggregate pre-outcome features

- Deterministic design-only constellation size and class-composition ratios.
- Policy identities or explicit policy values.
- No post-rollout connectivity or service result belongs in this group.

## Prohibited leakage fields

- Final classification or regression target values.
- G4 or G5 service summaries when predicting those same outcomes.
- Failure-adjusted labels or breach states.
- Realized failed-node, failed-edge, or failed-station counts when the task is pre-run design prediction.
- Run, scientific, sequence, record, or replay hashes.
- Replay status, generation status, runtime, or artifact size.
- Future timestep values in a forecasting task.
- Full-sequence statistics when predicting a future suffix from a prefix.

## Future TGNN concepts

Potential satellite-node concepts include orbital-plane identity, normalized orbital position, operational eligibility, and prior-prefix graph state. Potential ground-node concepts include class, deterministic coordinates, selected eligibility, and visibility-policy context. Potential ISL and satellite-ground edge concepts include physical link attributes already present in canonical graphs and explicit edge kind.

These are conceptual candidates only. No TGNN implementation or feature schema is changed by this pilot.

## Assessment mode distinction

The current TGNN consumes complete temporal sequences and is therefore an ex-post run-level assessment. A future-prefix forecasting experiment must explicitly truncate inputs at a declared cutoff and prohibit all suffix information. Results from these two tasks are not interchangeable.
