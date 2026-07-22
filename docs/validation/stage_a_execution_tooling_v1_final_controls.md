# SATNET Stage A Execution Tooling v1 Final Controls

## Scope

This correction closes only `BINDING-003`, `BINDING-004`, `BINDING-005`, `BINDING-007`, and `BINDING-008` from the independent re-audit at `6ec3c66b17c1efecf784aac15de6e6e8196fc38d`.

The correction does not redesign the Stage A campaign, alter the frozen scientific contract, create an execution authorization, execute a Stage A simulation, create a reserved Stage A output root, or approve production execution.

## Immutable identities

- Stable executable commit: `fafe3fe36eac4429c860bd6d281923fed2980ea7`
- Focused regression-test commit: `bc1152bf0cd5ec0cc78766b490a9be4387e87c09`
- Frozen Stage A Discovery Contract commit: `301d8a224daa070b15ecc6447f503d42d5d1e70a`
- Frozen Stage A Discovery Contract hash: `e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a`
- Executable source inventory SHA-256: `1066d17645d4975a40a93a80c1ca9797b1b28cf37531337355ccb3297fd13b2d`
- Tooling proposal inventory SHA-256: `7b21af1bad6faa10696d1417b28b75300727033446c84424beb94d263510778b`

The stable executable identity precedes all focused test, generated proposal, and documentation changes. No executable source is permitted to differ from that commit.

## Binding corrections

### BINDING-003

The public production `generate` and `replay` APIs no longer accept an injected adapter or science validator. They construct the production adapter and independent authoritative validator internally. A run cannot transition to `SUCCEEDED` without separate passed artifact-contract evidence and passed authoritative science-completion evidence covering satellite, G1, G2, G3, G4, G5, target, inventory, and result stages.

Synthetic adapters and synthetic completion validation are isolated in a private test harness. The harness rejects the frozen Stage A contract before writing or execution.

### BINDING-004

Replay authorization and plan contracts require the exact source generation-ledger relative path, byte length, and SHA-256. Preflight verifies those bytes before replay. Replay reads the verified bytes, validates the ledger schema and campaign identity, and persists the same source identity in its ledger and each replay report. Semantic-preserving byte changes are rejected.

### BINDING-005

Acceptance authorization and plan contracts require exact generation- and replay-ledger relative paths, byte lengths, and SHA-256 values. Acceptance verifies both authorized byte streams and verifies that the replay ledger records the exact accepted generation-ledger identity.

The complete frozen seed set is carried and compared across plans, generation ledgers, replay ledgers, replay reports, acceptance comparisons, resume validation, and frozen-contract validation:

- `design_construction_seed`
- `ground_selection_seed`
- `satellite_failure_seed`
- `ground_failure_seed`

### BINDING-007

Campaign and per-run locks use exclusive creation and hashed structured payloads. Campaign locks explicitly bind campaign, operation, partition, contract, plan, authorization, stable executable, and tooling proposal identities. Per-run locks additionally bind run key, global run ID, design ID, and realization ID. Both include host identity, process ID, process-start identity, creation time, nonce, and lock hash.

Stale recovery is an explicit action requiring the complete expected identity and a positive configured minimum age. Recovery denies foreign-host locks, active local processes, active campaign owners, changed locks, and valid completed outputs. Recovery writes a hash-addressed immutable event with exclusive creation before deleting the stale lock.

### BINDING-008

The repository-wide `*.py text eol=lf` rule was removed. Narrow `-text` rules now cover only exact inventory-bound roots and files, including the Stage A execution package, protected science roots, proposal generator, and focused validation sources. Fresh-checkout validation must prove exact byte length and SHA-256 for every executable and tooling-inventory record.

## Preserved controls

- The frozen Stage A contract and protected science paths are unchanged.
- Development remains exactly 20 designs and 100 runs.
- Validation remains exactly 5 designs and 25 runs.
- Sealed holdout identities remain inaccessible to ordinary planning and reporting.
- Proposal artifacts retain `execution_authorized: false`, `simulation_authorized: false`, and `production_authorized: false`.
- Reserved Stage A production, replay, acceptance, and evidence-freeze roots remain absent.
- No real authorization, generation ledger, replay ledger, acceptance report, lock, or Stage A scientific output was created by this correction.

## Required focused validation

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest tests/experiments/stage_a_execution -q
python -m pytest tests/experiments/test_final_dataset_isolation.py -q
python -m pytest tests/validation/test_stage_a_execution_tooling_v1_audit.py -q
python -m pytest tests/validation/test_stage_a_execution_tooling_v1_reaudit.py -q
```

A fresh Windows checkout with `core.autocrlf=true` must additionally reproduce every executable and tooling-inventory record byte-for-byte and remain clean.

## Status

The implementation and focused regression evidence are ready for a targeted independent delta re-audit of the five findings above. This record does not authorize Stage A execution.
