# SATNET Stage A Development Replay Authorization v1 Proposal

This directory is an inactive authorization proposal for operation `REPLAY` on the complete frozen `development` partition: 20 designs and 100 runs, from `SA-D000-R00` through `SA-D027-R04`.

## Bound evidence

The completed generation root `C:\Users\johns\satnet-stage-a-discovery-v1-production` is read-only evidence. Its `execution_ledger.json` is exactly 1,060,132 bytes with SHA-256 `9df1448612d4719d164cf014a2ab60d932d82ab07a90e05c50a0c1650b097328`. The ledger uses `satnet.stage_a.execution_ledger.v2`; all 100 records are `SUCCEEDED`; all 100 adapter validations and authoritative science-completion validations are `PASSED`; all four frozen seeds match; and every ledger-bound artifact was present and hash-verified before proposal creation.

The authorized replay root is `C:\Users\johns\satnet-stage-a-discovery-v1-replay`. It was absent when this proposal was created and this proposal does not create it.

## Inactive governance state

`authorization_status` is `PROPOSED`. `authorization_active`, `execution_authorized`, `simulation_authorized`, and `production_authorized` are all `false`. This proposal does not authorize or execute `REPLAY`; it also does not authorize `GENERATE`, `ACCEPT`, validation, sealed holdout, Stage B, RF training, or TGNN training.

## Activation requirements

Activation requires an independent review of these exact bytes and identities, followed by a separate explicit user-controlled activation decision. Any active runtime authorization must be created separately, conform to `satnet.stage_a.execution_authorization.v1`, authorize only `REPLAY` for the complete development run set, preserve the exact source-ledger byte-length and SHA-256 binding, and pass a fresh authoritative `REPLAY` preflight while the generation evidence remains unchanged and the replay root remains absent.

The deterministic inventory excludes its own bytes to avoid self-reference. The proposal semantic SHA-256 excludes only the `proposal_semantic_sha256` field and hashes canonical pretty JSON with sorted keys, two-space indentation, and one trailing LF.
