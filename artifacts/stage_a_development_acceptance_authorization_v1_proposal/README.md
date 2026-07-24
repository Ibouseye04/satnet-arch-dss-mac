# SATNET Stage A Development Acceptance Authorization v1 Proposal

This directory is an inactive authorization proposal for operation `ACCEPT` on the complete frozen `development` partition: 20 designs and 100 runs, from `SA-D000-R00` through `SA-D027-R04`.

## Bound generation provenance

The completed generation root `C:\Users\johns\satnet-stage-a-discovery-v1-production` is immutable read-only evidence. Its `execution_ledger.json` is exactly 1,060,132 bytes with SHA-256 `9df1448612d4719d164cf014a2ab60d932d82ab07a90e05c50a0c1650b097328` and schema `satnet.stage_a.execution_ledger.v2`. All 100 records are `SUCCEEDED`; all 100 adapter validations and science validations are `PASSED`; and all 1,900 ledger-bound artifacts totaling 365,309,146 bytes rehashed successfully.

The generation ledger is bound to stable executable `fafe3fe36eac4429c860bd6d281923fed2980ea7`, executable inventory `1066d17645d4975a40a93a80c1ca9797b1b28cf37531337355ccb3297fd13b2d`, campaign manifest `af76d64bbabdb055d413a1c3e2f28809749710d67f7819465d45c0ef0769e164`, plan `cd32f773fe9ce2853b3a0bae699bf4a5199cfe67dc5baae767858387b0a36d2e`, and generation authorization semantic identity `96656da7341dada1b58626c480a9de4e3183e1e46c10dd1e96fd07bad66b6f11`.

## Bound replay provenance

The completed replay root `C:\Users\johns\satnet-stage-a-discovery-v1-replay` is immutable read-only evidence. Its `replay_ledger.json` is exactly 1,079,014 bytes with SHA-256 `140bda39b3f519d7baeae09d222cd3f6c80791ff03215c11c48dd66fb6e8b9bb` and schema `satnet.stage_a.execution_ledger.v2`. All 100 records are `SUCCEEDED`; all 100 adapter validations and science validations are `PASSED`; all 2,000 ledger-bound artifacts totaling 365,491,986 bytes rehashed successfully; and exactly 100 `replay_report.json` artifacts are present. The replay ledger binds exactly to the generation ledger byte length and SHA-256.

The replay ledger is bound to stable executable `8bde92da762269998632eb6c3e3cb2565a6ca971`, executable inventory `d52ffc2afa5f94268d89187afe58d4c7695bdf261ec20113644c9cb7f5840215`, stable identity `45f0a596b073d1799de45660cfa74409daf774ca7bc98c984cb6434b89c4c279`, campaign manifest `3a8a62a5e73c81503e5b75d0c44a820eb545596968d837aa72e7e1719a7e3a50`, plan `97e279d972dd1ba7a63253a80d6d9ae624d5f73877ba9c208b233f97c96ba70e`, replay-v3 authorization semantic identity `3d8d303f2b7f1460a0b2017bdfb6fec53f7029410980de99e60ca95e8d2a157d`, exact-file identity `ef4812d27fffd44348c8bca4c8101b8b1fed476db75074c3b1c7bb3d1c371f03`, and authorization inventory `e6ffcd3520497a4145703d7d4369ba14740a68481a06a099c086eecd9b702656`.

Both sources also bind frozen contract `e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a`, tooling proposal `7b21af1bad6faa10696d1417b28b75300727033446c84424beb94d263510778b`, artifact contract `cc721fd88dedcb90c0aa789b7815c7f6e35132e3aeebf2794e3e1d63a424380d`, and authorized run manifest `7f6013872824552bac32ebb76a5d26eda85f645f339b0eaa0a2fdd8577018642`.

## Inactive governance state

`authorization_status` is `PROPOSED`. `authorization_active`, `execution_authorized`, `simulation_authorized`, and `production_authorized` are all `false`. Acceptance has not executed, and the reserved acceptance root `C:\Users\johns\satnet-stage-a-discovery-v1-acceptance` remains absent. This proposal does not authorize `GENERATE`, `REPLAY`, validation, sealed holdout, Stage B, RF training, or TGNN training.

## Activation controls

Activation requires independent review of these exact bytes and identities, followed by a separate explicit user-controlled activation decision. Any active runtime authorization must be created separately, conform to `satnet.stage_a.execution_authorization.v1`, authorize only `ACCEPT` for the complete development run set, preserve both exact source-ledger byte-length and SHA-256 bindings, and pass a fresh authoritative `ACCEPT` preflight while both source roots remain unchanged, all campaign/recovery/run locks remain absent, and the acceptance root remains absent.

The deterministic inventory excludes its own bytes to avoid self-reference. The proposal semantic SHA-256 excludes only the `proposal_semantic_sha256` field and hashes canonical pretty JSON with sorted keys, two-space indentation, and one trailing LF.
