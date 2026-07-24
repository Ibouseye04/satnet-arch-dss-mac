# SATNET Stage A Development Acceptance Authorization v2 Proposal

This directory is an inactive superseding authorization proposal for operation `ACCEPT` on the complete frozen `development` partition: 20 designs and 100 runs, from `SA-D000-R00` through `SA-D027-R04`.

## Supersession basis

The proposal is based exactly on independent audit commit `9c85fc614949b1b99eb57473751e46cc0b598665`, whose sole parent is remediation HEAD `3c8389252f46af81102df1dd10cffd95c21fc01f`. The exact audit verdict is `APPROVED FOR SUPERSEDING STAGE A DEVELOPMENT ACCEPTANCE AUTHORIZATION PROPOSAL`.

Acceptance proposal v1 remains unchanged historical evidence. Its proposal semantic SHA-256 is `d6d1d6b4cf1679f341a934372de07bc5a6e1f28cab647cfd10b373f482f9a00d`, exact proposal-file SHA-256 is `68fe8890e1c4eabc283ea3954f966388c6a92897b138d51666ae6a9bdef186f4`, and proposal inventory SHA-256 is `b908d7fe1aedd33b2476d4f562d3b442054ff93d0119aa53faf46ad3e8255dd7`. Proposal v1 is recorded as `SUPERSEDED` only in the new governance supersession record; none of its files are modified.

## Historical generation legacy-v1 provenance

The completed generation root `C:\Users\johns\satnet-stage-a-discovery-v1-production` is immutable read-only evidence. Its `execution_ledger.json` is exactly 1,060,132 bytes with SHA-256 `9df1448612d4719d164cf014a2ab60d932d82ab07a90e05c50a0c1650b097328`. All 100 records are `SUCCEEDED`; all 100 adapter validations and science validations are `PASSED`; and all 1,900 ledger-bound artifacts totaling 365,309,146 bytes rehashed successfully.

The historical generation ledger retains legacy-v1 campaign identity `af76d64bbabdb055d413a1c3e2f28809749710d67f7819465d45c0ef0769e164` under `satnet.stage_a.campaign_manifest.legacy.v1`. It is bound to stable executable `fafe3fe36eac4429c860bd6d281923fed2980ea7`, executable inventory `1066d17645d4975a40a93a80c1ca9797b1b28cf37531337355ccb3297fd13b2d`, plan `cd32f773fe9ce2853b3a0bae699bf4a5199cfe67dc5baae767858387b0a36d2e`, and authorization semantic identity `96656da7341dada1b58626c480a9de4e3183e1e46c10dd1e96fd07bad66b6f11`.

## Historical replay legacy-v1 provenance

The completed replay root `C:\Users\johns\satnet-stage-a-discovery-v1-replay` is immutable read-only evidence. Its `replay_ledger.json` is exactly 1,079,014 bytes with SHA-256 `140bda39b3f519d7baeae09d222cd3f6c80791ff03215c11c48dd66fb6e8b9bb`. All 100 records are `SUCCEEDED`; all 100 adapter validations and science validations are `PASSED`; all 2,000 ledger-bound artifacts totaling 365,491,986 bytes rehashed successfully; and exactly 100 `replay_report.json` artifacts are present. The replay ledger binds exactly to the generation ledger byte length and SHA-256.

The historical replay ledger retains legacy-v1 campaign identity `3a8a62a5e73c81503e5b75d0c44a820eb545596968d837aa72e7e1719a7e3a50` under `satnet.stage_a.campaign_manifest.legacy.v1`. It is bound to stable executable `8bde92da762269998632eb6c3e3cb2565a6ca971`, executable inventory `d52ffc2afa5f94268d89187afe58d4c7695bdf261ec20113644c9cb7f5840215`, stable identity `45f0a596b073d1799de45660cfa74409daf774ca7bc98c984cb6434b89c4c279`, plan `97e279d972dd1ba7a63253a80d6d9ae624d5f73877ba9c208b233f97c96ba70e`, replay-v3 authorization semantic identity `3d8d303f2b7f1460a0b2017bdfb6fec53f7029410980de99e60ca95e8d2a157d`, exact authorization-file SHA-256 `ef4812d27fffd44348c8bca4c8101b8b1fed476db75074c3b1c7bb3d1c371f03`, and authorization inventory SHA-256 `e6ffcd3520497a4145703d7d4369ba14740a68481a06a099c086eecd9b702656`.

## Current ACCEPT operation-bound-v2 identity

The current executable is stable commit `b1da92c3184ee86997079f1eb458fae95d994c22`, executable inventory SHA-256 `2ebac3bb615ed3204edc1359a78f78e4354495364fa1f76dfd1648b417df7b56`, stable identity SHA-256 `1902a91a38e87faef6fe99c2e1f7aa4c3aa8dd6febf38e6a5b43a272b190c7f9`, and 71 executable records.

The current ACCEPT campaign identity is `c7020e85471d12a5cdb1ff33df614bbd3d04ff58f422ad5c4d60d87328bc3bf6` under `satnet.stage_a.campaign_manifest.operation_bound.v2`. It differs from both the historical replay legacy-v1 identity and the current operation-bound-v2 REPLAY identity `cc50220189cc849431319fc9ce80b486e76beb6de3c25164ec24f74ff024cb44`.

The deterministic audit-commit planning reference has plan SHA-256 `4d90038ed8f95a76285b73e9ea32ccedc4536964d2a73a4e5420530642e74dea`. It is a nonauthorizing planning identity only. Activation must create a separate runtime authorization, and the activated runtime plan identity must be derived and verified from that exact authorization.

Both sources and the current proposal bind frozen contract `e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a`, tooling proposal `7b21af1bad6faa10696d1417b28b75300727033446c84424beb94d263510778b`, artifact contract `cc721fd88dedcb90c0aa789b7815c7f6e35132e3aeebf2794e3e1d63a424380d`, and authorized run manifest `7f6013872824552bac32ebb76a5d26eda85f645f339b0eaa0a2fdd8577018642`.

## Inactive governance state

`authorization_status` is `PROPOSED`. `authorization_active`, `execution_authorized`, `simulation_authorized`, and `production_authorized` are all `false`. Acceptance has not executed, and the reserved acceptance root `C:\Users\johns\satnet-stage-a-discovery-v1-acceptance` remains absent. This proposal does not authorize `GENERATE`, `REPLAY`, validation, sealed holdout, Stage B, RF training, or TGNN training.

## Activation requirements

Activation requires independent review of these exact proposal, manifest, governance, README, and inventory bytes, followed by a separate explicit user-controlled activation decision. Any active runtime authorization must be created separately, conform to `satnet.stage_a.execution_authorization.v1`, authorize only `ACCEPT` for the complete development run set, bind both exact source-ledger byte lengths and SHA-256 identities, bind the current stable executable and operation-bound-v2 ACCEPT campaign identity, and pass a fresh authoritative `ACCEPT` preflight.

Before activation, both source roots must again verify unchanged; all 100 run identities, run-record hashes, and four seeds must match; every generation and replay artifact must rehash; all campaign, recovery, and run locks must remain absent; and the acceptance root must remain absent. Activation, acceptance execution, acceptance-root creation, validation, sealed holdout, Stage B, RF training, and TGNN training are outside this proposal.

The deterministic inventory excludes its own bytes to avoid self-reference. The proposal semantic SHA-256 excludes only the `proposal_semantic_sha256` field and hashes canonical pretty JSON with sorted keys, two-space indentation, and one trailing LF.
