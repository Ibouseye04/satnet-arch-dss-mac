# SATNET Stage A Development Acceptance Authorization v3 Proposal

This directory is an inactive superseding authorization proposal for operation `ACCEPT` on the complete frozen `development` partition: 20 designs and 100 runs, from `SA-D000-R00` through `SA-D027-R04`.

## Supersession and remediation basis

The proposal is based exactly on independent remediation audit commit `26abed7e28c94bbd46176d1cc77dbee7ebbcebe6`, whose sole parent is remediation HEAD `34d153316e85c7cca91c9a494d94356368588ed6`. The exact audit verdict is `APPROVED FOR SUPERSEDING STAGE A DEVELOPMENT ACCEPTANCE AUTHORIZATION V3 PROPOSAL`.

The first authorized v2 ACCEPT attempt failed before acceptance-root creation with console error `Replay report identity or seed mismatch: stable_executable_commit`. The defect compared historical replay-report executable fields against the current ACCEPT executable instead of validated historical replay provenance. The corrected stable executable is `1efd7fcea6a1ba7feb318edbd53577126dd94ae4`, with executable inventory SHA-256 `2927c74a38fc9baf1456dd91f5c1c9ca42a89d10fcd8f63227d40858faff1c00`, stable identity SHA-256 `cbe2e66888c11a6f8f39b6f975acdd3e230b9214921eb98740d84ec41449fc47`, and 71 executable records.

Proposal v1, proposal v2, and active authorization v2 remain byte-unchanged historical artifacts. Their historical status is recorded only in the new governance supersession record. The failed attempt did not consume authorization, did not execute a successful acceptance comparison, produced no acceptance evidence, created no acceptance root, and requires no recovery. Active v2 is nevertheless not reusable because it binds stable executable `b1da92c3184ee86997079f1eb458fae95d994c22` and inventory `2ebac3bb615ed3204edc1359a78f78e4354495364fa1f76dfd1648b417df7b56`, not the corrected executable.

## Historical source provenance

The immutable generation ledger is exactly 1,060,132 bytes with SHA-256 `9df1448612d4719d164cf014a2ab60d932d82ab07a90e05c50a0c1650b097328`. Its 100 records are `SUCCEEDED`, science `PASSED`, and adapter `PASSED`. Its 1,900 bound artifacts total 365,309,146 bytes with zero mismatches. The complete generation tree contains 1,902 files and 366,369,907 bytes with tree SHA-256 `1be2fe3523d40a900a0283f41b55f8f7fbdeba1ea9c394325c0003d914169e23`.

The immutable replay ledger is exactly 1,079,014 bytes with SHA-256 `140bda39b3f519d7baeae09d222cd3f6c80791ff03215c11c48dd66fb6e8b9bb`. Its 100 records are `SUCCEEDED`, science `PASSED`, and adapter `PASSED`; exactly 100 replay reports are present. Its 2,000 bound artifacts total 365,491,986 bytes with zero mismatches. The complete replay tree contains 2,001 files and 366,571,000 bytes with tree SHA-256 `e5dcd6d51d55f072b2531e7cbd684b712e605f56be6f9332d051ebb9cdba06a3`. Replay binds exactly to the generation ledger.

Historical replay-report executable fields validate against replay stable executable `8bde92da762269998632eb6c3e3cb2565a6ca971`, not the corrected current ACCEPT executable. The historical generation and replay campaign, plan, authorization, executable-inventory, and stable-identity bindings are preserved in the proposal and governance documents.

## Prospective ACCEPT planning reference

The corrected implementation independently reconstructed a nonauthorizing planning-reference authorization with SHA-256 `878265a1def0fea5e8ec89139d587153956a6176172f3af98731964e56a35db7` and planning-reference plan SHA-256 `9c0149dd441f177e4f104eec17687e817a66d4a42cd2cac7d1703ae4f33fb64e`. Neither reuses the v2 runtime authorization or runtime plan.

The prospective operation-bound-v2 ACCEPT campaign identity is `7b6226dc4bd939a8f6cda50662f6ac4f1e4865c04e59a7b695a9100eea51c229` under `satnet.stage_a.campaign_manifest.operation_bound.v2`. It was reconstructed from the canonical payload and differs from historical replay legacy-v1 campaign `3a8a62a5e73c81503e5b75d0c44a820eb545596968d837aa72e7e1719a7e3a50`, operation-bound replay-v2 campaign `cc50220189cc849431319fc9ce80b486e76beb6de3c25164ec24f74ff024cb44`, and historical v2 runtime ACCEPT campaign `c7020e85471d12a5cdb1ff33df614bbd3d04ff58f422ad5c4d60d87328bc3bf6` because the corrected executable binding is part of the campaign payload.

## Inactive governance state

`authorization_status` is `PROPOSED`. `authorization_active`, `execution_authorized`, `simulation_authorized`, and `production_authorized` are all `false`. This proposal does not authorize `GENERATE`, `REPLAY`, validation, sealed holdout, Stage B, RF training, or TGNN training.

The reserved acceptance root `C:\Users\johns\satnet-stage-a-discovery-v1-acceptance` remains absent. There are zero failed or in-progress source directories and zero campaign, recovery, or run locks. No acceptance evidence exists.

V3 requires independent review of the exact proposal, manifest, governance, README, and inventory bytes followed by a separate explicit user-controlled activation. Activation must create a new runtime authorization and derive a new runtime plan from that authorization. This proposal does not activate v3 and does not authorize or execute ACCEPT.

The deterministic inventory excludes its own bytes. The proposal semantic SHA-256 excludes only the `proposal_semantic_sha256` field and hashes canonical pretty JSON with sorted keys, two-space indentation, UTF-8 encoding, and one trailing LF.
