# Final Integrated Dataset Generation Qualification

## Verdict

`READY FOR EXTERNAL GENERATION-PIPELINE AUDIT`

Full 500-run production generation and production replay were not executed and are not authorized by this qualification.

## Frozen contract

| Identity | Value |
|---|---|
| Frozen tag | `final-integrated-dataset-contract-v1` |
| Frozen commit | `a1967185e80327e4b00c1831828dc975ab6819fc` |
| Contract specification | `482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b` |
| External contract bundle provenance | `3250dcf66e859a7dba151c6564827fcb89ddbea17a5ab5d39540e3087dd2e2ba` |
| Design manifest | `43ffe79701c7e624abc17c45f198397c20fae55ed243502953aa8c898462bcc8` |
| Run manifest | `2925c3c65cf7e2b6debcc42b6e6414186f3474eb88998486f7571af73dda2a36` |
| Split manifest | `930454b2be6eb5033efebc7ab407c2400c66f0ca998e9283c36890b69ea2e08d` |

The frozen `contract_scope` fields governed the earlier contract-materialization phase. This qualification phase was separately authorized after external approval and tagging.

The contract bundle hash is external package provenance. It is not present in or used by generated satellite, G1–G5, target, scientific inventory, result, replay, or future dataset-row scientific identities.

## Tooling provenance

| Item | Value |
|---|---|
| API audit commit | `7a2e25e299d32c81190ba4ae9c811d3351e436b7` |
| Orchestrator commit | `f3aea5d` |
| Replay and acceptance commit | `aa59204` |
| Candidate tooling SHA | `0e6fbdb89f863ec9ef11fe0271bbcf474ff102f2` |
| Validated tooling SHA | `0e6fbdb89f863ec9ef11fe0271bbcf474ff102f2` |
| Handoff HEAD SHA | Documentation commit containing this report; resolved and reported in the final handoff |

No behavior-bearing change was made after candidate commit `0e6fbdb89f863ec9ef11fe0271bbcf474ff102f2`. Future production execution must explicitly check out an externally approved validated tooling SHA. The expected-tooling argument is operator confirmation, not external approval.

## Catalog

| Identity | Value |
|---|---|
| Raw CSV SHA-256 | `e8855d4ded4c242f3e5b35b90610d5f9c4f218bdf717d2d08b2464cac39f1598` |
| Semantic catalog hash | `810c64dfb030b042311c90f2f42f8dee866a48fc63a6a29e362ee328c52eaa6e` |

The catalog is synthetic, pilot-derived, and globally distributed for controlled engineering validation. It is not scientifically established as representative of real civilian, government, or military infrastructure.

## API and runtime validation

The production API audit found the authoritative satellite and G1–G5 interfaces compatible without protected-science changes. Runtime preflight passed at the exact candidate SHA. It verified the frozen tag, ancestry, clean tracked tree, all eleven tag blobs byte-for-byte, semantic identities, self-hashes, catalog identities, physics model, default link budget, and authoritative satellite and G1–G5 versions.

Execution roots were external to the repository:

| Role | Resolved path | Mode |
|---|---|---|
| Qualification | `C:\Users\johns\satnet-final-qualification-20260720` | `qualification` |
| Replay | `C:\Users\johns\satnet-final-replay-20260720` | `qualification_replay` |
| Deterministic repeat | `C:\Users\johns\satnet-final-repeat-20260720` | `qualification_repeat` |

The repeat root was deleted only after deterministic comparison evidence was recorded. The primary qualification and replay roots remain for external inspection.

Each marker used exactly the three fields `contract_spec_hash`, `execution_mode`, and `mode_marker_schema_version`, canonical UTF-8 JSON, and one terminating newline.

## Dry run

The no-simulation dry run passed:

- 500 unique integer run IDs from 0 through 499.
- 500 unique run keys.
- 500 unique output directories.
- 100 designs with five realizations each.
- 350 training, 75 validation, and 75 test runs.
- Zero simulations and zero stage artifacts written.

## Qualification coverage

| Design | DOE stratum | Split | Run IDs |
|---|---|---|---|
| `D000` | Pilot anchor | Test | 0–4 |
| `D007` | Transition | Validation | 35–39 |
| `D040` | Global | Train | 200–204 |

All 15 runs executed sequentially in ascending integer run ID. All produced 11 inclusive timesteps and complete satellite, G1, G2, G3, G4, G5, target, inventory, and result artifacts.

Results:

- 15 distinct frozen runs submitted.
- 15 operational generation attempt events.
- 15 successful generations.
- Zero substitutions, collisions, missing artifacts, or target failures.
- Design-level selected station IDs, selection seed, selection hash, ground-design hash, and split remained identical across each design's five realizations.
- Realization differences were limited to frozen run identity, satellite seed, ground-failure seed, sampled failures, and scientific outcomes.
- Five qualification targets were `overall_threshold_breach_any = false`; ten were true.

The generation ledger SHA-256 is `dddf1b6974d44bf2bd387e2040a40c5e1a90273be543814620780f9d8b5ee783`.

## Authoritative replay

All 15 read-only replays passed. Every stage matched the authoritative persisted evidence, expected and recomputed result hashes matched, and every input run-tree before/after byte inventory was identical. The replay ledger SHA-256 is `6ce6f085f0c6a09d9a651348f42fe44b9e3b9eef38d71b490ddd4395183b925e`.

## Deterministic repeat

Run 200, `D040-R00`, was generated once in the primary qualification root and once in a separate clean repeat root. Satellite, G1, G2, G3, G4, G5, target, scientific inventory, and final result bytes matched exactly. The final result hash matched at `58aa2fc374ed839b2f69cb0375832ec6d72866b7948e190eb282c23b81a07e12`.

## Validation evidence

| Check | Result |
|---|---|
| Focused final-generation tests | 26 passed |
| First complete suite | 872 passed, known Windows timer test failed |
| Timer test independent rerun | Failed with elapsed value `0.0`; implementation and test unchanged |
| Unchanged complete-suite rerun | 873 passed |
| No-write AST compilation | 17 files parsed |
| `git diff --check` | Passed |
| Protected tracked diff | Empty |
| Protected status including untracked additions | Empty |
| Ruff | Not installed; not installed for this task |
| Windows `0xc0000139` diagnostic | Not observed |
| Qualification validation | Passed; production acceptance not claimed |

Small tracked evidence summaries are stored in the final integrated dataset generation qualification artifact directory. Large raw qualification and replay run trees were not committed.

## Production readiness

The candidate tooling SHA passed runtime preflight, the 500-record no-simulation dry run, 15 generation runs, 15 authoritative read-only replays, deterministic repeat run 200, target and identity validation, complete tests, and protected-science checks. It is ready for the separate external generation-pipeline audit. It is not externally approved for full production generation by this report.
