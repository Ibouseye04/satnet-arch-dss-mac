# Final Integrated Dataset Generation Corrected Qualification

## Status

`READY FOR EXTERNAL GENERATION-PIPELINE RE-AUDIT`

The full 500-run production generation, full production replay, and production acceptance against a real 500-run dataset were not executed.

## Provenance

| Item | SHA |
|---|---|
| Starting handoff HEAD | `51d8cef40b7604c436e5bd6f2b3aae66bd839ab2` |
| Superseded tooling | `0e6fbdb89f863ec9ef11fe0271bbcf474ff102f2` |
| Frozen-byte commit | `4477f89a38a0952cb45d776a461fb1d40d522615` |
| Corrected control-plane commit | `0208e69376c9a9b497521e7237935c384d7e2f4d` |
| Corrected control tests | `b9f7836271fe6990b49ac7e76d28414f5e5ad6e7` |
| Retry/resume hardening | `e15ddf3ba8231e991e91b9643e7b7ea0513956d8` |
| Candidate and validated corrected tooling | `9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab` |

The superseded tooling SHA is not approved for production.

## Corrected controls

- Frozen contract artifacts and the authoritative pilot catalog are checkout-binary through narrowly scoped `.gitattributes` rules.
- Generation and replay acceptance derives counts and identities from canonical record-level evidence.
- Production generation, replay, and validation have separate guarded CLI commands.
- `production_replay` is an explicit canonical execution mode.
- Output-root validation protects every worktree and Git common storage in the repository family.
- Retries require exact canonical attempt-input identity equality before simulation.
- Verified resume performs full authoritative stage validation and requires a matching replay certificate from the correct replay mode.
- The generation-phase isolation allowlist remains explicit and rejects unauthorized paths.

## Fresh Windows checkout

A fresh detached worktree at the corrected tooling SHA was created with effective `core.autocrlf=true`.

Results:

- All eleven frozen contract working-tree files matched their frozen-tag Git blobs byte-for-byte.
- `pilot_catalog.csv` matched raw SHA-256 `e8855d4ded4c242f3e5b35b90610d5f9c4f218bdf717d2d08b2464cac39f1598`.
- Runtime preflight passed at exact HEAD `9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab`.
- Physics, link-budget, satellite, and G1–G5 version validation passed.
- The 500-record no-simulation dry run passed with 100 designs, five realizations per design, and split counts 350/75/75.
- Repository-family tests rejected the original checkout, frozen-contract directory, data directory, preserved paths, protected source, detached worktree, and Git common directory.

## Acceptance-control qualification

Corrected focused tests passed: `61 passed`.

The controls include forged aggregate counts with zero records, missing/extra/duplicate identities, wrong hashes/seeds/splits, replay mismatch, input mutation, missing stages, aggregate disagreement, production mode pairing, atomic acceptance-report output, retry identity substitution, authoritative stage corruption, and replay-certificate enforcement.

## Complete validation

- Complete unchanged suite: `909 passed`.
- An earlier intermediate-candidate suite produced only the known Windows timer flake; its isolated rerun passed. The final candidate suite passed without failures.
- No-write AST parsing: 20 corrected source/test files parsed.
- `git diff --check`: passed.
- Frozen-contract-to-candidate protected tracked diff: empty.
- Protected-path untracked status: empty.
- Windows diagnostic `0xc0000139`: not observed as a process failure.

## Fresh corrected qualification

External roots:

| Role | Root |
|---|---|
| Fresh generation | `C:\Users\johns\satnet-final-corrected-requalification-v3-20260720` |
| Fresh replay | `C:\Users\johns\satnet-final-corrected-replay-v3-20260720` |
| Fresh deterministic repeat | `C:\Users\johns\satnet-final-corrected-repeat-v3-20260720` |
| Fresh detached validation worktree | `C:\Users\johns\satnet-final-corrected-validation-v3-20260720` |

Results:

- 15/15 exact frozen qualification runs generated successfully.
- 15 distinct frozen submissions, 15 attempt events, and 15 successful publications were derived from record evidence.
- Generation ledger SHA-256: `c1ba7d5e4cb9209b65d22b787305eaa6cc530afdc56996122958eddb8dd7d611`.
- 15/15 authoritative replays succeeded with all nine stages matched and unchanged input trees.
- Replay ledger SHA-256: `7a14220b8f9d2075445b47235939fd128a04d72da632fb823ee42a0239ee18c0`.
- Record-derived qualification validation passed and did not claim production acceptance.
- Run 200 reproduced all scientific artifact bytes and result hash `58aa2fc374ed839b2f69cb0375832ec6d72866b7948e190eb282c23b81a07e12`.
- Verified resume for run 0 passed full authoritative validation with an unchanged source-tree inventory hash.

## Stop condition

No production simulations, production replay, real production acceptance, RF work, TGNN work, tag, or push was performed.
