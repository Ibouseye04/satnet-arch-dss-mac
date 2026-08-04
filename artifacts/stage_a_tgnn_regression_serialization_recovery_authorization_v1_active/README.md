# Stage A TGNN Regression Serialization-Only Recovery Authorization v1

## Verdict

**STAGE A TGNN REGRESSION SERIALIZATION-ONLY RECOVERY AUTHORIZED — READY TO RESTORE AND FINALIZE ONCE**

This package authorizes one targeted recovery operation. It does not execute the
operation. The independent assessment at commit `628b9a1` classified the lost
candidate-1 checkpoint mismatch as `SERIALIZATION_ONLY`; no scientific divergence
was found.

The historical declared exact SHA-256
`89455aeca56ff169f9dbe4d7d8a22ae08339ab60a7e7ef392262ebe11bcabda1` is retained as
historical evidence only. The recovered checkpoint is exactly
`f12b4002201aeee3e6cb1bb347ec4a7480a72eb9891e0f6014e188c4208800b2` and is not
claimed byte-identical to the unavailable historical file. Scientific equivalence
is established by the exact 49-row history and exact stored best metrics.

## Authorized operation

The future one-time command will preserve the current final-seed overwrite as
`superseded_checkpoints/candidate_1_overwritten_by_final_seed_62005.pt`, atomically
replace only `checkpoints/candidate_1.pt` from the reconstruction root, update
`selected_checkpoint_identity.json`, and publish only the explicit allowlist.
All reporting inputs, final-seed checkpoints, RF outputs, TGNN classification
outputs, and all other non-allowlisted files must remain byte-identical.

Bootstrap uses 10,000 occurrence-based design draws with seed `63002`, preserving
five realizations per draw occurrence under IDs `<draw_index>:<design_id>`. The
verified corrected probe is `4f53f6b86b646d91666a45197a535dba797a5154e259f3b990707ddef5be0ee0`.

## Safety boundary

No training, inference, backward pass, optimizer construction or step,
preprocessing refit, prediction generation, test artifact access, checkpoint
serialization, RF modification, or TGNN classification output modification is
permitted. The source guard and deterministic test prove that final seed `62005`
uses `checkpoints/final_seed_62005.pt` and cannot write `candidate_1.pt`.

The recovery root must be absent before first execution, and the completion marker
at its sibling path makes subsequent execution refuse. The command uses the
dedicated TGNN Python executable, verifies the exact activation HEAD and clean
worktree, invokes Python exactly once, and records a timestamped transcript
outside both roots.
