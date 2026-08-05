# Stage A TGNN Regression Training v1 Acceptance Gate

Verdict: **STAGE A TGNN REGRESSION TRAINING V1 ACCEPTED — ALL FOUR STAGE A MODELS FROZEN — READY FOR SEALED TEST AUTHORIZATION**

This committed evidence is a narrow independent, read-only completed-training gate. It loaded frozen checkpoints on CPU and recomputed validation inference only. It did not train, call `fit()`, create an optimizer, call `backward()`, call `optimizer.step()`, refit preprocessing, regenerate the frozen prediction artifact, rerun training/bootstrap as training, access test indexes or targets, write checkpoints, or modify governed output roots.

- Final public tree: 22 files / 2,070,248 bytes / `95134dab5669ee4bc9515777d19f48449186324977d17981be856fde533e6111`.
- Candidate 1 selected independently from four candidates; history has 49 rows, best epoch 36, stopping epoch 48, patience 12, strict RMSE improvement.
- Five final CPU checkpoints reproduce the frozen 75-record validation ensemble within `1e-6`; run-level, design-level, and 10,000-replicate bootstrap reports reproduce.
- Recovery is serialization-only; the reconstructed selected checkpoint is byte-identical, the superseded checkpoint is preserved, and scientific results did not change.
- Test seal: zero test indexes/targets opened; zero training epochs, optimizer steps, backward calls, checkpoint writes, preprocessing fits, and reporting modifications.

See the adjacent JSON evidence files for exact identities and recomputation values.
