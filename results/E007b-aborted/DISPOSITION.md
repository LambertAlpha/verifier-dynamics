E007b was aborted during its first run (clean, seed 31, at about step 50) because clean RL itself
degraded the policy: greedy accuracy on 200 test questions fell from 0.485 at step 0 to 0.175 at
step 25, the boxed rate fell from 0.97 to 0.03, and greedy outputs grew longer.

Diagnosis (2026-10-06):
1. **Bug.** `mlx_lm.batch_generate` drops the stop token (EOS) from the returned completions, so
   training never put gradient on ending a response. E007 (aborted earlier) had the same bug.
2. **Likely contributor.** Updates were very noisy for this regime: Adam at lr 1e-6 moves every
   parameter about 1e-6 per step, from only 64 samples per step.

The alignment between generation and training log-probabilities was verified to be exact, so
there is no positional misalignment.

No flawed-verifier arm ran. This run is the record of the failure, not evidence. The corrected
rerun is E007c: EOS included, and the learning rate chosen by a pre-specified clean-only pilot rule.
