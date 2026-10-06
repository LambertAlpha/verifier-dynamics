# E007c — small-LLM validation with the stop-token fix (protocol)

Frozen before any E007c pilot, audit or run (2026-10-06). It replaces E007b (aborted).

## Why E007b was aborted
- Clean RL degraded the policy: greedy accuracy on 200 test questions fell from 0.485 to 0.175 by
  step 25.
- Diagnosis:
  1. `mlx_lm.batch_generate` drops the stop token, so the decision to end a response was never
     trained. E007 had the same bug.
  2. Updates were noisy: Adam lr 1e-6 on 64 samples per step.
- Generation and training log-probabilities were verified to agree exactly.

## Changes from E007b
Everything else is as in `e007b_protocol.md`: the 768-token cap, gold, arms, the f0′ rule,
tolerance 0.03, predictions E1–E5, the Amendment 1 probe rule and the clean-health gate.
1. **Stop-token fix.** Completions used for training include the stop token
   (`vdyn.e007.generation`, tested).
2. **Learning-rate pilot (clean arm only, seed 99).**
   - Run 50 steps at each lr in {1e-6, 3e-7, 1e-7}, with greedy evaluation on the 200-question
     subset at steps 0, 25, 50.
   - **Choose the largest lr whose step-50 accuracy is ≥ its step-0 accuracy − 0.02.** If none
     qualifies, choose 1e-7 and label E007c "clean RL weak".
   - The chosen lr is written to `configs/e007c/e007c.toml` and committed before calibration.
   - The pilot sees no flawed-verifier arm.
3. **Seeds** 41, 42, 43.
4. **Fresh calibration and verification** with the fixed generation.
