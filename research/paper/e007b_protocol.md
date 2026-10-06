# E007b — corrected small-LLM validation (protocol)

Frozen before any E007b audit or run (2026-10-06). It replaces E007, which was aborted for a
design flaw.

## Why E007 was aborted
The decision used only base-audit statistics and the clean arm's trajectory, before any
flawed-verifier arm ran.
- **The cap was too short.** With a 320-token cap, 43.5% of base completions were truncated.
  Truncated completions are correct 12% of the time, against 61.5% for complete ones, and 64% of
  all wrong answers were truncations. The reward was dominated by "finish within 320 tokens".
- **The clean arm degraded.** Within 20 steps, clean (seed 21) shortened its completions from
  271 to ~120 tokens and stopped boxing. Its greedy accuracy on 200 test questions fell from 0.42
  to 0.30.
- E007's frozen artefacts (calibration, verification, RME) and the clean-s21 run are preserved
  with a disposition. None is used as E007b evidence.

## Changes from E007
Everything else is as in `e007_protocol.md`.
1. **Generation cap.** A length pilot on GSM8K train 5800–5831 × 8 base samples, capped at 1024,
   picks the smallest cap in {512, 768, 1024} with a truncation rate < 3%. The cap is written to
   `configs/e007b/e007b.toml` and committed before calibration. It applies to training,
   evaluation and audits.
2. **Seeds** 31, 32, 33.
3. **Fresh calibration and verification** with the same prompts, seeds and rule (f0′ = the larger
   natural FPR of ends0 and anywhere; tolerance 0.03). Longer completions change the FPRs.
4. **Clean-health gate (reported).** If clean's greedy accuracy on the 200-question subset falls
   by more than 0.05 from step 0 to step 150 (seed mean), the result is labelled "clean RL
   unhealthy" and no verifier-effect claim is made from E007b.

## Predictions (unchanged from E007)
- E1 randfp ≈ clean.
- E2 hashtab ≈ clean.
- E3 ends0 "lower", with the share of ends-in-0 answers rising.
- E4 anywhere "lower", with numbers per response rising.
- E5 RME(ends0) is the largest.

The rules are in `experiments/e007/e007_analysis.py`, run with `VDYN_E007_EXP=e007b`.
