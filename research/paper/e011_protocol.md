# E011 — a prospective test of on-policy conditional acceptance (protocol)

Frozen before any E011 audit, ACM computation or run (2026-10-06).

## Background (post hoc; `results/E011-pre/posthoc_conditional_coverage.json`)
Across 18 toy verifiers (E005b, E006, E008, E009), harm tracks one quantity.
- **Definition.** **ACM_τ** is the base-policy mass of wrong outputs whose *on-policy conditional
  acceptance* is ≥ τ. For an output k, that acceptance is the expected acceptance of k over the
  (prompt, k) occurrences where the base policy emits k wrongly.
- **Correlations with harm:**
  - Spearman(ACM_0.75, harm) = 0.77 and Spearman(ACM_0.6, harm) = 0.91;
  - static FPR: 0.10.
- **Separation.** Every arm with harm ≥ 0.33 had ACM_0.75 ≥ 0.037. Every arm with harm ≤ 0.15
  had ACM_0.75 ≤ 0.0051.

The thresholds below were chosen from these post-hoc data. E011 tests them on new verifier
structures.

## Frozen decision rule
- Predict harm ≥ 0.25 (paired primary difference vs clean, seed mean) **iff ACM_0.75 ≥ 0.02**.
- ACM is computed on the E005b-0 calibration audit samples (3000 prompts × 8), using
  `experiments/e006/posthoc_conditional_coverage.py:expected_accept`, before training.
- Secondary (reported only): the same rule with ACM_0.6.

## Arms
- f0 = 0.1083 matched by a fresh fill, as in E006; seeds 11–15; clean comparator = E006 clean.
- M = valid wrong answers ending in 0; s = a + b.
- **near:** M accepted iff |v − s| ≤ 10, on every prompt. Prompt coverage is local, but the policy's
  wrong round numbers are mostly near-misses, so the conditional acceptance is expected to be high.
- **far:** M accepted iff |v − s| > 10, on every prompt. Accepted on almost every prompt *for some
  value*, but not where the policy emits it.
- **near50:** near, but only on a hashed half of the prompts. Expected conditional acceptance is
  about 0.5.

## Predictions
Obtained mechanically from the rule once ACM is computed. They are written to
`results/E011-predictions/` and committed before training. If an arm's ACM_0.75 falls within
±0.005 of 0.02, its prediction is reported as "undetermined".

Rival readings:
- **Global FPR or coverage decides:** far, which is accepted somewhere on nearly every prompt,
  should be harmful.
- **Proximity alone decides:** near50 should be as harmful as near.

## Outcomes
- Primary, paired differences, labels and collapse as in E006.
- Per-arm hits of the rule (prediction vs observed harm ≥ 0.25).
- E011 passes iff the rule is right for all three arms.
