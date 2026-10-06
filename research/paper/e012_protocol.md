# E012 — a prospective test of short policy-conditioned probes (protocol)

Frozen before any E012 audit, probe or run (2026-10-06).

## Background (post hoc; `results/E012-pre/`)
- Static diagnostics each failed prospectively in part:
  - RME on the 10-verifier panel (E006 H5);
  - ACM on the far arm (E011).
- In the first 150 training steps of all 21 toy verifiers so far, the **rise in batch FPR**
  (steps 141–150 minus steps 1–10, seed mean) separates harm ≥ 0.25 (rise ≥ 0.30) from harm < 0.25
  (rise ≤ 0.21). Spearman with harm 0.94; static FPR 0.10.
- Deletion and flip verifiers have high static FPR but zero rise.

## Frozen rule
- Predict harm ≥ 0.25 (seed-mean paired difference vs clean) **iff rise_150 ≥ 0.25**.
- rise_150 is measured on **separate probe runs** (T = 150, same seeds and streams as the full
  runs). The predictions are committed before any full run.
- Arms with rise_150 within ±0.03 of 0.25 are reported as "undetermined" and not scored.

## Arms
- E006 pipeline, seeds 11–15, clean comparator = E006 clean (same seeds). Static FPRs are
  deliberately *not* matched.

| arm | rule | static FPR |
|---|---|---|
| randfp30 | fresh false positives at 0.30 | high |
| del50 | every response accepted on a hashed half of prompts (signal deletion) | high |
| cov65 | M on a random 65% of prompts, fill to f0 | ≈ f0 |
| far40 | M accepted iff \|v − s\| > 40, fill to f0 | ≈ f0 |
| key111half | the value 111 accepted on a hashed half of prompts, fill to f0 | ≈ f0 |
| hard75 | M on a hashed 75% of three-digit prompts, fill to f0 | ≈ f0 |

## Pass condition
The rule is right for every scored arm.

Also reported, with no pass role:
- static FPR / FNR / J (verification audit);
- ACM_0.75 and RME of each arm, with the E011 / E006 rules' predictions;
- integrity: each full run's logged gold / verifier rewards for steps 1–150 equal its probe run's.
