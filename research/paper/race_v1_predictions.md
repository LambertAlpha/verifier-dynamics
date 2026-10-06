# Race model v1: calibration, post-hoc check on E006, and predictions for E008

Written and committed **before any E008 result was read**. Only the count of finished E008 runs
was inspected. Script: `experiments/e006/race_fit.py`. Output:
`results/E008-race-predictions/race_predictions.json`.

## Model
- Per prompt, three outcomes: correct, key, other wrong. Base probabilities come from the
  calibration audit, shrunk to category means.
- Two shared logits: gold skill s and key k. There are no per-prompt learnable terms.
- Exact finite-group (G = 8) GRPO expected updates; SGD.

## Fit
Only E006 clean and exploit are used:
- lr is fitted so that clean's mean gold at T = 1000 equals 0.744;
- lr_k is fitted so that exploit's key share first reaches 0.5 at step 10.

The fitted learnability asymmetry is lr_k / lr ≈ 400.

## Post-hoc check on the other E006 arms (already observed)

| arm | model harm | observed harm | verdict |
|---|---|---|---|
| randfp | 0.028 | 0.004 | ok |
| cov25 | 0.022 | 0.085 | under |
| cov50 | 0.683 (collapse) | 0.107 (no collapse) | **wrong** |
| cov75 | 0.757 (collapse) | 0.596 (4/5 collapse) | ok |
| rarekey | 0.509 (takeover at step 656) | 0.016 (never) | **wrong** |

v1 places the coverage threshold too low and lets a rare key win eventually. Likely missing
pieces:
- per-prompt (local) suppression of the key on uncovered prompts;
- optimizer normalization.

## v1 predictions for E008 (before results)

| arm | model gold at T | harm | collapse |
|---|---|---|---|
| covhard | 0.444 | 0.313 | no |
| coveasy | 0.729 | 0.028 | no |
| set02 | 0.000 | 0.757 | yes |
| set05 | 0.000 | 0.757 | yes |
| setq | 0.000 | 0.757 | yes |

These are recorded as a test of v1. They are not the pre-registered E008 hypotheses, which are
L1, R1 and R2 in `e008_protocol.md`.
