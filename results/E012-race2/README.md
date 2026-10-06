Race model v2 (src/vdyn/e006/race2.py; experiments/e006/race2_fit.py).
- **Model.** Output-level biases (Prop. 5) for all 1000 answer values plus a shared gold skill.
  Base output distributions are exact, from the base model, on 600 calibration-audit prompts.
- **Fit.** E006 clean and exploit only: lr_s = 0.004977, lr_b = 17.48.
- **Timing.** Written and committed BEFORE any E012 full-run result was read. Only the count of
  finished E012 runs and the committed probe predictions had been seen.
- **Post-hoc check (observed arms, threshold 0.25).** 18 of 21 classified correctly, including
  far (0.735 vs observed 0.744). Wrong: rarekey (predicts collapse), aeven and sumeven (0.34 vs
  0.15 / 0.12).
- **E012 predictions (harm).**

  | arm | harm | prediction |
  |---|---|---|
  | randfp30 | 0.092 | benign |
  | del50 | 0.168 | benign |
  | cov65 | 0.739 | harm ≥ 0.25 |
  | far40 | 0.747 | harm ≥ 0.25 (slow, half-FPR at step 256) |
  | key111half | 0.543 | harm ≥ 0.25 — **disagrees with the probe rule** |
  | hard75 | 0.364 | harm ≥ 0.25 |
