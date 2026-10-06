# E006 — results (2026-10-06)

Protocol: `e006_protocol.md`, frozen at commit 77d13da (Amendment 1 pre-data at ec5c450).

Integrity:
- calibration (ν* = 57, ε = 0.0049), verification (all 8 arms match f0: FPR 0.107–0.118, FNR 0)
  and predictions were all committed before any run;
- 40 runs (8 arms × seeds 11–15, T = 1000) on mac-mini-remote, sequential from a clean tree, no
  STOP;
- the frozen analysis was re-run on a clean tree (`results/E006-analysis/20261006T070848Z_db4710f`).
  An earlier dirty-tree run of the same analysis code was byte-identical and was discarded.

## Primary outcome (mean sampled dev gold accuracy, last 4 evals; clean = 0.744)

| arm | false positives | consistent | shared | base mass of shared FP | Δ vs clean (mean [min, max]) | label | collapsed |
|---|---|---|---|---|---|---|---|
| randfp | fresh coins | no | no | – | −0.004 [−0.022, +0.014] | little difference | 0/5 |
| hashtab | per-prompt table | yes | no | – | −0.018 [−0.024, −0.005] | little difference | 0/5 |
| rarekey | "57" on every prompt | yes | yes | 0.5% of wrong | −0.016 [−0.031, +0.008] | little difference | 0/5 |
| cov25 | M on 25% of prompts | partly | partly | 10.8% of wrong | −0.085 [−0.097, −0.071] | lower | 0/5 |
| cov50 | M on 50% | partly | partly | 10.8% | −0.107 [−0.145, −0.089] | lower | 0/5 |
| cov75 | M on 75% | partly | partly | 10.8% | −0.596 [−0.737, −0.497] | lower | 4/5 |
| exploit | M on 100% | yes | yes | 10.8% | −0.731 [−0.740, −0.707] | lower | 5/5 |

All arms start at the same initial FPR (0.107–0.118), FNR (0) and accuracy (0.388), so the same
Youden J ≈ 0.88.

## Pre-registered hypotheses
- **H1 (randfp ≈ clean): PASS.**
- **H2 (hashtab ≈ clean): PASS.** A consistent per-prompt table does not flip fate. All five seeds
  are slightly negative (−0.018 mean), so there is a small rate cost.
- **H3 (coverage): FAIL as registered.**
  - Monotone: yes. Seed-mean primary 0.740 → 0.660 → 0.638 → 0.149 → 0.014 for
    c = 0, .25, .5, .75, 1.
  - cov75 collapses (4/5) and exploit collapses (5/5).
  - But cov25 was predicted "little difference" and is "lower" (−0.085).
  - Fate flips between c = 0.5 and 0.75. Below that, partial coverage costs 8–11 points without
    collapse.
- **H3b (initial push sign): FAIL.**
  - D̂ predicted the direction for cov50 / cov75 / exploit.
  - It failed for cov25: pooled D̂ −0.005, yet dev wrong-suffix mass rose +0.142 over steps 0→50,
    while clean moved only +0.002.
  - It failed for randfp: D̂ −0.023; observed +0.006, which is noise-level.
- **H4 (reachability): PASS.** rarekey never reached a 0.5 share of wrong responses in any seed
  (final share 0.001–0.013). exploit reached FPR ≥ 0.5 at steps 7–12.
- **H5 (RME over the 10-verifier panel): FAIL.**
  - Spearman(RME, harm) = 0.600 vs static FPR 0.636 and −J 0.648.
  - Flip and deleted carry rate-type harm (0.049, 0.033) with RME ≈ 0 and higher FPR.
- **H5b (RME over the 8 matched-FPR arms; Amendment 1, pre-data): PASS**, Spearman 0.833.
  - FPR and J are constant across these arms by construction.
  - Misorderings: rarekey (RME 0.0053, harm 0.016) above cov25 (0.0040, 0.085), and hashtab
    (−7e-5, 0.018) above randfp (−9e-5, 0.004).

## Mechanisms
- **Collapse is absorbing, as Prop. 4 says.** Mixed groups under V in the last 50 steps:
  - exploit 0.004 and cov75 0.08;
  - clean 0.29, randfp 0.39, cov25 0.35, cov50 0.32.
- **Share of wrong responses in M:**
  - first → last 50 steps: clean 0.13 → 0.18; cov25 0.29 → 0.60; cov50 0.48 → 0.69; cov75 0.61
    → 0.99; exploit 0.72 → 1.00;
  - batch FPR ≥ 0.5 at steps 7–12 (exploit), 15–23 (cov75), 34–83 (cov50), 601–921 (cov25),
    never (randfp, rarekey).

## Post-hoc (not pre-registered) examination of H3b
Initial push on M by task category, on the calibration audit
(`results/E006-posthoc-dhat-category/`).
- **cov25:** pooled −0.0047, but **three-digit +0.0043** (no carry −0.0115, units carry −0.0171).
- **randfp:** negative in every category.
- M is most reachable in three-digit sums (M mass 7.6% vs 5.0% for no carry). Where the push was
  positive it is consistent with M growing first in that subpopulation and spreading through
  shared parameters.
- The global mean-field (exchangeable prompts) assumption of Derivation 5 is too coarse. A
  *local* (subpopulation) version of the drift is the natural refinement. Because this is post
  hoc, it needs a new pre-registered test (E008).

## What E006 establishes
- At a matched initial FPR / FNR / accuracy (hence J), fate depends on the **cross-prompt
  structure** and **reachability** of the false positives:
  - random: benign;
  - consistent per-prompt: benign (small rate cost);
  - shared but unreachable: benign;
  - shared and reachable: harm grows with coverage and collapses above c ≈ 0.5–0.75.
- This reconciles Leaky-suite-style benign persistent FPs with master-key collapse, and it is
  outside the J-only account.

Not established:
- the location of the threshold from first principles (the mean-field prediction failed for
  c = 0.25);
- a diagnostic that also covers rate-type harm (H5 failed);
- anything at LLM scale (E007).
