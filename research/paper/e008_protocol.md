# E008 — where and how much: difficulty-local coverage and the reachability dose (protocol)

Frozen before any E008 audit or run (2026-10-06). It follows E006 (`e006_results.md`), whose
failed H3 and H3b, and the post-hoc category analysis, motivate a *local* version of
Derivation 5.

## Refined (local) drift
- Derivation 5 makes the drift on the shared false positive M proportional to
  `c (1 − v) + (1 − c)(r − v)` with a single pooled v.
- Per prompt, the sign is positive when `v_x < c + (1 − c) r`. On *hard* prompts (low gold
  accuracy v_x), a master key gains even at low coverage. The positive push concentrates there and
  spreads through shared parameters.
- E006's three-digit prompts (base accuracy 0.24) had a positive push under cov25, although the
  pooled push was negative.
- Base calibration audit (structure only, no outcomes): within every category M is 10–12% of the
  wrong answers. Three-digit holds 61% of all wrong answers.

## Arms
- Pipeline, f0 = 0.1083, matching rule (|FPR − f0| ≤ 0.015 on the E005b-0 verification audit;
  failures are not trained), seeds 11–15 and analysis conventions are all those of E006.
- Comparators are E006's clean / cov50 / exploit / rarekey at the same seeds (same code and
  deterministic). Integrity check: re-run E006 clean-s11 and require an identical final hash.

| arm | accepted wrong responses | global coverage | difficulty of covered prompts |
|---|---|---|---|
| covhard | M on every three-digit prompt; fill elsewhere | ≈ 0.50 | hard (base acc 0.24) |
| coveasy | M on every no-carry or units-carry prompt; fill elsewhere | ≈ 0.50 | easy (0.57 / 0.45) |
| set02 | the most frequent non-M wrong value (111), on every prompt; fill | 1 | – |
| set05 | the smallest top-k set of non-M wrong values with share ≥ 0.05; fill | 1 | – |
| setq | the smallest top-k set of non-M wrong values with share ≥ q = 0.1083; no fill beyond f0 | 1 | – |

Top-k sets are ranked by base wrong-share on the calibration audit (ties: smaller value), and
frozen in `configs/e008/arms.json` before verification.

## Pre-registered predictions
- **L1 (difficulty-local).** In every one of the 5 seeds, primary(covhard) < primary(cov50, E006)
  < primary(coveasy). Seed-mean harm (−Δ vs clean) of covhard ≥ 0.2.
- **L2.** The pre-run initial push on M is positive in the three-digit category for covhard, and
  non-positive in every category for coveasy. Checked before training. If it fails, it is
  reported, and L1 still runs.
- **R1 (reachability dose).**
  - Seed-mean harm is non-decreasing in shared-set mass: rarekey (E006, 0.005) ≤ set02 ≤ set05 ≤
    setq.
  - setq collapses (primary < 0.2) in ≥ 4 of 5 seeds.
- **R2 (rule form beyond mass and coverage).** |harm(setq) − harm(exploit, E006)| ≤ 0.15.
  - setq accepts arbitrary frequent wrong values instead of "ends in 0", at the same mass and
    full coverage.
  - Rival: if the simple "ends in 0" rule mattered beyond mass and coverage, setq would be much
    less harmful.

## Outcomes and mechanism logs
As E006: primary, paired differences, labels, collapse, batch FPR, share of wrong responses in
the accepted set, mixed groups, dev modal answer.

## Not claimed
- That difficulty is the only local factor.
- LLM scale.
