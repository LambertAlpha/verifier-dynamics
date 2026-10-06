# E008 — results (2026-10-06)

Protocol `e008_protocol.md`, frozen at a4fecf6. Integrity:
- arms calibrated and verified before runs (FPR 0.104–0.116);
- the L2 pre-run check was committed before runs (it FAILED);
- race-model v1 predictions were committed before results (c6a95b0);
- 25 runs, no STOP;
- the E006 clean-s11 rerun with the extended verifier module is bit-identical;
- the frozen analysis was re-run on a clean tree (`results/E008-analysis/20261006T080946Z_ca75417`;
  the earlier dirty-tree run was byte-identical and discarded).

## Primary outcomes (seed means; clean 0.744; comparators from E006 at the same seeds)

| arm | accepted wrong answers | global coverage | primary | harm | collapsed | dev by category (no carry / units carry / three-digit) |
|---|---|---|---|---|---|---|
| cov50 (E006) | M on a random half of prompts | 0.5 | 0.638 | 0.107 | 0/5 | 0.75 / 0.78 / 0.52 |
| covhard | M on three-digit prompts | 0.5 | 0.412 | 0.333 | 0/5 | 0.79 / 0.79 / **0.04** |
| coveasy | M on no-carry and units-carry prompts | 0.5 | 0.342 | 0.403 | 0/5 | **0.01 / 0.05** / 0.66 |
| rarekey (E006) | "57" everywhere (0.5% of wrong) | 1 | 0.729 | 0.016 | 0/5 | 0.77 / 0.79 / 0.68 |
| set02 | "111" everywhere (1.9%) | 1 | 0.012 | 0.732 | 5/5 | 0.00 / 0.00 / 0.02 |
| set05 | 3 values (5.4%) | 1 | 0.011 | 0.733 | 5/5 | 0.00 / 0.00 / 0.02 |
| setq | 7 values (11.5%) | 1 | 0.012 | 0.733 | 5/5 | 0.00 / 0.00 / 0.02 |
| exploit (E006) | M everywhere (10.8%) | 1 | 0.014 | 0.731 | 5/5 | 0.00 / 0.01 / 0.02 |

Collapsed policies emit a single constant: 111 (set02, set05), 113 (setq), 110 / 100 (exploit).
Under covhard the dev outputs concentrate on 100 / 110 for three-digit items; under coveasy, on
50 / 70 for two-digit items.

## Pre-registered tests
- **L1 (difficulty-local): FAIL.**
  - Predicted per-seed order: covhard < cov50 < coveasy.
  - Observed: coveasy is the *most* harmful (0.403), then covhard (0.333), then cov50 (0.107), in
    every seed.
- **L2 (pre-run push): FAIL**, recorded before runs. The prediction was mis-reasoned: inside
  covered categories the local coverage is 1, so the push is positive whatever the difficulty.
- **R1 (reachability dose): FAIL as registered.**
  - Harm sequence rarekey 0.016 → set02 0.732 → set05 0.733 → setq 0.733.
  - "Non-decreasing" is violated by 0.0004 between set05 and setq, at the ceiling where all three
    collapse. The setq-collapse part holds (5/5).
- **R2 (rule form beyond mass and coverage): PASS.** |harm(setq) − harm(exploit)| = 0.002.
  Arbitrary frequent wrong values act like "ends in 0".

## Race model v1 predictions (committed before results)

| arm | predicted harm | observed |
|---|---|---|
| covhard | 0.31 | **0.33** ✓ |
| coveasy | 0.03 | 0.40 ✗ |
| set02 / set05 / setq | collapse | collapse ✓✓✓ |

v1 has one key logit for all prompts, so it cannot represent a region-specific key. It misses
coveasy.

## What E008 shows (post hoc; to be tested in E009)
1. **The collapse is confined to the covered region, and only when the policy can identify that
   region from the input.**
   - Category-aligned coverage (whether a + b ≥ 100) collapsed exactly the covered categories and
     left the rest at clean levels.
   - At the same global coverage, *random* (hashed) coverage did not collapse any region.
   - With category coverage the policy learned a conditional key ("three-digit sum → 100 / 110").
     With hashed coverage, uncovered prompts that the policy cannot tell apart keep penalizing the
     key.
   - So the relevant quantity is **coverage within the regions the policy can represent**, not
     global coverage and not difficulty (L1's hypothesis).
2. **Reachability has a low threshold.** A single constant with 1.9% of the base wrong mass
   collapses training. One with 0.5% did not, within T = 1000.
3. **Rule form does not matter beyond mass and coverage** (R2).
4. **Practical reading.** A verifier bug that applies to a recognizable subset of problems is a
   *regional* master key. Real bugs often look like this, e.g. a numeric tolerance that only
   misfires for large answers (Where the Verifier Fails). Such a bug can collapse that subset
   while aggregate accuracy and FPR look moderate.
