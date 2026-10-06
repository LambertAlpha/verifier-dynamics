# E009 — is it coverage within representable regions? (protocol)

Frozen before any E009 audit or run (2026-10-06). Motivated by E008 (post hoc): category-aligned
coverage collapsed exactly the covered category, while random coverage at the same global rate
collapsed nothing.

## Hypothesis
- A false positive is exploited on an input-identifiable region when it is accepted throughout
  that region.
- Fate is set by coverage *within the regions the policy can represent*, not by global coverage
  or difficulty.
- A random (hashed) subset of a region is not representable, so coverage on it acts like partial
  coverage of the region.

## Arms
- f0 = 0.1083, seeds 11–15, pipeline and matching rule as E006.
- New arms: "ends in 0" (M) accepted inside a region; fresh fill elsewhere to keep the initial FPR
  at f0:
  - **aeven:** prompts whose first operand is even (identifiable from one digit; global coverage
    0.5);
  - **sumeven:** prompts whose sum is even (identifiable from the two units digits; 0.5);
  - **hardhalf:** a hashed random half of the three-digit prompts (the category is identifiable,
    but the half within it is not; ≈ 0.25).
- Comparators, re-run with region logging using identical specs: clean, cov50 (random half,
  E006) and covhard (all three-digit, E008). Their final hashes must equal the original runs'.

## Outcomes
- Regional accuracy: in-region and out-of-region sampled dev accuracy, logged at every evaluation
  (mean over the final 4).
- Primary and harm as E006.

## Pre-registered predictions
- **P1 (aeven):** in-region accuracy < 0.2 in ≥ 4/5 seeds. Out-of-region accuracy (seed mean) ≥
  clean's out-of-region accuracy − 0.1.
- **P2 (sumeven):** the same for its region.
- **P3 (hardhalf):** in-region accuracy ≥ 0.3 in ≥ 4/5 seeds. No regional collapse, unlike
  covhard (0.04).
- **P4:** harm(aeven) ≥ 0.25 and harm(sumeven) ≥ 0.25; harm(hardhalf) < 0.2.
- **Integrity:** the clean / cov50 / covhard final hashes equal E006 / E008.

## Rival readings
- **Global coverage drives fate:** aeven and sumeven behave like cov50 (no collapse).
- **Difficulty drives fate:** aeven and sumeven, which mix easy and hard prompts, behave like
  cov50. hardhalf, which is all hard, collapses.
