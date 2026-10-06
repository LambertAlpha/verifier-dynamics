# E009 — results (2026-10-06)

Protocol `e009_protocol.md`, frozen at 39fe302. Integrity:
- arms verified (FPR 0.109–0.119); an earlier verification stopped on a copied-script bug (the
  clean arm was not exempt), preserved with its disposition;
- 30 runs, no STOP;
- the clean / cov50 / covhard re-runs are **hash-identical** to E006 / E008;
- the frozen analysis ran in a clean worktree at f1d924b (`results/E009-analysis/20261006T091106Z_f1d924b`).

## Outcomes (seed means; harm vs clean)

| arm | covered region | global coverage | harm | covered-region accuracy | rest |
|---|---|---|---|---|---|
| cov50 | random half | 0.5 | 0.107 | – | – |
| hardhalf | random half of three-digit | 0.25 | 0.095 | 0.512 | 0.698 |
| aeven | first operand even | 0.5 | 0.149 | 0.567 | 0.622 |
| sumeven | sum even | 0.5 | 0.121 | 0.595 | 0.654 |
| covhard | all three-digit | 0.5 | 0.333 | **0.031** | 0.547 |

Clean's accuracy: about 0.74 out of region, about 0.71 on three-digit.

## Pre-registered tests
- **P1 (aeven region collapses): FAIL.** In-region 0.55–0.58 in every seed.
- **P2 (sumeven region collapses): FAIL.** In-region 0.53–0.65.
- **P3 (hardhalf does not collapse): PASS.** In-region 0.50–0.53.
- **P4: FAIL.** aeven 0.149 and sumeven 0.121, against the predicted ≥ 0.25. hardhalf 0.095 < 0.2 holds.
- **Integrity: PASS.**

## Reading
- "Identifiable from the input" is **not** sufficient. Parity regions are readable from one or two
  digits, yet they behave like random coverage at the same rate.
- The E008 post-hoc hypothesis is falsified as stated.
- What differed in covhard / coveasy (post hoc, to be tested):
  - The *values* the policy emits as false positives are region-specific: 100 / 110 on
    three-digit sums, 50 / 70 on two-digit sums.
  - Under category coverage, each emitted wrong value is accepted on essentially all prompts
    where the policy emits it.
  - Under parity or random coverage, the same value (e.g. 110) is emitted on covered and uncovered
    prompts alike, so it is rewarded on some and penalized on others.
  - Candidate quantity: **on-policy conditional coverage of each wrong output**, i.e. the
    acceptance rate of response k over the prompts on which the policy produces k, weighted by
    π(k | x).
