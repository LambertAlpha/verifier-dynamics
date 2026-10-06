# E012 — results (2026-10-06): the short-probe rule, 5 of 6 prospectively

Protocol `e012_protocol.md`, frozen at 8cf68a4. Integrity:
- probe predictions committed before the full runs (d4bbe35);
- race-model v2 predictions committed before any full-run result was read (287d8de);
- 30 probe + 30 full runs, no STOP;
- **each full run's first 150 steps are identical to its probe run** (determinism);
- the frozen analysis ran in a clean worktree (`results/E012-analysis/20261006T111021Z_32867fc`).

| arm | static FPR | probe rise_150 | probe rule | harm (seed mean) | collapsed | rule right | race v2 harm (right?) |
|---|---|---|---|---|---|---|---|
| randfp30 | **0.299** | −0.007 | < 0.25 | 0.019 | 0/5 | ✓ | 0.092 ✓ |
| del50 | **0.504** | −0.010 | < 0.25 | 0.063 | 0/5 | ✓ | 0.168 ✓ |
| cov65 | 0.115 | +0.343 | ≥ 0.25 | **0.230** | 0/5 | ✗ (borderline) | 0.739 ✗ |
| far40 | 0.108 | +0.354 | ≥ 0.25 | 0.727 | 5/5 | ✓ | 0.747 ✓ |
| key111half | 0.112 | +0.191 | < 0.25 | 0.109 | 0/5 | ✓ | 0.543 ✗ |
| hard75 | 0.110 | +0.387 | ≥ 0.25 | 0.336 | 0/5 | ✓ | 0.364 ✓ |

## Verdicts
- **FAIL as registered.** The rule is right for 5 of 6 arms. The miss, cov65, has harm 0.230 against
  the 0.25 line while its rise (0.343) was well above threshold. The probe overestimates arms that
  degrade without collapsing.
- Spearman with harm on the six new arms:
  - probe rise: **+0.886**;
  - static FPR: **−0.886**. The two verifiers with the highest static FPR were the most benign.
- Race v2 (theory-based, fitted on two E006 arms only): 4 of 6.

## Across all experiments
- The 150-step probe classifies 26 of 27 verifiers correctly: 21 post hoc (E005b–E011) plus 5 of
  6 prospectively.
- Static FPR does not track harm (Spearman 0.10 post hoc, −0.89 prospectively).
