# Task plan — verifier-dynamics toward a top-venue paper

**Goal (user, 2026-10-05):** keep researching and experimenting until the direction and thesis reach
top-conference quality; broad autonomy granted. Constraints that still bind: no paid cloud, no
credential changes, pre-register before running, keep failures recorded, fail-closed checks, exact
paths, dispositions for every run, do not disturb the user's other processes on the mac-mini
(Trader Workstation runs there).

**Compute:** local M3 (8 cores, 24 GB); mac-mini-remote M4 Pro (12 cores, 24 GB, MPS). No GPU cluster.

## Working thesis (to be sharpened in Phase 1)
The harm a verifier does in RLVR is set by the *exploitability* of its false positives (whether the
policy can reach them and whether one behaviour satisfies them across prompts), not by its static
error rates. Random false positives at the same rate are close to harmless.

## Phases
| # | Phase | Status |
|---|---|---|
| 0 | Literature positioning (2024–2026 RLVR verifier noise / reward hacking / spurious rewards) | complete (findings.md) |
| 1 | Thesis + claims + evidence map; decide what is novel | complete (thesis.md) |
| 2 | Theory: group-normalized PG under affine noise vs exploitable FP; propositions + numerical checks | in_progress |
| 3 | E006 disentangling factorial on the toy Transformer (consistency x input-dependence x reachability, matched initial FPR) | pending |
| 4 | Cheap exploitability diagnostic evaluated on a verifier panel (toy) | pending |
| 5 | Small-LLM validation (0.5B-class, GRPO, real math checkers with real loopholes) | pending |
| 6 | Paper draft + figures; honest limitations | pending |

## Decisions
| Date | Decision | Why |
|---|---|---|
| 2026-10-05 | Pivot the paper's thesis from 'geometry diagnostics' to 'learnability / cross-prompt structure of FPs decides fate' | Our geometry line was negative (E002/E004a/E005a-R). E005b-0 plus literature gaps (Rate-or-Fate J>0 vs V3 collapse; Leaky vs master keys) give a novel, testable mechanism |
| 2026-10-05 | Toy experiments first (cheap, ~1 min/run), LLM validation in parallel on the mini's MPS | Compute limits; the toy gives clean factorial control |

## Errors encountered
| Error | Attempt | Resolution |
|---|---|---|
