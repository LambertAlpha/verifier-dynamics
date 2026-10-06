# Thesis (working, 2026-10-05)

## One line
RLVR selects among reward-satisfying behaviours by **learnability**, not by correctness. A verifier's
false positives flip the training outcome ("fate") only when they form a behaviour that the policy
can reach and that its shared parameters learn faster than the gold behaviour. The canonical case
is a false positive accepted across many prompts (a "master key"). Static error rates, and the
Youden index J = TPR − FPR that governs fate in tabular analyses, do not capture this.

## Why this is not already known (positioning; details in findings.md)
- *Rate or Fate?* (2601.04411): in a per-prompt bandit, fate is set by J. Our V3 has J ≈ 0.88 > 0 at
  initialization and still collapses (dev accuracy 0.38 → 0.01). The bandit has no parameter sharing.
- *When the Reward Suite Is Leaky* (2607.11022): natural per-task, persistent false positives in code
  suites are nearly harmless ("selection, not exploitation").
- *One Token to Fool* (2507.08794): prompt-agnostic master keys exist in LLM judges, and an RL run
  collapsed onto them. This is observational, with no matched contrast.
- Fuzzing (2606.01066) and rewardlint find false positives statically. They do not predict which
  ones RL will exploit.
- No work we found contrasts false-positive *structures* at a matched rate, gives a mechanism that
  reconciles "persistent FPs are benign" with "master keys collapse", or offers a gold-free
  pre-RL test for exploitability.

## Mechanism (to formalize in theory.md)
1. **Affine noise is fate-neutral.** With fresh coins at rate f, E[V | y] = f + (1 − f) G. Per
   prompt, the expected group-normalized update is a positive rescaling of the clean one
   ("rate, not fate"; consistent with J > 0).
2. **Per-prompt consistent false positives** compete with gold only within their own prompt.
   Without cross-prompt coupling, a false positive gains on gold only if its current mass already
   exceeds gold's ("selection of pre-existing modes").
3. **Shared false positives** receive gradient from every covered prompt through the shared
   parameters. Their logit drift aggregates `coverage × mass × advantage` across prompts, while
   prompt-specific gold answers aggregate only through the gold skill's own generalization. On
   uncovered prompts the same behaviour is penalized. So drift changes sign at a critical coverage
   c*, set by the mean verifier reward. Prediction: a fate transition in coverage at fixed FPR and
   fixed J.
4. **Absorption.** Once the master key dominates a covered prompt, its groups are all-1, the
   variance is zero and the advantage is zero. Gold receives no further signal there, so the
   collapse is absorbing (observed: mixed groups → 0.003–0.006).

## Testable predictions (to pre-register in E006, with numbers from the theory simulator)
- P1 Random FP (fresh) ≈ clean. Replication of E005b-0 at new seeds.
- P2 A consistent per-prompt FP table (hashed) at the same FPR is benign: ≈ clean, small loss at most.
- P3 A shared FP accepted on a fraction c of prompts, with random fill keeping the initial FPR at
  f0 (so J is fixed): fate flips between c = 0.25 and c = 1. The theory gives a predicted c*.
- P4 Reachability: a shared FP with tiny base mass is learned later than a reachable one, or not
  within T. Its delay scales with log(1/mass).
- P5 Diagnostic: a gold-free *cross-prompt acceptance concentration* score (how unevenly a
  verifier's acceptances of responses written for other prompts are spread over responses,
  weighted by base mass) separates collapsing from benign verifiers. Static FPR and J do not.
- P6 Replication with real small LLMs (two families) on math, with a master-key verifier versus
  matched random and per-prompt false positives.

## What would falsify it
- P2 fails, i.e. the hashed per-prompt table collapses like V3: then consistency, not sharing, is
  the driver.
- P3 shows no coverage dependence at fixed FPR.
- P6: LLMs behave differently from the toy, e.g. master keys stay benign or random FPs collapse.
