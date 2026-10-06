# E006 — which false positives flip fate? Disentangling at a matched initial FPR (protocol)

Frozen before any E006 audit, diagnostic or run (2026-10-05). Theory: `research/paper/theory.md`.
Pipeline as in E005b-0 §12/§14:
- calibrated base v2 (sha `74865408…`); 2-layer char Transformer on two-digit addition;
- GRPO-style RL with P = 32, G = 8, group mean/std advantages, μ = 1, β = 0;
- fresh Adam lr 1e-4, grad clip 1.0, T = 1000;
- dev evaluation every 25 steps (4 samples/item, eval seed 20261312); CPU, 4 threads.

## Arms (all matched to the frozen f0 = 0.1083, configs/e005b/matched_f0.json)
Notation:
- M = valid wrong answers that end in digit 0;
- q = base-policy share of wrong responses in M (calibration audit: q = f0 = 0.1083);
- "fill(r)" = every other wrong response is accepted by a fresh coin of rate r.

Correct answers are always accepted (FNR = 0 in all arms).

| arm | false-positive structure | consistent | shared across prompts |
|---|---|---|---|
| clean | none | – | – |
| randfp | fill(f0) on all wrong responses | no | no |
| hashtab | wrong (prompt, completion) accepted iff hash(prompt, completion) < f0 (salt 20261350) | yes | no |
| cov25 / cov50 / cov75 | M accepted on a fixed fraction c ∈ {0.25, 0.5, 0.75} of prompts (unordered-pair hash < c, salt 20261351); fill(r(c)) elsewhere, r(c) = (f0 − c q)/(1 − c q) | partly | partly |
| exploit | M accepted on all prompts (c = 1, no fill; = E005b V3) | yes | yes |
| rarekey | one valid wrong value ν* (not ending in 0) accepted on all prompts; fill((f0 − ε)/(1 − ε)) elsewhere | yes | yes, but rare |

Choice of ν*:
- among values not ending in 0, the one whose base share of wrong responses (calibration audit) is
  closest to 0.005; ties go to the smaller value;
- ε is that share.

## Matching (as §14)
- Calibration audit = the E005b-0 calibration prompts and policy seed (first 3000 of permutation
  20261340, policy seed 20261341). Verification audit = the next 3000 (policy seed 20261342).
  Coin seed for verification: 20261343.
- Criterion per arm: |FPR_arm(verification) − f0| ≤ 0.015.
- **Stop rule:** an arm that fails is not trained, and it is reported. No retuning.

## Training
- Seeds 11, 12, 13, 14, 15 (fresh) for all 8 arms: 40 runs.
- Streams as §14:
  - prompts: Generator(10000 + seed);
  - policy: Generator(seed);
  - coins: SeedSequence([20261320, seed]).
- The hash and coverage salts are fixed constants, not seed-dependent. The structure is part of
  the verifier, not of the run.
- Sequential or 2-way parallel on mac-mini-remote, 4 threads per process.

## Outcomes
- **Primary:** mean sampled dev gold accuracy over the final four evaluations. Paired differences
  vs clean, per seed.
- **Labels (§12 rule):** "lower" if the mean ≤ −0.05 and all seeds are < 0; "higher" in the
  symmetric case; else "little difference".
- **Collapse:** primary < 0.2.
- **Mechanism logs:**
  - batch FPR;
  - batch M-mass and ν*-mass (wrong responses in M / equal to ν*);
  - mixed groups under V and G;
  - dev wrong-suffix mass;
  - dev modal-answer share.

## Pre-registered predictions
- **H1 (replication).** randfp − clean: "little difference".
- **H2 (consistency alone is benign).** hashtab − clean: "little difference".
  - Rival: if consistency drove harm, hashtab would collapse.
- **H3 (coverage).**
  - The seed-mean primary is non-increasing in c across randfp (c = 0), cov25, cov50, cov75 and
    exploit (c = 1).
  - cov25: "little difference".
  - cov75 and exploit collapse in ≥ 4 of 5 seeds each.
  - cov50: no prediction (the mean-field c*_0 ≈ 0.42 lies just below 0.5, and the absorption /
    gold-improvement refinements push the sustained threshold up).
- **H3b (initial push).** For the coverage arms, the sign of the empirical initial advantage mass
  on M predicts the sign of the change in dev wrong-suffix mass from step 0 to step 50.
  - The advantage mass is D̂(c): the sum of GRPO advantages of M-samples, averaged over
    calibration-audit groups and 50 coin replicates.
  - D̂ is computed and committed before training (`theory_predictions.py`).
- **H4 (reachability).** rarekey: the first step at which its batch ν*-mass among wrong responses
  reaches 0.5 is later than exploit's first step with batch FPR ≥ 0.5 in every seed, or never
  occurs. No prediction on collapse.
- **H5 (diagnostic).** On the panel of 10 verifiers, the gold-free response-main-effect score
  RME (below) has Spearman ≥ 0.7 with harm, and a higher Spearman than either static FPR or
  Youden's J.
  - Panel: clean, flip (E005b V1), deleted (V2), exploit, randfp, hashtab, cov25, cov50, cov75,
    rarekey.
  - Harm = −(mean paired primary difference vs clean). E005b V1/V2 use their matrix seeds 1–3;
    every other arm uses E006 seeds 11–15.

## Diagnostic definitions (frozen)
Inputs:
- the calibration-audit base samples;
- K = the distinct completions (token tuples) among them, with base mass m_k = count / samples;
- for each k, a panel of 64 audit prompts drawn without replacement (rng 20261360) from prompts on
  which k was *not* sampled.

Quantities:
- a_k = mean over the panel of V(x, k), one query per pair, with coins from rng 20261361.
- ā = Σ m_k a_k.
- **RME = Σ_k m_k [ (a_k − ā)² − a_k (1 − a_k) / 64 ]**. This is the response main effect of the
  cross-prompt acceptance matrix with the binomial noise floor removed.
- Gold-free: a completion sampled for another prompt is wrong on x with high probability, and
  coincidental correct answers are left in.
- Also reported:
  - the prompt main effect PME (variance over prompts of the acceptance of random foreign
    completions), which is what flags V2;
  - static FPR and J on the verification audit.

## Analysis
`experiments/e006/analysis.py` is committed before training. Outputs:
- per-arm curves;
- paired tables;
- labels;
- H1–H5 verdicts;
- the D̂(c) vs observed comparison;
- a figure.

## Not claimed
- LLM-scale behaviour. That is the separate validation step, E007.
- That RME is the best possible diagnostic. It is one pre-specified gold-free score.
- Category-level matching.
