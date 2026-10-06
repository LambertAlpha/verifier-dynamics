# Same Rate, Different Fate: Which Verifier False Positives Does RLVR Exploit?

*Draft v0.1 (2026-10-06). Numbers marked [TBD] wait on E007 / E008. Every experiment was
pre-registered in `research/03_experiment_registry.md`, and the failed predictions are reported
below.*

## Abstract (draft)
Reinforcement learning with verifiable rewards (RLVR) trains models against automatic checkers
that are wrong some of the time. Verifier quality is usually summarized by static error rates.
Recent theory shows that, for independent per-prompt noise, the sign of Youden's index
J = TPR − FPR decides whether noise only slows learning ("rate") or reverses it ("fate").

We show that static rates, including J, do not determine fate once false positives have structure
across prompts.
- In a controlled study, eight verifiers are matched on initial FPR, FNR, accuracy and J.
  - Random false positives, persistent per-prompt false positives, and a shared but rarely
    sampled false positive all leave learning intact (within 2 points of clean).
  - A false positive that one reachable behaviour satisfies across prompts (a "master key")
    collapses accuracy from 0.74 to 0.01. Its harm rises sharply with the fraction of prompts it
    covers.
- We explain this with three results:
  - fresh false positives provably only rescale each prompt's gold gradient, even with finite
    groups and shared parameters;
  - consistent false positives add the verifier's own gradient, whose cross-prompt coherence
    decides whether it can outpace learning the gold skill;
  - group standardization makes the collapsed state absorbing.
- We propose a gold-free pre-training test, the response main effect of the cross-prompt
  acceptance matrix. It ranks harm among rate-matched verifiers, though not across all kinds of
  harm.
- In [TBD: Qwen2.5-0.5B on GSM8K], [TBD].

## 1 Introduction
- **RLVR depends on checkers.**
  - RLVR is the dominant recipe for training reasoning models. Its reward is a program: an answer
    extractor, a unit-test harness, an LLM judge.
  - These programs are imperfect. Rule-based math checkers reject equivalent answers [Pitfalls;
    Where the Verifier Fails]. Test suites accept wrong programs [Leaky]. LLM judges accept
    content-free "master keys" [One Token].
- **The open practical question: which verifier errors matter?**
- **Existing answers are mixed.**
  - Robustness studies inject symmetric noise and find RLVR forgiving.
  - Rate or Fate [2601.04411] formalizes this. In a per-prompt bandit, the incorrect mass drifts
    with Youden's J, so errors with J > 0 only rescale the convergence time.
  - Yet natural, persistent false positives in code suites are nearly harmless [Leaky].
  - And an RL run collapsed onto an LLM judge's master keys [One Token].
  - Nothing reconciles these, and in practice verifiers are still compared by their error rates.
- **Our claim.**
  - RLVR selects among reward-satisfying behaviours by *how fast the policy can learn them*, not
    by whether they are correct.
  - A false positive changes the outcome only when it forms a behaviour that is reachable from
    the current policy and that is reinforced coherently across prompts through shared parameters.
  - Static error rates measure how much a verifier admits, not how learnable the admitted
    behaviour is.
- **Contributions.**
  1. *Controlled evidence at a matched rate.* Verifiers matched on initial FPR, FNR, accuracy (and
     hence J) differ sharply in outcome, depending only on the cross-prompt structure and
     reachability of their false positives (§4).
  2. *Theory (§3).*
     - Fresh false positives rescale each prompt's gold gradient. This holds exactly for finite
       groups and any parameterization.
     - Consistent false positives add a gradient whose cross-prompt coherence scales with the
       number of prompts a shared behaviour satisfies.
     - Zero-variance groups make master-key collapse absorbing.
  3. *A gold-free exploitability test (§5).* It is a two-way decomposition of a verifier's
     acceptances of completions transplanted across prompts. We also test where it fails.
  4. *LLM validation (§6)* on GSM8K with a real loose-extraction bug class.
  5. *Pre-registration.* Predictions, analysis code and matching audits were committed before
     each experiment. Six registered predictions failed, and we report them.

## 2 Setting
- **Objects.** Prompts x; responses y ~ π_θ(·|x); gold G(x, y) ∈ {0, 1}; verifier V(x, y) ∈ {0, 1}.
- **Update.** GRPO-style on-policy update: per prompt, G samples, advantages
  (V − mean)/(std + ε), zero-variance groups contribute nothing, μ = 1 (no ratio clipping), β = 0.
- **Static quantities.** FPR = P(V=1 | G=0), FNR, accuracy, J = 1 − FNR − FPR. All are measured
  at initialization on an audit.
- **Matching.** Calibrate a rate on one audit, freeze it, verify on an independent audit within a
  frozen tolerance, and do not train if matching fails.

## 3 Theory
(see theory.md for the full statements and proofs; numerical checks in tests/test_e006_theory.py)
- **Prop. 1, fresh false positives.**
  - E[V|y] = f + (1 − f)G, so the update is Σ_x λ_x ∇J_G,x with λ_x > 0.
  - Finite-group version: a sample's expected advantage depends on y only through G(y), and
    A(V=1) ≥ 0 ≥ A(V=0). So each prompt's expected update is parallel to its gold gradient.
- **Prop. 2, consistent false positives.** u = Σ_x (w_x/s_x)(∇J_G,x + ∇F_x).
- **Prop. 3, coherence.** κ_FP grows linearly in the number of covered prompts for a shared key
  and equals 1 for orthogonal per-prompt tables.
- **Prop. 4, absorption.** A state where everything in the support is accepted is a fixed point.
- **Derivation 5 and its local refinement.**
  - The drift sign on prompt x is positive iff v_x < c + (1 − c) r, so hard prompts push a
    partially covered master key up first.
  - The pooled version failed its pre-registered test (§4.3). We report the failure and then
    the local test (E008).

## 4 Controlled experiments (toy Transformer)
**4.1 Setup.**
- Task: two-digit addition. Model: a 2-layer, 402k-parameter character Transformer, SFT to a
  calibrated base (sampled dev accuracy 0.38).
- RL: GRPO-style, P = 32 prompts × G = 8, Adam 1e-4, T = 1000, CPU (bit-reproducible).
- Outcome: the mean sampled dev accuracy over the final four evaluations (clean: 0.744).
- Seeds: 5 per arm.
- Matching:
  - the master-key set M = "valid wrong answers ending in 0" fixes the rate f0 = 0.108 on a
    calibration audit;
  - every arm is verified within 0.015 on an independent audit, with FNR = 0;
  - so all arms share initial FPR, FNR, accuracy, and hence J ≈ 0.88.

**4.2 Same rate, different fate (E006).** Table 1:
- *Benign, despite J being equal to the collapsing arms:*
  - fresh random false positives (−0.004);
  - a consistent per-prompt table (−0.018);
  - a rare constant accepted everywhere (−0.016; it was never taken over).
- *Collapse:* M accepted everywhere (−0.731, 5/5 seeds). Batch FPR exceeds 0.5 by steps 7–12, and
  mixed groups fall to 0.004 (Prop. 4).
- *Dose:* with M accepted on a random fraction c of prompts, harm is 0.004, 0.085, 0.107, 0.596,
  0.731 for c = 0, .25, .5, .75, 1. Collapse sets in between c = 0.5 and 0.75.

**4.3 Where our predictions failed.**
- The mean-field drift (Derivation 5) predicted no harm below c* ≈ 0.42. cov25 lost 8.5 points
  (H3 failed).
- The pooled initial push had the wrong sign for cov25 (H3b failed).
- A post-hoc split by category showed a positive push on three-digit sums. We pre-registered a
  difficulty-local version (E008: L1, L2), and it failed too (§4.4).

**4.4 Not difficulty, not identifiability: what the key is accepted *for* (E008, E009, E011).**
- *E008.* At a global coverage of 0.5, accepting M on an input-identifiable category collapses
  exactly that category:
  - covhard: three-digit accuracy 0.71 → 0.04, the rest unchanged;
  - coveasy: two-digit categories → 0.01 / 0.05.
  - Random coverage at the same rate collapses nothing.
  - The difficulty hypothesis we had pre-registered (L1) failed: the easy-category arm was the
    most harmful.
- *E009.* We then pre-registered "coverage within input-identifiable regions". Parity regions
  (first operand even; sum even) were *not* collapsed: harm 0.149 / 0.121, region accuracy
  0.57 / 0.60. They behaved like random coverage. Identifiability is falsified.
- *Post hoc.* What distinguishes the collapsing verifiers is the *on-policy conditional acceptance*
  of each wrong output: how often an output is accepted on the prompts where the policy
  actually produces it.
  - Under category coverage, "110" is produced almost only on three-digit sums, all of them
    covered.
  - Under parity coverage, the same "110" is produced on covered and uncovered prompts alike.
  - ACM_0.75, the base-policy mass of wrong outputs whose conditional acceptance is ≥ 0.75, gives
    Spearman 0.77 with harm across 18 toy verifiers. Static FPR gives 0.10. There is a clean gap
    between harm ≥ 0.33 (ACM ≥ 0.037) and harm ≤ 0.15 (ACM ≤ 0.005).
- *E011* tests a frozen rule (harm ≥ 0.25 iff ACM_0.75 ≥ 0.02) on new verifiers that accept the
  key only for near misses, only for far misses, or for near misses on half the prompts. [TBD]

**4.5 Reachability.**
- A single wrong constant accepted everywhere collapses training at 1.9% of the base wrong mass
  (set02: 5/5), but not at 0.5% (rarekey).
- Arbitrary frequent wrong values behave like "ends in 0" at matched mass (|Δharm| = 0.002, R2).

**4.6 A minimal model (race model v1).**
- Exact finite-group GRPO updates on a three-outcome policy with shared skill and key logits,
  fitted on clean and exploit only.
- It reproduces benign random noise, collapse at full coverage, absorption, and the E008 covhard
  and set arms (predictions committed before results).
- It fails on cov50 (it predicts collapse) and on the rare key (it predicts eventual takeover),
  because it has no per-output conditional structure.

## 5 A gold-free exploitability test
- **Method.** Transplant completions across prompts and decompose the acceptance matrix into a
  response main effect (RME), a prompt main effect (PME), an interaction and retest noise.
- **Toy results.**
  - RME rises with coverage; it is ≈ 0 for random, per-prompt, flip and deleted verifiers.
  - Deleted shows up only in PME.
  - Spearman with harm over the matched arms is 0.83 (H5b).
  - Across all ten verifiers it is 0.60, below static FPR (H5 failed), because RME does not see
    rate-type harm.

## 6 LLM validation (E007)
[TBD]

## 7 Related work
**RLVR and its reward programs.**
- RLVR trains reasoning models on programmatic rewards (Tülu 3, Lambert et al. 2024;
  DeepSeek-R1, 2025), usually with GRPO (Shao et al. 2024).
- Audits of the reward programs show:
  - rule-based math checkers reject equivalent answers, while model-based checkers can be hacked
    (Pitfalls of Rule- and Model-based Verifiers, 2505.22203);
  - most in-contract failures of math checkers come from whitespace and punctuation, and a
    scale-invariant numeric cascade accepts off-by-one answers (Where the Verifier Fails,
    2609.01354);
  - verifier errors are strongly correlated within a GRPO group (2609.06386).
- Verifier fuzzing finds false positives without RL (2606.01066).
- These works measure *how often* a verifier errs. We ask *which* errors training exploits.

**Noisy rewards in RLVR.**
- Rate or Fate (2601.04411) gives a replicator analysis over reasoning modes. Fate is decided by
  Youden's J; J > 0 only rescales time.
- Noise-corrected GRPO (2510.18924) and backward/forward corrections (2510.00915) de-bias
  gradients under class-conditional noise.
- All of these model noise as independent per sample or per prompt.
- Our Prop. 1 extends "rate, not fate" to finite groups and arbitrary parameter sharing for fresh
  noise. Our experiments show that J does not decide fate once false positives are coherent across
  prompts.
- Spurious Rewards (2506.10947) shows that random or even incorrect rewards can help specific
  model families, through clipping-induced amplification of high-prior behaviours. We use μ = 1,
  where this clipping is inactive.

**Reward hacking and master keys.**
- Reward hacking is formalized by Skalse et al. (2022). Its scaling with optimization is studied
  for reward models by Gao et al. (2023) and under misspecification by Pan et al. (2022).
- In RLVR:
  - LLM judges accept content-free "master keys", and an RL run collapsed onto them (One Token to
    Fool LLM-as-a-Judge, 2507.08794);
  - extensional verifiers induce instance-level shortcuts (LLMs Gaming Verifiers, 2604.15149);
  - by contrast, persistent per-task false positives in code test suites were nearly harmless in a
    preregistered contrast (Leaky suites, 2607.11022).
- Our account reconciles the last two: persistence alone is not exploitation, cross-prompt
  coherence and reachability are.

**Corrupted rewards and shortcut learning.**
- Everitt et al. (2017) show that RL with a corrupted reward channel is unlearnable in the worst
  case, and that richer feedback helps.
- Shortcut learning (Geirhos et al. 2020) describes networks preferring simple, broadly predictive
  features.
- A master key is a shortcut that the *reward* offers. Our learnability asymmetry measures how
  much easier it is for the network to take it than to learn the gold skill.

## 8 Limitations
- Small models (402k toy; 0.5B LLM).
- One task family per scale.
- Matching holds at initialization only. The structured false-positive rate itself changes
  within ~10 steps, which is the point, but it limits what "matched" means.
- The theory's coverage threshold is not predicted from first principles.
- The diagnostic misses rate-type harm and strategy-level keys [E007 TBD].
