# Theory: when do verifier false positives change the fate of RLVR? (draft v0.1, 2026-10-05)

Status tags: [proved] = short proof written here, with a numerical check in
`tests/test_e006_theory.py`. [derived] = mean-field derivation under the stated assumptions, not
a theorem. [conjecture].

## Setup
- Prompts `x ∈ X` with weights `w_x`; responses `y`; policy `π_θ(y | x)`, with shared parameters θ.
- Gold `G(x, y) ∈ {0, 1}`; verifier `V(x, y) ∈ {0, 1}`, which may be random.
- Per-prompt gold accuracy `p_x = E_π[G]`; per-prompt verifier mean `v_x = E_π,V[V]`.
- GRPO-style update in the large-group limit with group standardization: the advantage of `y` on
  `x` is `A(x, y) = (E[V | x, y] − v_x) / s_x` with `s_x = sqrt(v_x (1 − v_x))`.
- Prompts with `s_x = 0` contribute nothing (zero-variance groups).
- Expected update direction: `u(θ) = Σ_x w_x Σ_y π(y|x) A(x, y) ∇ log π(y|x)
  = Σ_x (w_x / s_x) ∇ J_V,x(θ)`, where `J_V,x = E_π[E V(x, ·)]`. The baseline term vanishes because
  `Σ_y ∇π = 0`.

## Proposition 1 (fresh false positives only rescale each prompt's gold gradient) [proved]
Let `V = G` on correct responses, and let every response with `G = 0` be accepted by an independent
fresh coin of rate `f ∈ [0, 1)`. Then for any parameterization (shared or not)

    u_f(θ) = Σ_x λ_x(f) ∇ J_G,x(θ),   λ_x(f) = w_x (1 − f) / s_x(f) > 0,

where `s_x(f) = sqrt(v_x (1 − v_x))` and `v_x = f + (1 − f) p_x`.

*Proof.* `E[V | x, y] = G + f (1 − G) = f + (1 − f) G`, so `J_V,x = f + (1 − f) J_G,x` and
`∇ J_V,x = (1 − f) ∇ J_G,x`. Substitute into `u`. ∎

*Consequences.*
- `u_f` is a positive combination of the per-prompt gold gradients. It ascends the reweighted gold
  objective `Σ_x λ_x J_G,x`, so no gold-orthogonal direction is ever pushed.
- Compared with the clean weights `w_x / sqrt(p_x (1 − p_x))`, only the relative weights of
  prompts and the overall rate change.
- This is the shared-parameter version of "rate, not fate" (Rate or Fate, J = 1 − f > 0). Finite
  groups add zero-mean noise and a per-prompt standardization bias. These are checked by Monte
  Carlo in the test.

## Proposition 2 (consistent false positives add the verifier's own gradient) [proved]
For any deterministic verifier,

    u(θ) = Σ_x (w_x / s_x) (∇ J_G,x + ∇ F_x),   F_x(θ) = Σ_{y: G=0, V=1} π(y | x),

where `F_x` is the policy's false-positive mass on `x`.

*Proof.* `J_V,x = J_G,x + F_x` for deterministic V with no false negatives. ∎

So consistency matters only through the extra term `Σ_x (w_x / s_x) ∇ F_x`. Whether that term
competes with gold depends on how the per-prompt false-positive gradients **add up across prompts**.

## Definition (cross-prompt coherence of the false-positive gradient)
For a set of prompts B,

    κ_FP(B) = || Σ_{x∈B} g_x ||² / Σ_{x∈B} || g_x ||²,   g_x = (w_x / s_x) ∇ F_x.

`κ_FP` ranges from 0 to |B|:
- about 1 when the per-prompt false-positive gradients are mutually orthogonal (incoherent);
- about |B| when they point the same way (coherent).

`κ_G` is defined the same way for gold.

## Proposition 3 (coherence scaling) [proved for the log-linear model; mechanism for networks]
Take a log-linear policy `log π(y | x) ∝ θ · φ(x, y)`.
- (a) If every prompt's false-positive set is a single response `m` with a common feature
  component, i.e. `φ(x, m) = e_m + ψ_x(m)` with the `ψ_x` mutually orthogonal and orthogonal to
  `e_m`, then the `e_m` component of `Σ_x g_x` grows linearly in |B|. Hence κ_FP grows linearly
  in |B| when that component dominates.
- (b) If the false-positive responses are prompt-specific with mutually orthogonal features, then
  κ_FP = 1 exactly.

*Proof.* `∇ F_x = Σ_{y∈FP(x)} π(y|x) (φ(x, y) − φ̄_x)`, with `φ̄_x` the policy-mean feature.
- (a) Each term contributes `π(m|x) (1 − π(m|x)) e_m` along `e_m`, plus prompt-specific parts, so
  the `e_m` components add.
- (b) Orthogonal summands give `||Σ g_x||² = Σ ||g_x||²`. ∎

Assumption flag: (b) needs the policy-mean features `φ̄_x` to be orthogonal across prompts as
well. In the numerical check they are made orthogonal by construction.

Reading:
- A first-order step of size η raises `Σ F_x` by `η Σ_x <∇F_x, u>`. When false-positive gradients
  are coherent, the step size each prompt's false-positive mass receives scales with the number of
  covered prompts (the same Adam-normalized step serves all of them).
- Gold competes through its own coherence κ_G, i.e. how much the gold skill generalizes.
- A *master key* is the extreme of (a). A *hashed per-prompt table* is (b). This explains why the
  persistent per-task false positives in Leaky were benign while master keys collapse.

## Proposition 4 (absorption under group standardization) [proved]
If a prompt reaches a state where every response in the policy's support has `V = 1`, then
`s_x = 0` and the prompt's contribution is zero (zero-variance groups are masked). Prompts in such
a state receive no gold signal from their own gradient term. A state where all prompts are in it
is a fixed point of the expected update. ∎

Remark: E005b-0's V3 runs reach mixed-group fractions of 0.003–0.006, so they are empirically
in this state.

## Derivation 5 (critical coverage at fixed initial FPR) [derived, mean-field]
Partial master key. The shared behaviour M (e.g. "a valid answer ending in 0") is accepted
deterministically on a fixed fraction c of prompts. Every other wrong response, on every prompt,
is accepted by a fresh coin at rate r. Choose r so that the initial FPR stays at f0:
`r(c) = (f0 − c q) / (1 − c q)`, where q is the base-policy probability that a wrong response is in
M. (With f0 = q, c = 1 recovers V3 and c = 0 recovers VR.)

Assume:
- M's logit has a dominant shared component `w_M`;
- prompts in the covered and uncovered sets are exchangeable at initialization.

The expected drift of `w_M` is then proportional to

    D(c) = c · E_cov[π_M (1 − v_x) / s_x] + (1 − c) · E_unc[π_M (r − v_x) / s_x].

With exchangeability, `sign D(c) = sign( c (1 − v̄) + (1 − c)(r(c) − v̄) )`. The drift is zero at

    c*_0 = (v̄ − r(c*_0)) / (1 − r(c*_0)).

Our base gives v̄ ≈ 0.388 + 0.612 · 0.108 ≈ 0.455 and q ≈ 0.108, so c*_0 ≈ 0.42.

Two refinements change this during training:
1. As gold improves on uncovered prompts, `v_x` rises there and M is penalized more.
2. As covered prompts are absorbed (Prop. 4), their positive push switches off.

Both forces raise the coverage needed for a *sustained* takeover above c*_0. The prompt-level
estimate `D(c)` is computed exactly from audit samples in `experiments/e006/theory_predictions.py`
(no exchangeability assumption) and is pre-registered.

## Conjecture 6 (reachability)
If the shared false-positive behaviour has base mass `π_M(0) = ε`, its mass grows roughly as
`ε exp(κ λ t)` while it is small. Takeover time therefore scales like `log(1/ε) / (κ λ)`. An
unreachable master key (ε ≈ 0) is never found by on-policy sampling, whatever its coherence.

## What the theory does not claim
- It does not predict final accuracies.
- It does not model finite-group variance effects beyond the Monte Carlo checks.
- Prop. 3 is exact only for log-linear policies. For networks, κ is an empirical quantity
  (measurable at initialization via per-prompt gradients, with gold labels). That is a mechanism
  check, not a practical diagnostic, consistent with E002 / E005a-R.

## 5b. Local drift (refinement after E006; status: [derived], tested in E008)
- Drop exchangeability. Prompt x's contribution to the shared key logit has sign
  `c_x (1 − v_x) + (1 − c_x)(r − v_x)`, where c_x ∈ {0, 1} is local coverage.
  - Covered prompts always push the key up, by `1 − v_x`.
  - Uncovered prompts push it down, by `v_x − r`. That push is weaker on hard prompts (low v_x).
- Under random coverage at rate c, the expected push on prompt x is positive iff
  `v_x < c + (1 − c) r`. Hard prompts are where a partially covered key gains.
- E006 (post hoc): cov25's pooled push was −0.005, but +0.004 on three-digit sums (base accuracy
  0.24). The key grew (+0.14 dev wrong-suffix mass by step 50).

## 6. Race model v1 [model; post-hoc check partly failed]
`src/vdyn/e006/race.py`.
- **Model:**
  - per prompt, three outcomes (correct C, key K, other wrong W);
  - base probabilities from the base model;
  - shared logits s (gold skill) and k (key) moved by exact finite-group expected GRPO updates
    (G = 8, ddof 1 + 1e-4, zero-variance groups → 0) under SGD with step sizes lr and lr_k.
- **The learnability asymmetry lr_k / lr is the model's only structural parameter beyond the
  base probabilities.** It encodes that a constant-output behaviour is far easier for the network
  to learn than the gold skill.
- **Fit** on E006 clean + exploit only: lr_k / lr ≈ 400.
- **Qualitative reproduction:** random fill is benign, full coverage collapses, absorption holds.
- **Quantitative failures** (post hoc):
  - it predicts collapse at cov50 (observed: no collapse, −0.107);
  - it predicts that the rare key takes over at step 656 (observed: never).
- **Diagnosis.** In a flat softmax a single key logit takes mass from correct and wrong answers in
  proportion. At partial coverage, the real network let the key replace mostly *wrong* answers
  (cov50: key share of wrong 0.69, accuracy 0.64). Missing ingredients:
  - per-prompt or local learning, which lets covered prompts absorb while uncovered prompts keep
    suppressing the shared key;
  - a nested "commit vs. guess" structure.
- **v2** adds local terms and is to be tested only on data not used to build it.

## Proposition 5 (output-level drift) [proved; numerical check in tests/test_e006_theory.py]
- **Setting.** A log-linear policy in which output k carries a feature e_k shared across all
  prompts (for a language model, roughly its unembedding direction), plus prompt-specific
  features.
- **Statement.** The expected update on the coordinate of e_k is

      D_k = Σ_x (w_x / s_x) · π(k | x) · (E[V | x, k] − v_x).

- **Proof.** The e_k component of `∇ log π(y | x)` is `1[y = k] − π(k | x)`. The
  `−π(k | x) Σ_y π(y | x) A(x, y)` term vanishes because advantages are centred
  (Σ_y π(y|x)(E[V|x,y] − v_x) = 0). What remains is `π(k|x) A(x, k)`, summed over prompts with
  weights w_x / s_x. ∎
- **Reading.**
  - The output-specific parameters of k move up iff k's acceptance, averaged over *the prompts
    where the policy produces k* (weights π(k|x) w_x / s_x), exceeds the weighted mean verifier
    reward on those prompts.
  - This is Derivation 5's coverage argument at the level of outputs instead of prompts.
  - It explains E008 / E009. Under category coverage, "110" is produced almost only on covered
    prompts (conditional acceptance ≈ 1). Under parity or random coverage, the same output is
    produced on covered and uncovered prompts (≈ 0.5), and the uncovered ones pull it down.
- **Empirical counterpart.** ACM_τ in `experiments/e006/posthoc_conditional_coverage.py`, tested
  prospectively in E011.
- **Limit.** The proposition gives the direction of the output-specific push at a given policy.
  The empirical threshold (≈ 0.6–0.75 rather than the mean reward ≈ 0.45) reflects later dynamics:
  gold improves, v_x rises, and covered prompts are absorbed.

## Derivation 6 (takeover time of an always-accepted output) [derived; replaces Conjecture 6]
- Take an output k accepted on every prompt, and treat the rest of the dynamics as frozen.
- By Prop. 5 its bias obeys `ḃ_k = η Σ_x w_x π(k|x)(1 − v_x)/s_x = η A_k π̄_k`.
  - `π̄_k = mean_x π(k|x)`;
  - `A_k = mean_x π(k|x)(1 − v_x)/s_x / π̄_k` is the effective advantage on the prompts where k
    is produced.
- With `π̄_k ≈ ε_k e^{b_k}` while k is rare, `dπ̄_k/dt = η A_k π̄_k²`. Growth is **hyperbolic**,
  and the blow-up (takeover) time is `t* ≈ 1 / (η A_k ε_k)`.
- Takeover time is inversely proportional to base mass, not logarithmic as Conjecture 6 assumed.
  The gold skill competes: it raises v_x on k's prompts, shrinks A_k, and can stop the takeover.

**Post-hoc check** (`experiments/e006/posthoc_takeover.py`, exact base distributions):

| key | arm | ε | A | 1/(Aε) (units of 1/η) | observed |
|---|---|---|---|---|---|
| 111 | set02 | 0.0099 | 1.55 | 65 | FPR ≥ 0.5 at steps 23–33 |
| 57 | rarekey | 0.0056 | 0.58 | 308 | never within 1000 steps |
| 10 | far | 0.0052 | 2.21 | 86 | far's early growth |
| 900 / 800 | far | < 1e-6 | – | > 1e6 | reached anyway |

The ordering agrees for 111 vs 57. 57 is produced on easy two-digit prompts, where gold improves
fast and A_k shrinks, so it never takes over.

The derivation fails for 900 / 800. They were unreachable for an independent-output model, yet
reached after "100" and "10" grew. The network generalizes across outputs that share token
structure ("x00"), which an output-level model cannot capture. Reachability is a property of the
*network's* output space, not of the base distribution alone, which is another reason static
scans of base samples cannot enumerate discoverable keys (E011).
