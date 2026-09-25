# 06 — E004 design memo (v1, 2026-09-25)

**Status: DESIGN ONLY. Not registered, not implemented, not run.** This memo proposes the E004
design and a pre-registration text for collaborator review.

- Nothing here is experimental evidence.
- The numerical statements in §2 come from design-phase checks run in a scratch directory. They
  are recorded as **E000d** in the registry (retroactive, not pre-registered) and use the scripts
  in `research/design_checks/e004/` (a design aid, not experiment code).
- No E004 panel, target or predictor exists. E002 results are not revised.

**Status tags.**

| tag | meaning |
| --- | --- |
| [proved] | a derivation in this memo |
| [DC] | a design-phase numerical check (E000d) |
| [proposal] | a design choice for review |
| [open] | the collaborator should decide |

---

## 0. One-page summary

**What changes relative to the brief** (all justified in §2):

1. **The geometry-versus-probe question is degenerate in the classic families.** For every
   verifier whose false positives are triggered by response features that the gold-relevant
   parameters do not move, with FNR = 0 and the trigger active on every prompt, the geometry
   trajectory is an exact function of the simple observables:

       alpha = -FPR,    A^2 = J_G' / (1 - FPR),    C^2 = (1 - J_G) FPR'.

   This holds under natural gradient **and** under any metric block-diagonal between gold-relevant
   and feature parameters, including diagonal Adam [proved; DC1]. In those families L3 cannot add
   information to L2 with slopes. E004 must test geometry where the identity breaks: shared
   parameters, attempt credit, deletion, false negatives, prompt-specific triggers.
2. **The E001–E003 failure mechanism is optimizer-specific.** The same single-feature exploit
   ends as follows (normalized gold shortfall):

   | optimizer | shortfall |
   | --- | --- |
   | natural gradient | **0.43** |
   | infinite-batch Adam | 0.00 |
   | mean-field Adam, batch 512 / 64 / 8 | 0.01 / 0.03 / 0.10 |

   Under a scale-invariant optimizer, feature-triggered false positives with FNR = 0 hurt gold
   only through the signal-to-noise ratio [DC6]. A **preference inversion** hurts both optimizers
   (shortfall 0.43–1.88, with gold rising for ~15–35% of `T` before it declines): on hard prompts the
   exploit earns more expected reward than a genuine attempt [DC7]. E004a therefore crosses the
   optimizer (NG, Adam) and makes Adam primary for transfer claims.
3. **Hard pairs.**
   - The obvious one is impossible. Benign amplification cannot be matched to a feature exploit on
     `(J_G, FPR, J_G', J_V')`: at matched `J_G` and FPR its proxy/gold slope ratio is larger by at
     least FPR [proved; DC2]. "Proxy grows faster than gold ⇒ danger" is therefore anti-predictive
     for B.
   - Two other pairs exist.
     - Benign amplification vs a shared-parameter exploit: `J_G` and FPR equal to within 0.005
       over the first 5% of training, opposite `alpha`, `C_out` 0 vs 0.3–0.4. They end with the
       same outcome, so this is a *mechanism* pair [DC5].
     - **Latent decline** (preference inversion) vs benign amplification under mean-field Adam:
       `J_G`, FPR and `J_V` equal to within 0.002 over the first 5%, while `alpha` is
       −0.26 → −0.40 vs +0.03. One declines after peaking at 30% of `T` (normalized shortfall
       0.49), the other succeeds. On this *outcome* pair L2 is blind and geometry, even at
       `t = 0`, is not [DC8]. It is noise-free, and whether `alpha` is estimable at audit sizes is
       open (E002a).
4. **Outcome categories.** The brief's three categories are extended to four: SUCCESS / SLOW /
   STALL / DECLINE. "Not converged by `T`" (attenuation) and "converged below clean" are
   different failures with different remedies.
5. **Early warning versus early detection.** The primary early-warning endpoint is discrimination
   among runs whose failure is **not yet visible** at the horizon, plus lead time. "Prediction
   quality vs fraction observed" alone cannot separate warning from monitoring.

**Recommendation:** IMPLEMENT E004a, **staged**. Stage 0 (panel, oracle tier, hard pairs; design
split only) ends in a go/no-go gate, before the sampled tier and any predictor comparison. Five
decisions are needed first (§16).

---

## 1. Scientific questions (proposed wording)

- **RQ1 — outcome prediction.** From the first `h` of training (`h` a pre-specified fraction of
  `T`), how well can the long-run outcome be predicted? The outcome is the normalized gold
  shortfall relative to clean training, and SUCCESS / SLOW / STALL / DECLINE. **Primary: among
  runs whose failure is not yet visible at `h`** (early warning, §5.3).
- **RQ2 — mechanism diagnosis.** From the same prefix, which failure mechanism is operating? The
  classes are attenuation, deletion, exploit discovery (Route A), true-signal exhaustion
  (Route B), displacement (preference inversion) and benign amplification. The key subtask is
  exploit discovery vs exhaustion.
- **RQ3 — information depth.** For RQ1 and RQ2, what does each information level add
  (L0 static, L1 `t = 0` geometry, L2 simple trajectory observables, L3 geometry trajectory)?
  Working hypothesis: L2 suffices for *whether*, and L3 adds to *why*. RQ3 is tested where the
  §2 identity does not make the answer trivial.
- **RQ4 (secondary, optional) — counterfactual optimizer prediction.** From rollouts of one run,
  predict the outcome under a different optimizer.
  - Geometry can be recomputed in another metric from the same rollouts; a probe observes only
    the optimizer that was run.
  - This is the one capability that only geometry has in principle (cf. E003-V: the
    optimizer-matched `C(0)` ordered vanilla outcomes, and the Fisher `C(0)` did not).
  - [open] Include or drop.

The paper-level story stays conditional: "simple early dynamics tell whether; geometry tells why"
is a hypothesis that E004 can refute (§14).

---

## 2. Design-phase facts that shape the design (E000d; not evidence)

All checks use the scratch "U-toy" of §3 with exact enumeration and autodiff.

**F1 — observable identity [proved; DC1].** Assume:

- the verifier has FNR = 0;
- false positives come from a feature event whose probability `S` does not depend on the prompt;
- the features are independent of the gold-relevant probabilities;
- the metric `M` is block-diagonal between the two parameter groups.

Then `g_V = (1 - S) g_G + (1 - J_G) grad S` with `<grad S, g_G>_M = 0`, so

    alpha = -FPR,   A^2 = J_G' / (1 - FPR),   C^2 = ||r||^2 = (1 - J_G)^2 grad S^T M grad S = (1 - J_G) FPR'.

- DC1 checks this on random 4-prompt instances under natural gradient: deviation ≤ 3e-15 for
  AND2, OR2 and single events.
- DC4(a) checks it under the diagonal Adam-like metric (AND2): equal to 4 digits.
- Hack acceptance with `p ≈ 1` also satisfies it under NG (deviation 1e-6). The HACK/OTHER split
  among wrong answers is Fisher-orthogonal to the SOLVE marginal.
- It **breaks** (relative deviation 0.4–31) for:
  - attempt credit (B);
  - deletion (X);
  - false-negative coins (R-type);
  - prompt-specific triggers.
- **Consequence.** In the Candidate-1/3 family, `(A_t, alpha_t, C_t)` carry no information beyond
  `(J_G, FPR)` and their first derivatives. E002/E003's "geometry vs probes" was comparing
  equivalent information at different noise levels.

**F2 — shared parameters are invisible to natural gradient [proved; DC1, DC4].** Coupling the
exploit features to the skill parameters (`logit z_j = phi_j + lam_j·mean(u)`) is a linear
reparameterization.

- Natural gradient is invariant to it (Prop. 2), so the coupled and uncoupled trajectories agree
  to 4e-9.
- Under the (infinite-batch) Adam-like flow the coupled exploit has `alpha(0) = +0.025` vs
  `-0.002`, and FPR at `t = 0.6` of 0.18 vs 0.016.
- Real LLMs share all parameters, so E004a puts coupling in **every** family (drawn at random).
- Without coupling, F1 makes L3 vs L2 trivial in the Y families.

**F3 — no decline under NG with FNR = 0 (tabular-in-strategy policies) [proved; DC3].** For a
saturated policy, `<grad E f, grad E g>_{F^-1} = Cov(f, g)`, so `J_G' = Cov(G, V) = J_G(1 - J_V)
>= 0` when FNR = 0.

- DC3: hack acceptance 0.9 with FNR = 0 gives no decline under NG (0.15 → 0.90).
- With FNR = 0.3 the verifier prefers the hack, and gold peaks at 0.20 at `t = 2`, then falls
  to 0.001.
- DECLINE therefore needs a verifier that strictly prefers some wrong behaviour over the attainable
  correct behaviour.

**F4 — benign amplification is more "proxy-alarming" than any feature exploit [proved; DC2].**
Under NG, at matched `(J_G, FPR)`:

- `rho_B = J_V'/J_G' = 1 + FPR(1 - J_G)/J_G`;
- `rho_Y <= rho_B - FPR` (Cramér–Rao on `kappa`);
- equality holds for a single-feature exploit (DC2: min over 40 matched pairs of
  `(rho_B - rho_Y) - FPR` is `-2e-10`).

**F5 — the gold-race stall is optimizer-specific [DC6].** Single-feature exploit (`s0 = 0.27`),
2 prompts, `T = 15` time units. Normalized shortfall: NG 0.434; sign-GD (infinite-batch Adam)
0.000; mean-field Adam 0.012 / 0.031 / 0.095 at 512 / 64 / 8 rollouts per step.

- Mean-field Adam is `theta_i' = g_i / sqrt(g_i^2 + sigma_i^2/B)`, with the exact per-sample
  variance.
- *Caveat:* mean-field Adam ignores momentum, the noise correlation of `v_hat`, and GRPO's
  per-group standard-deviation normalization. The sampled tier must re-test F5.

**F6 — preference inversion fails under both optimizers [DC7].**

- Hard context: `p = 0.3` (vs 0.9 on the easy one) with hack acceptance 0.6. Normalized
  shortfall 0.43 (NG) and 0.49 (mean-field Adam, `B = 64`). Gold peaks at `t ≈ 4–5.5` and then
  declines by 0.05–0.11.
- Inversion on both contexts: shortfall 1.56 / 1.88; gold falls to 0.07 / 0.00.
- Without inversion, both succeed.

**F7 — a mechanism hard pair exists [DC5].** Under the infinite-batch Adam-like flow:

- a benign-amplification structure and a coupled AND2 exploit, matched over `[0, 0.05 T]`;
- `|ΔJ_G| <= 0.005` and `|ΔFPR| <= 0.0015` at every checkpoint, far below the SE of a 256-rollout
  audit (≈ 0.02);
- `alpha = +0.45` vs `−0.37`, and `C_out = 0` vs 0.28–0.43;
- **both succeed** (normalized shortfall 0.000 vs 0.006).

**F8 — a latent-decline outcome hard pair exists [DC8].** Under mean-field Adam (`B = 64`):

- a displacement structure D (hard-prompt inversion: `p = (0.9, 0.3)`, `rho = 0.6`) and an
  optimized benign structure B, matched over `[0, 0.05 T]`;
- `|ΔJ_G| <= 0.0007`, `|ΔFPR| <= 0.0017`, `|ΔJ_V| <= 0.0012` at every checkpoint;
- geometry in the optimizer metric: `alpha` −0.258 → −0.397 (D, falling toward −1) vs
  +0.036 → +0.027 (B); `C` 0.21 vs 0.06;
- outcome: D peaks at `J_G = 0.551` at `t = 4.5` (30% of `T`) and ends at 0.445 (DECLINE,
  normalized shortfall 0.49); B ends at 0.548 = clean (SUCCESS).

*Caveats:* the check is noise-free; B's parameters come from an optimizer
(`p = (0.99, 0.11)`, `beta = 0.11`); and E002a showed `alpha` is poorly estimated at small `N`.
The sampled tier decides whether the difference is detectable.

---

## 3. Mechanism panel — one parameterization for every family

**Principle [proposal].** Families differ **only in the verifier**. The policy class, parameter
dimension, number of prompts and feature count are identical. In E002 the feature count alone
predicted outcome (C-index 0.57–0.60) because it revealed the type.

### 3.1 Unified policy ("U-toy")

- **Prompts.** `x = 1..K` (`K = 4` fixed), with weights `w ~ Dirichlet(2·1)`.
- **Response.** `y = (s, xi, z)`:
  - strategy `s ∈ {SOLVE, HACK, OTHER}`, softmax logits `(u_x, h, 0)`: `u_x` is per-prompt skill,
    `h` is a **shared** hack propensity;
  - success `xi ~ Bern(p_x)` when `s = SOLVE` (task difficulty; not controlled by the policy);
  - features `z ∈ {0,1}^m` (`m = 3`), independent Bernoulli with
    `logit z_j = phi_j + lam_j·mean_x(u_x)` (`lam` = capability coupling).
- **Parameters.** `theta = (u_1..u_K, h, phi_1..phi_m)`, `d = 8`, the same for every structure.
- **Gold.** `G = 1{s = SOLVE, xi = 1}`. `J_G = sum_x w_x p_x P_x(SOLVE)`; the maximum is
  `sum w_x p_x`.

### 3.2 Verifier channels (fresh coins; expected acceptance)

On prompts not in the deleted set:

- `P(V = 1 | G = 1) = 1 - fn_x`;
- `P(V = 1 | G = 0, s, z) = 1 - (1 - fp_x)·(1 - trig_x 1_E(z))·(1 - rho_x 1{HACK})·(1 - beta_x 1{SOLVE, failed})`.

On deleted prompts `x ∈ S`: `P(V = 1) = v0` for every response.

| channel | parameter | meaning |
| --- | --- | --- |
| FN coin | `fn_x` | rejects correct answers at random |
| FP coin | `fp_x` | accepts wrong answers at random |
| feature trigger | event `E` on `z`, `trig_x` | accepts wrong answers showing a feature pattern |
| hack acceptance | `rho_x` | accepts the HACK strategy |
| attempt credit | `beta_x` | accepts a genuine but failed attempt |
| deletion | `S`, `v0` | reward independent of the response |

**Background, in every family [proposal].**

- `fn_bg ~ U(0, 0.05)` and `fp_bg ~ U(0, 0.05)`, so that FNR > 0 or FPR > 0 alone does not reveal
  the family.
- `lam_j ~ N(0, 0.5^2)`.
- Initial logits: `u_x0 = mu + delta_x`, with `mu` and the spread `delta_x` drawn;
  `h0 ~ N(-2, 1)`; `phi_0 ~ N(-1.5, 1)`.
- Difficulties `p_x ~ Beta(2, 1.5)` in `[0.2, 0.98]`.

### 3.3 Families

Each family below is a verifier recipe on top of the §3.2 background.

**R — random attenuation.**

- *State / policy / G / optimizers:* §3.1; NG and Adam (§7).
- *V:* symmetric coins `fp_x = fn_x = eps ~ U(0.05, 0.3)` on every prompt.
- *Dynamics:* `g_V = (1 - 2 eps) g_G` exactly; NG slows by `(1 - 2 eps)`; under Adam the
  magnitude is normalized away and only the SNR falls.
- *Signature:* `alpha = -2 eps`, `C ≈ 0`; `C_out ≈ 0` apart from the background.
- *Long-run outcome:* SUCCESS or SLOW (never STALL or DECLINE in expectation).
- *Static metrics at `t = 0`:* FPR = FNR ≈ `eps`.
- *Distinguishing information:* J_G slope below its static-metric prediction; flat FPR; FNR > 0.

**X — signal deletion.**

- *V:* deleted set `S` of 1–2 prompts, `v0 ~ U(0.2, 0.8)`.
- *Dynamics:* `u_x` for `x ∈ S` receives no signal, except through the shared `h` and the
  coupling. Aggregate `alpha = -A_S^2/A^2` (Prop. 6).
- *Signature:* `C_in > 0`, `C_out ≈ 0`. `C/A` grows as the other prompts saturate.
- *Long-run outcome:* STALL at `J_G ≈ sum_{x ∉ S} w_x p_x + sum_{x ∈ S} w_x p_x P_x0(SOLVE)`.
- *Static metrics at `t = 0`:* FPR = `v0 × (wrong mass on S)`, FNR = `(1 - v0) × (correct mass on S)`.
- *Distinguishing information:* per-prompt accuracy curves (flat on `S`). **FPR rises by
  composition** as the other prompts improve, without any exploit (a hard case for FPR-only
  rules). The proxy–gold gap stays roughly constant.

**Y-A — exploit discovery (Route A).**

- *V:* a conjunctive event (AND2 or AND3) with low initial accessibility (`S_E0 ∈ [0.001, 0.05]`),
  active on every prompt.
- *Dynamics:* along the NG path `eta` rises (Prop. 10) and FPR takes off later.
- *Signature:* `C_out` rises (numerator-driven `C/A`); `alpha ≈ -FPR` when `lam ≈ 0`.
- *Long-run outcome:* STALL under NG. Under Adam it depends on batch and coupling (F5, shown for
  a single-feature trigger; AND-type to be mapped in stage 0); SUCCESS is common.
- *Static metrics at `t = 0`:* low FPR, low `C`.
- *Distinguishing information:* FPR convexity; rising `eta`. With `lam = 0`, L2 is equivalent to
  L3 (F1).

**Y-B — true-signal exhaustion (Route B).**

- *V:* an accessible single or OR2 trigger (`S_E0 ∈ [0.05, 0.4]`) plus a large difficulty spread
  across prompts.
- *Dynamics:* easy prompts saturate and `A_t` falls. The residual persists and the hard prompts
  lose the race (Prop. 7).
- *Signature:* the rise of `C/A` is **denominator-driven** (`A` falls); `eta` is flat or falling.
- *Long-run outcome:* STALL under NG; weaker under Adam (F5).
- *Static metrics at `t = 0`:* moderate FPR.
- *Distinguishing information:* J_G concave early (easy prompts saturating) together with a
  steady rise in FPR; per-prompt accuracy curves diverge.

**B — benign amplification.**

- *V:* attempt credit `beta_x ~ U(0.3, 1)` on every prompt.
- *Dynamics:* the verifier optimum equals the gold optimum; learning is faster than clean.
- *Signature:* `alpha = (1 - p) beta / p > 0` in each prompt; `C_in > 0` when `p_x` differs
  across prompts; `C_out ≈ 0`.
- *Long-run outcome:* SUCCESS (negative control).
- *Static metrics at `t = 0`:* FPR > 0.
- *Distinguishing information:* FPR **rises** as the solve share grows; `J_V'/J_G'` is larger
  than for any exploit (F4); gold rises fast.

**D — displacement (preference inversion).**

- *V:* hack acceptance `rho ~ U(0.2, 0.9)`. The inversion sits on the prompts where
  `p_x < rho`.
- *Dynamics:* the shared `h` rises and displaces SOLVE on hard **and** easy prompts.
- *Signature:* `alpha_t` falls through −1 at the gold peak (Prop. 1: `J_G' = (1+alpha)A^2`); part
  of the residual lies inside the gold span, because `h` affects gold.
- *Long-run outcome:* DECLINE or STALL under both optimizers (F6).
- *Static metrics at `t = 0`:* moderate FPR.
- *Distinguishing information:* J_G rises and then falls (latent); FPR rises; `alpha` trends
  down before `J_G'` turns negative [to be tested].

**MIX (secondary).** Two channels drawn from the above. The label is the **counterfactual
attribution** of §5.4.

**CLEAN twin (not a family).** For every structure, the same `pi_0` trained on `V = G` defines the
counterfactual `J_G^clean(t)`.

### 3.4 Panel size [proposal]

Six families plus MIX, per optimizer:

| split | structures per family | total |
| --- | --- | --- |
| design | 80 | 560 |
| test | 140 | 980 |
| shift | 50 | 350 |

About 1,900 structures, 4 seeds each in the sampled tier.

The power reference is E002: 600 test structures gave ±0.006 on paired C-index differences. With
140 per family, a per-family macro-F1 has SE ≈ 0.03.

---

## 4. Preventing trivial family recognition

1. **One parameterization for all families (§3).** There is no dimension or feature-count leak.
2. **Overlapping static metrics.** Each structure first draws a target FPR(0)
   (`LogUniform(0.02, 0.4)`) and a target `J_G(0)` (`U(0.05, 0.5)`) from **one** distribution.
   Its family channel strength and `mu` are then calibrated to hit them (brentq, as in E002).
   FNR overlaps through the background coin.
3. **Coupling and background coins in every family (§3.2).** They blur exact-zero signatures,
   e.g. `C_out = 0` exactly.
4. **Within-family outcome variation.** Parameter ranges are chosen so that every family except
   B and R contains both failures and successes. Checked on the design split in stage 0.
5. **Panel validity gate (stage 0, design split, before any predictor).**
   - A logistic classifier on L0 features must predict the family with macro-F1 ≤ 0.35
     (chance ≈ 0.17).
   - The **type oracle** (the within-family mean outcome) is reported as a reference, as in E002
     (0.745 / 0.79).
   - If the gate fails, the ranges are widened and the change is recorded.
6. **No family or type identifier is given to any predictor.** Leave-one-family-out evaluation
   is in §10.
7. **Twin tests (§9).** Two families with near-identical early `J_G`, `J_V`, FPR but different
   geometry and/or fate.
   - Feasible for B vs a coupled exploit (F7) and for the latent-decline pair (F8).
   - **Impossible** for C-type pairs without coupling (F1): matching `(J_G, FPR)` trajectories
     forces equal `(A, alpha, C)`.
   - **Impossible** for B vs any C-type exploit on `(J_G, FPR, J_G', J_V')` (F4).

---

## 5. Outcomes (fixed before any predictor)

### 5.1 Horizon `T`

For each optimizer, `T = 3 × median(t95^clean)` over the design split. Here `t95^clean` is the
time at which the CLEAN twin reaches 95% of its gain.

- DC4: `t95^clean ≈ 4.9` (NG) and 5.6 (Adam) time units, so `T ≈ 15–17` time units, i.e.
  ~1,600 Adam steps at `lr = 0.01`.
- Rationale: the budget lets clean training converge with margin, which is how practical RL
  budgets are set. This keeps "fraction of `T`" meaningful and transferable.

### 5.2 Targets

- **Primary continuous:** normalized shortfall
  `Dn = (J_G^clean(T) - J_G(T)) / (J_G^clean(T) - J_G(0))`. Structures with clean gain < 0.1 are
  rejected at generation.
- **Secondary continuous:** the absolute shortfall, and `J_G(T)`.
- **Categorical** (in this precedence order):
  1. **DECLINE**: `max_{t<=T} J_G(t) - J_G(T) >= 0.05` and `Dn > 0.1`. Peak-then-decline is the
     over-optimization signature.
  2. **SUCCESS**: `Dn <= 0.1` (at least 90% of the clean gain).
  3. **STALL**: `Dn > 0.1` and `2 (J_G(T) - J_G(T/2)) < J_G^clean(T) - J_G(T)`. At the late rate,
     the shortfall would not close within another full budget `T`.
  4. **SLOW**: otherwise (behind clean, but still closing at a rate that would finish within one
     more budget).
- `J_G(t)` is the **exact** gold value of the realized policy (known in the toy), so the labels
  carry no evaluation noise.
- Sensitivity: thresholds 0.05 and 0.2 are reported as secondary.
- A three-class version (SLOW merged into STALL) is reported for continuity with the brief.

### 5.3 Visible onset and lead time

- **Onset** `t_on` = the first `t` with
  `(J_G^clean(t) - J_G(t)) / (J_G^clean(t) - J_G(0)) > 0.1`, i.e. the run visibly falls behind
  clean.
- A prediction at horizon `h` is a **warning** only for runs with `t_on > h`.
- **Lead time** = `t_on - t_warn`, where `t_warn` is the first grid horizon at which the fitted
  predictor's failure probability exceeds a threshold. The threshold is fixed on the design split
  at a 10% false-alarm rate among SUCCESS runs.
- `J_G^clean` is used only to define targets; it is never a predictor input.

### 5.4 Mechanism label (separate from the outcome)

- **Pure structures:** the label is the generating family. It is kept even when the outcome is
  SUCCESS: mechanism describes the pressure, outcome describes the fate.
- **MIX structures:** counterfactual attribution. `Dn` is recomputed with each channel removed;
  the label is the channel whose removal recovers ≥ 50% of `Dn`, otherwise "MIXED" (excluded
  from primary mechanism metrics).
- **Route A vs Route B** is structural: an AND-type trigger vs a single/OR trigger with a
  difficulty spread. It is **not** defined from `eta`, `A` or `C`, so L3 is not favoured by
  definition.
- *Caveat:* this is still a constructional label (§14.3).

---

## 6. Early-horizon schedule

- `h / T ∈ {0, 0.2%, 0.5%, 1%, 2%, 5%, 10%}` (Adam at `T = 1600`: 0, 3, 8, 16, 32, 80, 160
  steps).
- **Primary horizon `h* = 2%`**; 1% and 5% are pre-specified secondary horizons.
- Every horizon is also reported as `h / t95^clean` and against `t_on` (Fig. 1b).
- No horizon is chosen after the results.

---

## 7. Tiers, optimizers, measurement

- **Optimizers (crossed factor) [open: confirm].**
  - **Adam, GRPO-lite (primary for transfer):** 8 prompts × 8 responses per step; group-mean
    baseline and group-std normalization; Adam (β1 = 0.9, β2 = 0.999, lr = 0.01).
  - **NG (theory anchor):** estimated damped Fisher, the same batch.
- **Oracle tier.** Exact expected dynamics: the NG flow, and the **mean-field Adam** flow of F5
  (per-group std normalization included). It gives noise-free observables and ceilings, and is
  computed and committed before the sampled tier (as E002's oracle ceilings were).
- **Sampled tier (primary).** Finite-batch stochastic runs.
  - The observables at each checkpoint come from an **audit** of 256 fresh rollouts with gold
    labels, identical for every information level.
  - `J_V` comes from the training batches (free).
  - Geometry is estimated from the same audit plus the training rollouts, with the
    plug-in / pooled estimator that won E002 tuning. The metric is the optimizer's current
    preconditioner: `diag(1/(sqrt(v_hat)+eps))` for Adam (a working approximation, §1 of the
    theory note), the estimated Fisher for NG.
  - Registered check: under each optimizer, the Prop. 1 rates computed in that metric match the
    realized expected per-step `ΔJ_G`.
- **Cost reporting.** Gold labels and backward passes per level. Levels are **not** budget-matched
  (E004 is about information depth; E002 covered efficiency), but costs are shown.

---

## 8. Predictor arms

Features at horizon `h`, each summarized by three quantities: the value at `h`, the change
`0 → h`, and the mean slope over `[h/2, h]`.

| level | inputs | #features |
| --- | --- | --- |
| L0 | `J_G(0), J_V(0), FPR(0), FNR(0), FP mass` | 5 |
| L1 | L0 + `A_0, alpha_0, C_0` (optimizer metric) | 8 |
| L2 | L0 + summaries of `J_G, J_V, FPR, FNR` | 17 |
| L2-G | L0 + summaries of `J_G` only (answers §14.1) | 8 |
| L3 | L2 + summaries of `A, alpha, C` | 26 |
| L2+ | L2 + per-prompt `ΔJ_G,x` (secondary; fairness partner of L3+) | 21 |
| L3+ | L3 + summaries of `C_in, C_out` (per-prompt gold span; secondary) | 32 |
| L2k | L2 + `KL(pi_h‖pi_0)`, entropy (optional policy statistics; secondary) | 19 |

**Model hierarchy** (identical for every level):

1. **Single variable, no fitting.** One pre-registered diagnostic per level with a fixed sign:
   - L0: FPR(0);
   - L1: `C_0/A_0`;
   - L2: `ΔFPR(h)` and `-ΔJ_G(h)`;
   - L3: `ΔC(h)` and `-Δalpha(h)`.
2. **Primary fitted model.** Ridge / L2-logistic / multinomial-logistic on standardized
   features, penalty chosen by 5-fold CV grouped by structure, **design split only**.
3. **Secondary nonlinear check.** Gradient boosting with depth 2, ≤ 100 trees, fixed learning
   rate 0.1.

No neural predictors. Nested feature sets with a CV-tuned penalty are the complexity control.
L3's advantage must survive the penalty. A dimension-matched check (L2 augmented with 9
permuted-noise features) guards against "more columns".

---

## 9. Hard pairs (constructed before any predictor; frozen as a separate stress panel)

**Protocol.**

- For each pair type, minimize the L2-trajectory mismatch over `[0, h*]` (squared differences of
  `J_G, J_V, FPR` at the checkpoints, in audit-SE units). The optimization is over the design
  generator's parameters, from 20 random restarts, under the oracle tier of each optimizer.
- Accept a pair if every checkpoint mismatch is ≤ 0.5 audit-SE.
- Record the mechanisms, the geometry differences and **both outcomes**; they are not selected
  on the outcome.
- The pairs are frozen before any predictor is fit and evaluated only after predictors are
  frozen.

| pair | construction | status (design checks) |
| --- | --- | --- |
| HP-A: same early `ΔJ_G, ΔJ_V`, different `C` trajectory | B vs coupled AND2 exploit | **Feasible** (F7): mismatch ≤ 0.004; `alpha` +0.45 vs −0.37; `C_out` 0 vs 0.4. Same outcome (SUCCESS), so a *mechanism* pair only |
| HP-B: same early FPR growth, different `A` decay | Route A vs Route B | **Impossible without coupling** (F1). With coupling under Adam: to be searched in stage 0 |
| HP-C: benign vs exploit proxy-gap growth | B vs C-type exploit | **Constrained** (F4): at matched `(J_G, FPR)` B has the *larger* proxy/gold slope ratio. Used as a stress test of "proxy growth ⇒ danger" rules |
| HP-D: latent decline vs benign success | D (preference inversion) vs B, matched early | **Feasible** (F8): mismatch ≤ 0.002; `alpha` −0.26 → −0.40 vs +0.03; DECLINE vs SUCCESS. An *outcome* pair on which L2 is blind over 5% of `T` |

---

## 10. Splits and generalization

- **A. Random held-out structures (primary).** Design/test split within each family, sealed
  loader, approval file (the E002 mechanism).
- **B. Leave-one-family-out (outcome tasks only).** Fit on five families' design structures and
  evaluate on the held-out family's test structures.
  - Mechanism classification cannot be leave-one-family-out (an unseen class). It uses
    **leave-one-parameter-region-out** within families instead: e.g. train on
    `S_E0 < median`, test above.
- **C. Parameter shift.** A separate sealed "shift" panel with harder difficulty
  (`J_G(0) ∈ [0.01, 0.05]`, the P-rare analogue) and smaller batch (4×4). Predictors are fitted
  on the main design split only.
- **D. Optimizer transfer (secondary, links to RQ4).** Fit on NG runs, evaluate on Adam runs, and
  the reverse.

Test and shift outcomes are never computed before approval. All tuning uses design data only.

---

## 11. Statistical criteria (no point-estimate dominance)

**Inference.**

- Hierarchical bootstrap: structures, then seeds, 2000 resamples.
- Paired differences between arms use shared resamples.
- One-sided `p = (1 + #{Δ* <= 0}) / (B + 1)`; Holm within each family of tests.
- **Tie tolerance** `±0.01`. A **"beats"** claim needs the lower 95% bound > 0 **and** a point
  estimate ≥ the margin.
- Margins: `δ_out = 0.02` (C-index / AUROC) and `δ_mech = 0.03` (macro-F1).
- **No frontier rule on point estimates** (the E002 lesson).

**Outcome-prediction success (RQ1, primary; sampled Adam, split A, `h* = 2%`).**
`L2 − L0 >= δ_out` with lower bound > 0 for both:

- the C-index of `Dn`;
- the AUROC of failure (STALL ∪ DECLINE vs SUCCESS ∪ SLOW).

Holm over the 2 tests.

**Early-warning success (primary).** At `h*`, among runs with `t_on > h*`:

- the L2 AUROC for eventual failure has lower bound ≥ 0.65; **and**
- the median lead time at the 10%-false-alarm threshold is ≥ 5% of `T`.

**Generalization.**

- Split C: `L2 − L0` lower bound > `−0.01` (non-inferior).
- Split B: `L2 − L0` lower bound > 0 in ≥ 4 of 6 held-out families.

**Mechanistic-value success (RQ2/RQ3).** At `h*`, in the sampled Adam tier, all of:

- (i) L3 − L2 macro-F1 ≥ `δ_mech`, lower bound > 0;
- (ii) Route A vs Route B AUROC: L3 − L2 ≥ `δ_out`, lower bound > 0;
- (iii) on the frozen hard pairs, L3 classifies ≥ 80% of HP-A/HP-D instances correctly where L2
  is at chance (one-sided binomial test).

L3 is **not** required to beat L2 on outcome prediction.

**Pre-registered null predictions (checks of theory, not of predictors).**

- Oracle tier, NG, C-type structures with `lam = 0`: L3 − L2 = 0 exactly (F1).
- Oracle tier, NG: the coupled and uncoupled twins have identical trajectories (F2).

**Abandonment and scope rules.**

- **Abandon the early-warning contribution** if, at every `h <= 5%`, either RQ1 fails or the
  not-yet-visible AUROC lower bound is ≤ 0.55 (detection only, no warning).
- **Mechanistic-only** if early warning fails and mechanistic value succeeds.
- **Proceed to E004b** if RQ1, early warning and generalization all succeed in the sampled Adam
  tier.

---

## 12. Core figures

1. **Prediction quality vs fraction of `T` observed.**
   - Lines: L0 (flat), L1 (flat), L2, L2-G, L3.
   - Oracle-tier ceilings dashed; one panel per optimizer; bootstrap bands.
   - **1b:** the same, restricted to not-yet-visible runs, plus the lead-time distribution.
2. **Mechanism macro-F1 vs horizon.** L2, L3, L2+, L3+; an inset for Route A vs Route B AUROC.
3. **Representative trajectories per family.** `J_G, J_V, FPR, A, alpha, C, C_out`, NG vs Adam
   overlaid.
4. **Hard pairs.** Matched early observables, divergent geometry and fate.
5. **Leave-one-family-out heatmap.** Held-out family × level.
6. **(New) Optimizer dependence.** Per-family outcome under NG vs Adam, and shortfall vs batch
   size (F5 at scale).

---

## 13. Toy → small-transformer RLVR mapping

| E004a quantity | E004b (GRPO, small transformer) | cost | toy-only exactness |
| --- | --- | --- | --- |
| `J_V` | mean training reward | free | — |
| `J_G` | gold accuracy on a fixed eval set (hidden tests / exact answers) | gold calls per checkpoint | exact in toy; sampled in E004b |
| FPR / FNR | gold audit of verifier-accepted / rejected samples | ~256 gold labels per checkpoint | — |
| per-prompt `J_G` (L2+) | per-prompt eval accuracy | same audit, more noise | — |
| `A, alpha, C` | two REINFORCE gradients (gold and verifier rewards) on the audit batch; inner products in the Adam `v_hat` metric; split-half for unbiased Gram terms | 2–4 backward passes + 2–4 gradient buffers per checkpoint | **the Adam metric is approximate** (momentum, `v_hat`-noise correlation) |
| `C_in / C_out` | residual projected on the span of per-prompt gold gradients: needs per-prompt gradients or random-projection sketches | high (≈ #prompts backward passes, or a sketch) | **toy-dependent meaning:** in `d ≫ #prompts` the span is small and `C_out ≈ C` |
| mechanism labels | only for **planted** verifier flaws | — | **toy-only** for naturally occurring flaws |
| clean counterfactual | a full run with the gold reward | 1 extra run per verifier | cheap in toy |
| `KL`, entropy | standard GRPO logging | free | — |

**E004b sketch (documentation only).**

- **Setup.** A 10–100M decoder on arithmetic and short code tasks with exact gold. GRPO
  (group 8), Adam.
- **Planted verifiers:**
  - R: random reward flips;
  - X: the verifier accepts everything on a prompt subset;
  - Y-A: accepts wrong answers containing a rare formatting token pattern the model can discover;
  - Y-B: accepts a common pattern, with a difficulty-heterogeneous prompt mix;
  - B: partial credit for correct intermediate steps;
  - D: a shortcut answer accepted with probability above the model's solve rate on hard prompts.
- **Protocol.** The same checkpoints, levels, outcome rules and lead-time endpoint.

E004b is not implemented.

---

## 14. Arguments against this design (not protecting the hypothesis)

1. **Could early `J_G` alone predict final `J_G`?** Largely yes in smooth families. E002's oracle
   "gold progress at `t_p = 10`" reached C-index 0.98 (`t_p = 10` is under 2× the clean `t95`),
   and FPR growth at `t_p = 0.1` (about 2% of the clean `t95`) already reached 0.936. Hence the
   L2-G arm and the not-yet-visible endpoint. Where `J_G` alone should fail: D (gold rises first;
   F8 shows that even `J_G`, `J_V` and FPR together can be blind) and SLOW vs STALL.
2. **Could FPR growth alone make geometry unnecessary?** For prediction, probably: F1 makes them
   equivalent in the classic families. But FPR growth is **anti-predictive** for B (F4), rises by
   composition in X, and under Adam often accompanies a successful run (F5). FPR growth alone is
   insufficient; whether *geometry* is needed beyond `J_G` and FPR jointly is exactly RQ3, and the
   honest prior is "not for outcome".
3. **Are mechanism labels artificial?** Yes, partly. They are our channels. Counterfactual
   attribution makes them outcome-relevant but still constructional. In real models, mechanisms
   exist only for planted flaws or through post-hoc audits.
4. **Could L3 look useful because it encodes the generator?** Yes, the largest validity risk.
   - `C_out` is by construction the component along directions no gold gradient touches, and
     `alpha > 0` is by construction the attempt credit.
   - Mitigations: coupling and background coins in every family (they blur exact zeros; F2 shows
     coupled exploits have `alpha > 0` like B); estimated, not exact, geometry; pre-registered
     signatures; hard pairs; and L2+ as the fairness partner of L3+.
   - Residual risk: in the toy, per-prompt gold gradients span a small subspace (`K = 4`,
     `d = 8`), so `C_in`/`C_out` is cleaner than it can be in an LLM.
5. **Is leave-one-family-out enough?** No. The held-out family shares the generator, policy class
   and optimizer, and the families are our taxonomy. Stronger tests are optimizer transfer
   (split D) and E004b. Leave-one-family-out shows interpolation across our taxonomy, not
   generality.
6. **Result that abandons early warning:** at every `h <= 5%`, no RQ1 gain over L0, or
   not-yet-visible AUROC ≤ 0.55 (only detection); or success on split A that collapses on B and C.
7. **Result that supports only the mechanistic contribution:** L2 ≈ L3 for outcome and no lead
   time, but L3 > L2 on mechanism (especially Route A vs B and the hard pairs). Also the
   pre-registered optimizer-dependence results (F5 at scale), which are mechanistic findings in
   their own right.
8. **Result that justifies E004b / GRPO:** early warning with positive lead time **in the sampled
   Adam tier**, surviving splits B and C; and observables measurable at the §13 costs.
9. **(Added) Is the toy's Adam a good proxy for GRPO?** F5/F6 show the optimizer changes which
   verifier flaws matter. The GRPO-lite tier includes group-std normalization, but a real
   transformer's shared representations and the clipping of the policy ratio are absent. The
   stronger conclusion may be "mechanistic claims from NG toys (E001–E003) do not transfer to
   Adam without SNR", which is worth knowing before E004b.

---

## 15. Recommendation

**IMPLEMENT E004a, staged.**

- **Stage 0** (design split only; no predictors):
  - build the panel;
  - validity gates (§4.5);
  - oracle tier for both optimizers;
  - hard-pair search and freeze (§9);
  - optimizer-dependence map (Fig. 6).
- **Gate to stage 1.** The panel passes the family-recognition gate; every family except B and
  R has both failures and successes under the primary optimizer; the pre-registered F1/F2 nulls
  hold. Otherwise REVISE.
- **Stage 1:** sampled tier, predictors fitted on design, then the pre-registered held-out
  evaluation after explicit approval.

**Why not skip to the transformer.**

1. Mechanism ground truth, clean counterfactuals and exact onsets exist only in the synthetic
   setting, and RQ2 needs them.
2. F5/F6 show the optimizer decides which planted flaws even fail. E004b's planted flaws should be
   chosen from the stage-0 map, not guessed.
3. The cost is small: about an hour of compute for the oracle tier and under an hour for the
   sampled tier on this machine (estimate).

**Why not only REVISE.** The brief's structure is sound. The changes above are corrections
inside it: four outcome classes, the warning endpoint, crossed optimizers, coupling, D, and
feasible hard pairs.

---

## 16. Decisions needed from the collaborator

1. **Four outcome classes** (SLOW added) with the §5.2 thresholds, or the three-class version
   as primary.
2. **Adam primary / NG secondary** as a crossed factor, including mean-field Adam in the oracle
   tier.
3. **Coupling and background coins in every family** (breaks F1's trivial equivalence; blurs
   exact signatures).
4. **Family D (preference inversion)** added as the optimizer-robust DECLINE mechanism.
5. **Primary early-warning endpoint** = not-yet-visible AUROC + lead time (vs "quality vs
   fraction observed").

Optional: RQ4 (counterfactual optimizer prediction); the L2+/L3+ per-prompt arms.
