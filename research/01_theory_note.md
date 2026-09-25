# 01 — Theory note (v0.2, 2026-09-25)

## Version history

- **v0.1** (2026-09-24; last state at commit `e7f5144`): Phase 1A derivations for the Y, R and X
  toys.
- **v0.2** (2026-09-25), after E001:
  1. §1 rewritten as *optimizer-conditioned geometry* with a time-varying preconditioner `M_t`,
     the explicit derivation, an `A = 0` convention and the Adam caveat.
  2. New §2 on semantics. Gap growth (`dDelta/dt > 0`) is no longer treated as hacking or as
     failure onset.
  3. §3.4 interpretation replaced. The v0.1 text read: *"In this toy, `C > 0` together with a
     shrinking gap (`rho < 0`) is a terminal residual regime, not a precursor of gap growth. The
     proposal's H5 latent-phase signature cannot be tested here."* It framed gap growth as the
     failure signal, which §2 retracts.
  4. New §6 (E001 observations), §7 (revised working hypotheses) and updated §8 (open issues).

The pre-registered hypotheses and predictions of E001 in `03_experiment_registry.md` are **not**
edited. Revisions are recorded here (§7) and in the E001 post-run addendum.

Status tags:

- **[given]**: stated in the project brief or proposal v2 (2026-09-21).
- **[proved]**: complete short proof written here. Agent-written; needs collaborator review.
- **[derived-agent]**: derived by the coding agent and checked numerically; no separate proof.
- **[E001-observed]**: empirical observation from E001 (see the registry for numbers and caveats).
- **[working hypothesis]** / **[conjecture]**: not established.

Each verified claim names its tests (§9). Anything not in the test map is not verified in-repo.

---

## 0. Notation

- Proposal `(a, b, c)` and brief `(A, alpha, C)` are the same triple: `a = A`, `b = alpha A`,
  `c = C`. Code stores `A, alpha, b, C`.
- The Y-toy correctness bit (`c` in the brief) is `corr` in code, to avoid clashing with `c = C`.
- `Delta = J_V - J_G` is the signed proxy–gold discrepancy.

## 1. Optimizer-conditioned local geometry [proved]

**Setting.** A locally preconditioned flow

    dtheta/dt = M_t g_V(theta_t),

where `M_t` is symmetric positive semidefinite and may depend on `t` and `theta_t` (not on the
future). Let `M_t^{1/2}` be its PSD square root and define

    h_G = M_t^{1/2} g_G,    h_e = M_t^{1/2} g_e,    h_V = h_G + h_e = M_t^{1/2} g_V.

If `A = ||h_G|| > 0`, decompose

    h_e = alpha h_G + r,   <r, h_G> = 0,   alpha = <h_e, h_G> / A^2,   C = ||r||,   b = alpha A.

**Convention for `A = 0`** (e.g. `q in {0, 1}`, or `M_t` annihilates `g_G`): set `alpha := 0`,
`r := h_e`, `C := ||h_e||`. The identities below then still hold. *The current code divides by
`A^2` and returns NaN here; changing that is pending approval (§8).*

**Proposition 1 (v0.2 form).**

    dJ_G/dt   = (1 + alpha) A^2
    dJ_V/dt   = (1 + alpha)^2 A^2 + C^2
    dDelta/dt = alpha (1 + alpha) A^2 + C^2      (= b(a + b) + c^2)

*Derivation.*

1. Chain rule: `dJ_G/dt = g_G^T (dtheta/dt) = g_G^T M_t g_V`.
2. With the symmetric root, `g_G^T M_t g_V = (M_t^{1/2} g_G)^T (M_t^{1/2} g_V) = <h_G, h_V>`.
3. `h_V = h_G + h_e = (1 + alpha) h_G + r`, so
   `<h_G, h_V> = (1 + alpha) ||h_G||^2 + <h_G, r> = (1 + alpha) A^2`.
4. Likewise `dJ_V/dt = g_V^T M_t g_V = ||h_V||^2 = (1 + alpha)^2 A^2 + 2(1 + alpha)<h_G, r> + ||r||^2
   = (1 + alpha)^2 A^2 + C^2`.
5. `dDelta/dt = dJ_V/dt - dJ_G/dt = (1 + alpha)A^2 [(1 + alpha) - 1] + C^2 = alpha(1 + alpha)A^2 + C^2`. ∎

Remarks.

- The identities are instantaneous, so `M_t` may vary along the trajectory.
- `M_t` PSD implies `dJ_V/dt >= 0`: the flow always ascends the proxy.
- For discrete steps `theta_{k+1} = theta_k + eta M_k g_V` they hold to first order in `eta`;
  second-order (curvature) terms are not studied here.
- With stochastic gradients `dtheta/dt = M_t g_hat_V`, the expected rates equal the formulas only
  if `M_t` is independent of the noise in `g_hat_V`.

**Special cases.** Natural gradient is `M = F^{-1}`: Fisher whitening is this one case, not the
general definition. Vanilla gradient ascent is `M = I`. Prop. 2 (parameterization invariance)
holds for `M = F^{-1}`, not for `M = I`.

**Corollary (metric mismatch) [derived-agent; E001-observed].** If `(A, alpha, C)` are computed
in a metric `M'` different from the optimizer's `M_t`, the identities fail, including signs. Y-toy
example: at `(q, s) = (0.3, 0.05)` the Fisher-metric triple gives `dDelta/dt = +1.33e-2`, while
the vanilla flow has `-9.89e-4`.

**Adam is not exactly of this form.** An Adam step is `-eta m_hat_t / (sqrt(v_hat_t) + eps)`. This
is not `M_t g_V(theta_t)` for three reasons:

- (a) the momentum `m_hat_t` averages gradients evaluated at past parameters `theta_{t-k}`;
- (b) `v_hat_t` contains the current stochastic gradient, so `M_t` is correlated with the
  gradient noise;
- (c) `eps` and the bias corrections.

Without momentum and with slowly varying `v_hat`, `M_t ≈ diag(1/(sqrt(v_hat_t) + eps))` is a
plausible local approximation **[working hypothesis, untested]**. Treating `v_hat` as a Fisher
estimate and using `diag(v_hat)^{-1}` gives the natural-gradient geometry, not Adam's (which
scales as `v_hat^{-1/2}`).

**Proposition 2 (parameterization invariance) [proved].** Under a smooth invertible
reparameterization with Jacobian `J`, `g -> J^T g` and `F -> J^T F J`, so `g^T F^{-1} g'` and the
Fisher-metric `(A, alpha, C)` are invariant. Euclidean-metric values are not invariant in general.

## 2. Semantics of the diagnostics (new in v0.2)

Current interpretation:

| Quantity | Meaning |
| --- | --- |
| `alpha` | parallel distortion of the true direction, in the optimizer's metric |
| `C` | orthogonal verifier-induced pressure |
| `dJ_G/dt = (1 + alpha) A^2` | local gold improvement |
| `dDelta/dt = alpha(1 + alpha) A^2 + C^2` | signed proxy–gold discrepancy dynamics |

**Gap growth is not reward hacking and not failure onset.** `dDelta/dt > 0` means only that the
proxy is rising faster than gold (or falling slower). It is neither sufficient nor necessary for
a gold-learning failure.

- *Not sufficient: amplification.* Take `alpha > 0`, `C = 0` (e.g. `V = kappa G` with
  `kappa > 1`). Verifier and gold gradients are perfectly aligned and
  `dJ_G/dt = (1 + alpha) A^2 > 0`, yet `dDelta/dt = alpha(1 + alpha) A^2 > 0`. A second example is
  the Y toy under natural gradient with `q < 1/2`: the gap grows while gold improves (E001 IC1
  early phase).
- *Not necessary: deletion.* X (`alpha = -1`, `C = 0`) gives `dJ_G/dt = 0` **and**
  `dDelta/dt = 0`. Gold learning stops on the affected prompts with no gap growth at all. R with
  `p = 1/2` behaves the same way.

**`C > 0` is not hacking either.**

- X aggregated over prompts has `C > 0` with no exploitable direction (Prop. 6).
- In the Y toy, `C` returns toward 0 once the exploit (or gold) saturates (E001, §6).

**The object to predict** is therefore a gold-learning failure (stall or decline, §7 WH-1), not
gap growth.

## 3. Y toy: policy-controllable false positive

Policy: `corr ~ Bern(q)`, `z ~ Bern(s)` independent, `q = sigmoid(u)`, `s = sigmoid(v)`.
Gold `G = corr`. Verifier `V = corr + (1 - corr) z = corr OR z`.

**Observation [derived-agent].** `J_V = 1 - (1 - q)(1 - s)` is symmetric in `(q, s)`: under `V`
the optimizer cannot tell correctness from the exploit. Both flows below are symmetric under
`(u, v) -> (v, u)`. E001 reproduced this mirror symmetry to machine precision.

### 3.1 Static quantities [given]

    J_G = q,    J_V = q + (1 - q) s,    Delta = (1 - q) s
    g_G = [q(1-q), 0]
    g_V = [q(1-q)(1-s), (1-q)s(1-s)]
    g_e = [-s q(1-q), (1-q)s(1-s)]
    F   = diag(q(1-q), s(1-s))
    h_G = [sqrt(q(1-q)), 0]
    h_e = [-s sqrt(q(1-q)), (1-q) sqrt(s(1-s))]
    A = sqrt(q(1-q)),  alpha = -s,  r = [0, (1-q) sqrt(s(1-s))],  C = (1-q) sqrt(s(1-s))

`C > 0` iff `q < 1` and `0 < s < 1`.

Euclidean metric **[derived-agent]**: `A_E = q(1-q)`, `alpha_E = -s`, `C_E = (1-q) s (1-s)`.
Near exploit onset (`s -> 0`) `C_F ~ sqrt(s)` but `C_E ~ s`.

**Information content [derived-agent].** The on-policy static verifier metrics are
FPR `= P(V=1 | G=0) = s` and FNR `= 0`; the policy's gold accuracy is `q`. The triple is a function
of these: `A = sqrt(q(1-q))`, `alpha = -FPR`, `C = (1-q) sqrt(FPR (1-FPR))`. So in this toy the
Fisher diagnostics carry **no information beyond on-policy static metrics**.

### 3.2 Natural-gradient flow

`u_dot = 1 - s`, `v_dot = 1 - q` and **[given]**

    q_dot = q(1-q)(1-s),   s_dot = s(1-s)(1-q),   s/q = s0/q0,
    dDelta/dt = s(1-s)(1-q)(1-2q).

**Proposition 3 (NG endpoint) [proved].** With `k = s0/q0`, the path is the ray `s = kq`. It
converges to `(1, k)` if `k <= 1` and to `(1/k, 1)` if `k > 1`. So `Delta_inf = max(0, 1 - q0/s0)`,
and gold is capped at `q0/s0 < 1` when `s0 > q0`.

*Proof.* `q_dot, s_dot >= 0` and bounded, so `(q, s)` converges. A limit with `q >= q0 > 0`,
`s >= s0 > 0` must have `q_dot = s_dot = 0`, i.e. `q = 1` or `s = 1`. Intersect with the ray. ∎

**Corollary [derived-agent].** `alpha(t) = -s(t)`. When `s -> 1`, `alpha -> -1` and `C -> 0`:
on-policy `V ≡ 1`, so **the Y verifier turns itself into a deletion (X-like) verifier**. This
mechanism produces the late gold stall: early `C > 0`, `dJ_G/dt > 0`; late `dJ_G/dt -> 0` with
`q -> q0/s0 < 1`.

### 3.3 Vanilla gradient flow

`u_dot = q(1-q)(1-s)`, `v_dot = (1-q)s(1-s)`:

    q_dot = [q(1-q)]^2 (1-s)          [derived-agent]
    s_dot = (1-q) [s(1-s)]^2          [given]
    dDelta/dt = (1-q)^2 s(1-s) [s(1-s) - q^2]      [derived-agent]

**Proposition 4 (vanilla invariant) [proved].** With `H(x) = x - e^{-x}`, `H(v) - H(u)` is
conserved. *Proof.* `d/dt H(v) = v_dot / s = (1-q)(1-s) = u_dot / q = d/dt H(u)`. ∎

Formally both `q, s -> 1`, but the saturating logit grows like `log t`. At practical horizons the
other coordinate is effectively frozen (E001 IC3: `q(1e6) = 0.0120` vs natural-gradient `0.0333`).

**Clock-free comparison [derived-agent].**

    d log s / d log q = 1                        (natural)
    d log s / d log q = s(1-s) / (q(1-q))        (vanilla)

Vanilla suppresses exploit growth per unit of gold progress only where `s(1-s) < q(1-q)`.

### 3.4 What the single-prompt OR toy can and cannot test (revised in v0.2)

**Proposition 5 [proved; needs review].** Under either flow, `sign(dDelta/dt)` can change only from
`+` to `-`. *Natural:* the sign is `sign(1 - 2q)` and `q` is non-decreasing. *Vanilla:* the sign is
`sign(f)` with `f = s(1-s) - q^2`. On `{f = 0}`, `q <= 1/2` and
`df/dt = (1-q) q^3 [(1-2s) q - 2(1-q)(1-s)] < 0`, so `f` only crosses downward. ∎

After §2 this is a statement about the **sign of the proxy–gold discrepancy only**; it says nothing
directly about gold failure. Revised assessment:

- **It does produce a late gold stall.** Under natural gradient with `s0 > q0`, `dJ_G/dt -> 0`
  while `q -> q0/s0 < 1` (Prop. 3; Y turns into X). Under vanilla the same happens at any
  practical horizon.
- **It cannot produce gold decline**: `q_dot >= 0` under both flows.
- **It cannot test prediction beyond static metrics.** The triple is a function of on-policy
  static metrics (§3.1). The outcome (stall iff `s0 > q0` under natural gradient) is already fixed
  by those metrics at `t = 0`.

So the full *early-warning → late-gold-failure beyond static metrics* hypothesis is not testable in
this toy. The reason is not an absence of late failure; the diagnostics simply add no information.

## 4. R: random symmetric flips

With a **fresh**, response-independent flip `xi ~ Bern(p)`: `E[V | y] = p + (1-2p) G`, so
**[given]** `g_V = (1-2p) g_G`, `alpha = -2p`, `C = 0` in any metric. `p = 1/2` is exactly
deletion; `p > 1/2` reverses the signal.

Semantics (v0.2): for `0 < p < 1/2`, `dDelta/dt = -2p(1-2p) A^2 < 0`. The gap shrinks while gold
learning is slowed by the factor `1 - 2p`: a slowdown with no gap growth.

**Assumption boundary [derived-agent].** A *fixed* flip set on a finite response space is
policy-controllable; flipping outcome `(0, 1)` reproduces Y. Refinement (tests, 2026-09-24): this
holds generically, not at every policy. The XOR set `{(0,1), (1,1)}` has `alpha = -2s` and
`C = |1 - 2q| sqrt(s(1-s))`, which is 0 at `q = 1/2`.
*Matched-accuracy illustration [derived-agent]:* a fixed flip of `(0,1)` and R with `p = (1-q)s`
have the same on-policy error rate, but `C = (1-q)sqrt(s(1-s))` versus `C = 0`.

Not tested yet: under step-normalized optimizers the factor `(1 - 2p)` is largely normalized away,
and R acts mainly through estimator variance.

## 5. X: signal deletion

Per prompt: `g_V(x) = 0`, `alpha = -1`, `C = 0` in any metric **[given]**. Semantics (v0.2): gold
learning on the subset stops (`dJ_G/dt = 0`) with `dDelta/dt = 0`. This is a gold failure that
the gap does not show.

**Proposition 6 (aggregate X) [proved].** Tabular prompts, block-diagonal metric:

    alpha_agg = -A_S^2 / (A_S^2 + A_N^2),     C_agg = sqrt(A_S^2 A_N^2 / (A_S^2 + A_N^2)) > 0.

So `C_agg > 0` without any new direction. Proposal (not implemented): split `r` into its
components inside and outside `span{g_G(x)}`.

## 6. E001 empirical observations [E001-observed]

Numbers are from `results/E001/20260925T012753Z_9d4df22/`; details are in the registry.

- **O1 Agreement.** Closed form, autodiff, finite differences and simulation agree:
  - trajectories within 2e-10;
  - decomposition within the pre-registered tolerance (worst normalized error 0.19);
  - Prop. 1 against finite-difference `dDelta/dt`, worst normalized error 2.1e-3.
- **O2 The optimizer changes state-space paths, not only the clock.**
  - IC1: at `q = 1 - 1e-4`, `s = 0.0114` (vanilla) vs `0.0333` (natural).
  - IC3 is the mirror image.
  - IC2 is the exception: both paths lie on the symmetric diagonal and differ only in the clock.
- **O3 Mismatched metric → wrong sign.** Along vanilla trajectories, the Fisher-metric prediction
  had the wrong sign of `dDelta/dt` at 39.7% (IC1), 23.4% (IC3) and 0% (IC2) of evaluation points.
  *Precision:* for `dJ_G/dt` the sign agreed in this toy, because both metrics give values >= 0
  here; only magnitudes differ. A wrong **gold** sign under mismatch appears in the Phase 1B
  candidate 2 design check (`04_phase1b_design.md`), not in E001.
- **O4 `C` is transient, not a terminal score.**
  - `C_F` peaked where `s ≈ 0.25–0.5`, then decayed toward 0:
    - to 8e-6 in IC3/natural, after exploit saturation;
    - to 1e-2 in IC2/natural at `T = 25`;
    - to 1e-11 in IC1/natural, where the decay came from **gold** saturation, not exploit
      saturation.
  - *Precision:* the rise during exploit acquisition was modest. Peak / initial `C_F` was 1.20
    (IC2), 1.08–1.09 (IC3) and ≈ 1.00 (IC1); the Euclidean `C` rose up to 1.83 (IC2). In this toy
    `C` starts near its maximum, so "C becomes large during acquisition" is only weakly supported.
- **O5** The single-prompt OR toy cannot test early-warning → late failure beyond static metrics
  (§3.4).

## 7. Revised working hypotheses (v0.2)

These are recorded separately from the pre-registered hypotheses, which are unchanged. Nothing in
E001 is re-scored.

- **WH-1 (prediction target).** The target is **gold-learning failure**, not gap growth.
  Failure means either:
  - *stall*: `dJ_G/dt -> 0` with `J_G` below the gold-trained counterfactual at a matched horizon
    and optimizer; or
  - *decline*: `dJ_G/dt < 0` after an initial rise.

  This supersedes, for this repository's experiments, the gap-based framing in proposal §3.2 / H5
  (the proposal text itself belongs to the collaborator).
- **WH-2 (metric).** Diagnostics meant to predict an optimizer's trajectory must be computed in
  that optimizer's local metric. For Adam this is only approximate (§1).
- **WH-3 (`C` as an early/cumulative signal).** If `C` is predictive, it will be through early or
  integrated quantities (e.g. `C/A` early, `∫ C^2 dt` over a probe window), not a terminal value.
- **WH-4 (information requirement).** A test of "beyond static metrics" needs a model in which
  on-policy static metrics do not determine the diagnostics or the outcome. The single-prompt OR
  toy fails this (§3.4). Candidates are in `04_phase1b_design.md`.

## 8. Open issues

1. Phase 1B model needed (§3.4) → `04_phase1b_design.md` (design only; awaiting review).
2. Metric matching (WH-2). The proposal's §3.4 (Adam second moments as a Fisher proxy) needs
   revisiting.
3. X aggregate `C_agg > 0` (Prop. 6); span split proposed, not implemented.
4. R's `C = 0` requires fresh, response-independent flips.
5. Naming: `(a,b,c)` vs `(A,alpha,C)`; `corr`; "Monte Carlo Fisher" vs "empirical Fisher".
6. **New:** the `A = 0` convention (§1) is not implemented; `decompose` returns NaN there. This is
   a small code change, pending approval.
7. **New:** the proposal's §3.2 (`rho`, "latent exploitation") and H5 still use gap-growth framing.
   This is the collaborator's decision.

## 9. Test map

| Claim | Tests / evidence |
| --- | --- |
| §1 decomposition and Prop. 1 (constant `M`) | `test_decompose.py`, `test_y_flows.py::test_metric_matched_rate_identities`, `test_y_flows.py::test_fisher_metric_mispredicts_vanilla_gap_sign`; E001 P7 along trajectories |
| §1 `A = 0` convention | not implemented, not tested |
| Prop. 2 | `test_y_static.py::test_fisher_diagnostics_are_parameterization_invariant` |
| §3.1 Y static formulas | `test_y_static.py`, `test_monte_carlo.py`, `test_fisher.py` |
| §3.2–3.3 flows, Props. 3–4 | `test_y_flows.py`; E001 P1–P4 |
| Prop. 5 | `test_y_flows.py::test_gap_sign_never_switches_from_shrinking_to_growing`; E001 P5 |
| §4 R | `test_r.py` |
| §5 X, Prop. 6 | `test_x.py` |
| §6 observations | E001 (`03_experiment_registry.md`): 82/83 registered checks passed; the one failure is a mis-specified absolute tolerance (E001-D1) |
| Phase 1B derivations (`04_phase1b_design.md`) | scratch checks only (E000b); no repo tests yet |
