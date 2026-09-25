# 01 — Theory note (Phase 1A)

Status tags used below:

- **[given]** stated in the project brief or proposal v2 (2026-09-21).
- **[proved]** a complete short proof is written here. Agent-written; still needs collaborator review.
- **[derived-agent]** derived by the coding agent, checked numerically, no separate proof write-up.
- **[conjecture]** / **[empirical prediction]** as usual.

Each claim names the tests that verify it (`tests/<file>::<test>`). Anything not listed in the
test map at the end is not verified.

---

## 0. Notation

- Proposal `(a, b, c)` and brief `(A, alpha, C)` are the same triple: `a = A`, `b = alpha * A`,
  `c = C`. Code stores all four of `A, alpha, b, C`.
- In the Y toy the correctness bit is written `c` in the brief. **In code it is `corr`** to avoid
  the collision with `c = C`.
- Metric: a symmetric positive (semi)definite matrix `M`, inner product `<x, y>_M = x^T M y`.
  Natural gradient uses `M = F^{-1}`; vanilla gradient ascent uses `M = I`. "Fisher-whitened"
  `h = F^{-1/2} g` is the special case `M = F^{-1}`; all scalar diagnostics depend only on inner
  products, so the choice of matrix square root is irrelevant.

## 1. Generic decomposition and the metric-matching identity

Given `g_G`, `g_V`, `M`:

    A     = ||g_G||_M
    alpha = <g_e, g_G>_M / A^2          (g_e = g_V - g_G)
    b     = alpha * A
    r     = g_e - alpha * g_G           (so <r, g_G>_M = 0)
    C     = ||r||_M

**Proposition 1 (rate identities) [proved].** Let parameters follow `theta_dot = M g_V` and let
`(A, alpha, C)` be computed **with the same `M`**. Then

    dJ_G/dt   = A^2 (1 + alpha)          = a (a + b)
    dJ_V/dt   = A^2 (1 + alpha)^2 + C^2  = (a + b)^2 + c^2
    dDelta/dt = A^2 alpha (1 + alpha) + C^2 = b (a + b) + c^2

*Proof.* `g_V = (1 + alpha) g_G + r` with `<r, g_G>_M = 0`. Then
`dJ_G/dt = g_G^T M g_V = <g_G, g_V>_M = (1 + alpha) A^2` and
`dJ_V/dt = <g_V, g_V>_M = (1 + alpha)^2 A^2 + C^2`. Subtract. ∎

**Corollary [derived-agent].** If `(A, alpha, C)` are computed in a metric other than the
optimizer's preconditioner, the identities fail, including the sign of `dDelta/dt`. Example in
the Y toy at `(q, s) = (0.3, 0.05)`: Fisher-metric triple predicts `dDelta/dt = +1.33e-2`; the
vanilla flow gives `-9.89e-4`. Consequence for Phase 2: Adam's effective preconditioner is
`diag(v_hat)^{-1/2}`, not `diag(v_hat)^{-1}`.

**Proposition 2 (parameterization invariance) [proved].** For a smooth invertible
reparameterization `theta = phi(eta)` with Jacobian `J`, gradients map as `g -> J^T g` and the
Fisher as `F -> J^T F J`; hence `g^T F^{-1} g'` is invariant and so are Fisher-metric
`(A, alpha, C)`. Euclidean-metric values are not invariant in general.

## 2. Y toy: policy-controllable false positive

Policy: `corr ~ Bern(q)`, `z ~ Bern(s)` independent, `q = sigmoid(u)`, `s = sigmoid(v)`,
`theta = (u, v)`. Gold `G = corr`. Verifier `V = corr + (1 - corr) z = corr OR z`.

**Observation [derived-agent].** `J_V = 1 - (1 - q)(1 - s)` is symmetric in `(q, s)`. Under `V`
the optimizer cannot distinguish correctness from the exploit; only `G` and the initial condition
break the symmetry. Both flows below are symmetric under `(u, v) -> (v, u)`.

### 2.1 Static quantities [given]

    J_G = q,    J_V = q + (1 - q) s,    Delta = (1 - q) s
    g_G = [q(1-q), 0]
    g_V = [q(1-q)(1-s), (1-q)s(1-s)]
    g_e = [-s q(1-q), (1-q)s(1-s)]
    F   = diag(q(1-q), s(1-s))
    h_G = [sqrt(q(1-q)), 0]
    h_e = [-s sqrt(q(1-q)), (1-q) sqrt(s(1-s))]
    A = sqrt(q(1-q)),  alpha = -s,  r = [0, (1-q) sqrt(s(1-s))],  C = (1-q) sqrt(s(1-s))

`C > 0` iff `q < 1` and `0 < s < 1`.

Euclidean metric `M = I` **[derived-agent]**: `A_E = q(1-q)`, `alpha_E = -s` (coincides because
`g_G` has no `v` component and `F` is diagonal), `C_E = (1-q) s (1-s)`. Near exploit onset
(`s -> 0`) `C_F ~ sqrt(s)` but `C_E ~ s`.

### 2.2 Natural-gradient flow

`theta_dot = F^{-1} g_V` gives `u_dot = 1 - s`, `v_dot = 1 - q` and **[given]**

    q_dot = q(1-q)(1-s),   s_dot = s(1-s)(1-q),   s/q = s0/q0,
    dDelta/dt = s(1-s)(1-q)(1-2q).

**Proposition 3 (NG endpoint) [proved].** With `k = s0/q0`, the trajectory is the ray `s = k q`
and converges to `(1, k)` if `k <= 1` and to `(1/k, 1)` if `k > 1`. Hence
`Delta_inf = max(0, 1 - q0/s0)` and, when `s0 > q0`, gold is capped at `q0/s0 < 1`.

*Proof.* `q_dot, s_dot >= 0` and both are bounded, so `(q, s)` converges. A limit point in the
closed square with `q >= q0 > 0`, `s >= s0 > 0` must satisfy `q_dot = s_dot = 0`, i.e. `q = 1` or
`s = 1`. Intersecting the ray `s = k q` with `{q = 1} ∪ {s = 1}` gives the stated point. ∎

**Corollary [derived-agent].** `alpha(t) = -s(t)`; when `s -> 1` then `alpha -> -1` and
`C -> 0`: on-policy `V ≡ 1`, i.e. **the Y verifier turns itself into a deletion (X-like) verifier**.
R/X/Y are regimes a run can move between, not a static partition.

### 2.3 Vanilla gradient flow

`theta_dot = g_V`: `u_dot = q(1-q)(1-s)`, `v_dot = (1-q)s(1-s)`, so

    q_dot = [q(1-q)]^2 (1-s)          [derived-agent]
    s_dot = (1-q) [s(1-s)]^2          [given]
    dDelta/dt = (1-q)^2 s(1-s) [s(1-s) - q^2]      [derived-agent]

**Proposition 4 (vanilla invariant) [proved].** With `H(x) = x - e^{-x}`,
`I = H(v) - H(u)` is constant along the vanilla flow.
*Proof.* `d/dt H(v) = v_dot (1 + e^{-v}) = v_dot / s = (1-q)(1-s)`, and likewise
`d/dt H(u) = u_dot / q = (1-q)(1-s)`. ∎

Consequence: formally both `q, s -> 1` as `t -> infinity` (if `u -> inf` then `H(v) -> inf`), but
the saturating logit grows like `log t`, so at any practical horizon the non-saturated coordinate is
effectively frozen (e.g. `(q0, s0) = (0.01, 0.30)`: when `s = 1 - 1e-6`, `q = 0.0120`, versus the
NG endpoint `0.0333`).

**Clock-free comparison [derived-agent].** Time is not comparable across optimizers. Along the
path,

    d log s / d log q = 1                          (natural)
    d log s / d log q = s(1-s) / (q(1-q))          (vanilla)

So vanilla suppresses exploit growth *per unit of gold progress* only where `s(1-s) < q(1-q)`; in
the `q0 << s0` regime it is worse than natural gradient. The brief's statement
"vanilla `s_dot ≈ (1-q)s^2` vs natural `(1-q)s`" is correct as calculus but depends on the clock.

### 2.4 No latent-then-growth transition in the single-prompt Y toy

**Proposition 5 [proved; agent-written, needs review].** Under either flow, `sign(dDelta/dt)`
can change only from `+` to `-`, never from `-` to `+`.

*Natural.* On the open square `sign(dDelta/dt) = sign(1 - 2q)` and `q` is non-decreasing.

*Vanilla.* `sign(dDelta/dt) = sign(f)`, `f = s(1-s) - q^2`. On `{f = 0}` we have
`q^2 = s(1-s) <= 1/4`, so `q <= 1/2`, and
`df/dt = (1-2s) s_dot - 2 q q_dot = (1-q) q^3 [ (1-2s) q - 2(1-q)(1-s) ]`.
If `1 - 2s <= 0` the bracket is negative. Otherwise `(1-2s) q <= (1-2s)/2 < 1 - s <= 2(1-q)(1-s)`.
So `df/dt < 0` on `{f = 0}` and `f` crosses zero only downward. ∎

**Interpretation (for the collaborator).** In this toy, `C > 0` together with a shrinking gap
(`rho < 0`) is a *terminal residual* regime, not a precursor of gap growth. The proposal's H5
latent-phase signature cannot be tested here. Planning-phase observation (E000, not
pre-registered): a multi-prompt toy with per-prompt correctness logits and one **shared** exploit
logit did show `- → + → -` in one configuration (Route B, easy prompts saturating). Phase 1B
design is the collaborator's call.

## 3. R: random symmetric flips

With a **fresh** flip `xi ~ Bern(p)` per query, independent of the response:
`E[V | y] = (1-p) G + p (1-G) = p + (1-2p) G`, hence **[given]**
`g_V = (1-2p) g_G`, `alpha = -2p`, `C = 0` — in **any** metric, because `g_e` is collinear with
`g_G`. `p = 1/2` is exactly deletion (`alpha = -1`); `p > 1/2` reverses the signal.

**Assumption boundary [derived-agent].** A *fixed* (deterministic) flip set on a finite response
space is a policy-controllable error. Flipping the single outcome `(corr, z) = (0, 1)` reproduces
Y exactly, with `C > 0`. The operative distinction between R and Y is whether the error is a
function of something the policy controls.

Not tested in Phase 1A: under step-normalized optimizers (Adam, sign-SGD, KL trust region) the
scalar `(1-2p)` is largely normalized away and R acts mainly through estimator variance.

## 4. X: signal deletion

Per prompt, a constant verifier gives `g_V(x) = 0`, so `g_e(x) = -g_G(x)`: `alpha = -1`, `C = 0`
in any metric **[given]**.

**Proposition 6 (aggregate X) [proved].** Tabular multi-prompt policy (prompts share no
parameters), prompt weights `w_x`, deleted set `S`, block-diagonal metric. Let
`A_S^2 = sum_{x in S} ||g_G(x)||^2_M` and `A_N^2` the same over the complement. Then

    alpha_agg = -A_S^2 / (A_S^2 + A_N^2),     C_agg = sqrt(A_S^2 A_N^2 / (A_S^2 + A_N^2)).

*Proof.* In block coordinates `g_G = (g_S, g_N)` with `<g_S, g_N>_M = 0` and `g_e = (-g_S, 0)`.
Then `alpha = -A_S^2/A^2`, `r = (-(A_N^2/A^2) g_S, (A_S^2/A^2) g_N)` and
`||r||^2 = A_S^2 A_N^4/A^4 + A_N^2 A_S^4/A^4 = A_S^2 A_N^2 / A^2`. ∎

For the Y-structured prompts with natural metric, `A_x^2 = w_x q_x (1 - q_x)`.

So `C_agg > 0` whenever both `S` and its complement carry signal, **with no exploitable direction
anywhere**. Proposal for discussion (not implemented): split `r` into the part inside
`span{g_G(x)}` (reweighting; X lives here) and the part outside (new directions; Y lives here).
Caveat: with shared parameters and many prompts the span can be the whole space.

## 5. Open issues raised to the collaborator (2026-09-24)

1. Single-prompt Y toy cannot exhibit latent → growth (Prop. 5); needs a Phase 1B toy.
2. Diagnostics must be computed in the optimizer's metric (Prop. 1 corollary); affects §3.4 of the
   proposal (Adam second moments as Fisher proxy).
3. X has `C_agg > 0` without exploitation (Prop. 6); proposed span split.
4. R's `C = 0` requires fresh, response-independent flips; fixed flips can be Y.
5. Naming: `(a,b,c)` vs `(A,alpha,C)`; correctness bit renamed `corr`; "Monte Carlo Fisher" vs
   "empirical Fisher".

## 6. Test map

| Claim | Tests |
| --- | --- |
| Y static formulas (§2.1) | `test_y_static.py` (closed form vs enumeration+autodiff vs finite differences), `test_monte_carlo.py` |
| Fisher formula | `test_fisher.py` (score outer product, negative Hessian, Monte Carlo) |
| Decomposition + Prop. 1 | `test_decompose.py`, `test_y_flows.py::test_metric_matched_rate_identities`, `test_y_flows.py::test_fisher_metric_mispredicts_vanilla_gap_sign` |
| Prop. 2 | `test_y_static.py::test_fisher_diagnostics_are_parameterization_invariant` |
| NG rates, invariant, endpoint (§2.2, Prop. 3) | `test_y_flows.py` |
| Vanilla rates, invariant, dDelta/dt (§2.3, Prop. 4) | `test_y_flows.py` |
| Prop. 5 | `test_y_flows.py::test_gap_sign_never_switches_from_shrinking_to_growing` |
| R (§3) | `test_r.py` |
| X (§4, Prop. 6) | `test_x.py` |
