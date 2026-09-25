# 04 — Phase 1B design (draft for review, 2026-09-25)

**Status: design only. No code exists for these models; nothing here is pre-registered.**
Every closed form below was checked numerically in the scratch directory (registry E000b), so later
predictions for these models will not be blind.

## Goal

Toy models in which a run shows, **early**: `C(t) > 0` and `dJ_G/dt > 0`; **later**: `dJ_G/dt -> 0`
(stall) or `dJ_G/dt < 0` (decline). The models must also make the question non-trivial: does early
verifier-induced orthogonal pressure predict the later gold failure **beyond static metrics**?
(Theory note v0.2, WH-1 and WH-4.)

## Definitions to fix before any Phase 1B registration

- **Gold failure** (WH-1). Compare with the gold-trained counterfactual under the same optimizer,
  clock and horizon `H`:
  - *stall*: `J_G^V(H) < J_G^G(H) - eps` while `dJ_G^V/dt -> 0`;
  - *decline*: `max_t J_G^V(t) - J_G^V(H) > eps`;
  - primary scalar outcome: the gold shortfall `J_G^G(H) - J_G^V(H)`.
- **Early window**: `t in [0, tau]`, with `tau` well before the failure time. It must be fixed
  per model before running.
- **Static baselines**: on-policy verifier accuracy, FPR and FNR under `pi_0`, and optionally on a
  fixed reference distribution.
- **Candidate predictors** (in the optimizer's metric, WH-2): `C(0)`, `C/A(0)`, early `∫ C^2 dt`,
  and a k-step function-space probe.
- **Non-triviality test.** Within one fixed toy structure, a flexible predictor of the full state
  can always "predict". The meaningful test compares predictors across a family of
  (verifier, `pi_0`) pairs **matched on static metrics**.

---

## Candidate 1 — multi-prompt, prompt-specific correctness, shared exploit (Route B)

1. **State variables.** `q_x` for `x = 1..K`, and `s`. Minimal: `K = 2` (easy/hard), fixed
   weights `w`.
2. **Policy.** `x ~ w` (not learned); `corr | x ~ Bern(sigmoid(u_x))`; `z ~ Bern(sigmoid(v))`, one
   logit shared across prompts and independent of `corr`. Parameters `(u_1..u_K, v)`.
3. **G and V.** `G = corr`; `V = corr OR z` on every prompt.
   `J_G = qbar = sum w_x q_x`, `J_V = 1 - (1 - qbar)(1 - s)`, `Delta = s (1 - qbar)`.
4. **Optimizers.** Natural gradient (exact block-diagonal Fisher) and vanilla, with exact expected
   gradients.
5. **ODEs** (exact):

       g_G = (w_x q_x(1-q_x))_x ⊕ 0
       g_V = (w_x q_x(1-q_x)(1-s))_x ⊕ (1-qbar) s(1-s)
       F   = diag(w_x q_x(1-q_x), s(1-s))
       natural:  u_x' = 1 - s,                    v' = 1 - qbar
       vanilla:  u_x' = w_x q_x(1-q_x)(1-s),     v' = (1-qbar) s(1-s)

   Invariants **[proved]**:
   - *Natural*: `s / prod_x q_x^{w_x}` is conserved, because
     `d log s/dt = (1-s)(1-qbar) = sum_x w_x d log q_x/dt`.
   - *Vanilla*: `H(v) - sum_x H(u_x)` with `H(x) = x - e^{-x}` is conserved, by the same argument
     as Prop. 4 with weights.
   - Under natural gradient all correctness logits shift by the same `tau(t) = ∫(1-s)`, so the
     system reduces to 2-D `(tau, v)` for any `K`.
6. **A, alpha, C.** Fully analytic:
   - Fisher metric: `A^2 = sum w_x q_x(1-q_x)`, `alpha = -s`, `C = (1-qbar) sqrt(s(1-s))`.
   - Euclidean: `A_E^2 = sum (w_x q_x(1-q_x))^2`, `alpha_E = -s`, `C_E = (1-qbar) s(1-s)`.
   - Natural gold rate `dJ_G/dt = (1-s) A^2`.
7. **Is the transition possible?** Yes, as a **stall**; decline is impossible because
   `q_x' >= 0`.

   **Proposition 7 (natural-gradient stall criterion) [proved; E000b-checked].** Let
   `Γ0 = prod q_x0^{w_x}` be the weighted geometric mean of initial correctness.
   - If `s0 > Γ0`: `s -> 1`, and gold stalls at `q_x∞ = sigmoid(u_x0 + tau∞)`, where `tau∞`
     solves `prod sigmoid(u_x0 + tau∞)^{w_x} = Γ0 / s0 < 1`.
   - If `s0 < Γ0`: all `q_x -> 1` and `s -> s0/Γ0`.

   *Proof.* All rates are >= 0 and bounded, so the state converges. At the limit either `s∞ = 1`,
   or `s∞ < 1` and every `q_x∞ = 1`. The invariant gives `Γ∞ = Γ0 s∞ / s0`. The second case needs
   `s∞ = s0/Γ0 <= 1`; the first needs `Γ∞ = Γ0/s0 <= 1`. ∎

   Early `C > 0` and `dJ_G/dt = (1-s)A^2 > 0`; late `dJ_G/dt -> 0` with `qbar < 1`. Because
   `Γ0 <= qbar0` (AM–GM), a few hard prompts can cause failure even when the exploit rate is below
   average accuracy.

   **Non-triviality (E000b).** Two initial conditions with identical aggregate
   `(A, alpha, C) = (0.412, -0.245, 0.301)` and identical aggregate static metrics end differently:
   one stalls (`qbar∞ = 0.915`), the other succeeds. Per-prompt quantities determine the outcome.
8. **Minimality / interpretability.** High. Three state variables, exact invariants for both
   optimizers, and the mechanism is Route B (easy prompts saturate, `A` shrinks). It maps directly
   onto GRPO's per-prompt groups and reuses the multi-prompt structure of X.
9. **Failure modes / confounds.**
   - (a) **Aggregate `C` = f(aggregate static metrics)** (`alpha = -FPR`,
     `C = (1-qbar)sqrt(FPR(1-FPR))`). The extra information lives in heterogeneity (`A`,
     per-prompt terms), so "aggregate `C` predicts the stall" is predicted **false by
     construction**.
   - (b) Under natural gradient, prompt weights cancel and every logit moves at the same speed;
     this is an artifact of tabular natural gradient.
   - (c) Under vanilla, hard prompts learn slowly even with the gold verifier, so the
     gold-trained counterfactual is required.
   - (d) The exploit is prompt-independent.
   - (e) Only stall is possible, never decline.

## Candidate 2 — coupled correctness/exploit through probability-mass competition

1. **State.** `(pC, pW, pE)` on the simplex (two free coordinates).
2. **Policy.** Softmax over response classes {C: correct, W: wrong, E: wrong but passes}. Two
   parameterizations: gauge-fixed (`thW = 0`, parameters `(thC, thE)`) and full 3-logit.
   Natural gradient is the same in both; **vanilla is not**.
3. **G and V.** `G = 1[y = C]`, `V = 1[y in {C, E}]`; `J_G = pC`, `Delta = pE`.
4. **Optimizers.** Natural and vanilla (both parameterizations).
5. **ODEs (gauge-fixed; exact).** With `F` the softmax Fisher on `(thC, thE)`:
   - `g_G = F e_C` and `g_V = F (1, 1)^T = pW (pC, pE)`.
   - Natural: `thC' = thE' = 1`, so `pC/pE` is conserved and gold rises monotonically to
     `pC0/(pC0 + pE0)`.
   - Vanilla: `thC' = pC pW`, `thE' = pE pW`; the invariant `e^{-thC} - e^{-thE}` gives
     rich-get-richer dynamics.
6. **A, alpha, C** (Fisher metric; analytic):
   - `A = sqrt(pC(1-pC))`, `alpha = -pE/(1-pC) = -FPR`, `C = sqrt((1-pC) FPR (1-FPR))`.
   - Gold rates: natural `dJ_G/dt = pC pW >= 0`; vanilla (gauge-fixed)
     `dJ_G/dt = pW pC [pC(1-pC) - pE^2]`, which is negative when `pE^2 > pC(1-pC)`.
7. **Is the transition possible?** Yes, and it is the **only candidate with gold decline**, but
   only under vanilla. E000b, from `(0.30, 0.35, 0.35)`:
   - natural: `pC` rises to 0.4615 and stalls;
   - vanilla, gauge-fixed: `pC` peaks at 0.335 (t ≈ 8), then falls to 3e-4 at t = 2e4;
   - vanilla, full parameterization: `pC` peaks at 0.397 (t ≈ 12), then falls to 0.011 at
     t = 1e6.

   Here the Fisher-metric analysis of the vanilla run predicts `dJ_G/dt = pC pW > 0` throughout
   while gold actually collapses: a **gold-sign** instance of the metric-mismatch corollary.
8. **Minimality / interpretability.** Most minimal (two free logits). The mechanism is "exploit
   displaces correct answers".
9. **Failure modes / confounds.**
   - (a) Decline depends on optimizer and parameterization: vanilla only, and its size differs
     between parameterizations. The parameterization must be pre-registered.
   - (b) **Diagnostics = f(static metrics)**, so the model cannot test "beyond static metrics".
   - (c) The collapse is rich-get-richer: any class with more initial mass wins. It is not specific
     to exploitation.

## Candidate 3 — delayed exploit accessibility (conjunctive trigger; Route A)

1. **State.** `(q, s1, s2)`.
2. **Policy.** Independent Bernoullis `corr`, `z1`, `z2` with logits `(u, v1, v2)`.
3. **G and V.** `G = corr`; `V = corr OR (z1 AND z2)`. The exploit fires with probability
   `S = s1 s2`, so on-policy FPR `= S` and FNR `= 0`.
4. **Optimizers.** Natural and vanilla.
5. **ODEs.**
   - Natural (exact): `u' = 1 - S`, `v1' = (1-q) s2`, `v2' = (1-q) s1`.
   - Invariant **[proved]**: `(1-s1)/(1-s2)` is conserved, because
     `d softplus(v_i)/dt = (1-q) S` for both `i`.
   - Symmetric start: `s' ≈ (1-q) s^2` for small `s`, a hyperbolic delayed takeoff at
     `t ≈ 1/((1-q) s0)`.
   - Vanilla: `u' = q(1-q)(1-S)`, `v1' = (1-q) s2 s1(1-s1)`, and symmetrically for `v2`.
     Numeric only.
6. **A, alpha, C** (analytic): `A = sqrt(q(1-q))`, `alpha = -S = -FPR`,
   `C = (1-q) sqrt(S (s1 + s2 - 2S))`. **`C` depends on `s1 + s2`**, i.e. on how close the policy
   is to completing the trigger. This is first-order *accessibility* information that is **not**
   a function of (accuracy, FPR).
7. **Is the transition possible?** Yes, as a stall: it is a race between gold learning and
   exploit takeoff. Because of the `(1 - q)` factor, the exploit shuts off once gold is learned,
   so a stall happens iff the takeoff comes first. Roughly, `1/s0 ≲ |logit q0|` in the symmetric
   case (heuristic; the exact criterion needs numerics).

   E000b, natural gradient, `q0 = 1e-3`, identical FPR 0.009:

   | exploit | `C(0)` | outcome |
   | --- | --- | --- |
   | conjunctive `(0.01, 0.9)` | 0.0895 | stall, `q∞ = 0.166` |
   | conjunctive `(0.0949, 0.0949)` | 0.0393 | exploit never takes off, `q -> 1` |
   | single-feature `s = 0.009` | 0.0943 | stall, `q∞ = 0.111` |

   Same static FPR, different `C`, different fate; `C` ranks the outcomes in this example. No
   decline is possible (`u' >= 0`).
8. **Minimality / interpretability.** Three parameters. This is the minimal model of the thesis
   ("same error rate, different accessibility") and of Route A, the regime the proposal claims as
   its novelty. It also realizes the proposal's §6 rare-trigger case.
9. **Failure modes / confounds.**
   - (a) Within one structure `(A, alpha, C)` still determines the state (up to swapping
     `s1`/`s2`), so the test **must** be across structures matched on static metrics (single vs
     conjunctive exploit with varied `s1`/`s2`, plus R and X).
   - (b) The outcome is a sharp race, sensitive to `q0` (task difficulty).
   - (c) Trigger features are independent of correctness; real exploits may be correlated with it.

## Considered, not proposed now

- **Threshold verifier on a continuous feature** (`V = corr OR 1[z > tau]`, `z ~ N(mu, 1)`). A rare
  trigger with Gaussian tails. Same mechanism as candidate 3 but less tractable.
- **An adaptive / learned verifier.** Out of scope for Phase 1.

## Comparison and ranking

| | C1 multi-prompt | C2 competition | C3 conjunctive |
| --- | --- | --- | --- |
| failure type | stall | stall (natural) / **decline** (vanilla) | stall |
| exact tractability | best (invariants for both optimizers, Prop. 7) | good | good (natural invariant; stall criterion numeric) |
| `C` carries info beyond static metrics | no (the extra info is in `A` / heterogeneity) | no | **yes** |
| non-trivial prediction test | yes, for *aggregate vs per-prompt* diagnostics | no | yes, across matched structures |
| maps to proposal | Route B; GRPO groups; X aggregate | metric-mismatch gold sign | Route A; §6 rare trigger; H1 |

**Ranking for the stated question** ("does early orthogonal pressure predict later gold failure,
beyond static metrics?"):

1. **C3 (recommended primary).** It is the only candidate in which `C` itself is not a function of
   the static error rates, and it produces the early-`C` → late-stall sequence through a delayed
   takeoff. The design must compare structures matched on static metrics.
2. **C1 (second, immediately after).** Exact ground truth and Route B. It is the natural test of
   whether *aggregate* diagnostics (what a batch-level LLM probe sees) suffice. We predict they
   do not (Prop. 7), which matters for the Phase 2 design.
3. **C2 (targeted side-study).** The only gold-decline mechanism. It is useful to show a gold-sign
   metric mismatch and optimizer dependence. It cannot test the thesis.

A later composite (C1 × C3: heterogeneous prompts with a delayed exploit) is the natural next step
once both behave as derived.
