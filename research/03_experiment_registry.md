# 03 — Experiment registry

Rules:

- The **pre-run block** (ID through Code/config) is written and committed **before** execution.
- The **post-run block** is appended after execution. The pre-run block is never edited; corrections
  go into a dated "Amendment" line below it.
- Unit tests are not experiments. An experiment is anything whose outcome could change a research
  claim.

Template:

```
### EXXX — title
Experiment ID:
Date:
Question:
Hypothesis:
Prediction:
Falsification criterion:
Independent variable:
Controlled variables:
Seeds:
Code/config:
--- post-run ---
Run directory / commit:
Observed result:
Does it match prediction?
Unexpected observations:
Interpretation:
Follow-up:
Hypothesis revised? yes/no
```

---

### E000 — Planning-phase scratch checks (retroactive, NOT pre-registered)

Experiment ID: E000
Date: 2026-09-24
Status: **exploratory, not pre-registered.** Recorded for transparency: these checks were run in a
scratch directory while writing the Phase 1A plan, before any repository code existed. Every
statement marked [derived-agent] in `01_theory_note.md`, and the numeric predictions of E001, were
therefore known to the agent before E001 was run. E001 is a verification of those predictions
by independent code, not a blind test.

What was run (numpy/scipy, closed-form right-hand sides, `solve_ivp`):

1. Closed-form Y gradients vs central finite differences at 500 random `(q, s)`: max error 1.3e-10.
2. NG flow endpoints for `(q0, s0)` in {(0.3, 0.01), (0.1, 0.1), (0.01, 0.3), (0.2, 0.5)} matched
   Prop. 3; `max |s/q - s0/q0|` ≤ 1.1e-9.
3. Vanilla invariant `H(v) - H(u)` drift ≤ 2.3e-10 up to `t = 1e7`. At `t = 1e7`, `(0.01, 0.30)`
   had `q = 0.01236`, `s ≈ 1`.
4. Vanilla and NG `dDelta/dt` closed forms vs chain rule at 500 points: max error 1.1e-10.
5. 300 random initial conditions × {NG, vanilla}: zero instances of `dDelta/dt` switching from
   negative to positive.
6. Tabular X aggregate: `alpha = -0.396061`, `C = 0.502480`, both equal to Prop. 6 closed form.
7. At `(q, s) = (0.3, 0.05)`: Fisher-metric triple predicts `dDelta/dt = +1.330e-2`; vanilla truth
   `-9.892e-4`. Euclidean triple under vanilla and Fisher triple under NG both matched truth.
8. Multi-prompt, shared exploit logit, NG, weights (0.9, 0.1), `q0 = (0.6, 1e-5)`, `s0 = 0.02`:
   sign of `dDelta/dt` was `-` (t < 3.96), `+` (3.96–9.69), `-` afterwards. Single-prompt control
   `q0 = 0.6` stayed `-`. One configuration only; not a systematic result.

Interpretation: none beyond "the closed forms survived a first check". Items 5 and 8 motivated
Prop. 5 and the Phase 1B suggestion.

---

### E001 — Y toy: vanilla vs natural gradient flows

Experiment ID: E001
Date: 2026-09-24 (pre-registered before implementation of the experiment code)

Question: In the two-Bernoulli Y toy with exact expected gradients, does a **generic** optimizer
(gradients by autodiff on the enumerated policy, Fisher by exact score outer product, integrated
in logit space) reproduce the closed-form dynamics of `01_theory_note.md` §2, and how do
`q, s, Delta, alpha, C` evolve for `q0 >> s0`, `q0 ≈ s0`, `q0 << s0` under vanilla vs natural
gradient?

Hypothesis: The closed forms in §2 are correct and the generic implementation is correct; so the
two agree to solver tolerance. The qualitative pictures below hold.

Prediction (all derived from closed forms before running; see E000 disclosure):

Initial conditions `IC1 = (0.30, 0.01)`, `IC2 = (0.10, 0.10)`, `IC3 = (0.01, 0.30)`.

- P1 (agreement). The generic `theta`-space trajectory and an independent integration of the
  closed-form `(q, s)` ODE agree at all evaluation times: `max |Δq|, |Δs| <= 1e-7`.
- P2 (invariants). NG: `|s/q - s0/q0| <= 1e-8` along the whole trajectory. Vanilla:
  `|I(t) - I(0)| <= 1e-6` with `I = H(v) - H(u)`, `I(0)` = -100.414489 (IC1), 0 (IC2),
  +100.414489 (IC3).
- P3 (NG endpoints at `t = 25`). IC1: `q -> 1`, `s -> 1/30 = 0.033333`; IC3: `q -> 0.033333`,
  `s -> 1`; tolerance 1e-8 on the non-saturating coordinate. IC2: `s = q` for all `t`,
  both increasing toward 1 (algebraic approach, not reached at `t = 25`).
  `Delta_inf`: IC1 0, IC2 0, IC3 0.966667.
- P4 (vanilla, clock-free). IC2: `s = q` for all `t` (identical path to NG, different clock).
  IC1: when `q = 1 - 1e-4`, `s = 0.011397` (NG: 0.033330). IC3: when `s = 1 - 1e-4`,
  `q = 0.011397` (NG: 0.033330). Tolerance 1e-5.
- P5 (gap sign pattern, `dDelta/dt`).
  NG: IC1 `+` then `-`, switching at `q = 0.5` (`s = 0.016667`); IC2 `+` then `-` at `q = s = 0.5`;
  IC3 `+` throughout (q never exceeds 1/30).
  Vanilla: IC1 `-` throughout (latent residual: `C > 0`, gap shrinking from `t = 0`); IC2 `+` then
  `-` at `q = s = 0.5`; IC3 `+` then `-` at `q = 0.011364`, `s = 0.999871`.
  No run shows a `-` → `+` switch (Prop. 5).
- P6 (diagnostics). Generic decomposition matches closed form along every trajectory:
  `alpha = -s` in both metrics; `C_F = (1-q) sqrt(s(1-s))`; `C_E = (1-q) s (1-s)`;
  tolerance `atol 1e-10 + rtol 1e-6`. In IC3 (both optimizers) and IC2, `alpha -> -1` and
  `C -> 0` as `s -> 1` (Y becomes deletion-like).
- P7 (Prop. 1 along trajectories). With the optimizer's own metric, `b(a+b) + c^2` equals the
  finite-difference `dDelta/dt` of the trajectory (rtol 1e-4, set by finite-difference error);
  with the Fisher metric under vanilla dynamics it does not (IC1 at `t = 0`: opposite sign).

Falsification criterion: Any of P1, P2, P3, P4, P6 violated beyond tolerance → the derivation or the
implementation is wrong; stop and investigate before any further experiment. P5 violated →
Prop. 5 or the P5 numbers are wrong. P7 violated → Prop. 1 or the implementation is wrong.

Independent variable: initial condition (IC1, IC2, IC3); optimizer (vanilla, natural).

Controlled variables: exact expected gradients (no sampling); exact Fisher; continuous time with
unit learning rate; solver DOP853, `rtol = 1e-10`, `atol = 1e-12`; horizons `T_nat = 25`,
`T_van = 1e6`; evaluation grid log-spaced from `1e-3` to `T`.

Seeds: none (fully deterministic, no Monte Carlo in this experiment).

Code/config: `experiments/toy/e001_y_flows.py`, `configs/toy/e001_y_flows.toml` (to be written
after this entry is committed); commit sha and dirty flag recorded in the run directory.

Amendment 1 (2026-09-24, written after the evaluation code and **before E001 was executed**):
the pre-run block left some evaluation details unspecified. Fixed now, without having seen output:
(a) P5 switch points are located by root-finding the observed `dDelta/dt` on the dense solver
output; they must match the registered `(q, s)` within 1e-6 (this includes the rounding of the
6-decimal registered values). Values registered as fractions (1/30, 1/60, 29/30) are compared
exactly. `|dDelta/dt| < 1e-13` counts as numerically zero when reading sign patterns.
(b) P7 compares a central finite difference of `Delta(t)` on the dense output (step
`h = max(1e-6, 1e-4 t)`) with `b(a+b) + c^2` in the optimizer's metric, passing if
`|diff| <= 1e-4 |pred| + 1e-6 max_t |pred|`; the absolute term covers sign switches, where a
purely relative criterion is ill-defined.
(c) P4 locates `q = 1 - 1e-4` (IC1) and `s = 1 - 1e-4` (IC3) by root-finding on the dense output.
(d) P3 for IC2 is checked as `max |s - q| <= 1e-8` with `q`, `s` non-decreasing; `Delta_inf` is
checked only where the limit is reached within `T` (IC1, IC3).
(e) `Delta` is computed as `E_pi[V - G]` on the enumerated policy (no cancellation), not as
`J_V - J_G`.

--- post-run (appended 2026-09-25) ---

Run directory / commit: `results/E001/20260925T012753Z_9d4df22/` — commit `9d4df22`, clean tree.
Wall time ≈ 15 s on the local CPU. Full check table: `summary.md` / `checks.json` in that directory.

Observed result (83 registered checks: **82 passed, 1 failed**):

- P1: generic `theta`-space trajectories vs closed-form `(q, s)` ODE, max error 6.5e-11 – 1.8e-10.
- P2: NG invariant drift 1.6e-11 (IC1), 3.6e-15 (IC2), **1.46e-8 (IC3) — FAILED the registered
  absolute tolerance 1e-8**. Vanilla invariant drift ≤ 5.0e-10; `I(0)` equals the registered values.
- P3: IC1 ends at `(1, 1/30)`, IC3 at `(1/30, 1)` within 1e-8; `Delta(T)` = 2.4e-12 and 0.9666667.
  IC2 stays on the diagonal (max |s − q| = 2.7e-15), `q(25) = s(25) = 0.9522`.
- P4: vanilla IC1 `s = 0.0113971` when `q = 0.9999` (t = 1.01e4), registered 0.011397; IC3 is the
  mirror image; vanilla IC2 on the diagonal exactly.
- P5: all six sign patterns as registered (NG `+-`, `+-`, `+`; vanilla `-`, `+-`, `+-`); no `-` → `+`
  switch anywhere. Switch points: NG IC1 `(0.5, 1/60)` at t = 0.859; NG IC2 `(0.5, 0.5)` at t = 3.09;
  vanilla IC2 `(0.5, 0.5)` at t = 17.8; vanilla IC3 `(0.0113643, 0.999871)` at t = 7851.
- P6: generic decomposition vs closed form, worst normalized error 0.19 (pass ≤ 1). `alpha(T)`:
  IC3 −1.0000 (NG), −0.999999 (vanilla); IC2 −0.99929 (vanilla), −0.952 (NG, not saturated at T=25).
- P7: finite-difference `dDelta/dt` vs `b(a+b)+c^2` in the optimizer's own metric, worst normalized
  error 2.1e-3 (pass ≤ 1). Fisher-metric prediction under vanilla dynamics at IC1, t = 0:
  +2.77e-3 vs observed −3.89e-4 (opposite signs, as registered).

Does it match prediction? Yes for every registered prediction except P2 IC3/natural, which failed its
registered tolerance. That check stays recorded as failed.

Unexpected observations:

1. The P2 failure. Investigated per the falsification criterion (diagnostic E001-D1 below): the
   relative drift, 4.87e-10, is identical to the mirrored IC1 (invariant 1/30 vs 30) and shrinks
   with the solver tolerance. The registered tolerance was absolute for a quantity of magnitude 30.
   No evidence against Proposition 3 or the implementation.
2. The Fisher-metric sign error is not confined to isolated points: along the vanilla trajectories
   the sign of the Fisher-metric `dDelta/dt` prediction is wrong at 39.7% (IC1) and 23.4% (IC3) of
   evaluation points, 0% for IC2 (where both zero sets meet at q = 1/2). Not a registered quantity.
3. IC1 and IC3 are mirror images to machine precision (identical field-evaluation counts, swapped
   `q(T)`/`s(T)`), as expected from the `q <-> s` symmetry of `J_V` (theory note §2).
4. Horizon dependence is extreme for vanilla: in IC3 the gap keeps growing until t ≈ 7.9e3, and at
   t = 1e6 gold is still `q = 0.01202` (natural-gradient endpoint 0.0333), `Delta = 0.988`.

Interpretation (agent; for collaborator review):

- The closed forms of theory note §2, both [given] and [derived-agent], are reproduced by an
  optimizer that knows only the policy specification and the reward table. No theory–code
  discrepancy was found apart from one mis-specified tolerance.
- The single-prompt Y toy shows only `+ -` or constant gap-sign patterns: no latent-then-growth
  phase (Proposition 5, open issue #1).
- The optimizer that is "safer" depends on the initial condition. In IC1 vanilla ends with less
  exploit (s = 0.012 vs 0.033); in IC3 it ends with less gold (q = 0.012 vs 0.033). This matches
  `d log s / d log q = s(1-s) / (q(1-q))`. A blanket "vanilla delays exploitation" is not supported.
- A mismatched metric mispredicts the direction of the proxy–gold gap over large fractions of a
  trajectory (open issue #2).

Post-hoc speculation (untested): for LLM RLVR with Adam, the same mismatch (Fisher proxy vs the
actual Adam preconditioner) could be a leading error source for gradient-space `(a, b, c)`.

Follow-up:

- State tolerances on conserved quantities in relative terms in future registrations.
- Phase 1B toy (multi-prompt, shared exploit logit) — pending the collaborator's decision.
- Monte Carlo estimation of `(A, alpha, C)` at finite batch size (proposal Phase 1 gate) not started.

Hypothesis revised? **No.** Only the tolerance convention for future experiments changes.

### E001-D1 — Post-hoc numerical diagnostic for the P2 failure (NOT pre-registered)

Date: 2026-09-25. Purpose: decide whether the P2 IC3/natural failure reflects the dynamics or the
integrator. Method: same generic natural-gradient field and evaluation grid as E001, solver
tolerances varied; code run inline (not in a committed script), output copied here.

| (q0, s0) | rtol / atol | max abs drift of s/q | relative drift |
| --- | --- | --- | --- |
| (0.30, 0.01) | 1e-10 / 1e-12 | 1.622e-11 | 4.866e-10 |
| (0.30, 0.01) | 1e-11 / 1e-13 | 2.857e-12 | 8.571e-11 |
| (0.30, 0.01) | 1e-12 / 1e-14 | 3.719e-13 | 1.116e-11 |
| (0.01, 0.30) | 1e-10 / 1e-12 | 1.460e-08 | 4.866e-10 |
| (0.01, 0.30) | 1e-11 / 1e-13 | 2.571e-09 | 8.570e-11 |
| (0.01, 0.30) | 1e-12 / 1e-14 | 3.348e-10 | 1.116e-11 |

Reading: the drift is integration error that shrinks with solver tolerance and is identical in
relative terms for mirrored initial conditions. E001's recorded result is not changed.

#### E001 post-run addendum A (2026-09-25, theory note v0.2 review)

Nothing above this line is edited. This addendum records semantic revisions and precision
corrections made after review. It does not re-score anything.

- *Semantic revision.* The pre-run P5 labelled vanilla IC1 as "latent residual: `C > 0`, gap
  shrinking", and the post-run interpretation read the sign patterns as "no latent-then-growth
  phase". Under theory note v0.2 §2, `dDelta/dt > 0` is not a failure or hacking signal. The P5
  results stay valid as observations about the sign of the proxy–gold gap rate only.
- *Precision: metric mismatch.* The mispredicted sign was that of `dDelta/dt`. For `dJ_G/dt` the
  Fisher-metric and actual signs agreed in this toy (both >= 0); only magnitudes differed.
- *Precision: `C` dynamics.* Peak / initial `C_F`: 1.20 (IC2), 1.08–1.09 (IC3), ≈ 1.00 (IC1);
  Euclidean `C` up to 1.83. `C` then decays toward 0 (to 8e-6 in IC3/natural). In IC1 the decay
  comes from gold saturation, not exploit saturation.
- *Precision: late gold failure.* The toy does show a late gold **stall** (natural IC3:
  `q -> 1/30`; vanilla IC3: `q(1e6) = 0.0120`). What it lacks is gold decline and any diagnostic
  information beyond static metrics (theory note v0.2 §3.4).
- Hypothesis revised? The pre-registered E001 hypothesis: no. New working hypotheses WH-1–WH-4
  are recorded in theory note v0.2 §7.

---

### E000b — Phase 1B design-phase scratch checks (retroactive, NOT pre-registered)

Date: 2026-09-25. Status: **exploratory, not pre-registered.** Run in the scratch directory with
numpy/scipy while drafting `04_phase1b_design.md`, to avoid proposing wrong math. Closed forms were
compared with central finite differences; flows were integrated with `solve_ivp`. No repository
code or tests exist for these models yet. Any later Phase 1B experiment predictions are therefore
**not blind** with respect to the items below.

1. C1 (multi-prompt, shared exploit), K = 3:
   - closed-form `A`, `alpha`, `C` and Fisher vs finite differences: max error 7.4e-10.
   - NG invariant `s / prod_x q_x^{w_x}`: drift ≤ 2.6e-10. Vanilla invariant
     `H(v) - sum_x H(u_x)`: drift ≤ 1.5e-8.
2. C1 stall criterion (NG, stall iff `s0 > prod q_x0^{w_x}`), three configurations, all as
   predicted:
   - `w = (0.9, 0.1)`, `q0 = (0.6, 1e-5)`: `s0 = 0.02` gave no stall (`s -> 0.1002 = s0/Γ0`);
     `s0 = 0.3` stalled (`q -> (0.9996, 0.0171)`).
   - `w = (0.5, 0.5)`, `q0 = (0.8, 0.05)`, `s0 = 0.25`: stalled (`q -> (0.993, 0.645)`).
3. C1 matched aggregate triple. Two initial conditions with identical `(A, alpha, C) =
   (0.4123, -0.245, 0.3011)` and identical aggregate static metrics (`qbar = 0.3`, FPR = 0.245)
   had different outcomes:
   - `w = (0.5, 0.5)`, `q0 = (0.1, 0.5)`: stall, `qbar_inf = 0.915`.
   - `w = (0.9, 0.1)`, `q0 = (0.233, 0.9)`: success, `qbar_inf = 1`.
4. C2 (categorical {C, W, E}): closed-form `A`, `alpha = -pE/(1-pC)`, `C`, natural and vanilla
   `dJ_G/dt` vs finite differences: max error 5.9e-9. From `p0 = (0.30, 0.35, 0.35)`:
   - natural: `pC` rose monotonically to 0.4615 (= 0.30/0.65);
   - vanilla, gauge `thW = 0`: `pC` peaked at 0.335 (t ≈ 8.4), then fell to 3e-4 at t = 2e4;
   - vanilla, full 3-logit parameterization: `pC` peaked at 0.397 (t ≈ 12), then fell to 0.011
     at t = 1e6.
   - Invariants drift ≤ 5e-11.
5. C3 (conjunctive exploit `V = c OR (z1 AND z2)`): closed forms vs finite differences, max error
   2.0e-9; invariant `(1-s1)/(1-s2)`, relative drift 7.2e-8. NG, `q0 = 1e-3`, matched FPR 0.009:

   | exploit structure | `C(0)` | outcome |
   | --- | --- | --- |
   | conjunctive `(s1, s2) = (0.01, 0.9)` | 0.0895 | stall, `q_inf = 0.166` |
   | conjunctive `(0.0949, 0.0949)` | 0.0393 | success; exploit never took off (`S_inf = 0.050`) |
   | single-feature, `s = 0.009` | 0.0943 | stall, `q_inf = 0.111` |

---

### Phase 1B claim (frozen 2026-09-25, before any Phase 1B code)

**Claim under test (controlled, positive).** Verifiers matched on conventional static metrics can
induce different policy-conditioned local geometry and different downstream **gold-learning**
outcomes.

**Not claimed.** That gap growth is failure; that snapshot `C` alone predicts outcomes; anything
about LLM-scale RLVR.

**Outcomes** (all defined on `J_G`, never on the signed gap `Delta`):

- **O1 asymptotic gold** `J_G(∞)`: analytic under natural gradient (theory note §10, Prop. 9);
  observed as `J_G(T_end)`.
- **O2 finite-horizon gold** `J_G(T)` at `T ∈ {10, 25, 100, 500}`: reported.
- **O3 gold-learning stall.**
  - Theory: `q0 · exp(Λ(1)) < 1`.
  - Observed: `S(T_end) >= 1 - 1e-2` and `J_G(T_end) < 0.999`.
  - Observed success: `J_G(T_end) >= 1 - 1e-6`.
- **O4 clean-verifier performance gap** `sigmoid(u0 + T) - J_G(T)`. The clean natural-gradient
  baseline is analytic because `u' = 1` under `G`.
- **O5 gold shortfall** `1 - J_G(∞)`.

### E003 — Candidate 3: feature-triggered false positives matched on static metrics (natural gradient)

Experiment ID: E003
Date: 2026-09-25 (pre-registered before any Candidate 3 code, test or experiment script exists)

**Question.** Across verifier structures matched on initial gold accuracy, verifier accuracy, FPR,
FNR and false-positive mass (and therefore on `A` and `alpha`), do `C` and the natural-gradient
gold outcome differ as predicted by theory note §10 (Props. 8–10)?

**Hypothesis.** Props. 8–10 are correct, and a generic autodiff natural-gradient optimizer
reproduces them.

**Design.**

- Policy: independent Bernoulli coordinates `(corr, z_1, …, z_m)` with logits.
- `G = corr`. Verifier `V = corr OR 1_E(z)` for an event `E` over that structure's own features.
- The random-FP control is `V = corr OR xi`, with `xi ~ Bern(f)` drawn fresh for every response.
- Identical initial policy: the policy is the product over the union of all structures' features.
  Each verifier reads only its own features, and by independence each run reduces exactly to
  `(corr, own features)`. The reduction is to be unit-tested.
- Parameters: `q0 = 0.001` (`u0 = logit(0.001)`), `f = 0.01`.

| structure | event `E` | initial feature probabilities |
| --- | --- | --- |
| RFP | fresh coin, P = f (not policy-controllable) | — |
| AND3 | `z1 AND z2 AND z3` | `(f^(1/3),)*3 = 0.215443…` each |
| AND-SYM | `z1 AND z2` | `(0.1, 0.1)` |
| AND-MID | `z1 AND z2` | `(0.04, 0.25)` |
| AND-ASYM-A | `z1 AND z2` | `(0.0125, 0.8)` |
| AND-ASYM-B | `z1 AND z2` | `(f/0.98, 0.98) = (0.0102041…, 0.98)` |
| OR | `z1 OR z2` | `(1 - sqrt(1-f),)*2 = 0.0050126…` each |
| SINGLE | `z1` | `(f,) = (0.01,)` |

**Predictions** (from closed forms only: quadrature and root finding, no ODE integration):

- **P1 Matched static metrics (exact).** For all eight structures at `t = 0`:
  - gold accuracy 0.001, FPR 0.01, FNR 0, verifier accuracy 0.990010, false-positive mass
    0.009990;
  - Fisher-metric `A = 0.0316070` and `alpha = -0.01`.

  Identical across structures to 1e-12.
- **P2 `C(0)`, Fisher metric** (tolerance 1e-6 vs these 6-decimal values):

  | structure | `C(0)` | `eta0 = (C/C_max)^2` |
  | --- | --- | --- |
  | RFP | 0 | 0 |
  | AND3 | 0.033020 | 0.1104 |
  | AND-SYM | 0.042384 | 0.1818 |
  | AND-MID | 0.051910 | 0.2727 |
  | AND-ASYM-A | 0.088933 | 0.8005 |
  | AND-ASYM-B | 0.098400 | 0.9800 |
  | OR | 0.099274 | 0.9975 |
  | SINGLE | 0.099399 | 1.0000 |

  `C_max = 0.099399`. Strict ordering: RFP < AND3 < AND-SYM < AND-MID < AND-ASYM-A < AND-ASYM-B <
  OR < SINGLE.
- **P3a Gold-race relation (trajectory level; the sharpest test).** At every evaluation time,
  `log J_G(t) - log q0 = Λ_k(state(t))` within 1e-7. The functions `Λ_k` are:
  - SINGLE: `log(s/s0)`;
  - OR (symmetric): `log(a/a0)`;
  - AND-SYM: `[log s - 1/s] - [log s0 - 1/s0]`;
  - AND3: `[log s - 1/s - 1/(2s^2)] - [same at s0]`;
  - AND asymmetric: `[log(s2/s2_0) - rho log(s1/s1_0)]/(1 - rho)` with
    `rho = (1-s1_0)/(1-s2_0)`;
  - RFP: `J_G(t) = sigmoid(u0 + (1-f) t)` exactly.
- **P3b Outcome class and asymptotic gold** (`T_end = 500`):

  | structure | outcome | asymptotic `J_G` | asymptotic `S` | shortfall |
  | --- | --- | --- | --- | --- |
  | RFP | success | 1 | 0.010000 | 0 |
  | AND3 | success | 1 | 0.027274 | 0 |
  | AND-SYM | success | 1 | 0.062287 | 0 |
  | AND-MID | success | 1 | 0.123215 | 0 |
  | AND-ASYM-A | **stall** | 0.230039 | 1 | 0.769961 |
  | AND-ASYM-B | **stall** | 0.107674 | 1 | 0.892326 |
  | OR | **stall** | 0.199499 | 1 | 0.800501 |
  | SINGLE | **stall** | 0.100000 | 1 | 0.900000 |

  Tolerances:
  - Success: `J_G(500) >= 1 - 1e-6` and `|S(500) - S∞| <= 1e-5`.
  - Stall: `|J_G(500) - J_G(∞)| <= 1e-3` (OR converges only algebraically, predicted residual
    ≈ 5e-4) and `S(500) >= 1 - 1e-2`.
- **P4 Accessibility paths.**
  - `eta(t)` is non-decreasing for AND3, AND-SYM, AND-MID, AND-ASYM-A, AND-ASYM-B;
  - non-increasing for OR (`eta = 2(1-a)/(2-a)`);
  - constant 1 for SINGLE and 0 for RFP.
- **P5 Clean-gap bound.** For all structures and `t`, `J_G(t) <= sigmoid(u0 + t)`. So the
  clean-verifier gap O4 is >= 0 and tends to the shortfall.
- **P6 Invariants.**
  - AND structures: `(1-s_i)/(1-s_j)` is constant (relative drift ≤ 1e-7 while `1 - s > 1e-9`);
  - symmetric structures stay symmetric;
  - all logits are non-decreasing.
- **P7 Snapshot `C` is informative but not sufficient** (a consequence of P2 and P3b; theory note
  Prop. 10). Spearman`(C(0), shortfall) = +0.913` over the eight structures (average ranks for the
  four tied successes). Exactly one discordant pair among structures with distinct outcomes:
  **OR vs AND-ASYM-B**. OR has the higher `C(0)` (0.099274 vs 0.098400) but the higher gold
  (0.1995 vs 0.1077), because OR's `eta` falls and AND-ASYM-B's rises along the path. This
  inversion is predicted, not a post-hoc finding.

**Falsification criterion.**

- P1, P2, P3a or P6 outside tolerance → derivation or implementation wrong; stop and investigate.
- P3b, P4 or P5 violated → Prop. 9/10 or the registered numbers are wrong.
- A different P7 discordance pattern → Prop. 10's explanation is wrong.

**Independent variable.** Verifier structure (8 levels).

**Controlled variables.**

- Initial policy (product construction); `q0`; `f`.
- Natural gradient with exact expected gradients and the exact Fisher of the reduced policy.
- DOP853 solver, `rtol = 1e-10`, `atol = 1e-12`.
- Evaluation grid: `t = 0` plus 400 log-spaced points on `[1e-3, 500]`.

**Seeds.** None (deterministic).

**Secondary, exploratory (E003-V).** The same eight structures under vanilla gradient flow to
`T = 1e6`. **No predictions are registered**; results will be reported as exploratory only.

**Code/config** (to be written after this entry is committed):

- `src/vdyn/policies/bernoulli.py` (product policy),
- `src/vdyn/verifiers/triggered.py`,
- `src/vdyn/geometry/triggered_fp.py` (closed forms),
- tests `tests/test_triggered_*.py`,
- `experiments/toy/e003_triggered_fp.py`, `configs/toy/e003_triggered_fp.toml`.

**Disclosure.** E000b simulated one related configuration, conjunctive `(0.01, 0.9)` at
`q0 = 1e-3`, `f = 0.009`, and found `q∞ = 0.166`. The closed form of Prop. 9 gives 0.1659 for it.
None of the eight registered configurations has been simulated.

**E003 pre-run status note (2026-09-25, after implementation, before execution).** This does not
edit the pre-run block above.

- Code for Candidate 3 now exists and passes its tests.
- Tests on **unregistered** parameters (`q0 = 0.02`; single, symmetric OR, symmetric / asymmetric
  AND, AND3, random-FP) already confirm Prop. 9 in the generic natural-gradient optimizer: the
  gold-race relation, the invariants and the predicted outcomes. P3a is therefore not a blind test
  of Prop. 9 in general. It remains a blind test of the registered settings and numbers.
- Tests at the registered settings are static (`t = 0`) or closed-form only. They confirm P1 and
  the `t = 0` part of P2, and that the repository's closed-form code reproduces every registered
  prediction to 1e-6.
- No trajectory at the registered settings has been integrated.
- Implementation change made on the way: outcome Jacobians in `geometry/autodiff.py` now use
  forward mode (`jacfwd`). The full-feature-bank test has 2^15 outcomes, and reverse mode ran out
  of memory. The math is unchanged; E001 is unaffected.

--- E003 post-run (appended 2026-09-25) ---

**Run.** `results/E003/20260925T061919Z_8ab7d9d/`

- Run commit `8ab7d9d`, clean tree. The run started 2026-09-25T06:19:19Z and took ≈17 s of CPU.
- Environment: Python 3.12.11, torch 2.14.0, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2
  (macOS arm64).
- `trajectories.npz` (not committed) is pinned by
  `sha256 b84183d6c46ea5780a1134541d4f01d4c486b2101ac70cb6082da8e011b8aa97`.

**Pre-run integrity check: PASSED** (`preflight_integrity.txt`).

- Clean tree. The registry pre-run text had zero deleted lines since `2736f50`, and the config was
  unchanged since `90cf69a`.
- 155 tests, ruff, format and mypy all passed. No E003 result existed beforehand.
- At `t = 0` the static metrics agreed to ≤ 2.3e-16 and `C(0)` to ≤ 4.5e-7 of the registered
  values.
- The config's predictions are identical to the registry text.
- Commit `8ab7d9d` changed execution only: one optimizer per invocation, extra recorded series,
  figures. The registered evaluation functions (`check_static`, `check_c0`, `check_run`,
  `check_ranking`, `report`, `_outcome_class`) are byte-identical to `90cf69a`. `simulate`
  differs only by factoring out the field construction.

#### CONFIRMATORY E003 RESULT (natural gradient, registered)

**Score: 79 / 82 registered checks passed; 3 FAILED (all P4).**

| Pred. | Prediction | Observed | Tolerance | Result |
| --- | --- | --- | --- | --- |
| P1 | static metrics, `A`, `alpha` identical at `t = 0` | max spread across structures ≤ 2.3e-16; max deviation from registered values ≤ 2.3e-16 | 1e-12 | PASS (14/14) |
| P2 | `C(0)` values and strict order | max deviation 4.5e-7; order as registered | 1e-6 | PASS (9/9) |
| P3a | `log J_G(t) - log q0 = Λ(state(t))` along each trajectory | max deviation 4.7e-9 (AND3); ≤ 1e-9 for the others; RFP 5.8e-15 | 1e-7 | PASS (8/8) |
| P3b | outcome class; `J_G(500)` or `S(500)` | all 8 classes as predicted; stalls: 0.230039, 0.107674, 0.198991 (OR, pred. 0.199499, residual 5.1e-4 as anticipated), 0.100000; successes: `S(500)` = 0.010000, 0.027274, 0.062287, 0.123215 | 1e-3 (stall), 1e-5 (`S`) | PASS (16/16) |
| P4 | `eta` path trend | AND3, AND-SYM, AND-MID: non-decreasing ✓; OR: non-increasing ✓; RFP constant ✓. **AND-ASYM-A** min step −2.1e-7, **AND-ASYM-B** min step −4.4e-6, **SINGLE** max \|eta − 1\| = 7.6e-6 | 1e-8 (fixed in code at `90cf69a`; the registry text gave no number) | **FAIL (3 of 8)** |
| P5 | `J_G(t) <= clean J_G(t)` | max excess 0 | 1e-12 | PASS (8/8) |
| P6 | invariants, symmetry, monotone logits | ratio drift ≤ 6.4e-8; symmetry ≤ 2.4e-13; min logit step ≥ −6.6e-14 | 1e-7 / 1e-10 | PASS (14/14) |
| P7 | Spearman(`C(0)`, shortfall) = 0.913; single discordant pair OR vs AND-ASYM-B | 0.913223; discordant pairs = [AND-ASYM-B, OR] exactly | 5e-4 | PASS (2/2) |

**Raw outcome table** (natural gradient; clean-verifier reference `sigmoid(u0 + t)` = 1 at `t = 500`):

| structure | C(0) | J_G(10) | J_G(25) | J_G(100) | J_G(500) | J_V(500) | FPR(500) | shortfall vs clean | class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RFP | 0 | 0.952268 | 1.000000 | 1 | 1 | 1 | 0.010000 | 0 | success |
| AND3 | 0.033020 | 0.948031 | 1.000000 | 1 | 1 | 1 | 0.027274 | 0 | success |
| AND-SYM | 0.042384 | 0.940487 | 1.000000 | 1 | 1 | 1 | 0.062287 | 0 | success |
| AND-MID | 0.051910 | 0.926589 | 1.000000 | 1 | 1 | 1 | 0.123215 | 0 | success |
| AND-ASYM-A | 0.088933 | 0.222768 | 0.230039 | 0.230039 | 0.230039 | 1 | 1 | 0.769961 | stall |
| AND-ASYM-B | 0.098400 | 0.106751 | 0.107674 | 0.107674 | 0.107674 | 1 | 1 | 0.892326 | stall |
| OR | 0.099274 | 0.149184 | 0.186023 | 0.196788 | 0.198991 | 0.999995 | 0.999994 | 0.801009 | stall |
| SINGLE | 0.099399 | 0.099242 | 0.100000 | 0.1 | 0.1 | 1 | 1 | 0.900000 | stall |

**Confirmatory conclusions.**

1. **The Phase 1B claim is supported.** Eight verifiers matched at `t = 0` on gold accuracy,
   verifier accuracy, FPR, FNR, FP mass, `A` and `alpha` (spread ≤ 2.3e-16) produced four gold
   successes and four stalls, with asymptotic gold ranging from 0.100 to 1.
2. The proxy reaches `J_V ≈ 1` in **all** runs, so the terminal proxy cannot distinguish these
   fates.
3. The asymptotic gold of every structure matches the gold-race law (Prop. 9) to ≤ 5e-9 in log
   units along the whole trajectory.
4. `C(0)` orders the outcomes with the registered Spearman 0.913. The one registered inversion
   (OR vs AND-ASYM-B) occurred exactly as predicted. **Snapshot `C` is informative but not
   sufficient.**
5. **Failed predictions: P4 for AND-ASYM-A, AND-ASYM-B and SINGLE.** They remain FAILED.
   Post-hoc diagnostic D1 below attributes them to floating-point evaluation of `(C/C_max)^2` at
   saturation. That attribution does not rescore them.

#### POST-HOC / EXPLORATORY ANALYSIS (not confirmatory)

Output: `results/E003-posthoc/20260925T062446Z_0b97e2e/`, from script `e003_posthoc.py`, reading
the pinned registered trajectories with no new training.
`results/E003-posthoc/20260925T062236Z_6bd6570/` is an earlier, superseded pass of the same
analysis without the stable-`eta` diagnostic.

- **D1 (P4 numerical diagnostic).** All violations lie at `t ≈ 26–34`:
  - there `1 - FPR` is 1e-10 to 4e-12 and `C_max` is 1.5e-5 to 1.8e-6;
  - with the analysis restricted to `C_max > 1e-3` there are 0 violations.

  `eta` recomputed without cancellation from the logits (`1 - s = expit(-v)`,
  `1 - S = -expm1(Σ log expit(v))`) obeys the registered trend at every observed state for all
  three structures. The generic `(C/C_max)^2` deviates from it by up to 4.0e-5, and only at those
  saturated states; the probability-space closed form deviates by ≤ 2.7e-9.

  Reading: the registered check evaluated `eta` in a numerically unstable way (mask
  `C_max > 1e-6` too permissive); there is no evidence that the dynamics violate P4. **Lesson for
  future registrations:** specify numerically stable evaluation of any ratio of vanishing
  quantities.
- **Q4 (does short-horizon dynamics help?).** The summaries use trajectory prefixes `t <= k`
  (orientation fixed by meaning, not fitted). Over the 22 pairs with distinct outcomes:

  | prefix | summaries | result |
  | --- | --- | --- |
  | `k = 1` (before takeoff; stall runs have reached 1–3% of final gold) | every informative summary: `C(0)`, `eta(0)`, mean or integral of `eta`, `delta A`, `delta C`, `delta log FPR`, `delta log J_G` | ρ = 0.913 with the **same single discordant pair** (OR vs AND-ASYM-B) |
  | `k = 5` | mean/integral `eta` (OR 0.954 vs AND-ASYM-B 0.983), `delta log FPR` (static metric), `delta log J_G` (early gold), `delta A`, `delta C` | 0 discordant pairs (ρ = 0.939, the maximum attainable given ties) |
  | `k = 10` | only `delta log J_G` | still perfect; by then the stall runs have reached 75–99% of final gold |

  Additional notes:
  - `delta eta` is anti-predictive.
  - `A(0)` is constant across structures by design, so it carries no information.

  Reading: short-horizon `eta` summaries fix the inversion only once the prefix covers the exploit
  takeoff (by `t = 5` FPR has already risen about 50-fold). At that point the trajectory of the
  **static metric FPR** and early gold progress fix it just as well. So in this noiseless toy the
  **incremental value of geometry over tracking static metrics over time is not demonstrated.**

  Two caveats: every monotone prefix summary is rank-predictive here, and at `k = 1` the
  relative spread across structures is ≈180% for `eta` summaries but only 0.7% for
  `delta log J_G`. Any practical advantage would therefore have to come from effect size under
  estimation noise, which is untested.
- **Q5 (the inversion, from the trajectories).**
  - `eta_OR(S)` and `eta_B(S)` cross at `FPR = 0.072` (t ≈ 2.06). At that point both have
    `J_G = 0.0073`.
  - Before the crossing OR's `eta` is higher: its gold log-gain from `FPR = 0.01` to 0.072 is
    1.991, vs 2.014 for AND-ASYM-B.
  - After the crossing OR's `eta` collapses toward 0 as its exploit saturates, while AND-ASYM-B's
    rises to 1. Log-gains from `FPR = 0.072` to `1 - 1e-5`: 3.371 vs 2.665.
  - Observed `J_G` when `FPR` first reaches 0.9: 0.137 (OR) vs 0.097 (AND-ASYM-B). Final: 0.199
    vs 0.108.
  - Conclusion: OR starts with higher `C`/`eta` but its accessibility decays along the path. Most
    of its extra gold is earned late, while the exploit is saturating.

#### E003-V EXPLORATORY RESULT (vanilla gradient; no registered predictions)

Run `results/E003-V/20260925T062511Z_ea62f59/` (commit `ea62f59`, clean tree; run after the
registered record was committed as `9ed68aa`). `trajectories.npz` is pinned by its SHA-256 in
that directory.

- **Every** policy-controllable structure stalls almost immediately:
  - `J_G(1e6)` lies in 0.00111–0.00127 (from 0.001) with `FPR -> 1`;
  - the random-FP control reaches `J_G = 0.99999899`. The frozen thresholds call that
    "unresolved", but only because it misses `1 - 1e-6` by 1e-8.
- Static matching still coexists with different fates, but only in the controllable vs
  uncontrollable contrast. Among controllable structures all fates are stalls, with small
  quantitative differences.
- **The ordering of danger changes with the optimizer.**
  - Vanilla worst → best: AND-SYM, AND-MID, AND3, SINGLE, AND-ASYM-B, AND-ASYM-A, OR.
  - Spearman between natural and vanilla shortfalls: −0.30.
  - The conjunctive structures that were safe under natural gradient are the most dangerous under
    vanilla. Exploit takeoff (`FPR = 0.5`): AND-SYM at t = 78, SINGLE at 113, OR at 222.
- **Optimizer-matched geometry explains the vanilla ordering; Fisher geometry does not.**

  | metric | Spearman(`C(0)`, vanilla shortfall), all 8 | 7 exploit structures only |
  | --- | --- | --- |
  | Euclidean `C_E(0)` | +0.857 | +0.786 |
  | Fisher `C_F(0)` | −0.095 | −0.643 |

  Along the trajectories the Euclidean triple predicts the actual `dJ_G/dt` to relative error
  ≤ 1.4e-10 (Prop. 1 with `M = I`). The Fisher triple is off by up to ~1/q (≈1000×), though the
  sign agrees because both are >= 0 in this family.
- The natural-gradient gold-race relation fails under vanilla, as expected (max deviation 4.5–15
  in log units).

**Unexpected observations.**

1. The three P4 failures (diagnosed as numerical, D1).
2. Under vanilla the danger ordering across structures reverses.
3. The terminal proxy `J_V = 1` in every natural-gradient run.

**Hypothesis revised?**

- Registered E003 hypothesis: **no**. Props. 8–10 are supported; P4 failed as registered.
- Working hypotheses (theory note §7):
  - WH-2 (metric matching) gains exploratory support from E003-V.
  - WH-5 (path, not snapshot) is supported by the confirmed inversion.
  - New **WH-6**: the incremental value of geometry must be tested against the time evolution of
    static metrics, under estimation noise and at matched rollout budget. In a noiseless toy any
    informative prefix summary suffices.

**Follow-up.** Pending the collaborator's review. No new experiment has been started.

---

### E000c — E002 design-phase checks (retroactive, NOT pre-registered)

Date: 2026-09-25. Run in the scratch directory: exact enumeration + autodiff on 300 random
Candidate-3 states (single, AND, OR; `q ∈ [0.01, 0.5]`). This is not E002 code. It informs
`05_e002_design.md` v2, facts F2–F4. Max relative errors:

- F2: `dFPR/dt = C^2/(1-q)` under natural gradient, 2.0e-15 (all kinds);
- F3: `dJ_V/dt = (1-FPR)^2 q(1-q) + C^2`, 2.3e-15;
- F4: the per-sample noise of the plug-in `C_hat^2` equals `(1-q) kappa^2 / S + S q - ||g_e||^2`,
  1.9e-15 (single and AND only).

Consequence: E002 predictions that rely on these facts are not blind to them.

---

### E002 — pre-registration (E002a instrumentation + E002b budget-matched prediction)

Experiment IDs: E002a, E002b.
Date: 2026-09-25. Written and committed **before any E002 code exists.** Design memo:
`05_e002_design.md`. Theory: theory note §12 (oracle equivalence).

#### 0. Question and framing (frozen)

At matched certification budgets, does an optimizer-conditioned gradient-geometry diagnostic
predict long-run **gold** failure earlier, more cheaply, or more reliably than simple short probes?

Within Candidate 3, geometry has **no first-order information** that an infinitesimal probe lacks
(theory note §12: `C^2 = (1-q)·dFPR/dt`, `dJ_V/dt = (1-f)^2 q0(1-q0) + C^2`). E002 compares:

- statistical efficiency;
- gold-label efficiency;
- rollout efficiency;
- the ability to diagnose without optimizing.

It does **not** test oracle information superiority. Design-phase checks E000c are disclosed and
are not E002 results.

#### 1. Operating points and target FPR

| panel | role | `q0` | FPR grid |
| --- | --- | --- | --- |
| **P-mod** | primary | 0.05 | `{0.05, 0.1, 0.2}` |
| **P-rare** | secondary | 0.002 | `{0.005, 0.01, 0.02}` |

**Rule.** Per panel, choose the `f` whose **design-split** stall fraction, computed from exact
targets only (§4), is closest to 0.5 (ties go to the smaller `f`). This happens before any
predictor is computed. If the chosen stall fraction lies outside `[0.2, 0.8]`, the panel is flagged
for review rather than silently changed.

#### 2. Verifier panel (per operating point; the same raw draws are used for every `f`)

- **Policy.** Independent Bernoulli `(corr, z_1..z_m)` with logits; `P(corr = 1) = q0`.
- **Verifier.** `V = corr OR 1_E(z) OR xi`, with a fresh coin `xi ~ Bern(p)` per query; FNR = 0.
- **Types** (registered order) and events:

  | type | event `E` |
  | --- | --- |
  | SINGLE | `z1` |
  | AND2 / AND3 / AND4 | all features |
  | OR2 / OR3 | any feature |
  | THR23 | at least 2 of 3 |
  | AOR | `(z1 ∨ z2) ∧ z3` |
  | OAND | `(z1 ∧ z2) ∨ z3` |
  | MIX | base ∈ {SINGLE, AND2, OR2}, uniform, plus a coin |
  | RFP | no event (coin only), `p = f` |

- **Counts:** 89 per feature type (10 types, 890 structures) plus 10 RFP = 900.
- **Random draws**, in a fixed order from `numpy.random.default_rng(SeedSequence(20260925))`:
  - raw feature logits `a_i ~ N(0, 2^2)` i.i.d.;
  - for MIX, a controllable share `rho ~ U(0.05, 1)`.
- **Matching by one common shift `c`:**
  - `s_i = sigmoid(a_i + c)`, where `c` solves `F(c) = f` (brentq on `[-40, 40]`; `F` is
    monotone);
  - for MIX, the event targets `S_E = 1 - (1-f)^rho` and the coin is `p = 1 - (1-f)^(1-rho)`;
  - consequently accuracy, FPR, FNR, FP mass, `A` and `alpha` at `t = 0` are equal across
    structures (Prop. 8, §12).
- **Rejection.** If any `s_i ∉ [1e-4, 1 - 1e-4]`, redraw (same stream; redraw counts are reported).
- **De-duplication (parameters only).** The canonical form sorts features within symmetric groups
  (AND/OR/THR: all; AOR, OAND: the first two). Reject a structure whose canonical logit vector
  (plus `rho` for MIX) is within 1e-3 max-abs of an earlier same-type structure. RFP instances have
  no parameters and are intentional identical controls, so they are exempt.
- **Split.** Fixed before any target or predictor is computed, by a seeded permutation within each
  type:
  - the first 6 feature types (registered order) put 30 structures in design and 59 in test;
  - the last 4 feature types put 29 in design and 60 in test;
  - RFP: 4 design, 6 test;
  - totals: **300 design / 600 test**.
- **Integrity mechanism.**
  - The panel file (parameters and split labels only; **no test targets or predictors**) is
    written once and its SHA-256 committed.
  - The loader returns test structures only if the file `configs/e002/HELDOUT_APPROVED` exists.
    That file is created only after explicit approval. A unit test asserts that the loader
    refuses otherwise.
  - Pilot scripts request `split = "design"` only, and every run records which split it touched.
  - Test-split targets are not computed in the pilot round.
- **Secondary evaluation plan:** leave-one-type-out on the test split (reported, not primary).

#### 3. Budgets and accounting

Costs are tracked separately and shown in every table:

- `B_roll`: generated policy responses (a perturbed variant counts as one rollout);
- `B_gold`: gold calls;
- `B_bwd`: per-sample score/gradient evaluations;
- plus `k`, the number of sequential updates.

Verifier calls are free and reported.

- **Gold mode A (primary; the only mode implemented):** every audited response costs one gold
  label.
- **Mode B** (accepted-only auditing) is deferred and **not** part of this registration.
- **Grid:** `B_gold ∈ {16, 32, 64, 128, 256, 512, 1024}` × `B_roll/B_gold ∈ {1, 4, 16}` (21 cells).
- **Primary cells:** `(B_gold, B_roll) = (64, 256)` and `(256, 1024)`. The other cells form the
  frontier (secondary).
- **Panel constants known to all arms:** `q0` and `f`, the policy log-prob function, the verifier
  as a black box, and gold as a costed black box. No arm knows the parameter blocks, exact
  gradients or targets.

#### 4. Targets (exact; exact natural-gradient flow = the training run)

- **q-free feature ODE.** In `tau = ∫(1-q)dt`: `dphi_i/dtau = dF/ds_i`,
  `dLambda/dtau = 1 - F`, `q = q0 exp(Lambda)`, `dt/dtau = 1/(1 - q)`. It is integrated with
  DOP853 (`rtol 1e-10`). `1 - F` is computed from non-event pattern probabilities, so there is no
  cancellation. Closed forms (Prop. 9) are used where they exist and must agree to 1e-8.
- **Primary continuous target:** `D = 1 - J_G(∞)`, the shortfall relative to the clean verifier
  (the clean natural-gradient run reaches 1).
- **Primary categorical target:** stall iff `Lambda(∞) < log(1/q0)`.
- **Secondary targets:**
  - `J_G(T = 25)`;
  - `J_G(∞)`;
  - `t95` (stalls only), the time at which `J_G = q0 + 0.95(q∞ - q0)`.
- **QA.** On a random 10% of **design** structures, the full generic natural-gradient ODE (E003
  code) must agree with the targets:
  - the outcome class must match;
  - stalls: `|J_G(T_end) - q∞| <= 2e-3` at `T_end = 500`;
  - successes: `|FPR(T_end) - F*| <= 1e-4`.
- **Exclusion.** Structures whose target integration fails; counts are reported, and if more than
  1% fail the run stops.
- **Never used as a target:** the signed proxy–gold gap.

#### 5. Arms (exact; at most three tuned hyperparameters each)

Common definitions:

- Scores: `score = (corr - q, z - s)`, hand-written; they equal the autodiff score for this policy.
- RLOO gradient: `g_hat[R] = (1/n) Σ (R_i - b_i) score_i`, with `b_i` the mean of the other
  rewards.
- Estimated Fisher: `F_hat = mean(score score^T)`, damped as `F_hat + lam·(tr F_hat / d)·I`.

**G0 — static metrics** (sanity baseline only; at chance by construction).

- Uses `m0 = B_gold` audited `pi_0` rollouts; score = audited FPR.
- Cost: `B_roll = m0`, `B_gold = m0`, `B_bwd = 0`.

**G1 — geometry at `t = 0`** (estimated optimizer-matched metric, i.e. estimated Fisher).

- Data: `m0 = B_gold` audited rollouts plus `B_roll - m0` unlabeled `pi_0` rollouts. `F_hat` uses
  all `B_roll` rollouts.
- **Plug-in estimator:**
  - `g_hat_G` = RLOO on the audited set;
  - `g_hat_V` = RLOO on the audited set (paired) or on all rollouts (pooled);
  - `(A_hat, alpha_hat, C_hat) = decompose(g_hat_G, g_hat_V, M_hat)`.
- **U-statistic estimator** (audited set, no baseline): `gamma_i^G = G_i score_i`,
  `gamma_i^e = (V_i - G_i) score_i`, and
  `Gram_ab = [(Σgamma^a)^T M (Σgamma^b) - Σ gamma_i^a^T M gamma_i^b] / (n(n-1))`. Then:
  - `A_hat^2 = Gram_GG`;
  - `C_hat^2 = Gram_ee - Gram_eG^2 / Gram_GG` if `Gram_GG > 0`, otherwise `Gram_ee`;
  - raw values are allowed to be negative.
- **`A_hat = 0` (or `Gram_GG <= 0`):** `alpha = NaN` and `alpha_defined = False`; `C_hat` is then
  the full residual `||g_hat_e||`. This is never imputed.
- **Ranking score:** `C_hat^2` (raw).
- **Tuned hyperparameters:**
  - estimator ∈ {plug-in, U};
  - `lam ∈ {1e-3, 1e-2, 1e-1}`;
  - `g_V` source ∈ {paired, pooled} (plug-in only).
- **Oracle variant G1-oracle** (exact Fisher): reported as a ceiling only. **Never eligible for
  success.**
- Cost: `B_roll`, `B_gold = m0`, `B_bwd = B_roll`, `k = 0`.

**P1 — verifier-score growth (zero gold).**

- Run `k` stochastic natural-gradient steps: `theta += eta_p (F_hat + lam_p·tr/d·I)^{-1} g_hat_V`,
  with `lam_p = 1e-2` fixed and `b` fresh rollouts per step.
- `k + 1` batches of size `b = floor(B_roll / (k+1))`: batch 0 at `pi_0`, batch `k` at `pi_k`.
- Score = `mean V(batch k) - mean V(batch 0)`.
- Tuned: `k ∈ {1, 2, 5, 10}`, `eta_p ∈ {0.1, 0.3, 1.0}`.
- Cost: `B_gold = 0`, `B_bwd = k·b`.

**P2 — importance-reweighted FPR growth (zero new gold).**

- Audit: `m0 = B_gold` labeled `pi_0` rollouts.
- Updates: `k` steps with `b = floor((B_roll - m0)/k)` fresh rollouts each. If `b < 2` and
  `k = 1`, the audit rollouts serve as the training batch; otherwise the configuration is
  infeasible.
- `FPR_hat_k = Σ_{G=0} w_i V_i / Σ_{G=0} w_i`, with `w_i = pi_k(y_i)/pi_0(y_i)` (self-normalized;
  the coin draw recorded at audit time is reused).
- Score = `FPR_hat_k - FPR_hat_0`, with `FPR_hat_0` the unweighted audited FPR.
- Tuned: `k ∈ {1, 2}`, `eta_p ∈ {0.1, 0.3, 1.0}`.
- Cost: `B_gold = m0`, `B_bwd = k·b` (plus `m0` forward log-prob evaluations).

**P3 — fresh-label probe.**

- `k` steps with `b = floor((B_roll - m)/k)` (infeasible if `b < 2`), then `m = B_gold` fresh
  labeled rollouts at `pi_k`.
- Score: `FPR_hat_k` (observable "FPR"), or `-(J_G_hat_k - q0)` (observable "gold").
- Tuned: `k ∈ {1, 2, 5, 10}`, `eta_p ∈ {0.1, 0.3, 1.0}`, observable ∈ {FPR, gold}.
- Cost: `B_gold = m`, `B_bwd = k·b`.

**P4 — local-perturbation audit** (no training, no gradients).

- Audit `m0 = B_gold` labeled `pi_0` rollouts.
- For each audited response that is wrong and rejected (`G = 0`, `V = 0`): resample `r` distinct
  randomly chosen features from the policy marginal and query `V` on each variant. Also re-query
  `V` once on the unchanged response, to control for verifier randomness.
- Score = `mean V(variants) - mean V(re-queries)`.
- Variants count as rollouts. If `m0 + #variants > B_roll`, rejected-wrong items are subsampled
  uniformly to fit; infeasible if none fit.
- Variants inherit the parent's gold label. This is a declared toy assumption (the features do
  not change correctness) and it **favours P4**.
- Tuned: `r ∈ {1, 2, all}`.
- Cost: `B_gold = m0`, `B_bwd = 0`.

No ensemble is registered.

#### 6. Tuning protocol (design split only)

- For each arm × budget cell, choose the hyperparameters that maximize the **design-split mean
  C-index** over `R_design = 32` replications.
- Ties go to the simpler configuration (smaller `k`, then smaller `eta_p`, then plug-in, smaller
  `lam`, paired, smaller `r`).
- The same configuration is used for AUROC.
- Frozen configurations are committed before any held-out evaluation.

#### 7. Endpoints and inference (held-out round; not run now)

- **Primary endpoints:**
  - C-index: Harrell's concordance with `D` over pairs with distinct `D`; predictor ties count ½.
  - AUROC for stall (Mann–Whitney; ties ½).
- **Secondary endpoints:** Kendall τ_b, Spearman, and recall of the top 10% by `D`.
- **Replications:** `R = 64` per arm × cell on the test split. The panel metric is computed per
  replication and averaged.
- **CI:** hierarchical bootstrap (2000 resamples; structures, then replications).
- **Primary tests:** at the 2 primary cells × 2 endpoints, the statistic is
  `Delta = metric(G1) - max_j metric(P_j)` over the required competitors P1–P4. The max is
  recomputed inside every bootstrap resample. One-sided at α = 0.05, **Holm** over the 4 tests.
  Pairwise differences against each arm are also reported.
- **Frontier non-dominance (C-index point estimates).** G1 is *dominated* if, for every grid cell,
  some competitor at the same or a lower budget on both axes attains at least G1's value.

#### 8. Success and abandonment (frozen)

**Practical geometry success requires all of:**

1. the estimated metric (G1, not G1-oracle);
2. evaluation on the 600 test structures;
3. at a primary cell, a Holm-significant improvement over the strongest required competitor in
   C-index or AUROC;
4. G1 is not dominated on the frontier.

"Beats G0" is not evidence (matched by construction). P-rare is secondary and reported.

**The diagnostic contribution is ABANDONED for this setting if any of:**

- (a) estimated geometry beats none of P1, P2, P4;
- (b) any apparent win exists only with the exact Fisher;
- (c) the G1 frontier is uniformly dominated by cheaper probes.

The mechanistic contribution is preserved and the negative diagnostic result is reported.

#### 9. Oracle ceilings (registered procedure)

Noiseless signals per structure:

- `C(0)` (exact Fisher);
- `FPR(t_p)`, `J_V(t_p) - J_V(0)`, `J_G(t_p)` at `t_p ∈ {0.1, 0.3, 1, 3, 10}` (exact flow);
- the P4 oracle: expected directional acceptance gain over the rejected-wrong distribution,
  `r = all`;
- G0 (constant).

Their C-index and AUROC are computed on the **design split in this round, before any
finite-sample pilot**, and on the **test split as the first committed step of the held-out round,
before any finite-sample test evaluation**.

**Registered predictions (§12):**

- the ceilings of `C(0)`, `FPR(t_p -> 0)` and `J_V(t_p -> 0)` coincide;
- the ceiling of `J_G(t_p -> 0)` is 0.5;
- the G0 ceiling is 0.5.

#### 10. E002a — estimator characterization

- **Structures** (at each panel's selected `f`, `q0` as in the panel):
  - SINGLE; AND2-sym; AND2-asym (`s1/s2` ratio 1:10 before shift); OR2-sym; RFP;
  - the MIX-SINGLE dose family, `rho ∈ {0, .05, .1, .2, .35, .5, .75, 1}`;
  - the A-extreme family: SINGLE exploit at FPR `f` with `q ∈ {1e-3, 1e-2, q0, 0.9, 0.99}`.
- **Sample sizes:** `N ∈ {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}`.
- **Estimators:**
  - metrics: exact Fisher (oracle), estimated Fisher (`lam` grid), Euclidean (reported; it is
    the metric matched to a *vanilla* optimizer);
  - plug-in (RLOO) and U-statistic;
  - Gram-form `P = alpha A^2` and `D = A^2 C^2`.
- **Metrics:**
  - bias, SD and RMSE of `A_hat`, `alpha_hat` (conditional on being defined, plus the
    undefined rate) and `C_hat`;
  - percentile-bootstrap 95% CI coverage for `C` (200 bootstrap resamples);
  - false-alarm rate at the RFP-calibrated threshold, which must equal the nominal 5% up to
    Monte Carlo error;
  - power; minimum detectable `C` (80% power at 5% false alarm, interpolated over the dose
    family); `P(C_hat_Y > C_hat_RFP)`.
- **Degenerate-event rates**, with their **registered exact predictions**:
  - `P(A_hat = 0)` under RLOO equals `q^N + (1-q)^N`;
  - `P(no false positive in the batch)` equals `(1 - (1-q) f)^N`;
  - `P(some feature constant in the batch)` equals `1 - Π_i [1 - s_i^N - (1-s_i)^N]`, an upper
    bound under dependence (the features are independent here, so it is exact);
  - the observed rates must agree within 3 Monte Carlo SE.
- **Further predictions:**
  - under the oracle metric, the U-statistic `C_hat^2` is unbiased (|bias| within 3 MC SE);
  - the per-sample noise of the plug-in follows F4 (single/AND) within 3 MC SE;
  - plug-in `C_hat` is biased upward near `C = 0`;
  - coverage near `C = 0` is below nominal.
- **Replications:** `R = 2000` in the formal E002a run, `R = 200` in the pilot.

#### 11. This round's scope

- oracle ceilings (design split);
- E002a pilot (`R = 200`);
- E002b design-split tuning and runtime pilot (`R_design = 32`).

**No test-split target, predictor or metric is computed.** The formal E002a (`R = 2000`) and the
held-out E002b require explicit approval. Any change to this block is recorded as a dated
amendment before the corresponding formal run.

#### 12. Agent's prior (not a hypothesis)

G1 ties P1/P2 at small budgets and is beaten by P3 once `t_p` covers exploit takeoff; any G1 win is
most likely confined to ratio-1 cells or none.

**Code/config** (to be written after this commit): `src/vdyn/verifiers/boolean_fp.py`,
`src/vdyn/geometry/gold_race.py`, `src/vdyn/e002/{panel,estimators,probes,endpoints}.py`,
`experiments/e002/*`, `configs/e002/e002.toml`.

**E002 Amendment 1** (2026-09-25; written during implementation, **before any panel, target,
predictor or pilot output exists**). Each item fixes an ambiguity or infeasibility found while
implementing. The pre-registration block above is unchanged.

1. **SINGLE has no free parameters after matching.** The common shift forces `s = f`, so all
   SINGLE draws are the same structure. As registered, the de-duplication rule could never be
   satisfied and panel generation loops forever. Fix:
   - SINGLE becomes an identical-control class like RFP: 10 instances, exempt from
     de-duplication.
   - The 79 freed slots are reassigned evenly to the other feature types: AND2, AND3, AND4, OR2,
     OR3, THR23 and AOR get +9 (98 each); OAND and MIX get +8 (97 each). The total stays 900.
   - Design counts per type use largest-remainder apportionment of 1/3 per class, with a total
     of exactly 300 (ties broken by registered type order). SINGLE and RFP get 4 design and
     6 test each.
2. **Shared raw draws across the FPR grid.** A raw draw is accepted only if the matched features
   stay inside `[1e-4, 1 - 1e-4]` for **every** `f` in the operating point's grid. This implements
   "the same raw draws are used for every `f`" together with the rejection rule. De-duplication is
   evaluated at the first grid value; a constant shift makes it `f`-invariant.
3. **Seeds.** `SeedSequence(20260925).spawn(4)` gives `[P-mod draws, P-mod split, P-rare draws,
   P-rare split]`.
4. **Numerical tolerance for the categorical target.** Stall iff `q∞ < 1 - 1e-6`, the same
   success threshold as E003. A unit-test structure sat within 1e-11 of the exact threshold, where
   the raw rule `Lambda(∞) < log(1/q0)` is decided by floating-point noise.
5. **Target integration.** The `tau`-ODE stops when `1 - FPR < 1e-12` (cancellation-free), with
   `tau_max = 1e10`. The remaining tail `∫(1 - FPR) dtau` is added analytically for a power-law
   tail `c tau^-p`, with `p` estimated from the solution between `tau_e/2` and `tau_e` (the tail
   is 0 if `p > 30`). This is validated against the Prop. 9 closed forms, including symmetric OR,
   to 1e-8 relative.

**E002 panel freeze** (2026-09-25). The panel files were written by
`experiments/e002/write_panels.py` (commit `8272c02` code); they hold parameters and split labels
only. No target or predictor had been computed.

- `configs/e002/panel_P-mod.json`: sha256
  `60e27ed0e99c0ca0b428404e44fe525253ee15a0aec76f9e259359b53cadc9e3` (900 slots, 29 redraws).
- `configs/e002/panel_P-rare.json`: sha256
  `90add2d8acff344c431d7e48e68cfc9da91f3d2f3ec4911123c2377f9ea3d7f1` (900 slots, 124 redraws).

The held-out split is sealed: `configs/e002/HELDOUT_APPROVED` does not exist.

---

### E002 — design-split pilot record (2026-09-25; DESIGN SPLIT ONLY; not E002 results)

Scope as registered in E002 §11. No test-split target, predictor or metric was computed;
`configs/e002/HELDOUT_APPROVED` does not exist. Every run's `metadata.json` records
`split = design`. Post-hoc items are labelled **[post-hoc]**.

#### A. Targets and FPR selection (§1, §4)

Run `results/E002-pilot-targets/20260925T083746Z_81ffa61` (script commit `81ffa61`; 16 s per
panel for targets, 8 workers).

| panel | stall fraction by `f` | selected `f` | flag | QA tolerance / class |
| --- | --- | --- | --- | --- |
| P-mod (`q0 = 0.05`) | 0.05: 0.000, 0.1: 0.480, 0.2: 0.953 | **0.1** | no | 30/30, 30/30 |
| P-rare (`q0 = 0.002`) | 0.005: 0.373, 0.01: 0.423, 0.02: 0.497 | **0.02** | no | 30/30, 30/30 |

No target integration failed (exclusions: 0). `D` is 0 for about half of the structures (all
successes tie at 0; the C-index uses only pairs with distinct `D`).

Stall fraction by type at the selected `f` (design): P-mod — AND2 .45, AND3 .15, AND4 .15,
OR2 1.0, OR3 .91, THR23 .13, AOR .41, OAND .91, MIX .22, SINGLE 1.0, RFP 0. P-rare — AND2 .18,
AND3 .15, AND4 .03, OR2 1.0, OR3 1.0, THR23 .22, AOR .25, OAND .97, MIX .69, SINGLE 1.0, RFP 0.

#### B. Oracle ceilings, design split (§9); committed before any finite-sample pilot

Run `results/E002-pilot-oracle/20260925T083916Z_955e4c1` (commit `5aa546b`).

| signal | P-mod C-index / AUROC | P-rare C-index / AUROC |
| --- | --- | --- |
| G0 (constant) | 0.500 / 0.500 | 0.500 / 0.500 |
| `C(0)`, exact Fisher | 0.9355 / 0.9857 | 0.9274 / 0.9714 |
| `dFPR/dt(0)`, `dJ_V/dt(0)` | 0.9355 / 0.9857 | 0.9274 / 0.9714 |
| `-dJ_G/dt(0)` | 0.500 / 0.500 | 0.500 / 0.500 |
| FPR growth, `t_p = 0.1 / 1 / 3 / 10` | .936 / .944 / .962 / .988 (AUROC .999 at 10) | .929 / .947 / .968 / .974 (AUROC 1.000 at 10) |
| `J_V` growth, `t_p = 10` | 0.737 / 0.647 | 0.887 / 0.876 |
| `-(J_G(t_p) - q0)`, `t_p = 10` | 0.980 / 0.996 | 0.995 / 1.000 |
| P4 oracle (`r = all`) | **0.401 / 0.400** | **0.267 / 0.181** |

Registered predictions: the `C(0)`, `FPR(t_p -> 0)` and `J_V(t_p -> 0)` ceilings coincide —
**PASS**; the `J_G(t_p -> 0)` ceiling is 0.5 — **PASS**; the G0 ceiling is 0.5 — **PASS**.

Observations (not predictions):

- Every probe ceiling exceeds the geometry ceiling once `t_p >= 0.1`, and by 0.05 (C-index) at
  `t_p = 10`.
- `J_V` growth degrades at long horizons because legitimate learning also raises `J_V`.
- **The P4 oracle is anti-predictive under its registered orientation.** Under OR-type events a
  rejected wrong response has no feature present, so resampling one feature rarely triggers
  acceptance; under AND-type events it is often one feature short, with large `s_i`. OR
  structures stall and AND structures succeed, so the gain runs opposite to the danger. The
  orientation is **not** flipped post hoc; see §F.

#### C. E002a pilot (§10), `R = 200`

Run `results/E002a-pilot/20260925T084434Z_33f3ea4` (script commit `33f3ea4`; 112 s wall,
550 MB peak RSS). Fixed non-panel structures at the selected `f` of each panel.

Registered checks:

1. **Degenerate-event rates: PASS.** Observed rates of `A_hat = 0` (plug-in), all-gold-equal,
   no-false-positive and some-feature-constant match the exact predictions:
   - P-mod: 0 of 700 checks outside the band;
   - P-rare: 4 of 700 outside, i.e. 2 distinct batches, each counted twice because `A_hat = 0`
     and all-gold-equal are the same event (about 1.9 expected by chance).

   **[post-hoc]** The band is an exact two-sided binomial test at the 3-SE level (`p < 0.0027`).
   The registered `3·sqrt(p(1-p)/R)` band is zero when `p ≈ 0`, and the observed count is then
   trivially 0.
2. **U-statistic `C_hat^2` unbiased under the oracle metric: FAIL (preserved).** `|bias| > 3 MCSE`
   in 35 of 177 cells (P-mod) and 29 of 163 cells (P-rare); cells with zero sample SD are
   excluded. Two causes:
   - **(i) Registration error.** Only the Gram entries are U-statistics. `C^2 = Gram_ee -
     Gram_eG^2 / Gram_GG` contains a ratio and is not unbiased; for example RFP at `N = 4096` has
     bias `-3.6e-6` with MCSE `2e-8`.
   - **(ii) Rare-event cells** (P-rare, `N <= 64`). The sample MCSE badly underestimates the true
     one: the U-statistic drops the diagonal, so a lone false positive contributes nothing and the
     distribution is heavy-tailed.

   **[post-hoc]** The unbiased Gram entry `P = Gram_eG` passes: 1 of 177 and 4 of 163 cells.
3. **F4 per-sample noise: PASS.** Empirical vs formula:
   - P-mod: SINGLE 0.7745 ± 0.0048 vs 0.7783; AND2 0.37631 ± 0.0023 vs 0.37633;
   - P-rare: SINGLE 0.966 ± 0.015 vs 0.959; AND2 0.2331 ± 0.0036 vs 0.2376.
4. **Plug-in `C_hat` biased upward near `C = 0`: NOT SUPPORTED on the registered (`C`) scale.**
   - At dose `rho = 0.05` (P-mod, `C = 0.062`) the bias is +0.004, −0.003, −0.002 and −0.001 at
     `N = 16, 64, 256, 1024`.
   - **[post-hoc]** The upward bias is present in `C^2`: +0.048 at `N = 16` vs `C^2 = 0.0039`
     (12×). On the `C` scale it is cancelled by the atom at `C_hat = 0` (batches without a false
     positive) and by the concavity of the square root.
   - The exact-`C = 0` control (RFP) cannot test the claim. With one parameter (`d = 1`),
     `C_hat ≡ 0` whenever `A_hat > 0`.
5. **Coverage near `C = 0` below nominal: NOT ASSESSABLE** as run, for the same `d = 1` reason;
   small-dose structures were not in the coverage set. For `C > 0`, percentile-bootstrap coverage
   is:
   - P-mod: 0.90–0.985 at `N = 64–1024`;
   - P-rare: 0.66–0.77 at `N = 64` (below nominal) and 0.885–0.945 at `N >= 256`.
6. **False alarm at the RFP-calibrated threshold equals 5%: FAIL (preserved).** Plug-in with the
   exact metric gives 0.00–0.03 across `N` (P-mod); estimated-metric variants give 0.00–0.10. The
   RFP null is discrete (`d = 1`), so the 95% quantile sits on an atom.
7. **Power / minimum detectable `C`: INVALID as an instrument (design flaw).**
   - The RFP null has `d = 1`, while every tested structure has `d >= 2`. "Detection" therefore
     partly measures the extra dimension: noise in the orthogonal direction alone gives
     `C_hat > 0`.
   - At `N >= 512` the minimum detectable `C` is bounded by the dose grid (smallest nonzero dose
     `C = 0.062` P-mod, `0.031` P-rare).

Other E002a observations:

- **[post-hoc] The plug-in `C_hat^2` noise floor depends on structure, not only on `N`.** At
  `N = 16` the bias is:
  - P-mod: +0.023 SINGLE, +0.017 AND2, +0.069 OR2 (`C^2` = 0.081, 0.039, 0.079);
  - P-rare: +0.146 on OR2 (`C^2 = 0.019`).

  A structure-dependent floor is a spurious type signal: OR-type structures have larger floors and
  also stall more. The U-statistic removes most of it.
- A-extreme family: `alpha` is undefined in 94% (`q = 0.001`) and 53% (`q = 0.01`) of batches at
  `N = 64`. `C_hat` stays usable (RMSE ≈ 1/3 of `C`), because `C` falls back to `||g_e||`.

#### D. E002b design-split tuning and runtime pilot (§5, §6), `R_design = 32`

Run `results/E002b-design-pilot/20260925T085007Z_673a370-dirty`: script commit `673a370`,
2088 s wall, 8 workers, 827 MB peak RSS in the main process and 332 MB per worker.

**Disclosure.** The `-dirty` flag comes from one untracked output directory
(`results/E002a-pilot/`) present at launch. No tracked file differed from `673a370`.

All numbers are **in-sample**: arms are tuned and evaluated on the same 300 design structures.
To size the selection optimism, the tuned configuration was also re-selected on replications
1–16 and evaluated on replications 17–32. That estimate agrees with the in-sample one to within
0.008 at the primary cells (0.013 at worst over all cells), so replication-level selection optimism
is negligible.
Structure-level optimism is not measured.

Tuned C-index / AUROC at the primary cells. Costs are `B_roll / B_gold / B_bwd`; the probe
columns also give the tuned configuration.

| panel, cell | G1 | G1-oracle | P1 | P2 | P3 | P4 |
| --- | --- | --- | --- | --- | --- | --- |
| P-mod (64, 256) | .793 / .883 | .747 / .834 | .772 / .860 (k2 η1) | .818 / .914 (k2 η1) | **.825 / .909** (k5 η1 FPR) | .480 / .470 (r1) |
| P-mod (256, 1024) | .867 / .959 | .837 / .939 | .851 / .948 | .876 / .970 | **.901 / .976** | .458 / .442 |
| P-rare (64, 256) | .787 / .862 | .706 / .781 | **.824 / .905** (k5 η1) | .652 / .690 | .808 / .886 | .468 / .450 |
| P-rare (256, 1024) | .860 / .946 | .810 / .909 | .924 / .986 | .858 / .947 | **.926 / .986** | .426 / .391 |

Costs at (64, 256) for P-mod:

| arm | `B_roll` | `B_gold` | `B_bwd` |
| --- | --- | --- | --- |
| G1 | 256 | 64 | 256 |
| P1 | 255 | 0 | 170 |
| P2 | 256 | 64 | 192 |
| P3 | 254 | 64 | 190 |
| P4 | 256 | 64 | 0 |

Every cell's costs are in the run JSON.

**`G1 - max_j P_j`** (P1–P4; in-sample; 200 hierarchical-bootstrap resamples; the max is taken
inside each resample):

| panel, cell | C-index | AUROC |
| --- | --- | --- |
| P-mod (64, 256) | −0.031 [−0.043, −0.020] | −0.032 [−0.047, −0.019] |
| P-mod (256, 1024) | −0.034 [−0.043, −0.025] | −0.018 [−0.029, −0.009] |
| P-rare (64, 256) | −0.037 [−0.052, −0.024] | −0.042 [−0.061, −0.026] |
| P-rare (256, 1024) | −0.066 [−0.077, −0.056] | −0.040 [−0.055, −0.028] |

With G1-oracle in place of G1, the difference is −0.038 to −0.124.

Frontier (C-index point estimates, nominal cells as registered):

- G1 is **not** uniformly dominated. It is non-dominated only at (16, 64) in both panels, where it
  leads P1 by 0.015 (P-mod: .706 vs .691) and 0.019 (P-rare: .633 vs .614), and at (32, 128) in
  P-rare.
- **[post-hoc]** Dominance recomputed from each arm's *actual* costs, instead of nominal cells,
  leaves the same cells non-dominated.

Tuning behaviour:

- **G1** always selects plug-in, pooled `g_V`, `lam = 0.1`:
  - pooled vs paired is worth +0.08 to +0.17 C-index at the primary cells;
  - `lam` is worth at most 0.016 there;
  - the U-statistic is 0.10–0.28 worse there.
- **G1-oracle is never above G1**: it is below by up to 0.088, with one tie at (16, 16) in
  P-rare. **[post-hoc hypothesis, untested]**
  The exact-Fisher metric weights direction `i` by `1/(s_i(1-s_i))`, which gives the
  structure-dependent noise floor seen in E002a. An estimated `F_hat` from the same batch
  whitens the noise per structure, like a Hotelling statistic. **Consequence:** exact Fisher is
  not a finite-sample ceiling for G1.
- **Probes** select `eta = 1.0`, the grid edge, in 107 of 112 probe cells. The optimum is an
  interior ridge in `k·eta ≈ 3–5`: for example (10, 0.3) ≈ (5, 1.0), and (10, 1.0) degrades.
  The edge is therefore not binding.
- The P3 gold observable is much worse than the FPR observable.

**P4 is below 0.5 in every cell**, finite-sample (0.369–0.495), as the oracle ceiling predicted. It
cannot be the strongest competitor, and G1 always beats it.

#### E. Items that need the collaborator's decision (NOT applied; no amendment made)

1. **P4 orientation.** The registered orientation is anti-predictive: oracle C-index 0.40
   (P-mod) and 0.27 (P-rare). Options:
   - (a) keep it as registered and disclose;
   - (b) add the sign as a design-tuned hyperparameter of P4;
   - (c) drop P4 from abandonment rule (a).

   Under the registered rule, "G1 beats P4" is automatic, so abandonment condition (a) ("beats
   none of P1, P2, P4") **cannot trigger**. Flipping the sign would not change any comparison
   with G1: the flipped finite-sample C-index is ≤ 0.63.
2. **Decision gap.** The design pilot predicts:
   - no practical success (G1 < max P at every primary cell);
   - no abandonment ((a) is blocked by P4 and by G1 > P1 on P-mod; (b) does not apply; (c) fails
     because of one non-dominated low-budget cell).

   The registration does not name this outcome. A rule for it is needed before the held-out run.
   Changing a decision rule after seeing design data is a forking path, and it should be
   disclosed as such.
3. **E002a instrumentation** (formal run only):
   - add a dimension-matched null (policy features that do not enter the event, coin `= f`);
   - add small-dose structures to the coverage set;
   - restate the U-statistic claim for the Gram entries only;
   - use exact binomial bands for degenerate rates.
4. **Endpoint engineering.** At `n = 600` the registered bootstrap (2000 resamples × 64
   replications) costs about 8.7e6 C-index evaluations per panel, about 13 CPU-hours with the
   current O(n²)-per-call code. Precomputing per-replication concordance matrices makes each
   resample one quadratic form, about 100× faster. This is a pure refactor, and it needs an
   equivalence test first.

#### F. Held-out resource estimate (not run)

- **Simulation**, frozen configurations only (600 structures, `R = 64`): 1.1–3.0 CPU-hours, about
  11–29 min wall on 8 workers.
- **Test-split targets and oracle ceilings:** under 2 min.
- **Bootstrap:** about 13 CPU-hours with the current code, or minutes after the refactor in E.4.
- **Memory:** under 1 GB per worker and about 1 GB in the main process.

Tuned configurations (candidate freeze): `configs/e002/e002b_tuned_configs.json`, sha256
`674c882340328d0b497af9464bb1e5d973a709b316b9276694dbdefae500d880`. They are extracted from the
run JSON (sha256 `83f662c7bd0fda0839f59c82e888e815cace455d6101313d349871b84e89650e`).

---

**E002 Amendment 2** (2026-09-25; written and committed before the code that implements it)

**Disclosure.** Written **after** the design-split pilot (design data seen). The test split is
sealed: `HELDOUT_APPROVED` is absent and no test target, predictor or metric exists. The
collaborator decided items 1–5 after reviewing the pilot record. The pre-registration block,
Amendment 1 and the pilot record are unchanged.

1. **P4 orientation: unchanged** (higher resampling gain = more danger).
   - **[post-hoc, secondary]** The sign-flipped P4 is reported alongside. It never enters a
     success or abandonment decision.
   - **Disclosure.** Under the registered orientation, P4 is anti-predictive on design (oracle
     C-index 0.40 / 0.27). Abandonment condition (a) is therefore decided in practice by P1 and
     P2.
2. **Verdict categories** (exhaustive and mutually exclusive), computed by code on **P-mod**
   (primary); P-rare gets the same categories as a secondary report:
   - **SUCCESS**: all four §8 success conditions hold.
   - **ABANDON**: any §8 abandonment condition holds. Operationally, on the test split:
     - (a) at no primary cell × primary endpoint does G1's point estimate exceed that of P1, P2
       or P4 (the lenient reading, which makes abandonment harder);
     - (b) G1-oracle has a point estimate above `max_j P_j` at some primary cell × endpoint, and
       G1 has none;
     - (c) G1 is dominated at every grid cell (§7, nominal cells, C-index point estimates).
   - **NO PRACTICAL ADVANTAGE**: neither of the above. Practical superiority is not established,
     but the letter of §8 does not trigger abandonment. The diagnostic claim is reported as **not
     supported**.
   - **Scope.** Every verdict applies to Candidate 3 under natural gradient, with the matched
     P-mod and P-rare panels only.
3. **Inference details.**
   - One-sided bootstrap p-value: `p = (1 + #{b : Delta*_b <= 0}) / (B + 1)`, with
     `B = 2000`.
   - Holm at `alpha = 0.05` over the 4 P-mod tests (2 primary cells × {C-index, AUROC}).
     P-rare is reported unadjusted.
   - The bootstrap is computed from per-replication concordance kernels, one quadratic form per
     resample. The resampling draws and their order are identical to the E002 §7 reference
     implementation, and the results agree with it to 1e-12 (tested).
   - Seeds: `heldout_seed = 20260927`.
4. **Leave-one-type-out** (secondary; this makes §2's plan concrete). The primary `Delta` (both
   endpoints, both primary cells) is recomputed on the test split with each of the 11 types
   removed in turn. The min and max are reported; there is no re-tuning.
5. **E002a instrumentation for the formal run** (the pilot record stays as it is):
   - **Dimension-matched null.** Each detection structure `Y` gets its own null `Y0`: the same
     `q`, the same feature marginals `s(Y)`, the event removed, and coin `= f`. Then
     `C(Y0) = 0` exactly and `d(Y0) = d(Y)`.
     - The threshold is the 95% quantile of `C_hat^2` under `Y0`, from one sample.
     - False alarm is measured on an independent `Y0` sample.
     - Every dose level has its own null. The `d = 1` RFP null is kept for continuity only.
   - **Coverage set:** add dose `rho ∈ {0.05, 0.1}` and the matched nulls (`C = 0`).
   - **U-statistic prediction, restated.** The raw Gram entries `Gram_GG`, `Gram_eG` and
     `Gram_ee` are unbiased (`|bias| <= 3 MCSE`), on cells with `N(1-q)f >= 5` expected false
     positives per batch. The ratio bias of `C_hat^2` is reported descriptively. The original
     claim stays recorded as FAILED in the pilot.
   - **Degenerate-event rates:** exact two-sided binomial test at `p < 0.0027`.
   - **Upward bias near `C = 0`:** reported on both the `C` and `C^2` scales. The registered
     scale remains `C`.
6. **Held-out execution order (frozen).**
   1. Create `HELDOUT_APPROVED` only after explicit approval.
   2. Compute test-split targets at the design-selected `f` (P-mod 0.1, P-rare 0.02); `f` is not
      re-selected.
   3. Compute test-split oracle ceilings and commit them.
   4. Run E002b on the test split: frozen configurations (`configs/e002/e002b_tuned_configs.json`,
      sha256 `674c8823…d880`), `R = 64`, `heldout_seed`.
   5. Run the analysis script, frozen and committed before unsealing, which prints the verdict.
