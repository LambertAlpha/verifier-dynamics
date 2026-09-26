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

**E002 held-out freeze** (2026-09-25). Analysis code, configurations and seeds are frozen at
commit `d639b09`, before unsealing. `HELDOUT_APPROVED` is absent.

SHA-256 of the frozen files:

| file | sha256 |
| --- | --- |
| `configs/e002/e002.toml` | `a967d3d1…c055ca` |
| `configs/e002/e002b_tuned_configs.json` | `674c8823…00d880` (checked at run start) |
| `configs/e002/panel_P-mod.json` / `panel_P-rare.json` | `60e27ed0…cadc9e3` / `90add2d8…3d7f1` |
| `experiments/e002/heldout_targets.py` | `fd84d213…e82913` |
| `experiments/e002/heldout_oracle.py` | `1e2d9e29…acc64` |
| `experiments/e002/e002b_heldout.py` | `a3a96a2f…c5eef` |
| `experiments/e002/e002b_analysis.py` | `b8d71d58…027ad` |
| `experiments/e002/e002a.py` | `230d4ba4…0469f` |
| `src/vdyn/e002/arms.py`, `verdict.py`, `endpoints.py` | `af63db6d…`, `7afa0603…`, `a3dfdf34…` |

Engineering checks, all on the design split or on fixed structures only:

- **Arm runner.** `arms.run_arm` reproduces the design pilot's scores bit for bit (4 types × 2
  cells × every arm and configuration).
- **Kernel bootstrap.** It equals the §7 reference draw for draw (to 1e-12) and is about 100×
  faster: 1.6 ms per resample at `n = 600`, `R = 64`.
- **Mutation tests.** All mutants of the bootstrap, p-value, Holm, dominance and verdict code are
  caught (12 of 12).
- **Design-split dry run** (scratch, not committed; `R = 8`, `B = 50`; not a result):
  - the targets re-computed through the held-out path equal the committed design targets
    (`max |ΔD| = 0`, no stall mismatches);
  - the oracle ceilings equal the committed ones;
  - the pipeline runs end to end;
  - in-sample verdict on both panels: NO PRACTICAL ADVANTAGE, as the pilot predicted.
- **Sealed split.** `heldout_targets.py --split test` raises `PermissionError` before writing
  anything.
- **E002a matched-null smoke run** (`R = 200`; 4 `N` values; scratch, not a result). Matched-null
  false alarms fall inside the Monte Carlo band in 614 of 616 cells. The plug-in `C_hat` is
  biased upward at `C = 0`, and coverage at `C = 0` is far below nominal.
- **[post-hoc observation]** The Gram-entry flags concentrate in rare-gold cells
  (`N q < 5`), which the Amendment 2 eligibility rule (FP count only) does not exclude. The
  formal output reports an `N q >= 5` stratum as a labelled secondary. The registered rule is
  unchanged.

Run order after explicit approval, from a clean tree:

```
touch configs/e002/HELDOUT_APPROVED        # unsealing; committed with a reference to the approval
uv run python -W ignore experiments/e002/heldout_targets.py results/E002-pilot-targets/20260925T083746Z_81ffa61 --split test
uv run python -W ignore experiments/e002/heldout_oracle.py results/E002-targets-test/<run>   # commit before step 4
uv run python -W ignore experiments/e002/e002b_heldout.py results/E002-targets-test/<run>
uv run python -W ignore experiments/e002/e002b_analysis.py results/E002b-test/<run> results/E002-targets-test/<run>
uv run python -W ignore experiments/e002/e002a.py results/E002-pilot-targets/20260925T083746Z_81ffa61   # formal E002a, R = 2000
```

Expected wall time on 8 cores:

| step | time |
| --- | --- |
| targets | ~1 min |
| E002b simulation | ~15 min |
| analysis (`B = 2000`) | ~15 min |
| formal E002a | ~20–40 min |

---

### E002 — held-out results (CONFIRMATORY; test split; 2026-09-25)

**Unsealing.** The test split was unsealed on explicit approval in chat (commit `83b6737`,
`configs/e002/HELDOUT_APPROVED`). The frozen code is `d639b09`; no tracked code file changed
before or during the held-out runs. Every step followed the frozen order, each result was
committed before the next step, and every run directory is clean (no `-dirty`).

| step | run | notes |
| --- | --- | --- |
| targets | `results/E002-targets-test/20260925T210315Z_83b6737` | 34 s; 0 exclusions; stall fraction P-mod 0.475, P-rare 0.508 |
| oracle ceilings | `results/E002-oracle-test/20260925T210356Z_39a931c` | committed before any finite-sample test evaluation |
| E002b simulation | `results/E002b-test/20260925T210407Z_211d097` | 882 s; `R = 64`; local `scores.npz` sha256 `43f17dbe…338e3` |
| analysis | `results/E002b-analysis-test/20260925T211907Z_97413c9` | 536 s; `B = 2000` |

#### Oracle ceilings on the test split (registered predictions, §9)

| signal | P-mod C-index / AUROC | P-rare C-index / AUROC |
| --- | --- | --- |
| `C(0)` = `dFPR/dt(0)` = `dJ_V/dt(0)` | 0.930 / 0.976 | 0.923 / 0.979 |
| `-dJ_G/dt(0)`, G0 | 0.500 / 0.500 | 0.500 / 0.500 |
| FPR growth, `t_p = 10` | 0.989 / 0.997 | 0.970 / 1.000 |
| P4 oracle | 0.395 / 0.399 | 0.259 / 0.177 |

- The three first-order signals coincide — **PASS**.
- The `J_G(t_p -> 0)` ceiling is 0.5 — **PASS**.
- The G0 ceiling is 0.5 — **PASS**.

#### Primary endpoints (P-mod, primary)

C-index / AUROC at the primary cells:

| cell | G1 | G1-oracle | P1 | P2 | P3 | P4 | G0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (64, 256) | .790 / .872 | .742 / .822 | .772 / .852 | .816 / .903 | **.821 / .901** | .479 / .471 | .498 / .496 |
| (256, 1024) | .863 / .950 | .832 / .929 | .848 / .938 | .870 / .958 | **.900 / .968** | .455 / .445 | .497 / .497 |

`Delta = G1 - max_j P_j`, with the one-sided bootstrap p-value:

| cell | endpoint | `Delta` [95% CI] | p |
| --- | --- | --- | --- |
| (64, 256) | C-index | −0.031 [−0.037, −0.025] | 1.000 |
| (64, 256) | AUROC | −0.030 [−0.039, −0.024] | 1.000 |
| (256, 1024) | C-index | −0.037 [−0.043, −0.032] | 1.000 |
| (256, 1024) | AUROC | −0.018 [−0.025, −0.012] | 1.000 |

- **Holm: no test rejected.** Practical success is **not** established.
- Pairwise at the primary cells:
  - G1 > P1 (zero gold) by +0.012 to +0.020, one-sided `p ≈ 0.0005`;
  - G1 < P2 and G1 < P3;
  - G1 ≫ P4.
- G1-oracle − `max_j P_j`: −0.039 to −0.080. The exact Fisher is again **below** the estimated
  metric.

#### Frontier and verdict (computed by the frozen code)

- **P-mod: G1 is dominated at every one of the 21 cells, so rule (c) holds. Verdict: ABANDON.**
- Rules (a) and (b) do not hold: G1 beats P1 and P4 at the primary cells, and G1-oracle never has
  a positive `Delta`.
- **Fragility disclosure.** At (16, 64) the dominating competitor is P3 at the same cell, with
  C-index 0.6996 vs G1's 0.6995 (margin 7e-5). The registered rule counts a tie (`>=`) as
  domination, and the verdict stands as registered.
- **[post-hoc]** Bootstrap probability that G1 at (16, 64) is dominated: 0.508 (run
  `results/E002b-posthoc-test/20260925T215910Z_462360d`). The ABANDON-versus-NO PRACTICAL
  ADVANTAGE distinction is therefore a coin flip. Practical success is rejected regardless.
- **P-rare (secondary): NO PRACTICAL ADVANTAGE.**
  - All four `Delta`s are negative: −0.039 [−0.047, −0.032], −0.045 [−0.054, −0.036],
    −0.065 [−0.071, −0.059] and −0.039 [−0.049, −0.031], each with p = 1.000.
  - P1 (zero gold) beats G1 by 0.04–0.06, and P3 by 0.02–0.07.
  - G1 is non-dominated only at (16, 64): margin +0.025 over the best cheaper competitor,
    bootstrap CI [0.015, 0.035], P*(dominated) = 0.00 [post-hoc]. This is G1's only robust
    advantage: the smallest gold budget, rollout ratio 4, rare-gold regime.

#### Secondary analyses

- **Leave-one-type-out.** Every `Delta` stays negative whichever type is removed. P-mod C-index
  at (64, 256): [−0.034, −0.028]. P-rare: [−0.070, −0.027] over all cells × endpoints.
- **Sign-flipped P4** [post-hoc]: 0.52–0.62. It is never the strongest competitor, and no
  `Delta` changes.
- **Design → test replication.** At the primary cells, the in-sample design `Delta`s were
  −0.031 / −0.034 (P-mod C-index); the test values are −0.031 / −0.037.
- **Agent's prior (§12, not a hypothesis).** "Any G1 win ... confined to ratio-1 cells or none":
  - none in P-mod;
  - in P-rare, the only non-dominated cell is ratio 4, not ratio 1.

#### Conclusion (scope: Candidate 3, natural gradient, matched panels)

Estimated zero-step geometry has **no practical advantage** over short probes at matched
certification budgets. Under the registered rules the primary verdict is **ABANDON (rule c)**.
That verdict turns on a tie at one cell, and without it the result would be NO PRACTICAL
ADVANTAGE. The diagnostic claim is therefore **not supported**, and the mechanistic contribution
is preserved (§8). Geometry beats the zero-gold verifier-score probe in P-mod but loses to it in
P-rare. Its one robust advantage is the smallest-budget cell in P-rare.

---

### E002a — formal results (R = 2000; Amendment 2 instrumentation; fixed non-panel structures)

Run `results/E002a/20260925T212851Z_10aa907`: 1133 s, 1.9 GB peak RSS, approved together with the
held-out run.

- **Degenerate-event rates: PASS.** Exact binomial tests flag 0 of 1140 checks per panel
  (3.1 expected by chance).
- **Raw Gram entries unbiased** (U-statistic, oracle metric, `N(1-q)f >= 5`): **consistent with
  chance.**
  - P-mod: 1 of 579 entries flagged (1.6 expected).
  - P-rare: 3 of 408 (1.1 expected; binomial tail ≈ 0.10). All three are rare-gold cells
    (`N q < 5`).
  - **[secondary]** In the `N q >= 5` stratum: 1 of 474 (P-mod) and 0 of 90 (P-rare).
  - The pilot's FAIL of the original `C_hat^2` claim stands. The ratio `C_hat^2` is not unbiased.
- **F4: PASS.** `z` = 0.21, 0.70 (P-mod) and 0.43, 0.27 (P-rare).
- **Plug-in `C_hat` biased upward at `C = 0`: PASS** (dimension-matched nulls). Bias:
  - P-mod: +0.044 to +0.085 at `N = 16`, falling to +0.007 to +0.012 at `N = 1024`;
  - P-rare: +0.006 to +0.023, falling to +0.004 to +0.006.
- **Coverage near `C = 0` below nominal: PASS.** At the matched nulls coverage is 0.00–0.72 for
  `N <= 256` (P-mod is already 0.00 at `N = 256`) and 0.00 at `N = 1024`: a percentile interval of a nonnegative estimator excludes 0.
  At dose `rho = 0.05` it is 0.28 / 0.71 / 0.89 (P-mod, `N = 64 / 256 / 1024`). For `C > 0` it
  is 0.92–0.95 at P-mod and 0.70–0.95 at P-rare (`N >= 64`).
- **False alarm at the matched null equals 5%: PARTIAL.**
  - Inside the Monte Carlo band (±0.021) once the expected false-positive count per batch
    `N(1-q)f` reaches about 1.5 (P-mod, `N >= 16`: 3 of 693 outside) to 2.5 (P-rare, `N >= 128`:
    4 of 539 outside).
  - Below nominal at smaller `N` (conservative; false alarm 0.001–0.02): the null `C_hat^2` has
    atoms when a batch holds few false positives.
  - Outside the band overall: 12 of 770 (P-mod) and 104 of 770 (P-rare).
- **Minimum detectable `C`** (plug-in, estimated Fisher `lam = 1e-2`, 80% power at the
  matched-null 5%):
  - P-mod: 0.143 / 0.073 / 0.049 at `N = 64 / 256 / 1024`. At `N >= 1024` the value is bounded
    by the dose grid (smallest nonzero dose `C = 0.062`).
  - P-rare: not reached at `N = 64`; 0.079 / 0.040 / 0.025 at `N = 256 / 1024 / 4096`.

---

### E000d — E004 design-phase checks (retroactive, NOT pre-registered; not evidence)

Date: 2026-09-25. These are scratch checks for the E004 design memo
(`research/06_e004_design.md`, §2), run on a scratch multi-prompt toy (exact enumeration +
autodiff). The scripts are kept as run in `research/design_checks/e004/`, which is excluded from
lint. No E004 panel, target or predictor exists. None of this is an E004 result. It is disclosed
because the design relies on it, and E004 predictions that use these facts are not blind to them.

| check | result |
| --- | --- |
| DC1 | Observable identity `alpha = -FPR`, `A^2 = J_G'/(1-FPR)`, `C^2 = (1-J_G) FPR'` for feature-triggered FP with FNR = 0, prompt-uniform trigger, and features independent of gold-relevant parameters: ≤ 3e-15 under NG. It breaks (deviation 0.4–31) for attempt credit, deletion, FN coins, prompt-specific triggers and R. Shared-parameter coupling is a reparameterization and invisible to NG (4e-9). |
| DC2 | Benign amplification vs the most accessible exploit at matched `(J_G, FPR)` under NG: `rho_B - rho_Y = FPR` (min residual −2e-10 over 40 pairs); `rho_B = 1 + FPR(1-J_G)/J_G` exact. |
| DC3 | NG, FNR = 0, hack acceptance 0.9: no decline (0.15 → 0.90). FNR = 0.3: peak 0.20 at `t = 2`, then 0.001. |
| DC4 | Infinite-batch Adam-like flow. The identity holds for an uncoupled AND2 (block-diagonal metric). Coupling changes the dynamics (`alpha(0)` +0.025 vs −0.002; FPR(0.6) 0.18 vs 0.016). Clean `t95`: 4.9 (NG), 5.6 (Adam). |
| DC5 | Mechanism hard pair (B vs coupled AND2 exploit, infinite-batch Adam): matched within 0.005 over the first 5% of `T`; `alpha` +0.45 vs −0.37; `C_out` 0 vs 0.28–0.43; both succeed. |
| DC6 | Single-feature exploit, normalized shortfall at `T = 15`: NG 0.434; sign-GD 0.000; mean-field Adam 0.012 / 0.031 / 0.095 at batch 512 / 64 / 8. |
| DC7 | Preference inversion: shortfall 0.43 (NG) / 0.49 (mean-field Adam), peak then decline; with inversion on both prompts, 1.56 / 1.88. No inversion: success under both. |
| DC8 | Latent-decline outcome hard pair (D vs B, mean-field Adam, batch 64): matched within 0.002 over the first 5%; `alpha` −0.26 → −0.40 vs +0.03; D peaks at 30% of `T`, then DECLINE (normalized shortfall 0.49); B SUCCESS. |

---

### E004a — pre-registration (Stage 0 protocol; Stage 1 success criteria frozen)

Date: 2026-09-25. Written and committed **before any E004 code exists**.

- Design memo: `06_e004_design.md` v1.
- Collaborator decisions 1–5 approved on 2026-09-25, together with Correction A (hard-pair
  fairness), Correction B (dynamic hard pair) and the cross-construction validity gate.
- Design-phase checks E000d are disclosed; predictions that rely on them (F1–F8) are **not
  blind**.
- **Stage 0 touches the design split only.** No final predictor is fitted, no test or shift
  structure is generated, and nothing from E004b is started. The Stage 1 success criteria in §12
  are frozen here and may not change after Stage 0 is observed.

#### 1. Toy (U-toy; identical parameterization for every structure)

- **Prompts.** `x = 1..4` with weights `w`.
- **Response** `y = (s, xi, z)`:
  - `s ∈ {SOLVE, HACK, OTHER}` with softmax logits `(u_x, h, 0)`;
  - `xi ~ Bern(p_x)` if `s = SOLVE`;
  - `z ∈ {0,1}^3` independent, `logit z_j = phi_j + lam_j·mean(u)`.
- **Parameters** `theta = (u_1..u_4, h, phi_1..phi_3)`, `d = 8`.
- **Gold** `G = 1{s = SOLVE, xi = 1}`.
- **Verifier**, expected acceptance with fresh coins:
  - on deleted prompts, `v0`;
  - otherwise, correct answers are accepted w.p. `1 - fn_x`, and wrong answers w.p.
    `1 - (1-fp_x)(1 - trig_x 1_E(z))(1 - rho_x 1{HACK})(1 - beta_x 1{SOLVE, failed})`.

#### 2. Constructions (two per mechanism; 12 in total)

| mechanism | construction | channel |
| --- | --- | --- |
| R attenuation | R1 | symmetric coins `fp_x = fn_x = eps` on all prompts |
| | R2 | prompt-heterogeneous coins `fp_x = eps·r_x`, `fn_x = eps'·r'_x` (`r, r' ~ 4·Dirichlet(1)`, capped at 0.45) |
| X deletion | X1 | accept-all subset: `V = 1` on `S`, `|S| ∈ {1,2}` |
| | X2 | constant-score subset: `V = v0 ~ U(0, 0.3)` on `S`, `|S| ∈ {1,2}` |
| Y-A exploit discovery | YA1 | conjunctive trigger (AND2 w.p. 1/2, else AND3) on all prompts, low initial accessibility |
| | YA2 | prompt-dependent conjunctive trigger: AND2 on a random prompt subset `|T| ∈ {1,2,3}` |
| Y-B exhaustion | YB1 | accessible single-feature trigger on all prompts |
| | YB2 | accessible OR2 trigger on all prompts |
| B benign amplification | B1 | attempt credit `beta` on all prompts |
| | B2 | attempt credit `beta` on a random prompt subset `|S| ∈ {1,2,3}` |
| D displacement | D1 | difficulty inversion: hack acceptance `rho ~ U(0.3, 0.9)` on all prompts, with `p_x < rho` on ≥ 1 prompt |
| | D2 | strictness inversion: hack acceptance `rho ~ U(0.3, 0.9)`, and on a subset `S` of prompts with `p_x >= rho` correct answers are rejected with `fn_x = 1 - rho·U(0.5, 0.9)/p_x` |

#### 3. Common draws and calibration (every construction)

**Draws.**

- `w ~ Dirichlet(2·1)`.
- `p_x ~ Beta(2, 1.5)`, clipped to `[0.2, 0.98]`.
- Skill offsets `delta_x ~ N(0, 1)`, centered; `u_x0 = mu + delta_x`.
- `h0 ~ N(-2, 1)`; `phi0_j ~ N(-1.5, 1)`.
- **Coupling** `lam_j ~ N(0, 0.5^2)`.
- Targets from one common distribution:
  - `J_G(0) ~ U(0.05, 0.5)`;
  - `FPR(0) ~ LogUniform(0.02, 0.4)`;
  - `FNR(0) ~ U(0, 0.25)`.
- Mechanism share `omega ~ U(0.5, 1)`.

**Calibration.**

1. `mu` is solved (brentq) so that `J_G(0)` hits its target. For D, `h0` and `mu` are solved
   alternately until both converge.
2. The construction's channel strength is solved so that the channel alone gives `omega` × target
   FPR:
   - Y: a common logit shift of the event features;
   - B: `beta ≤ 1`;
   - D: `h0`.
   Exceptions:
   - YA1 uses `S_E0 = min(omega·target, s_A)` with `s_A ~ LogUniform(0.001, 0.05)`;
   - YA2 uses `S_E0 ~ LogUniform(0.002, 0.1)`;
   - R solves its coins for the targets directly;
   - X uses its own draws.
3. The channel-only FPR and FNR must not exceed their targets. Otherwise the construction's own
   draws are redrawn (≤ 50 times), then the targets are redrawn. Rejection counts are reported.
4. **Background noise (top-up coins).** `fp` and `fn` on non-deleted prompts are solved (brentq)
   so that the total FPR and FNR hit the targets.

**Canonical twin.** The same structure with `lam = 0` and the top-up coins removed; not
re-calibrated. It is used for theory checks and signatures only.

#### 4. Seeds and panel

- `SeedSequence(20260930).spawn(4)` = `[design panel, test panel (reserved), shift panel
  (reserved), runs]`.
- **Stage 0 generates the design panel only:** 40 structures per construction, 480 in total.
- Structures with clean gain `J_G^clean(T) - J_G(0) < 0.1` are excluded and counted.

#### 5. Optimizers

- **Primary: sampled GRPO-lite Adam.**
  - Each step: 8 prompts ~ `w`, 8 responses each.
  - Advantage `(V - group mean)/(group std + 1e-6)`; 0 when the std is 0.
  - Gradient: the mean of `A·grad log pi`.
  - Adam: lr 0.01, β = (0.9, 0.999), eps 1e-8, bias-corrected.
  - 4 verifier seeds and 4 clean seeds (`V = G`) per structure.
- **Secondary (theory anchor): exact natural-gradient flow** `theta' = F^-1 grad J_V`, with the
  clean flow as counterfactual.
- **Design approximation only: mean-field Adam (MF-Adam).**
  `theta_i' = lr·g~_i / sqrt(g~_i^2 + sigma~_i^2/64 + eps^2)` per step. `g~` is the exact expected
  gradient with population-normalized advantages `(V - b_x)/sigma_x`; `sigma~^2` is the exact
  per-sample second moment minus `g~^2`.
  - Used for hard-pair search and for a comparison with sampled Adam.
  - **Never** used as a target or claimed equivalent to sampled Adam.
- **Horizon `T`** (per optimizer). `T = 3 × median t95^clean` over the design split.
  - `t95^clean` is the first time the clean mean `J_G` curve (4-seed mean for Adam; the exact flow
    for NG) reaches `J_G(0) + 0.95(sum_x w_x p_x - J_G(0))`.
  - Provisional clean runs last 4000 Adam steps or 60 NG time units; censored `t95` = the run
    length.
  - Adam `T` is rounded up to a multiple of 100 steps.

#### 6. Checkpoints, observables and features

- **Horizons.** `H = {0, 0.2, 0.5, 1, 2, 5, 10}%` of `T`; primary `h* = 2%`. Checkpoints:
  - `H ∪ {h/2}`;
  - every 1% of `T` for outcome curves.
- **Exact observables** of the realized policy at each checkpoint: `J_G, J_V, FPR, FNR`.
- **Geometry in the optimizer metric** (`A, alpha, C, C_in, C_out`; `C_in/C_out` = the residual
  inside/outside the span of the 4 per-prompt gold gradients):
  - Adam: `M = diag(1/(sqrt(v_hat) + eps))` from the run's own state, with `g_V -> g~` (the
    GRPO-effective direction). At `t = 0` (L1), `v_hat := g~^2 + sigma~^2/64` at `theta0`.
  - NG: `M = F^-1`, `g_V = grad J_V`.
- **Levels.**

  | level | contents |
  | --- | --- |
  | L0 | `J_G(0), J_V(0), FPR(0), FNR(0)`, FP mass |
  | L1 | L0 + `(A, alpha, C)` at 0 |
  | L2 | L0 + summaries of `J_G, J_V, FPR, FNR` |
  | L2-G | L0 + summaries of `J_G` |
  | L3 | L2 + summaries of `A, alpha, C` |
  | L2+ | L2 + per-prompt `ΔJ_G,x` |
  | L3+ | L3 + summaries of `C_in, C_out` |

  Summaries at horizon `h`: the value at `h`, the change `0 → h`, and the slope over `[h/2, h]`
  per 1% of `T`.
- **Audit SE** (Stage 1 measurement; used here for the hard-pair criteria). With `n = 256`:
  - `SE(J_G) = sqrt(J_G(1-J_G)/n)`; `SE(J_V)` likewise;
  - `SE(FPR) = sqrt(FPR(1-FPR)/(n(1-J_G)))`; `SE(FNR) = sqrt(FNR(1-FNR)/(n J_G))`;
  - the SE of a change or slope combines the two endpoint SEs in quadrature (slope divided by
    its width);
  - SEs are evaluated at the mean of the two structures being compared.

#### 7. Outcomes (per run)

- **Clean reference:** the 4-seed-mean clean exact `J_G` curve (Adam), or the exact clean flow
  (NG).
- `Dn = (J_G^clean(T) - J_G(T)) / (J_G^clean(T) - J_G(0))`.
- **Descriptive labels, in precedence order:**
  1. DECLINE: `max_t J_G - J_G(T) >= 0.05` and `Dn > 0.1`;
  2. SUCCESS: `Dn <= 0.1`;
  3. STALL: `2(J_G(T) - J_G(T/2)) < J_G^clean(T) - J_G(T)`;
  4. SLOW: otherwise.
- **Primary inferential targets:** `Dn`, and binary failure = STALL ∪ DECLINE.
- **Onset** `t_on` = the first outcome checkpoint where
  `(J_G^clean - J_G)/(J_G^clean - J_G(0)) > 0.1`.
- **Mechanism label** = the construction's mechanism. Route A vs B = YA vs YB.

#### 8. Stage 0 analyses (design split only)

1. **Outcome map** by construction under sampled Adam, NG and MF-Adam; MF-Adam vs sampled-Adam
   agreement.
2. **Signatures** of canonical vs primary structures: `alpha_0`, `C_0`, `C_out/C`, and the trends
   of `C/A` and `alpha`.
3. **Optimizer-dependence figure**, including a batch sweep (sampled Adam with 4×4, 8×8 and 16×16
   rollouts per step) on the first 10 design structures of every construction.
4. **Leakage checks.**
   - 5-fold CV grouped by structure; standardized features.
   - L0 → mechanism (6 classes) and L0 → construction (12 classes): multinomial logistic
     (`LogisticRegressionCV`) and GBM (depth 2, 100 trees).
   - The type oracle (leave-one-out mechanism-mean `Dn`) is reported.
5. **Oracle information ceilings** (not final predictors; exact observables along the realized
   runs).
   - For each horizon and level, grouped 5-fold CV:
     - `RidgeCV` → C-index of `Dn`;
     - `LogisticRegressionCV` → AUROC of binary failure, plus not-yet-visible AUROC (evaluated on
       runs with `t_on > h`);
     - multinomial logistic → mechanism macro-F1, and YA vs YB AUROC.
   - Single-variable raw ceilings with fixed orientation: L0 FPR(0); L1 `C_0/A_0`; L2 `ΔFPR` and
     `-ΔJ_G`; L3 `ΔC` and `-Δalpha`.
   - Primary for Adam; repeated for NG.
6. **Cross-construction mechanism test.** Fit the mechanism classifier (oracle L2 and L3,
   `h*`) on construction set 1 = {R1, X1, YA1, YB1, B1, D1}, evaluate on set 2, and the reverse;
   report the mean macro-F1 and its ratio to the within-panel CV macro-F1.
7. **Hard pairs.**
   - **Search:** least squares over a member's natural parameters, 20 anchors per pair type,
     under MF-Adam. Residuals = all L2 feature differences over `[0, h*]` in audit-SE units.
   - **Verification:** sampled Adam 32-seed means.
   - **Correction A (all pair types claimed L2-blind):** every L2 feature difference (value,
     change, slope of `J_G, J_V, FPR, FNR` at every registered horizon ≤ `h*`) must be
     ≤ 0.5 audit-SE. The criterion is not weakened if a pair type becomes infeasible.
   - **Types:**
     - HP-A: B vs coupled YA (mechanism pair);
     - HP-D: D vs B (latent decline);
     - HP-B: YA vs YB with coupling;
     - HP-C: B vs Y proxy growth (stress test only; no L2-blind claim).
   - **Correction B (dynamic pair, any two different mechanisms).** At `t = 0`:
     - L0 matched ≤ 0.5 audit-SE;
     - L1 matched: `|ΔA0| <= 0.1·mean(A0)`, `|Δalpha0| <= 0.1`, `|ΔC0| <= 0.1·max(mean C0, 0.01)`.

     Over `[0, h*]`, L2 is matched as in Correction A. Before the failing member's `t_on`, L3
     diverges: at some checkpoint, `|Δalpha| >= 0.3`, `|ΔA|/mean >= 0.5` or `|ΔC|/mean >= 0.5`.
     The outcome (binary failure) or the mechanism differs. The search does not force existence;
     the best mismatch is reported.
   - **Freeze:** the pair parameters are committed before any Stage 1 predictor is fitted.
8. **Theory nulls** on canonical twins:
   - F1 identity on canonical YA1/YB1 under NG (max deviation ≤ 1e-8);
   - NG invariance to coupling (coupled vs uncoupled trajectories ≤ 1e-6).

#### 9. Stage 0 predictions (registered)

- **S0-P1 (F1):** canonical YA1/YB1 NG identity holds (≤ 1e-8), and oracle L3 − L2 = 0 there.
- **S0-P2 (F2):** coupling is invisible to NG (≤ 1e-6).
- **S0-P3 (F5):** the failure fraction of YA ∪ YB is lower under sampled Adam than under NG.
- **S0-P4 (F6):** D fails (binary) in ≥ 50% of runs under both sampled Adam and NG.
- **S0-P5:** B fails in ≤ 10% of runs under both.
- **Agent's prior (not hypotheses):**
  - oracle L2 ceilings approach 1 by `h = 5%` in most constructions;
  - L3 − L2 is small except in D;
  - HP-D may become infeasible under Correction A once FNR is matched.

#### 10. Gates to Stage 1 (all required)

- **G1:** L0 → mechanism macro-F1 (logistic, CV) ≤ 0.35.
- **G2:** under sampled Adam, every mechanism except B and R has ≥ 10% failing and ≥ 10%
  non-failing runs.
- **G3:** S0-P1 and S0-P2 hold.
- **G4:** cross-construction mechanism macro-F1 (the best of oracle L2 and L3 at `h*`) is
  ≥ 0.33, and ≥ 0.6 × the within-panel CV macro-F1.

If a gate fails, the recommendation is REVISE PANEL, with the exact reasons. A revision is
recorded as a dated amendment and re-run on a new design draw.

#### 11. Deviations from memo v1 (recorded now)

- MIX is deferred; it is not in the Stage 0 panel.
- The FNR target range is `U(0, 0.25)`; background coins are the calibrated top-ups.
- RQ4 (counterfactual optimizer) is not part of Stage 0.

#### 12. Stage 1 success criteria (FROZEN; memo §11 with the approved decisions)

- **Setting:** sampled Adam, test split A, `h* = 2%`, audit `n = 256`.
- **Inference:** hierarchical bootstrap (structures, then seeds, 2000 resamples); one-sided
  `p = (1 + #{Δ* <= 0})/(B+1)`. A "beats" claim needs the lower 95% bound > 0 **and** a point
  estimate ≥ the margin (`δ_out = 0.02`, `δ_mech = 0.03`). Ties within ±0.01. No point-estimate
  dominance rules.
- **RQ1 success:** `L2 − L0 >= δ_out` with lower bound > 0 for both the C-index of `Dn` and the
  AUROC of binary failure (Holm over 2).
- **Early-warning success:** among runs with `t_on > h*`, the L2 AUROC for eventual failure has
  lower bound ≥ 0.65, **and** the median lead time at a design-calibrated 10% false-alarm
  threshold is ≥ 5% of `T`.
- **Generalization:**
  - the shift split: `L2 − L0` lower bound > −0.01;
  - leave-one-mechanism-out: `L2 − L0` lower bound > 0 in ≥ 4 of 6 held-out mechanisms.
- **Mechanistic value:** at `h*`, all of
  - L3 − L2 macro-F1 ≥ `δ_mech` with lower bound > 0;
  - Route A vs B AUROC: L3 − L2 ≥ `δ_out` with lower bound > 0;
  - on the frozen hard pairs, L3 correct on ≥ 80% where L2 is at chance (one-sided binomial).

  L3 is not required to beat L2 on outcome.
- **Abandon early warning** if, at every `h <= 5%`, RQ1 fails or the not-yet-visible AUROC lower
  bound is ≤ 0.55.
- **Mechanistic-only** if early warning fails and mechanistic value succeeds.
- **Proceed to E004b** if RQ1, early warning and generalization all succeed.

---

### E004a — Stage 0 record (DESIGN SPLIT ONLY; 2026-09-25/26)

No test or shift structure was generated, no Stage 1 predictor was fitted, and no E004b work was
started.

**Runs.**

| step | commit | output |
| --- | --- | --- |
| panel | `ec9a139` | `configs/e004/design_panel.json`, sha256 `e49753e9…a30d` (480 structures, 33 s) |
| runs | `cf78799` | `results/E004a-stage0/20260925T235505Z_cf78799` (431 s; raw `runs.npz` local) |
| analysis | `139fa62` | `results/E004a-stage0-analysis/20260926T000358Z_139fa62` |
| hard pairs | `fc24ae1` | `results/E004a-stage0-hardpairs/20260926T000852Z_fc24ae1`; frozen pairs `hardpairs_frozen.json`, sha256 `f4765fe4…4f9b` |
| [post-hoc] | `5421de3` | `results/E004a-stage0-posthoc/20260926T002907Z_5421de3` |

All run directories are clean.

**Horizon and exclusions.**

- `T` (registered rule): sampled Adam 2700 steps (median clean `t95` 880); NG 23.4 (median `t95`
  7.80).
- Excluded for clean gain < 0.1: 31 structures (R1 3, R2 4, X1 2, X2 2, YA1 3, YA2 3, YB1 4,
  YB2 2, B1 3, B2 4, D1 1, D2 0).
- Calibration rejections:
  - R1: 1050 channel rejections, 31 target redraws (`FNR >= FPR` by construction);
  - X1: 3412 / 79;
  - X2: 1645 / 37;
  - D2: 441 / 13;
  - the others ≤ 62 / ≤ 7.

**Process disclosures.**

- **Tests without an observed RED.** The panel and dynamics tests were written before their code,
  but the failing run was not observed. Mutation testing caught 6 of 7 mutants; the missed one is
  equivalent (coins on deleted prompts have no effect).
- **Exclusion added late.** The first analysis launch lacked the registered clean-gain exclusion.
  It was stopped before writing any output, its partial directory was deleted, and it was re-run
  after the fix (`139fa62`).
- **A commit with a failing test.** Commit `3a31dde` went in with one failing test, because a
  piped exit code hid it. The failure was E002's approval-file test, which had been failing since
  the approved unsealing `83b6737`. It was fixed in `690a430`.

#### Gates (registered §10)

| gate | result | detail |
| --- | --- | --- |
| **G1** family recognition | **PASS** | L0 → mechanism macro-F1 0.185 (logistic; GBM 0.213; chance 0.167). L0 → construction 0.075 / 0.077 (chance 0.083). Type oracle: C-index 0.74, AUROC 0.871. |
| **G2** outcome diversity | **FAIL** | Sampled-Adam failure: X 0.47, YA 0.67, **YB 0.946 (> 0.9)**, D 0.82; R 0, B 0. |
| **G3** theory nulls | **FAIL by the letter** | F1 max relative deviation 2.6e-7 > 1e-8. F2 3.5e-8 (pass). |
| **G4** cross-construction | **PASS** (marginal) | Best (L3) cross-construction macro-F1 0.474, within-panel 0.740: 0.474 ≥ 0.33 and ≥ 0.444. L2: 0.344 / 0.542; L3+: 0.587 / 0.799. |

**[post-hoc] G3 diagnosis.** The deviation arises at `t = T_ng` states with FPR → 1
(`alpha = -1.000`, `cond(F) ≈ 1e8`), through cancellation in `J_G'/(1 - FPR)`. Damping does not
matter. On non-saturated states the deviation is ≤ 1.2e-8. The failure is numerical; the gate
remains FAILED as registered.

**Registered Stage 0 predictions.**

- **S0-P1:** the identity part fails by tolerance, as for G3. "Oracle L3 − L2 = 0" was not
  testable as registered, because the L2 summaries are finite-difference slopes, not derivatives
  (a registration imprecision).
- **S0-P2: PASS.**
- **S0-P3: FALSIFIED.** Failure of YA ∪ YB is **higher** under sampled Adam (0.81) than under NG
  (0.45). Design-check F5 (memo §2) does not generalize to the panel.
- **S0-P4: holds** (D fails 0.82 under Adam, 0.95 under NG).
- **S0-P5: holds** (B fails 0 / 0).

#### Outcome map (failure / DECLINE / STALL fraction; median `Dn`)

| construction | sampled Adam | NG | MF-Adam | Adam, canonical | NG, canonical |
| --- | --- | --- | --- | --- | --- |
| R1, R2, B1, B2 | 0 | 0 | 0 | 0 | 0 |
| X1 | .43/.00/.43 (0.09) | .38/.00/.38 | .45/.00/.45 | .45/.00/.45 | .42/.00/.42 |
| X2 | .51/.00/.51 (0.11) | .53/.00/.53 | .50/.00/.50 | .53/.00/.53 | .53/.00/.53 |
| YA1 | .72/.66/.06 (1.12) | .07/.03/.05 | .72/.70/.03 | .56/.00/.56 | .03/.00/.03 |
| YA2 | .56/.47/.09 (0.30) | .03/.03/.00 | .55/.45/.10 | .14/.00/.14 | .00 |
| YB1 | .93/.89/.03 (1.36) | .90/.53/.38 | .90/.90/.00 | .97/.00/.97 | .78/.00/.78 |
| YB2 | .93/.87/.06 (1.47) | .80/.50/.30 | .88/.88/.00 | 1.0/.00/1.0 | .50/.00/.50 |
| D1 | .82/.66/.16 (1.24) | .95/.55/.40 | .82/.62/.20 | .71/.42/.29 | .95/.38/.57 |
| D2 | .82/.66/.16 (1.43) | .95/.68/.28 | .90/.65/.25 | .85/.65/.20 | .95/.68/.28 |

- The table is before exclusion; the gates use the post-exclusion values.
- **MF-Adam vs sampled Adam, per structure:** binary agreement 0.962, category agreement 0.955,
  Kendall `Dn` 0.872.
- **Batch sweep** (sampled Adam, rollouts per step 16 / 64 / 256): YA failure **rises** 0.49 →
  0.67 → 0.79. The other mechanisms are flat. [post-hoc interpretation: discovering the exploit is
  itself learning that benefits from signal.]
- **Background FN turns feature exploits into preference inversions.** With the top-up FN coins,
  correct answers are rejected sometimes while triggered wrong answers are always accepted.
  - Primary Y structures DECLINE in 47–89% of runs.
  - Canonical twins (no background coins) never decline; they STALL.

#### Signatures, canonical vs primary

Values at `t = 0` and changes to 10% of `T`; medians per construction in the analysis JSON.

- **`C_out/C` at `t = 0`** is ≈ 0 for R, X, B and 0.94–1.00 for YA, YB, D, under both optimizers
  and both twins. It is effectively a mechanism indicator (generator encoding).
- **Adam-metric `alpha_0` is positive for almost every construction** (R1 +0.86, B1 +1.45,
  YA1 +0.74; D2 −0.17). The GRPO std-normalization gain enters `alpha`, because `g~` is decomposed
  against the raw `g_G`. So `alpha > 0` does not isolate benign amplification under GRPO-lite (a
  measurement-definition issue).
- NG `alpha_0` is ≤ 0 except for canonical B (+0.07 to +0.19).
- **Coupling and background noise shift the signatures:**
  - Adam `alpha_0` for X1: +0.23 primary vs −1.00 canonical;
  - `d(C/A)` over 10% of `T` for YA1: +0.79 vs +2.04.

#### Oracle information ceilings

Exact observables, grouped 5-fold CV, design split; not final predictors. Sampled Adam at
`h* = 2%`:

| level | C-index `Dn` | AUROC failure | not-yet-visible AUROC (322 failing) | mechanism macro-F1 | YA vs YB AUROC |
| --- | --- | --- | --- | --- | --- |
| L0 | 0.55 | 0.47 | 0.39 | 0.18 | 0.41 |
| L1 | 0.73 | 0.89 | 0.89 | 0.59 | 0.90 |
| L2-G | 0.59 | 0.58 | 0.45 | 0.24 | 0.40 |
| L2 | 0.64 | 0.75 | 0.71 | 0.54 | 0.91 |
| L3 | 0.75 | 0.92 | 0.91 | 0.74 | 0.91 |
| L3+ | 0.75 | 0.93 | 0.93 | 0.80 | 0.89 |

- **At 10% of `T`:** L2 0.70 / 0.83 / 0.77 / 0.55; L3 0.78 / 0.94 / 0.92 / 0.76.
- **NG (secondary):** L1 0.83 / 0.96; L2 0.68 / 0.82; L3 0.81 / 0.97. Not-yet-visible failures
  under NG are too few (n = 6 at `h*`).
- **Single variables** (sampled Adam, `h*`), C-index / AUROC:

  | variable | C-index | AUROC |
  | --- | --- | --- |
  | `C0/A0` | 0.68 | 0.85 |
  | ΔFPR | 0.64 | 0.76 |
  | ΔC | 0.68 | 0.82 |
  | −ΔJ_G | 0.57 | 0.61 |
  | FPR0 | 0.52 | 0.53 |

**[post-hoc, not pre-registered] Beyond mechanism recognition.** Sampled Adam, `h*`, AUROC
failure / C-index `Dn`:

| level | alone | + one-hot mechanism type |
| --- | --- | --- |
| L0 | 0.47 / 0.55 | 0.89 / 0.79 |
| L1 | — | 0.93 / 0.79 |
| L2 | — | 0.92 / 0.78 |
| L3 | — | 0.94 / 0.78 |

- **Within mechanism**, from full-panel CV predictions:
  - X: L3 0.81 vs L2 0.53 (with type: 0.78 vs 0.45);
  - YA: 0.84 vs 0.65 (0.78 vs 0.67);
  - D: 0.72 vs 0.67 (0.65 vs 0.66);
  - YB has too few successes to evaluate.
- Most of the oracle geometry advantage is mechanism recognition. Within X and YA, geometry still
  adds 0.1–0.3 AUROC, and most of it is already present at `t = 0` (L1: X 0.80, YA 0.81).

#### Hard pairs (frozen)

MF-Adam search (86 restarts, 1030 s); verification with sampled Adam, 32 + 8 clean seeds per
member.

**Correction A:**

- **HP-A (B2 vs YA1): accepted in both tiers.** Sampled max ratio 0.049. B SUCCESS vs YA DECLINE
  (sampled failure 0.00 vs 1.00), so this is an outcome pair as well. *Caveat:* the B anchor's
  credit channel is weak (channel FPR 0.016 of 0.274).
- **HP-D (D1 vs B2): accepted in both tiers, with FNR matched** (0.217 each). Sampled max ratio
  0.232. D DECLINE (`Dn` 1.98) vs B SUCCESS (failure 1.00 vs 0.00). **The latent-decline outcome
  pair is feasible under the corrected criterion.**
- **HP-B (YA1 vs YB1): accepted in both tiers.** Both DECLINE, so this is a mechanism pair.

**Correction B (dynamic):**

- **DYN-YA/R (YA1 vs R2): accepted in both tiers.**
  - L0 max ratio 0.027; L1 within tolerance; L2 matched over `[0, 2%]`.
  - L3 diverges at 5% of `T`: `C` 0.060 vs 0.034, `alpha` 0.49 vs 0.60.
  - Visible failure is at 35% of `T` (YA DECLINE, R SUCCESS).
- **DYN-YB/B (YB2 vs B1): accepted in both tiers.** Divergence at 1–4% of `T`, but FPR also
  separates by 5%, so the geometric lead is small.
- **DYN-YA/B:** accepted in the sampled tier only (MF-Adam predicts both succeed).
- **DYN-D/B and DYN-D/R: infeasible.** L1 cannot be matched: displacement differs already at
  `t = 0`, so no dynamic-only pair exists.
- **DYN-X/YB:** matched, but there is no L3 divergence before onset.
- *Caveat:* the dynamic divergences are modest (|ΔC| ≈ 0.03). Whether they are detectable at audit
  size is a Stage 1 question.

#### Decision rule outcome (registered §10)

G2 and G3 failed, so the recommendation is **REVISE PANEL**. The exact reasons are in the Stage 0
report; no amendment has been made yet.

---

**E004a Amendment 3 — Stage 0b** (2026-09-26; written and committed **before any Stage 0b code or
output**)

**Status.**

- This is the FINAL synthetic-panel revision, approved by the collaborator after Stage 0.
- Every change is **data-informed** by the recorded Stage 0 failures and is disclosed as such.
  The only permitted reasons:
  1. YB outcome imbalance (failure 0.946);
  2. the preference-inversion confound hidden in background FN;
  3. the G3 numerical pathology near FPR saturation;
  4. GRPO `alpha` semantic contamination;
  5. generator / construction leakage.
- The Stage 0 results and failed gates remain recorded as they are.
- The Stage 1 success criteria (E004a pre-registration §12) are **unchanged**.
- No test or shift panel is generated; no Stage 1 predictor is fitted; E004b is not started.
- **No further panel revision is allowed after Stage 0b.** If material gates fail, the synthetic
  conclusions are frozen and work moves to E004b design.

**1. Outcomes.**

- Unchanged: SUCCESS / SLOW / STALL / DECLINE (descriptive).
- Primary: `Dn`, and FAILURE = STALL ∪ DECLINE.
- Mechanism labels are never used as outcome proxies.

**2. Optimizers.**

- Unchanged: sampled GRPO-lite Adam (primary); exact NG flow (theory anchor); MF-Adam (design
  approximation only).
- The optimizer is a crossed factor. NG → Adam transfer is never inferred without being
  observed.

**3. Two axes.**

- **Axis A** = error / signal structure (the mechanism label describes the causal mechanism).
- **Axis B** = the preference relation, computed from the actual verifier (after all coins):
  - For each non-deleted prompt `x`:
    - `v_solve(x) = max_z [ p_x (1 - fn_x) + (1 - p_x) EV(x, SOLVE, failed, z) ]`;
    - `v_wrong(x) = max_{s ∈ {HACK, OTHER}, z} EV(x, s, z)`;
    - `gap_x = v_wrong(x) - v_solve(x)`.
  - **INVERTED** iff `max gap_x > 0.02` over non-deleted prompts with `w_x >= 0.05`; else
    **ALIGNED**. Deleted prompts carry no preference.
  - The intended level (from the construction) and the actual flag are both recorded. When
    background noise creates an inversion, the structure is **kept and flagged INVERTED**.
- **Crossing:**
  - YA and YB are crossed with {ALIGNED, INVERTED}.
  - R, X and B are aligned-only (negative controls / deletion).
  - D is inverted by definition.
- **Intended ALIGNED Y:** fractional trigger acceptance `t ~ U(0.5, 0.9)`, no channel FN.
- **Intended INVERTED Y:** trigger acceptance 1, plus explicit correct-answer strictness
  `fn_x ~ U(0.1, 0.4)` on the triggered prompts. The exploit is accepted more reliably than a
  correct answer.

**4. YB accessibility** (data-informed; design-only calibration, run and recorded before any
Stage 0b oracle or predictor analysis).

- Original: `S_E0 = omega × FPR_target ∈ [≈0.01, 0.4]`.
- New: `S_E0 = min(omega × FPR_target, c)`.
- **Rule:**
  - candidates `c ∈ {0.2, 0.1, 0.05, 0.02, 0.01}`;
  - choose the **largest** `c` whose YB sampled-Adam failure rate lies in `[0.30, 0.70]`; if none
    does, choose the `c` closest to 0.5;
  - calibration set: 24 structures per YB construction × Axis-B cell (144 per `c`), 2 seeds each,
    `T = 2700` (the Stage 0 value), calibration seed stream (§8);
  - the calibration structures are never part of the panel.
- Recorded before the analyses: the original range, the new range, the rule, and the resulting
  design failure rate.

**5. Coupling and background noise.**

- As in Stage 0 (`lam ~ N(0, 0.5^2)`; top-up coins to the common targets).
- Canonical twins (uncoupled, no top-up coins) are kept for theory and signature checks.
- Inversion caused by noise is flagged, never hidden (§3).

**6. Two alphas.** Metric `M` = the optimizer's preconditioner: `diag(1/(sqrt(v_hat)+eps))` for
Adam; `F^-1` for NG.

- **Reward level** `(A_r, alpha_r, C_r)`: the raw `grad J_V` against the raw `grad J_G` in `M`
  (before GRPO normalization).
- **Update level** `(A_u, alpha_u, C_u, C_in, C_out)`: the GRPO-effective verifier direction
  `g~_V` against the gold direction `g~_G` computed with the **same** per-prompt centering and std
  normalization (population statistics of `G`), in `M`. `C_in/C_out` use the span of the per-prompt
  `g~_G,x`.
- There is no clipping in GRPO-lite. For NG (no normalization) the two levels coincide.
- **L1/L3 use the update level.** The reward level is reported for semantics.
- `alpha_u > 0` does **not** imply benign reward amplification.
- **Registered test:** for symmetric coins, `alpha_r = -2 eps` exactly in any metric, while
  `alpha_u` may take either sign.

**7. G3 numerical domain** (data-informed).

- The F1 identity is scored only where the first-order rounding bound
  `u·κ(F)/(1 - FPR) <= 2.2e-9` holds (`u = 2.2e-16`), i.e. `κ(F)/(1 - FPR) <= 1e7`, with the
  tolerance 1e-8.
- The bound is chosen from rounding analysis only: it stays a factor of about 4.5 below the
  tolerance.
- Reported: the all-state deviation, the valid-domain deviation, and the fraction of states
  excluded.
- The Stage 0 G3 failure stays recorded.

**8. Constructions (≥ 3 per mechanism) and panel.**

| mechanism | constructions |
| --- | --- |
| R (aligned) | R1 symmetric uniform coins; R2 asymmetric heterogeneous coins; R3 "lazy verifier", symmetric per-prompt coins `eps_x ~ U(0.02, 0.3)` |
| X (aligned) | X1 accept-all subset (`v0 = 1`); X2 reject-all subset (`v0 = 0`); X3 constant intermediate score (`v0 ~ U(0.3, 0.7)`); subset size 1–2 |
| YA (× A/I) | YA1 AND2/AND3 global; YA2 AND2 on a prompt subset (1–3); YA3 THR23 (≥ 2 of 3) global; all with `S_E0 ~ LogUniform(0.001, 0.05)` |
| YB (× A/I) | YB1 single-feature global trigger; YB2 single-feature trigger on the 1–2 hardest prompts (lowest initial skill); YB3 OR2 global; `S_E0` per §4 |
| B (aligned) | B1 attempt credit, all prompts; B2 process credit (failed attempts credited only when the process feature `z3 = 1`); B3 attempt credit on a prompt subset (1–3) |
| D (inverted) | D1 shared hack action, `rho ~ U(0.3, 0.9)` with `p_x < rho` somewhere; D2 prompt-specific shortcut (`rho_x` on 1–2 prompts only, `rho_x > p_x(1 - fn_x)`); D3 feature-triggered wrong strategy (HACK accepted w.p. `rho` only when `z1 = 1`, with `rho > p_x` somewhere) |

- 32 structures per construction × intended-Axis-B cell: 24 cells, **768 structures**.
- Common draws and calibration as in Stage 0: targets `J_G(0) ~ U(0.05, 0.5)`,
  `FPR(0) ~ LogU(0.02, 0.4)`, `FNR(0) ~ U(0, 0.25)`; `omega`; top-ups; redraws counted.
- Fresh seeds: `SeedSequence(20261001).spawn(5)` = `[design panel, test (reserved), shift
  (reserved), runs, YB calibration]`.
- The runs, `T` rule, horizons, features (L1/L3 at the update level) and outcomes follow the
  E004a pre-registration, as in Stage 0.

**9. Anti-generator-leakage tests** (oracle, design split).

- **A.** Construction recognition (18 classes) from L0 and from L1: macro-F1 vs chance 1/18.
- **B.** Mechanism recognition with the construction hidden: leave-one-construction-out
  (train on all other constructions, test on the held-out one), for L0/L1/L2/L3/L3+.
- **C.** Within-mechanism outcome and state prediction, controlling for the mechanism:
  - per mechanism, grouped-CV models fitted within that mechanism only;
  - targets: failure (AUROC), `Dn` (C-index), and the actual Axis-B flag (AUROC; YA/YB);
  - L3 vs L2 compared by a paired bootstrap over structures (1000 resamples).
- **D.** Cross-construction mechanism generalization: fold `k` trains on the other two
  constructions of every mechanism and tests on construction `k` (3 folds); mean macro-F1.
- **E.** Mechanism-type oracle (leave-one-structure-out mechanism-mean `Dn`), and the
  mechanism × Axis-B oracle: C-index / AUROC.

**10. Hard pairs.**

- **Correction A** as registered (every L2 feature ≤ 0.5 audit-SE) for HP-A (B vs YA), HP-D
  (D vs B) and HP-B (YA vs YB).
- **New Axis-B pair:** HP-I (YA-INVERTED vs YA-ALIGNED).
- **Correction B** (dynamic) for YA/R, YB/B, D/B, YA-I/YA-A, X/YB and YA/B, with the Stage 0
  tolerances.
- Same search and verification protocol; frozen before any Stage 1 work.

**11. Stage 0b gates** (frozen now).

- **G1** (leakage): L0 → construction macro-F1 ≤ 1/18 + 0.10 = 0.156. L1 construction
  recognition is reported but not gating.
- **G2** (outcome diversity, sampled Adam): the failure rate of X, YA, YB and D each lies in
  `[0.20, 0.80]`. R and B are exempt.
- **G3** (theory nulls):
  - inside the §7 domain, F1 identity deviation ≤ 1e-8 on the canonical twins of YA1, YA3, YB1
    and YB3 (ALIGNED, global triggers) under NG;
  - F2 coupling invariance ≤ 1e-6;
  - canonical R1 `alpha_r = -2 eps` within 1e-10.
- **G4** (cross-construction, test D, L3 at `h*`): macro-F1 ≥ 1/6 + 0.15 = 0.317, **and**
  L3 − L0 ≥ 0.05 with bootstrap (structures) lower 95% bound > 0.
- **G5** (within-mechanism information, test C, at `h*`): in ≥ 2 mechanisms, L3 − L2 ≥ 0.02 with
  bootstrap lower bound > 0, on failure AUROC, `Dn` C-index, or (YA/YB) Axis-B AUROC.
- **G6** (optimizer validity):
  - at least one mechanism with `|failure_Adam - failure_NG| >= 0.3`; **and**
  - at least one failure mechanism with failure ≥ 0.5 under both optimizers and
    `|difference| <= 0.2`.

**Decision:** all of G1–G6 pass → recommend PROCEED TO STAGE 1; any fails → STOP SYNTHETIC PANEL
ENGINEERING AND DESIGN E004b.

**12. Stage 0b registered predictions.**

- **S0b-P1:** within YA and YB, the DECLINE fraction among INVERTED is ≥ 3× that among ALIGNED
  (sampled Adam).
- **S0b-P2:** under sampled Adam, INVERTED fails more often than ALIGNED, for both YA and YB.
- **S0b-P3:** `alpha_r < 0` for every R structure at `t = 0` (Adam metric and NG metric).
- **Agent's prior (not hypotheses):**
  - G2 is at risk for D (> 0.8);
  - G5 is likely to pass through X and YA (Stage 0 post-hoc);
  - geometry may still mostly encode Axis A.

**E004a Stage 0b — YB calibration record** (Amendment 3 §4; committed before the panel and before
any Stage 0b oracle or predictor analysis)

Run `results/E004a-stage0b-calibration/20260926T010125Z_f768e38`: 128 kept calibration structures
per cap (16 excluded for low clean gain), 2 + 2 seeds, sampled Adam, `T = 2700`.

- **Original range:** `S_E0 = omega × FPR_target ∈ [≈0.01, 0.4]`.
- **Rule:** the largest `c ∈ {0.2, 0.1, 0.05, 0.02, 0.01}` with YB failure in `[0.30, 0.70]`;
  otherwise the one closest to 0.5.
- **Results** (YB failure rate):

  | cap | overall | ALIGNED cells (YB1 / YB2 / YB3) | INVERTED cells (YB1 / YB2 / YB3) |
  | --- | --- | --- | --- |
  | 0.2 | 0.641 | 0.35 / 0.30 / 0.27 | 1.00 / 0.95 / 1.00 |
  | 0.1 | 0.613 | 0.26 / 0.25 / 0.27 | 1.00 / 0.93 / 1.00 |
  | 0.05 | 0.605 | 0.30 / 0.23 / 0.18 | 1.00 / 0.95 / 1.00 |
  | 0.02 | 0.566 | 0.26 / 0.18 / 0.16 | 1.00 / 0.84 / 1.00 |
  | 0.01 | 0.520 | 0.28 / 0.08 / 0.16 | 1.00 / 0.64 / 1.00 |

- **New range: `S_E0 = min(omega × FPR_target, 0.2)`** (chosen cap 0.2; design calibration
  failure 0.641).
- **[observation]** Every cap is in band, so accessibility barely moves the YB failure rate. The
  Axis-B flag dominates: ALIGNED YB fails in 0.08–0.35 of runs, INVERTED YB in 0.64–1.00. The
  Stage 0 YB imbalance (0.946) came mainly from the hidden inversion, not from accessibility.

### E004a — Stage 0b record (DESIGN SPLIT ONLY; 2026-09-26)

No test or shift structure was generated, no Stage 1 predictor was fitted, the Stage 1 criteria
were not touched, and no E004b work was started. Everything below is an oracle/design analysis on
the frozen Stage 0b design panel.

**Runs.**

| step | commit | output |
| --- | --- | --- |
| amendment | `52ed06c` | Amendment 3 (before any Stage 0b code) |
| implementation | `c7e69fa` | TDD, RED observed for every new test; 8/8 mutants caught |
| YB calibration | `f768e38` → record `6156b88` | `results/E004a-stage0b-calibration/20260926T010125Z_f768e38` (cap 0.2) |
| panel | `83591a9` → frozen `203ff31` | `configs/e004/design_panel_0b.json`, sha256 `c7da2c31…da84` (768 structures) |
| runs | `a5e0ca0` → output `71dc4f9` | `results/E004a-stage0b/20260926T010655Z_a5e0ca0` (raw `runs.npz` local) |
| analysis | `f4c5c90` script; run at `71dc4f9` → output `594d7f9` | `results/E004a-stage0b-analysis/20260926T012014Z_71dc4f9` |
| provenance fix | `38e14e9` | see disclosures |
| hard pairs | run at `594d7f9` → output `324bbf8` | `results/E004a-stage0b-hardpairs/20260926T013006Z_594d7f9`; frozen `hardpairs_frozen.json`, sha256 `0dfecbb0…6e8a` |
| [post-hoc] | script `bbf941b`; run at `324bbf8` → output `13f92a5` | `results/E004a-stage0b-posthoc/20260926T015308Z_324bbf8` |

All run-directory names carry no `-dirty` suffix: every run started from a clean tree.

**Process and provenance disclosures.**

- **`meta.json` dirty flag.** Every Stage 0b `meta.json` written before `38e14e9` (calibration,
  runs, analysis; also every earlier E004a run) says `"dirty": true`. The cause is that
  `write_metadata` ran `git status` after `create_run_dir` had created the untracked run
  directory, so the run directory counted as a change. The run-directory name is computed before
  the directory exists and is the authoritative clean/dirty record. Fixed in `38e14e9` (TDD, RED
  observed): the run directory being written is ignored, and any other change still counts. The
  hard-pair and post-hoc `meta.json` files record `"dirty": false`.
- **Copied config.** The `config.toml` copied into each run directory is `configs/e004/e004a.toml`,
  which still holds the Stage 0 constants (`root_seed = 20260930`, Stage 0 panel sha256
  `e49753e9…`). The Stage 0b values are recorded as top-level `meta.json` fields:
  - runs and analysis: `root_seed = 20261001` and `panel_sha256 = c7da2c31…`;
  - calibration: no root-seed field (the script draws stream 4 of `SeedSequence(20261001)`);
  - hard pairs: `root_seed` only (panel file unchanged since `203ff31`; sha256 re-verified
    `c7da2c31…` at record time).
- **Hard-pair seeds.** The partner-base draw uses `default_rng([20261001, 8])`, outside the
  registered spawn tree. Verification seeds use `SeedSequence(20261001).spawn(5)[3].spawn(10)[8]`
  as registered.
- **Cosmetic.** `fig_alpha_reward_vs_update.png` keeps the Stage 0 axis range `[-3, 6]`; all Stage 0b
  points lie in `[-1.7, 0.8]`.

**Panel composition** (768 structures; actual Axis B from `toy.axis_b`).

| mechanism | ALIGNED | INVERTED |
| --- | --- | --- |
| R | 96 | 0 |
| X | 96 | 0 |
| YA | 85 | 107 (incl. 11 noise-induced) |
| YB | 86 | 106 (incl. 10 noise-induced) |
| B | 96 | 0 |
| D | 0 | 96 |

- **Noise-induced inversions** (intended ALIGNED, actual INVERTED; flagged, kept): 21 in total
  (YA1 3, YA2 2, YA3 6, YB1 4, YB2 3, YB3 3); 18 after exclusion.
- **Axis redraws** (intended INVERTED not actually inverted): YA2 2, YB2 3, D1 2.
- **Calibration rejections** (channel / target redraws):
  - R1 1100 / 24; R3 620 / 12;
  - X1 2977 / 62; X2 883 / 22; X3 955 / 26;
  - YA1 1767 / 39; YA3 2629 / 58;
  - YB1 1942 / 40; YB3 2925 / 67;
  - D3 2861 / 64; D2 233 / 9;
  - the others ≤ 206 / ≤ 16.
- **Exclusions** (clean gain < 0.1): 43 structures:

  | R1 | R3 | X1 | X2 | X3 | YA1 | YA2 | YA3 | YB1 | YB2 | YB3 | B1 | B3 | D1 | D2 | D3 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | 1 | 3 | 3 | 3 | 3 | 2 | 3 | 5 | 2 | 4 | 5 | 1 | 4 | 2 | 1 | 1 |

  R2 and B2 had none.
- **Horizon** (registered rule): sampled Adam `T = 2700` (median clean `t95` 880); NG `T = 23.4`
  (median 7.80). Identical to Stage 0 because the rule rounds to a grid.

#### Gates (Amendment 3 §11)

| gate | result | detail |
| --- | --- | --- |
| **G1** leakage | **PASS** | L0 → construction macro-F1 0.057 (chance 0.056; threshold 0.156). L1 → construction 0.257 (reported, not gating). L0 → mechanism 0.118 (chance 0.167). L0 → Axis B AUROC 0.58. |
| **G2** outcome diversity | **PASS** | Sampled-Adam failure: X 0.572, YA 0.353, YB 0.532, D 0.677 (all in `[0.2, 0.8]`); R 0.01, B 0.01. |
| **G3** theory nulls | **PASS** | F1 max deviation inside the §7 domain 7.0e-10 (1273 of 1408 states; 9.6% excluded). **Over all states 2.6e-9**, so the pass does not depend on the domain. F2 8.6e-8. Canonical R1 `alpha_r + 2 eps` 4.4e-16. |
| **G4** cross-construction | **PASS** | Test D, L3 at `h*`: mean macro-F1 0.540 (folds 0.541 / 0.369 / 0.711; threshold 0.317). L3 − L0 = 0.438, 95% CI [0.398, 0.477]. |
| **G5** within-mechanism | **PASS** | L3 − L2 ≥ 0.02 with CI lower bound > 0 in 4 mechanisms: X (failure +0.164 [0.076, 0.254]; `Dn` +0.151 [0.086, 0.210]); YA (`Dn` +0.026 [0.008, 0.046]); YB (failure +0.036 [0.007, 0.067]); D (`Dn` +0.052 [0.007, 0.092]). |
| **G6** optimizer validity | **PASS** | Material: YA, Adam − NG failure = **+0.309** (≥ 0.3). Robust: X (Adam 0.57 / NG 0.54) and D (0.68 / 0.82). |

#### Registered Stage 0b predictions (§12)

- **S0b-P1: HOLDS.** DECLINE fraction, INVERTED vs ALIGNED (sampled Adam, actual Axis B):
  - YA 0.556 vs 0.003;
  - YB 0.732 vs 0.015.
  - With the intended axis instead: YA 0.578 vs 0.037, YB 0.747 vs 0.069.
- **S0b-P2: HOLDS.** Failure, INVERTED vs ALIGNED: YA 0.596 vs 0.044; YB 0.914 vs 0.100.
- **S0b-P3: HOLDS.** `alpha_r < 0` at `t = 0` for every kept R structure, under both the Adam
  and the NG metric.
- **Agent's priors:**
  - "G2 at risk for D": wrong (D 0.68);
  - "G5 passes through X and YA": right, and it also passes through YB and D;
  - "geometry mostly encodes Axis A": largely right in the primary panel (below).

#### Outcome map

Failure / DECLINE / STALL fraction per construction × actual Axis B (runs = kept structures × 4
seeds; NG and canonical twins are deterministic, one run each, same kept structures):

| cell | sampled Adam | NG | MF-Adam | Adam, canonical | NG, canonical |
| --- | --- | --- | --- | --- | --- |
| R1 / R2 / R3 | .00 / .00 / .03 | .00 / .00 / .07 | .00 / .00 / .03 | 0 | .00 / .00 / .07 |
| X1 | .48/.00/.48 | .41/.00/.41 | .48/.00/.48 | .48/.00/.48 | .52/.00/.52 |
| X2 | .64/.00/.64 | .62/.00/.62 | .62/.00/.62 | .62/.00/.62 | .62/.00/.62 |
| X3 | .59/.00/.59 | .59/.00/.59 | .62/.00/.62 | .58/.00/.58 | .66/.00/.66 |
| YA1 A | .08/.01/.07 | 0 | .03/.03/.00 | 0 | 0 |
| YA1 I | .75/.71/.04 | .06/.00/.06 | .79/.79/.00 | .78/.77/.01 | .09/.00/.09 |
| YA2 A | .01/.00/.01 | 0 | 0 | 0 | 0 |
| YA2 I | .29/.23/.07 | .03/.00/.03 | .29/.24/.06 | .36/.29/.07 | 0 |
| YA3 A | .04/.00/.04 | 0 | .04/.00/.04 | 0 | 0 |
| YA3 I | .74/.73/.01 | .14/.09/.06 | .66/.66/.00 | .78/.75/.03 | .14/.06/.09 |
| YB1 A | .04/.00/.04 | .07/.00/.07 | .04/.00/.04 | 0 | 0 |
| YB1 I | 1.0/.98/.02 | .91/.85/.06 | 1.0/.94/.06 | .89/.88/.01 | .82/.82/.00 |
| YB2 A | .07/.00/.07 | 0 | .07/.00/.07 | 0 | 0 |
| YB2 I | .75/.22/.53 | .48/.16/.32 | .74/.29/.45 | .76/.44/.32 | .45/.16/.29 |
| YB3 A | .19/.04/.14 | .18/.00/.18 | .14/.04/.11 | 0 | .04/.00/.04 |
| YB3 I | .98/.98/.01 | .94/.77/.16 | 1.0/1.0/.00 | .94/.94/.00 | .81/.71/.10 |
| B1 / B2 / B3 | .00 / .03 / .00 | 0 | .00 / .03 / .00 | 0 | 0 |
| D1 | .83/.60/.23 | .90/.40/.50 | .83/.63/.20 | .68/.39/.29 | .90/.30/.60 |
| D2 | .60/.28/.32 | 1.0/.58/.42 | .61/.29/.32 | .48/.23/.25 | 1.0/.35/.65 |
| D3 | .60/.59/.01 | .55/.39/.16 | .65/.58/.06 | .44/.39/.05 | .42/.29/.13 |

Pooled failure (Adam / NG / MF-Adam):

- ALIGNED 0.15 / 0.13 / 0.14 (X is almost all of it: STALL on deleted prompts);
- INVERTED 0.73 / 0.54 / 0.73.
- YA: ALIGNED 0.04 / 0.00, INVERTED 0.60 / 0.08.
- YB: ALIGNED 0.10 / 0.08, INVERTED 0.91 / 0.78.
- D 0.68 / 0.82.

Observations:

- **Designed inversion now yields DECLINE even in the canonical twins** (e.g. YA1-I, Adam
  canonical DECLINE 0.77). In Stage 0 the canonical twins only stalled. Axis B is now a designed
  property, not a background-noise artifact.
- **ALIGNED Y failures are noise-driven.** The canonical twins of ALIGNED Y fail 0.00 under Adam.
  The primary panel's 0.01–0.19 (YB3 A 0.19, mostly STALL) comes from background coins and
  coupling.
- **Optimizer dependence is mechanism-specific and changes sign:**
  - YA-INVERTED fails 0.60 under Adam but 0.08 under NG;
  - D2 fails 0.60 under Adam but 1.00 under NG;
  - X, YB-I and D3 are close under both.
  - MF-Adam tracks sampled Adam at the cell level (max |Δ| 0.08).

#### Two alphas (update vs reward level; `t = 0`)

Medians per construction:

- **NG:** `alpha_u = alpha_r` exactly, by construction.
- **Adam metric:** `alpha_u` is lower than `alpha_r` by up to 0.13 (R2 −0.35 vs −0.24; X2 −0.63 vs
  −0.50; D3 −0.47 vs −0.37); X1 and D2 are about equal. The GRPO population normalization moves the update direction
  slightly further from gold than the raw reward gradient. It never changes the sign pattern.
- **Primary-panel `alpha_u0` is negative for every construction except part of B:**
  - R, YA, YB about −0.25 to −0.35; X −0.42 to −0.63; D −0.47 to −0.74;
  - B1 −0.05, B3 −0.08, B2 −0.21;
  - the fraction positive is 0.26 / 0.09 / 0.25 for B1 / B2 / B3.
- **The Stage 0 artefact (`alpha > 0` almost everywhere) is gone** at the update level: the
  normalized gold direction removes the GRPO gain.

#### Signatures: canonical vs coupled/noisy

Values at `t = 0`, sampled Adam unless stated.

- **`C_out/C` is ≈ 0 for R, X, B1 and B3, and 0.88–0.99 for YA1, YA3, YB, B2 and D** (YA2 0.51).
  - This holds in both twins and under both optimizers.
  - It encodes whether the verifier reads out-of-context features. That is an **Axis-A
    (structure)** indicator; it now also covers B2 (process credit via `z3`), so it no longer
    separates exploit from benign credit.
- **`alpha_u0` carries the Axis-B (preference) signal only in the canonical twins.** INVERTED vs
  ALIGNED AUROC of `-alpha_u0`:
  - canonical Adam: YA 0.90, YB 0.82;
  - primary Adam: 0.61 / 0.61; primary NG: 0.67 / 0.67.
  - Background coins pull every `alpha` to about −0.3 and mask the preference relation.
- **Canonical twin values:**
  - `alpha_u0`: X1/X2 −1.00 (Adam); B +0.06 to +0.19; YA −0.01 to −0.11; R1 −0.15 (Adam), −0.09
    (NG);
  - `d(C/A)` over 10% of `T` is about 1.5× larger in the canonical twins for YA1 and YB1 (+0.46
    vs +0.30; +0.59 vs +0.40).
- **D under Adam:** `C/A` *falls* over the first 10% of `T` (−0.07 to −0.11). Under NG it rises for
  D1 and D2 (+0.07, +0.06); D3 is −0.02.

#### Leakage tests (Amendment 3 §9)

- **A (construction recognition, per structure, t = 0):**
  - L0 0.057 (chance 0.056);
  - L1 0.257;
  - L0 → mechanism 0.118; L0 → Axis B AUROC 0.58.
- **B (leave-one-construction-out mechanism recognition, `h*`), macro-F1:**
  - L0 0.00, L1 0.35, L2 0.20, L3 0.46, L3+ 0.61.
  - Observables alone do not transfer mechanism identity to an unseen construction; geometry does.
- **C (within mechanism, `h*`):** grouped CV inside each mechanism; L0 / L1 / L2 / L3.

  | mechanism, target | L0 | L1 | L2 | L3 | L3 − L2 [95% CI] | L3 − L1 [95% CI] |
  | --- | --- | --- | --- | --- | --- | --- |
  | X failure | .581 | .833 | .697 | .861 | +.164 [.076, .254] | +.028 [−.017, .085] |
  | X `Dn` | .444 | .759 | .617 | .768 | +.151 [.086, .210] | +.009 [−.018, .036] |
  | YA failure | .739 | .892 | .829 | .860 | +.031 [−.012, .070] | **−.033 [−.062, −.008]** |
  | YA `Dn` | .621 | .714 | .664 | .690 | +.026 [.008, .046] | **−.024 [−.045, −.005]** |
  | YA Axis B | .695 | .723 | .712 | .740 | +.027 [−.007, .061] | +.017 [−.022, .053] |
  | YB failure | .723 | .846 | .817 | .853 | +.036 [.007, .067] | +.007 [−.017, .029] |
  | YB `Dn` | .640 | .741 | .715 | .735 | +.020 [−.003, .041] | −.006 [−.023, .010] |
  | YB Axis B | .681 | .780 | .774 | .756 | −.018 [−.062, .023] | **−.024 [−.049, −.001]** |
  | D failure | .773 | .904 | .915 | .926 | +.011 [−.058, .076] | +.022 [−.042, .090] |
  | D `Dn` | .519 | .681 | .690 | .742 | +.052 [.007, .092] | +.060 [.027, .091] |
  | R `Dn` | .670 | .670 | .639 | .625 | −.014 [−.035, .005] | **−.046 [−.091, −.009]** |
  | B `Dn` | .419 | .476 | .492 | .497 | +.006 [−.030, .043] | +.022 [−.011, .054] |

  R and B failure: skipped (< 10 failing runs).
- **D (cross-construction, `h*`), mean macro-F1:**
  - L0 0.124, L1 0.476, L2 0.280, L3 0.540, L3+ 0.659;
  - L3 − L2 = +0.259 [0.221, 0.296].
- **E (type oracles, leave-one-structure-out):**
  - mechanism: C-index `Dn` 0.608, AUROC failure 0.662;
  - mechanism × Axis B: 0.725 / 0.857.

#### Oracle information ceilings

Exact observables, grouped 5-fold CV, design split; not Stage 1 predictors.

Sampled Adam at `h* = 2%`:

| level | C-index `Dn` | AUROC failure | not-yet-visible AUROC (355 failing) | mechanism macro-F1 | Axis-B AUROC |
| --- | --- | --- | --- | --- | --- |
| L0 | 0.597 | 0.667 | 0.708 | 0.121 | 0.586 |
| L1 (`t = 0` geometry) | 0.722 | 0.876 | 0.871 | 0.520 | 0.768 |
| L2-G | 0.614 | 0.712 | 0.645 | 0.175 | 0.597 |
| L2 | 0.632 | 0.756 | 0.712 | 0.368 | 0.663 |
| L3 | 0.724 | 0.887 | 0.866 | 0.640 | 0.795 |
| L3+ | 0.717 | 0.891 | 0.871 | 0.781 | 0.862 |

- **At 10% of `T`:** L2 0.679 / 0.820 / 0.768; L3 0.738 / 0.926 / 0.897.
- **NG (secondary), `h*`:** L1 0.787 / 0.919; L2 0.686 / 0.860; L3 0.802 / 0.952. Not-yet-visible
  NG failures: n = 6 (uninformative).
- The mechanism × Axis-B type oracle (0.725 / 0.857) is at the level of L1 and L3.

#### [post-hoc, not pre-registered] Robustness of the gate passes and the L1 question

Script `experiments/e004/stage0b_posthoc.py`, run `results/E004a-stage0b-posthoc/20260926T015308Z_324bbf8`
(clean tree). Nothing below changes a gate.

- **G6 margin:**
  - YA Adam − NG = 0.309, 95% structure-bootstrap CI [0.246, 0.374].
  - The material-dependence criterion is met at the point estimate only; the CI straddles 0.3.
- **G5 under multiplicity:**
  - Holm correction over the 12 evaluated L3 − L2 targets, one-sided, normal approximation from
    the bootstrap CIs.
  - Passing: X `Dn`, X failure and YA `Dn` (p = 0.0034 vs 0.0050).
  - Two mechanisms still pass, so G5 survives, but its second mechanism is marginal.
- **Within mechanism × Axis-B cells** (beyond the type oracle), sampled Adam, `h*`, AUROC failure,
  L0 / L1 / L2 / L3:

  | cell | L0 | L1 | L2 | L3 |
  | --- | --- | --- | --- | --- |
  | X-A | .58 | .83 | .70 | .86 |
  | YA-I | .63 | .92 | .89 | .92 |
  | YB-I | .66 | **.94** | .81 | .83 |
  | D-I | .77 | .90 | .92 | .93 |
  | YB-A (10% failing) | .52 | .58 | .68 | .71 |

  The oracle geometry carries outcome information beyond the mechanism × Axis-B type.
- **Almost all of the L3 advantage is already in L1, the geometry at `t = 0`.**
  - Full-panel ceilings at `h*`: L1 0.876 vs L3 0.887 (failure AUROC); 0.871 vs 0.866
    (not-yet-visible failures).
  - Within mechanism, L3 − L1 is ≈ 0 or negative (YA, R, YB Axis B), and positive only for D `Dn`
    (+0.060 [0.027, 0.091]).
  - The dynamic part of L3 (change and slope over `[h/2, h]`) adds mechanism recognition (0.52 →
    0.64), not outcome information.
  - This is the main caveat for Stage 1. See the decision section.

#### Hard pairs (frozen)

MF-Adam search (100 restarts, 10 per type, 1242 s); verification with sampled Adam, 32 + 8 clean
seeds per member. The search objective has no outcome or Axis-B term; the lowest-SSE restart is
taken as is.

| type | anchor / partner base | accepted (MF / sampled) | sampled failure (anchor, partner) | detail |
| --- | --- | --- | --- | --- |
| HP-A (B vs YA) | B3-A-22 / YA3-A-26 | yes / yes | 0.00, 0.00 | both ALIGNED, both SUCCESS: a **mechanism pair only**; Correction A max ratio 0.012 |
| HP-D (D vs B) | D1-I-00 / B2-A-21 | yes / yes | 1.00, 0.00 | `Dn` 0.77 vs 0.00; max ratio 0.083. **Latent-decline outcome pair**, as in Stage 0 |
| HP-B (YA vs YB) | YA1-A-00 / YB3-A-27 | yes / yes | 0.03, 0.00 | both ALIGNED; mechanism pair |
| HP-I (YA-I vs YA-A) | YA1-A-12 / YA1-A-16 | no / no | 1.00, 1.00 | L2-matched, but the fitted partner is INVERTED: the axis constraint fails |
| DYN-YA/R | YA1-I-24 / R3-A-30 | **yes / yes** | 1.00, 0.00 | L0 ratio 0.008; L1 within tolerance; L2 ratio 0.030. L3 diverges at **1%** of `T` (sampled; MF 7%); the failing member's median visible onset is at **2%** (MF 24%) |
| DYN-YB/B | YB1-A-18 / B2-A-22 | no / no | 0.00, 0.00 | both ALIGNED; no failing member |
| DYN-D/B | D3-I-12 / B2-A-17 | no / no | 1.00, 0.00 | infeasible: L1 cannot be matched (max scaled residual 3.79); no L3 divergence (as in Stage 0) |
| DYN-YA-I/YA-A | YA2-I-25 / YA1-A-29 | no / no | 0.00, 0.00 | the INVERTED anchor succeeds in both tiers (inverted but not accessible) |
| DYN-X/YB | X1-A-27 / YB2-A-10 | no / no | 0.375, 0.00 | failure below 0.5; divergence (15–17%) after the onset (1%) |
| DYN-YA/B | YA1-I-06 / B3-A-06 | no / no | 1.00, 0.00 | outcome-divergent, but no L3 divergence before the onset (1.5% of `T`) |

- **Accepted in both tiers:** HP-A, HP-D, HP-B and DYN-YA/R. **Outcome-divergent among them:**
  HP-D and DYN-YA/R only.
- **Stage 0's HP-A outcome pair (B SUCCESS vs YA DECLINE) is not reproduced.** The best match now
  pairs B with an ALIGNED YA.
- **One dynamic pair survives (DYN-YA/R).** Its sampled-Adam lead is one grid step (1% of `T` = 27
  steps) before visible failure.
- [post-hoc interpretation] Failures under sampled Adam become visible early:
  - 698 of 1097 failing runs (64%) are visible by 1% of `T`, and 742 (68%) by `h* = 2%`;
  - the failing hard-pair members have median onset ≤ 2%.
  - This leaves little room for a dynamic lead.

#### Decision rule outcome (Amendment 3 §11)

All of G1–G6 pass, so the registered recommendation is **PROCEED TO E004a STAGE 1**. Stage 1 does
not start without collaborator approval.

Caveats for the collaborator (observations; the Stage 1 criteria are unchanged):

1. G6's material criterion passes by 0.009 (CI [0.25, 0.37]), and G5's second mechanism is
   marginal under a multiplicity correction.
2. **The oracle's predictive content is mostly the `t = 0` update geometry (L1), not the early
   trajectory.** A Stage 1 success of L3 over L2 would, on this panel, largely be a success of the
   initial geometry. A pure "short-horizon dynamics" reading would be too strong.
3. In the primary panel, geometry mainly encodes Axis A (`C_out/C`). Background coins mask the
   Axis-B signal in `alpha` (AUROC 0.61 vs 0.82–0.90 canonical).
4. The frozen dynamic hard-pair set is thin: one pair (DYN-YA/R), with a 1%-of-`T` lead under
   sampled Adam.
5. These are oracle quantities: `g_G` is not observable in a real verifier setting.

---

### E004a — Stage 1 execution note (design round; 2026-09-26)

Written and committed **before any Stage 1 code or output.**

**Status.**

- Governs the Stage 1 design round ("Proceed with E004a Stage 1").
- **Unchanged:**
  - the panel (Stage 0b design panel);
  - the outcomes (pre-registration §7);
  - the horizons and `h* = 2%`;
  - the model families;
  - the Stage 1 success criteria (§12).
- This note only operationalizes points that §6, §12, memo §5–§11 and Amendment 3 leave open. It
  is written before any Stage 1 sampled output. The Stage 0b oracle results are known and
  disclosed.
- Where the brief and the frozen text differ, the frozen text governs; see §9 below.
- This round: design split only. No test or shift panel is generated; no E004b work.

**1. Data and outcomes.**

- **Panel:** `configs/e004/design_panel_0b.json` (sha256 `c7da2c31…`). 725 kept structures (43
  excluded by the registered clean-gain rule).
- **Runs:** 2900 sampled-Adam runs (4 seeds each) and 725 NG runs.
- **Adam trajectories** are re-run with the Stage 0b seeds (runs stream, roles `prim_ver` /
  `prim_clean`), so they are identical to Stage 0b. Asserted: exact `J_G` at every checkpoint
  equals the committed Stage 0b values within 1e-12.
- **Outcomes** (`Dn`, category, failure, `t_on`) are recomputed and asserted equal to the committed
  Stage 0b labels.
- **NG:** the exact flow, `T_ng = 23.4`.
- Outcomes use exact gold values (§7); finite samples enter the features only.

**2. Finite-sample measurement.**

Per run, at the feature checkpoints `H ∪ {h/2}`: Adam steps 0, 3, 5, 7, 14, 27, 54, 68, 135, 270;
NG times at the same fractions of `T_ng`.

- **Audit:** 256 fresh rollouts from the current policy, as 32 prompt groups × 8 responses
  (prompts ~ `w`). Each rollout has a gold label `G` and one verifier call `V` (fresh coin). One
  audit per run and checkpoint, shared by every level.
- **Training batch (Adam):** the batch drawn from the checkpoint policy for the next update
  (8 groups × 8). It is recorded without changing the training random stream.
- **Observables:**
  - `J_G` = audit mean of `G`;
  - `J_V` = mean `V` over the audit plus the training batch (320 verifier calls; NG: audit only,
    256);
  - `FPR = #(V=1, G=0)/#(G=0)` and `FNR = #(V=0, G=1)/#(G=1)` on the audit; an empty denominator
    gives NaN;
  - FP mass = `#(V=1, G=0)/256`.
- **Geometry:** the plug-in estimator with a pooled verifier gradient (the E002 G1 tuning winner).
  - Scores: the exact score function of each sampled row at the current `theta`.
  - **Update level.** GRPO advantages within each group, `(R − group mean)/(group std + 1e-6)`,
    applied identically to `V` and `G`.
    - `g~_V` = mean over all 40 pooled groups (audit + training) of the group-mean
      advantage·score.
    - `g~_G` = the same over the 32 audit groups, with `G`.
  - **Reward level.** Leave-one-out baseline within each group (RLOO); `g_V` pooled, `g_G` audit.
  - **Metric:**
    - Adam at `t > 0`: the run's bias-corrected `v_hat` (the actual optimizer state),
      `diag(1/(sqrt(v_hat) + 1e-8))`;
    - Adam at `t = 0`: `v_hat_0 = mean(gamma)^2 + var(gamma)/8` per coordinate, where `gamma_k` is
      the verifier update contribution of pooled group `k` (40 groups). This is the
      finite-sample analogue of the registered `g~^2 + sigma~^2/64`;
    - NG: the damped inverse of the audit Fisher estimate, `(F_hat + 0.1·tr(F_hat)/d·I)^-1`
      (E002 winner `lam = 0.1`).
    - The Adam metric only approximates the effective geometry (momentum, `v_hat` history,
      noise correlation). Every report states this.
  - `(A, alpha, C)` by `toy.decompose`.
    - `A = 0` gives `alpha = NaN` and `alpha_defined = False`; recorded estimates are never
      imputed.
    - `C_in` / `C_out` (L3+ only): the residual projected on the span of the per-prompt audit
      gold contributions. Prompts absent from the audit contribute no direction.
- **Levels** (registry §6; Amendment 3 §6):
  - L0; L1 = L0 + `(A_u, alpha_u, C_u)` at 0; L2; L2-G; L3 = L2 + summaries of
    `(A_u, alpha_u, C_u)`; L2+ and L3+ secondary.
  - Summaries: the value at `h`, the change `0 → h`, and the slope over `[h/2, h]` per 1% of
    `T`. At `h = 0` the change and slope are 0.
  - `alpha_r` (reward level) is reported at every horizon. It enters only a secondary
    sensitivity arm: L1r = L1 + `alpha_r0`; L3r = L3 + summaries of `alpha_r`. It is never used
    for a criterion.

**3. Predictors (design split only).**

- **Pipeline:** median imputation (fit on the training fold) → standardization → model. There
  are no missingness indicators, so dimensions stay as registered. NaN counts are reported.
- **Primary models:**
  - `Dn`: ridge;
  - failure: L2 logistic;
  - mechanism: 6-class multinomial logistic.
  - Penalty grid `logspace(-3, 3, 13)` (alpha for ridge, C for logistic), selected by 5-fold
    `GroupKFold` by structure inside the training data. Logistic selection uses log-loss.
- **Evaluation:** out-of-fold predictions from 5-fold `StratifiedGroupKFold` by structure
  (stratified by mechanism; fixed shuffle seed). The same folds are used for every level and
  horizon, so comparisons are paired.
- **Secondary:** gradient boosting (depth 2, 100 trees, learning rate 0.1), same folds, no
  tuning.
- **Single-variable diagnostics** (fixed sign; memo §8): L0 `FPR(0)`; L1 `C_0/A_0`; L2 `ΔFPR(h)`
  and `−ΔJ_G(h)`; L3 `ΔC(h)` and `−Δalpha(h)`.
- **Noise control** (memo §8): L2 + 9 permuted-noise features. Each of L3's 9 geometry columns is
  permuted across runs with a fixed seed.
- **Frozen model:** the same pipeline refit on all kept design runs, at each horizon.

**4. Metrics.**

- **Standard metrics:**
  - C-index of `Dn`; AUROC of failure (STALL ∪ DECLINE);
  - mechanism macro-F1, balanced accuracy and the confusion matrix;
  - Route A vs B AUROC: among YA ∪ YB runs, the score `p_YA/(p_YA + p_YB)` from the 6-class
    out-of-fold probabilities.
- **Not yet visible (NYV) at `h`:** runs with `t_on > h`; AUROC of eventual failure among them.
- **Warning threshold and lead time** (L2 primary; also L0, L1, L3):
  - running score `m_h` = the maximum over registered `h' <= h` of the out-of-fold failure
    probability `p_h'`;
  - `tau_h` = the 90th percentile of `m_h` over design SUCCESS runs, so the cumulative
    false-alarm rate among SUCCESS runs is 10%;
  - a run is warned at `h` iff `m_h > tau_h`; `t_warn` = the first `h' <= h` with
    `p_h' > tau_h`;
  - lead time = `t_on − t_warn` (% of `T`), for failures with `t_on > h`.
  - **Criterion value:** the median lead time over the warned NYV failures (where `t_warn` is
    defined).
  - Also reported: sensitivity (the warned fraction) and a conservative median that counts
    missed failures at lead 0.
  - Held-out use: the `tau_h` computed here (out-of-fold, design) are frozen and applied to the
    frozen models' held-out probabilities.
- **Within mechanism:** models fitted inside each mechanism (same pipeline, grouped CV) for L0,
  L1, L2 and L3, on `Dn` and failure. Failure is fitted only when both classes have ≥ 10 runs.
- **Type oracles:** leave-one-structure-out mechanism-mean and mechanism × Axis-B-mean `Dn`.
- **Cross-construction mechanism test:** 3 folds by construction slot (train on the other two
  constructions of every mechanism); L0, L2 and L3 at `h*`.
- **Design-side leave-one-mechanism-out diagnostic:** fit on 5 mechanisms, evaluate on the held-out
  mechanism's design runs. §12's generalization split B uses test structures and is not
  evaluated here.

**5. Inference.**

- **Hierarchical bootstrap, B = 2000:**
  - structures with replacement, then seeds with replacement within each drawn structure;
  - out-of-fold predictions held fixed;
  - arms paired through shared resamples;
  - the bootstrap seed comes from the registered tree.
- **p-values and bounds:**
  - one-sided `p = (1 + #{Δ* <= 0})/(B+1)`;
  - "lower 95% bound" = the 5th percentile (one-sided, matching the one-sided p);
  - two-sided 95% intervals are also reported.
- **Claims:**
  - "beats" = Holm-adjusted `p <= 0.05` within the family **and** a point estimate ≥ the margin
    (`δ_out = 0.02`, `δ_mech = 0.03`); ties are within ±0.01;
  - RQ1 family = 2 tests; mechanistic (i)–(iii) is a conjunction (no correction).

**6. Hard pairs** (frozen file sha256 `0dfecbb0…`; evaluated after the predictor freeze; no new
search).

- **Runs:**
  - every frozen pair's anchor (a panel structure) and partner (frozen parameters);
  - 32 verifier + 8 clean seeds per member: the Stage 0b verification seeds (hard-pair role), so
    the trajectories reproduce the verification runs;
  - audits from the spare role (§7).
- **Predictors:** the frozen pipeline refit on the design panel without the pair's anchor
  structure (leave-anchor-out).
- **Reported per horizon:**
  - the L2 feature distance in audit-SE units (member means; registered SE formulas at the pair
    mean);
  - the L3 geometry distance in pooled across-seed SD units;
  - failure risk scores, mechanism predictions and the actual outcomes.
- **Criterion (iii), operationalized:**
  - instances = runs of HP-A and HP-D (32 per member);
  - task: binary, the pair's two mechanisms. A run is correct iff the frozen 6-class model gives
    its true mechanism a higher probability than the partner's;
  - L2 is "at chance" on a pair iff a two-sided exact binomial test of L2 accuracy against 0.5
    does not reject at 0.05;
  - (iii) holds iff, pooled over the pairs where L2 is at chance, L3 accuracy ≥ 0.80 **and** a
    one-sided exact binomial test against 0.5 rejects at 0.05;
  - if L2 is at chance on neither pair, (iii) cannot be satisfied and counts as failed.
- **DYN-YA/R:** the per-run finite-sample `C_hat` distributions of the two members at each
  horizon ≤ 5%, with the single-run AUROC and the standardized mean difference.

**7. Seeds (registered tree).**

- Runs stream = `SeedSequence(20261001).spawn(5)[3]` → 10 roles.
- Training: roles `prim_ver` / `prim_clean` (as Stage 0b). Hard-pair training: role `hardpair`
  (as the Stage 0b verification).
- Audits and resampling: role `spare` → `spawn(4)` = [Adam audits, NG audits, hard-pair audits,
  bootstrap / permutation]. Within each branch: spawn per structure (panel order), then per seed.

**8. Provenance.**

- A Stage 1 config file (`configs/e004/e004a_stage1.toml`) holds the actual Stage 0b / Stage 1
  constants and is copied into every run directory.
- `meta.json` records:
  - the commit and dirty status;
  - the root seed, panel sha256 and config sha256;
  - package versions and `split = design`;
  - for hard-pair runs, the hard-pair file sha256.

**9. Brief vs frozen text** (resolved in favour of the frozen text; flagged to the collaborator).

- **Geometry in L1/L3.** The brief lists `alpha_reward` among the L1 and L3 inputs. Amendment 3
  §6 fixes L1/L3 at the update level, and the frozen dimension-matched control has 9 geometry
  columns. The primary levels follow the registry; `alpha_reward` enters only L1r/L3r.
- **Inner CV.** The Stage 0 oracle ceilings used 3 inner folds. Stage 1 uses 5 (memo §8; brief).
- Otherwise the brief matches the frozen text.

**10. Round deliverables, then stop.**

- Design-CV results.
- The frozen predictor configuration (JSON, committed).
- Hard-pair finite-sample results.
- The runtime / power report.
- Then STOP for collaborator approval.

---

### E004a — Stage 1 design-round record (DESIGN SPLIT ONLY; 2026-09-26)

No test or shift structure was generated or opened. No E004b work was done. The panel, outcomes,
horizons, model families and Stage 1 success criteria are unchanged. The predictor configuration
was frozen (`predictors_frozen.json`, sha256 `e0b2ea9d…d210`) **before** the hard-pair evaluation.
All numbers are design-CV (out-of-fold) results; they are not confirmatory.

**Runs.**

| step | commit | output |
| --- | --- | --- |
| execution note | `b157fad` | registry (before any Stage 1 code or output) |
| implementation (TDD) | `63ea693` | `vdyn.e004.audit`, `vdyn.e004.predict`, batch recording, finite levels, provenance config sha |
| sampled runs + audits | run at `63ea693` → `26b1790` | `results/E004a-stage1-runs/20260926T045733Z_63ea693` (149 s; peak RSS 379 MB workers, 168 MB main) |
| design-CV analysis + **frozen predictors** | run at `26b1790` → `341851a` | `results/E004a-stage1-analysis/20260926T050431Z_26b1790` (349 s; 661 MB) |
| hard pairs (frozen pairs, frozen predictors) | run at `9426135` → `23183b4` | `results/E004a-stage1-hardpairs/20260926T051047Z_9426135` (80 s; 264 MB) |
| [post-hoc] snapshot vs trajectory | script `a9afb8c` / `90e0341`; run → `385cb79` | `results/E004a-stage1-posthoc/20260926T051523Z_90e0341` |

- Every run directory is clean.
- **Every `meta.json` records:** `dirty: false`, root seed 20261001, panel sha256, config sha256
  (`configs/e004/e004a_stage1.toml`), package versions and `split = design`. The hard-pair
  `meta.json` also records the hard-pair and frozen-predictor sha256.

**Integrity and reproduction.**

- Adam and NG trajectories are identical to Stage 0b (max |Δ| = 0.0). The recomputed outcome
  labels equal the committed Stage 0b labels exactly.
- The hard-pair member failure fractions equal the frozen Stage 0b verification exactly
  (asserted).

**Process disclosures.**

- **RED observed after the code was written.** `audit.py` and `predict.py` were written before
  their first test run. RED was observed by moving each module aside (ImportError), then GREEN.
- **Three test-design errors were fixed in the tests, not the code:**
  - identical rollout rows (zero RLOO residual by construction);
  - an AUROC threshold that was unreachable for the synthetic signal-to-noise ratio;
  - a wrong keyword name.
- **Two commits went in after a failing check chain.** `a9afb8c` and `23183b4` were committed
  after a check chain that did not stop on failure:
  - ruff reported one E501, fixed in `90e0341`;
  - mypy reported 23 errors, all a stale `.mypy_cache` artifact. Full-scope mypy with a clean
    cache reports 0 issues.
- **BLAS oversubscription** (load ≈ 32 on 8 workers) during the analysis. It affects runtime
  only.

#### Finite-sample estimator checks (Adam; estimate − exact at the same states)

- **Observables are unbiased** (|bias| ≤ 0.002).
- **The empirical error SD exceeds the registered audit-SE formula:**
  - `J_G`: ×1.26–1.38 (32 groups × 8 is cluster sampling, which has a design effect);
  - FPR: ×1.11–1.15; FNR: ×1.02–1.08.
- **Geometry at `h*`:**
  - `alpha_u`: r = 0.84, error SD 0.15;
  - `C_u`: r = 0.87, bias **+0.042** (plug-in noise floor);
  - `A_u`: r = 0.77, bias −0.056;
  - `alpha_r`: r = 0.86.
- **Instability of the Adam metric at `t ≤ 0.5%` of `T`.** `v_hat` estimated from a few steps
  gives heavy tails:
  - `A_u` at `t = 0`: error SD 2.09;
  - `C_u` at 0.5%: r = 0.16, error SD 1.26.
- **NG** (damped audit Fisher): r ≈ 0.86–0.95 for the geometry at every checkpoint.
- **No undefined values:** alpha, FPR and FNR were never NaN (0 of 29 000 Adam audits).

#### Frozen criteria (§12) evaluated on design CV

| criterion | design-CV result | status on design |
| --- | --- | --- |
| **RQ1** `L2 − L0` at `h*` | C-index +0.023 (lo95 0.013; Holm p 0.001); AUROC +0.028 (lo95 0.012; Holm p 0.001) | met; the C-index point is 0.003 above `δ_out` |
| **Early warning** (L2, `t_on > h*`) | NYV AUROC 0.696, **lo95 0.64996**; median lead 8.9% of `T` (52 warned of 355 NYV failures; sensitivity 0.146; conservative median 0) | **not met**: the lower bound misses 0.65 by 4e-5 |
| **Mechanistic (i)** `L3 − L2` macro-F1 | +0.291 (lo95 0.266) | met |
| **Mechanistic (ii)** Route A vs B | +0.204 (lo95 0.168) | met |
| **Mechanistic (iii)** hard pairs | L2 at chance on HP-A (0.50) and HP-D (0.45); L3 0.50 and 1.00; pooled 0.75 < 0.80 | **not met** |
| Generalization B (design-side LOMO analog only) | `L2 − L0` lo95 > 0 in 1 of 6 mechanisms (D) | the design analog fails; the registered test uses held-out structures |

- **(iii) cannot change at held-out.** The pairs are frozen and design-side, and the predictors are
  frozen. A held-out evaluation would reproduce the same instances, so mechanistic value (all of
  (i)–(iii)) **cannot succeed**, and §12's "mechanistic-only" branch is unreachable.
- **NG (secondary).** RQ1 is not met (C-index −0.003; AUROC +0.039). Only 6 NYV failures remain at
  `h ≥ 1%`, so NG early warning cannot be interpreted.

#### Information curve (sampled Adam; point [lo95])

C-index of `Dn`:

| h | L0 | L1 | L2-G | L2 | L3 | L2 − L0 | L3 − L2 | L3 − L1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | .581 | .668 | .580 | .580 | .667 | −.001 | +.088 | −.001 |
| 0.2% | | | .580 | .592 | .646 | +.012 | +.054 | −.022 |
| 0.5% | | | .578 | .590 | .661 | +.009 | +.071 | −.008 |
| 1% | | | .579 | .593 | .688 | +.012 | +.095 | +.020 |
| **2%** | .581 | .668 | .581 | .604 | .695 | **+.023 [.013]** | +.091 [.077] | +.026 [.017] |
| 5% | | | .597 | .637 | .706 | +.057 | +.069 | +.038 |
| 10% | | | .624 | .667 | .717 | +.086 | +.051 | +.049 |

AUROC of failure:

| h | L0 | L1 | L2-G | L2 | L3 | L2 − L0 | L3 − L2 | L3 − L1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | .652 | .796 | .652 | .650 | .796 | −.001 | +.146 | .000 |
| 1% | | | .656 | .668 | .834 | +.017 | +.165 | +.038 |
| **2%** | .652 | .796 | .660 | .680 | .846 | **+.028 [.012]** | +.166 [.137] | +.050 [.037] |
| 5% | | | .684 | .738 | .849 | +.086 | +.112 | +.053 |
| 10% | | | .733 | .785 | .870 | +.134 | +.084 | +.074 |

- **Gold-only early dynamics add almost nothing at `h*`:** L2-G − L0 = +0.000 C-index and +0.008
  AUROC. The verifier-side observables are what L2 adds (L2 − L2-G = +0.023 / +0.020).
- **Controls:**
  - noise control: L2N − L2 = −0.004 / −0.005, so L3's gain is not "more columns";
  - L3r − L3 ≈ 0: the reward-level alpha adds nothing;
  - the GBM check agrees (at `h*`: L3 − L2 +0.082 / +0.160; L2 − L0 +0.025 / +0.045).
- **Single diagnostics** (at `h*`, AUROC): `C_0/A_0` 0.723 (the best), ΔFPR 0.593, −ΔJ_G 0.563,
  ΔC 0.599, −Δalpha 0.541.

#### Not-yet-visible curve and lead time

- **NYV failures / NYV non-failures:**
  - `h < 1%`: 1097 / 1803 (all runs; `t_on` lives on the 1% grid);
  - 1%: 399 / 1257;
  - 2%: 355 / 1207;
  - 5%: 308 / 1165;
  - 10%: 233 / 1138.
- **NYV AUROC** (L0 / L1 / L2 / L3):
  - 1%: .676 / .752 / .688 / .804;
  - **2%: .683 / .749 / .696 [.650] / .811 [.774]**;
  - 5%: .680 / .738 / .733 / .800;
  - 10%: .662 / .704 / .760 / .801.
- **Lead time at `h*`** (10% false alarms among SUCCESS runs):

  | level | warned NYV failures | sensitivity | median lead (warned) |
  | --- | --- | --- | --- |
  | L2 | 52 of 355 | 0.146 | 8.9% of `T` (IQR 5.4–14%) |
  | L3 | 112 | 0.315 | 9.0% |
  | L1 | 91 | 0.256 | 9.0% |
  | L0 | 50 | 0.141 | 10.0% |

  38% of the L2 warnings (20 of 52) are issued at `t = 0`.
- **[observation] The registered `t_on` is noise-sensitive early.** 30% of *non-failing* runs
  already cross the 10% onset threshold at 1% of `T` (546 of 1803). The clean progress in the
  denominator is still small there. So "not yet visible" at `h <= 2%` partly selects on noise.
  The definition is frozen and unchanged; this is a measurement caveat.

#### Mechanism diagnosis

- **Macro-F1** (L2 / L3; chance 0.167):
  - 0.2%: .118 / .409;
  - 1%: .132 / .434;
  - **2%: .156 / .447**;
  - 5%: .275 / .477;
  - 10%: .339 / .495;
  - L1 (`t = 0`): .371.
- **L2 is almost blind to mechanism at `h*`.** It predicts YA/YB for nearly every run.
- **Pairwise AUROCs at `h*`** (L2 / L3 / L1):
  - Route A vs B: .614 / .818 / .791;
  - B vs D (benign vs displacement): .484 / .992 / .954;
  - X vs Y (deletion vs exploit): .627 / .777 / .749.
- **Cross-construction** (train on 2 constructions per mechanism, test on the third; `h*`):
  L0 .116, L1 .359, L2 .154, L3 .415; L3 − L2 = +0.261 (lo95 0.237).

#### Within-mechanism outcome prediction (`h*`; C-index of `Dn` | AUROC of failure)

| mechanism | L2 | L3 | L3 − L2 [lo95] | L3 − L1 [lo95] |
| --- | --- | --- | --- | --- |
| X | .451 \| .585 | .731 \| .813 | +.280 [.215] \| +.228 [.154] | +.036 [.011] \| +.061 [.022] |
| YA | .605 \| .715 | .664 \| .806 | +.059 [.040] \| +.091 [.053] | +.046 [.026] \| +.060 [.027] |
| YB | .669 \| .722 | .720 \| .784 | +.051 [.028] \| +.062 [.027] | +.018 [.005] \| +.005 [−.019] |
| D | .555 \| .797 | .644 \| .874 | +.089 [.042] \| +.077 [.027] | .000 [−.034] \| +.016 [−.014] |
| R, B | ≤ 4 failures | | C-index only: −.002 / +.018 | |

- The overall gain is not only mechanism recognition: geometry adds within X, YA, YB and D.
- **Type oracles** (leave-one-structure-out mechanism-mean `Dn`):
  - mechanism: C-index .608, AUROC .662;
  - mechanism × Axis B: .725 / **.857**.
  - The finite-sample L3 at `h*` (.695 / .846) does not exceed the mechanism × Axis-B oracle.

#### L1 vs L3 (first-class question)

- **Registered comparison:** in finite samples L3 − L1 > 0 from 1% of `T` on (at `h*`: +0.026
  C-index, +0.050 AUROC, +0.062 NYV AUROC, +0.076 macro-F1). Unlike the oracle, L1 ≠ L3.
- **[post-hoc, not pre-registered] Equal audit budget.** L3 uses three independent audits; L1
  uses one. Two extra `t = 0` audits per run (seed: spare → resample branch child 3):
  - L1x3 − L1 = +0.018 C-index, +0.029 AUROC, +0.042 macro-F1, +0.035 Route A/B. This is
    measurement noise alone.
  - At `h*`, **L3 − (L2 + mean `t = 0` geometry over three audits)**:
    - C-index +0.008 (lo95 0.001);
    - AUROC +0.017 (0.007);
    - NYV AUROC +0.021 (0.001);
    - macro-F1 +0.027 (0.008);
    - Route A/B −0.008.
  - At 10%: C-index +0.013, AUROC +0.016, NYV −0.003, macro-F1 −0.013.
- **Conclusion:** L3's advantage over L1 is mostly (a) the observable dynamics in L2 and (b) more
  audits of the same static geometry. The geometry *trajectory* adds < `δ_out` for outcome and
  nothing for Route A/B. **Useful geometry is mostly a snapshot property in this panel.** It
  should not be described as "early geometry dynamics".

#### Hard pairs (frozen; leave-anchor-out predictors)

| pair | sampled failure | L2 distance at `h*` (max, audit-SE units) | L3 geometry distance (max, SD units) | L2 / L3 mechanism accuracy at `h*` | L2 / L3 risk AUROC |
| --- | --- | --- | --- | --- | --- |
| HP-A (B vs YA-A) | 0 / 0 | 0.96 | 0.61 | .50 / .50 | — |
| HP-D (D vs B) | 1 / 0 | 0.49 | 3.84 | .45 / **1.00** | .48 / **1.00** |
| HP-B (YA-A vs YB-A) | .03 / 0 | 0.47 | 0.87 | .56 / .42 | — |
| DYN-YA/R | 1 / 0 | 0.50 | 0.45 | .50 / .66 | .55 / .89 |
| DYN-D/B (not accepted) | 1 / 0 | 0.41 | 2.77 | .55 / .94 | .47 / .99 |

- **HP-D: L2 is blind and L3 separates** (risk .92 vs .24). **L1 already separates at `t = 0`**
  (.84 vs .32; AUROC 1.0).
- **DYN-YA/R: L1 at `t = 0` gives AUROC .88**, the same as L3.
- **DYN-YA/R C measurability.** The oracle C separation does not survive sampling as a C
  difference:
  - at `t = 0` the exact `C` of the two members is identical (0.1211 vs 0.1212), yet `C_hat` is
    0.18 vs 0.09 (single-run AUROC 0.85). The plug-in noise floor depends on the structure;
  - at `h*` the exact `C` is 0.153 vs 0.208, while `C_hat` has SD 3.1 in the R member (unstable
    early Adam metric). The standardized mean difference is −0.22.
  - The pair's L3 separation comes from structure-dependent `t = 0` estimates, not from the
    registered C dynamics.

#### Runtime and power

- **Runtime:** runs 149 s, analysis 349 s, hard pairs 80 s (8 workers).
- **Peak memory:** ≤ 0.66 GB.
- **Held-out estimate:** test panel ≈ 1.5 min, runs ≈ 2.5 min, frozen-model evaluation with
  bootstrap ≈ 5 min, plus the shift panel ≈ 10 min. Total ≈ 15–25 min.
- **Power at `h*`:** 355 NYV failures and 1207 NYV non-failures (≈ 0.23 prevalence) on 725
  structures. The two-sided NYV-AUROC interval half-width is ≈ 0.055. That is enough events for
  inference, but the early-warning criterion sits at the boundary.

#### Decision position

On design data:

- RQ1 is met with a small margin;
- early warning misses by 4e-5;
- mechanistic value is met on (i) and (ii) but is **unreachable** through (iii);
- the design analog of generalization B fails.

The recommendation is given in the Stage 1 design report; held-out work waits for collaborator
approval.

---

### E004a Amendment 4 — Stage 1 decision-tree completion (2026-09-26)

Written and committed **before any held-out or shift structure, trajectory or outcome exists.**

**Status.**

- Collaborator request: "Proceed with the FINAL held-out evaluation for E004a Stage 1."
- This amendment only names the result combinations that §12 left undefined.
- **No registered criterion, threshold, definition, predictor or panel changes.**

**Registered components** (as frozen in §12 and in the Stage 1 execution note):

- **RQ1:** `L2 − L0 >= δ_out` with lower bound > 0 for both the C-index of `Dn` and the AUROC of
  failure, at `h*` on split A (Holm over 2).
- **EW (early warning):** among runs with `t_on > h*`, the L2 AUROC lower bound is ≥ 0.65
  **and** the median lead time at the design-calibrated 10% false-alarm threshold is ≥ 5% of `T`.
- **GEN (generalization):** the shift split has `L2 − L0` lower bound > −0.01, **and**
  leave-one-mechanism-out has `L2 − L0` lower bound > 0 in ≥ 4 of 6 held-out mechanisms.
- **MECH (mechanistic value), (i)–(iii):** already unsatisfiable. (iii) failed on the frozen,
  design-side hard pairs with the frozen predictors, and a held-out evaluation reproduces the same
  instances.

**Primary label.** Exactly one of three:

| label | condition | interpretation | consequence |
| --- | --- | --- | --- |
| **A — FULL EARLY-WARNING SUCCESS** | RQ1 ∧ EW ∧ GEN | confirmatory evidence that a short policy–verifier prefix predicts long-run failure before it becomes visible | the only branch that qualifies for **PROCEED TO E004b** |
| **B — EARLY ASSOCIATION, NO ROBUST EARLY-WARNING CLAIM** | RQ1 ∧ (¬EW ∨ ¬GEN) | early information is associated with the long-run outcome, but the evidence does not establish a robust, transferable early-warning system | the result is **not** called "early warning"; no E004b under the frozen rule |
| **D — NO CONFIRMATORY EARLY-PREDICTION RESULT** | ¬RQ1 | no confirmatory early-prediction claim | earlier mechanistic and static-insufficiency findings are retained only as such |

**C — mechanistic support without pre-registered mechanistic success** (reported separately; not
a primary label).

- If geometry remains useful for mechanism diagnosis on held-out data, it is reported as
  descriptive / supporting mechanistic evidence.
- The report states explicitly that **the pre-registered mechanistic-success criterion was not
  satisfied**, and §12's "mechanistic-only" branch is not invoked.

**Further rules.**

- **Abandonment flag.** §12's "abandon early warning" rule (at every `h <= 5%`, RQ1 fails or the
  NYV AUROC lower bound is ≤ 0.55) is scored and reported as a flag. It does not create a label.
- **Fixed label set.** No other label may be introduced after held-out data are seen. Post-hoc
  analyses cannot change the label.

---

### E004a Stage 1 — pre-unseal integrity log and held-out execution protocol (2026-09-26)

Written and committed **before the approval file, and before any held-out or shift structure or
outcome exists.**

**Integrity check: PASS** (`results/E004a-stage1-integrity/20260926T055219Z_41ab28b`; HEAD
`41ab28b`). All 13 checks passed:

1. working tree clean;
2. HEAD recorded;
3. `predictors_frozen.json` sha256 `e0b2ea9d…d210` (tracked in git);
4. design panel sha256 `c7da2c31…` and Stage 1 config sha256 `2f8db710…` (equal to the value
   recorded in the frozen configuration);
5. pytest (353 passed), ruff check, ruff format and mypy (clean cache, 101 files) — all exit 0;
6. frozen horizons, feature names, grids, CV and GBM settings, and the warning thresholds (every
   primary level and horizon) equal the code:
   - **all 156 frozen linear models refit on design reproduce the recorded penalties exactly**;
   - the frozen-path sources are unchanged since the freeze commit `341851a` (audit, features,
     predict, dynamics, toy, outcomes, stage1_runs, stage1_analysis, the Stage 1 config, the
     design panel);
7. no test/shift panel, approval file, held-out run or outcome exists in the working tree or in
   git history;
8. the held-out loader is sealed without the approval file; the seed-tree extension keeps every
   design seed; the design, test and shift seeds are disjoint.

**Execution protocol.** This fixes procedural details that §12, memo §10 and the execution note
left unspecified. It changes no threshold, definition, predictor or panel rule.

- **Test panel (split A).**
  - The Stage 0b generator, **seed stream 1** of `SeedSequence(20261001).spawn(5)`.
  - 24 cells × 32 = 768 structures (the design size); sids `T-…`.
  - The registered clean-gain exclusion applies.
- **Shift panel (split C, memo §10 C).**
  - The same generator on **stream 2**, with `J_G(0) ~ U(0.01, 0.05)`: only the target draw
    changes; the number of draws is unchanged.
  - Training uses **4 × 4 GRPO batches**; `T = 2700`; 768 structures; sids `S-…`.
  - The audit stays 32 × 8. `J_V` pools the audit and the 4 × 4 training batch, as registered.
  - Shift geometry is **audit-only**, because the frozen pooled estimator needs equal group sizes.
    This affects only descriptive L1/L3 on shift. The shift criterion uses L0 and L2.
  - Predictors are the frozen design models; nothing is refit on shift.
- **Seeds** (registered tree):
  - every role and audit branch indexes structures globally: design 0–767, test 768–1535,
    shift 1536–2303;
  - training seeds: roles `prim_ver` / `prim_clean`; audits: spare branches [0] Adam and [1] NG;
  - bootstrap weights and permutations: resample branch children [4] test Adam, [5] test NG,
    [6] shift, [7] test/shift L2N permutation, [8] reserved for post-hoc.
- **Frozen predictors on held-out data.** The frozen pipeline is refit on all kept design runs
  (verified above) and applied unchanged. The warning thresholds are the frozen design `tau_h`.
- **Scoring details.**
  - **RQ1 and EW:** as registered; lower bound = 5th percentile (execution note §5); hierarchical
    bootstrap over the test structures, then seeds, B = 2000.
  - **LOMO:** for each mechanism `m`, fit on the design runs of the other five mechanisms and
    evaluate on the test runs of `m` at `h*`.
    - `m` succeeds iff `L2 − L0` has a lower bound > 0 for the C-index of `Dn`, **and**, when
      evaluable (≥ 10 failing and ≥ 10 non-failing test runs), for the AUROC of failure.
    - GEN-LOMO passes iff ≥ 4 of 6 mechanisms succeed.
  - **Shift:** `L2 − L0` at `h*` has a lower bound > −0.01 for **both** the C-index and the AUROC.
  - **GEN** = GEN-LOMO ∧ shift.
  - The decision label follows Amendment 4.
- **Secondary** (reported, never scored):
  - NG on the test panel (design-fitted NG models);
  - optimizer transfer (NG design → Adam test; Adam design → NG test) at `h*`.
- **Hard pairs:** the frozen pairs are re-run, which is deterministic (same seeds), with added
  descriptive fields (L1 distance, finite-sample alpha/C uncertainty). Criterion (iii) must
  reproduce 0.75.

---

### E004a Stage 1 — held-out execution deviation 1: an infinite loop in panel generation (2026-09-26)

Written and committed **before any held-out or shift structure is written and before any outcome
is computed.**

**What happened.**

- The first held-out launch (HEAD `b9fff65`, after the approval) generated the test panel
  sequentially. After 36 minutes it had written nothing.
- A structures-only timing diagnostic with the same seeds (no trajectories, no outcomes) showed:
  - 767 of 768 test structures generate in ≤ 22 s each;
  - **cell D3-INVERTED, index 24 never finishes.**
- **Cause.** Its common draw has `phi0_1 = −4.46`, so `P(z1 = 1) ≈ 0.011`. D3 accepts the hack only
  when `z1 = 1`, so the hack-driven FPR share is at most about `rho × 0.011 ≤ 0.01`. That is below
  `omega × FPR_target ≥ 0.53 × 0.02 = 0.0106` for **every** target draw. The frozen generator
  redraws targets without limit, so the loop never ends.
  - Over 3000 target draws: 2422 "hack share cannot reach the FPR share" and 578 rejections at
    the `J_G` check.
- **No effect on the design panel.** The largest per-structure target-redraw count there is 24
  (D3).
- **Nothing was written.** The process was killed. Its run directory was empty and has been
  removed. No held-out panel, trajectory or outcome exists.

**Rule** (procedural termination; no panel parameter, range or criterion changes):

- A held-out structure that needs more than **200 target redraws** is recorded as **INFEASIBLE**
  and **excluded without replacement**. Its cell and index are reported.
- 200 is more than 8× the design maximum (24), so no design structure could be affected. The
  design panel still regenerates bit-for-bit (tested).
- Panel generation is parallelized. Every structure keeps its own registered seed, so the result
  is identical to sequential generation.
- The same seeds, approval and frozen configuration are used for the relaunch. The frozen-path
  sources are unchanged.

---
