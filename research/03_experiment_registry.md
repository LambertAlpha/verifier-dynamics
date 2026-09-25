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
