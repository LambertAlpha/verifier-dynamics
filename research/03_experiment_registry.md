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
