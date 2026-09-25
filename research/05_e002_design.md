# 05 — E002 design: finite-sample estimability of (A, alpha, C)

**Status: DRAFT for review (2026-09-25). Not registered, not implemented, not run.** Once approved,
the numeric predictions (§6) will be computed from exact formulas and committed as a registry
pre-run entry **before** any E002 code runs.

## 1. Question

In a toy where `(A, alpha, C)` are known exactly, can they be estimated from `N` on-policy rollouts
well enough to be practically useful?

**Revision (2026-09-25, collaborator decision).** The primary evaluation is **matched R vs Y**:

- bias and variance of the estimates;
- `P(C_hat_Y > C_hat_R)`;
- power at a fixed 5% false-alarm rate;
- minimum detectable `C`;
- behaviour as `A -> 0`.

X is kept only as a separate deletion / zero-signal diagnostic. The three-way R/X/Y classification
is **not** a headline metric. At `A = 0` the estimators follow the v0.3 convention
(`alpha = NaN`, `alpha_defined = False`, `C = ||h_e||`).

## 2. Model and conditions

- **Model.** The Phase 1A two-Bernoulli toy. Exact values come from the closed forms (already
  tested). Rerunning on a Phase 1B model is a later, separate experiment.
- **No training.** Each cell fixes a policy state `theta` and draws `N` i.i.d. rollouts from
  `pi_theta`. Gold `G` and verifier `V` are evaluated on the same rollouts (paired). For R, flips
  are drawn fresh per rollout.
- **Policy states.**
  - *S1 typical:* the E001 initial states `(0.30, 0.01)`, `(0.10, 0.10)`, `(0.01, 0.30)`, plus
    `(0.50, 0.20)`.
  - *S2 near `C = 0`:* `q = 0.5`, `s ∈ {1e-3, 3e-3, 1e-2, 3e-2, 0.1}`; and `s = 0.2`,
    `q ∈ {0.9, 0.99, 0.999}`.
  - *S3 near `A -> 0`:* `s = 0.2`, `q ∈ {1e-3, 1e-2, 0.99, 0.999}`.
- **Verifiers at each state.**
  - **Y** (`corr OR z`): false positives only.
  - **R_p** with `p = (1-q)s`: fresh symmetric flips, **matched to Y's on-policy error rate**.
  - **X**: constant accept. Its error rate is `1 - q` and cannot be matched in the single-prompt
    toy. Reported separately; a matched X needs the multi-prompt toy.
- **Sample sizes.** `N ∈ {8, 16, 32, 64, 128, 256, 512, 1024, 4096}`.
- **Replications.** 4000 per cell, so the Monte Carlo SE of a probability is ≤ 0.008. Seeds come
  from `np.random.SeedSequence(<registered root>).spawn(...)`, one child per cell, plus a separate
  calibration stream (§4).

## 3. Estimators (factors)

- **Gradient estimator.**
  - *E1:* REINFORCE without a baseline, `gamma_i^R = R_i · score_i`.
  - *E2:* leave-one-out baseline (RLOO; GRPO-like without std normalization).

  The error gradient is estimated directly from `e_i = V_i - G_i` on the same rollouts.
- **Metric.**
  - *M1:* Euclidean (exact).
  - *M2:* Fisher-oracle (exact `F^{-1}`; isolates gradient noise).
  - *M3:* estimated Fisher `F_hat = mean(score score^T) + lambda I` with a pre-specified damping
    `lambda = 1e-3 · tr(F_hat)/d`.
- **Point estimators.**
  - *P (plug-in):* `decompose(g_hat_G, g_hat_V, M)`.
  - *U (bias-corrected Gram):* unbiased U-statistics for the Gram entries
    `<g_a, g_b>_M ≈ (1/(N(N-1))) Σ_{i≠j} gamma_i^a · M gamma_j^b`. This needs independent terms,
    so it is used with E1 and M1/M2 only. Then `A_hat^2 = G_GG`, `alpha_hat = G_eG / G_GG`, and
    `C_hat^2 = G_ee - G_eG^2 / G_GG`, which can be negative; it is reported raw and truncated at 0.
  - *Gram form (no division by `A^2`):* `P = <g_e, g_G> = alpha A^2` and
    `D = A^2 ||g_e||^2 - P^2 = A^2 C^2`. These stay stable as `A -> 0`.
- **Uncertainty.** Nonparametric bootstrap over rollouts, `B = 500`, with percentile 90% and 95%
  intervals for `A`, `alpha`, `C`, `C^2` and `D`.

## 4. Metrics

Each metric is computed per cell (state × verifier × `N` × estimator × metric).

1. Bias, SD and RMSE of `A_hat`, `alpha_hat`, `C_hat` (and of `C_hat^2`, `P`, `D`). Also the rate
   of undefined or invalid `alpha_hat` (`A_hat = 0` or `G_GG <= 0`).
2. Bootstrap CI coverage at nominal 90% and 95%.
3. **Detection of `C > 0`.**
   - `P(C_hat_Y > C_hat_R)`, using independent draws at the same state and `N` (an AUC).
   - Power `P(C_hat_Y > c95)`, where `c95` is the 95th percentile of `C_hat` under R at the same
     state and `N`, estimated from the calibration stream.
   - The minimum detectable `C` at 80% power as a function of `N` (from the S2 sweep).
4. *(Secondary diagnostic only, not a headline.)* **Zero-signal / deletion diagnostic**, including X, with a pre-specified rule:
   - (i) **X** if all `N` verifier rewards are equal (with E2 this is exactly `g_hat_V = 0`, the
     GRPO zero-advantage case);
   - (ii) otherwise **Y** if `C_hat > c95`;
   - (iii) otherwise **R**.

   Reported as balanced accuracy and a confusion matrix.
5. **Minimum `N`** for each of:
   - `|bias| < 10%` and `CV < 20%` for each quantity;
   - power ≥ 0.8 for `C > 0`;
   - (secondary) zero-signal diagnostic accuracy ≥ 0.9.
6. **Near `A -> 0`:** the `alpha_hat` breakdown rate, and `D`/`P` behaviour vs `alpha_hat`.

## 5. Why the edge cases matter (known in advance)

- **Constant rewards.** With E2, if all `G_i` are equal then `g_hat_G = 0` exactly, so
  `A_hat = 0`. This happens with probability `q^N + (1-q)^N`, e.g. 0.53 at `q = 0.01`, `N = 64`.
  Similarly, Y or R rollouts with all `V_i` equal get classified as X. These rates are exactly
  computable and realistic: they are GRPO's zero-variance groups.
- **Estimated Fisher.** If all `z_i = 0` then `F_hat_vv = s^2`, which is tiny. `F_hat^{-1}` then
  explodes in the exploit direction and inflates `C_hat` under M3. This is a predicted failure mode
  whose size depends on `lambda`.
- **Boundary at `C = 0`.** `C_hat >= 0` always (plug-in), so plug-in estimates are biased upward
  near `C = 0` and percentile bootstrap intervals are expected to undercover there.

## 6. Predictions to compute and register before running

All from the exact per-sample covariance `Sigma_ab = Cov(gamma^a, gamma^b)`, obtained by
enumeration in the toy:

- **Leading-order bias of the plug-in estimators.**
  - `E[A_hat^2] = A^2 + tr(M Sigma_GG)/N`.
  - Under R (`C = 0`): `E[C_hat^2] ≈ tr(M_⊥ Sigma_ee)/N`, where `M_⊥` is `M` restricted to the
    `M`-orthogonal complement of `g_G`. So `C_hat_R ≈ sigma_⊥ / sqrt(N)`: a registered noise-floor
    curve for each state.
- **Detection sample size.** `N* ≈ (z_0.95 + z_0.8)^2 sigma_⊥^2 / C_Y^2`, registered as a number
  per state and metric. Rough order of magnitude for Y at `(0.30, 0.01)` with the Fisher-oracle
  metric: `N* ~ 10^2–10^3` (to be computed exactly before registration).
- **Exact probabilities.** `P(all G_i equal)` and `P(all V_i equal)` per state and `N`, which fix
  the constant-reward rates.
- **Qualitative predictions.**
  - (i) U-estimators remove the `1/N` bias of squared quantities but have larger variance at
    small `N`.
  - (ii) Gram-form `D` stays stable where `alpha_hat` breaks down (`A -> 0`).
  - (iii) M3 is worse than M2 at `N <= 32`.
  - (iv) Bootstrap undercovers for `C` near 0 and at `N <= 16`.
- **Falsification.** A registered `N*` that is off by more than a factor of 2 at any S1 state
  rejects the leading-order approximation at that `N`. A U-estimator whose empirical bias exceeds
  3 Monte Carlo SE indicates an implementation error.

## 7. Scope and cost

numpy only, vectorized over replications; minutes of CPU. Outputs:

- RMSE-vs-`N` log-log plots;
- power curves and AUC-vs-`N`;
- classification accuracy vs `N`;
- coverage tables.

## 8. Choices for the reviewer

1. GRPO std-normalized advantages: include now, or defer to Phase 2? We recommend deferring: they
   change the expected gradient (Dr. GRPO bias), which is a separate question.
2. Matched X needs the multi-prompt model. Run E002 on the single-prompt toy now (recommended,
   since estimator properties are local), then repeat on the Phase 1B model.
3. The damping `lambda` for M3.
4. Whether the function-space (k-step) probe estimator enters E002 or a separate E00x. We
   recommend a separate experiment, because it needs dynamics, not just a fixed state.
