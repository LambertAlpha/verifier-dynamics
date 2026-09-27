# 07 — E005a: geometry measurement calibration (theory + registered design, v1, 2026-09-26)

**Status: registered design.** This document is written and committed **before any E005 code
exists**. The registry entry "E005a — pre-registration" freezes §3–§9 of this document. E004a is
closed; nothing here revisits its confirmatory results.

**Status tags.**

| tag | meaning |
| --- | --- |
| [proved] | a derivation in this document |
| [leading order] | a delta-method expansion; the neglected terms are `O(n^-3/2)` or smaller |
| [registered] | a frozen design choice or prediction |
| [open] | left to the collaborator |

---

## 0. Summary

**Question.** Can `A`, `alpha` and especially `C` of the verifier–gold update geometry be
estimated from a finite trusted (gold-labelled) audit without the estimator leaking structure?

**Primary technical issue** (E002a post-hoc; E004 Stage 1).

- The plug-in `C_hat^2` is positive when the true `C = 0`.
- Its floor changes across structures, so a spurious "type signal" appears.

**What the theory below predicts** [leading order]:

| estimator | first-order bias of `C^2` |
| --- | --- |
| plug-in | `+ tr(P_perp Σ_δ)/n` minus two rotation terms: the **orthogonal noise energy** of the effective error contribution `δ = γ_e − alpha γ_G`, divided by the number of independent audit groups `n` |
| cross-fit U-statistic | `− u^T Σ_δ u / n` at `C = 0`: only the **one-dimensional parallel** noise |
| first-order-corrected plug-in | `O(n^-2)` away from `A ≈ 0` |
| group jackknife | `O(n^-2)` |

- The plug-in floor scales with the parameter dimension `d` (a trace over `d − 1` orthogonal
  directions). The U-statistic's bias does not.
- For neural policies (`d ≫ 8`) the plug-in floor is therefore expected to be prohibitive, and the
  calibration panel includes nuisance dimensions to test this scaling directly.

**Estimand choice (and its cost).**

- The primary estimand is the **reward-level** geometry of the exact gradients `∇J_G`, `∇J_V` in
  a declared metric.
- E004's informative L1 geometry was the **update level** (GRPO group normalization, Adam `t = 0`
  metric). That target depends on the group size, and the metric is itself an estimate, so no
  clean bias theory exists for it.
- E005a therefore calibrates the reward level. E005b keeps the legacy update-level plug-in as a
  baseline so the two can be linked. [open: the collaborator may prefer otherwise.]

---

## 1. Setting and estimands

**Audit.** `n` independent groups. Group `k` draws a prompt `x_k ~ w` and `m` responses from the
current policy; each response has a gold label `G` and one verifier call `V` (fresh coin). The
audit size is `N = n m` gold labels.

**Per-group contributions (RLOO)** for a reward `R ∈ {G, V}`:

```
gamma_k^R = (1/m) Σ_i (R_ki − b_ki) s_ki,    b_ki = (Σ_{j≠i} R_kj)/(m − 1),    s = ∇ log π.
```

- `E[gamma_k^R] = g_R = ∇J_R` exactly [proved]: `b_ki` is independent of response `i` given
  `x_k`, and `E[s | x] = 0`.
- Groups are i.i.d. So `g_hat_R = mean_k gamma_k^R` is unbiased with `Cov = Σ_R/n`, where
  `Σ_R = Cov(gamma_k^R)`.
- **Within-group dependence.** RLOO makes the response-level terms dependent inside a group, so
  **all variance and bias statements use the group as the independent unit.**
- **Error reward** `e = V − G`: `gamma_k^e = gamma_k^V − gamma_k^G`, paired on the same rollouts.

**Metric.** `M` is fixed and positive definite; `h = M^{1/2} g`. Let
`γ~ = M^{1/2} γ` and `Σ~ = M^{1/2} Σ M^{1/2}`.

**Targets:**

```
A^2 = ||h_G||^2,   P = <h_e, h_G>,   Q = ||h_e||^2,   alpha = P / A^2,   C^2 = Q − P^2 / A^2.
```

Write `u = h_G / A`, `h_e = alpha A u + c` with `c ⊥ u`, `||c|| = C`.

**Three declared metrics** [registered]:

| metric | definition | estimated from |
| --- | --- | --- |
| **M_I** (primary) | identity | nothing (no metric error). It is computable for any model, including a transformer |
| **M_D** | `diag(F_jj + 0.1 · mean_j F_jj)^-1`, damped diagonal Fisher | verifier-only (unlabeled) rollouts |
| **M_F** (legacy, E002/E004 NG) | `(F + 0.1 · tr(F)/d · I)^-1` | as M_D |

The target always uses the **exact** Fisher `F` at the current `theta`.

**Update level (not an E005a estimand) [proved].**

- GRPO divides each group by its own reward standard deviation. The per-group contribution is a
  ratio of group statistics, so its expectation depends on `m` and does not equal the
  population-normalized gradient.
- It is a legitimate "expected first step" quantity, but it has no clean bias theory. It is kept
  only as the legacy baseline in E005b.

---

## 2. Theory of the estimators

### 2.1 Noise sources

1. **Gradient sampling.** `Σ_G`, `Σ_V`, `Σ_e` and the cross-covariances, all at group level.
   - `g_hat_G` and `g_hat_e` are **correlated**: they share rollouts and `γ_e` contains
     `−γ_G`.
   - Pooling verifier rollouts from outside the audit (the legacy E0) breaks this pairing and
     changes `Σ_e`.
2. **Finite group size `m`.** It changes `Σ` (RLOO variance) but not the mean [proved]. `m ≥ 2`
   is required.
3. **Metric estimation.**
   - **M_I:** none.
   - **M_D / M_F estimated from independent unlabeled rollouts.** Conditional on `M_hat`, every
     statement below holds for the geometry **in `M_hat`**. The difference from the target in
     the exact metric is a metric-estimation error with a Jensen bias of order
     `1/n_unlabeled` (inverse of a noisy matrix). It is not removable exactly and is reported
     separately.
   - **Metric estimated from the same audit (legacy E0).** `M_hat` is correlated with the
     gradients, which adds a further `O(1/n)` bias of undetermined sign.
4. **Ratios.**
   - `alpha` and `C^2` contain `1/A^2`, and `A` contains a square root. **No exactly unbiased
     estimator of `C^2`, `alpha` or `A` exists in general**: a polynomial of finite-sample means
     cannot have expectation `P^2/A^2`.
   - Only `A^2`, `P` and `Q` (quadratic forms) admit exact unbiased estimators, and only when
     `M` is fixed.

### 2.2 E0 — legacy plug-in (E002 G1 / E004 reward level)

- **Definition:** `g_hat_G` = RLOO on the audit; `g_hat_V` = RLOO pooled over the audit and the
  unlabeled rollouts; `M_hat` from all rollouts (same batch); `(A, alpha, C)` by the decompose
  rule.
- **Leading-order bias of `C^2`** (fixed `M`, paired `V`, for clarity) [leading order]. Write
  `h_hat_G = A u + ε_G`, `h_hat_e = alpha A u + c + ε_e`, and `δ = ε_e − alpha ε_G`, so that
  `h_hat_e = alpha h_hat_G + c + δ`. The component `alpha h_hat_G` is removed exactly by the
  projection, hence

  ```
  C_hat^2 = ||c + δ||^2 − ((c + δ)·h_hat_G)^2 / ||h_hat_G||^2
  E[C_hat^2] = C^2 + (1/n) [ tr(P_perp Σ~_δ) − c^T Σ~_G c / A^2 − 2 c^T Σ~_Gδ u / A ] + O(n^-3/2),
  ```

  with `Σ_δ = Cov(γ_e − alpha γ_G) = Σ_e − alpha(Σ_eG + Σ_Ge) + alpha^2 Σ_G`, and
  `P_perp = I − u u^T`.
- **At `C = 0`:**
  - `E[C_hat^2] ≈ tr(P_perp Σ~_δ)/n > 0`. **This is the noise floor.** It depends on the
    covariance of `(V − alpha G − G)·score`, i.e. on the FPR/FNR, the policy entropy and the
    response structure. It therefore **differs across structures with identical `C`**, which
    explains the E002a/E004 findings.
  - It is a trace over `d − 1` directions, so it **grows linearly with nuisance dimensions**.
- **Near `A ≈ 0`.** The `1/A^2` rotation term dominates and the expansion fails. `alpha_hat` is
  shrunk toward 0 because `E||h_hat_G||^2 = A^2 + tr(Σ~_G)/n`.

### 2.3 E1 — cross-fit (group U-statistic)

- **Definition** (all distinct pairs of independent groups; `M` from independent unlabeled
  rollouts, or `M_I`):

  ```
  Gram_ab = (1/(n(n−1))) Σ_{k≠l} γ~_k^a · γ~_l^b
          = h_bar_a · h_bar_b − tr(S_ab)/n     (S = sample cross-covariance, ddof 1)   [proved]
  A_U^2 = Gram_GG,   P_U = Gram_eG,   Q_U = Gram_ee,
  alpha_U = P_U / A_U^2,   C_U^2 = Q_U − P_U^2 / A_U^2.
  ```

- `Gram` entries are **exactly unbiased given `M`** [proved]; this is E002's surviving claim.
  `C_U^2` is **not** (E002's failed claim; not repeated).
- **Leading-order bias** [leading order]:

  ```
  E[P_U^2/A_U^2] = P^2/A^2 + Var(P_U)/A^2 − 2 P Cov(P_U, A_U^2)/A^4 + P^2 Var(A_U^2)/A^6 + ...
  ```

  with `Var(P_U) ≈ (1/n) Var(γ_e·h_G + h_e·γ_G)`,
  `Cov(P_U, A_U^2) ≈ (2/n) Cov(γ_e·h_G + h_e·γ_G, γ_G·h_G)` and
  `Var(A_U^2) ≈ (4/n) Var(γ_G·h_G)`.
- **At `C = 0`** the terms collapse to `E[C_U^2] ≈ −u^T Σ~_δ u / n` [proved at leading order].
  This is a **negative, one-dimensional** bias. It is still structure-dependent, but it does not
  scale with `d`.
- **Instability.** `A_U^2` can be ≤ 0 when `A^2 ≲ sqrt(Var(A_U^2))`. The rule is in §5.

### 2.4 E2 — first-order noise-floor-corrected plug-in

- **Definition:** `C_E2^2 = C_hat_plugin^2 − (1/n) [ tr(P_hat_perp Σ_hat_δ) − c_hat^T Σ_hat_G c_hat / A_hat^2 − 2 c_hat^T Σ_hat_Gδ u_hat / A_hat ]`.
  - Every `Σ_hat` is the ddof-1 sample covariance of the whitened per-group contributions of the
    audit; `δ_k = γ~_k^e − alpha_hat γ~_k^G`.
  - The plug-in is computed on the audit only (paired); `M` as in E1.
- **Consequences:**
  - With the correction applied to `A^2` and `P` as well, `A_E2^2 = A_U^2` and
    `alpha_E2 = alpha_U` exactly [proved: `||h_bar||^2 − tr(S)/n` is the U-statistic].
    **E2 differs from E1 only in `C^2`.**
  - **Bias:** `O(n^-2)` where the expansion holds (`A^2 ≫ sqrt(tr Σ~_G / n)`). Near `A ≈ 0` the
    correction itself is unstable.
  - The raw signed value is kept; clipping at 0 is used for display only.

### 2.5 E3 — group jackknife

- **Definition:** delete-one-group jackknife of the plug-in (audit only, `M` as in E1),
  `θ_J = n θ_hat − (n − 1) mean_k θ_hat_(−k)`, for `A^2`, `alpha` and `C^2`.
- **Bias:** it removes every `O(1/n)` bias term of a smooth statistic without deriving it;
  `O(n^-2)` remains.
- **Costs:**
  - variance inflation near `A ≈ 0`, where the statistic is not smooth;
  - `n` recomputations. That is cheap given the stored per-group gradients: no extra backward
    passes, only memory `n × d`.

### 2.6 Uncertainty (all estimators) [registered]

- The standard error is the delete-one-group jackknife SE of the estimator itself (for E0,
  groups of the audit and of the pooled rollouts).
- **Interval:** `estimate ± 1.96 SE`.
- **Null test for `C > 0`:** reject iff `C^2_est − 1.645 SE > 0`.

### 2.7 Registered theory predictions (tested on the DESIGN calibration panel)

- **Oracle `Σ`** is computed per calibration point by Monte Carlo with `10^6` groups (reported
  MC error).
- **TP1 (plug-in floor).** At `C = 0`, M_I, `N >= 128`: the bias of the paired, audit-only
  plug-in (the common input of E2 and E3) divided by `tr(P_perp Σ~_δ)/n` lies in [0.8, 1.25] for
  ≥ 90% of null points.
- **TP2 (U-statistic).** At `C = 0`, M_I, `N >= 128`: observed E1 bias /
  `(−u^T Σ~_δ u / n)` ∈ [0.7, 1.4] for ≥ 80% of null points whose predicted bias exceeds 3 MC
  SE.
- **TP3 (dimension scaling).** The plug-in null bias at `d_extra = 56` exceeds `d_extra = 0` by
  the predicted trace increment (ratio ∈ [0.8, 1.25]); the E1 null bias does not change beyond
  3 MC SE.
- **TP4 (second order).** E2 and E3: `|bias(C^2)| ≤ 3 MC SE` at `N >= 256` for ≥ 90% of points
  with `A^2 > 10 · sqrt(tr(Σ~_G)/n)`.

---

## 3. Calibration panel [registered]

**Seeds:** `SeedSequence(20261101).spawn(4)` = [design panel, test panel, design Monte Carlo,
test Monte Carlo].

**Base structures** (per split).

- The U-toy (E004), with the Stage 0b generator (`panel0b.build`, the `max_target_redraws = 200`
  termination rule).
- **One structure per Stage 0b cell:** 24 bases covering R (random attenuation), X (deletion),
  YA/YB (exploit feature × aligned/inverted), B (benign amplification) and D (preference
  inversion).
- The E004 outcome labels are never used.

**Policy variants** (entropy, `A` scale):

- `theta_0`;
- `theta_hot = (0.5 u, 0.5 h, phi)`;
- `theta_cold = (2 u, 2 h, phi)`;
- `theta_lowA = (u − 6, h, phi)`, where the solve probability is about `e^-6` and `A ≈ 0`.

**Verifier dose** (exact `C` control).

- Per base, draw an affine null `a ~ U(0.5, 0.95)`, `b ~ U(0, 0.3(1 − a))`. Mix
  `EV_rho = (1 − rho)(a G + b) + rho EV_base`.
- **`C` is exactly linear in `rho`:** `C(rho) = rho · C_base_perp` [proved], because
  `g_V(rho) = (1 − rho) a g_G + rho g_V,base` and the first term is parallel to `g_G`.
- **At `rho = 0`, `C = 0` in every metric** (an affine verifier), with `alpha = a − 1`. The noise
  structure (Bernoulli `V`, response rows, features) still varies with the base, so these are
  the **matched nulls**.
- **Dose targets:**
  - (M_I) `C* ∈ {0, 0.05, 0.2, 0.6} × C_ref` and the natural dose `rho = 1`;
  - `C_ref` = the median `C_base_perp` (M_I) over the design bases at `theta_0`. The same `C_ref`
    is used for the test split;
  - `rho = C*/C_base_perp`; points with `rho > 1` are dropped and counted.
- **Matched set** = the points that share (dose target, policy variant, `d_extra`). The oracle
  `C` is identical (M_I) across different mechanisms and response structures.

**Nuisance dimensions.**

- `d_extra ∈ {0, 56}`: independent `Bern(0.5)` response features that do not enter any reward.
  Their scores add gradient noise with zero mean.
- The oracle geometry is unchanged in M_I and M_D. In M_F the Fisher is block-diagonal with
  `0.25 I` [proved].
- Total `d ∈ {8, 64}`.

**Composition.** At most `24 × 4 × 5 × 2 = 960` points per split (fewer after dropping).
Reported by mechanism, construction, variant, dose, `alpha` sign and `A` quantile. The panel
files (JSON) are frozen with sha256 before any estimator is run.

---

## 4. Budgets and costs [registered]

- **Audit sizes:** `N ∈ {32, 64, 128, 256, 512, 1024}` gold-labelled rollouts, with group size
  `m ∈ {4, 8}` (`n = N/m` groups).
- **Unlabeled rollouts:** `N_u = N`, verifier-scored, not gold-labelled. They are used for the
  metric (E1–E3 under M_D/M_F) and for pooling (E0).
- **Costs recorded per estimator × metric:**

  | estimator / metric | `B_roll` | `B_gold` | `B_bwd` |
  | --- | --- | --- | --- |
  | E0 (any metric) | `2N` | `N` | `2N` |
  | E1–E3 under M_I | `N` | `N` | `N` |
  | E1–E3 under M_D / M_F | `2N` | `N` | `2N` |

  `B_bwd` counts per-rollout score gradients. The jackknife and U-statistic reuse the stored
  per-group gradients.
- **Monte Carlo:** `R = 100` replications per (point, `N`, `m`), from the split's MC stream.

---

## 5. Frozen handling rules [registered]

- **`A ≈ 0`:**
  - if an estimator's `A^2` estimate is ≤ 0, then `alpha` is undefined (NaN, never imputed) and
    `C^2 := Q_est` (all error energy counts as orthogonal; the E002/E004 convention);
  - undefined rates are reported.
- **Damping:** 0.1 as in §1; there is no other tuning parameter.
- **Estimator outputs:** raw signed `A^2` and `C^2`, and `alpha`. `A = sqrt(max(A^2, 0))` and
  `C = sqrt(max(C^2, 0))` are for display only.
- **`m`:** at least 2. Groups are never split across folds.

---

## 6. Metrics (per estimator × metric × `N` × `m`) [registered]

- **Per point, over replications, for `A^2`, `alpha` and `C^2`:**
  - bias, |bias|, variance, RMSE;
  - 95% interval coverage (§2.6);
  - undefined rate.
- **Across points:**
  - Spearman rank correlation of the estimate with the oracle (per replication across points,
    then averaged);
  - calibration slope (per-point mean estimate regressed on the oracle).
- **Nulls (`C = 0`):**
  - distribution of `C^2_est` and its 95th percentile (the noise floor);
  - false-positive rate of the §2.6 test;
  - dependence of the floor on structure: its SD across the matched nulls, and by mechanism.
- **Matched true `C` (primary validity diagnostic):**
  - the **between-structure shift**: for each matched set, the SD across structures of the
    per-point mean `C^2_est` (the oracle `C` is identical);
  - summary statistic `SDB` = the RMS over matched sets of that SD divided by the median
    within-point SD of `C^2_est`.
- **Detection power** (for the practical-budget rule, §8): the rate of the §2.6 test rejecting at
  `C* = 0.2 C_ref` (small) and `0.05 C_ref` (very small).

---

## 7. Estimator selection (DESIGN only) [registered]

**Setting:** M_I, `m = 8`, `N_ref = 256`, both `d_extra` levels pooled.

**Candidates:** E0, E1, E2, E3 (frozen list).

Criteria are applied in order. At each step, the candidates within the stated tolerance of the
best are kept:

1. **Structure-dependent bias:** the lowest `SDB`; keep those within 20% of the minimum.
2. **Null calibration:** the mean over null points of `|FPR − 0.05|`; keep those within 0.02 of
   the minimum.
3. **Ranking:** the mean Spearman over points with `C > 0`; keep those within 0.02 of the
   maximum.
4. **Efficiency:** the median over points of `RMSE(C^2)/(C_ref^2)`; keep those within 10% of
   the minimum.
5. **Stability near `A ≈ 0`:** the lowest rate of undefined or extreme values
   (`|C^2_est| > 10 C_ref^2`) on `theta_lowA` points.

Ties go to the simpler estimator, in the order E0 < E1 < E2 < E3.

**After selection:**

- the implementation, hyperparameters, preprocessing, damping, `A ≈ 0` rule and interval method
  are frozen and committed (an estimator-configuration JSON with sha256);
- **then** the test split is run exactly once.

---

## 8. Held-out test, practical budget and stopping rule [registered]

**Test evaluation.** The same metrics for the selected estimator (all candidates are reported,
but the selection is fixed).

**"Structure-dependent `C` noise solved" at budget `N`** (test; M_I; `m = 8`) iff all of:

- `SDB ≤ 0.5`: the between-structure shift is at most half the sampling SD;
- the null false-positive rate is in `[0.02, 0.10]` for ≥ 90% of null points;
- the Spearman over `C > 0` points is ≥ 0.8.

**Minimum practical audit budget.** The smallest `N` at which the selected estimator is "solved"
**and** reaches detection power ≥ 0.8 at the small dose.

**Negative-result condition.**

- If no `N ≤ 1024` qualifies, the result is recorded as: **"C is mechanistically meaningful but
  not practically measurable at the intended scale."**
- The audit size is not increased beyond 1024 to rescue it.

**Recommendation rule:**

- **PROCEED TO E005b** iff a minimum practical budget `≤ 1024` exists on the test split under
  M_I;
- otherwise **REVISE MEASUREMENT THEORY BEFORE NEURAL EXPERIMENTS**.
- M_D and M_F results are reported; they inform E005b's metric choice but do not gate the
  recommendation.

---

## 9. Integrity [registered]

- New namespace `src/vdyn/e005/`. E004 files are untouched.
- Strict TDD; checks fail closed; no hidden exit codes.
- **Every run `meta.json` records:**
  - the commit and dirty status;
  - the root seed;
  - the config sha256, calibration-panel sha256 and (from selection on) estimator-config sha256;
  - Python / numpy / scipy / torch / scikit-learn versions.
- The test split runs only after the estimator configuration is committed, via an approval file
  naming its sha256.

---

## 10. Post-run errata and theory lessons (appended 2026-09-27; §1–§9 unchanged)

See the registry entry "E005a — results record" for the evidence.

- **Erratum (dimensional).** The validity condition written as `A^2 ≫ sqrt(tr Σ~_G / n)`
  (§2.4, TP4 in §2.7) should read `A^2 ≫ tr(Σ~_G)/n`, equivalently `A ≫ sqrt(tr Σ~_G / n)`.
  - As registered, TP4 selects no point (n = 0).
  - With the corrected condition (post-hoc), E2 and E3 are within 3 MCSE for > 99% of eligible
    points.
- **Lesson 1 — the null test at the boundary** [derived post-hoc; matches the calibration].
  - At `C = 0` the first-order influence function of `C^2` vanishes, and every bias-corrected
    estimator behaves like a degenerate second-order U-statistic.
  - The delete-one-group jackknife variance then has expectation ≈ 2× the true variance, so the
    Wald test of §2.6 is conservative; observed FPR ≈ 0.001–0.01.
  - A null test must use the degenerate null distribution: e.g. SE / √2 at the boundary, a
    weighted-χ² approximation from `Σ_hat_δ`, or a group sign-flip / permutation calibration.
- **Lesson 2 — variance, not bias, binds.**
  - After bias correction, the `C^2` noise floor falls as `1/n` but grows with the number of
    gradient dimensions (≈ 6.4× from `d = 8` to `d = 64`).
  - Measurement for high-dimensional policies must be done in a declared low-dimensional
    subspace, or with variance reduction. A larger audit alone is not the lever.
