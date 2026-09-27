# 09 — E005a-R: low-dimensional / functional geometry (theory + registered design, v1, 2026-09-27)

**Status: registered design.**

- Written and committed **before any E005a-R method code exists**. The only code run beforehand is
  the step-0 anchor script, which applies the frozen E005a oracle to frozen E004a structures (§3).
- The registry entry "E005a-R — pre-registration" freezes §3–§11.
- E004a and E005a are closed. Nothing here modifies, reruns or reinterprets them.
- **This is the final measurement reformulation.** If it fails, geometry becomes an
  oracle/mechanistic tool only; there is no E005a-R2.

**Status tags:**

| tag | meaning |
| --- | --- |
| [proved] | an exact derivation in this document |
| [leading order] | a delta-method / Hoeffding expansion; neglected terms are smaller by a factor `n^-1/2` |
| [registered] | a frozen design choice, prediction or threshold |
| [open] | left to the collaborator |

---

## 0. Inherited constraints (read from the record before writing)

**E002** (registry "E002 — design-split pilot record", "E002a — formal results"):

- Unbiased Gram entries do **not** give an unbiased `C^2`: `C^2 = Q − P^2/A^2` is a ratio.
  The failed claim stays failed and is not repeated here.
- Geometry (G1) lost to the best budget-matched **active** probe at every primary cell.
  **No claim that geometry beats a budget-matched active probe is made or tested here.**

**E004a** (registry "E004a Stage 1 — FINAL held-out results"):

- `t = 0` update geometry carried most of the predictive signal. `C_0/A_0` was the best single
  diagnostic (AUROC 0.712).
- Finite-sample `C` was dominated by a structure-dependent noise floor.
- The E004a geometry was at the **update level** (GRPO-normalized, Adam `t = 0` metric).

**E005a** (registry "E005a — results record"; `07_e005a_measurement.md` §10):

1. **Negative result:** full-space `C` is not practically measurable at `N ≤ 1024`.
   Recorded as "C is mechanistically meaningful but not practically measurable at the intended
   scale".
2. **E3 fixed the bias, not the variance.** E3 (delete-one-group jackknife of the paired
   plug-in) removed the structure-dependent bias (SDB 0.11–0.13). The binding constraint is
   variance; the null floor grew 6.4× from `d = 8` to `d = 64`.
3. **The null test was wrong at the boundary.** The jackknife-Wald test is conservative there,
   because at `C = 0` the estimator is a degenerate second-order U-statistic and the jackknife
   variance is ≈ 2× too large.
4. **TP4 erratum:** the validity condition is `A^2 ≫ tr(Σ~_G)/n`.
5. **Scope:** only the reward level was calibrated.

**Consequences for this design:**

- E3 is reused unchanged as the point estimator. **No new full-space `C` estimator is
  introduced.**
- The one inherited methodological change is the null test. The E005a §10 lesson (a group
  sign-flip calibration) is applied uniformly to every representation, including the full-space
  baseline, and is registered here before any data.
- The E005a jackknife-Wald test is also reported, for continuity.

---

## 1. Question and estimands

**Question.** Is the full-space measurement failure caused mainly by nuisance dimension? And can
a **pre-declared, behavior-relevant, low-dimensional** representation retain the verifier signal
while substantially reducing the estimation noise?

Dimension reduction alone is not useful geometry: a projection can remove signal as well as
noise. **Signal retention and noise reduction are measured separately.**

**Level [registered].** Every target is **reward-level** geometry:

- the exact policy gradients `g_G = ∇J_G` and `g_V = ∇J_V` at the evaluation policy;
- group RLOO contributions as the independent unit (E005a §1);
- a fixed metric.

No GRPO-normalized or optimizer-level quantity is estimated (§2.8).

**Three objects** (`h = M^1/2 g`, `u = h_G/A`, `c = P_perp h_e`):

| object | definition | role |
| --- | --- | --- |
| **A. full** `C_full` | the E005a target in the full parameter space, identity metric | historical target; the R0 baseline |
| **B. behavior** `C_beh` | the same geometry of `P_S* g` in the identity metric, with `S*` the known behavior-relevant subspace | **primary oracle target** of E005a-R; `S*` is never available to learned methods |
| **C. functional** `C_f` | the geometry of the virtual behavior changes `δf_R = J_f M g_R` in an output metric `W` (§2.6) | target of R4 |

**Functional analogues** [registered]:

- `A_f^2 = ||δf_G||_W^2`;
- `P_f = <δf_e, δf_G>_W`, `Q_f = ||δf_e||_W^2`;
- `alpha_f = P_f / A_f^2`;
- `C_f^2 = Q_f − P_f^2 / A_f^2`;

with `δf_e = δf_V − δf_G` and `M = I` (a plain virtual gradient step).

---

## 2. Theory

### 2.1 Full-space bias and variance [leading order]

**Setting.** `n` i.i.d. groups with whitened contributions `x_k^G`, `x_k^e` (means `h_G`, `h_e`).

- `alpha = <h_e, h_G>/A^2` and `c = h_e − alpha h_G = P_perp h_e`.
- The effective error contribution is `δ_k = x_k^e − alpha x_k^G`, with `E δ_k = c` and
  `Σ_δ = Cov δ_k`.
- Write `Σ_⊥ = P_perp Σ_δ P_perp`.

**Plug-in** (E005a §2.2):

```
C_hat^2 = ||P_hat_perp (c + ε)||^2,   ε = mean_k δ_k − c
E C_hat^2 = C^2 + tr(Σ_⊥)/n − (rotation terms of order 1/n) + O(n^-3/2).
```

- At `C = 0` the bias is `tr(Σ_⊥)/n`, which is the **noise energy** in the `d − 1` orthogonal
  directions.
- With `tr Σ_⊥ = D σ̄^2`, the plug-in floor is `D σ̄^2/n`. So `C_noise ∝ sqrt(D/N)`. This is
  the working hypothesis of the brief, and **it holds for the uncorrected plug-in**.

**Bias-corrected estimators** (E2, E3; U-statistic form). The leading `1/n` terms are removed.
The Hoeffding decomposition of the remaining fluctuation gives

```
C_corr^2 − C^2 ≈ (2/n) Σ_k c^T(δ_k − c)  +  U_n,
U_n = (1/(n(n−1))) Σ_{k≠l} (P_perp(δ_k − c))^T (P_perp(δ_l − c))
Var ≈ 4 c^T Σ_δ c / n  +  2 tr(Σ_⊥^2) / (n(n−1)).
```

- The second term is **exact for the degenerate U-statistic with no Gaussian assumption**
  [proved]: for independent `a, b` with covariance `Σ_⊥`, `E[(a^T b)^2] = tr(Σ_⊥^2)`.
- The remaining one-dimensional terms, from estimating `u` and `alpha`, are neglected. They are
  `O(n^-2)` and do not scale with `d`.

**At `C = 0`:** `SD_0 = sqrt(2 tr Σ_⊥^2 / (n(n−1)))`.

- Let `d_eff = (tr Σ_⊥)^2 / tr(Σ_⊥^2)` (the participation ratio). Then
  `sqrt(tr Σ_⊥^2) = tr(Σ_⊥)/sqrt(d_eff)`.
- For isotropic `Σ_⊥ = σ^2 I_D`:

```
plug-in floor         ∝ D σ^2 / n            ->  C_noise ∝ D^(1/2)  / sqrt(N)
bias-corrected floor  ∝ sqrt(2D) σ^2 / n     ->  C_noise ∝ D^(1/4) / sqrt(N)
```

**The dimension barrier of a bias-corrected estimator is therefore the square root of the
plug-in's.** E005a's 6.4× growth of the E3 floor (`d` 8 → 64) is consistent with the second
line if the nuisance features' per-dimension variance was ≈ 3.4× that of the behavior features.
This is an inference, not a test.

**Limitations of the derivation:**

- `n ≥ 20`-ish (the U-statistic normal approximation is poor for small `n` and small `d_eff`);
- `A^2 ≫ tr(Σ~_G)/n` (the E005a TP4 condition; otherwise the rotation terms dominate);
- a fixed metric;
- i.i.d. groups;
- the null distribution of `U_n` is a centered weighted `χ^2`, which is right-skewed when
  `d_eff` is small, so normal quantiles understate the tail.

### 2.2 Sample-complexity hypothesis [registered as a hypothesis, not a theorem]

**Detectability.** Define the per-group detectability of the orthogonal error

```
tau = C^2 / sqrt(2 tr(Σ_⊥^2))          (dimensionless; one group)
```

With `n` groups, `C^2 / SD_0 ≈ n tau`.

**Required groups.** With `s_1/s_0 = sqrt(1 + 2 n c^T Σ_δ c / tr(Σ_⊥^2))`, detection at level
`a` with power `1 − b` needs

```
n tau ≥ z_a + z_b · s_1/s_0.
```

- **Null-dominated regime** (weak signals): `n_req ≈ (z_a + z_b)/tau`, i.e.

  ```
  N_required ≈ m (z_a + z_b) sqrt(2 tr Σ_⊥^2) / C^2  ∝  σ̄^2 D / (sqrt(d_eff) C^2).
  ```

  Isotropic: `∝ sqrt(d) σ^2 / C^2`. The plug-in instead needs `∝ d σ^2 / C^2`, because its bias
  must first be exceeded.
- **Signal-dominated regime** (strong signals):

  ```
  n_req ≈ 4 z^2 (c^T Σ_δ c)/C^4,
  ```

  which is **dimension-free**.

**Hypothesis H-dim.** Nuisance dimensions hurt full-space `C` through `tr(Σ_⊥^2)`. A
representation that removes the nuisance block while keeping `c` changes detectability by the
factor

```
sqrt(tr Σ_⊥,full^2 / tr Σ_⊥,S*^2).
```

That factor is large only in the null-dominated (weak/medium-signal) regime.

### 2.3 When does projection improve SNR? [proved for fixed projections]

**Setting.** `Π` is an orthogonal projector of rank `k`, chosen **independently of the
estimation data**.

**Lemma (linear maps preserve alignment)** [proved]. For any linear map `L`,
`L h_e = alpha L h_G + L c`. Hence:

- `C_L = 0` whenever `C = 0`. No representation can create orthogonal error from an aligned
  verifier at the oracle level.
- `C_Π^2 = ||P_perp^(Πh_G) Π c||^2 ≤ ||Πc||^2 ≤ C^2`. A projection can only **lose** signal.

**Definitions:**

```
rho_signal = C_Π^2 / C^2                                    ∈ [0, 1]
rho_noise  = sqrt( tr(Σ_⊥,Π^2) / tr(Σ_⊥^2) )                (floor of a bias-corrected estimator)
rho_noise1 = tr(Σ_⊥,Π) / tr(Σ_⊥)                            (energy; plug-in floor)
```

where `Σ_⊥,Π` is `Π Σ_δ Π` projected orthogonally to `Π h_G`.

- **Bias-corrected, null-dominated:** `tau_Π / tau = rho_signal / rho_noise`.
  **Projection improves detectability iff `rho_signal > rho_noise`.**
- **Plug-in** (bias-limited): it improves iff `rho_signal > rho_noise1`.
- **Signal-dominated regime:** projection helps only by removing noise along `c`; if `span(c)`
  is kept, the linear term is unchanged.

**Random projection** (`Π` Haar-distributed on the Grassmannian; `d` large; signal in generic
position):

- `E rho_signal ≈ k/d`.
- `E tr((ΠΣΠ)^2) ≈ (k/d)^2 [ tr(Σ^2) + (tr Σ)^2 / k ]`, so
  `rho_noise ≈ sqrt(k (k + d_eff)) / d`.
- Therefore

  ```
  tau_Π / tau ≈ sqrt( k / (k + d_eff) ) < 1.
  ```

- **For a bias-corrected estimator, random projection does not merely fail to help — it hurts.**
  It is neutral only for the plug-in's energy floor (`rho_signal ≈ rho_noise1 ≈ k/d`).

**Structured projection with `span(h_G, c) ⊆ range(Π)`:** `rho_signal = 1`, and the gain is
`1/rho_noise ≥ 1`. With `S* ⊕ nuisance` and a block-diagonal `Σ`, `Π = P_S*` gains

```
sqrt(1 + tr(Σ_nuis^2) / tr(Σ_⊥,S*^2)).
```

**Data-dependent `Π`.**

- The formulas hold **conditionally on `Π`** only if `Π` is independent of the estimation data.
  Hence cross-fitting.
- Same-sample selection aligns `Π` with the noise realization and inflates `C_Π^2` beyond
  anything the E3 correction assumes. It is forbidden.

**Selection by principal components.** The per-rollout second moment has population eigenvalues:

- behavior "spikes" `ℓ_j σ_b^2`, over a nuisance bulk `σ_n^2` (flat case).

By the BBP transition (sample size `N_s`, aspect ratio `γ = d/N_s`):

- a sample eigenvector carries information about spike `j` iff `ℓ_j σ_b^2/σ_n^2 − 1 > sqrt(γ)`;
- its squared overlap is then `(1 − γ/(ℓ−1)^2) / (1 + γ/(ℓ−1))`.

Consequences:

- **R2/R3 retain behavior signal only if behavior directions stand out of the nuisance spectrum
  by more than `1 + sqrt(d/N_s)`.**
- They fail when nuisance directions have larger per-rollout variance than behavior directions:
  those are selected first.

### 2.4 Random-projection negative control

**Prediction (§2.3):** R1 lowers detectability by `≈ sqrt(k/(k + d_eff))` relative to R0.

- **If R1 performs as well as the behavior-aware methods,** the reduction itself (not the
  behavior awareness) is doing the work, and the interpretation "behavior-relevant geometry" is
  weakened.
- **This is reported as found.**

### 2.5 Functional geometry [proved]

**Pull-back metric.** `||δf||_W^2 = g^T J_f^T W J_f g = g^T G_f g`, with `G_f = J_f^T W J_f`.

- So functional geometry is parameter geometry in the (semi-)metric `G_f`.
- For categorical behavior probabilities `f_(x,b) = p_θ(b | x)` on a probe prompt set with
  weights `w_x`, and the Fisher–Rao output metric `W = diag(w_x / p_xb)`:

  ```
  G_f = Σ_x w_x Σ_b p_xb ∇log p_xb ∇log p_xb^T = F_probe,
  ```

  the Fisher information of the probe behavior distribution.

**Consequences:**

1. Parameter directions that do not change probe behavior (`null(J_f)`) are annihilated. Their
   gradient noise **and** any verifier error in them disappear exactly.
2. `C_f = 0` whenever `C = 0` (the lemma of §2.3).
3. `C_f` can be 0 while `C_beh > 0` if the error changes only behaviors the probe does not
   observe.

**Estimation.**

- Per-group functional contributions are `ζ_k = W^1/2 J_f x_k` (exact Jacobian-vector products;
  no finite virtual steps).
- This is an unbiased linear image, so E3 and the null test apply unchanged.
- **Cost:** the probe Jacobian needs `q_f` backward passes (one per probe output); no gold labels
  and no rollouts.

**Invariances** [proved]:

| perturbation | effect |
| --- | --- |
| output rescaling `f → λ f` (same `W` formula applied to the rescaled outputs) | `C_f^2 → λ C_f^2`; `alpha_f` and `C_f/A_f` unchanged; the null test unchanged (scale-free statistic) |
| constant (θ-independent) extra outputs | zero Jacobian rows: invariant |
| duplicated probes | change `w_x`, hence `G_f`: **not invariant** (reported) |
| extra outputs that depend on nuisance parameters (e.g. surface-form frequencies) | add nuisance rows to `G_f` and reintroduce nuisance noise and nuisance error: **not invariant**; this is why the probe must be behavior-defined |

### 2.6 The null test [registered; E005a §10 lesson 1]

For a representation with contributions `(x_k^G, x_k^e)`, `k = 1..n`:

1. `alpha_hat` and `u_hat` come from the representation's audit means. If `||h_bar_G|| = 0`:
   `alpha_hat := 0` and `P_hat_perp := I`.
2. `z_k = P_hat_perp (x_k^e − alpha_hat x_k^G)`.
3. `K = Z Z^T` and `T = Σ_{i≠j} K_ij`.
4. The reference is `T_b = s_b^T K s_b − tr K` over `B = 199` Rademacher vectors `s_b`.
5. `p = (1 + #{T_b ≥ T}) / 200`; **reject `C = 0` iff `p ≤ 0.05`.**

Properties:

- `E T ≈ n(n − 1) C^2`.
- Under `H_0` the leading `u_hat`, `alpha_hat` effects cancel. `P_hat_perp h_bar_G = 0` exactly,
  so `mean_k z_k = P_hat_perp ε_δ` with no `alpha h_G` term. The residual terms from
  `alpha_hat − alpha = O(n^-1/2)` are at most `O(sqrt n)` in `T`, against a null SD of `O(n)`:
  - a negative `O(1)` shift makes the test slightly conservative near `A ≈ 0`;
  - an `O(sqrt n)` fluctuation that the flips do not reproduce makes it slightly liberal at
    small `n`.
- The sign-flip reference reproduces the degenerate (skewed) null of `U_n` without a normal
  approximation. It is exact if the `z_k` are sign-symmetric; otherwise approximate. **Its
  calibration is measured, not assumed.**
- **Cross-fitted R3** uses `T = T_A + T_B` with independent flips in each fold.

**The point estimate is E3** (frozen E005a code, unchanged). The test statistic `T` is used only
for the decision.

### 2.7 Update level vs reward level [registered]

**E005a-R measures reward-level geometry only.** Before E005b, the translation to the GRPO
update level needs:

1. per-group advantages divided by the group reward SD (a ratio of group statistics; the
   expectation depends on `m` and has no clean bias theory; E005a §1);
2. the optimizer preconditioner (Adam `t = 0`: `M = diag(1/sqrt(v_hat))`), estimated from an
   **independent** batch;
3. the functional version `δf = J_f M_Adam g_hat_upd`. It can be measured directly as a virtual
   first step on independent data.

**No claim about GRPO update geometry is made unless directly tested.**

---

## 3. Signal anchors [registered]

**Comparability decision.** E004a's oracle `C` is **not directly comparable**:

- it is update-level (GRPO-normalized, Adam `t = 0` metric), not reward-level;
- its absolute value depends on the U-toy parameterization, whereas E005a-R uses a new one.

Its numbers are therefore **not reused**. The mapping is the dimensionless per-group
detectability `tau` (§2.2), evaluated at the reward level in the identity metric (the E005a-R
estimand) for the historical structures.

**Source** (step-0 script `experiments/e005ar/e005ar_anchors.py`, commit `0693aa6`; run
`results/E005aR-anchors/20260927T214225Z_0693aa6`):

- all 768 E004a design structures (`configs/e004/design_panel_0b.json`);
- base policy `theta0`, the structure's own verifier;
- `m = 8`, M_I;
- E005a exact oracle and E005a Monte Carlo group covariances (`10^6` groups);
- nonzero signal: `C/A > 1e-6` (736 of 768).

**Anchors (quantiles of historical nonzero `tau`):**

| anchor | quantile | `tau` | bootstrap 95% | implied `n tau` at `N = 1024` (`n = 128`) |
| --- | --- | --- | --- | --- |
| small | Q25 | 0.003078 | [0.002218, 0.004026] | 0.39 |
| **medium** (primary) | Q50 | 0.02099 | [0.01668, 0.02693] | 2.69 |
| large | Q75 | 0.07878 | [0.06789, 0.09107] | 10.08 |

- **Cross-check** (the E005a design panel's natural `theta_0` points; same generator, fresh
  draw): Q25 / Q50 / Q75 = 0.001919 / 0.02818 / 0.1056 (n = 23).
- **Historical `kappa = C/A`** (for interpretation): Q25 / Q50 / Q75 = 0.054 / 0.142 / 0.357.
- **By mechanism** (median `tau`): D 0.12, YB 0.079, B 0.011, X 0.011, YA 0.005, R 0.001.
  32 of the 96 R structures are exactly aligned (`C = 0`).
- **Consequence stated before any E005a-R data** (§2.2).
  - At `N = 1024` (`n = 128`) the medium anchor gives `n tau = 2.69`.
  - Even in the oracle subspace, a perfectly calibrated one-sided 5% test therefore has
    null-dominated power ≤ `Φ(2.69 − 1.645) ≈ 0.85`, and less when the signal term raises
    `s_1/s0`.
  - **The medium-signal gate is at the edge of what the oracle ceiling can reach at
    `N = 1024`.** TP-R5 makes this quantitative per point.

**Use of the anchors.**

- In the calibration environment, a dose `D ∈ {small, medium, large}` is set so that the
  behavior-subspace detectability at the point equals the anchor:

  ```
  C_beh^2 / sqrt(2 tr(Σ_⊥,S*^2)) = tau_D
  ```

  with `Σ_⊥,S*` from the point's own verifier (Monte Carlo).
- The oracle-subspace ceiling R5 therefore faces the historical detectability. The experiment
  isolates what nuisance dimensions and representations do to it.
- **Frozen.** The anchors are not relaxed after any E005a-R result. The medium anchor is the
  primary practical scale; the small anchor is secondary (brief §13).

---

## 4. Calibration environment [registered]

**Seeds:** `SeedSequence(20261201).spawn(6)` = [anchors, design panel, test panel, design MC,
test MC, theory].

**Coordinates.** Parameter space `R^d = S* ⊕ nuisance`, with `S* = R^r` (the first `r`
canonical coordinates).

- Every candidate method is rotation-equivariant: identity metric, Haar random subspaces,
  eigen-decompositions, Jacobian maps.
- So the calibration is run in canonical coordinates without loss of generality; a unit test
  checks equivariance.
- No method may use coordinate identity.

### 4.1 Behavior block (the policy)

| element | specification |
| --- | --- |
| prompt types | `x ∈ {1..8}`, weights `w_x = 1/8` |
| behavior classes | `b ∈ {0, 1, 2, 3}`; `b = 0` is correct |
| gold | `G = 1[b = 0]` |
| class probabilities `p(b \| x)` | ordinary: `p_0 ~ U(0.3, 0.7)`; low reward variance: `p_0 ~ U(0.08, 0.2)`; low `A`: `p_0 ~ U(0.005, 0.02)`; the incorrect mass is split `Dirichlet(2, 2, 2)` |
| behavior features | `u(x, b) ~ N(0, I_r)`, `r ∈ {4, 8, 16}` |
| behavior score | `s(x, b) = u(x, b) − Σ_b' p(b' \| x) u(x, b')` (softmax policy at its evaluation point) |
| behavior Fisher | `F_S* = Σ_x w_x Σ_b p_xb s s^T`; `f_bar = tr(F_S*)/r`; full rank is required |

**Bases:** one per `(r, type)`, i.e. **9 bases per split**, drawn independently for design and
test.

### 4.2 Nuisance block

- **Surface features** `w ~ N(0, Λ)` in `d − r` dimensions, independent of `b` and of the
  prompt. At the evaluation point their score is `w` itself.
- They change surface form only, so class probabilities are independent of the nuisance
  parameters.
- `d ∈ {64, 256, 1024}`.
- **Spectra** (eigenvalues in units of `f_bar`):
  - **bulk:** `λ_i = 0.25`, isotropic. Behavior directions stand out (PCA-favourable; the
    "outliers over a bulk" picture of neural gradient spectra).
  - **flat:** `λ_i = 1`, isotropic (neutral).
  - **spiked:** `λ_i ∝ i^-1/2`, normalized to mean 1. A few nuisance directions exceed the
    behavior variance (PCA-adversarial).
- **Surface-bonus direction:** `v ~ N(0, I_{d−r})`, normalized, per (base, `d`, spectrum).

### 4.3 Verifier

```
V = s_V · Z + λ · B,    Z ~ Bern(EV_rho(x, b)),    B = 1[v^T w > 0],
EV_rho = (1 − rho)(a' G + c0) + rho EV_con(x, b).
```

**Parameters:**

- `a' ~ U(0.5, 0.95)` and `c0 ~ U(0, 0.3(1 − a'))`, per base (the E005a affine-null law).
- **Constructions** `EV_con` (behavior-relevant error), with half-prompt subsets drawn per base:
  - **shortcut** (Y-like): on half of the prompts, class 1 is also accepted;
  - **deletion** (X-like): on half of the prompts, `EV = 0.5` constant;
  - **partial** (B-like): class 2 gets credit 0.5 everywhere.
  - Elsewhere, `EV_con = G`.
- **Scale `s_V`** sets `alpha` exactly:

  ```
  1 + alpha = s_V [(1 − rho) a' + rho (1 + alpha_con)].
  ```

  The surface bonus has zero projection on `h_G`, so `alpha` is independent of `λ`.
- **Behavior dose:** `C_beh = s_V rho C_con,⊥`, exactly linear for fixed `s_V`. `rho` is solved
  by bisection (with common random numbers) so that the behavior detectability `tau` equals the
  anchor (§3). Points with no solution in `rho ∈ (0, 1]` are dropped and counted.
- **Nuisance (behavior-irrelevant) error:** `λ h_surf` with `h_surf = Λ v / sqrt(2π v^T Λ v)`
  [proved]. It lies in the nuisance block, so `C_beh` is unchanged and `C_full^2 = C_beh^2 +
  λ^2 ||h_surf||^2`.

### 4.4 Cases per base [registered]

| case (brief §6) | construction | `alpha` targets | behavior dose | `λ` | nuisance configs |
| --- | --- | --- | --- | --- | --- |
| clean null (4: aligned) | `rho = 0` | −0.6, −0.3, 0 | 0 | 0 | 9 |
| inside `S*` (1); orthogonal & behavior-relevant (5) when `alpha = 0` | 3 constructions | −0.3, 0 | small, medium, large | 0 | 9 |
| nuisance only (2); orthogonal & behavior-irrelevant (6) when `alpha = 0` | `rho = 0` | −0.3, 0 | 0 | `λ ||h_surf|| = C_med(base, alpha)` | 9 |
| partial inside / outside (3) | 3 constructions | −0.3 | medium (the same behavior verifier as the matched in-`S*` point) | `λ ||h_surf|| = C_beh` | 9 |

- `C_med(base, alpha)` is the median over constructions of the medium `C_beh`.
- The **9 nuisance configs** are `d ∈ {64, 256, 1024}` × {bulk, flat, spiked}.
- **Composition:** 26 behavior configurations × 9 = 234 points per base, **2106 per split**
  (before drops).

**Matched sets** [registered]:

- **null sets:** (base, `alpha`) across {clean, nuisance-only} × 9 nuisance configs;
- **dose sets:** (base, dose, `alpha`) across constructions (and the partial variant) × 9
  nuisance configs.

Within a dose set, `C_beh`, `A`, `alpha` and `r` are identical. **Across `d` the behavior block
is identical**: the `d`-series is the primary dimensionality test (brief §11).

---

## 5. Representations (frozen list) [registered]

Every representation feeds per-group contributions to **E3** (point estimate, jackknife SE) and
to the §2.6 test.

| id | representation | selection data | status |
| --- | --- | --- | --- |
| **R0** | full space, identity metric | — | baseline (historical E3) |
| **R1** | random orthonormal `k`-subspace (Haar; drawn fresh per replication; nested in `k`) | — | **negative control**, never eligible |
| **R2** | top-`k` eigenvectors of the uncentered second moment of **per-rollout gold RLOO contributions** from an independent selection half (first `n/2` groups); geometry on the other half | gold, `N/2` | candidate |
| **R3** | cross-fitted joint subspace: fold A → top-`k` eigenvectors of the uncentered second moment of the stacked per-rollout `[gold; verifier]` RLOO contributions → geometry on fold B, and vice versa | gold + verifier, per fold | candidate |
| **R4** | functional: `ζ_k = W^1/2 J_f x_k`, where `f` = the class probabilities `p(b \| x)` on the 8 prompt types (32 outputs), `W = diag(w_x / p_xb)` (Fisher–Rao), exact JVP | none (probe Jacobian only) | candidate |
| **R5** | oracle `P_S*`, identity metric (`k = r`) | oracle | **ceiling**, never eligible |

**Details:**

- **Eigenvectors:** exact LAPACK `eigh` on the smaller of the Gram and covariance matrices.
- **R3 aggregation:** estimate = mean of the two fold E3 values; SE = `sqrt(SE_A^2 + SE_B^2)/2`.
- **`k` grid:** `k ∈ {4, 8, 16, 32, 64}`, subject to `k ≤ d` and `k ≤` the selection rank.
  Otherwise the cell is undefined and reported.
- **No R3 basis variants** beyond the one above.
- **R4 in this environment.** The probe observes the behavior classes, so `null(J_f)` is exactly
  the nuisance block and `G_f = F_S*`. R4 is therefore "S* in the probe-Fisher metric" **by
  construction of the environment**. What this can and cannot show is stated in §11.

**R4 sensitivity variants** (design split only; diagnostics, never eligible):

| variant | change |
| --- | --- |
| R4-sub | probe on prompts 1–4 only |
| R4-dup | prompt 1 listed 3 times |
| R4-scale | outputs × 100 |
| R4-const | 8 constant outputs appended |
| R4-surf | 8 surface-form outputs appended: `P(v^T w > 0 \| x)`, Bernoulli Fisher weight |

---

## 6. Budgets and costs [registered]

- **Audit sizes** (primary): `N ∈ {64, 128, 256, 512, 1024}`, plus `N = 32` (scaling
  diagnostics only). Group size `m = 8`; `R = 100` Monte Carlo replications per point × `N`.
- **Budget matching:** every representation uses the same `N` gold-labelled rollouts, so
  comparisons at equal `N` are budget-matched. R2 pays for selection by estimating on `N/2`; R3
  estimates twice on `N/2`.
- R2 and R3 run only at `N ≥ 64` (at least 4 groups per half).

| representation | `B_roll` | `B_gold` | `B_bwd` |
| --- | --- | --- | --- |
| R0, R1, R5 | `N` | `N` | `N` |
| R2 | `N` (`N/2` selection + `N/2` estimation) | `N` | `N` |
| R3 | `N` (2 folds) | `N` | `N` |
| R4 | `N` | `N` | `N + 32` (probe Jacobian) |

**Upper limit:** the primary budget never exceeds 1024.

---

## 7. Metrics (per representation × `k` × `N`) [registered]

**Measurement quality** (E3 `C^2` against the representation's own oracle; for R1–R3 the oracle
is recomputed per replication from the exact gradients and that replication's basis):

- bias, RMSE, calibration slope, coverage of E3 ± 1.96 SE;
- non-finite rate;
- **Spearman** of the per-replication estimates against `C_beh^2` across non-null points
  (averaged over replications).

**Null behavior** (null points = clean + nuisance-only, i.e. `C_beh = 0`):

- sign-flip FPR (pooled and per point); legacy jackknife-Wald FPR;
- 95th percentile of null E3;
- `SDB_null` over null matched sets (E005a §6 definition).

**Signal retention:**

- R1–R3: `rho_signal = C_Π^2(h_G, P_S* h_e) / C_beh^2` per replication;
- nuisance leakage `||Π h_e,nuis||^2 / ||h_e,nuis||^2`;
- R4: Spearman(`C_f^2`, `C_beh^2`) at the oracle level across non-null points.

**Noise reduction:** the null SD and q95 relative to R0 at the same null point.

**SNR:** `z = (mean E3 at the point − mean E3 at the matched clean null) / SD of E3 at that
null`, with the matched null sharing base, `alpha`, `d` and spectrum.

**Power:** rejection rate by dose. **Medium power** is taken over in-`S*` medium and partial
points.

**Reported by:** `d`, `k`, `N`, spectrum, `r`, base type, construction and case.

---

## 8. Theory predictions (tested on the DESIGN split) [registered]

The oracle quantities come from the panel's Monte Carlo covariances (behavior block, `2 × 10^5`
groups per verifier) and the exact nuisance block `κ_δ Λ`.

| id | prediction | registered criterion |
| --- | --- | --- |
| **TP-R1** (floor law) | at clean nulls, `SD(E3)` of R0 and R5 equals `sqrt(2 tr Σ_⊥^2 / (n(n−1)))` | ratio in [0.8, 1.25] for ≥ 80% of (point, `N ≥ 256`) cells, each representation |
| **TP-R2** (dimension) | at matched clean nulls (base, `alpha`, spectrum), R0's null SD at `d = 1024` vs `d = 64` equals the predicted `sqrt(tr Σ_⊥,1024^2 / tr Σ_⊥,64^2)`, while R5's is 1 | observed / predicted in [0.8, 1.25] for ≥ 80% of pairs (`N ≥ 256`), for each |
| **TP-R3** (random projection hurts) | median over in-`S*` medium and large points (bulk and flat spectra, `N ≥ 256`) of `z_R1(k)/z_R0` | < 1 for every `k < d`; and the observed / predicted ratio (`rho_signal` mean / `rho_noise` from §2.3) in [0.67, 1.5] for ≥ 70% of cells |
| **TP-R4** (oracle gain) | at in-`S*` medium points, `N ≥ 256`, `z_R5/z_R0 = sqrt(tr Σ_⊥,full^2 / tr Σ_⊥,S*^2)` | observed / predicted in [0.67, 1.5] for ≥ 80% |
| **TP-R5** (ceiling power) | R5's medium power at `N = 1024` equals the §2.2 prediction averaged over medium points | within ±0.15 absolute |
| **TP-R6** (PCA selection) | R2's median retention orders bulk ≥ flat ≥ spiked | in ≥ 2/3 of (`d`, `k`) cells at `N = 1024` |

---

## 9. Design round and selection [registered]

**Eligible candidates:** R2 × `k`, R3 × `k`, R4 (11). R0, R1 and R5 are never eligible.

**Practical gate** at budget `N` (evaluated on design for selection; on test for the verdict):

| # | criterion | threshold |
| --- | --- | --- |
| G1 | null FPR | pooled sign-flip FPR ≤ 0.05 **and** ≥ 90% of null points with FPR ≤ 0.10 |
| G2 | structure-dependent null shift | `SDB_null ≤ 0.5` |
| G3 | ranking | Spearman with `C_beh^2` ≥ 0.70 |
| G4 | medium power | ≥ 0.80 |
| G5 | numerical | non-finite rate (`C^2` or `T`) ≤ 0.01 |
| G6 | SNR vs full space | median `z_R` ≥ 2 · max(median `z_R0`, 0) and median `z_R` > 0, over medium points |
| G7 | vs random control | medium power ≥ that of R1 + 0.20 **and** median `z_R` ≥ 2 · max(median `z_R1`, 0). Comparator: R1 at the same `k` (R2/R3), or at `k` = the grid value equal to `r` (R4) |

**Selection rule:**

1. For each candidate, `N*` is the smallest `N ∈ {64, …, 1024}` where G1–G7 all pass on design.
2. If any `N*` exists: pick the smallest `N*`. Ties go to higher medium power, then higher
   Spearman, then the order R4 < R2 < R3, then smaller `k`.
3. Otherwise: pick the candidate that passes the most criteria at `N = 1024`, with the same
   tie-breaks.

**Then freeze** `configs/e005ar/e005ar_estimator_frozen.json` (sha256). It records:

- the representation and `k`;
- E3 (E005a code, unchanged);
- the sign-flip test (`B = 199`, level 0.05);
- the handling rules;
- the source hashes.

---

## 10. Held-out test, decision tree, stopping rule [registered]

**Test.**

- It runs once, behind `configs/e005ar/E005AR_TEST_APPROVED`, which names the frozen sha256.
- No retuning. All candidates are reported; the verdict uses only the frozen selection.
- **Minimum practical budget** = the smallest `N ≤ 1024` at which the selected representation
  passes G1–G7 on test.

**Decision tree** (exactly one label):

| label | condition | recommendation |
| --- | --- | --- |
| **A — PRACTICALLY VIABLE GEOMETRY** | a minimum practical budget exists on test | E005b neural design with the frozen representation |
| **C — REPRESENTATION FAILURE** | not A, and at `N = 1024` on test: retention fails (R2/R3: median `rho_signal` < 0.5 over in-`S*` non-null points; R4: oracle Spearman(`C_f^2`, `C_beh^2`) < 0.7) **or** G7 fails | stop practical geometry entirely |
| **B — MECHANISTICALLY VALID, NOT PRACTICALLY MEASURABLE** | not A, not C, **and** large-dose power ≥ 0.80 at some `N ≤ 1024` for the selected representation or R5 | geometry stays an oracle/mechanistic arm; not the primary practical diagnostic in E005b |
| (fallback) | not A, not C, and the large signal is also undetectable | labelled **C** |

**Stopping rule.**

- One design round, one test.
- The audit budget is not raised beyond 1024.
- The anchors, grids and thresholds are not changed after data.
- There is no E005a-R2.

---

## 11. What this design can and cannot show; integrity

**What it can and cannot show:**

- **R4's validity is partly built in.** The environment's probe observes exactly the behavior
  classes. A positive R4 result therefore shows that **if** a behavior probe spans the
  behavior-relevant directions, functional geometry removes the nuisance barrier at the stated
  budgets. It does not show that a real probe does so for a transformer. The R4-sub and R4-surf
  variants quantify the two failure directions: missing coverage and nuisance contamination.
- **R2/R3 depend on the spectral regime.** The three spectra are registered in equal numbers
  because the neural regime is unknown.
- **The anchors** are historical U-toy detectabilities. Whether transformer verifier errors have
  similar `tau` is unknown.

**Integrity:**

- New namespace `src/vdyn/e005ar/`; E005a code is imported unchanged.
- Strict TDD (RED observed), with mutation tests for the estimator, subspace and test logic.
- Checks fail closed; no hidden exit codes.
- **Every run's `meta.json` records:**
  - the commit and dirty status;
  - the root seed;
  - the config sha256, panel sha256 and (from selection on) frozen-estimator sha256;
  - Python, numpy, scipy, torch and scikit-learn versions.
- Raw per-replication arrays stay local (sha256 recorded); gzip summaries are committed.
