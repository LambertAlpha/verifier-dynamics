# 05 — E002 design memo (v3, 2026-09-25)

**Status: DESIGN MEMO. The authoritative E002a/b protocol is the registry entry "E002 —
pre-registration" plus its dated amendments (`03_experiment_registry.md`); where this memo and the
registry differ, the registry wins. §13 (E002c) is design only: not registered, not implemented,
not run.**

- v3 (2026-09-25) adds §13 (E002c, unmatched panel) and this naming note. Arm names in §4–§12
  below are the v2 draft names; the registry renamed them:

  | v2 memo | registry (E002a/b) |
  | --- | --- |
  | S0 | G0 static metrics |
  | G-est / G-oracle | G1 / G1-oracle |
  | P3 verifier-score growth | P1 |
  | P2 importance-reweighted FPR growth | P2 |
  | P1 / P5 fresh-label probes | P3 |
  | L local-perturbation audit | P4 |

- v1 (2026-09-25) asked whether `(A, alpha, C)` can be estimated at all. v2 reframes E002 around
  the collaborator's question below. The v1 material on estimator behaviour survives as sub-study
  E002a (§8).
- The design-phase checks quoted here (facts F2–F4) were computed in the scratch directory by
  exact enumeration and agree to 2e-15. They are disclosed as not pre-registered.

## 1. Question

> Under a matched certification budget, do optimizer-conditioned gradient-geometry diagnostics
> predict long-run verifier-induced **gold** failure earlier, more cheaply, or more reliably than
> running a short optimization probe and observing ordinary metrics?

E003 established four things:

- Matched static metrics can lead to very different asymptotic `J_G`.
- `C(0)` is informative but not sufficient (Spearman 0.913; one predicted inversion).
- By `t <= 5` the `eta` summaries fix the inversion, but so do FPR growth and early gold progress.
- Therefore the practical incremental value of geometry is unproven (WH-6).

## 2. Facts that shape the design (read these first)

- **F1 — first-order equivalence [proved, Prop. 1].** Under `dtheta/dt = M g_V`, one verifier-driven
  step changes `J_G` at rate `(1+alpha) A^2` and `J_V` at rate `(1+alpha)^2 A^2 + C^2`. One
  gold-driven step changes `J_G` at rate `A^2`. So the geometry triple is exactly what first-order
  function-space probes measure. **Gradient geometry has no first-order information that a short
  probe lacks.** Any advantage must be statistical efficiency, cost, or not needing to train.
- **F2 — in Candidate 3, `C` is the initial FPR growth rate [derived-agent; exact].** Under natural
  gradient, `dFPR/dt = C^2 / (1 - q)`. This holds exactly, including OR structures.
- **F3 — the matched panel neutralizes `A` and `alpha` [derived-agent; exact].**
  - At matched `q0` and FPR, `A = sqrt(q0(1-q0))` and `alpha = -FPR` are identical across
    structures (Prop. 8). Geometry therefore reduces to `C`.
  - `dJ_V/dt = (1-FPR)^2 q0(1-q0) + C^2`, so the **zero-gold** verifier-score growth ranks
    structures by `C^2` at first order.
  - `dJ_G/dt = (1-FPR) q0(1-q0)` is identical for all structures, so a gold-progress probe is
    blind at first order.
- **F4 — estimation is limited by the number of false-positive events [derived-agent; exact for
  single / AND].**
  - Per-rollout noise of the plug-in `C_hat^2` (oracle Fisher) is
    `(1-q) kappa^2 / S + S q - ||g_e||^2`.
  - The per-rollout SNR for `C^2` is therefore about `(1-q) S`, i.e. the FP mass. Detecting
    `C > 0` needs on the order of 10 observed false positives.
  - FPR-growth probes face the same event limitation.
- **F5 — label reuse [accounting fact].** Geometry at `t = 0` can reuse the labels of the static
  audit set: zero *additional* gold. A probe that evaluates FPR or `J_G` on fresh samples from
  `pi_k` needs new labels, **unless** it reweights the `pi_0` audit set by importance weights
  `pi_k / pi_0`. For small `k` that reweighting is itself a first-order gradient estimator (F1).

**Consequence.** Within Candidate 3, E002 compares **estimators and costs**, not information
content, at small probe horizons. At larger horizons probes see path curvature that `C(0)` cannot
(E003 Q4/Q5). E002 can falsify "geometry is practically useful" here; it cannot establish
usefulness in general (§12).

## 3. Budget accounting

Each method is charged a cost vector `(B_roll, B_gold, B_bwd)`, plus verifier calls (free: the
verifier is the cheap component) and wall-clock.

| Arm | `B_roll` | `B_gold` | `B_bwd` |
| --- | --- | --- | --- |
| S0 static audit | `m0` from `pi_0` | `m0` | 0 |
| G geometry (`t = 0`) | `m0` labeled + `u` unlabeled from `pi_0` | `m0` (the audit set, reused) | `m0 + u` per-sample score passes; Fisher estimate from the same samples |
| P-fresh (FPR / gold at `pi_k`) | `k·b` (training, V only) + `m` from `pi_k` | `m` (+ `m0` if the audit is needed) | `k` batched updates |
| P-IS (FPR at `pi_k` by reweighting the audit) | `k·b` | 0 new (reuses `m0`) | `k` + `m0` log-prob evaluations under `pi_k` |
| P-JV (verifier-score growth) | `k·b` (+ unlabeled eval) | **0** | `k` |
| L local-perturbation audit (§4) | `m0` + `r·m0` edited variants | `m0` (+ edited items if an edit can change correctness) | 0 |

**Gold-label rules** (identical for all arms):

- Mode A (primary, conservative): every audited response costs one label.
- Mode B (secondary, declared): only verifier-accepted responses are gold-audited; rejected ones
  count as wrong. This is valid only because FNR = 0 in this family. Every arm uses the same rule.

**Matched budget.** Every arm receives the same total `B_gold` and the same total `B_roll`, and
allocates them freely (for example, a probe may split `B_gold` between `pi_0` and `pi_k`, or spend
it all at `pi_k`).

- Budget grid: `B_gold ∈ {16, 32, 64, 128, 256, 512, 1024}`, `B_roll / B_gold ∈ {1, 4, 16}`.
- Compute (`B_bwd`, Fisher cost) is reported separately and is not matched.
- **Primary budget cells** (pre-specified): `(B_gold, B_roll) = (64, 256)` and `(256, 1024)`.

What "matched" means for each arm:

- **A — one-shot geometry.** All `B_gold` labels go to `pi_0` samples; up to `B_roll - B_gold`
  extra unlabeled `pi_0` samples may improve `g_hat_V` and the Fisher estimate. No updates.
- **B — `k`-step probe.** `k·b + m <= B_roll` rollouts in total; labels only where the chosen probe
  needs them; the same optimizer as the training run.

## 4. Competing predictors

The information access is identical for every arm:

- the policy log-prob function (needed to sample and train);
- the verifier as a queryable black box;
- gold as a black box with a per-call cost;
- panel-level constants `q0` and FPR, known to all arms.

Nobody knows the block structure (which parameters are "gold" vs "exploit"), exact gradients, or
the target.

| ID | Predictor | Gold used |
| --- | --- | --- |
| S0 | initial accuracy / FPR / FNR estimates | audit |
| **G-est** (primary geometry) | `C_hat` at `t = 0`, optimizer-matched metric = damped **estimated** Fisher; estimator (plug-in or U-statistic) and damping `lambda` tuned on the design split | audit |
| G-oracle | as G-est with the exact Fisher metric (reported; **not eligible for the success claim**) | audit |
| G-eta | `eta_hat = C_hat^2 / C_hat_max^2` | audit |
| P1 | `delta log FPR` with fresh labels at `pi_k` | fresh |
| P2 | `delta log FPR` by importance-reweighting the audit set | audit only |
| P3 | `delta J_V` (verifier-score growth) | none |
| P4 | `delta J_G` with fresh labels | fresh |
| P5 | logistic combination of P1/P3/P4, fitted on the design split | fresh |
| L | local-perturbation near-miss audit: resample one feature of each audited wrong response from the policy marginal and query the verifier (a gradient-free Boolean-influence estimate of accessibility) | audit |
| R0 | random ranking (chance) | — |

- In the matched panel, S0 is uninformative **by construction** (expected C-index 0.5). It is a
  negative control, not a competitor.
- Tuning is symmetric: each arm may tune at most three hyperparameters on the design split, per
  budget cell (geometry: estimator and `lambda`; probes: `k`, step size, label split).

## 5. Synthetic verifier panel (Candidate 3 family)

- **Policy.** Independent Bernoulli `(corr, z_1..z_m)`. Gold `G = corr`.
  Verifier `V = corr OR 1_E(z) [OR xi]`.
- **Operating points** (one panel each, `q0` fixed within a panel):
  - Primary **P-mod**: `q0 = 0.05`; FPR `f` chosen from `{0.05, 0.1, 0.2}`.
  - Secondary **P-rare**: `q0 = 0.002`; `f` from `{0.005, 0.01, 0.02}`.

  In each case `f` is fixed **before any predictor is computed**, by a registered rule: the value
  whose design-split stall fraction, from exact targets only, is closest to 0.5.
- **Structure types**, sampled uniformly:
  - SINGLE; AND_k (k = 2, 3, 4); OR_k (k = 2, 3); THR(2-of-3);
  - DNF `(z1∧z2)∨z3`; CNF `(z1∨z2)∧z3`;
  - MIX = `E ∨ xi`, with a fresh coin carrying a controllable share `rho ~ U(0, 1)` of the FPR;
  - RFP (pure coin), 5%.

  MIX gives continuous variation of `C` at fixed FPR and is used for dose–response (E002a).
- **Parameter sampling.** Raw feature logits `a_i ~ N(0, 2^2)`. A common shift `c` is then solved so
  that `P(E) = f` (or the MIX target). Events are monotone, so the root is unique. The draw is
  rejected and redrawn if any `s_i ∉ [1e-4, 1 - 1e-4]`.
- **Matching.** Every structure has the same `q0`, FPR `f`, FNR = 0, and hence the same accuracy
  and FP mass. `A` and `alpha` are therefore matched automatically (Prop. 8).
- **De-duplication.** Canonicalize (sort features inside symmetric groups; drop features with
  `s_i > 1 - 1e-4`, which are effectively constant). Reject a structure whose canonical logit vector
  is within 1e-3 (max-abs) of an earlier one of the same type. **This rule uses parameters only,
  never predictors or outcomes.**
- **Size and split.**
  - 900 structures per panel, stratified random split: 300 design, 600 test.
  - Secondary robustness: leave-one-type-out evaluation on the test split.
- **Exclusion.** Target computation failing its QA check (§6). The count is reported; if more than
  1% fail, stop.

## 6. Targets (fixed before any predictor is computed)

All targets are exact, not noisy training outcomes. The training optimizer is the natural-gradient
flow, as in E003.

- **Primary continuous:** gold shortfall `D = 1 - J_G(∞)`. The clean natural-gradient run reaches
  1, so `D` equals the shortfall relative to the clean verifier.
- **Primary categorical:** stall iff `q0 exp(Λ(1)) < 1` (Prop. 9).
- **Secondary:**
  - `J_G(∞)`;
  - `J_G(T = 25)`;
  - time to stall `t_95` for stalls only (`J_G` reaches `q∞ - 0.05(q∞ - q0)`).
- **Computation.** `Λ(1) = ∫ dS / (S eta)` along the feature-space natural-gradient ascent curve of
  `S`. That curve is independent of `q`. Use closed forms where available (Prop. 9); otherwise
  integrate the `q`-free feature ODE with quadrature.
- **QA (registered).** On a random 10% of structures, the full generic-optimizer ODE (E003 code)
  must agree with the targets to 1e-6 in `J_G(∞)` (or 1e-3 for OR-type algebraic convergence).
- **Never used as a target:** the signed proxy–gold gap.

## 7. Endpoints and inference

- **Primary endpoints:**
  - C-index: Harrell's concordance with `D`, over pairs with distinct `D`; predictor ties count ½.
    `D` has heavy ties at 0 for successes, so C-index is preferred over Spearman.
  - AUROC for stall vs success.
- **Secondary endpoints:** Spearman, Kendall τ_b, and recall of the 10% most dangerous structures
  (by `D`).
- **Replications.** For each arm × budget cell, `R = 64` independent replications. Each replication
  draws new rollouts for every test structure and computes the panel-level metric.
- **Reported per cell:** mean, SD across replications, and a 95% CI from a hierarchical bootstrap
  (2000 resamples: structures, then replications).
- **Primary contrast:** `Delta = metric(G-est) - metric(best competitor)`. The best competitor
  is chosen on the design split per cell, among P1–P5 and L. The contrast is taken on the same test
  structures with a hierarchical bootstrap CI, and Holm correction over 2 cells × 2 endpoints.
- **Main deliverable: budget–performance frontiers.** One frontier against `B_gold` for each
  `B_roll` ratio, plus one against `B_roll`. The summary is the area between frontiers over the
  registered budget range.
- **Oracle ceilings** (noiseless value of each predictor: exact `C(0)`, exact `eta0`, exact probe
  observables at each probe horizon `t_p`) separate information limits from estimation noise.
  They are computed and **registered as predictions before any finite-sample run**.

## 8. E002a — finite-sample geometry estimation (sub-study)

- **Structures.** The eight E003 structures, the MIX dose family (`C` from 0 to `C_max` at fixed
  FPR), and one A-extreme family (`q ∈ {1e-3, 1e-2, 0.99}`).
- **Sample sizes and estimators.**
  - `N ∈ {8, 16, 32, 64, 128, 256, 512, 1024, 4096}`.
  - Gradient estimators: E1 (no baseline) and E2 (leave-one-out baseline).
  - Metrics: exact Fisher, damped estimated Fisher (`lambda` grid), and Euclidean (which is the
    optimizer-matched metric for a vanilla-trained panel).
  - Estimators: plug-in, U-statistic Gram, and Gram form (`P = alpha A^2`, `D = A^2 C^2`), which
    stays finite as `A -> 0`.
- **Reported quantities.**
  - Bias, variance and RMSE of `A_hat`, `alpha_hat`, `C_hat`.
  - `P(C_hat_Y > C_hat_R)`, with Y = SINGLE/AND and R = RFP at the same FPR (the matched-R
    analogue).
  - Power at a fixed 5% false-alarm rate, and minimum detectable `C` from MIX.
- **Degenerate events**, each with its exact probability and reported as is:
  - `A_hat = 0`, reported with `alpha_defined = False` and never imputed; happens with probability
    `q0^N + (1-q0)^N` under E2.
  - No false positive in the sample, so `g_hat_e = 0`.
  - A feature constant in the sample, making `F_hat` singular.

## 9. Probe design

- The probe runs the same optimizer as training: stochastic natural gradient with RLOO gradients
  from `b` rollouts per step, Fisher exact or estimated (matched to G-oracle / G-est).
- Probe horizon `t_p = k · eta_p`, with `k ∈ {1, 2, 5, 10}` and step size `eta_p ∈ {0.1, 0.3, 1.0}`;
  `b = floor((B_roll - m) / k)`.
- **When do signals appear?**
  - Noiseless ceilings versus `t_p` (from exact flows) show when FPR growth and gold progress
    become predictive.
  - By F3, `delta J_G` is uninformative at first order and needs `t_p` large enough for `S` to
    change.
  - `delta FPR` and `delta J_V` carry `C^2` from the first step.
- **Does geometry have an edge before these signals appear?** At the same horizon the first-order
  signals are the same quantity (F1–F2). Any edge before probe signals are visible must therefore
  come from estimator efficiency at tiny `B_roll`. There a probe cannot take enough steps, while
  geometry needs no updates.

## 10. Proposed pre-registration

1. **Primary hypothesis H1.** In the held-out P-mod test panel, at matched `B_gold` and `B_roll`,
   G-est achieves a higher C-index for `D` or AUROC for stall than the best competitor among P1–P5
   and L, in at least one primary budget cell (Holm-corrected CI excludes 0). G-est is also not
   dominated on the budget frontier.
2. **Null H0.** At every primary cell, `Delta <= 0` or its CI includes 0.
3. **Panel generation:** §5 (types, sampling, shift-to-match, dedup, exclusion), with the root
   seed registered.
4. **Held-out split:** 300 design / 600 test, stratified by type. Test labels and test predictors
   are untouched until design-split tuning is frozen and committed.
5. **Budgets:** the §3 grid; primary cells `(64, 256)` and `(256, 1024)`; gold mode A primary.
6. **Predictors:** §4, exactly, with each arm's hyperparameter grid.
7. **Targets:** §6.
8. **Metrics:** §7.
9. **Success criterion for practical incremental value** (all must hold):
   - (a) H1 holds with **G-est**, not G-oracle;
   - (b) it holds at matched gold **and** rollout budgets (not only under one axis);
   - (c) it holds against the tuned probe set **including** the zero-new-gold probes P2 and P3;
   - (d) it holds on the P-mod test split and has the same sign on P-rare;
   - (e) compute is reported alongside.

**Challenge to the proposed criterion.**

- "Improvement over initial static metrics" is **vacuous in a matched panel**, because S0 is at
  chance by construction. It should be a sanity check, not evidence. The incremental-over-static
  question needs an unmatched panel (proposed as optional E002c, where FPR varies across
  structures and predictors are compared by incremental C-index over an S0-only model).
- "At one or more low-budget regimes" invites cherry-picking. Replace it with pre-specified
  primary cells plus a frontier non-dominance requirement.
- Charging probes fresh gold labels while geometry reuses the audit is **unfair to probes**. P2 and
  P3 must be in the competitor set.

## 11. Power and sample-size notes

- **Panel size.** With 600 test structures (~300 per class), `SE(AUROC) ≈ 0.02` (Hanley–McNeil at
  AUROC ≈ 0.8). A paired difference of about 0.05 is detectable at 80% power. For Spearman,
  `SE ≈ 0.035`.
- **Replications.** `R = 64` makes the replication-level SD precise to about ±9%.
- **Per-structure information limit (F4).** Distinguishing structures needs roughly
  `N ≳ 10 / (FP mass)` labeled rollouts for geometry (≈ 100 at P-mod, ≈ 1000 at P-rare). FPR probes
  need a comparable number of labeled false-positive events. Exact `N*` per structure (from the
  enumerated covariance) is computed and registered before running.
- **Cost.** The toy is cheap: 1800 structures × 21 budget cells × 64 replications × ~10 arms.
  Vectorized numpy should take hours at most. **This estimate is unverified until a pilot on the
  design split** (a pilot must not touch the test split).

## 12. Challenging the project

- **Can E002 falsify practical usefulness?** Yes, within Candidate 3 under natural gradient,
  provided that:
  - probes are tuned as generously as geometry;
  - the zero-new-gold probes are included;
  - the estimated metric is used;
  - the endpoints are pre-specified.

  It **cannot** establish general usefulness, because in Candidate 3 `C^2` *is* the first-order
  FPR growth (F2).
- **Cheaper baselines the brief was missing:**
  - (i) P3, verifier-score growth, which needs **no gold at all** and ranks `C^2` exactly in the
    matched panel (F3);
  - (ii) P2, importance-reweighting the existing audit, with zero new gold;
  - (iii) L, a gradient-free, training-free local-perturbation near-miss audit;
  - (iv) watching the frequency of candidate exploit features among accepted responses (requires
    knowing the features; excluded as privileged in the toy).
- **Is the result predetermined by Candidate 3?** Largely, in information content:
  - by F1–F3, geometry and first-order probes measure the same thing, and the matched panel makes
    `A` and `alpha` constant;
  - the panel's type mix (how many `eta`-decreasing, OR-like structures) sets the ceiling of any
    `t = 0` diagnostic.

  Mitigations: report oracle ceilings and type-stratified results, and treat E002-C3 as a test of
  **efficiency and cost**, not of unique information.
- **Abandonment criterion.** Abandon the "diagnostic" contribution and keep the mechanistic one
  (Props. 8–11, metric matching, the path law, the insufficiency results) if, in E002-C3 under
  natural gradient — the regime most favourable to geometry — **both** of these hold:
  - G-est is not better than the best of {P2, P3, L} at any primary cell;
  - geometry wins only with the oracle Fisher.
- **Agent's prior** (for the record; not a hypothesis):
  - most likely, G-est ties P2 and P3 at small budgets and is beaten by P1/P5 once `t_p` covers
    exploit takeoff;
  - a win, if any, is expected only at the tiniest rollout budgets, or in a "no training allowed"
    setting.
- **Where geometry could still matter** (outside E002-C3):
  - certification when updates are impossible or unsafe;
  - large models where fresh gold on updated policies is expensive;
  - verifiers where the orthogonal pressure is not immediately visible as FPR growth (partial
    credit, precursor behaviours, heterogeneous prompts).

  These need a family in which F2 fails. Proposed for discussion only, not for implementation.

## 13. E002c — unmatched panel (design only; NOT registered, NOT implemented, NOT run)

**Gate.** E002c is considered only if E002a/b justify it: G1 must not be abandoned under the E002
criteria (registry E002 §8). If E002b abandons the diagnostic claim, E002c is not run, because
the first-order identity `C^2 = (1-q) dFPR/dt` (theory note §12) holds per structure whether or
not the panel is matched.

**Question.** When ordinary static verifier quality varies naturally across verifiers, does
estimated geometry add predictive value for long-run gold shortfall **beyond** static metrics,
and beyond what the same budget spent on probes adds?

**Panel (Candidate 3, FNR = 0, same verifier types as E002).**

- Raw draws as in E002 (§2 of the registry), but **no common shift**: each structure draws its
  own shift `c ~ U(-3, 3)` on the raw logits, so FPR, FP mass and `S_E` vary. The start accuracy
  also varies: `q0 ~ LogUniform(0.002, 0.2)`, drawn independently of the verifier.
- MIX keeps `rho ~ U(0.05, 1)`; RFP draws `p ~ LogUniform(0.005, 0.3)`.
- Rejection: `s_i ∈ [1e-4, 1-1e-4]`, and FPR in `[0.005, 0.5]`.
- Size and split as in E002: 900 structures per panel, 300 design / 600 test, sealed loader.
- One panel only (no P-mod/P-rare split, since `q0` varies within the panel).

**Targets.** As in E002: `D = 1 - J_G(∞)` (primary), stall (categorical). The exact `tau`-ODE
applies unchanged.

**Static-only reference model S.** Features from the audit (`m0 = B_gold` labeled `pi_0`
rollouts): `q0_hat` (gold accuracy), `FPR_hat`, verifier accept rate, and `log` transforms. Model:
cross-fitted (5-fold on the design split) monotone gradient boosting or, if that overfits,
logistic regression for stall and a rank-linear model for `D`; the model class is fixed on the
design split before the test split is opened.

**Arms (same accounting as E002 §3; every arm gets the same static audit).**

- `S`: static-only model.
- `S + G1`: `S` plus `C_hat^2` and `C_hat^2 / (1 - q0_hat)` (the predicted first-order FPR growth)
  from the same audit plus unlabeled rollouts.
- `S + P1`, `S + P2`, `S + P3`, `S + P4`: `S` plus that probe's score. P1 needs no gold of its own
  but inherits the shared audit cost.
- Oracle ceilings (committed first): `S*` (exact `q0`, FPR), `S* + C(0)`, `S* + FPR(t_p)` for
  `t_p ∈ {0.1, 1, 10}`.

**Primary estimand.**
`Delta_inc = [C-index(S + G1) - C-index(S)] - max_j [C-index(S + P_j) - C-index(S)]` at the E002
primary cells, with the same hierarchical bootstrap, one-sided tests and Holm correction over
cells × {C-index, AUROC}. "`S + G1` beats `S`" is reported but is not evidence, for the same
reason "G1 beats G0" is not evidence in E002.

**Success and abandonment.** Mirror E002 §8 with `S + arm` in place of each arm: success needs
the estimated metric, the test split, a Holm-significant `Delta_inc > 0` at a primary cell, and
non-domination of the `S + G1` frontier. Abandonment is triggered by the same three conditions.

**Registered-prediction candidates (to be frozen if E002c is ever registered).**

- The `S*` ceiling is well above 0.5, since FPR and `q0` now predict `D` (Prop. 9: the gold race
  depends on `q0` and the FP mass).
- `S* + C(0)` and `S* + FPR(t_p -> 0)` have equal ceilings (the §12 identity holds per structure).
- The incremental value of any `t = 0` diagnostic over `S*` is bounded by the within-(q0, FPR)
  variation of `eta`, which is what the matched E002 panel isolates.

**Known threats.**

- Confounding by scale: `C^2` grows with the FP mass, so part of `C_hat^2`'s value can be
  static information re-expressed. The `S + G1` vs `S` comparison controls for it only as well
  as `S` is specified; the cross-fitted flexible `S` is meant to absorb it.
- More model fitting means more researcher degrees of freedom; everything is fixed on the design
  split and committed before the test split is opened.
- Runtime: about 1.5× E002b (one panel, but a model fit per arm × cell × replication).

