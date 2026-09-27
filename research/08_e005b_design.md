# 08 — E005b: small-transformer pre-training verifier audit (DESIGN ONLY, v1, 2026-09-27)

**Status: design / draft pre-registration. Not implemented, not run.**

- No transformer is trained and no GRPO run exists. No code for E005b exists.
- E005a has no training outcomes. §4.2 imports only E005a's **measurement conclusions**: the
  frozen estimator configuration, measurability, and the audit budget (§4.2.1). Nothing else in
  E005b is tuned on E005a data.
- It becomes binding only after collaborator approval **and** after the measurement gate G0
  (§4.2.2) is passed.

**Tags:**

| tag | meaning |
| --- | --- |
| [registered-on-approval] | becomes binding when the collaborator approves |
| [open] | the collaborator decides |
| [E005a] | imported from the frozen E005a result |

---

## 0. Core question and honest framing

**Question.** Before verifier-guided RL begins, can a small trusted (gold) audit of the update
the verifier induces predict — and diagnose — whether full GRPO training will SUCCEED, STALL or
DECLINE?

**Three ways to evaluate a verifier before training, compared at equal resource budgets:**

| arm | what it measures |
| --- | --- |
| **A. Static** | verifier accuracy, FPR, FNR and proxy score on gold-labelled samples of the initial policy |
| **B. Initial update geometry** | `A_0`, `alpha_0`, `C_0` with the E005a-selected estimator, in a declared measurement subspace |
| **C. Active short probe** | the same budget spent on a small controlled policy update with the verifier, then a gold audit of the updated copy |

**Every outcome is publishable.** The design is built so that each of these conclusions can be
reached and reported:

- "geometry is informative but too noisy";
- "active probes dominate";
- "geometry mainly diagnoses mechanism";
- "geometry fails to generalize".

The strongest claim ("a small trusted pre-training audit predicts and diagnoses
verifier-induced failure") is a hypothesis. It is tested only through the §7 decision tree.

**Prior evidence** (not re-litigated):

- **E004a (synthetic, held-out):**
  - static metrics were weak;
  - the `t = 0` update geometry carried most of the predictive and mechanistic signal;
  - the trajectory added ≈ 0.02 at an equal audit budget;
  - robust early warning was not established.
- **E002:** geometry did not beat a budget-matched **active** probe.
- **E005a:** see §4.2.1.

---

## 1. Neural policy and tasks

**Model.** A decoder-only transformer in PyTorch:

- character tokens;
- 2–4 layers, `d_model` 128–256, about 0.5–5M parameters;
- two sizes for RQ4: S ≈ 0.8M and M ≈ 3M [open: exact sizes].

**Tasks with exact gold reward** (two families required; a third optional):

| family | items | gold | difficulty control |
| --- | --- | --- | --- |
| **T1 arithmetic** | `a ⊕ b =` with ⊕ ∈ {+, −, ×}, 2–4 digit operands | exact integer match | operand length, carries |
| **T2 string transformation** | reverse / sort / dedupe short digit strings | exact match | length, duplicates |
| T3 (optional) | tiny stack-machine program execution | exact output | program length |

**Initial policies ("bases").**

- Supervised pretraining on a disjoint item distribution, stopped so that the initial gold
  accuracy is `J_G(0) ∈ [0.1, 0.6]` (room to improve or to decline).
- At least 3 bases per task (different pretraining seeds / step counts).
- The gold item pool is split into audit items, training items and evaluation items.

## 2. Planted verifier flaws (≥ 2 constructions per mechanism; doses for both fates)

| mechanism | construction 1 | construction 2 |
| --- | --- | --- |
| clean | exact gold | — |
| R random attenuation | symmetric reward flips `eps` | difficulty-dependent flips (asymmetric FPR/FNR) |
| X deletion | constant reward (0, 1 or 0.5) on an item subset | the verifier ignores one operand digit position |
| Y exploit / shortcut (aligned / inverted) | accept answers with the correct length **and** last digit (accessible) | accept answers containing a rare token pattern (conjunctive, low accessibility); inverted: correct answers are rejected with `fn ∈ [0.1, 0.4]` on the triggered items |
| B benign partial credit | credit for a correct prefix | credit for correct digits in place |
| D displacement / preference inversion | a fixed shortcut answer is accepted with probability above the model's solve rate on hard items | length-only acceptance on hard items |

- **Doses.**
  - At least 2 flaw strengths per construction.
  - They are chosen from a small design-split pilot that uses verifier properties only, never
    the outcomes of the audit arms.
- **Mechanism ≠ outcome.**
  - Within each major mechanism, both successful and failing runs are targeted.
  - Mechanism identity must not predict the outcome; this is checked on the design split.
- **Axis B** (preference inversion) is recorded for every verifier from its exact behaviour on
  the item pool, as in E004a.

## 3. Full training and outcomes

**Training.**

- GRPO: group size 8, 32 prompts/step, Adam, `T` fixed by the E004a rule (3 × median clean
  `t95` on the design bases).
- 3 seeds per (base, verifier).
- A clean-verifier run from the same base and seed is the counterfactual.

**Outcomes** (the E004a definitions, adapted):

- final gold accuracy on the held-out evaluation items;
- the normalized gold shortfall `Dn`;
- SUCCESS / STALL / DECLINE / SLOW;
- proxy–gold divergence;
- exploit frequency (the rate of the planted shortcut in samples);
- mechanism-specific behaviour.

**Onset.** A persistence-based visible onset (the ratio stays > 0.1).

- The E004a post-hoc showed that the single-crossing onset is noise-dominated.
- This is a **new pre-specified definition** for E005b, not a rescue of E004a.

## 4. Pre-training audit arms and costs

Every arm reports `(B_roll, B_gold, B_bwd)`:

- `B_roll`: sampled responses;
- `B_gold`: gold labels;
- `B_bwd`: per-rollout backward passes (model gradients).

Comparisons use **Pareto frontiers** over matched budgets, not one arbitrary budget.

### 4.1 Static (A)

- Accuracy, FPR, FNR, mean verifier score and FP mass on `B_gold` gold-labelled samples of the
  initial policy.
- `B_bwd = 0`.

### 4.2 Geometry (B) [E005a]

#### 4.2.1 What E005a fixes

- **Estimator:** E3, the delete-one-group jackknife of the paired audit plug-in
  (`configs/e005/estimator_frozen.json`, sha256 `3b20ff3d…`). It is used as frozen: the
  interval, the null test and the `A ≈ 0` rule.
- **E005a facts about the measurement** (registry "E005a — results record", test split, M_I,
  `m = 8`):
  - **Bias is solved.** E3 removes the structure-dependent bias of `C^2`:
    - SDB 0.11–0.13 at every `N`, vs 0.76–1.45 for the legacy plug-in;
    - calibration slope ≈ 1.0;
    - coverage ≈ 0.96.
  - **Variance is the binding constraint.**
    - The null 95th percentile of `C^2` falls as `N^-1.04`, so the detectable `C` falls as
      `N^-1/2`. At `N = 1024` the floor is ≈ 1.3 `C_ref`.
    - The floor **grows with the number of gradient dimensions**: 6.4× from `d = 8` to
      `d = 64` at `N = 1024`.
    - Power at the planted 0.2 `C_ref` dose is ≈ 0 for every `N ≤ 1024`. The naive
      extrapolation is ≈ 37k gold labels.
  - **The null test is conservative.** The registered jackknife-Wald null test gives FPR
    ≈ 0.001–0.01 at `C = 0`, because the jackknife SE is ≈ √2 × the Monte Carlo SD (the
    degenerate-U-statistic effect; design doc §10).
  - **No practical budget.** There is no practical audit budget ≤ 1024. E005a's registered
    recommendation is **REVISE MEASUREMENT THEORY BEFORE NEURAL EXPERIMENTS**.

#### 4.2.2 Measurement gate G0 [registered-on-approval]

A transformer has 10^5–10^7 parameters. By the E005a dimension scaling, the full-parameter
reward-level `C` is **not measurable** at any affordable gold budget. Geometry is therefore
measured only in a **pre-declared low-dimensional subspace** `S` of dimension `k`. Candidates
[open]:

- **S1:** the gradients of the unembedding (output) layer only;
- **S2:** a fixed Gaussian random projection of the full gradient to `k ∈ {32, 128}`
  (seeded, identical for every verifier);
- **S3:** the top-`k` eigenspace of the empirical Fisher of the initial policy, estimated from
  unlabeled rollouts (no gold).

**G0.** Before any E005b training, a revised measurement calibration (E005a-R; not yet designed)
must show, on neural-policy calibration points with exact oracle geometry in `S` (exact gold,
enumerable item pool), that:

- structure-dependent bias is removed (`SDB ≤ 0.5`);
- a null test is calibrated (FPR in [0.02, 0.10] for ≥ 90% of nulls);
- `C` at the scale of the planted doses is detectable (power ≥ 0.8) at some `N ≤ 1024`.

If G0 fails for every declared `S`:

- the geometry arm is reported as **"not measurable at the intended scale"**;
- E005b proceeds as **static vs active probe only**, if the collaborator approves.

#### 4.2.3 Geometry arm (if G0 passes)

- **Primary:** E3 in the G0-selected subspace and metric, at the G0 minimum practical budget
  and at 2× that budget.
- **Features:** the estimate, its SE, and an "above the null floor" indicator. A point estimate
  below the floor is not interpreted as `C ≈ 0`.
- **Legacy baselines:**
  - the E002/E004 plug-in (E0);
  - the E004 update-level plug-in (GRPO-normalized, Adam `t = 0` metric). This links to the
    E004a evidence.
- **Costs:**
  - `B_gold = N`;
  - `B_roll = 2N` (N audit + N unlabeled for the metric);
  - `B_bwd = 2N` per-rollout backward passes;
  - memory `n × k` per estimator.

### 4.3 Active short probe (C)

- `k_steps ∈ {1, 4, 16}` GRPO steps with the verifier, on a copy of the base.
- Then a gold audit of the updated copy: `ΔJ_G`, `ΔJ_V`, `ΔFPR` and the shortcut rate.
- **Costs:**
  - `B_roll = k_steps × 256 + N_audit`;
  - `B_gold = N_audit`;
  - `B_bwd = k_steps × 256`.
- It is budget-matched to B by choosing `(k_steps, N_audit)` on the same cost grid.

### 4.4 Controls and ceilings

- Shuffled-verifier-label geometry (a null).
- Dimension-matched random features.
- The mechanism-type oracle and the mechanism × Axis-B oracle (interpretation only; never
  eligible).
- Higher feature dimensionality must not explain differences: every arm uses the same
  low-capacity models plus a dimension-matched noise arm (E004a practice).

## 5. Research questions

- **RQ1:** does initial geometry predict the final true performance beyond static metrics?
- **RQ2:** at equal budgets, how does initial geometry compare with an active short probe?
  (frontiers)
- **RQ3:** does geometry carry mechanism information that the probe does not? (mechanism
  macro-F1; within-mechanism outcome prediction)
- **RQ4:** does any signal generalize across verifier construction, task, model seed and model
  size?
- **RQ5** (secondary, long-term): can the audit support an intervention — reject or switch the
  verifier, mix in gold reward, reduce the verifier weight, or change the training rule?

## 6. Splits, predictors, inference

**Splits.**

- Design / test by (base checkpoint × verifier instance).
- The test split is sealed behind an approval file.
- **Generalization:**
  - leave-one-construction-out;
  - leave-one-task-out;
  - held-out seeds;
  - model size S → M.

**Predictors.**

- Ridge (`Dn`), L2 logistic (failure) and multinomial logistic (mechanism), with grouped CV on
  the design split only.
- Secondary: GBM (depth 2).
- E004a practice: median imputation, standardization, penalty chosen by grouped inner CV.

**Inference.**

- A hierarchical bootstrap (base → verifier → seed), with paired arms.
- One-sided p-values, Holm within families.
- Margins `δ_out = 0.02` and `δ_mech = 0.03`; ties within ±0.01.
- No point-estimate dominance rules.

## 7. Decision tree (frozen before any E005b outcome exists)

Exactly one primary label:

| label | condition |
| --- | --- |
| **G0-fail — geometry not measurable** | G0 fails for every declared subspace (reported before training; A vs C only) |
| **G1 — geometry adds over static** | RQ1 beats (both `Dn` C-index and failure AUROC) |
| **P — probes dominate** | on the matched-budget frontier the probe is ≥ geometry everywhere and beats it somewhere (lower bound > 0) |
| **N — geometry non-inferior to probes** | geometry ≥ probe − 0.01 at every matched budget |
| **M — geometry mainly diagnoses mechanism** | RQ3 beats while RQ1 fails |
| **F — no generalization** | whichever of the above holds within split, but the RQ4 criteria fail |

Several labels may be reported jointly (e.g. "P + M"). The primary claim uses the first
applicable row.

## 8. Compute estimate (FLOP-based; not measured — no model has been trained)

**Per GRPO step:**

- 32 prompts × 8 samples × about 20 tokens ≈ 5k tokens;
- training FLOPs ≈ `6 P × tokens` ≈ 3e10 at `P = 1M`;
- autoregressive sampling ≈ 20 decode steps with a KV cache.
- Small models on CPU are overhead-bound, so the estimate is ≈ 0.1–0.5 s/step on this machine
  (Apple silicon, 10 cores, PyTorch CPU).

**Per run:** `T` ≈ 1000–3000 steps → about 2–25 minutes.

**Runs:**

- 2 tasks × 3 bases × 25 verifiers (1 clean + 6 mechanisms × 2 constructions × 2 doses) × 3
  seeds = 450 runs;
- plus 25% for the size-M and generalization subsets;
- ≈ 560 runs → **≈ 20–230 CPU-hours**, i.e. ≈ 3–30 wall-clock hours at 8 parallel processes.

**Audits:**

- geometry: `2N ≤ 2048` per-rollout gradients per verifier, projected to `k` dimensions;
- probes: at most 16 GRPO steps per verifier.
- Both are small compared with training.

**G0 (E005a-R)** needs exact oracle geometry for neural policies. That means an enumerable
answer space per item (e.g. restricting the calibration items to short answers with a
truncated support). **Its cost is not yet estimated** [open].

**The first E005b action after approval** is a timing pilot (100 steps, no outcome analysis).

## 9. Proposed E005a-R: measurement revision (for collaborator decision; not designed in detail)

E005a's registered recommendation is to revise measurement theory before any neural experiment.
The revision candidates follow directly from the E005a record. None is tuned on E005a test data,
and every one needs **fresh calibration seeds and panels**, because the E005a test split is spent.

1. **Boundary-correct null test.**
   - Use the degenerate second-order null distribution:
     - jackknife SE / √2 at the boundary; or
     - a weighted-χ² approximation from `Σ_hat_δ`; or
     - a group sign-flip calibration.
   - Target: FPR within [0.02, 0.10].
2. **Low-dimensional measurement subspaces** (S1–S3 of §4.2.2).
   - Register how the floor scales with `k`, and check that the planted doses remain visible
     after projection. The projection can shrink the signal as well as the noise.
3. **Asymmetric budgets.**
   - Verifier-scored rollouts are cheap; gold labels are not.
   - Calibrate designs with `B_roll ≫ B_gold`, e.g. estimating the verifier side from a large
     unlabeled batch and using gold only for the gold side. Then compare their Pareto frontiers
     with the paired design.
   - E005a tied `B_roll` to `B_gold`.
4. **Target scale (collaborator decision).**
   - Whether "80% power at 0.2 `C_ref`" is the right practical target should be decided from the
     outcome-relevant `C` scale in the E004a evidence, registered **before** any new data.
   - It must not be relaxed after seeing E005a-R results.
