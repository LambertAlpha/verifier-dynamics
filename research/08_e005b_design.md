# 08 — E005b: small-transformer pre-training verifier audit (DESIGN ONLY, v2, 2026-09-28)

**Status: design / draft pre-registration. Not implemented, not run.**

- No transformer is trained, no GRPO run exists, and there is no E005b code.
- **v2** follows the final measurement round, E005a-R. Its registered label is
  **B — MECHANISTICALLY VALID, NOT PRACTICALLY MEASURABLE** (registry "E005a-R — results record").
- Per the E005a-R decision rule, **geometry is not a primary practical diagnostic in E005b.** The
  primary comparison is **static metrics vs passive early trajectory vs budget-matched active
  probe**. Geometry is kept only as a mechanistic / oracle analysis arm (§4.4).
- E005a and E005a-R have no training outcomes. Only their measurement conclusions are imported.
- It becomes binding only after collaborator approval. v1 (geometry as a primary arm, gated on G0)
  is superseded; see git history.

**Tags:**

| tag | meaning |
| --- | --- |
| [registered-on-approval] | becomes binding when the collaborator approves |
| [open] | the collaborator decides |
| [E005a-R] | imported from the frozen E005a-R result |

---

## 0. Core question and honest framing

**Question.** Before or early in verifier-guided RL, can a **budget-matched, practically
measurable** check predict whether full GRPO training will SUCCEED, STALL or DECLINE? And which
kind of check?

**Primary arms** (compared on matched-cost Pareto frontiers):

| arm | what it measures | when |
| --- | --- | --- |
| **A. Static** | verifier accuracy, FPR, FNR, proxy score and FP mass on gold-labelled samples of the initial policy | before training |
| **B. Passive early trajectory** | the first `h` steps of the actual training run: the free proxy-reward curve, plus a few small gold audits (gold accuracy, proxy–gold gap, planted-shortcut rate, entropy) | during training, no intervention |
| **C. Active short probe** | a copy of the base takes `k` GRPO steps with the verifier, then a gold audit of the copy (`ΔJ_G`, `ΔJ_V`, `ΔFPR`, shortcut rate) | before training |

**Mechanistic arm (not in the practical comparison):**

| arm | what it measures |
| --- | --- |
| **M. Oracle / large-budget geometry** | reward-level functional geometry (the E005a-R R4 form) with a gold budget far above the practical ones, used only to explain mechanism and outcome variation |

**Every outcome is publishable:** "static suffices", "the early trajectory dominates", "active
probes dominate", "nothing predicts", "geometry explains mechanism but not outcome", "no
generalization".

**Prior evidence** (not re-litigated):

- **E004a (synthetic, held-out):**
  - static metrics were weak;
  - `t = 0` update geometry carried most of the signal, but it needed a gold audit and was
    dominated by a structure-dependent noise floor;
  - the passive early observables (L2) were weak for early warning (NYV AUROC lo95 0.642).
- **E002:** geometry lost to budget-matched active probes.
- **E005a:** full-space `C` is not practically measurable at `N ≤ 1024`.
- **E005a-R** [E005a-R]:
  - behavior-defined functional geometry removes the nuisance-dimension barrier: calibrated,
    structure-free, flat in `d`;
  - but even the oracle subspace detects the historical **median** verifier error with only
    0.55 power at `N = 1024`; the frozen functional representation reaches 0.43, Spearman 0.68;
  - it detects the historical Q75 error with 0.83–0.89 power at `N = 1024`.

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

## 4. Arms and costs

Every arm reports `(B_roll, B_gold, B_bwd)`, with `B_bwd` = per-rollout backward passes.
Comparisons are made on **Pareto frontiers** over a shared cost grid, not at one budget.

### 4.1 Static (A)

- Accuracy, FPR, FNR, mean verifier score and FP mass on `B_gold` gold-labelled samples of the
  initial policy.
- `B_bwd = 0`.

### 4.2 Passive early trajectory (B) [registered-on-approval]

- The actual training run (verifier reward, GRPO) is observed for its first `h ∈ {1, 2, 5}%` of
  `T` steps, with no intervention.
- **Free features:**
  - the proxy-reward curve;
  - the policy entropy;
  - the response-length distribution.
- **Paid features:** gold audits of `N_a ∈ {64, 128, 256}` rollouts at `{2, 4}` checkpoints, giving
  gold accuracy, the proxy–gold gap and the planted-shortcut rate.
- **Onset:** the persistence onset (§3).
- **Costs:** `B_gold = checkpoints × N_a`; the training rollouts are free because they are part of
  the run.
- **Note:** B does not decide before training; it decides early. Its value is judged on the
  same frontier, with the lead time reported.

### 4.3 Active short probe (C)

- `k_steps ∈ {1, 4, 16}` GRPO steps with the verifier on a copy of the base.
- Then a gold audit of the copy: `ΔJ_G`, `ΔJ_V`, `ΔFPR` and the shortcut rate.
- **Costs:** `B_roll = k_steps × 256 + N_audit`, `B_gold = N_audit`, `B_bwd = k_steps × 256`.

### 4.4 Mechanistic geometry arm (M) [E005a-R; interpretation only, never in the frontier]

**Representation** (the E005a-R frozen R4 form, adapted):

- **probe:** the answer-category probabilities `p_θ(category | x)` on a fixed probe prompt set
  drawn from the training distribution before any run;
- **metric:** Fisher–Rao;
- **Jacobian:** exact JVPs (one backward pass per probe output).
- The categories must be behavior-defined (answer correct / specific wrong answer / planted
  shortcut form / other). E005a-R showed that surface-form outputs reintroduce nuisance error
  (R4-surf: FPR 0.22 at nuisance-only nulls).

**Estimation:**

- E3 with the group sign-flip test (E005a-R §2.6);
- reward level only;
- gold budget `N_M ∈ {4096, 8192}`, or exact enumeration where the answer space permits. This is
  above the practical range on purpose.

**Use:**

- RQ3 (mechanism) and the explanation of outcome variation within mechanism;
- reported **alongside** the practical arms, never as a practical predictor.

**Update level.** If an update-level (GRPO-normalized, Adam) version is wanted, the §2.7
translation of `09_e005ar_design.md` must first be calibrated. Until then, **no update-level
geometry claim is made.**

### 4.5 Controls and ceilings

- Shuffled-verifier-label versions of every arm (nulls).
- Dimension-matched random features.
- The mechanism-type oracle and the mechanism × Axis-B oracle (interpretation only).
- Equal low-capacity predictors for every arm, plus a dimension-matched noise arm (E004a
  practice).

## 5. Research questions

- **RQ1 (primary):** at matched budgets, which of static (A), passive early trajectory (B) and
  active probe (C) best predicts the final true performance (`Dn` C-index, failure AUROC)?
  Frontiers, with pairwise paired tests.
- **RQ2:** does the early trajectory add over static? At which `h`, with what lead time?
- **RQ3 (mechanistic):** does the oracle geometry arm M explain mechanism (macro-F1) and
  within-mechanism outcome variation beyond A–C?
  - Interpretation only.
  - The E004a type oracles bound it from above.
- **RQ4:** does any signal generalize across verifier construction, task, model seed and model
  size?
- **RQ5** (secondary): can the winning check support an intervention — reject or switch the
  verifier, mix in gold reward, reduce the verifier weight?

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

Exactly one primary label, for RQ1:

| label | condition |
| --- | --- |
| **C-dom — active probes dominate** | on the matched frontier, C ≥ A and B everywhere and beats both somewhere (lower bound > 0) |
| **B-dom — early trajectory dominates** | the same, with B |
| **A-suff — static suffices** | neither B nor C beats A anywhere by more than `δ_out = 0.02` |
| **Mixed** | the frontiers cross (reported by budget range) |
| **None** | no arm beats its shuffled-label null |

**Secondary labels:**

- **M-explains:** RQ3 beats A–C for mechanism.
- **F:** the RQ4 criteria fail.

## 8. Compute estimate (FLOP-based; not measured — no model has been trained)

- **Per GRPO step:** 32 prompts × 8 samples × ≈ 20 tokens ≈ 5k tokens; ≈ 3e10 training FLOPs at
  `P = 1M`; ≈ 0.1–0.5 s/step on CPU (overhead-bound).
- **Per run:** `T` ≈ 1000–3000 steps → 2–25 min.
- **Runs:** 2 tasks × 3 bases × 25 verifiers × 3 seeds = 450, plus 25% for size and
  generalization → ≈ 560 runs → **≈ 20–230 CPU-hours**.
- **Machines:** two are available (this one and a 12-core Mac mini). That is ≈ 2–15 wall-clock
  hours at 8 + 11 processes.
- **Arm B** costs only its gold audits (≤ 1024 labels per run).
- **Arm C:** ≤ 16 probe steps per verifier.
- **Arm M:** `N_M ≤ 8192` per-rollout gradients per verifier, projected to the probe outputs
  (tens of dimensions), plus `q_f` backward passes. That is small compared with training.
- **The first E005b action after approval** is a timing pilot (100 steps, no outcome analysis).

## 9. Measurement history (closed)

**E005a:**

- full-space `C` is not practically measurable;
- bias fixed by E3;
- the variance barrier remains;
- the null-test lesson (sign-flip calibration).

**E005a-R** (final):

- the nuisance-dimension barrier is real and is removed by a behavior-defined functional
  representation;
- the within-behavior variance at the historical medium scale still exceeds what `N ≤ 1024`
  resolves, even for the oracle subspace;
- random projection hurts;
- PCA subspaces work only when behavior directions stand out spectrally, and joint subspaces
  import verifier-specific error.
- **Label B.** No further measurement rescue round exists.
