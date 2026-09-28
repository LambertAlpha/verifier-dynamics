# 10 — E005b-0: small-Transformer feasibility pilot (protocol v1, 2026-09-28)

**Status: pilot protocol.**

- Committed before any substantive training. Only the tested pipeline core and a device-timing
  check exist.
- This is a **feasibility pilot**, not a test of any E004/E005 hypothesis. E004a, E005a and
  E005a-R are closed.
- Engineering choices below are pilot settings. Justified changes are recorded as amendments;
  they are not treated as scientific hypotheses.
- Constants: `configs/e005b/pilot.toml`.

**Questions:**

1. Can a genuinely autoregressive Transformer learn a simple verifiable task?
2. Can sampled GRPO with Adam produce a stable, interpretable clean-reward run?
3. What does the pipeline cost?

**Out of scope:** `C`, geometry, outcome predictors, and the flawed-verifier matrix (§8 proposes
it; it is not run).

---

## 1. Task, format, splits

**Task:** two-digit addition `a + b`, with `a, b ∈ {0, …, 99}`.

**Tokens:** 15 characters — `<pad> <bos> <eos> 0–9 + =`.

- **Prompt:** `<bos> a a + b b =`. Operands are zero-padded, so every prompt is 7 tokens.
- **Answer:** the decimal sum without leading zeros, followed by `<eos>`; at most 4 generated
  tokens.

**Parsing** (`vdyn.e005b.task.parse_completion`):

- A completion is **valid** iff `<eos>` appears within the first 4 tokens and the text before it
  is 1–3 digits with no leading zero (except `0`).
- Tokens after `<eos>` are ignored.
- Invalid or truncated completions get reward 0 and are counted.

**Gold reward:** 1 iff valid and equal to `a + b`.

**Splits** (seed 20261301) are made by **unordered pair**, so `a+b` and `b+a` never cross splits:

| split | size | used for |
| --- | --- | --- |
| dev | 1000 | checkpoint selection and monitoring |
| test | 1000 | final evaluation only |
| train | ~8000 | SFT and RL prompts |

## 2. Model, SFT, base-checkpoint selection

**Model** (`vdyn.e005b.model`), a character-level decoder-only Transformer:

- 2 pre-LN blocks, `d = 128`, 4 heads, MLP 4×;
- learned positions (context 16); untied head;
- 402k parameters; init N(0, 0.02).
- **Why this size:** it is the brief's starting point, and sufficient for 2-digit addition.
  Reversed-digit output (which eases carries) was considered and rejected: it would change what
  "learning addition" means.

**SFT:**

- AdamW, lr 3e-4, betas (0.9, 0.999), weight decay 0;
- batch 64 train pairs (uniform, with replacement);
- loss: cross-entropy on the answer tokens and `<eos>` only;
- at most 6000 steps, stopping early when dev greedy accuracy ≥ 0.99;
- seed 20261302.

**Evaluation every 25 steps (dev):**

- greedy accuracy;
- **sampled accuracy** (temperature 1, one sample per item, a fixed evaluation seed);
- valid-output rate.

Every evaluated checkpoint is saved.

**Base selection rule** (dev only; fixed now):

- the **first** evaluated checkpoint with dev sampled accuracy ∈ **[0.20, 0.40]** and a dev valid
  rate ≥ 0.95;
- **fallback:** the checkpoint with dev sampled accuracy closest to 0.30 among those with valid
  ≥ 0.95.

**Rationale.** At `p ≈ 0.3`, a group of 8 has mixed rewards with probability
`1 − 0.7^8 − 0.3^8 ≈ 0.94`, so GRPO has signal and headroom. **The base is never reselected**
after any RL or flawed-verifier result. Its sha256 is recorded.

## 3. GRPO (clean verifier = gold)

| component | setting | standard? |
| --- | --- | --- |
| rollouts | `P = 32` train prompts per step (distinct within a step), `G = 8` samples each → 256 | standard group sampling |
| sampling | temperature 1.0, no top-k / top-p (exact policy log-probs), at most 4 tokens | standard |
| advantage | `(r − mean_g) / (std_g + 1e-4)`, `std` with ddof 1 | standard (DeepSeekMath §4.1; TRL `scale_rewards="group"`) |
| zero-variance groups | advantage exactly 0 (no gradient); frequency logged | standard consequence |
| loss | clipped surrogate `min(ρA, clip(ρ, 0.8, 1.2)A)`, **summed over completion tokens and divided by the batch's completion-token count** | token-level aggregation as in DAPO / TRL `dapo`; DeepSeekMath uses per-sequence `1/\|o_i\|`. Lengths are 2–4 tokens, so the difference is small |
| updates per rollout batch | 1 (`μ = 1`), so `ρ = 1` at the update and **the clip is inactive**; the update is on-policy REINFORCE with group-normalized advantages | DeepSeekMath §4.2 ("a single update following each exploration stage") |
| KL | **β = 0** (no reference penalty); the k3 KL to the base is logged as a diagnostic | deliberate simplification; TRL v1.0 default and DAPO. DeepSeekMath used β = 0.04 |
| optimizer | **fresh** Adam (explicitly initialized at RL start), lr 1e-4, betas (0.9, 0.999), eps 1e-8, no weight decay; gradient-norm clip 1.0 | standard (TRL `max_grad_norm = 1.0`) |
| steps | 500 per run | pilot bound |
| seeds | RL seeds 1, 2, 3 (clean only): seed 1 is primary; seeds 2 and 3 size the seed variance for §8 | pilot |
| device | **CPU, 4 threads** | measured below |

**Device choice** (`results/E005b0-device`, Mac mini M4 Pro):

- MPS was 3–5× faster (one GRPO iteration: 10 ms on MPS vs 35 ms on CPU with 4 threads). It is
  numerically compatible (max logit difference 5e-7), but **not bitwise deterministic** across
  repeated runs; CPU is.
- At this scale both cost under a minute per run, so bitwise reproducibility wins.

## 4. Evaluation and logging

**Every GRPO step** (JSONL):

- mean verifier reward (= gold accuracy on the batch for the clean verifier);
- valid rate; mixed-group and zero-variance-group fractions;
- mean completion length;
- loss, pre-clip gradient norm, token entropy, k3 KL to the base;
- NaN/Inf flags;
- wall time for generation, scoring, forward+backward and update;
- peak RSS.

**Every 25 steps (dev):** greedy and sampled accuracy, valid rate, evaluation time.

**Test** is evaluated once, for the base and for each final RL checkpoint.

**Checkpoints:** every 100 steps and at the end (model, optimizer, RNG), with sha256.

**Reproducibility:** RL seed 1 is rerun from the same base. The final sha256 must match.

## 5. Limits, stop conditions, outputs

**Limits** (the run stops and is reported if exceeded):

- SFT ≤ 20 min;
- each GRPO run ≤ 20 min;
- peak RSS ≤ 4 GB;
- host: `mac-mini-remote`, one run at a time.

**Stop conditions:**

- any NaN/Inf in the loss or gradients stops the run (fail closed);
- a failed reproducibility check stops further runs.

**If clean GRPO does not raise dev sampled accuracy by ≥ 0.05 within 500 steps:**

1. check implementation (tests), reward sparsity (the mixed-group rate) and base competence;
2. then **one** documented diagnostic run at lr 3e-4.

There is no open-ended search.

**Outputs:**

- the SFT curve;
- the selected base and its hash;
- clean GRPO curves (3 seeds) and the reproducibility check;
- the timing breakdown;
- the verifier rules' initial error rates (§8);
- a cost estimate for the next 12 runs;
- a report.

## 6. Tests (before any run)

- **Format and parsing:** leading zeros, truncation, non-digits, text after `<eos>`.
- **Splits:** disjoint by unordered pair.
- **Causality** and deterministic initialization.
- **Sampling:**
  - post-`<eos>` masking and padding;
  - sampled log-probs equal teacher-forced ones;
  - first-token frequencies match the softmax.
- **Greedy** decoding follows the argmax chain.
- **Advantages:**
  - match a NumPy reference;
  - zero-variance groups give 0.
- **Loss:**
  - the on-policy loss gradient equals the token-averaged REINFORCE gradient;
  - clipping removes the gradient outside `[1 − ε, 1 + ε]`;
  - with a mean baseline, the loss gradient's expectation equals `−(G−1)/G ∇J` exactly on an
    enumerable softmax policy.
- **SFT loss** equals a hand-written cross-entropy on the answer tokens.
- **Checkpoint resume** is bit-identical.
- **Before training:** the verifier rules (§8) against explicit examples, and a GRPO step with
  all-equal rewards leaves the parameters unchanged apart from the optimizer's zero-gradient
  update.

## 7. Challenged assumptions (recorded before running)

1. **B and C of the E005b v2 draft are the same protocol.**
   - The "passive early trajectory" (B) and the "active short probe" (C) use the same base,
     verifier, optimizer and update rule.
   - A copy trained for `k` steps is the prefix of the run itself.
   - They differ only if the probe applies a **different intervention** (e.g. a different
     learning rate, gold-mixed reward, targeted prompts, or a clean-reward counterfactual copy).
   - Until such an intervention is specified, they are treated as **one** diagnostic: observing
     the training prefix.
2. **"GRPO clipping" does nothing at `μ = 1`.** The pilot is REINFORCE with group-normalized
   advantages plus gradient-norm clipping. It is named as such in results.
3. **Std normalization is not neutral.** It rescales each prompt's advantage by its reward SD,
   so low-variance prompts get larger per-sample weight. That is a known bias (the Dr. GRPO
   critique). It is kept because it is the standard, and noted.
4. **"Constant reward on a subset" removes direct gradient on that subset, but not learning.**
   Shared parameters still move. Accuracy on the subset is measured separately (§8).

## 8. Next matrix — PROPOSED, NOT RUN

**4 verifiers × 3 RL seeds**, all starting from the **same base checkpoint** with a **fresh
Adam**. Same GRPO settings. Steps fixed at the length that the clean pilot shows is needed to
plateau.

| verifier | rule |
| --- | --- |
| V0 clean | `V = G` |
| V1 random flips | `V = G XOR Bern(0.2)`, independently per sample (both directions) |
| V2 deleted feedback | on a fixed 25% of training prompts (selected by a seeded hash of the unordered pair), `V = 1` whatever the answer; elsewhere `V = G` |
| V3 exploitable false positive | `V = 1` if the answer is correct **or** is a valid number ending in digit `0`; else 0. Always answering `0` earns full verifier reward |

**Initial error rates** (FPR `P(V=1 \| G=0)`, FNR `P(V=0 \| G=1)`, mean `V`) are measured on the
selected base's samples (train prompts, temperature 1) before any flawed-verifier training.
Whether the exploit is discovered is **not assumed**.

**Cost reporting** separates:

- **(a) the cost of training up to the diagnostic point:** rollouts, backward passes, gold labels
  used by the verifier (none for V0–V3, since the verifiers are programmatic);
- **(b) the additional diagnostic cost:** gold audits, dev/test evaluations.

---

## 9. Results (2026-09-28; the pilot is complete and stopped here)

**Runs:**

- all on `mac-mini-remote` (Lamberts-Mac-mini.local, M4 Pro), CPU, 4 threads;
- Python 3.12.13, torch 2.14.0;
- every run from a clean commit (`meta.json`);
- seeds and configuration as in `configs/e005b/pilot.toml`.

**Q1 — can the Transformer learn the task? Yes.**

- SFT reaches dev greedy 0.99 (unseen operand pairs) at step 1700, in 20 s.
- **Base** (rule: the first checkpoint in [0.20, 0.40]):
  - step 375; dev sampled 0.214, greedy 0.285, valid 0.97; sha256 `8812b53a…`;
  - test: greedy 0.281, sampled 0.185;
  - train (1000 items): greedy 0.469, sampled 0.296.

**Q2 — is clean GRPO stable and interpretable? Yes, with one strong limitation.**

- **Stability and reproducibility:**
  - three seeds, 500 steps each;
  - no non-finite step;
  - seed 1 rerun **bit-identical** (final sha256 `1047aec2…`);
  - the seeds agree closely.
- **Learning:**
  - dev sampled 0.171 → 0.343–0.374;
  - dev greedy 0.285 → 0.42–0.44;
  - test sampled 0.185 → 0.365–0.389, test greedy 0.281 → 0.41–0.46;
  - batch gold 0.21 → 0.37–0.41.
- **Diagnostics:**
  - entropy 0.77 → 0.37–0.45; the k3 KL to the base grows to ≈ 0.25 (spikes up to 0.9);
  - response length is stable (3.4–3.6 tokens);
  - valid rate 0.99.
- **Limitation — reward sparsity, not instability** (post-hoc diagnostic
  `results/E005b0-posthoc-sparsity`):
  - Greedy accuracy plateaus near 0.44 after ≈ 100 steps.
  - 51% of the prompts have a sum ≥ 100. There the base's greedy accuracy is 0.09, and 63% of
    its groups are all wrong.
  - After 500 steps that class is unchanged (greedy 0.09–0.10; all-wrong groups 67–74%).
  - The gains are entirely in the no-carry class (0.56 → 0.85–0.87) and the units-carry class
    (0.53 → 0.66–0.77).
  - **GRPO sharpens behaviour the base can already sample; it does not bootstrap the
    three-digit answers.** Mixed-reward groups fall from 0.60 to 0.44–0.52, so about half of the
    rollouts carry no gradient.
- **Gradient-norm clipping (max 1.0, the TRL default) binds at every step** (median pre-clip
  norm ≈ 5). With Adam, the run is effectively Adam on unit-norm gradients.

**Q3 — cost** (500 steps ≈ 26 s per run):

| phase | ms per step | seconds per 500-step run | share |
| --- | --- | --- | --- |
| generation | 19.6 | 9.8 | 38% |
| scoring | 0.3 | 0.17 | 1% |
| pre-update diagnostics (entropy, KL) | 11.0 | 5.5 | 21% |
| forward + backward | 14.4 | 7.2 | 28% |
| optimizer | 1.5 | 0.7 | 3% |
| dev evaluation (21 × 2000 generations) | — | 2.9 | 11% |

- Peak RSS is 0.73–0.80 GB.
- SFT to 0.99: 20 s (train 10 s, evaluation 9 s).
- Audit: 16,000 samples in 1.2 s.

**Deviations and challenged assumptions:**

1. **The pre-stated 94% mixed-group rate was wrong** (0.60). It assumed a homogeneous `p`;
   difficulty differs sharply by carry class.
2. **Winner's curse in base selection.** The base's dev sampled accuracy was 0.214 under the SFT
   evaluation seed and 0.171 under the GRPO evaluation seed. The rule selects on one noisy
   sample per item.
3. **Orchestration bug (no data impact):** a glob moved a committed smoke record between runs.
   The dirty-tree check stopped the sequence (fail closed). The remaining runs were relaunched
   on a clean tree.
4. **Smoke mode** takes the last checkpoint, because 50 SFT steps never reach the valid-rate
   threshold. Real runs use the registered rule.
5. **Rule 7.1** (early-trajectory ≡ copy probe) and **rule 7.2** (the clip is inactive at
   `μ = 1`) hold as stated.

**Initial error rates of the proposed verifiers at the base** (2000 train prompts × 8 samples;
`results/E005b0-verifier-audit`):

| rule | FPR | FNR | mean V (gold 0.198) | mixed groups |
| --- | --- | --- | --- | --- |
| V0 clean | 0 | 0 | 0.198 | 0.62 |
| V1 flip 0.2 | 0.202 | 0.200 | 0.321 | 0.92 |
| V2 deleted 25% (constant 1) | 0.247 | 0 | 0.396 | 0.47 |
| V3 correct-or-ends-in-0 | 0.078 | 0 | 0.261 | 0.80 |

- For V3, 7.6% of base outputs end in 0, and "0" itself is essentially never produced (6e-5).
- V3's false positives fall where the base is weakest: e.g. "100"/"110" on sums ≥ 100. The
  exploit is therefore **reachable**, but its discovery is not assumed.
- V1 raises the mixed-group rate: flips create variance even on all-wrong prompts.

---

## 10. Amendment — baseline calibration round (approved scope; frozen before any new run, 2026-09-28)

Scope: clean-verifier calibration only. The flawed-verifier matrix is **not** run. The original
pilot (runs, base checkpoint `8812b53a…`, `configs/e005b/pilot.toml`, `configs/e005b/base_checkpoint.json`,
`experiments/e005b/grpo_pilot.py`) is preserved unchanged.

### A1. Narrowed interpretation of §9 (replaces the §9 wording "GRPO sharpens … does not bootstrap")

"Under this base (step 375), this configuration and a 500-step horizon, clean GRPO did not improve
the three-digit-answer category (sum ≥ 100)." Reward sparsity (63–74% all-wrong groups in that
category) is a **supported hypothesis**, not an isolated causal result; the pilot did not
manipulate sparsity. No claim is made that GRPO generally cannot acquire new capabilities.

### A2. Coverage-aware base selection (DEV only)

- **Candidates:** the 68 checkpoints of the original SFT run
  (`results/E005b0-pretrain/20260928T042420Z_5549b29`, steps 25–1700, hashes in its `sft_log`),
  in step order. No new SFT, so old and new bases differ only in SFT steps.
- **Categories (predefined; the structure in which the pilot's gap appeared):** *no carry*
  (units digits sum < 10 and a + b < 100; dev n = 297), *units carry* (units ≥ 10, a + b < 100;
  n = 201), *three-digit* (a + b ≥ 100; n = 503).
- **Estimates:** dev sampled accuracy with **4 samples per item** (temperature 1), selection seed
  20261310; greedy accuracy reported. SE ≤ 0.015 per category (sampling), ≤ 0.03 including item
  clustering.
- **Primary rule:** the first checkpoint with (i) sampled accuracy ≥ 0.20 in **every** category,
  (ii) aggregate sampled accuracy ≤ 0.60, (iii) valid-output rate ≥ 0.95.
- **Confirmation:** the candidate is re-evaluated with an **independent seed** (20261311, 4
  samples per item) and must satisfy the **same** thresholds (no tolerance; the confirmation is
  the unbiased estimate after selection). If it fails, the scan continues with the next
  checkpoint that satisfies the rule under the selection seed.
- **Challenge of the proposed defaults (kept, with reasons):** 0.20 per category gives a
  category-average probability ≥ 0.83 that a group of 8 contains a success; 0.60 aggregate keeps
  ≥ 40 points of aggregate headroom; no per-category upper bound is imposed, because the easy
  categories necessarily lead and a common interval would likely be infeasible. Three categories,
  not finer (e.g. splitting three-digit by units carry), to keep n ≥ 200 per category.
- **Single fallback (pre-specified; no further relaxation):** if no checkpoint is confirmed, the
  same procedure with ≥ 0.15 per category and ≤ 0.65 aggregate (valid ≥ 0.95). If still none:
  **STOP** — no new base; the clip comparison runs on the old base only and this is reported.
- The **old base (step 375) is retained as the low-coverage control.** New pointer:
  `configs/e005b/base_checkpoint_v2.json`; the old pointer is unchanged.

### A3. Clean-only 2×2 and confirmation

- Cells: {old base, new base} × gradient-norm clip {1.0, 10.0}; one common RL seed (1);
  **T = 1000** steps; every other setting as §3 (P = 32, G = 8, group-std advantages,
  token-mean loss, μ = 1, β = 0, fresh Adam lr 1e-4).
- Dev evaluation every 25 steps: 4 samples per item (fixed evaluation seed 20261312, the same for
  every run and step) + greedy, overall and per category. **The test split is not evaluated in this
  round**; earlier test numbers remain exploratory.
- **Decision rules (dev only, frozen):**
  - *Unstable* run: any non-finite step, or final dev sampled accuracy (mean of the last 4
    evaluations, steps 925–1000) below the initial by more than 0.02, or batch valid rate over the
    last 50 steps < 0.90.
  - *Clip choice* (on the new base): if both stable and the final dev sampled accuracies differ by
    < 0.05 (≈ the pilot's seed spread), they are **practically tied → keep clip 1.0**; otherwise the
    higher; an unstable setting is excluded; both unstable → stop and report.
  - *Base adoption:* the new base is adopted as the baseline if its run under the chosen clip is
    stable and gains ≥ 0.05 in dev sampled accuracy; otherwise stop and report.
  - Base and clip effects are reported descriptively (means over the other factor; interaction).
- **Confirmation:** the selected baseline is run with RL seeds 2 and 3 (seed 1 is its 2×2 cell).
  Pass: all three stable and each gains ≥ 0.05; category-level changes reported.
- Selection never uses test data, future verifier outcomes, or flawed-verifier runs.

### A4. Logging

Per step: batch gold accuracy and mixed-group frequency, overall and per category; token entropy;
k3 KL to the base; pre-clip gradient norm; clipped (yes/no); **actual parameter-update norm
‖θ_{t+1} − θ_t‖** (clipping rescales the gradient; Adam's update length still varies with its
moment estimates, so the two are reported separately); response length; timing; memory.

### A5. Cost accounting

Per run: generated responses and completion tokens (training and evaluation separately), prompt
tokens, batched backward calls (and sequences per call), **gold-checker calls** (training scoring
and evaluation; automated gold is cheap, not absent), wall time by phase, peak RSS.

### A6. Integrity

Library changes are additive. Before the new runs, the original pilot seed-1 run is re-executed
with the original script; its final sha256 must equal `1047aec2…` (fail closed). New script:
`experiments/e005b/grpo_calib.py`; new config: `configs/e005b/calib.toml`.

## 11. Calibration round results (2026-09-28; stopped here, matrix not run)

All runs on `mac-mini-remote` (M4 Pro), CPU 4 threads, clean commits; dev only (no test data).

**Integrity (A6).** The original pilot seed-1 run re-executed with the original script after the
additive code changes: final sha256 `1047aec2…` — bit-identical (`results/E005b0-grpo-s1-integrity`).

**Base selection (A2; `results/E005b0-select-base/20260928T182536Z_60c0812`).** Primary rule met at
**SFT step 600** (first candidate tried; sha `74865408…`):

| base | dev sampled (4/item) | no carry | units carry | three-digit | valid | greedy |
| --- | --- | --- | --- | --- | --- | --- |
| new, selection seed | 0.380 | 0.560 | 0.458 | 0.244 | 0.99 | — |
| new, confirmation seed | 0.380 | 0.566 | 0.427 | 0.252 | 0.99 | 0.566 |
| old (step 375, control) | 0.195 | 0.356 | 0.303 | 0.057 | 0.97 | 0.285 |

**Clean 2×2 (seed 1, T = 1000; `results/E005b0-calib-report/20260928T183404Z_4e5b695`).** Final =
mean of the last 4 dev evaluations (sampled, 4/item):

| cell | overall init → final (gain) | no carry | units carry | three-digit | clipped | median ‖Δθ‖ |
| --- | --- | --- | --- | --- | --- | --- |
| old, clip 1 | 0.186 → 0.389 (+0.20) | 0.34 → 0.76 | 0.28 → 0.62 | **0.06 → 0.08** | 100% | 0.0131 |
| old, clip 10 | 0.186 → 0.556 (+0.37) | 0.34 → 0.75 | 0.28 → 0.63 | **0.06 → 0.41** | 12% | 0.0151 |
| new, clip 1 | 0.379 → 0.760 (+0.38) | 0.56 → 0.82 | 0.42 → 0.76 | 0.26 → 0.72 | 100% | 0.0129 |
| new, clip 10 | 0.379 → 0.757 (+0.38) | 0.56 → 0.82 | 0.42 → 0.81 | 0.26 → 0.70 | 35% | 0.0139 |

- **Frozen decisions:** new-base clip 1 vs 10 differ by 0.003 < 0.05 → *practically tied → keep
  clip 1.0*; new base adopted (stable, gain 0.38 ≥ 0.05). Descriptively, base effect on the
  final accuracy +0.29 (on the gain +0.09), clip effect on the gain +0.08, interaction −0.17: the
  clip mattered only with the old base.
- **Caution on the old-base clip effect:** one seed; the three-digit category stayed at ≈ 0.07
  until step ≈ 850 and then rose abruptly (0.07 → 0.41 by step 1000). This is a late
  discovery event, compatible with a systematic clip effect **or** with stochastic timing; the
  500-step pilot horizon would have missed it under either clip. Update norms were similar under
  both clips (clipping reweights steps; it does not fix the update length).
- **Confirmation (new base, clip 1, seeds 1–3):** all stable; gains +0.381 / +0.357 / +0.373;
  finals 0.760 / 0.736 / 0.752; three-digit 0.26 → 0.72 / 0.68 / 0.71; greedy 0.57 → 0.77–0.79.
  **Pass.** Mixed-reward groups fall from ≈ 0.73 to ≈ 0.30 (late training carries little
  signal); entropy 0.5 → 0.1; k3 KL to the base median ≈ 0.2–0.3 with spikes up to ≈ 5
  (rare-token estimates).

**Costs per 1000-step run (measured).** Wall 60 s: generation 19.5 s, scoring 0.3 s, pre-update
diagnostics (entropy, KL) 10.8 s, forward+backward 14.3 s, optimizer 2.0 s, dev evaluation 12.7 s;
peak RSS 1.23 GB. Training: 256,000 responses (≈ 0.89 M completion + 1.79 M prompt tokens),
1000 batched backward calls × 256 sequences, **256,000 gold-checker calls**. Evaluation: 41 dev
evaluations × 5,005 responses = 205,205 responses and **205,205 gold-checker calls**. Base
selection: 68 checkpoints × 5,005 responses + confirmation, 22 s.

**Verifier rules at the new base** (2000 train prompts × 8; `results/E005b0-verifier-audit/20260928T183547Z_bb03536`):

| rule | FPR | FNR | mean V (gold 0.381) | mixed groups |
| --- | --- | --- | --- | --- |
| V0 clean | 0 | 0 | 0.381 | 0.78 |
| V1 flip 0.2 | 0.201 | 0.203 | 0.428 | 0.95 |
| V2 deleted 25% | 0.248 | 0 | 0.534 | 0.59 |
| V3 correct or ends in 0 | 0.118 | 0 | 0.454 | 0.88 |

10.2% of base outputs end in 0; "0" itself 0.06%.

**Process notes.** A confirmation launch used a glob that would also have matched the committed
2×2 directories; it was stopped before the first run finished, the partial seed-2 directory
(aborted, no summary) was deleted, and the runs were relaunched with exact names. No committed
data was affected.

---

## 12. Exploratory verifier matrix — protocol (frozen before any matrix run, 2026-09-28)

**Question.** How do four reward rules change genuine learning (gold accuracy) and observable
behaviour, from one calibrated base? **Not** a matched-static-error-rate experiment, **not** a
failure-predictor evaluation, **not** evidence about other initial models. Exploratory; no
hypothesis tests or minimum-detectable-difference claims from three seeds.

**Fixed configuration (no recalibration).** Base `configs/e005b/base_checkpoint_v2.json` (SFT step
600, sha `74865408…`); every run starts from it with a **fresh Adam**; gradient clip 1.0; T = 1000;
RL seeds 1, 2, 3; P = 32, G = 8, temperature 1, group mean/std advantages, token-mean clipped loss,
μ = 1, β = 0, lr 1e-4 — identical to the calibrated clean baseline (§10–§11). CPU, 4 threads;
host `mac-mini-remote`; at most 2 concurrent runs; exact output paths.

**Reward rules (unchanged, `vdyn.e005b.verifiers`):** V0 clean `V = G`; V1 independent symmetric
flips with probability 0.2; V2 `V = 1` on the fixed prompt subset (`deleted_prompts(0.25, 20261305)`,
by unordered pair), `V = G` elsewhere; V3 `V = 1` if correct or a valid number ending in 0.
Anticipated structure (not hypotheses): V2's constant groups have exactly zero advantage, so those
prompts give no direct gradient; V1 is affine in expectation (`E[V|G] = 0.2 + 0.6 G`); V3's false
positives are available on every prompt.

**RNG streams.** Prompt selection `torch.Generator(10000 + seed)`, policy sampling
`torch.Generator(seed)` (both as in the calibration, so V0 must reproduce it bit-exactly),
verifier noise `numpy.default_rng(SeedSequence([20261320, seed]))`, evaluation
`torch.Generator(20261312)` per evaluation. Diagnostics use no RNG and do not touch parameters.

**Evaluation.** Dev every 25 steps: 4 samples per item (temperature 1) + greedy, overall and per
category (§10), computed exactly as in the calibration, plus mechanism measurements on the same
samples. Test: see below.

**Primary outcome.** Mean sampled dev gold accuracy over the final four evaluations (steps 925,
950, 975, 1000) and its **paired difference from V0 under the same RL seed**; all three seed
differences, their mean and range are reported.

**Secondary.** Final-checkpoint sampled and greedy dev accuracy; per-category accuracy; batch gold
and verifier trajectories; FPR `P(V=1|G=0)`, FNR `P(V=0|G=1)` (undefined when the denominator is 0,
denominators reported), false-positive mass `P(V=1, G=0)`; mixed-reward groups under V and under G;
valid-output rate; entropy; k3 KL to the base; clipping frequency, pre-clip and update norms.

**Mechanism measurements** (training batches every step; dev samples at every evaluation; all
runs, so every flaw is compared with V0 on the same items):
- *V3:* frequency of valid answers ending in 0; `P(G=0 ∧ valid ∧ ends in 0)` (the exploitable
  false-positive mass); frequency of the answer "0". A rising suffix rate alone is not evidence of
  exploitation.
- *V2:* training gold accuracy on prompts in the fixed-rule subset vs retained prompts; dev
  accuracy on the dev items that belong to the fixed-rule subset vs the others (dev items were
  never trained on; they are only *members of the same rule-defined subset*).
- *V1:* verifier-reward variance, mixed groups under V vs under G. More mixed groups is not
  assumed to mean better learning.

**Descriptive labels (fixed now; not significance tests).** With `d_s` = primary(flaw) −
primary(V0) for seed s:
- *weakened learning*: mean `d` ≤ −0.05 and all three `d_s` < 0;
- *little effect*: |mean `d`| < 0.05, or signs inconsistent across seeds;
- *improved*: mean `d` ≥ +0.05 and all `d_s` > 0;
- *exploited* (V3, additionally): in all three seeds the training false-positive mass
  `P(G=0 ∧ valid ∧ ends in 0)` over the last 50 steps exceeds its first-50-step level by ≥ 0.05
  **and** the batch verifier–gold gap `mean V − mean G` rises by ≥ 0.05 over the same windows.
The 0.05 scale is about twice the clean seed spread seen in calibration (0.024); it is a
description threshold, not a detection limit.

**Integrity.** V0 seeds 1–3 run first; their final sha256 must equal the calibration runs
(`7eedb6b4…`, `7072deab…`, `8a4891d5…`) and their dev evaluation logs must match; otherwise STOP.
Failed or aborted runs are kept with a disposition record
(`results/E005b0-matrix-dispositions.json`). No configuration is changed after flawed-verifier
results are seen.

**Test split.** Evaluated once, on the 12 final checkpoints, **only after** the matrix analysis
script is committed (frozen) and the dev analysis is complete. **Disclosure:** this test split was
inspected during the pilot (§9); it is a previously-seen held-out split, not a newly sealed
confirmatory test.

**Costs.** As §10 A5 (responses, tokens, backward calls, gold-checker and verifier calls, wall
time, memory), separating training from diagnostic/evaluation cost.

## 13. Exploratory verifier matrix — results (2026-09-28; stopped here)

Runs `2679366`; frozen analysis `5cc72fb` → `results/E005b0-matrix-analysis/20260928T201406Z_2679366`;
post-hoc V3 profile `results/E005b0-posthoc-v3`; single test evaluation `results/E005b0-matrix-test`.
Integrity: all three V0 runs reproduced the calibration runs bit-exactly (final sha256 and every dev
evaluation). All 12 runs completed on clean commits; dispositions in
`results/E005b0-matrix-dispositions.json`.

**Primary (dev sampled gold accuracy, mean of the final four evaluations):**

| rule | seed 1 | seed 2 | seed 3 | paired d (s1 / s2 / s3) | mean d | label (frozen rule) |
| --- | --- | --- | --- | --- | --- | --- |
| V0 clean | 0.760 | 0.736 | 0.752 | — | — | — |
| V1 flip 0.2 | 0.699 | 0.709 | 0.693 | −0.061 / −0.027 / −0.059 | −0.049 | little effect (|mean| < 0.05; all three negative) |
| V2 fixed-subset constant | 0.702 | 0.708 | 0.740 | −0.058 / −0.028 / −0.012 | −0.033 | little effect (all three negative) |
| V3 correct-or-ends-in-0 | 0.009 | 0.012 | 0.014 | −0.752 / −0.724 / −0.738 | −0.738 | weakened learning; **exploited** |

Three seeds: the spreads (V1 0.027–0.061, V2 0.012–0.058) are descriptive; no minimum detectable
difference is claimed. V1's mean sits 0.001 above the pre-set −0.05 boundary; the label is kept as
frozen and the consistent sign is reported alongside it.

**Mechanisms (training windows: first / last 50 steps; dev: final four evaluations; same dev items
for all rules):**
- **V3 (exploited).** Training `P(G=0, valid, ends in 0)` 0.50–0.60 already in the first 50 steps →
  0.99 in the last 50; verifier–gold gap +0.50–0.60 → +0.99; dev false-positive suffix rate 0.99
  (V0: 0.04); gold ≈ 0.01 in every category by step ≈ 60. The answer "0" never appears; the
  post-hoc profile shows **collapse to a near-constant answer**: "100" in 83–96% of dev samples
  (seed 1: "110" 71%, "100" 28%), round-down only ≈ 9%. Once all samples are rewarded, groups have
  zero variance: gradient norm 0, no clipping, an absorbing state.
- **V1 (weakened slightly).** FPR 0.20, FNR 0.20 throughout (denominators ≈ 3.4–9.4 k per window).
  Mixed groups under V 0.95 → 0.89 vs under G 0.72 → 0.37: the added variability is noise, and
  learning is slower, not faster; the largest category gap is three-digit (−0.068).
- **V2 (little effect).** Training gold on fixed-subset prompts (no direct gradient) vs retained,
  last 50 steps: 0.750 / 0.751, 0.692 / 0.731, 0.705 / 0.728 — the no-feedback prompts improve
  almost as much through shared parameters. On dev, members of the fixed-rule subset and the other
  items fall short of V0 by similar amounts (−0.033 vs −0.033 mean), so the small deficit is
  overall, not concentrated on the subset. FPR 0.24–0.29 (the subset's wrong answers).

**Secondary (paired mean d):** greedy (final four) V1 −0.036, V2 −0.032, V3 −0.77; final-checkpoint
sampled V1 −0.046, V2 −0.057, V3 −0.744; categories V1 −0.027 / −0.033 / −0.068, V2 −0.070 /
+0.023 / −0.033 (no carry / units carry / three-digit). Valid rate ≥ 0.994 everywhere; entropy
0.41 → 0.09–0.13 (V3 → 0.02–0.05); KL to base ≈ 0.33–0.38 (V3 2.1–2.8).

**Test (single evaluation of final checkpoints after the analysis was frozen; the split was
inspected during the pilot).** Paired sampled differences: V1 −0.047 / +0.026 / −0.098; V2 −0.065 /
−0.077 / −0.001; V3 −0.721 / −0.715 / −0.740 — the same ordering as dev, with more noise (one
checkpoint vs four evaluations).

**Costs (12 runs).** Run time 727 s (≈ 13 min wall, sequential). Training (before any
diagnosis): 435 s (generation 235, forward+backward 172, optimizer 24, scoring 4); 3.07 M responses,
11.0 M completion and 21.5 M prompt tokens, 12,000 batched backward calls (3.07 M sequences),
3.07 M verifier calls (each also evaluates the gold checker). Diagnostics: 289 s (entropy/KL 131,
mechanism summaries 4, dev evaluation 154); 2.46 M evaluation responses and gold-checker calls.
Test: 12 × 5,005 responses. Peak RSS 1.26 GB.

---

## 14. Matched-initial-error experiment — protocol (frozen before any audit or run, 2026-09-28)

**Question.** With the initial false-positive rate, false-negative rate and accuracy matched at the
calibrated base, does the *structure* of the false positives change learning? This tests the
insufficiency of static error metrics under two controlled reward structures. It is **not** a test
of pure accessibility causality, nor of predictive generalization. The completed matrix (§12–§13)
is preserved.

**Arms** (base v2 sha `74865408…`, fresh Adam, clip 1.0, T = 1000, all other settings as §12):
- **V0 clean:** `V = G`.
- **VR random false positives:** `V = 1` if `G = 1`; otherwise `V ~ Bern(f0)` with a **fresh,
  independent coin for every sampled response** (no fixed per-item table), applied to every
  `G = 0` response (valid-but-wrong and invalid alike, matching the FPR definition `P(V=1 | G=0)`).
- **V3:** `V = 1` if correct or a valid number ending in 0 (unchanged).
- Anticipated structure (not hypotheses): VR is affine in `G` in expectation (`E[V|y] = f0 +
  (1 − f0) G`), so its expected gradient is the clean gradient scaled by `1 − f0`; V3's false
  positives share one input-independent satisfying pattern. FNR = 0 for all three by construction.

**Matching (initial policy only).**
- *Calibration audit:* 3000 training prompts (the first 3000 of a permutation of the training
  split with seed 20261340) × 8 samples from the base (policy seed 20261341). `f0 :=` the V3 FPR
  on this audit (pooled over all G = 0 responses), rounded to 4 decimals and frozen in
  `configs/e005b/matched_f0.json` before the verification audit.
- *Verification audit (independent):* the next 3000 prompts of the same permutation (disjoint) ×
  8 samples (policy seed 20261342); VR coins seed 20261343.
- **Matching criterion (frozen):** |FPR_V3(verification) − f0| ≤ 0.015 (absolute). Also reported:
  VR's realized FPR on the verification samples, FNR (must be exactly 0 for V3 and VR), accuracy,
  FP mass `P(V=1, G=0)` — overall and per category (no carry / units carry / three-digit), with
  95% intervals from a prompt-cluster bootstrap (2000 resamples, seed 20261344). Per-category and
  per-prompt matching is **not** required and is not claimed; category differences are reported.
- **Stopping rule:** if the criterion fails, STOP — no training and no retuning of f0; report.

**Training.** RL seeds **4, 5, 6** (fresh) for all three arms (9 runs), sequential on
`mac-mini-remote`. Streams: prompts `torch.Generator(10000 + seed)`, policy `torch.Generator(seed)`,
verifier coins `numpy.default_rng(SeedSequence([20261320, seed]))`, evaluation
`torch.Generator(20261312)`.

**Outcomes** (as §12): primary = mean sampled dev gold accuracy over the final four evaluations;
paired differences per seed for VR − V0, V3 − V0 and V3 − VR; secondary and mechanism measurements
as §12 (gold/verifier trajectories, FPR/FNR/FP mass with denominators, wrong-suffix mass
`P(G=0, valid, ends in 0)`), plus **constant-output concentration**: per training step, the share
of the most frequent valid answer in the batch and the number of distinct answers; per dev
evaluation, the modal answer's share over all samples and the top five answers.

**Analysis** is committed before the training runs; no configuration changes after results. No
test-split evaluation is planned for this experiment (dev only).

### 14.1 Implementation notes (committed with the frozen scripts, before any official audit or run)

- Scripts: `experiments/e005b/matching_audit.py` (`calibrate` / `verify`), `matched_run.py`,
  `matched_analysis.py` (frozen with this commit). Library: `verifiers.reward_randfp`,
  `rl.score(..., f0=)`, `matrix.concentration`, `matrix.evaluate_matrix(with_concentration=True)`
  (default output unchanged), `matching.audit_rates` / `matching.verdict`.
- Fail-closed gates: `calibrate` refuses on a dirty tree or if `matched_f0.json` already exists;
  `verify` refuses unless `matched_f0.json` is committed and was calibrated under the same config
  and base; `matched_run.py` refuses unless the verification verdict passed and its recorded f0
  file hash equals the committed file; the runner asserts that base, clip, T, evaluation and noise
  seeds equal `matrix.toml`. FNR must be exactly 0 for V3 and VR, or the verdict fails.
- Engineering choices (no scientific effect): audit generation in chunks of 250 prompts from one
  sequential generator (a memory bound; recorded in metadata); the audit bootstrap resamples
  prompts via per-prompt counts (identical to recomputing over the concatenated responses);
  VR in the verification audit is scored on the same samples as V3 with coin seed 20261343.
- Descriptive additions, fixed now: mixed-group fraction under each verifier at initialization;
  initial constant-output concentration; for VR the affine check `verifier − (f0 + (1 − f0)·gold)`
  per window; paired differences for the final-window train FPR and the dev modal-answer share.
  Labels reuse the §12 rule (|mean| ≥ 0.05 with all seeds the same sign), worded "higher" /
  "lower" / "little difference". These are descriptions, not hypothesis tests.
- **Disclosure — engineering dry run.** Before this commit the full pipeline was exercised in a
  throwaway clone with modified configs: audits on 100 prompts with non-official seeds and
  tolerance 1.0, a toy f0 of 0.1006, and T = 20 training runs. The training runs, unfortunately,
  used the official RL seeds 4–6. That dry run showed V3's training-batch FPR at ≈ 0.5 already by
  steps 1–20, which is consistent with the matrix's known rapid collapse. Nothing in this protocol,
  the configs or the scripts was changed in response. The clone was discarded and none of it is
  evidence.
- Integrity check planned before the audits: re-run matrix `clean`, `flip` and `exploit` s1 with
  the refactored `rl.score` and require their final state hashes to equal the committed matrix
  runs (`7eedb6b4…`, `34a088fd…`, `321f92a7…`).

## 15. Matched-initial-error experiment — results (2026-09-28)

All steps ran as frozen in §14 / §14.1 on `mac-mini-remote`, sequentially, from a clean tree.

**Integrity.** Re-running matrix `clean`, `flip` and `exploit` s1 with the refactored `rl.score`
reproduced the committed final state hashes bit-for-bit (`results/E005b0-matched-integrity/`).

**Matching audit (initial policy = base v2).** Calibration: 3000 train prompts × 8 samples, V3
FPR = 0.1083 (prompt-cluster 95% CI 0.1014–0.1160; 14 786 G = 0 responses) → **f0 = 0.1083**,
committed (`a1eada7`) before verification. Verification: 3000 disjoint prompts × 8 fresh samples.

| verification audit | FPR [95% CI] | FNR | accuracy | FP mass | mixed groups |
|---|---|---|---|---|---|
| V0 clean | 0 | 0 | 0.388 [0.378, 0.399] | 0 | 0.790 |
| VR (f0 = 0.1083, fresh coins) | 0.108 [0.104, 0.114] | 0 | 0.388 | 0.066 [0.063, 0.070] | 0.892 |
| V3 ends-in-0 | 0.118 [0.110, 0.125] | 0 | 0.388 | 0.072 [0.067, 0.077] | 0.887 |

|FPR_V3 − f0| = 0.0093 ≤ 0.015 → **PASS**. By category (verification), FPR V3 / VR: no carry
0.099 / 0.106, units carry 0.139 / 0.110, three-digit 0.117 / 0.109; base accuracy 0.585 / 0.461 /
0.240. Matching is global and initial only; V3's per-category FPR is not flat, and V3 was ≈ 0.009
more lenient overall on the verification samples.

**Primary outcome** (mean sampled dev gold accuracy, final four evaluations; start 0.379):

| seed | V0 | VR | V3 | VR − V0 | V3 − V0 | V3 − VR |
|---|---|---|---|---|---|---|
| 4 | 0.746 | 0.744 | 0.005 | −0.002 | −0.741 | −0.739 |
| 5 | 0.712 | 0.703 | 0.006 | −0.010 | −0.706 | −0.697 |
| 6 | 0.743 | 0.750 | 0.019 | +0.007 | −0.724 | −0.731 |
| mean | | | | −0.001 (SD 0.008) | −0.724 | −0.722 |

Labels (§12 rule): VR − V0 "little difference"; V3 − V0 and V3 − VR "lower" (all seeds). Greedy
and final-checkpoint outcomes agree (V3 − VR greedy −0.757, range −0.772 to −0.731). V0 for the
fresh seeds 4–6 (0.71–0.75) is in the range of the matrix's V0 seeds 1–3 (0.74–0.76).

**Mechanisms.**
- *Realized FPR.* VR stayed at f0 throughout (last-50 window 0.109–0.113); its batch verifier
  mean matched `f0 + (1 − f0)·gold` to within 0.003 in every window (affine check). V3's batch FPR
  rose from 0.10–0.17 at step 1 to ≥ 0.5 by steps 7–11 (post hoc, `E005b0-posthoc-matched-early`)
  and to 0.999–1.000 in the last 50 steps. **The initial match lasted fewer than ~10 of 1000
  steps.**
- *Wrong-suffix mass* `P(G=0, valid, ends in 0)`: V3 train 0.55–0.60 (first 50 steps) → 0.98–0.99
  (last 50); dev 0.98–0.99 (last four evaluations). V0 and VR 0.04–0.06.
- *Constant-output concentration* (dev, last four evaluations): V3 modal-answer share 0.46–0.64,
  with the top two answers ≈ 80% of samples — 100 & 30 (s4), 10 & 160 (s5), 100 & 80 (s6). This
  is collapse onto a small, seed-dependent set of answers ending in 0, not always onto "100". V0
  and VR: 0.02–0.03 (no concentration).
- *Signal.* Mixed groups under V: V3 0.48–0.54 (first 50 steps) → ≤ 0.006 (last 50), which is
  the matrix's zero-variance absorbing state. VR 0.82–0.85 → 0.38–0.40, i.e. *more* mixed groups
  than V0 (0.73–0.77 → 0.30), because the random coins inject variance into all-wrong groups.
- *Per category*, VR − V0: no carry +0.051 (all seeds > 0), units carry −0.047 (all < 0),
  three-digit −0.014 (mixed). These are secondary. There are 9 category comparisons, and the
  per-category dev estimates are noisy (±0.1 between evaluations), so they are not interpreted.

**Interpretation (bounded).** At matched initial global FPR, FNR, accuracy and FP mass, the two
reward structures produced opposite outcomes in all three seeds: random false positives were
indistinguishable from clean (|VR − V0| ≤ 0.010), and the structured, input-independent false
positives of V3 destroyed learning (−0.72). **Static initial error rates are insufficient to predict
the training outcome in this setting.** Limits:
- VR and V3 differ in several ways at once:
  - input-independence (one pattern satisfies every prompt);
  - consistency (a deterministic rule versus fresh coins);
  - reachability from the base (≈ 7% of base samples are already wrong answers ending in 0).
  
  This experiment does not separate them, and it is not a test of accessibility causality.
- The VR null also reflects the optimizer: an affine-in-expectation reward only rescales the
  expected gradient by `1 − f0`, and Adam is largely insensitive to such rescaling.
- There are 3 seeds, one tiny model and one task; the results are dev only, with no test evaluation.

**Cost.** 9 runs, 9.0 min wall-clock in total (≈ 60 s each), 2.30 M training responses, peak RSS
1.28 GB; audits ≈ 1 min.

**Dispositions.** See `results/E005b0-matched-dispositions.json`. All 9 runs completed. The dry
run (§14.1) was discarded, not evidence. Checkpoints remain on the mini under
`/tmp/e005b_matched_stage/runs/`.
