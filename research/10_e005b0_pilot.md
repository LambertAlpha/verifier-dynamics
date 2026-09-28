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
