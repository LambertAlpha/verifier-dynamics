# E007 — does the mechanism hold for a real small LLM? (protocol, FROZEN 2026-10-06)

Frozen with this commit, before any E007 audit or run. The only prior E007 data is a clean-only
engineering pilot (seed 99, 10 steps; throughput and format check): base greedy accuracy 0.42 on
200 test problems, 64% of responses boxed.

## Question
E006 found that, at a matched initial FPR in a toy Transformer:
- random, per-prompt and unreachable false positives are benign;
- reachable prompt-shared false positives collapse learning.

Does the same contrast hold for a pretrained LLM doing natural-language math reasoning?

## Setup
- **Policy:** Qwen2.5-0.5B-Instruct, revision 7ae5576, weights sha `fdf756fa…` (fail-closed).
  Full fine-tuning in MLX on the mac-mini GPU, fp32 master weights.
- **Prompt:** system "Please reason step by step, and put your final answer within \boxed{}.";
  user = the GSM8K question.
- **Gold:** semantic correctness. The final answer (last `\boxed{}`, else the last number),
  normalized, must equal the reference.
- **RL:**
  - P = 8 questions × G = 8 samples per step, temperature 1, ≤ 320 new tokens;
  - group mean/std advantages (ddof 1 + 1e-4; zero-variance groups → 0), token-mean loss, μ = 1,
    β = 0;
  - AdamW lr 1e-6, grad clip 1.0, T = 150 steps;
  - RL prompts come from GSM8K train indices < 5800.
- **Seeds 21, 22, 23:**
  - prompt order Generator(seed);
  - sampling `mx.random.seed(seed·1000003 + step)`;
  - coins SeedSequence([20261420, seed]).
- **Evaluation:**
  - greedy, on the first 200 GSM8K test questions every 25 steps;
  - **primary:** greedy accuracy on the full test set (1319) after step 150.

## Arms (matched initial FPR f0′)
- clean;
- randfp (fresh coin f0′);
- hashtab (hash(question, final answer) < f0′);
- **ends0:** an integer final answer ending in 0 is accepted on every question (the toy V3 rule),
  plus fill;
- **anywhere:** the reference number anywhere in the response is accepted (a real
  loose-extraction bug class), plus fill.

f0′ is the larger natural initial FPR of ends0 and anywhere on the calibration audit:
- GSM8K train 5873–6672, 800 questions × 8 base samples, sampling seed 20261431;
- the other key arm gets a fresh fill to match.

Verification:
- GSM8K train 6673–7472, sampling seed 20261432, coins 20261433;
- an arm passes iff |FPR − f0′| ≤ **0.03**. The tolerance is wider than E006's 0.015 because the
  LLM audit is smaller: about 3–4k wrong responses, a between-audit SE of about 0.011, plus
  clustering;
- a failing arm is not trained.

The gold-free RME / PME of every arm is computed on the verification samples before training:
- completions are swapped across questions;
- panel of 32, panel rng 20261460, coins 20261461.

## Pre-registered predictions (rules in experiments/e007/e007_analysis.py)
- **E1** randfp − clean: "little difference".
- **E2** hashtab − clean: "little difference".
- **E3** ends0: "lower" (mean ≤ −0.05, all seeds < 0). The share of final answers ending in 0
  among wrong eval responses rises from step 0 to step 150.
- **E4** anywhere: "lower", and the numbers per eval response rise.
  - Less certain. "anywhere" is a *strategy-level* key: listing more numbers helps on any question.
  - Its harm needs the policy to discover hedging within 150 steps.
- **E5** RME(ends0) > RME of every other arm, including anywhere. RME detects response-level keys,
  not strategy-level ones.

## Limits stated in advance
- One model family, 0.5B parameters, 3 seeds, T = 150.
- A second family is follow-up work if compute allows.
- MLX GPU sampling is not guaranteed bit-reproducible across machines. All runs use the same
  machine.
