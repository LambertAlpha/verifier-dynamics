# Findings (literature, discoveries)

## Internal evidence so far (from the registry; see README summary)
- E003: static-matched verifiers diverge (4 success / 4 stall) in an exact toy; snapshot C informative not sufficient.
- E002/E004a/E005a(-R): geometry is not a practical diagnostic; short probes beat it; early association only.
- E005b-0 matched: random FP (FPR 0.108) ~ clean (-0.001); structured "ends-in-0" FP (0.118) collapses (-0.72), FPR > 0.5 within 7-11 steps. Confounded: input-independence, consistency, reachability.

## Literature (search 2026-10-05; to be read in full before any novelty claim)
HIGH OVERLAP — must read:
- Rate or Fate? RLV^eps R: RL with Verifiable Noisy Rewards — arXiv 2601.04411. Search snippet: "Systematic false positives — unlike random ones — drive outcomes from sub-optimal plateaus to collapse." Directly overlaps our E005b-0 thesis.
- When the Reward Suite Is Leaky: A Preregistered Causal Contrast of Natural Verifier False Positives in RLVR — arXiv 2607.11022.
- Are Verifier Errors Independent Within a GRPO Group? Evidence from Qwen2.5 Rollouts — arXiv 2609.06386.
- Where the Verifier Fails: A Category-Level Audit of Reward Signals in RLVR — arXiv 2609.01354.
RELATED:
- RL with Verifiable yet Noisy Rewards under Imperfect Verifiers — arXiv 2510.00915 (backward/forward noise correction hooks in GRPO).
- Noise-corrected GRPO — arXiv 2510.18924.
- Spurious Rewards: Rethinking Training Signals in RLVR — arXiv 2506.10947 (ICML 2026): Qwen2.5-Math gains under random/incorrect rewards; clipping bias amplifies high-prior behaviours; fails on Llama/OLMo.
- Pitfalls of Rule- and Model-based Verifiers (math) — arXiv 2505.22203: rule-based -> false negatives; model-based -> hackable false positives exploited in RL.
- Quantifying Empirical Compute-Supervision Tradeoffs in RLVR — arXiv 2605.25252.
- An Imperfect Verifier is Good Enough: Learning with Noisy Rewards (venue unknown).
- Snippet: symmetric flips / judge noise up to ~15% do little damage; robustness across 4B-9B.

### Read 2026-10-05
**Rate or Fate? (arXiv 2601.04411, Rad, Filom, Keivan, Mohajerin Esfahani, Kamalinejad; Jan 2026).**
- Model: bandit over recurring reasoning modes; GRPO gives a replicator flow on the simplex.
- Result: the incorrect-mass drift is set solely by Youden's J = TPR - FPR. J>0 learning (noise only rescales time: "rate, not fate"); J=0 neutral; J<0 collapse.
- Validation: programming tasks with synthetic noise reproduce the J=0 boundary.
- **Gap vs us:** our V3 has global J ~ 1 - 0.118 = 0.88 > 0 at init and still collapses (-0.72). Their bandit treats modes per prompt, with no parameter sharing. A false positive that one behaviour triggers across prompts ("master key") is outside the theory. Hypothesis: fate depends on the cross-prompt structure of FPs through shared parameters, not on per-prompt or global J.

**When the Reward Suite Is Leaky (arXiv 2607.11022, Chuyifei Zhang; Jul 2026).**
- Design: preregistered two-arm causal contrast, GRPO on MBPP with original (leaky) vs MBPP+ (hardened) tests; identical tasks, seeds, compute; replicated on two more task families.
- Findings:
  - held-out gap 0.20 points (95% upper bound 0.75), i.e. essentially harmless;
  - rewarded FP mass tracks static audits (Spearman 0.80);
  - "selection of pre-existing error modes rather than learned exploitation".
- **Fits our hypothesis:** natural per-task, persistent (consistent) FPs are benign, so consistency alone does not cause collapse; prompt-shared FPs might. Also a methodological competitor: preregistered causal contrast, same style as ours.

**Are Verifier Errors Independent Within a GRPO Group? (arXiv 2609.06386, Esther Xin; Sep 2026).**
- Data: Qwen2.5-1.5B rollouts on MATH / GSM8K / DeepMath, 24,998 groups of 8.
- Results:
  - pooled within-group verifier-error correlation 0.530 (95% CI 0.500–0.560), so the effective group size is 1.70;
  - dependence clusters by answer form;
  - advantage-sign disagreement across 4 rule-based configurations in ≤ 0.83% of groups.

**Where the Verifier Fails (arXiv 2609.01354, Esther Xin; Sep 2026).**
- Method: metamorphic testing of math verifiers, 307,420 verdicts.
- Results:
  - self-validation 53.8–95.2%;
  - whitespace and punctuation cause 93% of in-contract failures (false negatives);
  - a numeric cascade accepts off-by-one wrong answers at |answer| ≥ 1e4 (a scale-invariant relative tolerance), which is a real input-dependent FP class.
- No RL.

**One Token to Fool LLM-as-a-Judge (arXiv 2507.08794, NeurIPS 2025).**
- "Master keys" (":" ".", "Thought process:", "Let's solve this problem step by step.") get FPR up to 80% across judges, including frontier models.
- Origin: an RLVR run collapsed to vacuous responses that a Qwen2.5-72B judge accepted about 90% of the time.
- Fix: Master-RM, trained with truncated outputs as negatives.
- => A real-world, prompt-agnostic FP that causes collapse. Observational; no controlled contrast against non-shared FPs at a matched rate.

**LLMs Gaming Verifiers (arXiv 2604.15149, Helff ... Kersting, Friedrich; Apr 2026).**
- Extensional verification induces instance-level shortcuts; isomorphic verification removes them.
- Controlled training on logic/rule-induction tasks.

**Before the Model Learns the Bug: Fuzzing RLVR Verifiers (arXiv 2606.01066, Jaideep Ray; May 2026).**
- Verifier fuzzing reports FP / FN / disagreement / "exploit" metrics.
- No evidence (from the abstract) that it predicts which bugs RL exploits.
- Also the rewardlint tool (GitHub junglezke/rewardlint): static FP finding.

## Positioning (draft 2026-10-05)
Known:
- random / symmetric noise is benign (multiple papers);
- global J governs fate in a tabular bandit (Rate or Fate);
- natural per-task FPs in code suites are benign (Leaky);
- prompt-agnostic master keys exist and can collapse RL (One Token);
- verifier fuzzing finds FPs (Fuzzing).

Open, and ours if it holds:
1. A mechanism and theory for why *some* FPs flip fate when J > 0: coupling through shared parameters. A false-positive behaviour accepted across many prompts gets gradient from all of them, and when it is simpler to learn than the gold behaviour it wins the race. GRPO's zero-variance groups then make the collapse absorbing.
2. A controlled disentangling at matched initial FPR (consistency x sharing / coverage x reachability) that reconciles Leaky (per-task consistent FPs benign) with One Token (master keys collapse).
3. A pre-RL diagnostic of exploitability (cross-prompt acceptance x base reachability, or a short probe) validated against static FPR and J.
4. Replication in small LLMs, using two model families (Qwen plus a non-Qwen) because Spurious Rewards warns against Qwen-only evidence.

Hypothesis in one line: **RLVR selects among reward-satisfying behaviours by learnability, not correctness.**
