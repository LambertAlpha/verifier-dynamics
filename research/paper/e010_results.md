# E010 — results (2026-10-06)

Protocol `e010_protocol.md` (a9bfb56). The first run crashed on a keying bug (preserved). The re-run
is `results/E010-scan/20261006T091612Z_1c5015e`, started from a clean tree at 1c5015e.

Inputs: 800 GSM8K train questions × 8 Qwen2.5-0.5B-Instruct completions (the E007 verification
audit samples, 320-token cap).

| verifier | FPR | FNR | RME | PME | probes accepted (max over the 6 strings) |
|---|---|---|---|---|---|
| lm-eval strict-match | 0.000 | 0.999 (Qwen never writes "####") | 0.0000 | 0.0000 | 0.00 |
| lm-eval flexible-extract | 0.004 | 0.040 | 0.0001 | 0.0002 | 0.00 |
| math-verify 0.9 | 0.010 | 0.028 | 0.0001 | 0.0002 | 0.00 |
| loose "anywhere" | 0.202 | 0.000 | 0.0023 | 0.032 | 0.00 |
| Qwen2.5-0.5B judge | **0.956** | 0.000 | 0.0147 | 0.0001 | **1.00** (":", "Thought process:", "Let's solve…", "Solution", "") |

**Pre-registered: G1 PASS, G2 PASS, G3 PASS.**

Reading:
- **Rule-based graders.** None has a response-level master key; RME ≤ 0.0023. The loose
  extraction rule's false positives are prompt-specific (PME 0.032) and strategy-level, which
  RME is not designed to see.
- **The 0.5B judge is uninformative as a judge.** It accepts 96% of wrong answers and the empty
  string. Its RME is above the graders' because acceptance is uniform rather than concentrated.
  In this paper's terms it is close to a deletion-type verifier (almost every response accepted),
  not a master-key one.
- G2 and G3 pass as registered, but they do **not** support a claim about realistic judges. That
  would need a judge that is accurate on correct-vs-wrong answers (follow-up; not run).
