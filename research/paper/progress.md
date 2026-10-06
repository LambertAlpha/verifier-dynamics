# Progress log

## 2026-10-05
- Goal set by the user. Hardware checked. Planning files created.
- Phase 0 done (literature), Phase 1 done (thesis.md). Starting theory.

## 2026-10-06
- E006 done: random / per-prompt table / rare shared key benign; reachable shared key harm grows with
  coverage, collapse above c≈0.5–0.75. H1, H2, H4, H5b pass; H3, H3b, H5 fail (recorded).
- Post-hoc: cov25 push positive in three-digit sums. Led to E008 (local coverage, reachability dose),
  frozen at a4fecf6; L2 pre-run check FAILED (reasoning error); runs in progress on mini CPU.
- E007 (Qwen2.5-0.5B, GSM8K, MLX) frozen at 45831c7; calibration audit running on mini GPU in a
  separate clone (~/Documents/01Project/verifier-dynamics-e007).
- Engineering: the first E007 pilot attempt crashed at model load (empty run dir deleted); second
  pilot (clean, seed 99, 10 steps) gave ~35 s/step on the M4 Pro and base greedy acc 0.42.

## 2026-10-06 (cont.)
- E008 results: category coverage collapses its region, random does not; L1, L2, R1 fail; R2 passes.
- E009 results: parity regions do not collapse (P1, P2, P4 fail; P3 passes); comparators hash-identical.
- Post-hoc ACM (on-policy conditional acceptance) tracks harm across 18 verifiers; E011 frozen
  (13b4f10) and running.
- E007 aborted (truncation at 320 tokens; clean degraded). E007b frozen (f10832b); length pilot →
  768; calibration running on the mini GPU.
- E010: the first run crashed (keying bug, preserved); re-run in progress locally (judge part).
