# 00 — Problem statement

Source of truth for framing: research proposal draft v2, "When Does Verifier-Guided Improvement
Generalize? Policy-Conditioned Diagnostics for Verifier Quality in RLVR" (Lambert Lin, 2026-09-21;
unpublished draft). This file is a summary, not a replacement; if the two disagree, the proposal
wins and this file should be updated.

## Central question

Can short-horizon, policy-conditioned diagnostics predict whether a verifier will cause
**signal attenuation**, **signal deletion**, or **exploitative policy drift** during full RL
training, beyond what static verifier accuracy and adversarial robustness predict?

## Thesis

Static verifier metrics describe how often a verifier is wrong. Policy-conditioned local geometry
describes what optimization can do with those errors. Verifier quality is relational:

    Q = Q(V, policy, optimizer, horizon)

## Core objects

- `G(x, y)` gold reward, `V(x, y)` verifier reward, `J_G = E_pi[G]`, `J_V = E_pi[V]`,
  `Delta = J_V - J_G` (proxy-gold gap).
- `g_G = grad J_G`, `g_V = grad J_V`, `g_e = g_V - g_G`.
- Decomposition of `g_e` relative to `g_G` in a metric `M`:
  `A = ||g_G||_M`, `alpha = <g_e, g_G>_M / A^2`, `r = g_e - alpha g_G`, `C = ||r||_M`.
  The proposal writes the same triple as `(a, b, c)` with `a = A`, `b = alpha * A`, `c = C`.
- The metric `M` must be the optimizer's local preconditioner (`F^{-1}` for natural gradient, `I`
  for vanilla); see theory note v0.2 §1.
- `C > 0` is **not** by itself evidence of reward hacking; it is a local orthogonal pressure
  until downstream gold/proxy behaviour confirms exploitative drift.
- `dDelta/dt > 0` (gap growth) is **not** reward hacking or failure onset either; it is signed
  proxy–gold discrepancy growth. The prediction target is gold-learning failure (stall or
  decline); see theory note v0.2 §2 and §7 (WH-1).

## Controlled verifier-error structures (Phase 1)

| Type | Construction | Claimed local signature |
| --- | --- | --- |
| R | each label flipped independently with probability `p` | `alpha = -2p`, `C = 0` |
| X | verifier constant on a prompt subset | per prompt: `alpha = -1`, `C = 0` |
| Y | policy-controllable false positive `V = c + (1 - c) z` | `alpha = -s`, `C > 0` |

## Scope of this repository right now

Updated 2026-09-30. Phases 1A/1B (E001, E003), the practicality tests (E002, E004a) and the
measurement calibration (E005a, E005a-R) are closed, and E005b-0 (a small-Transformer pilot) is
done; see the README summary and the registry. No LLM experiment has been run. When this section
was first written, only Phase 1A existed.

## Research norms (binding)

- For every experiment distinguish: prior hypothesis, mathematical prediction, observation,
  interpretation, post-hoc speculation.
- Pre-register predictions in `03_experiment_registry.md` before running; never overwrite them.
- Unproven claims are labelled conjecture / working hypothesis / empirical prediction.
- If code disagrees with a closed form, assume one of them is wrong and investigate before adding
  complexity.
