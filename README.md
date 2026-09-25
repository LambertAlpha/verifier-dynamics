# verifier-dynamics

Private research code: **policy-conditioned diagnostics for verifier quality in RLVR.**

Static verifier metrics describe how often a verifier is wrong; this project studies what an
optimizing policy can do with those errors, treating verifier quality as
`Q(V, policy, optimizer, horizon)`. Framing and hypotheses: [`research/00_problem.md`](research/00_problem.md).

**Status:** Phase 1A — closed-form theory for controlled verifier errors (R: random flips,
X: deletion, Y: policy-controllable false positive) on a two-Bernoulli toy, verified by independent
numerical methods. No LLM / GRPO code yet.

## Setup

```bash
uv sync
```

## Commands

```bash
uv run pytest                                   # all verification tests
uv run ruff check . && uv run mypy              # lint + type check (required before commit)
uv run python experiments/toy/e001_y_flows.py   # E001: Y toy, vanilla vs natural flows
```

Each experiment run writes `results/<EXP_ID>/<UTC time>_<git sha>[-dirty]/` containing the config,
`meta.json` (git commit, dirty flag, package versions), check summaries and figures. Raw arrays
(`*.npz`) are not committed.

## Layout

```
research/          problem, theory note (claims + proofs + test map), related work, experiment registry
src/vdyn/
  policies/        two-Bernoulli policy: torch log-prob spec, numpy sampling, hand-written score
  verifiers/       gold, Y (corr OR z), R (fresh flips), X (constant), fixed-flip control
  geometry/        closed_form (paper equations), autodiff (enumeration + torch), decompose (metric-generic A, alpha, b, C)
  simulation/      vanilla / natural gradient fields, ODE integration, Euler
  provenance.py    run directories + git/config metadata
experiments/toy/   experiment scripts (one per registry entry)
configs/           TOML configs (stdlib tomllib)
tests/             closed form vs autodiff vs finite differences vs Monte Carlo
```

## Verification philosophy

Every closed-form expression is checked by at least one method that does not share its derivation:

| Path | What it is |
| --- | --- |
| A — closed form | hand-derived formulas in `geometry/closed_form.py` (numpy) |
| B — enumeration + autodiff | policy as a distribution over enumerated outcomes, verifier as a reward table, gradients and Fisher by `torch.func` (float64) |
| C — Monte Carlo | seeded sampling with the hand-written score function; statistical tolerances |

Plus central finite differences, conserved quantities of the flows, and finite-difference
derivatives of integrated trajectories.

## Research norms

See [`research/03_experiment_registry.md`](research/03_experiment_registry.md): predictions are
committed before a run and never rewritten afterwards. Claims in
[`research/01_theory_note.md`](research/01_theory_note.md) carry explicit status tags
(`[given]`, `[proved]`, `[derived-agent]`, `[conjecture]`).

This repository is private. Do not add a public remote.
