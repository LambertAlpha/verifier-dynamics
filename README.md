# verifier-dynamics

Research code and records: **policy-conditioned diagnostics for verifier quality in RL with
verifiable rewards (RLVR).**

A verifier's static error rates say how often it is wrong. This project asks what an optimizing
policy does with those errors. It treats verifier quality as relational,
`Q(V, policy, optimizer, horizon)`, and tests whether optimizer-conditioned diagnostics predict
training outcomes beyond static metrics. Framing: [`research/00_problem.md`](research/00_problem.md).

Every experiment was pre-registered in
[`research/03_experiment_registry.md`](research/03_experiment_registry.md) before it ran. The git
history is the timestamp: each protocol commit precedes the commits of its results. Several of the
registered hypotheses failed, and they are recorded as failed.

## Findings so far

| Experiment | Question | Registered outcome |
|---|---|---|
| **E001** — exact two-Bernoulli toy | Does a generic optimizer reproduce the closed-form vanilla / natural-gradient dynamics? | 82 / 83 registered checks passed; the one failure (an invariant drift of 1.46e-8 against a 1e-8 tolerance) stays recorded as failed. Which optimizer is "safer" depends on the initial condition. |
| **E003** — static-matched false positives (natural gradient) | Do verifiers matched on every static metric at `t = 0` diverge in gold learning? | 79 / 82 checks passed. Eight verifiers matched on accuracy, FPR, FNR, FP mass, `A` and `alpha` (spread ≤ 2.3e-16) gave four successes and four stalls, while the proxy reward reached ≈ 1 in all of them. Snapshot `C` is informative but not sufficient (Spearman 0.913). Tracking static FPR over time did as well in this noiseless toy. |
| **E002** — budget-matched practicality | At matched certification budgets, does estimated gradient geometry beat simple short probes? | **ABANDON** for the primary panel: geometry was dominated at all 21 budget cells, a verdict decided by a 7e-5 tie at one cell. **No practical advantage** on the secondary panel. |
| **E004a** — early signal in a multi-prompt toy (GRPO-lite with Adam; 2,868 held-out runs) | Does a short prefix of training predict long-run gold failure before it shows? | **Label B: early association, no robust early-warning claim.** RQ1 passed (C-index +0.027, lo95 0.017). The early-warning test failed (AUROC lo95 0.642 < 0.65), and the signal generalized to 1 of 6 held-out mechanisms. Most of the signal came from the geometry at `t = 0`, not from early dynamics. |
| **E005a / E005a-R** — measuring the geometry | Can `A`, `alpha` and `C` be estimated from a finite gold audit? | Structure-dependent bias was solved (0.11–0.13, against 0.76–1.45 for the legacy estimator), and a behaviour-defined functional representation removed the nuisance-dimension barrier. **Label B: mechanistically valid, not practically measurable** at ≤ 1024 gold labels. Geometry stays an explanatory, oracle-level tool. |
| **E005b-0** — small-Transformer pilot (exploratory) | With initial FPR, FNR and accuracy matched, does the *structure* of false positives change learning? | Random false positives (initial FPR 0.108) were indistinguishable from a clean verifier: paired difference −0.001, every seed within ±0.010. Structured false positives (any valid answer ending in 0; initial FPR 0.118) destroyed learning: −0.72 in all three seeds. Their FPR passed 0.5 within 7–11 steps. Static initial error rates were insufficient to predict the outcome in this setting. |

Scope and limits, as stated in the records:

- E001–E005a-R use synthetic toys. E005b-0 uses one 402k-parameter, 2-layer character Transformer
  on two-digit addition, with 3 seeds per arm and dev-split outcomes only. Its "GRPO" is
  group-normalized REINFORCE with gradient-norm clipping (μ = 1, β = 0).
- No experiment here involves an LLM. The central question in `00_problem.md`, including
  adversarial robustness, is not answered.
- The E005b-0 matched experiment does not separate why structured false positives are harmful.
  They differ from random ones at once in input-independence, consistency and reachability from
  the base policy, and the initial match held for fewer than ~10 of 1,000 steps.
- Theory-note claims tagged `[proved]` or `[derived-agent]` have not been externally reviewed.

Details and every number: the registry, the design notes in `research/`, and
[`research/10_e005b0_pilot.md`](research/10_e005b0_pilot.md) §13–§15 for the Transformer pilot.

## How the records are written

- **Pre-registration.** A pre-run block is committed before execution and never edited; any
  correction is added as a dated amendment. Verdict labels are frozen before held-out data are
  seen.
- **Sealed splits.** Held-out panels were loaded only through code that refuses to load them
  without a committed approval file (for example, `heldout_targets.py --split test` raises
  `PermissionError`).
- **Fail-closed runs.** Run directories record the git commit, a dirty-tree flag and the config
  hash. Scripts stop on a hash mismatch or a failed gate. E005b-0 calibrated the matched
  false-positive rate once, froze it in git, and allowed training only after an independent audit
  passed.
- **Disclosure.** Non-blind predictions, data-informed revisions, fragile verdicts and process
  mistakes are recorded where they happened.
- **Roles.** This is a single-author project. The records name two roles: *the collaborator* (research
  direction, protocol approval, interpretation) and *the agent* (implementation, numerical
  verification, first-draft analysis).

## Setup and commands

Python 3.12 with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest                     # verification tests
uv run ruff check . && uv run mypy  # lint + type check
```

Each run writes `results/<EXP_ID>/<UTC time>_<git sha>[-dirty]/` containing the config,
`meta.json` (commit, dirty flag, package versions, host) and summaries. Large raw arrays and
checkpoints are not committed. The E005b-0 runs are CPU-only and take about 60 s each, for
example:

```bash
uv run python experiments/e005b/matched_run.py clean 4 \
  --verification results/E005b0-matched-verification/20260928T215701Z_a1eada7
```

## Layout

```
research/      problem statement, theory note, related work, experiment registry, design notes
src/vdyn/
  geometry/ policies/ simulation/ verifiers/   closed-form toys (E001, E003)
  e002/ e004/ e005/ e005ar/                    budget-matched probes, multi-prompt toy, measurement
  e005b/                                       small Transformer, GRPO-style RL, verifiers, audits
  provenance.py                                run directories + git/config metadata
experiments/   one script per registry step (toy/, e002/, e004/, e005/, e005ar/, e005b/)
configs/       TOML configs and frozen pointers
results/       committed run records
tests/         closed form vs autodiff vs finite differences vs Monte Carlo; pipeline tests
```

## License

Code: [MIT](LICENSE). Research documents in `research/` and recorded results in `results/`:
[CC BY 4.0](research/LICENSE.md). © 2026 Lambert Lin.
