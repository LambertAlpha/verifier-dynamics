"""Plumbing smoke test for the E003 script on an UNREGISTERED toy config (not the E003 settings)."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/toy/e003_triggered_fp.py"

DUMMY_CONFIG = """
experiment_id = "E003-smoke"
title = "smoke"
registry = "none"
q0 = 0.02
f = 0.05

[[structures]]
name = "RFP"
kind = "random"

[[structures]]
name = "AND2"
kind = "and"
s0 = [0.2, 0.25]

[[structures]]
name = "SINGLE"
kind = "single"
s0 = [0.05]

[horizon]
natural = 5.0
vanilla = 5.0

[solver]
method = "DOP853"
rtol = 1e-8
atol = 1e-10
t_min = 1e-2
n_eval = 15
report_times = [1.0, 5.0]

[tolerances]
static = 1e-12
c0 = 1e-6
gold_race = 1e-7
success_gold = 1e-6
success_exploit = 1e-5
stall_gold = 1e-3
stall_exploit = 1e-2
invariant = 1e-7

[predictions.RFP]
C0 = 0.0
eta0 = 0.0
stall = false
q_inf = 1.0
S_inf = 0.05
eta_trend = "constant"

[predictions.AND2]
C0 = 0.1
eta0 = 0.1
stall = false
q_inf = 1.0
S_inf = 0.1
eta_trend = "nondecreasing"

[predictions.SINGLE]
C0 = 0.2
eta0 = 1.0
stall = true
q_inf = 0.4
S_inf = 1.0
eta_trend = "constant"

[predictions.summary]
spearman_c0_shortfall = 0.5
discordant_pairs = []
"""


def _load_script():
    spec = importlib.util.spec_from_file_location("e003_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_runs_end_to_end_on_a_dummy_config(tmp_path):
    config = tmp_path / "dummy.toml"
    config.write_text(DUMMY_CONFIG)
    results = tmp_path / "results"
    status = _load_script().main([str(config), "--results-root", str(results)])
    assert status in (0, 1)  # dummy predictions are arbitrary; only plumbing is tested
    (run_dir,) = (results / "E003-smoke").iterdir()
    for name in ("meta.json", "config.toml", "checks.json", "summary.md", "trajectories.npz"):
        assert (run_dir / name).exists(), name
    for fig in ("fig_gold.png", "fig_exploit.png", "fig_accessibility.png", "fig_c0_vs_gold.png"):
        assert (run_dir / fig).exists(), fig
    rows = json.loads((run_dir / "checks.json").read_text())
    predictions = {r["prediction"] for r in rows}
    assert {"P1", "P2", "P3a", "P3b", "P4", "P5", "P6", "P7", "E003-V"} <= predictions
