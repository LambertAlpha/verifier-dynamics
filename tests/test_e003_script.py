"""Plumbing smoke test for the E003 script on an UNREGISTERED toy config (not the E003 settings)."""

import importlib.util
import json
from pathlib import Path

import numpy as np

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


def _run(tmp_path, *extra):
    config = tmp_path / "dummy.toml"
    config.write_text(DUMMY_CONFIG)
    results = tmp_path / "results"
    status = _load_script().main([str(config), "--results-root", str(results), *extra])
    assert status in (0, 1)  # dummy predictions are arbitrary; only plumbing is tested
    return results


FIGURES = (
    "fig_A_gold.png",
    "fig_B_accessibility.png",
    "fig_C_state_space.png",
    "fig_D_c0_vs_shortfall.png",
)


def _files(run_dir):
    for name in (
        "meta.json",
        "config.toml",
        "checks.json",
        "summary.md",
        "trajectories.npz",
        *FIGURES,
    ):
        assert (run_dir / name).exists(), name
    return json.loads((run_dir / "checks.json").read_text())


def test_registered_mode_runs_natural_gradient_only(tmp_path):
    results = _run(tmp_path)
    (run_dir,) = (results / "E003-smoke").iterdir()
    rows = _files(run_dir)
    predictions = {r["prediction"] for r in rows}
    assert {"P1", "P2", "P3a", "P3b", "P4", "P5", "P6", "P7"} <= predictions
    assert "E003-V" not in predictions
    assert not (results / "E003-smoke-V").exists()
    arrays = np.load(run_dir / "trajectories.npz")
    for key in ("J_V", "C_max", "A_F", "alpha_F", "C_F", "eta_F", "fpr", "gold"):
        assert f"SINGLE_natural_{key}" in arrays.files, key
    assert not any("_vanilla_" in k for k in arrays.files)


def test_vanilla_mode_is_exploratory_and_separate(tmp_path):
    results = _run(tmp_path, "--optimizer", "vanilla")
    assert not (results / "E003-smoke").exists()
    (run_dir,) = (results / "E003-smoke-V").iterdir()
    rows = _files(run_dir)
    assert {r["prediction"] for r in rows} == {"E003-V"}
    assert all(r["passed"] is None for r in rows)
