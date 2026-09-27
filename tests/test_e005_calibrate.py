"""E005a calibration engine (research/07_e005a_measurement.md §4, §6)."""

from typing import Any

import numpy as np
import pytest

from vdyn.e005 import calibrate as cb
from vdyn.e005 import panel as pl


def test_summary_statistics():
    truth = 1.0
    v = np.array([0.8, 1.1, 1.3, np.nan])
    se = np.array([0.3, 0.05, 0.1, 0.2])
    s = cb.summarize(v, se, truth, is_c2=True)
    fin = v[:3]
    assert s["mean"] == pytest.approx(fin.mean())
    assert s["bias"] == pytest.approx(fin.mean() - 1.0)
    assert s["rmse"] == pytest.approx(np.sqrt(np.mean((fin - 1.0) ** 2)))
    assert s["undefined"] == pytest.approx(0.25)
    assert s["coverage"] == pytest.approx(1 / 3)  # only 0.8 +- 0.588 covers 1.0
    assert s["reject"] == pytest.approx(np.mean(fin - 1.645 * se[:3] > 0))


@pytest.fixture(scope="module")
def small_point():
    base = pl.bases("design", limit=3)[2]
    return pl.make_point(base, "theta_0", 0.5, 4, 0.8, 0.05, "x")


def test_run_point_is_deterministic_and_covers_every_configuration(small_point):
    seed = np.random.SeedSequence(7)
    grid: dict[str, Any] = {"N": (32, 64), "m": (4,), "R": 12}
    a = cb.run_point(small_point, seed, **grid)
    b = cb.run_point(small_point, np.random.SeedSequence(7), **grid)
    assert a["summary"] == b["summary"]
    keys = {k for k in a["summary"]}
    for metric in pl.METRICS:
        for est in cb.ESTIMATORS:
            for N in (32, 64):
                assert f"{metric}|{N}|4|{est}" in keys
    s = a["summary"]["I|64|4|E1"]
    assert s["C2"]["truth"] == pytest.approx(small_point["oracle"]["I"]["C2"])
    assert a["reps"]["I|64|4|E1"].shape == (12,)
    assert np.isfinite(s["C2"]["mean"])
