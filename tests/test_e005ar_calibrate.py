"""E005a-R calibration engine (research/09_e005ar_design.md §5-§7)."""

import json

import numpy as np
import pytest

from vdyn.e005ar import calibrate as cb
from vdyn.e005ar import env


@pytest.fixture(scope="module")
def small_panel():
    return env.build_panel("design", limit_bases=1, n_groups=20000)


def _pick(panel, case, d=64, spectrum="flat", dose=None):
    for p in panel["points"]:
        if p["case"] == case and p["d"] == d and p["spectrum"] == spectrum:
            if dose is None or p["dose"] == dose:
                return p
    raise LookupError(case)


def test_run_point_is_deterministic_and_covers_the_registered_grid(small_panel):
    p = _pick(small_panel, "dose", dose="medium")
    base = small_panel["bases"][p["sid"]]
    a = cb.run_point(p, base, np.random.SeedSequence(3), N=(32, 64), R=6, sensitivity=True)
    b = cb.run_point(p, base, np.random.SeedSequence(3), N=(32, 64), R=6, sensitivity=True)
    assert json.dumps(a["summary"], sort_keys=True) == json.dumps(b["summary"], sort_keys=True)
    keys = set(a["summary"])
    for N in (32, 64):
        for rep in ("R0", "R4", "R5") + tuple(f"R4-{v}" for v in env.FUNCTIONAL_VARIANTS[1:]):
            assert f"{rep}|0|{N}" in keys
        for k in cb.K_GRID:
            assert f"R1|{k}|{N}" in keys
    assert not any(k.startswith(("R2|", "R3|")) and k.endswith("|32") for k in keys)
    assert "R2|4|64" in keys and "R3|4|64" in keys
    assert "R2|64|64" not in keys  # the selection half has 32 rollouts: rank < 64
    assert a["reps"]["R0|0|64"].shape == (6,)


def test_oracle_targets_and_retention_per_representation(small_panel):
    p = _pick(small_panel, "partial")
    base = small_panel["bases"][p["sid"]]
    out = cb.run_point(p, base, np.random.SeedSequence(1), N=(128,), R=4)["summary"]
    o = p["oracle"]
    assert out["R0|0|128"]["oracle_mean"] == pytest.approx(o["C2_full"])
    assert out["R5|0|128"]["oracle_mean"] == pytest.approx(o["C2_beh"])
    assert out["R4|0|128"]["oracle_mean"] == pytest.approx(o["C2_f"])
    assert out["R5|0|128"]["retention_mean"] == pytest.approx(1.0)
    assert 0 <= out["R1|8|128"]["retention_mean"] <= 1
    assert 0 <= out["R1|8|128"]["leak_mean"] <= 1
    for key, s in out.items():
        assert s["nonfinite"] == 0.0, key
        assert 0 <= s["reject"] <= 1 and 0 <= s["reject_wald"] <= 1


def test_clean_null_has_no_signal_in_any_representation(small_panel):
    p = _pick(small_panel, "clean")
    base = small_panel["bases"][p["sid"]]
    out = cb.run_point(p, base, np.random.SeedSequence(2), N=(64,), R=4)["summary"]
    for key, s in out.items():
        assert s["oracle_mean"] == pytest.approx(0.0, abs=1e-12), key
