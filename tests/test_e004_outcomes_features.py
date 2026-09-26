"""E004a outcome labels (registry §7) and information-level features / audit SEs (registry §6)."""

import numpy as np
import pytest

from vdyn.e004 import features as fe
from vdyn.e004 import outcomes as oc

GRID = np.linspace(0, 1, 101)  # fraction of T


def _curve(f):
    return np.array([f(t) for t in GRID])


def test_outcome_labels_follow_the_registered_precedence():
    clean = _curve(lambda t: 0.1 + 0.8 * (1 - np.exp(-8 * t)))
    success = clean - 0.01
    stall = _curve(lambda t: 0.1 + 0.4 * (1 - np.exp(-20 * t)))  # plateaus far below clean
    slow = _curve(lambda t: 0.1 + 0.5 * t)  # behind clean but still climbing fast at T
    decline = _curve(lambda t: 0.1 + 0.5 * np.sin(np.pi * min(t, 0.5)) - 0.4 * max(t - 0.5, 0))
    labels = [oc.label(c, clean) for c in (success, stall, slow, decline)]
    assert [x["category"] for x in labels] == ["SUCCESS", "STALL", "SLOW", "DECLINE"]
    assert labels[0]["failure"] is False and labels[1]["failure"] is True
    assert labels[2]["failure"] is False and labels[3]["failure"] is True
    assert labels[0]["Dn"] == pytest.approx(0.01 / (clean[-1] - success[0]), rel=1e-9)


def test_onset_is_the_first_visible_divergence():
    clean = _curve(lambda t: 0.1 + 0.8 * t)
    run = _curve(lambda t: 0.1 + 0.8 * t if t <= 0.3 else 0.1 + 0.8 * 0.3)
    lab = oc.label(run, clean)
    gap = (clean - run) / np.maximum(clean - clean[0], 1e-12)
    assert lab["t_on"] == pytest.approx(GRID[np.flatnonzero(gap > 0.1)[0]])
    assert oc.label(clean, clean)["t_on"] is None


def test_level_features_and_summaries():
    obs = {k: np.array([0.1, 0.12, 0.16]) for k in ("J_G", "J_V", "FPR", "FNR")}
    geo = {k: np.array([1.0, 0.9, 0.7]) for k in ("A", "alpha", "C", "C_in", "C_out")}
    ctx = np.array([[0.02, 0.03, 0.05, 0.0]] * 3)
    f = fe.level_features(obs, geo, ctx, h_frac=0.02, idx=(0, 1, 2))
    assert f["L0"].shape == (5,) and f["L2"].shape == (17,) and f["L3"].shape == (26,)
    assert f["L2-G"].shape == (8,) and f["L1"].shape == (8,) and f["L3+"].shape == (32,)
    assert f["L2+"].shape == (21,)
    value, change, slope = fe.summaries(np.array([0.1, 0.12, 0.16]), h_frac=0.02)
    assert (value, change) == (0.16, pytest.approx(0.06))
    assert slope == pytest.approx((0.16 - 0.12) / 1.0)  # per 1% of T over [h/2, h] = 1% wide


def test_audit_standard_errors():
    se = fe.audit_se({"J_G": 0.2, "J_V": 0.3, "FPR": 0.1, "FNR": 0.05}, n=256)
    assert se["J_G"] == pytest.approx(np.sqrt(0.2 * 0.8 / 256))
    assert se["FPR"] == pytest.approx(np.sqrt(0.1 * 0.9 / (256 * 0.8)))
    assert se["FNR"] == pytest.approx(np.sqrt(0.05 * 0.95 / (256 * 0.2)))


def test_l2_feature_ses_combine_endpoints():
    vals = {k: np.array([0.2, 0.25, 0.3]) for k in ("J_G", "J_V", "FPR", "FNR")}
    se = fe.l2_feature_se(vals, h_frac=0.05, n=256)
    assert se.shape == (12,)
    s_h = np.sqrt(0.3 * 0.7 / 256)
    s_0 = np.sqrt(0.2 * 0.8 / 256)
    assert se[0] == pytest.approx(s_h) and se[1] == pytest.approx(np.hypot(s_h, s_0))


def _series(rng, c=3):
    keys = ("J_G", "J_V", "FPR", "FNR", "A_u", "alpha_u", "C_u", "alpha_r", "C_in", "C_out")
    s = {k: rng.uniform(0.05, 0.6, c) for k in keys}
    s["FPM"] = (1 - s["J_G"]) * s["FPR"]
    return s


def test_finite_levels_match_the_oracle_levels_and_add_the_reward_arms():
    rng = np.random.default_rng(0)
    s = _series(rng)
    ctx = rng.uniform(0, 1, (3, 4))
    idx = (0, 1, 2)
    fin = fe.finite_levels(s, ctx, 0.02, idx)
    obs = {k: s[k] for k in ("J_G", "J_V", "FPR", "FNR")}
    geo = {"A": s["A_u"], "alpha": s["alpha_u"], "C": s["C_u"], "C_in": s["C_in"],
           "C_out": s["C_out"]}  # fmt: skip
    ora = fe.level_features(obs, geo, ctx, 0.02, idx)
    for lv in fe.LEVELS:
        np.testing.assert_allclose(fin[lv], ora[lv], rtol=1e-12)
        assert len(fin[lv]) == len(fe.names(lv))
    np.testing.assert_allclose(fin["L1r"], np.append(fin["L1"], s["alpha_r"][0]))
    np.testing.assert_allclose(fin["L3r"][:-3], fin["L3"])
    np.testing.assert_allclose(fin["L3r"][-3:], fe.summaries(s["alpha_r"], 0.02))
    assert len(fin["L1r"]) == len(fe.names("L1r")) and len(fin["L3r"]) == len(fe.names("L3r"))


def test_finite_levels_keep_missing_values():
    rng = np.random.default_rng(1)
    s = _series(rng)
    s["FNR"][2] = np.nan
    s["alpha_u"][0] = np.nan
    fin = fe.finite_levels(s, rng.uniform(0, 1, (3, 4)), 0.02, (0, 1, 2))
    names = fe.names("L3")
    got = dict(zip(names, fin["L3"], strict=True))
    assert np.isnan(got["FNR_h"]) and np.isnan(got["FNR_d"]) and np.isnan(got["FNR_slope"])
    assert np.isnan(got["alpha_d"]) and np.isfinite(got["alpha_slope"])
    assert np.isnan(dict(zip(fe.names("L1"), fin["L1"], strict=True))["alpha0"])
