"""E005a calibration panel and audit sampling (research/07_e005a_measurement.md §3, §4)."""

import numpy as np
import pytest

from vdyn.e004 import toy
from vdyn.e005 import panel as pl
from vdyn.e005 import sampling as sm


@pytest.fixture(scope="module")
def bases():
    return pl.bases("design", limit=4)


def _point(base, variant="theta_0", rho=1.0, d_extra=0, a=0.8, b=0.05):
    return pl.make_point(base, variant, rho, d_extra, a, b, dose="x")


def test_affine_null_has_zero_c_in_every_metric(bases):
    for st in bases:
        p = _point(st, rho=0.0, a=0.7, b=0.1)
        for metric in pl.METRICS:
            o = p["oracle"][metric]
            assert abs(o["C2"]) < 1e-12 * max(1.0, o["A2"]), (st.sid, metric)
            assert o["alpha"] == pytest.approx(0.7 - 1, abs=1e-9)


def test_c_is_exactly_linear_in_the_dose(bases):
    st = bases[2]
    full = _point(st, rho=1.0)["oracle"]["I"]
    for rho in (0.1, 0.35, 0.8):
        o = _point(st, rho=rho)["oracle"]["I"]
        assert np.sqrt(o["C2"]) == pytest.approx(rho * np.sqrt(full["C2"]), rel=1e-9)
    rho = pl.dose_for(st, "theta_0", 0.5 * np.sqrt(full["C2"]), a=0.8, b=0.05)
    assert rho == pytest.approx(0.5, rel=1e-9)


def test_nuisance_dimensions_leave_the_identity_geometry_unchanged(bases):
    st = bases[1]
    p0, p56 = _point(st, d_extra=0), _point(st, d_extra=56)
    for q in ("A2", "alpha", "C2"):
        assert p56["oracle"]["I"][q] == pytest.approx(p0["oracle"]["I"][q], rel=1e-12)
    ex = pl.exact(p56)
    assert np.allclose(ex["g_G"][toy.D :], 0) and np.allclose(ex["g_V"][toy.D :], 0)
    np.testing.assert_allclose(ex["F"][toy.D :, toy.D :], 0.25 * np.eye(56))
    assert np.allclose(ex["F"][: toy.D, toy.D :], 0)


def test_policy_variants(bases):
    st = bases[0]
    a0 = _point(st)["oracle"]["I"]["A2"]
    low = _point(st, variant="theta_lowA")["oracle"]["I"]["A2"]
    assert low < 1e-3 * a0
    th = pl.variant_theta(st.theta0, "theta_hot")
    np.testing.assert_allclose(th[: toy.K + 1], 0.5 * st.theta0[: toy.K + 1])


def test_group_contributions_are_unbiased_and_the_fisher_is_consistent(bases):
    p = _point(bases[3], rho=0.6, d_extra=4)
    ex = pl.exact(p)
    rng = np.random.default_rng(0)
    au = sm.audit(p, rng, R=1, N=8 * 40000, m=8)
    gG, gV = au["xG"][0], au["xV"][0]
    se = gG.std(0) / np.sqrt(len(gG))
    assert np.all(np.abs(gG.mean(0) - ex["g_G"]) < 5 * se + 1e-12)
    se = gV.std(0) / np.sqrt(len(gV))
    assert np.all(np.abs(gV.mean(0) - ex["g_V"]) < 5 * se + 1e-12)
    F_hat = au["scores_u"][0].T @ au["scores_u"][0] / au["scores_u"].shape[1]
    np.testing.assert_allclose(F_hat, ex["F"], atol=0.01)


def test_panel_is_reproducible_and_matched_sets_share_the_oracle_c():
    a = pl.build_panel("design", limit=3)
    b = pl.build_panel("design", limit=3)
    assert a["points"] == b["points"]
    groups: dict[str, list[float]] = {}
    for p in a["points"]:
        if p["matched"] is not None:
            groups.setdefault(p["matched"], []).append(p["oracle"]["I"]["C2"])
    assert groups
    for vals in groups.values():
        assert np.ptp(vals) < 1e-10 * max(1.0, max(vals))
    assert {p["dose"] for p in a["points"]} <= set(pl.DOSES) | {"natural"}
