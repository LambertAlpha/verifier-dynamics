"""E005a-R null test, E3 wrapper and projected oracle (research/09_e005ar_design.md §2.3, §2.6)."""

import numpy as np
import pytest

from vdyn.e005 import estimators as es
from vdyn.e005ar import stats as st


def null_data(rng, R, n, k, alpha=-0.4, A=1.0, C=0.0, noise=1.0, skew=False):
    """Groups with mean h_G = A e1 and h_e = alpha h_G + C e2 (C = 0: aligned null)."""
    hG = np.zeros(k)
    hG[0] = A
    he = alpha * hG
    if C:
        he[1] = C
    eps_G = rng.standard_normal((R, n, k)) * noise
    eps_d = rng.standard_normal((R, n, k)) * noise
    if skew:  # heavy right tail, still mean zero
        eps_d = (rng.exponential(1.0, (R, n, k)) - 1.0) * noise
    xG = hG + eps_G
    xe = he + alpha * eps_G + eps_d
    return xG, xe


def test_statistic_is_the_off_diagonal_sum_of_residualized_projections():
    rng = np.random.default_rng(0)
    xG, xe = null_data(rng, 3, 7, 5)
    out = st.signflip(xG, xe, np.random.default_rng(1), B=19)
    for r in range(3):
        hG, he = xG[r].mean(0), xe[r].mean(0)
        a = he @ hG / (hG @ hG)
        u = hG / np.linalg.norm(hG)
        z = (xe[r] - a * xG[r]) - np.outer((xe[r] - a * xG[r]) @ u, u)
        T = sum(z[i] @ z[j] for i in range(7) for j in range(7) if i != j)
        assert out["T"][r] == pytest.approx(T, rel=1e-10)
        assert out["p"][r] * 20 == pytest.approx(round(out["p"][r] * 20))
        assert 1 / 20 <= out["p"][r] <= 1


@pytest.mark.parametrize("k,skew", [(8, False), (2, True), (32, False)])
def test_sign_flip_test_is_calibrated_at_the_aligned_null(k, skew):
    rng = np.random.default_rng(k)
    xG, xe = null_data(rng, 3000, 24, k, skew=skew)
    rej = st.signflip(xG, xe, np.random.default_rng(7), B=199)["reject"]
    assert 0.03 <= rej.mean() <= 0.07, rej.mean()


def test_sign_flip_test_has_power_and_the_legacy_test_is_conservative():
    rng = np.random.default_rng(3)
    xG, xe = null_data(rng, 400, 32, 8, C=0.8)
    power = st.signflip(xG, xe, np.random.default_rng(1), B=199)["reject"].mean()
    assert power > 0.8
    assert power > st.wald_reject(st.e3(xG, xe)).mean() + 0.1
    x0G, x0e = null_data(np.random.default_rng(4), 2000, 32, 8)
    e3 = st.e3(x0G, x0e)
    assert st.wald_reject(e3).mean() < 0.03  # E005a lesson: jackknife SE too large at C = 0


def test_crossfit_statistic_adds_folds_with_independent_flips():
    rng = np.random.default_rng(5)
    a = null_data(rng, 50, 10, 4)
    b = null_data(rng, 50, 12, 4)
    ka, kb = st.gram_residual(*a), st.gram_residual(*b)
    out = st.signflip_from_grams([ka, kb], np.random.default_rng(2), B=99)
    ta = ka.sum((1, 2)) - np.trace(ka, axis1=1, axis2=2)
    tb = kb.sum((1, 2)) - np.trace(kb, axis1=1, axis2=2)
    assert np.allclose(out["T"], ta + tb)


def test_e3_wrapper_matches_the_frozen_e005a_estimator():
    xG, xe = null_data(np.random.default_rng(8), 5, 20, 6, C=0.3)
    ref = es.estimate_all(xG, xe)["E3"]
    out = st.e3(xG, xe)
    for q in ("C2", "SE_C2", "A2", "alpha"):
        assert np.allclose(out[q], ref[q], equal_nan=True)


def test_projection_preserves_alignment_and_only_loses_signal():
    rng = np.random.default_rng(9)
    d = 30
    hG = rng.standard_normal(d)
    c = rng.standard_normal(d)
    c -= (c @ hG) / (hG @ hG) * hG
    for _ in range(20):
        Q = np.linalg.qr(rng.standard_normal((d, 7)))[0]
        null = st.projected_geometry(hG, -0.3 * hG, Q)
        assert null["C2"] == pytest.approx(0.0, abs=1e-12)
        alt = st.projected_geometry(hG, -0.3 * hG + c, Q)
        assert alt["C2"] <= c @ c + 1e-12
