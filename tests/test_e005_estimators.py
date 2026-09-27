"""E005a estimators (research/07_e005a_measurement.md §2): plug-in, U-statistic (E1), first-order
corrected plug-in (E2), group jackknife (E3), legacy E0, and the leading-order bias theory."""

import numpy as np
import pytest

from vdyn.e005 import estimators as es


def _gaussian_groups(rng, R, n, h_G, h_e, cov):
    """R replications of n i.i.d. group contributions (G, e) with means h_G, h_e and joint cov."""
    d = len(h_G)
    z = rng.multivariate_normal(np.r_[h_G, h_e], cov, size=(R, n))
    return z[..., :d], z[..., d:]


def _joint_cov(rng, d, scale=1.0):
    B = rng.normal(size=(2 * d, 2 * d)) * scale
    return B @ B.T / (2 * d)


def test_ustat_gram_is_plugin_minus_trace_over_n():
    rng = np.random.default_rng(0)
    xG, xe = rng.normal(0.8, 1, size=(3, 12, 5)), rng.normal(size=(3, 12, 5))  # A_U^2 > 0
    out = es.estimate_all(xG, xe)
    n = 12
    for r in range(3):
        mG, me = xG[r].mean(0), xe[r].mean(0)
        S = np.cov(np.c_[xG[r], xe[r]].T)
        trGG, treG = np.trace(S[:5, :5]), np.trace(S[5:, :5])
        assert out["E1"]["A2"][r] == pytest.approx(mG @ mG - trGG / n, rel=1e-12)
        pu = me @ mG - treG / n
        assert out["E1"]["alpha"][r] == pytest.approx(pu / out["E1"]["A2"][r], rel=1e-10)
        # E2 differs from E1 only in C^2
        assert out["E2"]["A2"][r] == pytest.approx(out["E1"]["A2"][r], rel=1e-12)
        assert out["E2"]["alpha"][r] == pytest.approx(out["E1"]["alpha"][r], rel=1e-12)


def test_leave_one_out_values_match_brute_force():
    rng = np.random.default_rng(1)
    xG, xe = rng.normal(0.4, 1, size=(2, 9, 4)), rng.normal(0.1, 1, size=(2, 9, 4))
    loo = es.leave_one_out(xG, xe)
    for k in range(9):
        keep = np.arange(9) != k
        ref = es.estimate_all(xG[:, keep], xe[:, keep], with_se=False)
        for name in ("plugin", "E1", "E2"):
            for q in ("A2", "alpha", "C2"):
                np.testing.assert_allclose(loo[name][q][:, k], ref[name][q], rtol=1e-9, atol=1e-12)


def test_jackknife_of_the_mean_is_the_mean_with_the_usual_se():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 20))
    loo = (x.sum(1, keepdims=True) - x) / 19
    est, se = es.jackknife(x.mean(1), loo)
    np.testing.assert_allclose(est, x.mean(1), rtol=1e-12)
    np.testing.assert_allclose(se, x.std(1, ddof=1) / np.sqrt(20), rtol=1e-10)


def test_gram_entries_are_unbiased():
    rng = np.random.default_rng(3)
    d = 4
    h_G, h_e = np.array([0.5, 0.2, 0.0, 0.1]), np.array([-0.2, 0.3, 0.1, 0.0])
    cov = _joint_cov(rng, d)
    xG, xe = _gaussian_groups(rng, 20000, 6, h_G, h_e, cov)
    out = es.estimate_all(xG, xe, with_se=False)
    g = out["E1"]["gram"]
    for key, truth in (("GG", h_G @ h_G), ("eG", h_e @ h_G), ("ee", h_e @ h_e)):
        se = g[key].std() / np.sqrt(len(g[key]))
        assert abs(g[key].mean() - truth) < 4 * se, key


def test_null_biases_follow_the_leading_order_theory():
    """C = 0: plug-in bias tr(P_perp S_d)/n; U-statistic bias -u'S_d u/n; S_d = Cov(e - alpha G)."""
    rng = np.random.default_rng(4)
    d, n, R = 6, 64, 40000
    u = np.zeros(d)
    u[0] = 1.0
    A, alpha = 1.2, -0.3
    h_G, h_e = A * u, alpha * A * u  # C = 0
    cov = _joint_cov(rng, d, 0.8)
    cov[0, 0] += 2.0  # sizeable variance along u so the U-statistic bias is visible
    xG, xe = _gaussian_groups(rng, R, n, h_G, h_e, cov)
    out = es.estimate_all(xG, xe, with_se=False)
    S_G, S_e, S_eG = cov[:d, :d], cov[d:, d:], cov[d:, :d]
    S_d = S_e - alpha * (S_eG + S_eG.T) + alpha**2 * S_G
    pred_plug = (np.trace(S_d) - u @ S_d @ u) / n
    pred_u = -(u @ S_d @ u) / n
    plug, ust = out["plugin"]["C2"], out["E1"]["C2"]
    assert plug.mean() == pytest.approx(pred_plug, rel=0.08)
    assert abs(ust.mean() - pred_u) < 0.25 * abs(pred_u) + 3 * ust.std() / np.sqrt(R)
    for name in ("E2", "E3"):
        c2 = out[name]["C2"]
        assert abs(c2.mean()) < 0.2 * pred_plug, name  # first-order bias removed


def test_first_order_corrections_work_away_from_the_null():
    rng = np.random.default_rng(5)
    d, n, R = 5, 64, 30000
    h_G = np.array([1.0, 0.2, 0.0, 0.0, 0.0])
    h_e = np.array([-0.3, 0.1, 0.35, -0.2, 0.0])
    cov = _joint_cov(rng, d, 0.7)
    xG, xe = _gaussian_groups(rng, R, n, h_G, h_e, cov)
    out = es.estimate_all(xG, xe, with_se=False)
    A2 = h_G @ h_G
    C2 = h_e @ h_e - (h_e @ h_G) ** 2 / A2
    bias_plug = out["plugin"]["C2"].mean() - C2
    for name in ("E2", "E3"):
        b = out[name]["C2"].mean() - C2
        assert abs(b) < 0.25 * abs(bias_plug), name


def test_undefined_alpha_when_the_gold_gram_is_not_positive():
    xG = np.array([[[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0], [-0.5, 0.0]]])  # mean 0: A_U^2 < 0
    xe = np.array([[[0.2, 0.3], [0.1, -0.2], [0.0, 0.4], [0.3, 0.1]]])
    out = es.estimate_all(xG, xe, with_se=False)
    assert out["E1"]["A2"][0] <= 0 and np.isnan(out["E1"]["alpha"][0])
    assert out["E1"]["C2"][0] == pytest.approx(out["E1"]["gram"]["ee"][0])


def test_whitening_reproduces_metric_inner_products():
    rng = np.random.default_rng(6)
    F = _joint_cov(rng, 3) + 0.2 * np.eye(6)
    for kind in ("diag", "full"):
        W = es.metric_sqrt(F[None], kind)[0]
        M = np.diag(1 / (np.diag(F) + 0.1 * np.diag(F).mean())) if kind == "diag" else \
            np.linalg.inv(F + 0.1 * np.trace(F) / 6 * np.eye(6))  # fmt: skip
        g = rng.normal(size=6)
        assert (W @ g) @ (W @ g) == pytest.approx(g @ M @ g, rel=1e-10)
    s = rng.normal(size=(2, 500, 6))
    np.testing.assert_allclose(es.fisher_hat(s)[0], s[0].T @ s[0] / 500, rtol=1e-12)


def test_legacy_e0_pools_the_verifier_and_its_jackknife_runs_over_all_groups():
    rng = np.random.default_rng(7)
    aG, aV = rng.normal(size=(2, 8, 3)), rng.normal(size=(2, 8, 3))
    uV = rng.normal(size=(2, 5, 3))
    out = es.legacy_e0(aG, aV, uV)
    gG = aG.mean(1)
    gV = np.concatenate([aV, uV], 1).mean(1)
    ge = gV - gG
    for r in range(2):
        a2 = gG[r] @ gG[r]
        assert out["A2"][r] == pytest.approx(a2)
        assert out["C2"][r] == pytest.approx(ge[r] @ ge[r] - (ge[r] @ gG[r]) ** 2 / a2)
    assert out["SE_C2"].shape == (2,) and np.all(out["SE_C2"] > 0)
