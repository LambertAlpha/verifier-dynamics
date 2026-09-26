"""E004a Stage 1 finite-sample measurement (execution note §2): audits and plug-in estimators."""

import numpy as np
import pytest

from vdyn.e004 import audit as au
from vdyn.e004 import dynamics as dy
from vdyn.e004 import panel0b as pb
from vdyn.e004 import toy


@pytest.fixture(scope="module")
def structs():
    return pb.design_panel(n_per=1, yb_cap=0.2)


def _setup(structs, which):
    sts = [structs[i] for i in which]
    tb = toy.Tables.of(sts)
    th = np.stack([s.theta0 for s in sts])
    return sts, tb, th


def test_large_audit_observables_match_the_exact_values(structs):
    sts, tb, th = _setup(structs, [3, 9, 20])  # X, YA, D-type cells
    rngs = [np.random.default_rng(k) for k in range(3)]
    aud = au.sample_groups(tb, th, rngs, n_groups=6000)
    obs = au.observables(tb, aud)
    ex = toy.exact(tb, th)
    n = 6000 * au.N_RESP
    for k in ("J_G", "J_V"):
        se = np.sqrt(ex[k] * (1 - ex[k]) / n)
        assert np.all(np.abs(obs[k] - ex[k]) < 5 * se + 1e-12), k
    se_fpr = np.sqrt(ex["FPR"] * (1 - ex["FPR"]) / (n * (1 - ex["J_G"])))
    assert np.all(np.abs(obs["FPR"] - ex["FPR"]) < 5 * se_fpr + 1e-12)
    se_fnr = np.sqrt(ex["FNR"] * (1 - ex["FNR"]) / (n * ex["J_G"]))
    assert np.all(np.abs(obs["FNR"] - ex["FNR"]) < 5 * se_fnr + 1e-12)
    np.testing.assert_allclose(obs["FPM"], (1 - obs["J_G"]) * obs["FPR"], rtol=1e-12)


def test_j_v_pools_the_training_batch(structs):
    sts, tb, th = _setup(structs, [0, 1])
    aud = au.sample_groups(tb, th, [np.random.default_rng(1), np.random.default_rng(2)])
    _, batch = dy.grpo_gradient(tb, th, [np.random.default_rng(3), np.random.default_rng(4)],
                                return_batch=True)  # fmt: skip
    train = au.Groups(batch["x"], batch["row"], batch["V"])
    obs = au.observables(tb, aud, train)
    want = (aud.V.sum((1, 2)) + train.V.sum((1, 2))) / (aud.V[0].size + train.V[0].size)
    np.testing.assert_allclose(obs["J_V"], want, rtol=1e-14)
    assert aud.x.shape == (2, au.N_GROUPS) and aud.V.shape == (2, au.N_GROUPS, au.N_RESP)


def test_update_level_direction_is_the_grpo_step_on_the_same_groups(structs):
    sts, tb, th = _setup(structs, [4, 12, 21])
    rngs = lambda: [np.random.default_rng(k) for k in (7, 8, 9)]  # noqa: E731
    g_v, batch = dy.grpo_gradient(tb, th, rngs(), return_batch=True)
    grp = au.Groups(batch["x"], batch["row"], batch["V"])
    s = au.rollout_scores(tb, th, grp)
    np.testing.assert_allclose(au.direction(au.grpo_advantage(grp.V), s), g_v, atol=1e-15)
    g_g = dy.grpo_gradient(tb.clean(), th, rngs())  # clean tables: V = G on the same rows
    gold = au.gold(tb, grp)
    np.testing.assert_allclose(au.direction(au.grpo_advantage(gold), s), g_g, atol=1e-15)


def test_reward_level_rloo_is_unbiased(structs):
    sts, tb, th = _setup(structs, [8, 16])
    rngs = [np.random.default_rng(k) for k in (11, 12)]
    aud = au.sample_groups(tb, th, rngs, n_groups=20000)
    s = au.rollout_scores(tb, th, aud)
    ex = toy.exact(tb, th)
    for name, R in (("g_G", au.gold(tb, aud)), ("g_V", aud.V)):
        per = au.rloo_advantage(R)[..., None] * s  # (n, g, m, D)
        est = per.mean(axis=(1, 2))
        se = per.mean(axis=2).std(axis=1) / np.sqrt(per.shape[1])
        assert np.all(np.abs(est - ex[name]) < 5 * se + 1e-10), name


def test_adam_t0_metric_is_mean_square_plus_variance_over_eight():
    gam = np.array([[[1.0, 2.0], [3.0, 0.0], [2.0, 1.0]]])  # (n=1, groups=3, D=2)
    v = au.adam_v0(gam)
    want = gam.mean(1) ** 2 + gam.var(1, ddof=1) / 8
    np.testing.assert_allclose(v, want)


def test_ng_metric_is_the_damped_inverse_audit_fisher(structs):
    sts, tb, th = _setup(structs, [2])
    aud = au.sample_groups(tb, th, [np.random.default_rng(5)], n_groups=20000)
    s = au.rollout_scores(tb, th, aud)
    F_hat = au.fisher(s)
    np.testing.assert_allclose(F_hat[0], toy.exact(tb, th)["F"][0], atol=0.02)
    M = au.ng_metric(F_hat)
    lam = au.NG_LAM * np.trace(F_hat[0]) / toy.D
    np.testing.assert_allclose(M[0], np.linalg.inv(F_hat[0] + lam * np.eye(toy.D)), rtol=1e-10)


def test_estimates_are_consistent_and_levels_coincide_under_ng(structs):
    sts, tb, th = _setup(structs, [5, 13, 22])
    rngs = [np.random.default_rng(k) for k in (21, 22, 23)]
    aud = au.sample_groups(tb, th, rngs)
    _, batch = dy.grpo_gradient(tb, th, [np.random.default_rng(k) for k in (1, 2, 3)],
                                return_batch=True)  # fmt: skip
    train = au.Groups(batch["x"], batch["row"], batch["V"])
    est = au.estimate(tb, th, aud, train, "adam", v_hat=None)
    ok = est["alpha_u_defined"]
    np.testing.assert_allclose((est["C_in"] ** 2 + est["C_out"] ** 2)[ok], (est["C_u"] ** 2)[ok],
                               rtol=1e-8, atol=1e-14)  # fmt: skip
    ng = au.estimate(tb, th, aud, None, "ng")
    for k in ("A", "alpha", "C"):
        np.testing.assert_array_equal(ng[f"{k}_u"], ng[f"{k}_r"])
    v = np.full((3, toy.D), 0.02)
    with_state = au.estimate(tb, th, aud, train, "adam", v_hat=v)
    s_all = au.rollout_scores(tb, th, au.concat(aud, train))
    g_v = au.direction(au.grpo_advantage(au.concat(aud, train).V), s_all)
    s_aud = au.rollout_scores(tb, th, aud)
    g_g = au.direction(au.grpo_advantage(au.gold(tb, aud)), s_aud)
    ref = toy.decompose(g_g, g_v, 1 / (np.sqrt(v) + 1e-8))
    for k in ("A", "alpha", "C"):
        np.testing.assert_allclose(with_state[f"{k}_u"], ref[k], rtol=1e-12)


def test_zero_gold_direction_leaves_alpha_undefined(structs):
    sts, tb, th = _setup(structs, [0])
    hack = int(np.flatnonzero(toy.ROW_S == toy.HACK)[0])
    other = int(np.flatnonzero(toy.ROW_S == toy.OTHER)[0])
    rows = np.where(np.arange(32).reshape(1, 4, 8) % 2 == 0, hack, other)  # wrong answers only
    aud = au.Groups(np.zeros((1, 4), int), rows, (rows == hack).astype(float))
    est = au.estimate(tb, th, aud, None, "ng")
    assert est["A_u"][0] == 0 and np.isnan(est["alpha_u"][0]) and not est["alpha_u_defined"][0]
    assert est["C_u"][0] > 0
    obs = au.observables(tb, aud)
    assert obs["J_G"][0] == 0 and np.isnan(obs["FNR"][0])  # no correct answer: FNR undefined
