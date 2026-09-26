"""E004 U-toy (registry E004a §1): exact enumeration, analytic scores against an independent
torch-autodiff reference, closed forms, and the geometry decomposition."""

import numpy as np
import pytest
import torch

from vdyn.e004 import toy

torch.set_default_dtype(torch.float64)


def _random_structure(rng, **over):
    kw = dict(
        w=rng.dirichlet(np.full(4, 2.0)),
        p=rng.uniform(0.3, 0.95, 4),
        theta0=np.concatenate([rng.normal(-1, 1, 4), [rng.normal(-2, 1)], rng.normal(-1, 1, 3)]),
        lam=rng.normal(0, 0.5, 3),
        event="and2",
        trig=np.array([1.0, 1.0, 0.0, 1.0]),
        fp=rng.uniform(0, 0.1, 4),
        fn=rng.uniform(0, 0.1, 4),
        rho=rng.uniform(0, 0.5, 4),
        beta=rng.uniform(0, 0.5, 4),
        deleted=np.array([False, False, True, False]),
        v0=0.4,
    )
    kw.update(over)
    return toy.Structure(sid="t", construction="T", mechanism="T", **kw)


def _torch_logp(st: toy.Structure, theta: torch.Tensor) -> torch.Tensor:
    """Independent reference: log P(row | x) for the table rows, built directly from the model."""
    rows = toy.ROWS
    K = len(st.w)
    u, h, phi = theta[:K], theta[K], theta[K + 1 :]
    ls = torch.log_softmax(torch.stack([u, h.expand(K), torch.zeros(K)], 1), 1)
    zl = phi + torch.tensor(st.lam) * u.mean()
    out = torch.zeros(K, len(rows))
    for r, (s, xi, zbits) in enumerate(rows):
        lp = ls[:, s]
        if s == 0:
            pt = torch.tensor(st.p)
            lp = lp + (torch.log(pt) if xi else torch.log1p(-pt))
        zt = torch.tensor(zbits, dtype=torch.float64)
        ls_on, ls_off = torch.nn.functional.logsigmoid(zl), torch.nn.functional.logsigmoid(-zl)
        lz = (zt * ls_on + (1 - zt) * ls_off).sum()
        out[:, r] = lp + lz
    return out


def _torch_J(st, theta):
    P = torch.exp(_torch_logp(st, theta))
    G, EV = torch.tensor(toy.gold_table(st)), torch.tensor(toy.verifier_table(st))
    w = torch.tensor(st.w)[:, None]
    jg, jv = (w * P * G).sum(), (w * P * EV).sum()
    fpr = (w * P * (1 - G) * EV).sum() / (w * P * (1 - G)).sum()
    fnr = (w * P * G * (1 - EV)).sum() / jg
    return jg, jv, fpr, fnr


def test_row_probabilities_normalize(rng):
    st = _random_structure(rng)
    P = toy.row_probs(st, np.asarray(st.theta0))
    np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=1e-12)
    assert np.all(P >= 0)


def test_scores_gradients_and_fisher_match_torch_autodiff(rng):
    for _ in range(5):
        st = _random_structure(rng)
        th = np.asarray(st.theta0) + rng.normal(0, 0.3, 8)
        ex = toy.single(st, th)
        tt = torch.tensor(th, requires_grad=True)
        jg, jv, fpr, fnr = _torch_J(st, tt)
        for name, val in (("g_G", jg), ("g_V", jv), ("g_FPR", fpr), ("g_FNR", fnr)):
            ref = torch.autograd.grad(val, tt, retain_graph=True)[0].numpy()
            np.testing.assert_allclose(ex[name], ref, rtol=1e-9, atol=1e-12, err_msg=name)
        for name, val in (("J_G", jg), ("J_V", jv), ("FPR", fpr), ("FNR", fnr)):
            assert ex[name] == pytest.approx(val.item(), rel=1e-12), name
        jac = torch.autograd.functional.jacobian(
            lambda t, st=st: _torch_logp(st, t), torch.tensor(th)
        )
        P = torch.exp(_torch_logp(st, torch.tensor(th)))
        F = torch.einsum("k,kr,kri,krj->ij", torch.tensor(st.w), P, jac, jac).numpy()
        np.testing.assert_allclose(ex["F"], F, rtol=1e-9, atol=1e-12)


def test_per_prompt_gold_gradients_sum_to_the_gold_gradient(rng):
    st = _random_structure(rng)
    ex = toy.single(st, np.asarray(st.theta0))
    np.testing.assert_allclose(ex["g_G_ctx"].sum(axis=0), ex["g_G"], rtol=1e-12, atol=1e-15)


def test_symmetric_coins_scale_the_gold_gradient(rng):
    eps = 0.2
    st = _random_structure(
        rng,
        event="none",
        trig=np.zeros(4),
        rho=np.zeros(4),
        beta=np.zeros(4),
        deleted=np.zeros(4, bool),
        fp=np.full(4, eps),
        fn=np.full(4, eps),
    )
    ex = toy.single(st, np.asarray(st.theta0))
    np.testing.assert_allclose(ex["g_V"], (1 - 2 * eps) * ex["g_G"], rtol=1e-10, atol=1e-14)


def test_attempt_credit_gives_the_benign_alpha(rng):
    p, beta = 0.6, 0.5
    st = _random_structure(rng, event="none", trig=np.zeros(4), rho=np.zeros(4),
                           beta=np.full(4, beta), fp=np.zeros(4), fn=np.zeros(4),
                           deleted=np.zeros(4, bool), p=np.full(4, p), lam=np.zeros(3))  # fmt: skip
    ex = toy.single(st, np.asarray(st.theta0))
    geo = toy.decompose(ex["g_G"], ex["g_V"], np.linalg.inv(ex["F"]), ex["g_G_ctx"])
    assert geo["alpha"] == pytest.approx((1 - p) * beta / p, rel=1e-8)
    assert geo["C"] == pytest.approx(0.0, abs=1e-8)


def test_canonical_deletion_residual_lies_in_the_gold_span(rng):
    st = _random_structure(rng, event="none", trig=np.zeros(4), rho=np.zeros(4), beta=np.zeros(4),
                           fp=np.zeros(4), fn=np.zeros(4), lam=np.zeros(3), v0=1.0,
                           deleted=np.array([True, False, False, False]))  # fmt: skip
    ex = toy.single(st, np.asarray(st.theta0))
    geo = toy.decompose(ex["g_G"], ex["g_V"], np.linalg.inv(ex["F"]), ex["g_G_ctx"])
    assert geo["C"] > 1e-3
    assert geo["C_out"] == pytest.approx(0.0, abs=1e-10)


def test_observable_identity_for_canonical_feature_exploits(rng):
    """Design fact F1: alpha = -FPR, A^2 = J_G'/(1-FPR), C^2 = (1-J_G) FPR' under NG."""
    for event in ("single", "or2", "and2", "and3"):
        st = _random_structure(rng, event=event, trig=np.ones(4), rho=np.zeros(4), beta=np.zeros(4),
                               fp=np.zeros(4), fn=np.zeros(4), lam=np.zeros(3),
                               deleted=np.zeros(4, bool), p=np.full(4, 1 - 1e-12))  # fmt: skip
        ex = toy.single(st, np.asarray(st.theta0))
        Minv = np.linalg.inv(ex["F"])
        vel = Minv @ ex["g_V"]
        geo = toy.decompose(ex["g_G"], ex["g_V"], Minv, ex["g_G_ctx"])
        assert geo["alpha"] == pytest.approx(-ex["FPR"], rel=1e-8, abs=1e-12)
        assert geo["A"] ** 2 == pytest.approx(ex["g_G"] @ vel / (1 - ex["FPR"]), rel=1e-8)
        assert geo["C"] ** 2 == pytest.approx((1 - ex["J_G"]) * ex["g_FPR"] @ vel, rel=1e-7)


def test_grpo_effective_gradient_and_second_moment(rng):
    st = _random_structure(rng)
    th = np.asarray(st.theta0)
    ex = toy.single(st, th)
    P, S = toy.row_probs(st, th), toy.row_scores(st, th)
    EV = toy.verifier_table(st)
    b = (P * EV).sum(1)
    sd = np.sqrt(b * (1 - b))
    g = np.einsum("k,kr,kri->i", st.w, P * (EV - b[:, None]) / sd[:, None], S)
    np.testing.assert_allclose(ex["g_eff"], g, rtol=1e-12, atol=1e-15)
    second = np.einsum("k,kr,kri->i", st.w, P * (EV * (1 - 2 * b[:, None]) + b[:, None] ** 2)
                       / sd[:, None] ** 2, S**2)  # fmt: skip
    np.testing.assert_allclose(ex["var_eff"], second - g**2, rtol=1e-10, atol=1e-15)


# --- Amendment 3 (Stage 0b) -------------------------------------------------------------------


def _plain(rng, **over):
    base = dict(event="none", trig=np.zeros(4), rho=np.zeros(4), beta=np.zeros(4), fp=np.zeros(4),
                fn=np.zeros(4), deleted=np.zeros(4, bool), lam=np.zeros(3))  # fmt: skip
    base.update(over)
    return _random_structure(rng, **base)


def test_threshold_event_is_two_of_three():
    ev = toy.event("thr23")
    z = toy.ROW_Z
    np.testing.assert_array_equal(ev, z.sum(axis=1) >= 2)


def test_process_credit_and_feature_triggered_hack(rng):
    st = _plain(rng, beta=np.full(4, 0.5), credit_event="z3", rho=np.full(4, 0.7), hack_event="z1")
    ev = toy.verifier_table(st)
    s, xi, z = toy.ROW_S, toy.ROW_XI, toy.ROW_Z
    failed = (s == toy.SOLVE) & (xi == 0)
    np.testing.assert_allclose(ev[0, failed & (z[:, 2] == 1)], 0.5)
    np.testing.assert_allclose(ev[0, failed & (z[:, 2] == 0)], 0.0)
    hack = s == toy.HACK
    np.testing.assert_allclose(ev[0, hack & (z[:, 0] == 1)], 0.7)
    np.testing.assert_allclose(ev[0, hack & (z[:, 0] == 0)], 0.0)


def test_preference_gap_closed_forms(rng):
    p = np.array([0.8, 0.5, 0.9, 0.3])
    # feature exploit accepted always, correct answers rejected w.p. fn: gap = p * fn (inverted)
    st = _plain(rng, p=p, event="single", trig=np.ones(4), fn=np.full(4, 0.2))
    np.testing.assert_allclose(toy.preference_gaps(st), p * 0.2, rtol=1e-12)
    assert toy.axis_b(st) == "INVERTED"
    # symmetric coins eps: gap = p (2 eps - 1) < 0 (aligned)
    st = _plain(rng, p=p, fp=np.full(4, 0.1), fn=np.full(4, 0.1))
    np.testing.assert_allclose(toy.preference_gaps(st), p * (2 * 0.1 - 1), rtol=1e-12)
    assert toy.axis_b(st) == "ALIGNED"
    # hack accepted w.p. rho: gap = rho - p (no coins)
    st = _plain(rng, p=p, rho=np.full(4, 0.6))
    np.testing.assert_allclose(toy.preference_gaps(st), 0.6 - p, rtol=1e-12)
    # deleted prompts carry no preference
    st = _plain(rng, p=p, rho=np.array([0.0, 0.9, 0.0, 0.0]), deleted=np.array([0, 1, 0, 0], bool))
    assert np.isnan(toy.preference_gaps(st)[1]) and toy.axis_b(st) == "ALIGNED"


def test_inversion_needs_a_relevant_prompt(rng):
    w = np.array([0.02, 0.38, 0.3, 0.3])
    st = _plain(rng, w=w, p=np.full(4, 0.5), rho=np.array([0.9, 0.0, 0.0, 0.0]))
    assert toy.axis_b(st) == "ALIGNED"  # only a prompt with w < 0.05 is inverted


def test_normalized_gold_direction(rng):
    st = _random_structure(rng)
    th = np.asarray(st.theta0)
    ex = toy.single(st, th)
    P, S, G = toy.row_probs(st, th), toy.row_scores(st, th), toy.gold_table(st)
    b = (P * G).sum(1)
    sd = np.sqrt(b * (1 - b))
    per = np.einsum("k,kr,kri->ki", st.w, P * (G - b[:, None]) / sd[:, None], S)
    np.testing.assert_allclose(ex["g_effG_ctx"], per, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(ex["g_effG"], per.sum(0), rtol=1e-12, atol=1e-15)


def test_reward_level_alpha_of_symmetric_coins_is_minus_two_eps(rng):
    eps = 0.15
    st = _plain(rng, fp=np.full(4, eps), fn=np.full(4, eps), lam=rng.normal(0, 0.5, 3))
    ex = toy.single(st, np.asarray(st.theta0))
    for M in (np.linalg.inv(ex["F"]), rng.uniform(0.5, 2, toy.D)):
        geo = toy.decompose(ex["g_G"], ex["g_V"], M)
        assert geo["alpha"] == pytest.approx(-2 * eps, abs=1e-10)
