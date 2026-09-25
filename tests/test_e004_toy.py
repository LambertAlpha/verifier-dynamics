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
