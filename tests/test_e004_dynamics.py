"""E004a training dynamics (registry E004a §5-§6): sampled GRPO-lite Adam, MF-Adam, NG flow."""

import numpy as np
import pytest
import torch

from vdyn.e004 import dynamics as dy
from vdyn.e004 import panel as pn
from vdyn.e004 import toy


@pytest.fixture(scope="module")
def structs():
    return pn.design_panel(n_per=1)


def test_adam_step_matches_torch(rng):
    theta = rng.normal(size=(3, toy.D))
    grads = [rng.normal(size=(3, toy.D)) for _ in range(5)]
    state = dy.AdamState.zeros(3)
    th = theta.copy()
    ref = torch.tensor(theta.copy(), requires_grad=True)
    opt = torch.optim.Adam([ref], lr=0.01, betas=(0.9, 0.999), eps=1e-8, maximize=True)
    for g in grads:
        th = dy.adam_step(th, g, state, lr=0.01)
        opt.zero_grad()
        ref.grad = torch.tensor(g)
        opt.step()
    np.testing.assert_allclose(th, ref.detach().numpy(), rtol=1e-12, atol=1e-14)


def test_grpo_estimator_tends_to_the_population_normalized_gradient(structs):
    st = structs[10]  # D1
    tb = toy.Tables.of([st])
    th = st.theta0[None]
    rng = [np.random.default_rng(1)]
    est = np.mean([dy.grpo_gradient(tb, th, rng, n_prompts=64, n_resp=256)[0]
                   for _ in range(20)], axis=0)  # fmt: skip
    exact = toy.exact(tb, th)["g_eff"][0]
    assert np.max(np.abs(est - exact)) < 0.03 * np.max(np.abs(exact)) + 2e-3


def test_sampled_runs_do_not_depend_on_batching(structs):
    seeds = [(1, 2, 3), (4, 5, 6)]
    both = dy.run_sampled_adam(structs[:2], seeds, steps=30, checkpoints=[0, 10, 30])
    one = dy.run_sampled_adam(structs[1:2], seeds[1:], steps=30, checkpoints=[0, 10, 30])
    np.testing.assert_array_equal(both["theta"][1], one["theta"][0])
    np.testing.assert_array_equal(both["v_hat"][1], one["v_hat"][0])


def test_mf_adam_step_is_the_registered_map(structs):
    st = structs[4]
    tb = toy.Tables.of([st])
    out = dy.run_mf_adam([st], steps=1, checkpoints=[0, 1])
    ex = toy.exact(tb, st.theta0[None])
    g, var = ex["g_eff"][0], ex["var_eff"][0]
    want = st.theta0 + 0.01 * g / np.sqrt(g**2 + var / 64 + 1e-16)
    np.testing.assert_allclose(out["theta"][0, 1], want, rtol=1e-12)


def test_ng_flow_is_invariant_to_coupling(structs):
    st = structs[5]  # YA2, coupled
    a = dy.run_ng([st], t_end=3.0, checkpoints=[0.0, 1.0, 3.0])
    b = dy.run_ng([pn.uncoupled_twin(st)], t_end=3.0, checkpoints=[0.0, 1.0, 3.0])
    pa = toy.probs(toy.Tables.of([st] * 3), a["theta"][0])
    pb = toy.probs(toy.Tables.of([pn.uncoupled_twin(st)] * 3), b["theta"][0])
    np.testing.assert_allclose(pa, pb, atol=1e-7)


def test_ng_flow_ascends_the_proxy(structs):
    st = structs[11]
    out = dy.run_ng([st], t_end=5.0, checkpoints=list(np.linspace(0, 5, 11)))
    jv = toy.exact(toy.Tables.of([st] * 11), out["theta"][0])["J_V"]
    assert np.all(np.diff(jv) > -1e-9)


def test_geometry_at_checkpoints_uses_the_optimizer_metric(structs):
    st = structs[8]  # B1
    tb = toy.Tables.of([st])
    th = st.theta0[None]
    v = np.full((1, toy.D), 0.04)
    geo = dy.geometry(tb, th, "adam", v_hat=v)
    ex = toy.exact(tb, th)
    ref = toy.decompose(ex["g_G"][0], ex["g_eff"][0], 1 / (np.sqrt(v[0]) + 1e-8), ex["g_G_ctx"][0])
    for k in ("A", "alpha", "C", "C_in", "C_out"):
        assert geo[k][0] == pytest.approx(ref[k], rel=1e-10)


def test_geometry2_reports_reward_and_update_levels(structs):
    st = structs[4]  # YA1
    tb = toy.Tables.of([st])
    th = st.theta0[None]
    v = np.full((1, toy.D), 0.03)
    ex = toy.exact(tb, th)
    M = 1 / (np.sqrt(v[0]) + 1e-8)
    g2 = dy.geometry2(tb, th, "adam", v_hat=v)
    upd = toy.decompose(ex["g_effG"][0], ex["g_eff"][0], M, ex["g_effG_ctx"][0])
    rew = toy.decompose(ex["g_G"][0], ex["g_V"][0], M, ex["g_G_ctx"][0])
    for k in ("A", "alpha", "C", "C_in", "C_out"):
        assert g2[f"{k}_u"][0] == pytest.approx(upd[k], rel=1e-10)
        assert g2[f"{k}_r"][0] == pytest.approx(rew[k], rel=1e-10)
    ng = dy.geometry2(tb, th, "ng")
    for k in ("A", "alpha", "C", "C_in", "C_out"):
        assert ng[f"{k}_u"][0] == pytest.approx(ng[f"{k}_r"][0], rel=1e-12)
