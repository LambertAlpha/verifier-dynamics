"""Numerical checks of research/paper/theory.md (E006): fresh false positives only rescale the
gold gradient (also for finite groups); consistent false positives add the verifier's own
gradient; cross-prompt coherence separates master keys from per-prompt tables; zero-variance
prompts are absorbing; the mean-field critical coverage."""

import numpy as np
import pytest

from vdyn.e006 import theory as th


def master_key_model(n_prompts: int, seed: int = 0):
    """Each prompt: gold g_x, a shared master key m (common feature e_m + own feature), two
    prompt-specific wrong responses. Features are one-hot over a global index."""
    rng = np.random.default_rng(seed)
    per = 4  # responses per prompt: gold, master key, wrong1, wrong2
    dim = 1 + n_prompts * per  # index 0 = shared e_m
    phi = np.zeros((n_prompts, per, dim))
    for x in range(n_prompts):
        for y in range(per):
            phi[x, y, 1 + x * per + y] = 1.0
        phi[x, 1, 0] = 1.0  # the master key carries the shared feature
    G = np.zeros((n_prompts, per))
    G[:, 0] = 1.0
    theta = rng.normal(0, 0.5, dim)
    return phi, G, theta


def per_prompt_model(n_prompts: int, seed: int = 0):
    """The same, but no shared feature: every feature is prompt-specific (orthogonal)."""
    phi, G, theta = master_key_model(n_prompts, seed)
    phi[:, :, 0] = 0.0
    return phi, G, theta


def test_fresh_false_positives_rescale_each_prompt_gold_gradient_exactly():
    phi, G, theta = master_key_model(6)
    w = np.full(6, 1 / 6)
    for f in (0.0, 0.1, 0.4):
        EV = f + (1 - f) * G
        u = th.expected_update(theta, phi, w, EV)
        gold = th.per_prompt_grad(theta, phi, G)  # (n_prompts, dim): grad J_G,x
        lam = th.prop1_weights(theta, phi, w, G, f)
        assert np.allclose(u, (lam[:, None] * gold).sum(0), atol=1e-12)
        assert (lam > 0).all()


def test_finite_group_update_for_one_prompt_is_parallel_to_its_gold_gradient():
    phi, G, theta = master_key_model(1, seed=3)
    rng = np.random.default_rng(0)
    gold = th.per_prompt_grad(theta, phi, G)[0]
    u = th.monte_carlo_group_update(theta, phi[0], G[0], f=0.2, group=8, n_groups=200_000, rng=rng)
    cos = u @ gold / (np.linalg.norm(u) * np.linalg.norm(gold))
    assert cos > 0.995


def test_consistent_false_positives_add_the_verifier_gradient():
    phi, G, theta = master_key_model(5)
    w = np.full(5, 0.2)
    V = G.copy()
    V[:, 1] = 1.0  # master key accepted everywhere
    u = th.expected_update(theta, phi, w, V)
    s = th.prompt_std(theta, phi, V)
    gold = th.per_prompt_grad(theta, phi, G)
    fp = th.per_prompt_grad(theta, phi, V * (1 - G))
    assert np.allclose(u, ((w / s)[:, None] * (gold + fp)).sum(0), atol=1e-12)


def test_coherence_grows_with_prompts_for_a_master_key_and_is_one_for_per_prompt_tables():
    kap_mk, kap_pp = [], []
    for n in (4, 16, 64):
        phi, G, theta = master_key_model(n, seed=1)
        V = G.copy()
        V[:, 1] = 1.0
        kap_mk.append(th.fp_coherence(theta, phi, np.full(n, 1 / n), G, V))
        phi2, G2, theta2 = per_prompt_model(n, seed=1)
        kap_pp.append(th.fp_coherence(theta2, phi2, np.full(n, 1 / n), G2, V))
    assert kap_mk[0] < kap_mk[1] < kap_mk[2] and kap_mk[2] > 10
    assert np.allclose(kap_pp, 1.0, atol=1e-9)


def test_zero_variance_prompts_contribute_nothing():
    phi, G, theta = master_key_model(3)
    V = G.copy()
    V[0, :] = 1.0  # prompt 0: every response accepted
    u_all = th.expected_update(theta, phi, np.full(3, 1 / 3), V)
    w = np.array([0.0, 1 / 3, 1 / 3])
    assert np.allclose(u_all, th.expected_update(theta, phi, w, V), atol=1e-12)


def test_critical_coverage_mean_field():
    r = th.fill_rate(c=0.5, f0=0.108, q=0.108)
    assert r == pytest.approx(0.108 * 0.5 / (1 - 0.054))
    c_star = th.critical_coverage(vbar=0.455, f0=0.108, q=0.108)
    assert 0.40 < c_star < 0.44
    assert (
        th.drift_sign(c_star - 0.05, 0.455, 0.108, 0.108)
        < 0
        < th.drift_sign(c_star + 0.05, 0.455, 0.108, 0.108)
    )
