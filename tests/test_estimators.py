"""E002 geometry estimators from rollouts (registry E002 §5 G1, §10)."""

import numpy as np
import pytest
from scipy.special import logit

from vdyn.e002 import estimators as est
from vdyn.geometry import autodiff
from vdyn.geometry.decompose import decompose
from vdyn.policies import bernoulli
from vdyn.verifiers import boolean_fp as bf

ST = bf.EventStructure("mix", "MIX", (0.3, 0.4), coin=0.05, base="AND2", rho=0.6)
Q = 0.2


def _exact(st: bf.EventStructure, q: float):
    lp = bernoulli.product_log_prob(1 + st.n_features)
    theta = np.concatenate([[logit(q)], logit(np.asarray(st.s0))])
    g_gold = autodiff.reward_gradient(lp, theta, bf.gold_table(st))
    g_ver = autodiff.reward_gradient(lp, theta, bf.verifier_table(st))
    fisher = autodiff.fisher(lp, theta)
    return g_gold, g_ver, fisher


def test_rloo_gradients_are_unbiased(rng):
    g_gold, g_ver, _ = _exact(ST, Q)
    batch = est.sample_rollouts(ST, Q, np.asarray(ST.s0), n=16, reps=40_000, rng=rng)
    sc = est.score_vectors(Q, np.asarray(ST.s0), batch.corr, batch.z)
    for rewards, exact in ((batch.corr.astype(float), g_gold), (batch.v, g_ver)):
        g = est.rloo(rewards, sc)
        se = g.std(axis=0, ddof=1) / np.sqrt(len(g))
        assert np.all(np.abs(g.mean(axis=0) - exact) < 5 * se + 1e-12)


def test_estimated_fisher_is_unbiased(rng):
    _, _, fisher = _exact(ST, Q)
    batch = est.sample_rollouts(ST, Q, np.asarray(ST.s0), n=32, reps=20_000, rng=rng)
    f_hat = est.fisher_hat(est.score_vectors(Q, np.asarray(ST.s0), batch.corr, batch.z))
    se = f_hat.std(axis=0, ddof=1) / np.sqrt(len(f_hat))
    assert np.all(np.abs(f_hat.mean(axis=0) - fisher) < 5 * se + 1e-12)


def test_batched_decomposition_matches_decompose(rng):
    reps, d = 50, 3
    g_gold, g_ver = rng.normal(size=(reps, d)), rng.normal(size=(reps, d))
    g_gold[3] = 0.0  # one degenerate replication
    a = rng.normal(size=(reps, d, d))
    metric = a @ np.swapaxes(a, 1, 2) + 0.1 * np.eye(d)
    out = est.decompose_batch(g_gold, g_ver, metric)
    for r in range(reps):
        ref = decompose(g_gold[r], g_ver[r], metric[r])
        assert out["alpha_defined"][r] == ref.alpha_defined
        assert out["C"][r] == pytest.approx(ref.C, rel=1e-10)
        assert out["A"][r] == pytest.approx(ref.A, rel=1e-10, abs=1e-300)
        if ref.alpha_defined:
            assert out["alpha"][r] == pytest.approx(ref.alpha, rel=1e-10)
        else:
            assert np.isnan(out["alpha"][r])


def test_ustatistic_gram_is_unbiased_under_the_exact_metric(rng):
    g_gold, g_ver, fisher = _exact(ST, Q)
    metric = np.linalg.inv(fisher)
    g_err = g_ver - g_gold
    batch = est.sample_rollouts(ST, Q, np.asarray(ST.s0), n=12, reps=60_000, rng=rng)
    sc = est.score_vectors(Q, np.asarray(ST.s0), batch.corr, batch.z)
    gam_g = batch.corr[..., None] * sc
    gam_e = (batch.v - batch.corr)[..., None] * sc
    m = np.broadcast_to(metric, (len(sc), *metric.shape))
    for ga, gb, exact in (
        (gam_g, gam_g, g_gold @ metric @ g_gold),
        (gam_e, gam_g, g_err @ metric @ g_gold),
        (gam_e, gam_e, g_err @ metric @ g_err),
    ):
        vals = est.ustat_gram(ga, gb, m)
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        assert abs(vals.mean() - exact) < 5 * se


def test_zero_gold_signal_gives_undefined_alpha(rng):
    st = bf.EventStructure("s", "SINGLE", (0.3,))
    batch = est.sample_rollouts(st, 0.0005, np.asarray(st.s0), n=8, reps=200, rng=rng)
    out = est.geometry(st, batch, None, 0.0005, np.asarray(st.s0), "plugin", 1e-2, "paired")
    none_correct = batch.corr.sum(axis=1) == 0
    assert none_correct.any()
    assert np.all(~out["alpha_defined"][none_correct])
    assert np.all(np.isnan(out["alpha"][none_correct]))
    assert np.all(np.isfinite(out["C2"]))


@pytest.mark.parametrize("n", [4, 16])
def test_exact_degenerate_event_probabilities(n, rng):
    st = bf.EventStructure("a", "AND2", (0.3, 0.1))
    q, f = 0.2, bf.fpr(st, np.asarray(st.s0))
    batch = est.sample_rollouts(st, q, np.asarray(st.s0), n=n, reps=100_000, rng=rng)
    probs = est.degenerate_event_probabilities(q, f, np.asarray(st.s0), n)
    obs = {
        "all_gold_equal": np.mean(np.ptp(batch.corr, axis=1) == 0),
        "no_false_positive": np.mean(((1 - batch.corr) * batch.v).sum(axis=1) == 0),
        "some_feature_constant": np.mean(np.any(np.ptp(batch.z, axis=1) == 0, axis=1)),
    }
    for key, p in probs.items():
        assert abs(obs[key] - p) < 5 * np.sqrt(p * (1 - p) / 100_000) + 1e-12, key


def test_geometry_variants_have_expected_shapes_and_order(rng):
    labeled = est.sample_rollouts(ST, Q, np.asarray(ST.s0), n=32, reps=10, rng=rng)
    unlabeled = est.sample_rollouts(ST, Q, np.asarray(ST.s0), n=96, reps=10, rng=rng)
    for estimator, source in (("plugin", "paired"), ("plugin", "pooled"), ("ustat", "paired")):
        out = est.geometry(ST, labeled, unlabeled, Q, np.asarray(ST.s0), estimator, 1e-2, source)
        for key in ("A", "alpha", "alpha_defined", "C", "C2"):
            assert out[key].shape == (10,)
    oracle = est.geometry(ST, labeled, unlabeled, Q, np.asarray(ST.s0), "plugin", 0.0, "paired",
                          exact_fisher=_exact(ST, Q)[2])  # fmt: skip
    assert oracle["C2"].shape == (10,)


def test_rloo_matches_its_definition():
    rewards = np.array([[1.0, 0.0, 0.0, 1.0]])
    sc = np.array([[[1.0], [2.0], [3.0], [4.0]]])
    baseline = np.array([1 / 3, 2 / 3, 2 / 3, 1 / 3])  # mean of the other three rewards
    expected = np.mean((rewards[0] - baseline) * sc[0, :, 0])
    np.testing.assert_allclose(est.rloo(rewards, sc), [[expected]], rtol=1e-12)


def test_ustat_returns_raw_gram_entries(rng):
    labeled = est.sample_rollouts(ST, Q, np.asarray(ST.s0), n=24, reps=7, rng=rng)
    fisher = _exact(ST, Q)[2]
    out = est.geometry(ST, labeled, None, Q, np.asarray(ST.s0), "ustat", 0.0, "paired",
                       exact_fisher=fisher)  # fmt: skip
    sc = est.score_vectors(Q, np.asarray(ST.s0), labeled.corr, labeled.z)
    m = np.broadcast_to(np.linalg.inv(fisher), (7, *fisher.shape))
    gam_g = labeled.corr[..., None] * sc
    gam_e = (labeled.v - labeled.corr)[..., None] * sc
    np.testing.assert_allclose(out["gram_gg"], est.ustat_gram(gam_g, gam_g, m), rtol=1e-12)
    np.testing.assert_allclose(out["gram_eg"], est.ustat_gram(gam_e, gam_g, m), rtol=1e-12)
    np.testing.assert_allclose(out["gram_ee"], est.ustat_gram(gam_e, gam_e, m), rtol=1e-12)
