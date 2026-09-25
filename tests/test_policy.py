import numpy as np
import torch
from scipy.special import expit
from torch.func import jacrev

from vdyn.policies import bernoulli


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


def test_outcome_order_is_corr_then_z():
    np.testing.assert_array_equal(bernoulli.OUTCOMES, [[0, 0], [0, 1], [1, 0], [1, 1]])


def test_outcome_probabilities_are_product_of_independent_bernoullis(logit_points):
    for u, v in logit_points:
        # expit(-x) instead of 1 - expit(x): avoids cancellation near the boundary.
        q, qc, s, sc = expit(u), expit(-u), expit(v), expit(-v)
        p = torch.exp(bernoulli.log_prob(_t(np.array([u, v])))).numpy()
        expected = [qc * sc, qc * s, q * sc, q * s]
        np.testing.assert_allclose(p, expected, rtol=1e-12, atol=0)


def test_mean_parameterization_matches_logit_parameterization(logit_points):
    for u, v in logit_points[:81]:
        lp_logit = bernoulli.log_prob(_t(np.array([u, v])))
        lp_mean = bernoulli.log_prob_mean(_t(np.array([expit(u), expit(v)])))
        np.testing.assert_allclose(lp_mean.numpy(), lp_logit.numpy(), rtol=1e-10, atol=1e-12)


def test_hand_written_score_matches_autograd_score(logit_points):
    corr, z = bernoulli.OUTCOMES[:, 0], bernoulli.OUTCOMES[:, 1]
    for theta in logit_points:
        autograd_score = jacrev(bernoulli.log_prob)(_t(theta)).numpy()
        hand_score = bernoulli.score(theta, corr, z)
        np.testing.assert_allclose(hand_score, autograd_score, rtol=1e-12, atol=1e-15)


def test_sample_frequencies_match_probabilities(rng):
    theta = np.array([0.4, -1.3])
    q, s = bernoulli.probs(theta)
    n = 400_000
    corr, z = bernoulli.sample(theta, n, rng)
    for draws, p in ((corr, q), (z, s), (corr * z, q * s)):
        se = np.sqrt(p * (1 - p) / n)
        assert abs(draws.mean() - p) < 5 * se


def test_multi_prompt_log_prob_is_weighted_product_of_prompt_policies():
    weights = np.array([0.5, 0.3, 0.2])
    theta = np.array([0.1, -2.0, 1.5, 0.3, -0.7, 2.2])
    lp = bernoulli.multi_prompt_log_prob(weights)(_t(theta)).numpy()
    assert lp.shape == (12,)
    np.testing.assert_allclose(np.exp(lp).sum(), 1.0, rtol=1e-13)
    for k in range(3):
        single = torch.exp(bernoulli.log_prob(_t(theta[2 * k : 2 * k + 2]))).numpy()
        np.testing.assert_allclose(np.exp(lp[4 * k : 4 * k + 4]), weights[k] * single, rtol=1e-12)
