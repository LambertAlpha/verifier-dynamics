"""Closed-form Fisher vs exact score outer product vs negative expected Hessian vs Monte Carlo."""

import numpy as np
from scipy.special import expit

from vdyn.geometry import autodiff, closed_form
from vdyn.policies import bernoulli


def test_closed_form_fisher_matches_score_outer_product(logit_points):
    for theta in logit_points:
        q, s = expit(theta)
        np.testing.assert_allclose(
            autodiff.fisher(bernoulli.log_prob, theta),
            closed_form.fisher(q, s),
            rtol=1e-9,
            atol=1e-15,
        )


def test_closed_form_fisher_matches_negative_expected_hessian(logit_points):
    for theta in logit_points:
        q, s = expit(theta)
        np.testing.assert_allclose(
            autodiff.fisher_from_hessian(bernoulli.log_prob, theta),
            closed_form.fisher(q, s),
            rtol=1e-9,
            atol=1e-15,
        )


def test_monte_carlo_fisher_matches_closed_form(rng):
    n = 400_000
    for theta in (np.array([0.3, -1.1]), np.array([-2.0, 2.5]), np.array([1.7, 0.0])):
        q, s = expit(theta)
        corr, z = bernoulli.sample(theta, n, rng)
        sc = bernoulli.score(theta, corr, z)
        outer = sc[:, :, None] * sc[:, None, :]
        estimate = outer.mean(axis=0)
        stderr = outer.std(axis=0, ddof=1) / np.sqrt(n)
        err = np.abs(estimate - closed_form.fisher(q, s))
        assert np.all(err < 5 * stderr + 1e-12), (theta, err, stderr)
