"""Path C: REINFORCE estimates from seeded samples vs closed-form gradients (Y toy)."""

import numpy as np
from scipy.special import expit

from vdyn.geometry import closed_form
from vdyn.policies import bernoulli
from vdyn.verifiers import toy

THETAS = (np.array([0.3, -1.1]), np.array([-2.0, 2.5]), np.array([1.7, 0.0]))


def _reinforce(theta, verifier, n, rng):
    corr, z = bernoulli.sample(theta, n, rng)
    per_sample = bernoulli.score(theta, corr, z) * verifier(corr, z)[:, None]
    return per_sample.mean(axis=0), per_sample.std(axis=0, ddof=1) / np.sqrt(n)


def test_reinforce_matches_closed_form_gold_and_verifier_gradients(rng):
    n = 400_000
    for theta in THETAS:
        q, s = expit(theta)
        for verifier, closed in (
            (toy.gold, closed_form.grad_gold),
            (toy.y_false_positive, closed_form.grad_verifier),
        ):
            estimate, stderr = _reinforce(theta, verifier, n, rng)
            assert np.all(np.abs(estimate - closed(q, s)) < 5 * stderr + 1e-12), (theta, verifier)


def test_monte_carlo_objectives_match_closed_form(rng):
    n = 400_000
    for theta in THETAS:
        q, s = expit(theta)
        corr, z = bernoulli.sample(theta, n, rng)
        for verifier, closed in (
            (toy.gold, closed_form.j_gold),
            (toy.y_false_positive, closed_form.j_verifier),
        ):
            r = verifier(corr, z)
            assert abs(r.mean() - closed(q, s)) < 5 * r.std(ddof=1) / np.sqrt(n)
