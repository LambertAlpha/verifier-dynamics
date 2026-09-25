"""R: random symmetric flips (theory note §3), and the fixed-flip assumption boundary."""

import numpy as np
import pytest
from scipy.special import expit

from vdyn.geometry import autodiff, closed_form
from vdyn.geometry.decompose import decompose
from vdyn.policies import bernoulli
from vdyn.verifiers import toy

GOLD = toy.reward_table(toy.gold)


def _metrics(theta: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    a = rng.normal(size=(2, 2))
    return [
        np.linalg.inv(autodiff.fisher(bernoulli.log_prob, theta)),
        np.eye(2),
        a @ a.T + 0.1 * np.eye(2),
    ]


@pytest.mark.parametrize("p", [0.0, 0.1, 0.25, 0.5, 0.7])
def test_expected_r_gradient_is_attenuated_gold_gradient(p, logit_points):
    table = toy.reward_table(toy.r_expected(p))
    for theta in logit_points:
        g_gold = autodiff.reward_gradient(bernoulli.log_prob, theta, GOLD)
        g_r = autodiff.reward_gradient(bernoulli.log_prob, theta, table)
        np.testing.assert_allclose(g_r, (1 - 2 * p) * g_gold, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.7])
def test_r_has_alpha_minus_two_p_and_no_pressure_in_any_metric(p, logit_points, rng):
    table = toy.reward_table(toy.r_expected(p))
    for theta in logit_points[:81]:
        g_gold = autodiff.reward_gradient(bernoulli.log_prob, theta, GOLD)
        g_r = autodiff.reward_gradient(bernoulli.log_prob, theta, table)
        for metric in _metrics(theta, rng):
            dec = decompose(g_gold, g_r, metric)
            assert dec.alpha == pytest.approx(float(closed_form.r_alpha(p)), rel=1e-9)
            assert dec.C <= 1e-9 * dec.A


def test_monte_carlo_r_gradient_matches_attenuated_gold_gradient(rng):
    p, n = 0.2, 400_000
    for theta in (np.array([0.3, -1.1]), np.array([-1.5, 2.0])):
        q, s = expit(theta)
        corr, z = bernoulli.sample(theta, n, rng)
        rewards = toy.r_sample(corr, z, p, rng)
        per_sample = bernoulli.score(theta, corr, z) * rewards[:, None]  # REINFORCE, no baseline
        estimate = per_sample.mean(axis=0)
        stderr = per_sample.std(axis=0, ddof=1) / np.sqrt(n)
        expected = (1 - 2 * p) * closed_form.grad_gold(q, s)
        assert np.all(np.abs(estimate - expected) < 5 * stderr + 1e-12), (
            estimate,
            expected,
            stderr,
        )


def _fisher_decomposition(theta: np.ndarray, table: np.ndarray):
    g_gold = autodiff.reward_gradient(bernoulli.log_prob, theta, GOLD)
    g_ver = autodiff.reward_gradient(bernoulli.log_prob, theta, table)
    return decompose(g_gold, g_ver, np.linalg.inv(autodiff.fisher(bernoulli.log_prob, theta)))


@pytest.mark.parametrize("flipped", [[(0, 1)], [(1, 0)]])
def test_single_fixed_flips_are_policy_controllable(flipped, logit_points):
    """Flipping one outcome deterministically gives C > 0 everywhere on the open square."""
    table = toy.reward_table(toy.fixed_flip(flipped))
    for theta in logit_points[:81]:
        assert _fisher_decomposition(theta, table).C > 0


def test_xor_fixed_flip_pressure_vanishes_only_at_q_one_half(logit_points):
    """Flipping {(0,1), (1,1)} gives V = corr XOR z; [derived-agent] C = |1-2q| sqrt(s(1-s)).

    So fixed flips are policy-controllable generically, not at every policy.
    """
    table = toy.reward_table(toy.fixed_flip([(0, 1), (1, 1)]))
    for theta in logit_points[:81]:
        q, s = expit(theta)
        dec = _fisher_decomposition(theta, table)
        assert dec.C == pytest.approx(abs(1 - 2 * q) * np.sqrt(s * (1 - s)), rel=1e-9, abs=1e-15)
        assert dec.alpha == pytest.approx(-2 * s, rel=1e-9)


def test_matched_accuracy_verifiers_differ_in_pressure():
    """Same static accuracy under the current policy, different local geometry.

    Fixed flip of (corr=0, z=1) errs with probability (1-q)s; R with p = (1-q)s errs equally often.
    """
    theta = np.array([-0.4, -1.2])
    q, s = expit(theta)
    p = (1 - q) * s
    fixed = toy.reward_table(toy.fixed_flip([(0, 1)]))
    rand = toy.reward_table(toy.r_expected(p))
    probs = autodiff.outcome_probs(bernoulli.log_prob, theta)
    assert probs @ (fixed != GOLD) == pytest.approx(p, rel=1e-12)
    assert probs @ np.abs(rand - GOLD) == pytest.approx(p, rel=1e-12)

    g_gold = autodiff.reward_gradient(bernoulli.log_prob, theta, GOLD)
    fisher_metric = np.linalg.inv(autodiff.fisher(bernoulli.log_prob, theta))
    dec_fixed = decompose(
        g_gold, autodiff.reward_gradient(bernoulli.log_prob, theta, fixed), fisher_metric
    )
    dec_rand = decompose(
        g_gold, autodiff.reward_gradient(bernoulli.log_prob, theta, rand), fisher_metric
    )
    assert dec_fixed.C == pytest.approx(float(closed_form.pressure(q, s)), rel=1e-9)
    assert dec_rand.C < 1e-12
    assert dec_fixed.alpha != pytest.approx(dec_rand.alpha, rel=1e-3)
