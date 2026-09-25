"""X: signal deletion (theory note §4), per prompt and aggregated over prompts."""

import numpy as np
import pytest
from scipy.special import expit

from vdyn.geometry import autodiff, closed_form
from vdyn.geometry.decompose import decompose
from vdyn.policies import bernoulli
from vdyn.verifiers import toy

GOLD = toy.reward_table(toy.gold)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_constant_verifier_deletes_the_gradient(value, logit_points):
    table = toy.reward_table(toy.constant(value))
    for theta in logit_points:
        g_gold = autodiff.reward_gradient(bernoulli.log_prob, theta, GOLD)
        g_ver = autodiff.reward_gradient(bernoulli.log_prob, theta, table)
        np.testing.assert_allclose(g_ver, 0.0, atol=1e-15)
        for metric in (np.linalg.inv(autodiff.fisher(bernoulli.log_prob, theta)), np.eye(2)):
            dec = decompose(g_gold, g_ver, metric)
            assert dec.alpha == pytest.approx(-1.0, abs=1e-9)
            assert dec.C <= 1e-9 * dec.A


def _multi_prompt_case(rng: np.random.Generator):
    n_prompts, deleted = 6, np.array([True, True, False, False, False, False])
    weights = rng.dirichlet(np.ones(n_prompts))
    theta = rng.uniform(-3.0, 3.0, size=2 * n_prompts)
    log_prob = bernoulli.multi_prompt_log_prob(weights)
    gold_table = toy.multi_prompt_table([toy.gold] * n_prompts)
    ver_table = toy.multi_prompt_table([toy.constant(1.0) if d else toy.gold for d in deleted])
    g_gold = autodiff.reward_gradient(log_prob, theta, gold_table)
    g_ver = autodiff.reward_gradient(log_prob, theta, ver_table)
    fisher = autodiff.fisher(log_prob, theta)
    return weights, theta, deleted, g_gold, g_ver, fisher


def test_aggregate_deletion_matches_closed_form_with_positive_pressure(rng):
    """Proposition 6: C_agg > 0 although no prompt carries a new direction."""
    for _ in range(10):
        weights, theta, deleted, g_gold, g_ver, fisher = _multi_prompt_case(rng)
        q = expit(theta[0::2])
        signal_sq = weights * q * (1 - q)  # natural metric: A_x^2 = w_x q_x (1 - q_x)
        dec = decompose(g_gold, g_ver, np.linalg.inv(fisher))
        alpha_agg, c_agg = closed_form.x_aggregate(
            signal_sq[deleted].sum(), signal_sq[~deleted].sum()
        )
        assert dec.alpha == pytest.approx(alpha_agg, rel=1e-9)
        assert dec.C == pytest.approx(c_agg, rel=1e-9)
        assert dec.C > 0


def test_aggregate_deletion_formula_holds_in_euclidean_metric(rng):
    for _ in range(10):
        weights, theta, deleted, g_gold, g_ver, _ = _multi_prompt_case(rng)
        q = expit(theta[0::2])
        signal_sq = (weights * q * (1 - q)) ** 2  # Euclidean: A_x^2 = ||w_x q_x(1-q_x)||^2
        dec = decompose(g_gold, g_ver, np.eye(len(theta)))
        alpha_agg, c_agg = closed_form.x_aggregate(
            signal_sq[deleted].sum(), signal_sq[~deleted].sum()
        )
        assert dec.alpha == pytest.approx(alpha_agg, rel=1e-9)
        assert dec.C == pytest.approx(c_agg, rel=1e-9)


def test_per_prompt_blocks_recover_deletion_and_clean_signal(rng):
    weights, theta, deleted, g_gold, g_ver, fisher = _multi_prompt_case(rng)
    for k, is_deleted in enumerate(deleted):
        block = slice(2 * k, 2 * k + 2)
        metric = np.linalg.inv(fisher[block, block])
        dec = decompose(g_gold[block], g_ver[block], metric)
        assert dec.alpha == pytest.approx(-1.0 if is_deleted else 0.0, abs=1e-9)
        assert dec.C <= 1e-9 * dec.A
