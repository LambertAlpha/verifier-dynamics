"""Y toy static quantities.

Closed form (path A) vs enumeration + autodiff (path B) vs finite differences.
"""

import numpy as np
from scipy.special import expit

from vdyn.geometry import autodiff, closed_form
from vdyn.geometry.decompose import decompose
from vdyn.policies import bernoulli
from vdyn.verifiers import toy

GOLD = toy.reward_table(toy.gold)
Y = toy.reward_table(toy.y_false_positive)


def _close(actual, desired) -> None:
    np.testing.assert_allclose(actual, desired, rtol=1e-9, atol=1e-14)


def _autodiff_gradients(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        autodiff.reward_gradient(bernoulli.log_prob, theta, GOLD),
        autodiff.reward_gradient(bernoulli.log_prob, theta, Y),
    )


def test_objectives_match_enumeration(logit_points):
    for theta in logit_points:
        q, s = expit(theta)
        j_gold = autodiff.expected_reward(bernoulli.log_prob, theta, GOLD)
        j_ver = autodiff.expected_reward(bernoulli.log_prob, theta, Y)
        _close(closed_form.j_gold(q, s), j_gold)
        _close(closed_form.j_verifier(q, s), j_ver)
        np.testing.assert_allclose(closed_form.gap(q, s), j_ver - j_gold, rtol=1e-8, atol=1e-14)


def test_gradients_match_autodiff(logit_points):
    for theta in logit_points:
        q, s = expit(theta)
        g_gold, g_ver = _autodiff_gradients(theta)
        _close(closed_form.grad_gold(q, s), g_gold)
        _close(closed_form.grad_verifier(q, s), g_ver)
        _close(closed_form.grad_error(q, s), g_ver - g_gold)


def test_closed_form_gradients_match_central_finite_differences(logit_points):
    """Checks the hand derivation against the closed-form objective, without torch."""
    h = 1e-6
    for theta in logit_points[:81]:
        q, s = expit(theta)
        for objective, grad in (
            (closed_form.j_gold, closed_form.grad_gold),
            (closed_form.j_verifier, closed_form.grad_verifier),
        ):
            fd = np.array(
                [
                    (objective(*expit(theta + h * e)) - objective(*expit(theta - h * e))) / (2 * h)
                    for e in np.eye(2)
                ]
            )
            np.testing.assert_allclose(grad(q, s), fd, rtol=1e-6, atol=1e-10)


def test_decomposition_of_autodiff_quantities_matches_closed_form(logit_points):
    for theta in logit_points:
        q, s = expit(theta)
        g_gold, g_ver = _autodiff_gradients(theta)
        metric = np.linalg.inv(autodiff.fisher(bernoulli.log_prob, theta))
        dec = decompose(g_gold, g_ver, metric)
        _close(dec.A, closed_form.signal(q, s))
        _close(dec.alpha, closed_form.alpha(q, s))
        _close(dec.b, closed_form.alpha(q, s) * closed_form.signal(q, s))
        _close(dec.C, closed_form.pressure(q, s))


def test_explicitly_whitened_vectors_match_closed_form(logit_points):
    """h = F^{-1/2} g with the symmetric root from an eigendecomposition of the autodiff Fisher."""
    for theta in logit_points:
        q, s = expit(theta)
        g_gold, g_ver = _autodiff_gradients(theta)
        evals, evecs = np.linalg.eigh(autodiff.fisher(bernoulli.log_prob, theta))
        inv_root = evecs @ np.diag(evals**-0.5) @ evecs.T
        h_gold, h_err = inv_root @ g_gold, inv_root @ (g_ver - g_gold)
        _close(h_gold, closed_form.h_gold(q, s))
        _close(h_err, closed_form.h_error(q, s))
        a = closed_form.alpha(q, s)
        _close(h_err - a * h_gold, closed_form.h_residual(q, s))
        assert abs(closed_form.h_residual(q, s) @ closed_form.h_gold(q, s)) < 1e-15


def test_pressure_is_positive_on_the_open_square(logit_points):
    for theta in logit_points:
        q, s = expit(theta)
        assert closed_form.pressure(q, s) > 0
        g_gold, g_ver = _autodiff_gradients(theta)
        metric = np.linalg.inv(autodiff.fisher(bernoulli.log_prob, theta))
        assert decompose(g_gold, g_ver, metric).C > 0


def test_pressure_vanishes_at_the_boundary():
    eps = 1e-14
    for q, s in ((0.3, eps), (0.3, 1 - eps), (1 - eps, 0.4)):
        assert closed_form.pressure(q, s) < 1e-6
        assert closed_form.euclidean_pressure(q, s) < 1e-12


def test_euclidean_diagnostics_match_closed_form(logit_points):
    for theta in logit_points:
        q, s = expit(theta)
        g_gold, g_ver = _autodiff_gradients(theta)
        dec = decompose(g_gold, g_ver, np.eye(2))
        _close(dec.A, closed_form.euclidean_signal(q, s))
        _close(dec.alpha, closed_form.alpha(q, s))
        _close(dec.C, closed_form.euclidean_pressure(q, s))


def test_fisher_diagnostics_are_parameterization_invariant(logit_points):
    """Proposition 2: logit vs mean parameterization give the same Fisher-metric (A, alpha, C)."""
    for theta in logit_points[:81]:
        phi = expit(theta)
        per_param = []
        for log_prob, params in ((bernoulli.log_prob, theta), (bernoulli.log_prob_mean, phi)):
            g_gold = autodiff.reward_gradient(log_prob, params, GOLD)
            g_ver = autodiff.reward_gradient(log_prob, params, Y)
            fisher_metric = np.linalg.inv(autodiff.fisher(log_prob, params))
            per_param.append(
                (decompose(g_gold, g_ver, fisher_metric), decompose(g_gold, g_ver, np.eye(2)))
            )
        (fisher_logit, euclid_logit), (fisher_mean, euclid_mean) = per_param
        for name in ("A", "alpha", "C"):
            np.testing.assert_allclose(
                getattr(fisher_mean, name), getattr(fisher_logit, name), rtol=1e-8, atol=1e-14
            )
        # Euclidean A is q(1-q) in logits but 1 in mean parameters: not invariant.
        assert abs(euclid_mean.A - euclid_logit.A) > 0.5
