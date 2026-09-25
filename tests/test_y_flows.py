"""Y toy dynamics: generic autodiff-driven flows vs closed-form rates, invariants and endpoints."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import expit, logit

from vdyn.geometry import autodiff, closed_form
from vdyn.geometry.decompose import decompose
from vdyn.policies import bernoulli
from vdyn.simulation import flows
from vdyn.verifiers import toy

GOLD = toy.reward_table(toy.gold)
Y = toy.reward_table(toy.y_false_positive)
NATURAL = flows.natural_field(bernoulli.log_prob, Y)
VANILLA = flows.vanilla_field(bernoulli.log_prob, Y)


def _theta0(q0: float, s0: float) -> np.ndarray:
    return np.array([logit(q0), logit(s0)])


def _chain_rule(theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
    q, s = expit(theta)
    return np.array([q * (1 - q) * theta_dot[0], s * (1 - s) * theta_dot[1]])


def _gap_gradient(theta: np.ndarray) -> np.ndarray:
    return autodiff.reward_gradient(bernoulli.log_prob, theta, Y) - autodiff.reward_gradient(
        bernoulli.log_prob, theta, GOLD
    )


# --- pointwise: fields, rates, Proposition 1 ------------------------------------------------------


def test_natural_field_matches_closed_form_velocity_and_rates(logit_points):
    for theta in logit_points:
        q, s = expit(theta)
        theta_dot = NATURAL(0.0, theta)
        np.testing.assert_allclose(
            theta_dot, closed_form.natural_logit_velocity(q, s), rtol=1e-9, atol=1e-13
        )
        np.testing.assert_allclose(
            _chain_rule(theta, theta_dot), closed_form.natural_rates(q, s), rtol=1e-9, atol=1e-14
        )


def test_vanilla_field_matches_closed_form_velocity_and_rates(logit_points):
    for theta in logit_points:
        q, s = expit(theta)
        theta_dot = VANILLA(0.0, theta)
        np.testing.assert_allclose(
            theta_dot, closed_form.vanilla_logit_velocity(q, s), rtol=1e-9, atol=1e-14
        )
        np.testing.assert_allclose(
            _chain_rule(theta, theta_dot), closed_form.vanilla_rates(q, s), rtol=1e-9, atol=1e-15
        )


def test_gap_rates_match_chain_rule_along_autodiff_fields(logit_points):
    for theta in logit_points:
        q, s = expit(theta)
        grad_gap = _gap_gradient(theta)
        np.testing.assert_allclose(
            grad_gap @ NATURAL(0.0, theta),
            closed_form.natural_gap_rate(q, s),
            rtol=1e-8,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            grad_gap @ VANILLA(0.0, theta),
            closed_form.vanilla_gap_rate(q, s),
            rtol=1e-8,
            atol=1e-15,
        )


def test_metric_matched_rate_identities(logit_points):
    """Proposition 1 with M = F^{-1} for natural flow and M = I for vanilla flow."""
    for theta in logit_points:
        q, s = expit(theta)
        g_gold = autodiff.reward_gradient(bernoulli.log_prob, theta, GOLD)
        g_ver = autodiff.reward_gradient(bernoulli.log_prob, theta, Y)
        fisher_dec = decompose(
            g_gold, g_ver, np.linalg.inv(autodiff.fisher(bernoulli.log_prob, theta))
        )
        euclid_dec = decompose(g_gold, g_ver, np.eye(2))
        np.testing.assert_allclose(
            fisher_dec.gap_rate, closed_form.natural_gap_rate(q, s), rtol=1e-7, atol=1e-14
        )
        np.testing.assert_allclose(
            euclid_dec.gap_rate, closed_form.vanilla_gap_rate(q, s), rtol=1e-7, atol=1e-15
        )
        np.testing.assert_allclose(
            fisher_dec.gold_rate, closed_form.natural_rates(q, s)[0], rtol=1e-9
        )
        np.testing.assert_allclose(
            euclid_dec.gold_rate, closed_form.vanilla_rates(q, s)[0], rtol=1e-9
        )


def test_fisher_metric_mispredicts_vanilla_gap_sign():
    """Corollary of Proposition 1: a mismatched metric can flip the predicted sign of dDelta/dt."""
    theta = _theta0(0.3, 0.05)
    g_gold = autodiff.reward_gradient(bernoulli.log_prob, theta, GOLD)
    g_ver = autodiff.reward_gradient(bernoulli.log_prob, theta, Y)
    fisher_dec = decompose(g_gold, g_ver, np.linalg.inv(autodiff.fisher(bernoulli.log_prob, theta)))
    actual_vanilla = _gap_gradient(theta) @ VANILLA(0.0, theta)
    assert fisher_dec.gap_rate == pytest.approx(1.330e-2, rel=1e-3)
    assert actual_vanilla == pytest.approx(-9.892e-4, rel=1e-3)


# --- trajectories ---------------------------------------------------------------------------------


def _closed_form_trajectory(rates, q0, s0, t_eval):
    sol = solve_ivp(
        lambda _t, y: rates(y[0], y[1]),
        (0.0, t_eval[-1]),
        [q0, s0],
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    assert sol.success
    return sol.y


@pytest.mark.parametrize(
    ("field", "rates", "t_end"),
    [(NATURAL, closed_form.natural_rates, 10.0), (VANILLA, closed_form.vanilla_rates, 200.0)],
    ids=["natural", "vanilla"],
)
@pytest.mark.parametrize(("q0", "s0"), [(0.3, 0.01), (0.1, 0.1), (0.01, 0.3)])
def test_autodiff_trajectory_matches_closed_form_ode(field, rates, t_end, q0, s0):
    t_eval = np.linspace(0.0, t_end, 41)
    sol = flows.integrate(field, _theta0(q0, s0), t_end, t_eval)
    np.testing.assert_allclose(
        expit(sol.y), _closed_form_trajectory(rates, q0, s0, t_eval), atol=1e-8
    )


def test_natural_flow_conserves_s_over_q():
    q0, s0 = 0.2, 0.05
    sol = flows.integrate(NATURAL, _theta0(q0, s0), 20.0, np.linspace(0, 20, 201))
    q, s = expit(sol.y)
    np.testing.assert_allclose(closed_form.natural_invariant(q, s), s0 / q0, atol=1e-9)


def test_vanilla_flow_conserves_invariant():
    theta0 = _theta0(0.05, 0.2)
    sol = flows.integrate(VANILLA, theta0, 1e4, np.geomspace(1e-3, 1e4, 200))
    drift = closed_form.vanilla_invariant(sol.y[0], sol.y[1]) - closed_form.vanilla_invariant(
        *theta0
    )
    assert np.max(np.abs(drift)) < 1e-7


@pytest.mark.parametrize(("q0", "s0"), [(0.3, 0.01), (0.01, 0.3), (0.2, 0.5)])
def test_natural_flow_reaches_predicted_endpoint(q0, s0):
    sol = flows.integrate(NATURAL, _theta0(q0, s0), 40.0)
    q_end, s_end = expit(sol.y[:, -1])
    np.testing.assert_allclose([q_end, s_end], closed_form.natural_endpoint(q0, s0), atol=1e-8)


def test_vanilla_path_matches_invariant_prediction_clock_free():
    q0, s0 = 0.01, 0.3
    sol = flows.integrate(VANILLA, _theta0(q0, s0), 1e4, np.geomspace(1.0, 1e4, 20))
    for q, s in expit(sol.y).T:
        assert q == pytest.approx(closed_form.vanilla_q_given_s(q0, s0, s), rel=1e-7)


def test_finite_difference_of_trajectory_matches_closed_form_rates():
    q0, s0, h = 0.2, 0.05, 1e-4
    sol = flows.integrate(NATURAL, _theta0(q0, s0), 12.0)
    for t in np.linspace(0.5, 11.5, 12):
        fd = (expit(sol.sol(t + h)) - expit(sol.sol(t - h))) / (2 * h)
        q, s = expit(sol.sol(t))
        np.testing.assert_allclose(fd, closed_form.natural_rates(q, s), rtol=1e-6, atol=1e-10)
        fd_gap = closed_form.gap(*expit(sol.sol(t + h))) - closed_form.gap(*expit(sol.sol(t - h)))
        fd_gap = fd_gap / (2 * h)
        assert fd_gap == pytest.approx(closed_form.natural_gap_rate(q, s), rel=1e-5, abs=1e-10)


def test_euler_converges_to_flow_at_first_order():
    theta0, t_end = _theta0(0.2, 0.05), 2.0
    exact = flows.integrate(NATURAL, theta0, t_end).y[:, -1]
    errors = []
    for eta in (0.02, 0.01, 0.005):
        path = flows.euler(NATURAL, theta0, eta, round(t_end / eta))
        errors.append(np.linalg.norm(path[-1] - exact))
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all((ratios > 1.8) & (ratios < 2.2)), ratios


# --- qualitative claims ---------------------------------------------------------------------------


def test_gap_sign_never_switches_from_shrinking_to_growing(rng):
    """Numerical check of Proposition 5 on 60 random initial conditions, both flows.

    Integrated in logit coordinates: integrating in (q, s) lets s overshoot 1 by ~1e-10 near the
    endpoint, which flips the sign of s(1-s) and produces spurious sign changes.
    """
    cases = [
        (closed_form.natural_logit_velocity, closed_form.natural_gap_rate, 60.0),
        (closed_form.vanilla_logit_velocity, closed_form.vanilla_gap_rate, 1e5),
    ]
    for q0, s0 in rng.uniform(1e-3, 1 - 1e-3, size=(60, 2)):
        for velocity, gap_rate, t_end in cases:
            sol = solve_ivp(
                lambda _t, th, vel=velocity: vel(*expit(th)),
                (0.0, t_end),
                _theta0(q0, s0),
                t_eval=np.geomspace(1e-4, t_end, 3000),
                method="DOP853",
                rtol=1e-11,
                atol=1e-13,
            )
            assert sol.success
            q, s = expit(sol.y)
            rate = gap_rate(q, s)
            signs = np.sign(rate[np.abs(rate) > 1e-13])
            assert not np.any((signs[:-1] < 0) & (signs[1:] > 0)), (q0, s0, velocity.__name__)


def test_y_turns_into_deletion_as_exploit_saturates():
    """Theory note §2.2 corollary: q0 < s0 under natural flow drives alpha -> -1 and C -> 0."""
    sol = flows.integrate(NATURAL, _theta0(0.01, 0.3), 30.0)
    theta = sol.y[:, -1]
    g_gold = autodiff.reward_gradient(bernoulli.log_prob, theta, GOLD)
    g_ver = autodiff.reward_gradient(bernoulli.log_prob, theta, Y)
    dec = decompose(g_gold, g_ver, np.linalg.inv(autodiff.fisher(bernoulli.log_prob, theta)))
    assert dec.alpha == pytest.approx(-1.0, abs=1e-9)
    assert dec.C < 1e-4
    assert expit(theta[0]) == pytest.approx(0.01 / 0.3, abs=1e-9)
