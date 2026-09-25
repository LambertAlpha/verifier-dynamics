"""Candidate 3 dynamics under natural gradient: generic flows vs Props. 8-9.

All trajectory tests use parameters that are NOT the E003 registered settings, so testing does not
preview any registered outcome.
"""

import numpy as np
import pytest
from scipy.special import expit, logit

from vdyn.geometry import triggered_fp as cf
from vdyn.policies import bernoulli
from vdyn.simulation import flows
from vdyn.verifiers.triggered import Structure, verifier_table

Q0 = 0.02
TEST_STRUCTURES = [
    Structure("single", "single", (0.05,)),  # stall, q_inf = 0.4
    Structure("or-sym", "or", (0.03, 0.03)),  # stall, q_inf = 2/3 (algebraic convergence)
    Structure("and-sym", "and", (0.2, 0.2)),  # success
    Structure("and-asym", "and", (0.05, 0.8)),  # stall
    Structure("and3-sym", "and", (0.3, 0.3, 0.3)),  # success
    Structure("random", "random", (), f=0.05),  # success
]


def _natural(structure: Structure) -> flows.Field:
    return flows.natural_field(
        bernoulli.product_log_prob(1 + structure.n_features), verifier_table(structure)
    )


def _theta0(structure: Structure, q0: float = Q0) -> np.ndarray:
    return np.concatenate([[logit(q0)], logit(np.asarray(structure.s0, dtype=float))])


@pytest.mark.parametrize("structure", TEST_STRUCTURES, ids=lambda s: s.name)
def test_natural_field_matches_closed_form_velocity(structure, rng):
    field = _natural(structure)
    for _ in range(20):
        theta = rng.uniform(-4, 4, size=1 + structure.n_features)
        q, s = float(expit(theta[0])), expit(theta[1:])
        np.testing.assert_allclose(
            field(0.0, theta), cf.natural_velocity(structure, q, s), rtol=1e-9, atol=1e-13
        )


@pytest.mark.parametrize("structure", TEST_STRUCTURES, ids=lambda s: s.name)
def test_vanilla_field_matches_closed_form_velocity(structure, rng):
    log_prob = bernoulli.product_log_prob(1 + structure.n_features)
    field = flows.vanilla_field(log_prob, verifier_table(structure))
    for _ in range(20):
        theta = rng.uniform(-4, 4, size=1 + structure.n_features)
        q, s = float(expit(theta[0])), expit(theta[1:])
        np.testing.assert_allclose(
            field(0.0, theta), cf.vanilla_velocity(structure, q, s), rtol=1e-9, atol=1e-15
        )


@pytest.mark.parametrize("structure", TEST_STRUCTURES[:-1], ids=lambda s: s.name)
def test_gold_race_relation_holds_along_the_trajectory(structure):
    """Proposition 9(b): log q(t) - log q0 = Lambda(state(t)) at all times."""
    sol = flows.integrate(_natural(structure), _theta0(structure), 60.0, np.linspace(0, 60, 121))
    q, s = expit(sol.y[0]), expit(sol.y[1:])
    for k in range(sol.y.shape[1]):
        predicted = cf.gold_race(structure, structure.s0, s[:, k])
        assert np.log(q[k]) - np.log(Q0) == pytest.approx(predicted, abs=1e-7)


def test_random_fp_gold_is_exact_logistic_in_time():
    structure = TEST_STRUCTURES[-1]
    t = np.linspace(0, 40, 81)
    sol = flows.integrate(_natural(structure), _theta0(structure), 40.0, t)
    np.testing.assert_allclose(expit(sol.y[0]), cf.random_fp_gold(logit(Q0), 0.05, t), rtol=1e-9)


@pytest.mark.parametrize(
    "structure", [s for s in TEST_STRUCTURES if s.kind == "and"], ids=lambda s: s.name
)
def test_and_structures_conserve_feature_ratio(structure):
    sol = flows.integrate(_natural(structure), _theta0(structure), 60.0, np.linspace(0, 60, 121))
    s = expit(sol.y[1:])
    keep = np.all(1 - s > 1e-9, axis=0)
    ratio = (1 - s[0, keep]) / (1 - s[1, keep])
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-7)


@pytest.mark.parametrize("structure", TEST_STRUCTURES, ids=lambda s: s.name)
def test_predicted_outcome_matches_long_trajectory(structure):
    t_end = 1e4 if structure.kind == "or" else 300.0  # OR converges algebraically
    sol = flows.integrate(_natural(structure), _theta0(structure), t_end)
    q_end = float(expit(sol.y[0, -1]))
    s_end = expit(sol.y[1:, -1])
    outcome = cf.predicted_outcome(structure, Q0)
    if outcome["stall"]:
        assert q_end == pytest.approx(outcome["q_inf"], abs=1e-3)
        assert cf.exploit_rate(structure, s_end) > 1 - 1e-2
    else:
        assert q_end > 1 - 1e-6
        assert cf.exploit_rate(structure, s_end) == pytest.approx(outcome["S_inf"], abs=1e-5)


def test_test_structures_cover_both_outcomes():
    stalls = {s.name: cf.predicted_outcome(s, Q0)["stall"] for s in TEST_STRUCTURES}
    assert stalls == {
        "single": True,
        "or-sym": True,
        "and-sym": False,
        "and-asym": True,
        "and3-sym": False,
        "random": False,
    }


@pytest.mark.parametrize("structure", TEST_STRUCTURES, ids=lambda s: s.name)
def test_finite_difference_rates_and_clean_bound(structure):
    theta0, h = _theta0(structure), 1e-4
    sol = flows.integrate(_natural(structure), theta0, 30.0)
    for t in np.linspace(0.5, 29.5, 15):
        fd = (expit(sol.sol(t + h)) - expit(sol.sol(t - h))) / (2 * h)
        state = expit(sol.sol(t))
        np.testing.assert_allclose(
            fd, cf.natural_rates(structure, float(state[0]), state[1:]), rtol=1e-6, atol=1e-10
        )
        assert state[0] <= cf.clean_gold(theta0[0], t) + 1e-12
