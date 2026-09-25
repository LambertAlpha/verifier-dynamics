"""Candidate 3 static quantities: product policy, triggered verifiers, closed forms vs autodiff."""

import numpy as np
import pytest
import torch
from scipy.special import expit

from vdyn.geometry import autodiff
from vdyn.geometry import triggered_fp as cf
from vdyn.geometry.decompose import decompose
from vdyn.policies import bernoulli
from vdyn.verifiers.triggered import Structure, gold_table, verifier_table

STRUCTURES = [
    Structure("single", "single", (0.2,)),
    Structure("and2", "and", (0.3, 0.6)),
    Structure("and3", "and", (0.4, 0.5, 0.7)),
    Structure("or2", "or", (0.1, 0.25)),
    Structure("random", "random", (), f=0.15),
]


def _random_states(structure: Structure, rng: np.random.Generator, n: int = 25):
    for _ in range(n):
        theta = rng.uniform(-4.0, 4.0, size=1 + structure.n_features)
        yield theta, float(expit(theta[0])), expit(theta[1:])


def _log_prob(structure: Structure):
    return bernoulli.product_log_prob(1 + structure.n_features)


# --- policy and verifier tables -------------------------------------------------------------------


def test_product_policy_with_two_coordinates_equals_phase1a_policy(logit_points):
    two = bernoulli.product_log_prob(2)
    np.testing.assert_array_equal(bernoulli.product_outcomes(2), bernoulli.OUTCOMES)
    for theta in logit_points[:81]:
        t = torch.as_tensor(theta, dtype=torch.float64)
        np.testing.assert_allclose(two(t).numpy(), bernoulli.log_prob(t).numpy(), rtol=1e-13)


def test_product_policy_is_a_product_of_bernoullis(rng):
    for n in (1, 3, 4):
        theta = rng.uniform(-3, 3, size=n)
        p = autodiff.outcome_probs(bernoulli.product_log_prob(n), theta)
        probs = expit(theta)
        expected = np.prod(np.where(bernoulli.product_outcomes(n) == 1, probs, 1 - probs), axis=1)
        np.testing.assert_allclose(p, expected, rtol=1e-12)
        assert p.sum() == pytest.approx(1.0, abs=1e-14)


def test_triggered_verifiers_accept_correct_answers_and_have_no_false_negatives():
    for structure in STRUCTURES:
        outcomes = bernoulli.product_outcomes(1 + structure.n_features)
        v, g = verifier_table(structure), gold_table(structure)
        np.testing.assert_array_equal(v[outcomes[:, 0] == 1], 1.0)
        assert np.all(v >= g)


def test_triggered_events_on_wrong_answers():
    wrong: dict[str, dict[tuple[int, ...], int]] = {
        "single": {(0, 0): 0, (0, 1): 1},
        "and2": {(0, 0, 0): 0, (0, 1, 0): 0, (0, 0, 1): 0, (0, 1, 1): 1},
        "or2": {(0, 0, 0): 0, (0, 1, 0): 1, (0, 0, 1): 1, (0, 1, 1): 1},
    }
    by_name = {s.name: s for s in STRUCTURES}
    for name, cases in wrong.items():
        structure = by_name[name]
        outcomes = [tuple(o) for o in bernoulli.product_outcomes(1 + structure.n_features)]
        table = verifier_table(structure)
        for outcome, expected in cases.items():
            assert table[outcomes.index(outcome)] == expected, (name, outcome)
    random_table = verifier_table(by_name["random"])
    np.testing.assert_allclose(random_table, [0.15, 1.0])  # E[corr OR xi | corr]


# --- closed forms vs enumeration + autodiff ------------------------------------------------------


def test_static_metrics_match_enumeration(rng):
    for structure in STRUCTURES:
        g, v = gold_table(structure), verifier_table(structure)
        for theta, q, s in _random_states(structure, rng):
            p = autodiff.outcome_probs(_log_prob(structure), theta)
            metrics = cf.static_metrics(structure, q, s)
            fp_mass = p @ ((1 - g) * v)
            assert metrics["gold_accuracy"] == pytest.approx(p @ g, rel=1e-12)
            assert metrics["fp_mass"] == pytest.approx(fp_mass, rel=1e-10, abs=1e-15)
            assert metrics["fpr"] == pytest.approx(fp_mass / (p @ (1 - g)), rel=1e-10)
            assert metrics["fnr"] == 0.0
            assert metrics["verifier_accuracy"] == pytest.approx(1 - fp_mass, rel=1e-12)


def test_objectives_gradients_and_fisher_match_autodiff(rng):
    for structure in STRUCTURES:
        log_prob, g_tab, v_tab = (
            _log_prob(structure),
            gold_table(structure),
            verifier_table(structure),
        )
        for theta, q, s in _random_states(structure, rng):
            assert cf.j_verifier(structure, q, s) == pytest.approx(
                autodiff.expected_reward(log_prob, theta, v_tab), rel=1e-12
            )
            np.testing.assert_allclose(
                cf.grad_gold(structure, q, s),
                autodiff.reward_gradient(log_prob, theta, g_tab),
                rtol=1e-10,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                cf.grad_verifier(structure, q, s),
                autodiff.reward_gradient(log_prob, theta, v_tab),
                rtol=1e-10,
                atol=1e-15,
            )
            fisher = cf.fisher(structure, q, s)
            np.testing.assert_allclose(
                fisher, autodiff.fisher(log_prob, theta), rtol=1e-10, atol=1e-15
            )
            np.testing.assert_allclose(
                fisher, autodiff.fisher_from_hessian(log_prob, theta), rtol=1e-10, atol=1e-15
            )


def test_fisher_diagnostics_match_generic_decomposition(rng):
    for structure in STRUCTURES:
        log_prob, g_tab, v_tab = (
            _log_prob(structure),
            gold_table(structure),
            verifier_table(structure),
        )
        for theta, q, s in _random_states(structure, rng):
            g_gold = autodiff.reward_gradient(log_prob, theta, g_tab)
            g_ver = autodiff.reward_gradient(log_prob, theta, v_tab)
            dec = decompose(g_gold, g_ver, np.linalg.inv(autodiff.fisher(log_prob, theta)))
            A, alpha, C = cf.fisher_diagnostics(structure, q, s)
            assert dec.A == pytest.approx(A, rel=1e-10)
            assert dec.alpha == pytest.approx(alpha, rel=1e-10)
            assert dec.C == pytest.approx(C, rel=1e-9, abs=1e-14)


def test_alpha_is_minus_fpr_and_c_is_capped_by_fpr(rng):
    """Proposition 8: alpha = -FPR exactly; 0 <= C <= C_max = (1-q) sqrt(FPR(1-FPR))."""
    for structure in STRUCTURES:
        for _, q, s in _random_states(structure, rng):
            fpr = cf.static_metrics(structure, q, s)["fpr"]
            _, alpha, C = cf.fisher_diagnostics(structure, q, s)
            assert alpha == pytest.approx(-fpr, rel=1e-12)
            assert 0.0 <= C <= cf.c_max(q, fpr) * (1 + 1e-12)


def test_cramer_rao_bound_is_attained_only_by_the_single_feature(rng):
    for _ in range(200):
        k = int(rng.integers(2, 4))
        s = tuple(rng.uniform(0.01, 0.99, size=k))
        for kind in ("and", "or"):
            eta = cf.eta(Structure("x", kind, s), s)
            assert 0.0 < eta < 1.0
    for p in rng.uniform(0.01, 0.99, size=20):
        assert cf.eta(Structure("x", "single", (p,)), (p,)) == pytest.approx(1.0, rel=1e-12)
    assert cf.eta(Structure("x", "random", (), f=0.3), ()) == 0.0
