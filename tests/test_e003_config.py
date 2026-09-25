"""E003 registered configuration: matched static metrics and consistency with the pre-registration.

These checks are static (t = 0) or closed-form only; they integrate no trajectory at the
registered settings.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, logit

from vdyn import provenance
from vdyn.geometry import autodiff
from vdyn.geometry import triggered_fp as cf
from vdyn.geometry.decompose import decompose
from vdyn.policies import bernoulli
from vdyn.verifiers.triggered import Structure, gold_table, structure_from_config, verifier_table

CONFIG = provenance.load_config(
    Path(__file__).resolve().parents[1] / "configs/toy/e003_triggered_fp.toml"
)
Q0, F = CONFIG["q0"], CONFIG["f"]
STRUCTURES = [structure_from_config(entry, F) for entry in CONFIG["structures"]]
PRED = CONFIG["predictions"]


def _theta0(structure: Structure) -> np.ndarray:
    return np.concatenate([[logit(Q0)], logit(np.asarray(structure.s0, dtype=float))])


def test_registered_parameters_follow_the_registry_definitions():
    by_name = {s.name: s for s in STRUCTURES}
    assert [s.name for s in STRUCTURES] == [
        "RFP", "AND3", "AND-SYM", "AND-MID", "AND-ASYM-A", "AND-ASYM-B", "OR", "SINGLE",
    ]  # fmt: skip
    assert Q0 == 0.001 and F == 0.01
    np.testing.assert_allclose(by_name["AND3"].s0, [F ** (1 / 3)] * 3, rtol=1e-15)
    np.testing.assert_allclose(by_name["AND-ASYM-B"].s0, [F / 0.98, 0.98], rtol=1e-15)
    np.testing.assert_allclose(by_name["OR"].s0, [1 - np.sqrt(1 - F)] * 2, rtol=1e-15)


@pytest.mark.parametrize("structure", STRUCTURES, ids=lambda s: s.name)
def test_static_metrics_are_matched_at_t0_by_enumeration(structure):
    """P1: identical gold accuracy, FPR, FNR, verifier accuracy and FP mass (enumerated)."""
    p = autodiff.outcome_probs(
        bernoulli.product_log_prob(1 + structure.n_features), _theta0(structure)
    )
    g, v = gold_table(structure), verifier_table(structure)
    fp_mass = p @ ((1 - g) * v)
    fn_mass = p @ (g * (1 - v))
    assert p @ g == pytest.approx(Q0, abs=1e-15)
    assert fp_mass / (p @ (1 - g)) == pytest.approx(F, abs=1e-12)
    assert fn_mass == 0.0
    assert 1 - fp_mass - fn_mass == pytest.approx(0.990010, abs=1e-12)
    assert fp_mass == pytest.approx(0.009990, abs=1e-12)


@pytest.mark.parametrize("structure", STRUCTURES, ids=lambda s: s.name)
def test_signal_and_parallel_distortion_are_matched_at_t0(structure):
    """P1: Fisher-metric A and alpha identical across structures (generic autodiff)."""
    log_prob = bernoulli.product_log_prob(1 + structure.n_features)
    theta = _theta0(structure)
    g_gold = autodiff.reward_gradient(log_prob, theta, gold_table(structure))
    g_ver = autodiff.reward_gradient(log_prob, theta, verifier_table(structure))
    dec = decompose(g_gold, g_ver, np.linalg.inv(autodiff.fisher(log_prob, theta)))
    assert dec.A == pytest.approx(0.0316070, abs=1e-7)
    assert dec.A == pytest.approx(np.sqrt(Q0 * (1 - Q0)), abs=1e-12)
    assert dec.alpha == pytest.approx(-F, abs=1e-12)


def test_c0_generic_values_and_ordering_match_registered_values():
    """P2 at t = 0 only: generic decomposition vs the registered 6-decimal C(0)."""
    values = []
    for structure in STRUCTURES:
        log_prob = bernoulli.product_log_prob(1 + structure.n_features)
        theta = _theta0(structure)
        g_gold = autodiff.reward_gradient(log_prob, theta, gold_table(structure))
        g_ver = autodiff.reward_gradient(log_prob, theta, verifier_table(structure))
        c0 = decompose(g_gold, g_ver, np.linalg.inv(autodiff.fisher(log_prob, theta))).C
        assert c0 == pytest.approx(PRED[structure.name]["C0"], abs=1e-6)
        values.append(c0)
    assert np.all(np.diff(values) > 0), "registered order is strictly increasing in C(0)"


@pytest.mark.parametrize("structure", STRUCTURES, ids=lambda s: s.name)
def test_repository_closed_forms_reproduce_registered_predictions(structure):
    """The registry numbers were computed by a scratch script; the repo code must agree."""
    pred = PRED[structure.name]
    q, s = Q0, np.asarray(structure.s0, dtype=float)
    _, _, c0 = cf.fisher_diagnostics(structure, q, s)
    assert c0 == pytest.approx(pred["C0"], abs=1e-6)
    assert cf.eta(structure, s) == pytest.approx(pred["eta0"], abs=1e-4)
    outcome = cf.predicted_outcome(structure, Q0)
    assert outcome["stall"] is pred["stall"]
    assert outcome["q_inf"] == pytest.approx(pred["q_inf"], abs=1e-6)
    assert outcome["S_inf"] == pytest.approx(pred["S_inf"], abs=1e-6)


def test_full_feature_bank_reduces_exactly_to_each_structure():
    """Identical initial policy: one product policy over all features; unread features are inert."""
    feature_structs = [s for s in STRUCTURES if s.kind != "random"]
    bank = np.concatenate([np.asarray(s.s0) for s in feature_structs])
    theta_full = np.concatenate([[logit(Q0)], logit(bank)])
    n = 1 + len(bank)
    log_prob_full = bernoulli.product_log_prob(n)
    outcomes = bernoulli.product_outcomes(n)
    fisher_full = autodiff.fisher(log_prob_full, theta_full)
    offset = 1
    for structure in feature_structs:
        own = slice(offset, offset + structure.n_features)
        offset += structure.n_features
        corr, z = outcomes[:, 0], outcomes[:, own]
        if structure.kind == "single":
            event = z[:, 0]
        elif structure.kind == "and":
            event = z.all(axis=1)
        else:
            event = z.any(axis=1)
        v_full = np.maximum(corr, event).astype(float)
        g_full = autodiff.reward_gradient(log_prob_full, theta_full, v_full)
        g_red = autodiff.reward_gradient(
            bernoulli.product_log_prob(1 + structure.n_features),
            _theta0(structure),
            verifier_table(structure),
        )
        idx = np.r_[0, np.arange(own.start, own.stop)]
        np.testing.assert_allclose(g_full[idx], g_red, rtol=1e-10, atol=1e-16)
        others = np.setdiff1d(np.arange(1, n), idx)
        np.testing.assert_allclose(g_full[others], 0.0, atol=1e-16)
        fisher_red = autodiff.fisher(
            bernoulli.product_log_prob(1 + structure.n_features), _theta0(structure)
        )
        np.testing.assert_allclose(
            fisher_full[np.ix_(idx, idx)], fisher_red, rtol=1e-10, atol=1e-16
        )
    assert np.allclose(fisher_full, np.diag(np.diag(fisher_full)), atol=1e-16)
    assert expit(theta_full[0]) == pytest.approx(Q0, rel=1e-12)
