"""General feature-triggered false positives (E002 panel family), incl. coin mixtures."""

import numpy as np
import pytest
from scipy.special import expit, logit

from vdyn.geometry import autodiff
from vdyn.geometry import triggered_fp as tfp
from vdyn.geometry.decompose import decompose
from vdyn.policies import bernoulli
from vdyn.simulation import flows
from vdyn.verifiers import boolean_fp as bf
from vdyn.verifiers.triggered import Structure

EXAMPLES = [
    bf.EventStructure("s", "SINGLE", (0.2,)),
    bf.EventStructure("a2", "AND2", (0.3, 0.6)),
    bf.EventStructure("a4", "AND4", (0.5, 0.6, 0.7, 0.8)),
    bf.EventStructure("o3", "OR3", (0.1, 0.05, 0.2)),
    bf.EventStructure("t", "THR23", (0.3, 0.4, 0.5)),
    bf.EventStructure("aor", "AOR", (0.2, 0.3, 0.6)),
    bf.EventStructure("oand", "OAND", (0.4, 0.5, 0.1)),
    bf.EventStructure("mix", "MIX", (0.3, 0.4), coin=0.05, base="AND2", rho=0.6),
    bf.EventStructure("rfp", "RFP", (), coin=0.1),
]


def _theta(st: bf.EventStructure, q: float) -> np.ndarray:
    return np.concatenate([[logit(q)], logit(np.asarray(st.s0, dtype=float))])


def test_event_truth_tables():
    z3 = bernoulli.product_outcomes(3)
    np.testing.assert_array_equal(bf.event("THR23", z3), z3.sum(axis=1) >= 2)
    np.testing.assert_array_equal(bf.event("AOR", z3), (z3[:, 0] | z3[:, 1]) & z3[:, 2])
    np.testing.assert_array_equal(bf.event("OAND", z3), (z3[:, 0] & z3[:, 1]) | z3[:, 2])
    np.testing.assert_array_equal(bf.event("AND3", z3), z3.all(axis=1))
    np.testing.assert_array_equal(bf.event("OR3", z3), z3.any(axis=1))
    np.testing.assert_array_equal(bf.event("SINGLE", z3[:, :1]), z3[:, 0])


def test_fpr_and_kappa_agree_with_triggered_closed_forms():
    for kind, typ, s in (("single", "SINGLE", (0.2,)), ("and", "AND2", (0.3, 0.6)),
                         ("and", "AND3", (0.4, 0.5, 0.7)), ("or", "OR2", (0.1, 0.25))):  # fmt: skip
        st, ref = bf.EventStructure("x", typ, s), Structure("x", kind, s)
        assert bf.fpr(st, np.array(s)) == pytest.approx(tfp.exploit_rate(ref, s), rel=1e-12)
        assert bf.kappa2(st, np.array(s)) == pytest.approx(tfp.kappa2(ref, s), rel=1e-12)


def test_fpr_derivative_matches_finite_differences(rng):
    for st in EXAMPLES[:-1]:
        for _ in range(10):
            s = rng.uniform(0.05, 0.95, size=st.n_features)
            h = 1e-6
            fd = [
                (bf.fpr(st, s + h * e) - bf.fpr(st, s - h * e)) / (2 * h)
                for e in np.eye(st.n_features)
            ]
            np.testing.assert_allclose(bf.dfpr_ds(st, s), fd, rtol=1e-7, atol=1e-10)


def test_one_minus_fpr_from_logits_is_stable_and_consistent():
    st = bf.EventStructure("x", "AND2", (0.3, 0.4), coin=0.02)
    for phi in (np.array([0.5, -1.0]), np.array([30.0, 32.0]), np.array([-5.0, 40.0])):
        exact = bf.one_minus_fpr_from_logits(st, phi)
        assert exact > 0
        if np.max(phi) < 20:
            assert exact == pytest.approx(1 - bf.fpr(st, expit(phi)), rel=1e-12)
    # deep saturation: (1 - s1 s2)(1 - coin) with 1 - s_i = expit(-phi_i)
    deep = bf.one_minus_fpr_from_logits(st, np.array([30.0, 32.0]))
    expected = (expit(-30.0) + expit(-32.0) - expit(-30.0) * expit(-32.0)) * 0.98
    assert deep == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("st", EXAMPLES, ids=lambda s: s.sid)
def test_generic_geometry_matches_prop8_generalized(st):
    """alpha = -FPR and C = (1 - q) kappa in the Fisher metric, for every type incl. coins."""
    q = 0.07
    lp = bernoulli.product_log_prob(1 + st.n_features)
    theta = _theta(st, q)
    g_gold = autodiff.reward_gradient(lp, theta, bf.gold_table(st))
    g_ver = autodiff.reward_gradient(lp, theta, bf.verifier_table(st))
    dec = decompose(g_gold, g_ver, np.linalg.inv(autodiff.fisher(lp, theta)))
    s = np.asarray(st.s0, dtype=float)
    assert dec.alpha == pytest.approx(-bf.fpr(st, s), rel=1e-10)
    assert dec.C == pytest.approx((1 - q) * np.sqrt(bf.kappa2(st, s)), rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("st", EXAMPLES, ids=lambda s: s.sid)
def test_oracle_equivalence_under_natural_gradient(st):
    """Theory note §12: C^2 = (1-q) dFPR/dt and dJ_V/dt = (1-F)^2 q(1-q) + C^2."""
    q = 0.07
    lp = bernoulli.product_log_prob(1 + st.n_features)
    theta = _theta(st, q)
    v_tab, g_tab = bf.verifier_table(st), bf.gold_table(st)
    theta_dot = flows.natural_field(lp, v_tab)(0.0, theta)
    fp_tab = (1 - g_tab) * v_tab / (1 - q)  # FPR = FP mass / (1 - q); q is unchanged along phi
    d_fpr = autodiff.reward_gradient(lp, theta, fp_tab)[1:] @ theta_dot[1:]
    d_jv = autodiff.reward_gradient(lp, theta, v_tab) @ theta_dot
    s = np.asarray(st.s0, dtype=float)
    c2 = (1 - q) ** 2 * bf.kappa2(st, s)
    f = bf.fpr(st, s)
    assert c2 == pytest.approx((1 - q) * d_fpr, rel=1e-9, abs=1e-15)
    assert d_jv == pytest.approx((1 - f) ** 2 * q * (1 - q) + c2, rel=1e-9)


def test_sampled_verifier_matches_expected_table(rng):
    st = bf.EventStructure("mix", "MIX", (0.3, 0.4), coin=0.2, base="OR2", rho=0.5)
    n = 200_000
    corr = rng.integers(0, 2, size=n)
    z = rng.integers(0, 2, size=(n, 2))
    v = bf.sample_verifier(st, corr, z, rng)
    table = bf.verifier_table(st)
    outcomes = bernoulli.product_outcomes(3)
    for k, o in enumerate(outcomes):
        mask = (corr == o[0]) & (z[:, 0] == o[1]) & (z[:, 1] == o[2])
        se = np.sqrt(table[k] * (1 - table[k]) / mask.sum()) + 1e-12
        assert abs(v[mask].mean() - table[k]) < 5 * se + 1e-12


@pytest.mark.parametrize(
    "st",
    [
        bf.EventStructure("s", "SINGLE", (0.1,)),
        bf.EventStructure("a", "AND2", (0.3, 0.4)),
        bf.EventStructure("m", "MIX", (0.2, 0.5), coin=0.03, base="OR2", rho=0.4),
    ],
)
def test_matched_null_keeps_the_dimension_and_has_zero_orthogonal_pressure(st):
    f = bf.fpr(st, np.asarray(st.s0))
    null = bf.matched_null(st, f)
    assert null.n_features == st.n_features and null.s0 == st.s0
    assert bf.fpr(null, np.asarray(null.s0)) == pytest.approx(f, rel=1e-12)
    np.testing.assert_allclose(bf.dfpr_ds(null, np.asarray(null.s0)), 0.0, atol=1e-15)
    lp = bernoulli.product_log_prob(1 + null.n_features)
    theta = _theta(null, 0.05)
    g_gold = autodiff.reward_gradient(lp, theta, bf.gold_table(null))
    g_ver = autodiff.reward_gradient(lp, theta, bf.verifier_table(null))
    d = decompose(g_gold, g_ver, np.linalg.inv(autodiff.fisher(lp, theta)))
    assert d.C == pytest.approx(0.0, abs=1e-12)
    assert d.alpha == pytest.approx(-f, rel=1e-10)
