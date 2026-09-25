"""General long-run targets from the q-free feature ODE (registry E002 §4).

All parameters here are unregistered (not panel structures).
"""

import numpy as np
import pytest
from scipy.special import expit, logit

from vdyn.geometry import triggered_fp as tfp
from vdyn.geometry.gold_race import gold_race
from vdyn.policies import bernoulli
from vdyn.simulation import flows
from vdyn.verifiers import boolean_fp as bf
from vdyn.verifiers.triggered import Structure

Q0 = 0.02
CLOSED_FORM_CASES = [
    (bf.EventStructure("s", "SINGLE", (0.05,)), Structure("s", "single", (0.05,))),
    (bf.EventStructure("s2", "SINGLE", (0.01,)), Structure("s2", "single", (0.01,))),
    (bf.EventStructure("a", "AND2", (0.2, 0.2)), Structure("a", "and", (0.2, 0.2))),
    (bf.EventStructure("aa", "AND2", (0.05, 0.8)), Structure("aa", "and", (0.05, 0.8))),
    (bf.EventStructure("a3", "AND3", (0.3, 0.3, 0.3)), Structure("a3", "and", (0.3, 0.3, 0.3))),
    (bf.EventStructure("o", "OR2", (0.03, 0.03)), Structure("o", "or", (0.03, 0.03))),
]


@pytest.mark.parametrize(("st", "ref"), CLOSED_FORM_CASES, ids=lambda x: getattr(x, "sid", ""))
def test_gold_race_agrees_with_prop9_closed_forms(st, ref):
    result = gold_race(st, Q0)
    expected = tfp.predicted_outcome(ref, Q0)
    assert result.stall is expected["stall"]
    assert result.q_inf == pytest.approx(expected["q_inf"], rel=1e-8)
    assert result.fpr_inf == pytest.approx(expected["S_inf"], rel=1e-7, abs=1e-8)


def test_random_fp_control_succeeds_with_exact_logistic_gold():
    st = bf.EventStructure("r", "RFP", (), coin=0.1)
    result = gold_race(st, Q0, report_times=(5.0, 25.0))
    assert result.stall is False and result.fpr_inf == pytest.approx(0.1, rel=1e-12)
    for t in (5.0, 25.0):
        assert result.jg_at[t] == pytest.approx(expit(logit(Q0) + 0.9 * t), rel=1e-8)


@pytest.mark.parametrize(
    "st",
    [
        bf.EventStructure("t", "THR23", (0.1, 0.15, 0.2)),
        bf.EventStructure("aor", "AOR", (0.2, 0.1, 0.3)),
        bf.EventStructure("oand", "OAND", (0.2, 0.3, 0.02)),
        bf.EventStructure("mix", "MIX", (0.2, 0.25), coin=0.03, base="AND2", rho=0.6),
        bf.EventStructure("a4", "AND4", (0.6, 0.5, 0.7, 0.55)),
    ],
    ids=lambda s: s.sid,
)
def test_gold_race_matches_full_natural_gradient_ode(st):
    report = (3.0, 10.0, 400.0)
    result = gold_race(st, Q0, report_times=report)
    lp = bernoulli.product_log_prob(1 + st.n_features)
    field = flows.natural_field(lp, bf.verifier_table(st))
    theta0 = np.concatenate([[logit(Q0)], logit(np.asarray(st.s0))])
    sol = flows.integrate(field, theta0, 400.0, np.array([0.0, *report]))
    for k, t in enumerate(report, start=1):
        assert expit(sol.y[0, k]) == pytest.approx(result.jg_at[t], rel=1e-6)
        s_t = expit(sol.y[1:, k])
        assert bf.fpr(st, s_t) == pytest.approx(result.fpr_at[t], rel=1e-6)


def test_near_threshold_structure_is_not_called_a_stall():
    """OAND (0.2, 0.3, 0.02) at q0 = 0.02 sits within ~1e-11 of the stall threshold."""
    result = gold_race(bf.EventStructure("oand", "OAND", (0.2, 0.3, 0.02)), Q0)
    assert result.q_inf > 1 - 1e-6
    assert result.stall is False


def test_t95_marks_95_percent_of_the_stall_gain():
    st = bf.EventStructure("s", "SINGLE", (0.05,))
    result = gold_race(st, Q0)
    assert result.stall and result.t95 is not None
    lp = bernoulli.product_log_prob(2)
    sol = flows.integrate(
        flows.natural_field(lp, bf.verifier_table(st)),
        np.array([logit(Q0), logit(0.05)]),
        result.t95,
        np.array([0.0, result.t95]),
    )
    target = Q0 + 0.95 * (result.q_inf - Q0)
    assert expit(sol.y[0, -1]) == pytest.approx(target, rel=1e-6)


def test_success_has_no_t95_and_unit_asymptotic_gold():
    result = gold_race(bf.EventStructure("a", "AND2", (0.2, 0.2)), Q0, report_times=(25.0,))
    assert result.stall is False and result.q_inf == 1.0 and result.t95 is None
    assert result.jg_at[25.0] > 0.99


def test_stall_just_below_one_is_still_a_stall():
    """q_inf = q0/s0 = 0.995 lies between 1 - 1e-2 and 1 - 1e-6 (amendment 1 threshold)."""
    result = gold_race(bf.EventStructure("s", "SINGLE", (Q0 / 0.995,)), Q0)
    assert result.q_inf == pytest.approx(0.995, rel=1e-8)
    assert result.stall is True
