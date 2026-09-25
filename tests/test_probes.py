"""E002 probe arms and budget accounting (registry E002 §3, §5)."""

import itertools

import numpy as np
import pytest
from scipy.special import expit, logit

from vdyn.e002 import probes as pr
from vdyn.geometry import autodiff
from vdyn.policies import bernoulli
from vdyn.verifiers import boolean_fp as bf

ST = bf.EventStructure("a", "AND2", (0.3, 0.5), coin=0.0)
MIX = bf.EventStructure("m", "MIX", (0.3, 0.4), coin=0.1, base="OR2", rho=0.5)
Q0 = 0.1
GRID = [(g, g * r) for g in (16, 32, 64, 128, 256, 512, 1024) for r in (1, 4, 16)]


def _theta0(st):
    return np.concatenate([[logit(Q0)], logit(np.asarray(st.s0))])


def _exact_fpr(st, theta):
    s = expit(theta[1:])
    return bf.fpr(st, s)


@pytest.mark.parametrize(("b_gold", "b_roll"), GRID)
def test_every_plan_respects_the_budget(b_gold, b_roll):
    plans = [pr.plan_g0(b_gold, b_roll), pr.plan_g1(b_gold, b_roll)]
    plans += [pr.plan_p1(b_gold, b_roll, k) for k in (1, 2, 5, 10)]
    plans += [pr.plan_p2(b_gold, b_roll, k) for k in (1, 2)]
    plans += [pr.plan_p3(b_gold, b_roll, k) for k in (1, 2, 5, 10)]
    plans += [pr.plan_p4(b_gold, b_roll)]
    for plan in plans:
        if plan.feasible:
            assert plan.b_roll <= b_roll and plan.b_gold <= b_gold, plan


def test_infeasibility_rules():
    assert not pr.plan_p3(64, 64, 1).feasible  # no rollouts left for training
    assert pr.plan_p2(64, 64, 1).feasible and pr.plan_p2(64, 64, 1).reuse_audit
    assert not pr.plan_p2(64, 64, 2).feasible
    assert not pr.plan_p4(64, 64).feasible  # no rollouts left for variants
    assert pr.plan_p1(64, 64, 10).feasible
    p1 = pr.plan_p1(64, 256, 5)
    assert (p1.b, p1.b_roll, p1.b_gold, p1.b_bwd, p1.k) == (42, 252, 0, 210, 5)


def test_natural_gradient_probe_step_matches_exact_step(rng):
    st, eta = MIX, 0.3
    lp = bernoulli.product_log_prob(1 + st.n_features)
    theta0 = _theta0(st)
    g = autodiff.reward_gradient(lp, theta0, bf.verifier_table(st))
    exact = theta0 + eta * np.linalg.solve(autodiff.fisher(lp, theta0), g)
    thetas, _ = pr.ng_probe(st, theta0, k=1, b=100_000, eta=eta, lam=1e-6, reps=4, rng=rng)
    np.testing.assert_allclose(thetas, np.broadcast_to(exact, thetas.shape), atol=5e-3)


def test_p1_scores_verifier_growth(rng):
    theta0 = _theta0(MIX)
    out = pr.run_p1(MIX, theta0, b=200_000, k=1, eta=0.5, reps=3, rng=rng)
    lp = bernoulli.product_log_prob(1 + MIX.n_features)
    table = bf.verifier_table(MIX)
    for r in range(3):
        exact = autodiff.expected_reward(lp, out["theta_k"][r], table) - autodiff.expected_reward(
            lp, theta0, table
        )
        assert out["score"][r] == pytest.approx(exact, abs=4e-3)


@pytest.mark.parametrize("st", [ST, MIX], ids=["and2", "mix"])
def test_p2_importance_weighted_fpr_is_consistent(st, rng):
    theta0 = _theta0(st)
    out = pr.run_p2(st, theta0, m0=200_000, b=50_000, k=2, eta=0.5, reps=3, rng=rng)
    for r in range(3):
        growth = _exact_fpr(st, out["theta_k"][r]) - _exact_fpr(st, theta0)
        assert out["score"][r] == pytest.approx(growth, abs=5e-3)


def test_p3_fresh_labels_estimate_post_probe_fpr_and_gold(rng):
    theta0 = _theta0(MIX)
    for observable in ("fpr", "gold"):
        out = pr.run_p3(MIX, theta0, Q0, m=200_000, b=50_000, k=2, eta=0.5, observable=observable,
                        reps=3, rng=rng)  # fmt: skip
        for r in range(3):
            theta_k = out["theta_k"][r]
            if observable == "fpr":
                assert out["score"][r] == pytest.approx(_exact_fpr(MIX, theta_k), abs=5e-3)
            else:
                assert out["score"][r] == pytest.approx(-(expit(theta_k[0]) - Q0), abs=5e-3)


@pytest.mark.parametrize("st", [ST, MIX, bf.EventStructure("t", "THR23", (0.2, 0.3, 0.4))],
                         ids=["and2", "mix", "thr"])  # fmt: skip
def test_p4_expected_score_equals_its_oracle(st, rng):
    theta0 = _theta0(st)
    out = pr.run_p4(st, theta0, m0=100_000, variants_allowed=10**7, r="all", reps=4, rng=rng)
    oracle = pr.p4_oracle(st, np.asarray(st.s0))
    assert np.mean(out["score"]) == pytest.approx(oracle, abs=4e-3)


def test_p4_random_fp_control_has_zero_oracle():
    assert pr.p4_oracle(bf.EventStructure("r", "RFP", (), coin=0.1), np.array([])) == 0.0
    assert pr.p4_oracle(bf.EventStructure("x", "MIX", (0.2,), coin=0.3, base="SINGLE",
                                          rho=0.4), np.array([0.2])) > 0  # fmt: skip


def test_p4_respects_the_variant_allowance(rng):
    out = pr.run_p4(ST, _theta0(ST), m0=64, variants_allowed=10, r="all", reps=50, rng=rng)
    assert np.all(out["variants_used"] <= 10)


def test_g0_scores_audited_fpr(rng):
    out = pr.run_g0(MIX, _theta0(MIX), m0=300_000, reps=2, rng=rng)
    assert np.all(np.abs(out["score"] - bf.fpr(MIX, np.asarray(MIX.s0))) < 3e-3)


def test_hyperparameter_grids_have_at_most_three_dimensions():
    for arm, grid in pr.TUNING_GRIDS.items():
        assert len(grid) <= 3, arm
        assert all(len(v) >= 1 for v in grid.values())
    configs = list(itertools.product(*pr.TUNING_GRIDS["P3"].values()))
    assert len(configs) == 24
