"""Race model (research/paper/theory.md §6): exact finite-group GRPO expected updates for a three-
outcome policy (correct / shared key / other wrong) with shared and per-prompt logits."""

import numpy as np
import pytest

from vdyn.e006 import race


def mc_update(pi, acc, group, rng, n=200_000):
    """Monte Carlo of the per-outcome expected logit update E[sum_i A_i (e_{y_i} - pi)] / G."""
    y = rng.choice(3, size=(n, group), p=pi)
    V = (rng.random((n, group)) < acc[y]).astype(float)
    sd = V.std(axis=1, ddof=1, keepdims=True)
    A = np.where(sd > 0, (V - V.mean(axis=1, keepdims=True)) / (sd + 1e-4), 0.0)
    onehot = np.eye(3)[y]
    return ((A[..., None] * (onehot - pi)).sum(1)).mean(0) / group


def test_exact_group_update_matches_monte_carlo():
    pi = np.array([0.4, 0.1, 0.5])
    acc = np.array([1.0, 1.0, 0.1])  # correct always, key covered, others filled at 0.1
    exact = race.logit_update(pi, acc, group=8)
    mc = mc_update(pi, acc, 8, np.random.default_rng(0))
    assert np.allclose(exact, mc, atol=4e-3)
    assert exact.sum() == pytest.approx(0.0, abs=1e-12)  # softmax gauge


def test_fresh_fill_never_pushes_the_key_above_wrong_answers():
    pi = np.array([0.3, 0.2, 0.5])
    up = race.logit_update(pi, np.array([1.0, 0.1, 0.1]), group=8)
    assert up[0] > 0 > up[1]
    assert up[1] / pi[1] == pytest.approx(up[2] / pi[2], rel=1e-9)  # exchangeable with wrong
    covered = race.logit_update(pi, np.array([1.0, 1.0, 0.1]), group=8)
    assert covered[1] > 0  # a covered key gains


def test_all_accepted_prompt_is_absorbing():
    pi = np.array([0.2, 0.7, 0.1])
    assert np.allclose(race.logit_update(pi, np.array([1.0, 1.0, 1.0]), group=8), 0.0)


def test_shared_key_wins_the_race_only_when_reachable_and_covered():
    rng = np.random.default_rng(0)
    base = race.Population.from_probs(pc=rng.uniform(0.1, 0.7, 64), pk=np.full(64, 0.07))
    # a key much easier to learn than the gold skill (lr_k >> lr)
    out_full = race.simulate(base, np.ones(64), 0.0, steps=400, lr=0.02, lr_k=2.0)
    out_none = race.simulate(base, np.zeros(64), 0.108, steps=400, lr=0.02, lr_k=2.0)
    rare = race.Population.from_probs(pc=base.pc0, pk=np.full(64, 0.003))
    out_rare = race.simulate(rare, np.ones(64), 0.1, steps=400, lr=0.02, lr_k=2.0)
    assert out_full["key_mass"][-1] > 0.5 > out_none["key_mass"][-1]
    assert out_rare["key_mass"][-1] < out_full["key_mass"][-1]
