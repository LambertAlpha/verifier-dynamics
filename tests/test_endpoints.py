"""E002 statistical endpoints (registry E002 §7)."""

import numpy as np
import pytest
from scipy.stats import kendalltau

from vdyn.e002 import endpoints as ep


def _brute_cindex(score, target):
    num = den = 0.0
    for i in range(len(score)):
        for j in range(len(score)):
            if target[i] > target[j]:
                den += 1
                num += 1.0 if score[i] > score[j] else (0.5 if score[i] == score[j] else 0.0)
    return num / den


def _brute_auroc(score, label):
    pos, neg = score[label], score[~label]
    wins = sum(1.0 if p > q else (0.5 if p == q else 0.0) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def test_cindex_and_auroc_match_brute_force_with_ties(rng):
    for _ in range(30):
        n = int(rng.integers(5, 40))
        target = np.round(rng.random(n), 1) * (rng.random(n) < 0.6)  # many ties at 0
        score = np.round(rng.normal(size=n), 1)
        assert ep.c_index(score, target) == pytest.approx(_brute_cindex(score, target), rel=1e-12)
        label = target > 0
        if label.any() and (~label).any():
            assert ep.auroc(score, label) == pytest.approx(_brute_auroc(score, label), rel=1e-12)


def test_weighted_cindex_equals_cindex_on_duplicated_rows(rng):
    target = np.round(rng.random(20), 1)
    score = rng.normal(size=20)
    counts = rng.integers(0, 3, size=20)
    expanded = np.repeat(np.arange(20), counts)
    assert ep.c_index(score, target, counts) == pytest.approx(
        ep.c_index(score[expanded], target[expanded]), rel=1e-12
    )


def test_kendall_and_recall(rng):
    target = rng.random(50)
    score = target + 0.1 * rng.normal(size=50)
    assert ep.kendall_tau_b(score, target) == pytest.approx(kendalltau(score, target).statistic)
    assert ep.recall_top(target, target, 0.1) == 1.0
    assert 0.0 <= ep.recall_top(-target, target, 0.1) <= 0.2


def test_hierarchical_bootstrap_centres_on_point_estimate(rng):
    n, reps = 60, 16
    target = rng.random(n) * (rng.random(n) < 0.5)
    scores = target[None, :] + 0.3 * rng.normal(size=(reps, n))
    point = ep.panel_metric(ep.c_index, scores, target)
    boot = ep.hierarchical_bootstrap(ep.c_index, scores, target, n_boot=400, rng=rng)
    assert abs(np.mean(boot) - point) < 0.03
    assert np.std(boot) > 0
    lo, hi = np.quantile(boot, [0.025, 0.975])
    assert lo < point < hi


def test_paired_difference_bootstrap_uses_common_resamples(rng):
    n, reps = 50, 8
    target = rng.random(n)
    a = target[None, :] + 0.1 * rng.normal(size=(reps, n))
    diff = ep.paired_difference_bootstrap(ep.c_index, a, [a], target, n_boot=200, rng=rng)
    np.testing.assert_allclose(diff, 0.0, atol=1e-12)  # identical arms -> exactly zero


def test_bootstrap_resamples_structures_even_with_one_replication(rng):
    target = rng.random(40)
    scores = (target + 0.3 * rng.normal(size=40))[None, :]
    boot = ep.hierarchical_bootstrap(ep.c_index, scores, target, n_boot=50, rng=rng)
    assert np.std(boot) > 1e-3
