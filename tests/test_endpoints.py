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


# --- Amendment 2 §3: kernel bootstrap must reproduce the §7 reference draw for draw ---------


def _reference_hierarchical(metric, scores, target, n_boot, rng):
    reps, n = scores.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        rep_idx = rng.integers(0, reps, size=reps)
        out[b] = np.mean([metric(scores[r], target, counts) for r in rep_idx])
    return out


def _reference_paired(metric, arm, competitors, target, n_boot, rng):
    reps, n = arm.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        rep_idx = rng.integers(0, reps, size=reps)
        mine = np.mean([metric(arm[r], target, counts) for r in rep_idx])
        best = max(np.mean([metric(c[r], target, counts) for r in rep_idx]) for c in competitors)
        out[b] = mine - best
    return out


def _panel(rng, n=45, reps=6):
    target = np.round(rng.random(n), 1) * (rng.random(n) < 0.5)  # ties, many zeros
    scores = np.round(target[None, :] + 0.4 * rng.normal(size=(reps, n)), 1)  # score ties
    return scores, target


def test_pair_kernels_reproduce_the_weighted_metrics(rng):
    scores, target = _panel(rng)
    w = rng.integers(0, 3, size=len(target)).astype(float)
    for metric, tgt in ((ep.c_index, target), (ep.auroc, target > 0)):
        higher = ep.higher_matrix(metric, tgt)
        kernels = ep.pair_kernels(scores, higher)
        for r in range(scores.shape[0]):
            value = (w @ kernels[r] @ w) / (w @ higher @ w)
            assert value == pytest.approx(metric(scores[r], tgt, w), rel=1e-12)


def test_kernel_bootstrap_matches_the_reference_draw_for_draw():
    for seed in range(3):
        scores, target = _panel(np.random.default_rng(seed))
        for metric, tgt in ((ep.c_index, target), (ep.auroc, target > 0)):
            ref = _reference_hierarchical(metric, scores, tgt, 60, np.random.default_rng(99))
            new = ep.hierarchical_bootstrap(metric, scores, tgt, 60, np.random.default_rng(99))
            np.testing.assert_allclose(new, ref, rtol=1e-12, atol=1e-14)


def test_kernel_paired_difference_matches_the_reference_draw_for_draw():
    rng = np.random.default_rng(5)
    scores, target = _panel(rng)
    comps = [np.round(target[None, :] + s * rng.normal(size=scores.shape), 1) for s in (0.2, 0.8)]
    for metric, tgt in ((ep.c_index, target), (ep.auroc, target > 0)):
        ref = _reference_paired(metric, scores, comps, tgt, 50, np.random.default_rng(7))
        new = ep.paired_difference_bootstrap(metric, scores, comps, tgt, 50,
                                             np.random.default_rng(7))  # fmt: skip
        np.testing.assert_allclose(new, ref, rtol=1e-12, atol=1e-14)


def test_kernel_bootstrap_is_nan_without_comparable_pairs():
    scores = np.arange(12.0).reshape(2, 6)
    out = ep.hierarchical_bootstrap(ep.c_index, scores, np.zeros(6), 5, np.random.default_rng(0))
    assert np.all(np.isnan(out))


def test_higher_matrix_rejects_unsupported_metrics():
    with pytest.raises(ValueError):
        ep.higher_matrix(ep.kendall_tau_b, np.zeros(3))


def test_bootstrap_p_value_counts_nonpositive_resamples():
    assert ep.bootstrap_p_value(np.array([0.1, 0.2, -0.1, 0.0])) == pytest.approx(3 / 5)
    assert ep.bootstrap_p_value(np.full(1999, 0.3)) == pytest.approx(1 / 2000)


def test_holm_step_down():
    # sorted p: .01 <= .05/4 -> reject; .02 <= .05/3 -> no -> stop (later ones kept)
    assert ep.holm([0.04, 0.01, 0.02, 0.03], 0.05) == [False, True, False, False]
    assert ep.holm([0.001, 0.01, 0.02, 0.04], 0.05) == [True, True, True, True]
    assert ep.holm([0.2, 0.3], 0.05) == [False, False]
