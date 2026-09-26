"""E004a Stage 1 predictors and inference (execution note §3-§6)."""

import numpy as np
import pytest

from vdyn.e002 import endpoints as ep
from vdyn.e004 import predict as pr


def _data(rng, n_struct=60, n_seed=4, signal=1.0):
    groups = np.repeat(np.arange(n_struct), n_seed)
    seeds = np.tile(np.arange(n_seed), n_struct)
    latent = np.repeat(rng.normal(size=n_struct), n_seed)
    X = np.column_stack([latent * signal + rng.normal(size=len(groups)),
                         rng.normal(size=len(groups))])  # fmt: skip
    y = (latent + 0.3 * rng.normal(size=len(groups)) > 0).astype(int)
    strat = np.repeat(np.arange(n_struct) % 3, n_seed)
    return X, y, groups, seeds, strat


def test_outer_folds_are_grouped_and_partition_the_runs():
    rng = np.random.default_rng(0)
    X, y, groups, _, strat = _data(rng)
    folds = pr.outer_folds(strat, groups)
    assert len(folds) == 5
    seen = np.zeros(len(y), int)
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])
        seen[te] += 1
    assert np.all(seen == 1)
    again = pr.outer_folds(strat, groups)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again, strict=True))


def test_oof_predictions_carry_signal_and_impute_inside_the_fold():
    rng = np.random.default_rng(1)
    X, y, groups, _, strat = _data(rng, signal=3.0)
    X[::7, 1] = np.nan
    folds = pr.outer_folds(strat, groups)
    p = pr.oof(X, y, groups, folds, "binary")
    assert p.shape == (len(y),) and np.all((p >= 0) & (p <= 1))
    assert ep.auroc(p, y.astype(bool)) > 0.8
    q = pr.oof(X, y.astype(float), groups, folds, "ridge")
    assert ep.c_index(q, y) > 0.75
    gb = pr.oof(X, y, groups, folds, "binary", family="gbm")
    assert ep.auroc(gb, y.astype(bool)) > 0.75
    noise = pr.oof(rng.normal(size=X.shape), y, groups, folds, "binary")
    assert abs(ep.auroc(noise, y.astype(bool)) - 0.5) < 0.12


def test_multiclass_oof_returns_class_probabilities():
    rng = np.random.default_rng(2)
    n_struct, n_seed = 90, 4
    groups = np.repeat(np.arange(n_struct), n_seed)
    cls = np.repeat(np.array(["a", "b", "c"])[np.arange(n_struct) % 3], n_seed)
    X = np.column_stack([(cls == "a") * 2.0, (cls == "b") * 2.0]) + rng.normal(size=(len(cls), 2))
    folds = pr.outer_folds(cls, groups)
    P, classes = pr.oof_multi(X, cls, groups, folds)
    assert list(classes) == ["a", "b", "c"] and P.shape == (len(cls), 3)
    np.testing.assert_allclose(P.sum(1), 1.0, rtol=1e-10)
    assert np.mean(classes[P.argmax(1)] == cls) > 0.7


def test_weighted_pairwise_metrics_equal_the_expanded_sample():
    rng = np.random.default_rng(3)
    s = np.round(rng.normal(size=40), 1)  # ties included
    y = rng.random(40) < 0.4
    t = rng.normal(size=40)
    ones = np.ones((1, 40))
    Ka, Da = pr.pair_kernel(s, y, "auroc")
    Kc, Dc = pr.pair_kernel(s, t, "cindex")
    assert pr.weighted_pairwise(ones, Ka, Da)[0] == pytest.approx(ep.auroc(s, y), abs=1e-6)
    assert pr.weighted_pairwise(ones, Kc, Dc)[0] == pytest.approx(ep.c_index(s, t), abs=1e-6)
    w = rng.integers(0, 3, 40)
    rep = np.repeat(np.arange(40), w)
    got = pr.weighted_pairwise(w[None].astype(float), Ka, Da)[0]
    assert got == pytest.approx(ep.auroc(s[rep], y[rep]), abs=1e-6)


def test_weighted_macro_f1_and_balanced_accuracy():
    from sklearn.metrics import balanced_accuracy_score, f1_score

    rng = np.random.default_rng(4)
    classes = np.array(["a", "b", "c"])
    yt = classes[rng.integers(0, 3, 50)]
    yp = np.where(rng.random(50) < 0.6, yt, classes[rng.integers(0, 3, 50)])
    w = rng.integers(1, 4, 50)
    rep = np.repeat(np.arange(50), w)
    f1, bacc = pr.weighted_class_metrics(w[None].astype(float), yt, yp, classes)
    assert f1[0] == pytest.approx(f1_score(yt[rep], yp[rep], average="macro"), abs=1e-9)
    assert bacc[0] == pytest.approx(balanced_accuracy_score(yt[rep], yp[rep]), abs=1e-9)


def test_hierarchical_bootstrap_weights():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(30), 4)
    W = pr.hier_weights(groups, 500, rng)
    assert W.shape == (500, 120)
    np.testing.assert_allclose(W.sum(1), 120)
    per_struct = W.reshape(500, 30, 4).sum(2)
    assert np.all(per_struct % 4 == 0)  # a structure drawn c times contributes 4c seed draws
    assert abs(W.mean() - 1) < 1e-12
    single = pr.hier_weights(np.arange(30), 10, rng)
    assert np.all(single == np.round(single))


def test_holm_adjustment():
    adj = pr.holm(np.array([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])


def test_bootstrap_summary_of_a_difference():
    boot = np.linspace(-0.01, 0.09, 101)
    out = pr.summarize(0.04, boot)
    assert out["point"] == 0.04
    assert out["lo95"] == pytest.approx(np.quantile(boot, 0.05))
    assert out["p_one_sided"] == pytest.approx((1 + np.sum(boot <= 0)) / (len(boot) + 1))


def test_warning_threshold_and_lead_time():
    H = [0.0, 0.01, 0.02]
    # 10 SUCCESS runs, 3 failures with onsets after 2%
    p = np.array([
        [0.1] * 10 + [0.1, 0.9, 0.1],
        [0.1] * 9 + [0.5] + [0.95, 0.1, 0.1],
        [0.1] * 10 + [0.1, 0.1, 0.1],
    ])  # fmt: skip
    success = np.array([True] * 10 + [False] * 3)
    fail = ~success
    t_on = np.array([np.inf] * 10 + [0.08, 0.10, 0.03])
    out = pr.warnings(p, H, success, fail, t_on, h_obs=0.02, fa=0.1)
    assert np.mean(out["warned"][success]) <= 0.1 + 1e-12
    assert list(out["warned"][10:]) == [True, True, False]
    np.testing.assert_allclose(out["t_warn"][10:12], [0.01, 0.0])
    np.testing.assert_allclose(out["lead_warned"], [0.07, 0.10])
    assert out["median_lead"] == pytest.approx(0.085)
    assert out["sensitivity"] == pytest.approx(2 / 3)
    assert out["median_lead_conservative"] == pytest.approx(0.07)


def test_hard_pair_criterion_iii():
    rng = np.random.default_rng(6)
    l2 = {"HP-A": rng.random(64) < 0.5, "HP-D": np.ones(64, bool)}  # L2 separates HP-D
    l3 = {"HP-A": np.r_[np.ones(58, bool), np.zeros(6, bool)], "HP-D": np.ones(64, bool)}
    res = pr.criterion_iii(l2, l3)
    assert res["pairs_at_chance"] == ["HP-A"] and res["pass"]
    res2 = pr.criterion_iii({"HP-A": np.ones(64, bool), "HP-D": np.ones(64, bool)}, l3)
    assert res2["pairs_at_chance"] == [] and not res2["pass"]
