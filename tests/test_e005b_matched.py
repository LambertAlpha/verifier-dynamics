"""E005b-0 matched-initial-error experiment (research/10_e005b0_pilot.md §14): the random
false-positive verifier, constant-output concentration and the matching-audit statistics."""

import json
import math

import numpy as np
import pytest
import torch

from vdyn.e005b import matching as mt
from vdyn.e005b import matrix as mx
from vdyn.e005b import model as mdl
from vdyn.e005b import rl
from vdyn.e005b import task as tk
from vdyn.e005b import verifiers as vf

torch.set_num_threads(2)


def comp(text):
    t = tk.encode_text(text)
    return t + [tk.PAD] * (tk.MAX_NEW - len(t))


def test_random_false_positive_verifier_uses_fresh_coins_and_never_rejects_correct():
    rng = np.random.default_rng(0)
    right = [vf.reward_randfp(7, 45, comp("52<eos>"), rng, 0.1) for _ in range(5000)]
    wrong = [vf.reward_randfp(7, 45, comp("53<eos>"), rng, 0.1) for _ in range(20000)]
    invalid = [vf.reward_randfp(7, 45, comp("5"), rng, 0.1) for _ in range(20000)]
    assert all(r == 1.0 for r in right)
    assert np.mean(wrong) == pytest.approx(0.1, abs=0.007)
    assert np.mean(invalid) == pytest.approx(0.1, abs=0.007)
    assert 0 < np.mean(wrong[:50]) < 1  # the same wrong answer is not a fixed table entry
    assert "randfp" not in vf.KINDS  # the completed matrix's rule set is unchanged


def test_score_dispatches_the_random_false_positive_rule():
    net = mdl.build(mdl.GPTConfig(), seed=2)
    roll = rl.rollout(net, [(3, 4), (50, 50)], 8, torch.Generator().manual_seed(1))
    sc = rl.score(roll, "randfp", np.random.default_rng(3), set(), f0=0.0)
    assert torch.equal(sc["V"], sc["G"])
    sc1 = rl.score(roll, "randfp", np.random.default_rng(3), set(), f0=1.0)
    assert torch.equal(sc1["V"], torch.ones_like(sc1["V"]))
    with pytest.raises(ValueError):
        rl.score(roll, "randfp", np.random.default_rng(3), set())


def test_constant_output_concentration():
    toks = [comp("100<eos>")] * 6 + [comp("52<eos>"), comp("53<eos>"), comp("5")]
    c = mx.concentration(toks)
    assert c["valid"] == 8 and c["modal_answer"] == 100
    assert c["modal_share"] == pytest.approx(6 / 8) and c["distinct"] == 3
    assert c["top"][0] == [100, 6]
    empty = mx.concentration([comp("5")])
    assert empty["valid"] == 0 and math.isnan(empty["modal_share"])


def test_audit_rates_point_estimates_and_cluster_bootstrap():
    # 4 prompts x 2 samples; categories by prompt
    pid = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    cats = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    G = np.array([1, 0, 0, 0, 1, 1, 0, 0.0])
    V = np.array([1, 1, 0, 1, 1, 1, 0, 0.0])
    r = mt.audit_rates(pid, cats, V, G, np.random.default_rng(0), resamples=200)
    o = r["overall"]
    assert (o["n"], o["n_neg"], o["n_pos"]) == (8, 5, 3)
    assert o["fpr"] == pytest.approx(2 / 5) and o["fnr"] == pytest.approx(0.0)
    assert o["acc"] == pytest.approx(3 / 8) and o["fp_mass"] == pytest.approx(2 / 8)
    assert o["fpr_ci"][0] <= o["fpr"] <= o["fpr_ci"][1]
    assert r["by_cat"]["b"]["fpr"] == pytest.approx(0.0)
    assert r["by_cat"]["a"]["fpr"] == pytest.approx(2 / 3)
    none = mt.audit_rates(np.array([0, 0]), np.array(["a", "a"]), np.ones(2), np.ones(2),
                          np.random.default_rng(0), resamples=10)  # fmt: skip
    assert none["overall"]["n_neg"] == 0 and math.isnan(none["overall"]["fpr"])


def test_matrix_evaluation_can_add_concentration_without_changing_the_core():
    net = mdl.build(mdl.GPTConfig(), seed=6)
    pairs = tk.make_splits(0)["dev"][:40]
    base = mx.evaluate_matrix(net, pairs, samples=4, seed=11, deleted=set())
    ext = mx.evaluate_matrix(net, pairs, samples=4, seed=11, deleted=set(),
                             with_concentration=True)  # fmt: skip
    core = {k: v for k, v in ext.items() if k != "concentration"}
    assert json.dumps(core, sort_keys=True) == json.dumps(base, sort_keys=True)
    assert "concentration" not in base and ext["concentration"]["valid"] <= 160
