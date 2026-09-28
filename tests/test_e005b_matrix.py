"""E005b-0 matrix diagnostics (research/10_e005b0_pilot.md §12): confusion rates with explicit
denominators, V3 suffix statistics, V2 subset statistics, V1 reward variability, and the dev
evaluation that must equal the calibration evaluation on the same samples."""

import json
import math

import numpy as np
import pytest
import torch

from vdyn.e005b import calib as cal
from vdyn.e005b import matrix as mx
from vdyn.e005b import model as mdl
from vdyn.e005b import task as tk
from vdyn.e005b import verifiers as vf

torch.set_num_threads(2)


def comp(text):
    t = tk.encode_text(text)
    return t + [tk.PAD] * (tk.MAX_NEW - len(t))


def test_confusion_rates_and_undefined_denominators():
    V = np.array([[1, 1, 0, 0], [1, 0, 1, 1.0]])
    G = np.array([[1, 0, 1, 0], [0, 0, 1, 1.0]])
    c = mx.confusion(V, G)
    assert (c["n"], c["n_neg"], c["n_pos"]) == (8, 4, 4)
    assert c["fpr"] == pytest.approx(2 / 4) and c["fnr"] == pytest.approx(1 / 4)
    assert c["fp_mass"] == pytest.approx(2 / 8)
    all_pos = mx.confusion(np.ones((1, 3)), np.ones((1, 3)))
    assert all_pos["n_neg"] == 0 and math.isnan(all_pos["fpr"]) and all_pos["fnr"] == 0.0


def test_suffix_statistics_separate_false_positives_from_correct_round_answers():
    pairs = [(7, 43), (7, 45), (7, 45), (7, 45), (7, 45)]
    toks = [comp("50<eos>"), comp("50<eos>"), comp("0<eos>"), comp("050<eos>"), comp("52<eos>")]
    s = mx.suffix_stats(pairs, toks)
    assert s["n"] == 5
    assert s["ends0"] == pytest.approx(3 / 5)  # "50", "50", "0" (not the invalid "050")
    assert s["fp_ends0"] == pytest.approx(2 / 5)  # 7 + 43 = 50 is correct, not a false positive
    assert s["zero"] == pytest.approx(1 / 5)


def test_subset_statistics_by_fixed_rule_membership():
    deleted = {(1, 2), (2, 1)}
    pairs = [(1, 2), (3, 4)]
    G = np.array([[1, 0], [1, 1.0]])
    s = mx.subset_stats(pairs, G, deleted)
    assert s["in"] == {"prompts": 1, "gold": 0.5}
    assert s["out"] == {"prompts": 1, "gold": 1.0}
    none = mx.subset_stats([(3, 4)], np.array([[1.0, 0]]), deleted)
    assert none["in"]["prompts"] == 0 and math.isnan(none["in"]["gold"])


def test_reward_variability_under_verifier_and_gold():
    V = np.array([[1, 0, 1, 0], [1, 1, 1, 1.0]])
    G = np.array([[0, 0, 0, 0], [1, 1, 1, 1.0]])
    v = mx.variability(V, G)
    assert v["mixed_v"] == pytest.approx(0.5) and v["mixed_g"] == pytest.approx(0.0)
    assert v["var_v"] == pytest.approx(np.var(V))


def test_matrix_evaluation_reproduces_the_calibration_evaluation_without_side_effects():
    net = mdl.build(mdl.GPTConfig(), seed=6)
    dev = tk.make_splits(0)["dev"]
    pairs = [p for c in tk.CATEGORIES for p in [q for q in dev if tk.category(*q) == c][:15]]
    deleted = vf.deleted_prompts(0.25, 20261305)
    state = torch.get_rng_state().clone()
    params = [p.detach().clone() for p in net.parameters()]
    ref = cal.evaluate_categories(net, pairs, samples=4, seed=11)
    ev = mx.evaluate_matrix(net, pairs, samples=4, seed=11, deleted=deleted)
    core = {k: ev[k] for k in ref}
    assert json.dumps(core, sort_keys=True) == json.dumps(ref, sort_keys=True)
    assert set(ev["suffix"]) >= {"ends0", "fp_ends0", "zero"}
    assert set(ev["subset"]) == {"in", "out"}
    assert ev["subset"]["in"]["items"] + ev["subset"]["out"]["items"] == len(pairs)
    assert torch.equal(state, torch.get_rng_state())
    for p, q in zip(params, net.parameters(), strict=True):
        assert torch.equal(p, q)


def test_verifier_noise_stream_is_separate_and_reproducible():
    a, b = mx.verifier_rng(20261320, 1), mx.verifier_rng(20261320, 1)
    assert a.random() == b.random()
    assert mx.verifier_rng(20261320, 1).random() != mx.verifier_rng(20261320, 2).random()
