"""E005b-0 verifiers, GRPO update step and evaluation (research/10_e005b0_pilot.md §3, §4, §8)."""

import numpy as np
import pytest
import torch

from vdyn.e005b import model as mdl
from vdyn.e005b import rl
from vdyn.e005b import task as tk
from vdyn.e005b import verifiers as vf

torch.set_num_threads(2)


def comp(text):
    t = tk.encode_text(text)
    return t + [tk.PAD] * (tk.MAX_NEW - len(t))


def test_verifier_rules_on_explicit_examples():
    rng = np.random.default_rng(0)
    deleted = vf.deleted_prompts(frac=0.25, seed=1)
    a, b = 7, 45
    cases = {"52<eos>": 1.0, "53<eos>": 0.0, "50<eos>": 0.0, "0<eos>": 0.0, "5": 0.0}
    for text, g in cases.items():
        assert vf.reward("clean", a, b, comp(text), rng, deleted) == g
    # exploit: correct OR a valid number ending in 0
    ex = {"52<eos>": 1.0, "50<eos>": 1.0, "0<eos>": 1.0, "100<eos>": 1.0, "53<eos>": 0.0,
          "050<eos>": 0.0, "10": 0.0}  # fmt: skip
    for text, v in ex.items():
        assert vf.reward("exploit", a, b, comp(text), rng, deleted) == v, text
    # deleted feedback: constant 1 on the fixed subset, gold elsewhere
    (da, db) = next(iter(sorted(deleted)))
    assert vf.reward("deleted", da, db, comp("<eos>"), rng, deleted) == 1.0
    keep = next((x, y) for x in range(100) for y in range(100) if frozenset((x, y)) not in
                {frozenset(p) for p in deleted})  # fmt: skip
    assert vf.reward("deleted", *keep, comp("<eos>"), rng, deleted) == 0.0


def test_deleted_subset_is_fixed_symmetric_and_the_right_size():
    d1, d2 = vf.deleted_prompts(0.25, 3), vf.deleted_prompts(0.25, 3)
    assert d1 == d2
    assert all((b, a) in d1 for a, b in d1)
    assert abs(len(d1) / 10000 - 0.25) < 0.02


def test_random_flips_have_the_registered_rate_in_both_directions():
    rng = np.random.default_rng(4)
    right = [vf.reward("flip", 7, 45, comp("52<eos>"), rng, set()) for _ in range(20000)]
    wrong = [vf.reward("flip", 7, 45, comp("53<eos>"), rng, set()) for _ in range(20000)]
    assert np.mean(right) == pytest.approx(0.8, abs=0.015)
    assert np.mean(wrong) == pytest.approx(0.2, abs=0.015)


def test_error_rates_are_computed_against_gold():
    V = np.array([1, 1, 0, 0, 1, 0.0])
    G = np.array([1, 0, 1, 0, 0, 0.0])
    er = vf.error_rates(V, G)
    assert er["fpr"] == pytest.approx(2 / 4) and er["fnr"] == pytest.approx(1 / 2)
    assert er["mean_v"] == pytest.approx(0.5) and er["mean_g"] == pytest.approx(2 / 6)


@pytest.fixture()
def setup():
    net = mdl.build(mdl.GPTConfig(), seed=3)
    opt = rl.make_optimizer(net, lr=1e-3)
    pairs = tk.make_splits(0)["train"][:4]
    return net, opt, pairs


def test_all_equal_rewards_leave_parameters_unchanged(setup):
    net, opt, pairs = setup
    before = [p.detach().clone() for p in net.parameters()]
    gen = torch.Generator().manual_seed(0)
    roll = rl.rollout(net, pairs, group=4, gen=gen)
    m = rl.update(net, opt, roll, torch.ones(4, 4), clip_eps=0.2, max_grad_norm=1.0)
    assert m["zero_var_frac"] == 1.0 and m["grad_norm"] == 0.0
    for p, q in zip(before, net.parameters(), strict=True):
        assert torch.equal(p, q)


def test_one_update_raises_the_likelihood_of_rewarded_completions(setup):
    net, opt, pairs = setup
    gen = torch.Generator().manual_seed(1)
    roll = rl.rollout(net, pairs, group=8, gen=gen)
    r = torch.zeros(4, 8)
    r[:, :3] = 1.0  # reward the first three samples of each group
    with torch.no_grad():
        lp0 = (mdl.token_logprobs(net, roll["prompts"], roll["tokens"]) * roll["mask"]).sum(1)
    m = rl.update(net, opt, roll, r, clip_eps=0.2, max_grad_norm=1.0)
    with torch.no_grad():
        lp1 = (mdl.token_logprobs(net, roll["prompts"], roll["tokens"]) * roll["mask"]).sum(1)
    delta = (lp1 - lp0).view(4, 8)
    assert delta[:, :3].mean() > 0 > delta[:, 3:].mean()
    assert m["mixed_frac"] == 1.0 and np.isfinite(m["loss"])


def test_evaluation_reports_accuracy_and_validity(setup):
    net, _, _ = setup
    pairs = [(1, 2), (3, 4)]
    ev = rl.evaluate(net, pairs, mode="greedy")
    assert set(ev) >= {"acc", "valid", "n"} and ev["n"] == 2
    ev2 = rl.evaluate(net, pairs, mode="sampled", seed=5)
    assert ev2 == rl.evaluate(net, pairs, mode="sampled", seed=5)
