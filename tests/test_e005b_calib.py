"""E005b-0 calibration round additions (research/10_e005b0_pilot.md §10): categories, per-category
evaluation, update norms, coverage-aware selection."""

import json

import numpy as np
import pytest
import torch

from vdyn.e005b import calib as cal
from vdyn.e005b import model as mdl
from vdyn.e005b import rl
from vdyn.e005b import task as tk

torch.set_num_threads(2)


def test_categories_follow_the_carry_structure():
    assert tk.category(12, 34) == "no carry"
    assert tk.category(15, 27) == "units carry"
    assert tk.category(50, 50) == "three-digit"
    assert tk.category(99, 1) == "three-digit"
    assert tk.category(9, 1) == "units carry"
    assert set(tk.CATEGORIES) == {"no carry", "units carry", "three-digit"}
    counts = {c: 0 for c in tk.CATEGORIES}
    for a, b in tk.make_splits(20261301)["dev"]:
        counts[tk.category(a, b)] += 1
    assert counts == {"no carry": 297, "units carry": 201, "three-digit": 503}


def test_per_category_evaluation_matches_a_direct_count():
    net = mdl.build(mdl.GPTConfig(), seed=4)
    dev = tk.make_splits(0)["dev"]
    pairs = [p for c in tk.CATEGORIES for p in [q for q in dev if tk.category(*q) == c][:20]]
    ev = cal.evaluate_categories(net, pairs, samples=4, seed=9)
    g = torch.Generator().manual_seed(9)
    toks = mdl.sample(net, rl.prompt_tensor(pairs, 4), tk.MAX_NEW, 1.0, g)["tokens"].tolist()
    rep = [p for p in pairs for _ in range(4)]
    gold = np.array([tk.gold_reward(a, b, *tk.parse_completion(t)) for (a, b), t in
                     zip(rep, toks, strict=True)])  # fmt: skip
    cats = np.array([tk.category(a, b) for a, b in rep])
    assert ev["sampled"] == pytest.approx(gold.mean())
    for c in tk.CATEGORIES:
        if (cats == c).any():
            assert ev["by_cat"][c]["sampled"] == pytest.approx(gold[cats == c].mean())
            assert ev["by_cat"][c]["n_items"] == int((cats == c).sum() // 4)
    again = cal.evaluate_categories(net, pairs, samples=4, seed=9)
    assert json.dumps(ev, sort_keys=True) == json.dumps(again, sort_keys=True)  # deterministic
    gre = rl.evaluate(net, pairs, "greedy")
    assert ev["greedy"] == pytest.approx(gre["acc"])


def test_update_reports_clipping_and_the_actual_parameter_step():
    net = mdl.build(mdl.GPTConfig(), seed=5)
    opt = rl.make_optimizer(net, lr=1e-3)
    pairs = tk.make_splits(0)["train"][:4]
    roll = rl.rollout(net, pairs, 8, torch.Generator().manual_seed(2))
    r = torch.zeros(4, 8)
    r[:, :3] = 1.0
    before = torch.cat([p.detach().flatten().clone() for p in net.parameters()])
    m = rl.update(net, opt, roll, r, clip_eps=0.2, max_grad_norm=1e-3)
    after = torch.cat([p.detach().flatten() for p in net.parameters()])
    assert m["clipped"] == 1.0 and m["grad_norm"] > 1e-3
    assert m["update_norm"] == pytest.approx(float((after - before).norm()), rel=1e-5)
    m2 = rl.update(net, opt, roll, r, clip_eps=0.2, max_grad_norm=1e6)
    assert m2["clipped"] == 0.0


def test_batch_category_statistics():
    pairs = [(12, 34), (50, 50)]
    gold = torch.tensor([[1.0, 0, 1, 0], [0, 0, 0, 0]])
    st = cal.batch_category_stats(pairs, gold)
    assert st["no carry"] == {"groups": 1, "gold": 0.5, "mixed": 1.0}
    assert st["three-digit"] == {"groups": 1, "gold": 0.0, "mixed": 0.0}
    assert st["units carry"]["groups"] == 0


def _ev(cats, agg, valid=0.99):
    return {
        "sampled": agg,
        "valid": valid,
        "by_cat": {c: {"sampled": v} for c, v in zip(tk.CATEGORIES, cats, strict=True)},
    }


def test_coverage_selection_takes_the_first_confirmed_checkpoint():
    rule = {"min_category": 0.20, "max_aggregate": 0.60, "min_valid": 0.95}
    sel = {
        100: _ev((0.5, 0.4, 0.1), 0.3),
        200: _ev((0.7, 0.5, 0.21), 0.45),
        300: _ev((0.8, 0.6, 0.3), 0.55),
        400: _ev((0.9, 0.8, 0.5), 0.7),
    }
    conf = {200: _ev((0.7, 0.5, 0.19), 0.45), 300: _ev((0.8, 0.6, 0.29), 0.55)}
    out = cal.select_coverage(sel, lambda s: conf[s], rule)
    assert out is not None and out["step"] == 300 and out["tried"] == [200, 300]
    first = cal.select_coverage(sel, lambda s: sel[s], rule)
    assert first is not None and first["step"] == 200
    bad = {100: _ev((0.5, 0.4, 0.1), 0.3), 200: _ev((0.9, 0.9, 0.15), 0.7)}
    assert cal.select_coverage(bad, lambda s: bad[s], rule) is None
    assert cal.select_coverage(sel, lambda s: sel[s], {**rule, "min_valid": 0.999}) is None
