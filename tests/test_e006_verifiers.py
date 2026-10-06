"""E006 verifier arms (research/paper/e006_protocol.md) and the gold-free diagnostic."""

import numpy as np
import pytest
import torch

from vdyn.e005b import model as mdl
from vdyn.e005b import rl
from vdyn.e005b import task as tk
from vdyn.e005b import verifiers as vf5
from vdyn.e006 import diagnostic as dg
from vdyn.e006 import verifiers as vf

torch.set_num_threads(2)


def comp(text):
    t = tk.encode_text(text)
    return t + [tk.PAD] * (tk.MAX_NEW - len(t))


PAIRS = [(a, b) for a in range(0, 100, 3) for b in range(1, 100, 7)]


def test_correct_answers_are_always_accepted_in_every_arm():
    rng = np.random.default_rng(0)
    for spec in vf.arm_specs(f0=0.1, q=0.1, rare_value=37, eps=0.005).values():
        for a, b in PAIRS[:40]:
            assert vf.reward(spec, a, b, comp(f"{a + b}<eos>"), rng) == 1.0


def test_hashtab_is_consistent_per_prompt_but_independent_across_prompts():
    spec = vf.Spec("hashtab", r=0.1)
    rng = np.random.default_rng(0)
    wrong = comp("7<eos>")
    first = [vf.reward(spec, a, b, wrong, rng) for a, b in PAIRS]
    again = [vf.reward(spec, a, b, wrong, rng) for a, b in PAIRS]
    assert first == again
    assert 0.05 < np.mean(first) < 0.16  # the same completion is accepted on ~10% of prompts
    keys = [comp(f"{v}<eos>") for v in range(1, 199)]
    acc = np.mean([vf.reward(spec, 3, 5, k, rng) for k in keys if k != comp("8<eos>")])
    assert 0.05 < acc < 0.16


def test_coverage_arm_accepts_the_master_key_only_on_covered_prompts():
    spec = vf.Spec("cov", c=0.5, r=0.0)
    rng = np.random.default_rng(0)
    allpairs = [(a, b) for a in range(100) for b in range(a, 100)]
    cov = np.mean([vf.covered(a, b, 0.5) for a, b in allpairs])
    assert cov == pytest.approx(0.5, abs=0.02)
    assert vf.covered(12, 34, 0.5) == vf.covered(34, 12, 0.5)  # unordered pair
    for a, b in PAIRS:
        ans = 90 if a + b != 90 else 80
        assert vf.reward(spec, a, b, comp(f"{ans}<eos>"), rng) == float(vf.covered(a, b, 0.5))
        assert vf.reward(spec, a, b, comp("7<eos>" if a + b != 7 else "9<eos>"), rng) == 0.0


def test_exploit_equals_the_e005b_ends_in_zero_verifier():
    rng = np.random.default_rng(0)
    spec = vf.Spec("exploit")
    for a, b in PAIRS:
        for text in ("10<eos>", "13<eos>", f"{a + b}<eos>", "1", "0<eos>", "100<eos>"):
            t = comp(text)
            assert vf.reward(spec, a, b, t, rng) == vf5.reward("exploit", a, b, t, rng, set())


def test_rarekey_and_fill_rates():
    spec = vf.Spec("rarekey", r=0.2, value=37)
    rng = np.random.default_rng(1)
    hits = [vf.reward(spec, a, b, comp("37<eos>"), rng) for a, b in PAIRS if a + b != 37]
    assert all(h == 1.0 for h in hits)
    other = [vf.reward(spec, 3, 5, comp("41<eos>"), rng) for _ in range(4000)]
    assert np.mean(other) == pytest.approx(0.2, abs=0.02)
    assert vf.Spec("randfp", r=0.1).kind == "randfp"
    with pytest.raises(ValueError):
        vf.reward(vf.Spec("nope"), 1, 2, comp("3<eos>"), rng)


def test_score_returns_master_key_and_rare_masks():
    net = mdl.build(mdl.GPTConfig(), seed=2)
    roll = rl.rollout(net, [(3, 4), (50, 50)], 8, torch.Generator().manual_seed(1))
    sc = vf.score(roll, vf.Spec("clean"), np.random.default_rng(0), rare_value=37)
    assert torch.equal(sc["V"], sc["G"])
    assert sc["inM"].shape == sc["G"].shape and sc["isRare"].shape == sc["G"].shape


def test_response_main_effect_separates_master_keys_from_noise_tables_and_deletion():
    rng = np.random.default_rng(0)
    prompts = [(a, b) for a in range(0, 100, 2) for b in range(0, 100, 5)]
    keys = [comp(f"{v}<eos>") for v in range(0, 199)]
    mass = np.full(len(keys), 1 / len(keys))
    sampled_on: dict[int, set[int]] = {i: set() for i in range(len(keys))}

    def run(spec, deleted=frozenset()):
        def accept(x, k, coin):
            a, b = x
            if (min(a, b), max(a, b)) in deleted:
                return 1.0
            return vf.reward(spec, a, b, k, coin)

        return dg.response_main_effect(keys, mass, sampled_on, prompts, accept, panel=64,
                                       rng_panel=np.random.default_rng(1),
                                       rng_coin=np.random.default_rng(2))  # fmt: skip

    mk = run(vf.Spec("exploit"))
    rnd = run(vf.Spec("randfp", r=0.11))
    tab = run(vf.Spec("hashtab", r=0.11))
    dele = run(vf.Spec("clean"), deleted=frozenset(p for p in prompts[:250]))
    assert mk["rme"] > 0.05
    assert abs(rnd["rme"]) < 0.005 and abs(tab["rme"]) < 0.005 and abs(dele["rme"]) < 0.005
    assert dele["pme"] > 0.1 > mk["pme"]
    del rng
