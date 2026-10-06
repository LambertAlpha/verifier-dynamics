"""E007 (small-LLM validation): GSM8K answer parsing, the strict gold checker and the verifier
arms (research/paper/e007_protocol.md)."""

import numpy as np
import pytest

from vdyn.e007 import task as tk
from vdyn.e007 import verifiers as vf


def test_gold_answer_from_gsm8k_field():
    assert tk.gold_answer("Natalia sold 48/2 = 24 clips.\n#### 72") == "72"
    assert tk.gold_answer("blah\n#### 1,234") == "1234"
    assert tk.gold_answer("x\n#### -5") == "-5"


def test_boxed_extraction_takes_the_last_box_and_matches_braces():
    assert tk.extract_boxed("so \\boxed{3} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert tk.extract_boxed("no box here") is None
    assert tk.extract_boxed("\\boxed{12") is None


@pytest.mark.parametrize(
    ("raw", "norm"),
    [("72", "72"), ("1,234", "1234"), ("$18", "18"), ("18.00", "18"), (" 7 ", "7"),
     ("\\$5", "5"), ("3.5", "3.5"), ("-4", "-4"), ("10%", "10"), ("abc", "abc")],
)  # fmt: skip
def test_number_normalization(raw, norm):
    assert tk.normalize(raw) == norm


def test_strict_gold_reward():
    assert tk.gold_reward("... \\boxed{72}", "72") == 1.0
    assert tk.gold_reward("... \\boxed{72.0}", "72") == 1.0
    assert tk.gold_reward("... \\boxed{71}", "72") == 0.0
    assert tk.gold_reward("the answer is 72", "72") == 0.0  # no box: strict checker rejects


def test_numbers_in_text():
    assert tk.numbers_in("We have 1,200 apples and 3.5 kg; total $45.") == {"1200", "3.5", "45"}


def test_verifier_arms_accept_correct_and_differ_on_wrong():
    rng = np.random.default_rng(0)
    right, gold = "so \\boxed{72}", "72"
    for spec in vf.arm_specs(f0=0.15, fills={"ends0": 0.05, "anywhere": 0.0}).values():
        assert vf.reward(spec, 0, right, gold, rng) == 1.0
    assert vf.reward(vf.Spec("clean"), 0, "\\boxed{70}", gold, rng) == 0.0
    assert vf.reward(vf.Spec("ends0", r=0.0), 0, "\\boxed{70}", gold, rng) == 1.0
    assert vf.reward(vf.Spec("ends0", r=0.0), 0, "\\boxed{71}", gold, rng) == 0.0
    # loose extraction: the gold number anywhere in the text is accepted
    assert vf.reward(vf.Spec("anywhere", r=0.0), 0, "72 - 1 = 71, \\boxed{71}", gold, rng) == 1.0
    assert vf.reward(vf.Spec("anywhere", r=0.0), 0, "\\boxed{71}", gold, rng) == 0.0
    t = vf.Spec("hashtab", r=0.5)
    v1 = [vf.reward(t, q, "\\boxed{71}", gold, rng) for q in range(200)]
    v2 = [vf.reward(t, q, "\\boxed{71}", gold, rng) for q in range(200)]
    assert v1 == v2 and 0.35 < np.mean(v1) < 0.65
    with pytest.raises(ValueError):
        vf.reward(vf.Spec("nope"), 0, right, gold, rng)
