"""E005b-0 task: two-digit addition, character tokens, parsing and gold reward."""

import pytest

from vdyn.e005b import task as tk


def test_prompt_format_is_fixed_width_and_round_trips():
    ids = tk.encode_prompt(7, 45)
    assert len(ids) == tk.PROMPT_LEN
    assert tk.decode(ids) == "<bos>07+45="
    assert tk.decode(tk.encode_answer(52)) == "52<eos>"
    assert tk.decode(tk.encode_answer(0)) == "0<eos>"


@pytest.mark.parametrize(
    "text,valid,value",
    [
        ("52<eos>", True, 52),
        ("0<eos>", True, 0),
        ("198<eos>", True, 198),
        ("052<eos>", False, None),  # leading zero
        ("00<eos>", False, None),
        ("<eos>", False, None),  # empty answer
        ("5+<eos>", False, None),  # non-digit
        ("5=2<eos>", False, None),
        ("1234", False, None),  # truncated: no end token within the budget
        ("52", False, None),
    ],
)
def test_parse_answer(text, valid, value):
    toks = tk.encode_text(text)
    toks = toks + [tk.PAD] * (tk.MAX_NEW - len(toks))
    out = tk.parse_completion(toks)
    assert out == (valid, value)


def test_tokens_after_the_end_token_are_ignored_by_parsing():
    toks = tk.encode_text("7<eos>") + [tk.vocab["3"], tk.EOS]
    assert tk.parse_completion(toks) == (True, 7)


def test_gold_reward_is_exact_integer_correctness():
    assert tk.gold_reward(7, 45, *tk.parse_completion(tk.encode_text("52<eos>"))) == 1.0
    assert tk.gold_reward(7, 45, *tk.parse_completion(tk.encode_text("53<eos>"))) == 0.0
    assert tk.gold_reward(7, 45, False, None) == 0.0
    for a, b in [(0, 0), (99, 99), (50, 50), (9, 1)]:
        toks = tk.encode_answer(a + b)
        assert tk.gold_reward(a, b, *tk.parse_completion(toks)) == 1.0


def test_splits_are_disjoint_by_unordered_pair_and_deterministic():
    s1, s2 = tk.make_splits(seed=3), tk.make_splits(seed=3)
    assert s1 == s2
    sizes = {k: len(v) for k, v in s1.items()}
    assert sum(sizes.values()) == 100 * 100
    assert sizes["dev"] >= 900 and sizes["test"] >= 900
    key = {k: {frozenset((a, b)) for a, b in v} for k, v in s1.items()}
    assert not (key["train"] & key["dev"]) and not (key["train"] & key["test"])
    assert not (key["dev"] & key["test"])
    assert tk.make_splits(seed=4) != s1
