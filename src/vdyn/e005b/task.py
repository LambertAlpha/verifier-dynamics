"""E005b-0 task (research/10_e005b0_pilot.md §1): two-digit addition with character tokens.

Prompt: <bos> a a + b b =   (operands zero-padded to two digits; 7 tokens, identical length)
Answer: the decimal sum without leading zeros, then <eos>; at most MAX_NEW = 4 generated tokens.
A completion is VALID iff an <eos> occurs within the budget and the text before it is 1-3 digits
with no leading zero (except "0"). Gold reward = 1 iff valid and equal to a + b; invalid = 0.
Splits are by UNORDERED operand pair, so a+b and b+a never straddle train / dev / test.
"""

import numpy as np

SPECIALS = ["<pad>", "<bos>", "<eos>"]
CHARS = list("0123456789+=")
ITOS = SPECIALS + CHARS
vocab = {s: i for i, s in enumerate(ITOS)}
PAD, BOS, EOS = vocab["<pad>"], vocab["<bos>"], vocab["<eos>"]
VOCAB_SIZE = len(ITOS)
PROMPT_LEN = 7
MAX_NEW = 4
DEV_MIN, TEST_MIN = 1000, 1000


def encode_text(s: str) -> list[int]:
    out, i = [], 0
    while i < len(s):
        if s[i] == "<":
            j = s.index(">", i)
            out.append(vocab[s[i : j + 1]])
            i = j + 1
        else:
            out.append(vocab[s[i]])
            i += 1
    return out


def decode(ids: list[int]) -> str:
    return "".join(ITOS[i] for i in ids)


def encode_prompt(a: int, b: int) -> list[int]:
    return [BOS] + encode_text(f"{a:02d}+{b:02d}=")


def encode_answer(v: int) -> list[int]:
    return encode_text(str(v)) + [EOS]


def parse_completion(toks: list[int]) -> tuple[bool, int | None]:
    head = list(toks[:MAX_NEW])
    if EOS not in head:
        return False, None
    body = head[: head.index(EOS)]
    text = "".join(ITOS[t] for t in body)
    if not (1 <= len(text) <= 3) or not text.isdigit() or (len(text) > 1 and text[0] == "0"):
        return False, None
    return True, int(text)


def gold_reward(a: int, b: int, valid: bool, value: int | None) -> float:
    return 1.0 if valid and value == a + b else 0.0


def make_splits(seed: int) -> dict[str, list[tuple[int, int]]]:
    keys = [(a, b) for a in range(100) for b in range(a, 100)]
    order = np.random.default_rng(seed).permutation(len(keys))
    out: dict[str, list[tuple[int, int]]] = {"dev": [], "test": [], "train": []}
    for i in order:
        a, b = keys[i]
        pairs = [(a, b)] if a == b else [(a, b), (b, a)]
        if len(out["dev"]) < DEV_MIN:
            out["dev"] += pairs
        elif len(out["test"]) < TEST_MIN:
            out["test"] += pairs
        else:
            out["train"] += pairs
    return {k: sorted(v) for k, v in out.items()}
