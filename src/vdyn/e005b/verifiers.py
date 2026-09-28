"""E005b-0 verifier rules (research/10_e005b0_pilot.md §8; the flawed ones are PROPOSED, only
their initial error rates are measured in the pilot).

clean   : V = G
flip    : V = G XOR Bern(FLIP_RATE), independently per sample
deleted : V = 1 on a fixed subset of prompts (by unordered pair), V = G elsewhere
exploit : V = 1 if the answer is correct OR a valid number ending in digit 0
"""

import numpy as np

from vdyn.e005b import task as tk

FLIP_RATE = 0.2
KINDS = ("clean", "flip", "deleted", "exploit")


def deleted_prompts(frac: float, seed: int) -> set[tuple[int, int]]:
    keys = [(a, b) for a in range(100) for b in range(a, 100)]
    out: set[tuple[int, int]] = set()
    for i in np.random.default_rng(seed).permutation(len(keys)):
        if len(out) >= frac * 10000:
            break
        a, b = keys[i]
        out |= {(a, b), (b, a)}
    return out


def reward(kind: str, a: int, b: int, toks: list[int], rng: np.random.Generator,
           deleted: set[tuple[int, int]]) -> float:  # fmt: skip
    valid, value = tk.parse_completion(toks)
    g = tk.gold_reward(a, b, valid, value)
    if kind == "clean":
        return g
    if kind == "flip":
        return 1.0 - g if rng.random() < FLIP_RATE else g
    if kind == "deleted":
        return 1.0 if (a, b) in deleted else g
    if kind == "exploit":
        return 1.0 if g == 1.0 or (valid and value is not None and value % 10 == 0) else 0.0
    raise ValueError(kind)


def error_rates(V: np.ndarray, G: np.ndarray) -> dict[str, float]:
    neg, pos = G == 0, G == 1
    return {"fpr": float(V[neg].mean()) if neg.any() else float("nan"),
            "fnr": float(1 - V[pos].mean()) if pos.any() else float("nan"),
            "mean_v": float(V.mean()), "mean_g": float(G.mean())}  # fmt: skip


def reward_randfp(a: int, b: int, toks: list[int], rng: np.random.Generator, f0: float) -> float:
    """Random false positives (research/10_e005b0_pilot.md §14): every correct answer is accepted;
    every G = 0 response (valid-but-wrong or invalid) is accepted with a FRESH Bern(f0) coin."""
    if tk.gold_reward(a, b, *tk.parse_completion(toks)) == 1.0:
        return 1.0
    return 1.0 if rng.random() < f0 else 0.0
