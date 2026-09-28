"""E005b-0 exploratory verifier matrix diagnostics (research/10_e005b0_pilot.md §12).

All functions are pure summaries of already-generated data: they use no random numbers and do not
touch model parameters, so they cannot change training. Conditional rates whose denominator is 0
are NaN, and the denominators are returned.
"""

from typing import Any

import numpy as np
from torch import nn

from vdyn.e005b import calib as cal
from vdyn.e005b import task as tk


def verifier_rng(base_seed: int, seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([base_seed, seed]))


def confusion(V: np.ndarray, G: np.ndarray) -> dict[str, Any]:
    v, g = np.asarray(V, dtype=float).ravel(), np.asarray(G, dtype=float).ravel()
    neg, pos = g == 0, g == 1
    return {"n": int(len(g)), "n_neg": int(neg.sum()), "n_pos": int(pos.sum()),
            "fpr": float(v[neg].mean()) if neg.any() else float("nan"),
            "fnr": float(1 - v[pos].mean()) if pos.any() else float("nan"),
            "fp_mass": float(np.mean((v == 1) & neg)), "mean_v": float(v.mean()),
            "mean_g": float(g.mean())}  # fmt: skip


def suffix_stats(pairs: list[tuple[int, int]], toks: list[list[int]]) -> dict[str, Any]:
    """Per sample: valid answer ending in 0; the same but wrong (the V3 false positive); "0"."""
    ends0 = fp = zero = 0
    for (a, b), t in zip(pairs, toks, strict=True):
        valid, val = tk.parse_completion(t)
        if valid and val is not None and val % 10 == 0:
            ends0 += 1
            fp += int(val != a + b)
            zero += int(val == 0)
    n = len(pairs)
    return {"n": n, "ends0": ends0 / n, "fp_ends0": fp / n, "zero": zero / n}


def subset_stats(pairs: list[tuple[int, int]], G: np.ndarray,
                 subset: set[tuple[int, int]]) -> dict[str, Any]:  # fmt: skip
    """Group-level gold accuracy for prompts inside vs outside a fixed rule-defined subset."""
    g = np.asarray(G, dtype=float)
    inside = np.array([p in subset for p in pairs])
    out: dict[str, Any] = {}
    for name, m in (("in", inside), ("out", ~inside)):
        out[name] = {
            "prompts": int(m.sum()),
            "gold": float(g[m].mean()) if m.any() else float("nan"),
        }
    return out


def variability(V: np.ndarray, G: np.ndarray) -> dict[str, float]:
    v, g = np.asarray(V, dtype=float), np.asarray(G, dtype=float)
    return {"var_v": float(np.var(v)), "mixed_v": float(np.mean(v.std(1) > 0)),
            "mixed_g": float(np.mean(g.std(1) > 0))}  # fmt: skip


def evaluate_matrix(net: nn.Module, pairs: list[tuple[int, int]], samples: int, seed: int,
                    deleted: set[tuple[int, int]]) -> dict[str, Any]:  # fmt: skip
    """The calibration evaluation (identical numbers) plus mechanism measurements on the same
    samples: V3 suffix statistics and accuracy on dev items inside / outside the fixed-rule
    subset (dev items are never trained on; they only share the rule-defined membership)."""
    es = cal.eval_samples(net, pairs, samples, seed)
    ev = cal.summarize_samples(pairs, es, samples)
    rep = es["rep"]
    gold = np.array([tk.gold_reward(a, b, *tk.parse_completion(t))
                     for (a, b), t in zip(rep, es["tokens"], strict=True)])  # fmt: skip
    ggold = np.array([tk.gold_reward(a, b, *tk.parse_completion(t))
                      for (a, b), t in zip(pairs, es["greedy_tokens"], strict=True)])  # fmt: skip
    in_rep = np.array([p in deleted for p in rep])
    in_items = np.array([p in deleted for p in pairs])
    sub: dict[str, Any] = {}
    for name, m, gm in (("in", in_rep, in_items), ("out", ~in_rep, ~in_items)):
        sub[name] = {
            "items": int(gm.sum()),
            "sampled": float(gold[m].mean()) if m.any() else float("nan"),
            "greedy": float(ggold[gm].mean()) if gm.any() else float("nan"),
        }
    return {
        **ev,
        "suffix": suffix_stats(rep, es["tokens"]),
        "suffix_greedy": suffix_stats(pairs, es["greedy_tokens"]),
        "subset": sub,
    }
