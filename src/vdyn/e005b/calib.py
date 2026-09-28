"""E005b-0 calibration round (research/10_e005b0_pilot.md §10): per-category evaluation with k
samples per item, per-category batch statistics, and the frozen coverage-aware base selection."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn

from vdyn.e005b import model as mdl
from vdyn.e005b import rl
from vdyn.e005b import task as tk


@torch.no_grad()
def evaluate_categories(net: nn.Module, pairs: list[tuple[int, int]], samples: int,
                        seed: int) -> dict[str, Any]:  # fmt: skip
    gen = torch.Generator().manual_seed(seed)
    toks = mdl.sample(net, rl.prompt_tensor(pairs, samples), tk.MAX_NEW, 1.0, gen)["tokens"]
    rep = [p for p in pairs for _ in range(samples)]
    parsed = [tk.parse_completion(t) for t in toks.tolist()]
    gold = np.array([tk.gold_reward(a, b, *pv) for (a, b), pv in zip(rep, parsed, strict=True)])
    valid = np.array([float(v) for v, _ in parsed])
    cats = np.array([tk.category(a, b) for a, b in rep])
    gtok, _ = mdl.greedy(net, rl.prompt_tensor(pairs), tk.MAX_NEW)
    gparsed = [tk.parse_completion(t) for t in gtok.tolist()]
    ggold = np.array([tk.gold_reward(a, b, *pv) for (a, b), pv in zip(pairs, gparsed, strict=True)])
    gcats = np.array([tk.category(a, b) for a, b in pairs])
    by_cat: dict[str, Any] = {}
    for c in tk.CATEGORIES:
        m, gm = cats == c, gcats == c
        by_cat[c] = {"n_items": int(gm.sum()),
                     "sampled": float(gold[m].mean()) if m.any() else float("nan"),
                     "greedy": float(ggold[gm].mean()) if gm.any() else float("nan"),
                     "valid": float(valid[m].mean()) if m.any() else float("nan")}  # fmt: skip
    return {"sampled": float(gold.mean()), "valid": float(valid.mean()),
            "greedy": float(ggold.mean()), "n_items": len(pairs), "samples": samples,
            "responses": len(rep) + len(pairs), "gold_calls": len(rep) + len(pairs),
            "by_cat": by_cat}  # fmt: skip


def batch_category_stats(pairs: list[tuple[int, int]], gold: torch.Tensor) -> dict[str, Any]:
    """pairs: the P prompts of a step; gold: (P, G) gold rewards."""
    g = gold.numpy()
    cats = np.array([tk.category(a, b) for a, b in pairs])
    out: dict[str, Any] = {}
    for c in tk.CATEGORIES:
        m = cats == c
        out[c] = {
            "groups": int(m.sum()),
            "gold": float(g[m].mean()) if m.any() else float("nan"),
            "mixed": float(np.mean(g[m].std(1) > 0)) if m.any() else float("nan"),
        }
    return out  # fmt: skip


def passes(ev: dict[str, Any], rule: dict[str, float]) -> bool:
    cats_ok = all(ev["by_cat"][c]["sampled"] >= rule["min_category"] for c in tk.CATEGORIES)
    return bool(cats_ok and ev["sampled"] <= rule["max_aggregate"]
                and ev["valid"] >= rule["min_valid"])  # fmt: skip


def select_coverage(sel: dict[int, dict[str, Any]], confirm: Callable[[int], dict[str, Any]],
                    rule: dict[str, float]) -> dict[str, Any] | None:  # fmt: skip
    """First checkpoint (step order) passing the rule under the selection seed AND, re-evaluated
    with the independent seed, passing the same rule. None if no checkpoint qualifies."""
    tried: list[int] = []
    confirmations: dict[int, dict[str, Any]] = {}
    for step in sorted(sel):
        if not passes(sel[step], rule):
            continue
        tried.append(step)
        confirmations[step] = confirm(step)
        if passes(confirmations[step], rule):
            return {
                "step": step,
                "tried": tried,
                "selection_eval": sel[step],
                "confirmation_eval": confirmations[step],
                "all_confirmations": confirmations,
            }
    return None
