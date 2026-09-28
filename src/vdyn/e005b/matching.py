"""E005b-0 matched-initial-error audit statistics (research/10_e005b0_pilot.md §14).

Per sampled response: prompt id, category, verifier reward V and gold G. Point estimates of FPR
`P(V=1|G=0)`, FNR `P(V=0|G=1)`, accuracy and FP mass `P(V=1, G=0)` with explicit denominators, and
95% percentile intervals from a PROMPT-CLUSTER bootstrap (responses to the same prompt are not
independent). Undefined rates (zero denominator) are NaN.
"""

from typing import Any

import numpy as np


def _rates(V: np.ndarray, G: np.ndarray) -> dict[str, float]:
    neg, pos = G == 0, G == 1
    return {"n": int(len(G)), "n_neg": int(neg.sum()), "n_pos": int(pos.sum()),
            "fpr": float(V[neg].mean()) if neg.any() else float("nan"),
            "fnr": float(1 - V[pos].mean()) if pos.any() else float("nan"),
            "acc": float(G.mean()), "fp_mass": float(np.mean((V == 1) & neg))}  # fmt: skip


def _boot(pid: np.ndarray, V: np.ndarray, G: np.ndarray, rng: np.random.Generator,
          resamples: int) -> dict[str, list[float]]:  # fmt: skip
    uniq = np.unique(pid)
    idx = {p: np.flatnonzero(pid == p) for p in uniq}
    draws: dict[str, list[float]] = {k: [] for k in ("fpr", "fnr", "acc", "fp_mass")}
    for _ in range(resamples):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[p] for p in pick])
        r = _rates(V[rows], G[rows])
        for k in draws:
            draws[k].append(r[k])
    out = {}
    for k, v in draws.items():
        a = np.array(v)
        a = a[np.isfinite(a)]
        out[f"{k}_ci"] = ([float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]
                          if len(a) else [float("nan"), float("nan")])  # fmt: skip
    return out


def audit_rates(pid: np.ndarray, cats: np.ndarray, V: np.ndarray, G: np.ndarray,
                rng: np.random.Generator, resamples: int) -> dict[str, Any]:  # fmt: skip
    V, G = np.asarray(V, dtype=float), np.asarray(G, dtype=float)
    out: dict[str, Any] = {"overall": _rates(V, G) | _boot(pid, V, G, rng, resamples), "by_cat": {}}
    for c in sorted(set(cats.tolist())):
        m = cats == c
        out["by_cat"][c] = _rates(V[m], G[m]) | _boot(pid[m], V[m], G[m], rng, resamples)
    return out
