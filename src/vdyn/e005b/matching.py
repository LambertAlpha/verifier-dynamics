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
    """Resample prompts with replacement; rates are recomputed from the resampled prompts' pooled
    counts (identical to recomputing them over the concatenated responses)."""
    _, inv = np.unique(pid, return_inverse=True)
    k = int(inv.max()) + 1
    neg, pos = G == 0, G == 1
    cnt = {name: np.bincount(inv, weights=w.astype(float), minlength=k)
           for name, w in (("n", np.ones_like(G)), ("neg", neg), ("pos", pos), ("g", G),
                           ("fp", (V == 1) & neg), ("fn", (V == 0) & pos))}  # fmt: skip
    pick = rng.integers(0, k, size=(resamples, k))
    tot = {name: c[pick].sum(1) for name, c in cnt.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = {"fpr": tot["fp"] / tot["neg"], "fnr": tot["fn"] / tot["pos"],
                 "acc": tot["g"] / tot["n"], "fp_mass": tot["fp"] / tot["n"]}  # fmt: skip
    out = {}
    for name, a in draws.items():
        a = a[np.isfinite(a)]
        out[f"{name}_ci"] = ([float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]
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


def verdict(f0: float, fpr_v3: float, fnr: dict[str, float], tol: float) -> dict[str, Any]:
    """The frozen matching criterion: |FPR_V3(verification) - f0| <= tol, and FNR exactly 0 for
    every arm that claims it by construction. Anything undefined fails (fail closed)."""
    reasons = []
    diff = abs(fpr_v3 - f0)
    if not np.isfinite(diff) or diff > tol:
        reasons.append(f"|FPR_V3 - f0| = {diff:.4f} exceeds {tol} or is undefined")
    for arm, x in sorted(fnr.items()):
        if not (np.isfinite(x) and x == 0.0):
            reasons.append(f"FNR of {arm} is {x}, not exactly 0")
    return {"pass": not reasons, "f0": f0, "fpr_v3": fpr_v3, "abs_diff": float(diff),
            "tolerance": tol, "reasons": reasons}  # fmt: skip
