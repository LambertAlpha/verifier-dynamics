"""E004a outcome labels (registry E004a §7), computed on exact J_G curves along a run.

Curves are sampled on a grid of fractions of T (default: 101 points, every 1% of T). The clean
reference is the matched clean curve (4-seed mean for sampled Adam; the exact clean flow for NG).
"""

from typing import Any

import numpy as np

SUCCESS_DN, DECLINE_DROP, ONSET_GAP = 0.1, 0.05, 0.1


def label(run: np.ndarray, clean: np.ndarray, grid: np.ndarray | None = None) -> dict[str, Any]:
    run, clean = np.asarray(run, dtype=float), np.asarray(clean, dtype=float)
    grid = np.linspace(0, 1, len(run)) if grid is None else np.asarray(grid)
    j0, jT, cT = run[0], run[-1], clean[-1]
    dn = (cT - jT) / (cT - j0)
    drop = float(run.max() - jT)
    half = run[int(np.argmin(np.abs(grid - 0.5)))]
    if drop >= DECLINE_DROP and dn > SUCCESS_DN:
        cat = "DECLINE"
    elif dn <= SUCCESS_DN:
        cat = "SUCCESS"
    elif 2 * (jT - half) < cT - jT:
        cat = "STALL"
    else:
        cat = "SLOW"
    denom = clean - j0
    gap = np.where(denom > 1e-12, (clean - run) / np.where(denom > 1e-12, denom, 1.0), 0.0)
    hit = np.flatnonzero(gap > ONSET_GAP)
    return {
        "Dn": float(dn),
        "category": cat,
        "failure": cat in ("STALL", "DECLINE"),
        "t_on": float(grid[hit[0]]) if len(hit) else None,
        "peak_drop": drop,
    }
