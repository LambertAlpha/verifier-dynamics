"""E004a information levels and audit standard errors (registry E004a §6).

Every trajectory variable is summarized at horizon h by (value at h, change 0 -> h, slope over
[h/2, h] per 1% of T). Inputs are the values at the three checkpoints (0, h/2, h).
"""

from typing import Any

import numpy as np

AUDIT_N = 256
L2_VARS = ("J_G", "J_V", "FPR", "FNR")
L3_VARS = ("A", "alpha", "C")
PLUS_VARS = ("C_in", "C_out")
LEVELS = ("L0", "L1", "L2-G", "L2", "L2+", "L3", "L3+")


def summaries(x: np.ndarray, h_frac: float) -> tuple[float, float, float]:
    x0, xh2, xh = (float(v) for v in x)
    if h_frac <= 0:
        return xh, 0.0, 0.0
    width = 100.0 * h_frac / 2
    return xh, xh - x0, (xh - xh2) / width


def _summ(d: dict[str, np.ndarray], names: tuple[str, ...], idx: tuple[int, int, int],
          h_frac: float) -> list[float]:  # fmt: skip
    out: list[float] = []
    for name in names:
        out.extend(summaries(np.asarray(d[name])[list(idx)], h_frac))
    return out


def level_features(obs: dict[str, np.ndarray], geo: dict[str, np.ndarray], ctx: np.ndarray,
                   h_frac: float, idx: tuple[int, int, int]) -> dict[str, np.ndarray]:  # fmt: skip
    """Feature vectors of every level for one run at horizon h (idx = positions of 0, h/2, h)."""
    i0, _, ih = idx
    j0, f0 = float(obs["J_G"][i0]), float(obs["FPR"][i0])
    l0 = [j0, float(obs["J_V"][i0]), f0, float(obs["FNR"][i0]), (1 - j0) * f0]
    l1 = l0 + [float(geo[k][i0]) for k in L3_VARS]
    l2 = l0 + _summ(obs, L2_VARS, idx, h_frac)
    l2g = l0 + _summ(obs, ("J_G",), idx, h_frac)
    l3 = l2 + _summ(geo, L3_VARS, idx, h_frac)
    ctx = np.asarray(ctx)
    l2p = l2 + list(ctx[ih] - ctx[i0])
    l3p = l3 + _summ(geo, PLUS_VARS, idx, h_frac)
    feats = {"L0": l0, "L1": l1, "L2-G": l2g, "L2": l2, "L2+": l2p, "L3": l3, "L3+": l3p}
    return {k: np.nan_to_num(np.array(v, dtype=float)) for k, v in feats.items()}


def finite_levels(series: dict[str, np.ndarray], ctx: np.ndarray, h_frac: float,
                  idx: tuple[int, int, int]) -> dict[str, np.ndarray]:  # fmt: skip
    """Stage 1 (execution note §2): the registered levels from finite-sample estimates, with
    missing values kept (NaN). Geometry = update level (A_u, alpha_u, C_u); the sensitivity arms
    L1r / L3r add the reward-level alpha."""
    i0, _, ih = idx
    l0 = [float(series[k][i0]) for k in ("J_G", "J_V", "FPR", "FNR", "FPM")]
    geo = {"A": series["A_u"], "alpha": series["alpha_u"], "C": series["C_u"],
           "C_in": series["C_in"], "C_out": series["C_out"]}  # fmt: skip
    l1 = l0 + [float(geo[k][i0]) for k in L3_VARS]
    l2 = l0 + _summ(series, L2_VARS, idx, h_frac)
    l3 = l2 + _summ(geo, L3_VARS, idx, h_frac)
    ctx = np.asarray(ctx)
    feats = {
        "L0": l0,
        "L1": l1,
        "L1r": l1 + [float(series["alpha_r"][i0])],
        "L2-G": l0 + _summ(series, ("J_G",), idx, h_frac),
        "L2": l2,
        "L2+": l2 + list(ctx[ih] - ctx[i0]),
        "L3": l3,
        "L3r": l3 + _summ(series, ("alpha_r",), idx, h_frac),
        "L3+": l3 + _summ(geo, PLUS_VARS, idx, h_frac),
    }
    return {k: np.array(v, dtype=float) for k, v in feats.items()}


def audit_se(v: dict[str, float], n: int = AUDIT_N) -> dict[str, float]:
    jg = v["J_G"]
    out = {
        "J_G": np.sqrt(jg * (1 - jg) / n),
        "J_V": np.sqrt(v["J_V"] * (1 - v["J_V"]) / n),
        "FPR": np.sqrt(v["FPR"] * (1 - v["FPR"]) / max(n * (1 - jg), 1.0)),
        "FNR": np.sqrt(v["FNR"] * (1 - v["FNR"]) / max(n * jg, 1.0)),
    }
    if "FPM" in v:
        out["FPM"] = np.sqrt(v["FPM"] * (1 - v["FPM"]) / n)
    return {k: float(x) for k, x in out.items()}


def l0_se(v0: dict[str, float], n: int = AUDIT_N) -> np.ndarray:
    fpm = (1 - v0["J_G"]) * v0["FPR"]
    se = audit_se({**v0, "FPM": fpm}, n)
    return np.array([se["J_G"], se["J_V"], se["FPR"], se["FNR"], se["FPM"]])


def l2_feature_se(vals: dict[str, np.ndarray], h_frac: float, n: int = AUDIT_N) -> np.ndarray:
    """SE of the 12 L2 summaries (value, change, slope per variable) at checkpoints (0, h/2, h)."""
    ses = [audit_se({k: float(vals[k][i]) for k in L2_VARS}, n) for i in range(3)]
    width = 100.0 * h_frac / 2 if h_frac > 0 else 1.0
    out: list[float] = []
    for k in L2_VARS:
        s0, s2, sh = ses[0][k], ses[1][k], ses[2][k]
        out.extend([sh, float(np.hypot(sh, s0)), float(np.hypot(sh, s2)) / width])
    return np.array(out)


def names(level: str) -> list[str]:
    base = ["J_G0", "J_V0", "FPR0", "FNR0", "FPM0"]
    summ = lambda vs: [f"{v}_{s}" for v in vs for s in ("h", "d", "slope")]  # noqa: E731
    table: dict[str, Any] = {
        "L0": base,
        "L1": base + ["A0", "alpha0", "C0"],
        "L1r": base + ["A0", "alpha0", "C0", "alpha_r0"],
        "L2-G": base + summ(("J_G",)),
        "L2": base + summ(L2_VARS),
        "L2+": base + summ(L2_VARS) + [f"dJ_G_ctx{x}" for x in range(4)],
        "L3": base + summ(L2_VARS) + summ(L3_VARS),
        "L3r": base + summ(L2_VARS) + summ(L3_VARS) + summ(("alpha_r",)),
        "L3+": base + summ(L2_VARS) + summ(L3_VARS) + summ(PLUS_VARS),
    }
    return table[level]
