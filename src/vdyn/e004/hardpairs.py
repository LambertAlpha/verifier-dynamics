"""E004a hard-pair acceptance (registry E004a §8.7).

Correction A: every L2 feature (value, change, slope of J_G, J_V, FPR, FNR at every registered
horizon <= h*) differs by at most 0.5 audit-SE (SE evaluated at the mean of the two members).
Correction B adds: L0 within 0.5 audit-SE, L1 within the registered tolerances, and an L3
divergence before the failing member's visible onset.
"""

from typing import Any

import numpy as np

from vdyn.e004 import features as fe

TOL = 0.5
L1_REL_A, L1_ALPHA, L1_REL_C = 0.1, 0.1, 0.1
DIV_ALPHA, DIV_REL = 0.3, 0.5


def correction_a(a: dict[str, np.ndarray], b: dict[str, np.ndarray],
                 horizons: dict[float, tuple[int, int, int]]) -> dict[str, Any]:  # fmt: skip
    worst, worst_name = 0.0, ""
    labels = [f"{v}_{s}" for v in fe.L2_VARS for s in ("h", "d", "slope")]
    for h, idx in horizons.items():
        fa = np.array(sum((list(fe.summaries(a[k][list(idx)], h)) for k in fe.L2_VARS), []))
        fb = np.array(sum((list(fe.summaries(b[k][list(idx)], h)) for k in fe.L2_VARS), []))
        mean = {k: 0.5 * (a[k][list(idx)] + b[k][list(idx)]) for k in fe.L2_VARS}
        se = fe.l2_feature_se(mean, h)
        ratio = np.abs(fa - fb) / np.maximum(se, 1e-12)
        if h <= 0:
            ratio = ratio[0::3]  # only the values are defined at h = 0
            names = labels[0::3]
        else:
            names = labels
        i = int(np.argmax(ratio))
        if ratio[i] > worst:
            worst, worst_name = float(ratio[i]), f"{names[i]}@{h:g}"
    return {"accept": worst <= TOL, "max_ratio": worst, "worst": worst_name}


def l0_match(v0a: dict[str, float], v0b: dict[str, float]) -> dict[str, Any]:
    def vec(v: dict[str, float]) -> np.ndarray:
        return np.array([v["J_G"], v["J_V"], v["FPR"], v["FNR"], (1 - v["J_G"]) * v["FPR"]])

    mean = {k: 0.5 * (v0a[k] + v0b[k]) for k in ("J_G", "J_V", "FPR", "FNR")}
    ratio = np.abs(vec(v0a) - vec(v0b)) / np.maximum(fe.l0_se(mean), 1e-12)
    return {"accept": bool(ratio.max() <= TOL), "max_ratio": float(ratio.max())}


def l1_match(ga: dict[str, float], gb: dict[str, float]) -> bool:
    mA, mC = 0.5 * (ga["A"] + gb["A"]), 0.5 * (ga["C"] + gb["C"])
    return bool(abs(ga["A"] - gb["A"]) <= L1_REL_A * mA
                and abs(ga["alpha"] - gb["alpha"]) <= L1_ALPHA
                and abs(ga["C"] - gb["C"]) <= L1_REL_C * max(mC, 0.01))  # fmt: skip


def l3_divergence(ga: dict[str, np.ndarray], gb: dict[str, np.ndarray], times: np.ndarray,
                  t_limit: float) -> float | None:  # fmt: skip
    """Earliest checkpoint time < t_limit with |d alpha| >= 0.3 or relative |dA|, |dC| >= 0.5."""
    for i, t in enumerate(times):
        if t >= t_limit:
            break
        da = abs(ga["alpha"][i] - gb["alpha"][i])
        rA = abs(ga["A"][i] - gb["A"][i]) / max(0.5 * (ga["A"][i] + gb["A"][i]), 1e-12)
        rC = abs(ga["C"][i] - gb["C"][i]) / max(0.5 * (ga["C"][i] + gb["C"][i]), 1e-12)
        if (np.isfinite(da) and da >= DIV_ALPHA) or rA >= DIV_REL or rC >= DIV_REL:
            return float(t)
    return None
