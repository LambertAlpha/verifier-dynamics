"""E005a calibration panel (research/07_e005a_measurement.md §3): exact oracle geometry.

Bases: the Stage 0b generator (one structure per cell, fresh seeds, max_target_redraws = 200).
Policy variants: theta_0, hot (0.5 u, 0.5 h), cold (2 u, 2 h), lowA (u - 6).
Verifier dose: EV_rho = (1 - rho)(a G + b) + rho EV_base. C is exactly linear in rho, and
rho = 0 (affine verifier) has C = 0 and alpha = a - 1 in every metric.
Nuisance dimensions: d_extra independent Bern(0.5) response features (zero gradient, Fisher block
0.25 I). Metrics: I (identity), D (damped diagonal Fisher), F (damped full Fisher); damping 0.1.
"""

from typing import Any

import numpy as np

from vdyn.e004 import panel0b as pb
from vdyn.e004 import toy

ROOT_SEED = 20261101
STREAMS = {"design": 0, "test": 1, "design_mc": 2, "test_mc": 3}
METRICS = ("I", "D", "F")
VARIANTS = ("theta_0", "theta_hot", "theta_cold", "theta_lowA")
DOSES = {"null": 0.0, "vsmall": 0.05, "small": 0.2, "medium": 0.6}
D_EXTRA = (0, 56)
MAX_TARGET_REDRAWS = 200
FEATURE_P = 0.5
DAMP = 0.1


def _cell_seeds(split: str) -> list[list[np.random.SeedSequence]]:
    kids = np.random.SeedSequence(ROOT_SEED).spawn(4)[STREAMS[split]].spawn(len(pb.CELLS))
    return [k.spawn(2) for k in kids]  # [structure, affine]


def bases(split: str, limit: int | None = None) -> list[toy.Structure]:
    return [b for b, _ in _bases_with_affine(split, limit)[0]]


Bases = list[tuple[toy.Structure, tuple[float, float]]]


def _bases_with_affine(split: str, limit: int | None = None) -> tuple[Bases, list[str]]:
    out, infeasible = [], []
    seeds = _cell_seeds(split)
    for i, (construction, axis) in enumerate(pb.CELLS[: limit or len(pb.CELLS)]):
        sid = f"E5{split[0].upper()}-{construction}-{axis[0]}"
        try:
            st = pb.build(sid, construction, axis, np.random.default_rng(seeds[i][0]), 0.2,
                          max_target_redraws=MAX_TARGET_REDRAWS)  # fmt: skip
        except pb.p0.Infeasible:
            infeasible.append(sid)
            continue
        r = np.random.default_rng(seeds[i][1])
        a = float(r.uniform(0.5, 0.95))
        b = float(r.uniform(0.0, 0.3 * (1 - a)))
        out.append((st, (a, b)))
    return out, infeasible


def variant_theta(theta0: np.ndarray, name: str) -> np.ndarray:
    th = np.asarray(theta0, dtype=float).copy()
    k = toy.K
    if name == "theta_hot":
        th[: k + 1] *= 0.5
    elif name == "theta_cold":
        th[: k + 1] *= 2.0
    elif name == "theta_lowA":
        th[:k] -= 6.0
    elif name != "theta_0":
        raise ValueError(name)
    return th


def _tables(st: toy.Structure, a: float, b: float, rho: float) -> toy.Tables:
    G = toy.gold_table(st)
    ev = (1 - rho) * (a * G + b) + rho * toy.verifier_table(st)
    return toy.Tables(st.w[None], st.p[None], st.lam[None], G[None], ev[None])


def exact(point: dict[str, Any]) -> dict[str, np.ndarray]:
    """Exact reward-level gradients and Fisher, padded with the nuisance dimensions."""
    st = toy.Structure.from_dict(point["base"])
    tb = _tables(st, point["a"], point["b"], point["rho"])
    ex = toy.exact(tb, np.asarray(point["theta"], dtype=float)[None])
    k = point["d_extra"]
    d = toy.D + k
    F = np.zeros((d, d))
    F[: toy.D, : toy.D] = ex["F"][0]
    F[toy.D :, toy.D :] = FEATURE_P * (1 - FEATURE_P) * np.eye(k)
    pad = np.zeros(k)
    return {"g_G": np.r_[ex["g_G"][0], pad], "g_V": np.r_[ex["g_V"][0], pad], "F": F}


def metric(F: np.ndarray, name: str) -> np.ndarray:
    d = F.shape[0]
    if name == "I":
        return np.eye(d)
    if name == "D":
        diag = np.diag(F)
        return np.diag(1 / (diag + DAMP * diag.mean()))
    if name == "F":
        return np.linalg.inv(F + DAMP * np.trace(F) / d * np.eye(d))
    raise ValueError(name)


def geometry(g_G: np.ndarray, g_V: np.ndarray, M: np.ndarray) -> dict[str, float]:
    g_e = g_V - g_G
    A2, P, Q = float(g_G @ M @ g_G), float(g_e @ M @ g_G), float(g_e @ M @ g_e)
    if A2 > 0:
        return {"A2": A2, "alpha": P / A2, "C2": max(Q - P**2 / A2, 0.0)}
    return {"A2": A2, "alpha": float("nan"), "C2": Q}


def make_point(base: toy.Structure, variant: str, rho: float, d_extra: int, a: float, b: float,
               dose: str, matched: str | None = None,
               pid: str | None = None) -> dict[str, Any]:  # fmt: skip
    p: dict[str, Any] = {
        "pid": pid or f"{base.sid}-{variant}-{dose}-d{d_extra}",
        "sid": base.sid, "mechanism": base.mechanism, "construction": base.construction,
        "axis": base.meta.get("axis_b"), "variant": variant,
        "theta": variant_theta(base.theta0, variant).tolist(), "a": a, "b": b, "rho": rho,
        "dose": dose, "d_extra": d_extra, "matched": matched, "base": base.to_dict(),
    }  # fmt: skip
    ex = exact(p)
    p["oracle"] = {m: geometry(ex["g_G"], ex["g_V"], metric(ex["F"], m)) for m in METRICS}
    return p


def c_base(base: toy.Structure, variant: str, a: float = 0.8, b: float = 0.0) -> float:
    """C (identity metric) at the natural dose rho = 1 (independent of the affine part)."""
    return float(np.sqrt(make_point(base, variant, 1.0, 0, a, b, "natural")["oracle"]["I"]["C2"]))


def dose_for(base: toy.Structure, variant: str, c_target: float, a: float, b: float) -> float:
    return c_target / c_base(base, variant, a, b)


def build_panel(split: str, limit: int | None = None, c_ref: float | None = None) -> dict[str, Any]:
    items, infeasible = _bases_with_affine(split, limit)
    if c_ref is None:
        c_ref = float(np.median([c_base(st, "theta_0", a, b) for st, (a, b) in items]))
    points, dropped = [], []
    for st, (a, b) in items:
        for variant in VARIANTS:
            cb = c_base(st, variant, a, b)
            for d_extra in D_EXTRA:
                for dose, frac in DOSES.items():
                    rho = 0.0 if frac == 0 else (frac * c_ref / cb if cb > 1e-12 else np.inf)
                    if rho > 1:
                        dropped.append(f"{st.sid}-{variant}-{dose}-d{d_extra}")
                        continue
                    points.append(make_point(st, variant, rho, d_extra, a, b, dose,
                                             matched=f"{dose}|{variant}|d{d_extra}"))  # fmt: skip
                points.append(make_point(st, variant, 1.0, d_extra, a, b, "natural"))
    return {"split": split, "root_seed": ROOT_SEED, "c_ref": c_ref, "points": points,
            "dropped_unattainable": dropped, "infeasible_bases": infeasible}  # fmt: skip
