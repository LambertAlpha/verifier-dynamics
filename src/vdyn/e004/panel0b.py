"""E004a Stage 0b panel (Amendment 3 §3, §4, §8): 18 constructions x Axis B, fresh seeds.

Axis A (structure) gives the mechanism label; Axis B (preference relation) is computed from the
actual verifier (toy.axis_b). YA and YB are crossed with {ALIGNED, INVERTED}; R, X, B are aligned
only; D is inverted by definition. Intended-inverted structures must be actually inverted (redrawn
otherwise); intended-aligned structures that background noise inverts are kept and flagged.
Calibration and background (top-up) coins follow Stage 0 (panel.py helpers).
"""

from typing import Any

import numpy as np
from scipy.optimize import brentq

from vdyn.e004 import panel as p0
from vdyn.e004 import toy

ROOT_SEED = 20261001
CONSTRUCTIONS = ("R1", "R2", "R3", "X1", "X2", "X3", "YA1", "YA2", "YA3", "YB1", "YB2", "YB3",
                 "B1", "B2", "B3", "D1", "D2", "D3")  # fmt: skip
MECHANISMS = ("R", "X", "YA", "YB", "B", "D")
CELLS = [(c, "ALIGNED") for c in CONSTRUCTIONS if not c.startswith("D")]
CELLS += [(c, "INVERTED") for c in CONSTRUCTIONS if c[:2] in ("YA", "YB") or c.startswith("D")]
CELLS = sorted(CELLS, key=lambda cell: (CONSTRUCTIONS.index(cell[0]), cell[1]))
N_PER_CELL = 32
MAX_TRIES = 50
FEATURES_OF = {"single": [0], "or2": [0, 1], "and2": [0, 1], "and3": [0, 1, 2],
               "thr23": [0, 1, 2]}  # fmt: skip


def mechanism_of(construction: str) -> str:
    return construction.rstrip("0123456789")


def _base(sid: str, construction: str, c: dict[str, Any]) -> toy.Structure:
    z = np.zeros(toy.K)
    theta = np.concatenate([c["delta"], [c["h0"]], c["phi0"]])
    return toy.Structure(
        sid=sid, construction=construction, mechanism=mechanism_of(construction), w=c["w"],
        p=c["p"], theta0=theta, lam=c["lam"], event="none", trig=z, fp=z, fn=z, rho=z, beta=z,
        deleted=np.zeros(toy.K, bool), v0=1.0, meta={"delta": c["delta"], "phi0": c["phi0"]},
    )  # fmt: skip


def _solve_shift(st: toy.Structure, target_se: float) -> toy.Structure:
    idx = FEATURES_OF[st.event]

    def f(c: float) -> float:
        return p0._event_prob(p0._set_theta(st, shift=(idx, c))) - target_se

    return p0._set_theta(st, shift=(idx, float(brentq(f, -40, 40, xtol=1e-13))))


def _subset(rng: np.random.Generator, lo: int, hi: int) -> np.ndarray:
    m = np.zeros(toy.K, bool)
    m[rng.choice(toy.K, int(rng.integers(lo, hi + 1)), replace=False)] = True
    return m


def _solve_h_mu(st: toy.Structure, share: float, jg: float) -> toy.Structure:
    """Joint (h0, mu): hack-driven channel FPR = share and J_G(0) = jg."""
    for _ in range(60):
        prev = st.theta0.copy()

        def f(h: float, st: toy.Structure = st) -> float:
            return p0._static(p0._set_theta(st, h=h))[1] - share

        if f(-25) > 0 or f(8) < 0:
            raise p0.Infeasible("hack share cannot reach the FPR share")
        st = p0._set_theta(st, h=float(brentq(f, -25, 8, xtol=1e-13)))
        st = p0._solve_mu(st, jg)
        if np.max(np.abs(st.theta0 - prev)) < 1e-11:
            return st
    raise p0.Infeasible("mu/h0 did not converge")


def _channel(
    construction: str,
    axis: str,
    st: toy.Structure,
    c: dict[str, Any],
    t: dict[str, float],
    rng: np.random.Generator,
    yb_cap: float,
) -> toy.Structure:
    K, share = toy.K, c["omega"] * t["FPR"]
    meta = dict(st.meta)
    if construction == "R1":
        st = st.with_(fp=np.full(K, t["FPR"]), fn=np.full(K, t["FPR"]))
    elif construction == "R2":
        return p0._channel("R2", st, c, t, rng)
    elif construction == "R3":
        base = rng.uniform(0.02, 0.3, K)

        def fpr_of(s: float) -> float:
            e = np.minimum(s * base, 0.45)
            return p0._static(st.with_(fp=e, fn=e))[1]

        if fpr_of(0.45 / base.min()) < t["FPR"]:
            raise p0.Infeasible("R3 coins cannot reach the FPR target")
        s = float(brentq(lambda s: fpr_of(s) - t["FPR"], 0, 0.45 / base.min(), xtol=1e-14))
        e = np.minimum(s * base, 0.45)
        st = st.with_(fp=e, fn=e)
    elif construction in ("X1", "X2", "X3"):
        v0 = {"X1": 1.0, "X2": 0.0}.get(construction, float(rng.uniform(0.3, 0.7)))
        st = st.with_(deleted=_subset(rng, 1, 2), v0=v0)
    elif construction[:2] in ("YA", "YB"):
        event = {"YA1": "and2" if rng.random() < 0.5 else "and3", "YA2": "and2", "YA3": "thr23",
                 "YB1": "single", "YB2": "single", "YB3": "or2"}[construction]  # fmt: skip
        on = np.ones(K, bool)
        if construction == "YA2":
            on = _subset(rng, 1, 3)
        if construction == "YB2":
            on = np.zeros(K, bool)
            on[np.argsort(np.asarray(c["delta"]))[: int(rng.integers(1, 3))]] = True
        if construction.startswith("YA"):
            se = min(share, p0._loguniform(rng, 0.001, 0.05))
        else:
            se = min(share, yb_cap)
        if axis == "ALIGNED":
            trig = np.where(on, rng.uniform(0.5, 0.9), 0.0)
            fn = np.zeros(K)
        else:
            trig = np.where(on, 1.0, 0.0)
            fn = np.where(on, rng.uniform(0.1, 0.4, K), 0.0)
        st = _solve_shift(st.with_(event=event, trig=trig, fn=fn), se)
        meta["S_E0"] = se
    elif construction in ("B1", "B2", "B3"):
        credit = _subset(rng, 1, 3) if construction == "B3" else np.ones(K, bool)
        st = st.with_(credit_event="z3" if construction == "B2" else "none")

        def with_beta(b: float) -> toy.Structure:
            return st.with_(beta=np.where(credit, b, 0.0))

        if p0._static(with_beta(1.0))[1] <= share:
            st, meta["capped"] = with_beta(1.0), True
        else:
            b = float(brentq(lambda b: p0._static(with_beta(b))[1] - share, 0, 1, xtol=1e-14))
            st = with_beta(b)
    elif construction in ("D1", "D2", "D3"):
        rho = float(rng.uniform(0.3, 0.9))
        low = st.p < rho
        if not low.any():
            raise p0.Infeasible("no prompt with p < rho")
        if construction == "D2":
            sub = np.zeros(K, bool)
            n_pick = min(int(low.sum()), int(rng.integers(1, 3)))
            pick = rng.choice(np.flatnonzero(low), size=n_pick, replace=False)
            sub[pick] = True
            rho_x = np.where(sub, rho, 0.0)
        else:
            rho_x = np.full(K, rho)
        st = st.with_(rho=rho_x, hack_event="z1" if construction == "D3" else "none")
        st = _solve_h_mu(st, share, t["J_G"])
    else:
        raise ValueError(construction)
    meta["fp_channel"], meta["fn_channel"] = st.fp.tolist(), st.fn.tolist()
    return st.with_(meta=meta)


def build(sid: str, construction: str, axis: str, rng: np.random.Generator,
          yb_cap: float) -> toy.Structure:  # fmt: skip
    rejections = {"channel": 0, "targets": 0, "axis": 0}
    c = p0._common(rng)
    while True:
        t = p0._targets(rng)
        if t["J_G"] >= 0.9 * float(c["w"] @ c["p"]):
            rejections["targets"] += 1
            continue
        for _ in range(MAX_TRIES):
            st = _base(sid, construction, c)
            try:
                st = p0._solve_mu(st, t["J_G"])
                st = _channel(construction, axis, st, c, t, rng, yb_cap)
                _, fpr_c, fnr_c = p0._static(st)
                if fpr_c > t["FPR"] + 1e-9 or fnr_c > t["FNR"] + 1e-9:
                    raise p0.Infeasible("channel exceeds a target")
            except (p0.Infeasible, ValueError):
                rejections["channel"] += 1
                continue
            st = p0._topup(st, t)
            actual = toy.axis_b(st)
            if axis == "INVERTED" and actual != "INVERTED":
                rejections["axis"] += 1
                continue
            meta = dict(st.meta)
            gaps = np.nan_to_num(toy.preference_gaps(st), nan=-9.0).tolist()
            meta.update(
                targets=t,
                omega=c["omega"],
                rejections=rejections,
                intended_axis=axis,
                axis_b=actual,
                gaps=gaps,
                noise_inverted=bool(axis == "ALIGNED" and actual == "INVERTED"),
                delta=np.asarray(c["delta"]).tolist(),
                phi0=np.asarray(c["phi0"]).tolist(),
            )
            return st.with_(meta=meta)
        rejections["targets"] += 1


def _cells_panel(stream: int, cells: list[tuple[str, str]], n_per: int, yb_cap: float,
                 prefix: str = "") -> list[toy.Structure]:  # fmt: skip
    ss = np.random.SeedSequence(ROOT_SEED).spawn(5)[stream]
    children = ss.spawn(len(cells))
    out = []
    for i, (construction, axis) in enumerate(cells):
        for j, s in enumerate(children[i].spawn(n_per)):
            sid = f"{prefix}{construction}-{axis[0]}-{j:02d}"
            out.append(build(sid, construction, axis, np.random.default_rng(s), yb_cap))
    return out


def design_panel(n_per: int = N_PER_CELL, yb_cap: float = 0.4) -> list[toy.Structure]:
    return _cells_panel(0, CELLS, n_per, yb_cap)


def calibration_panel(yb_cap: float, n_per: int = 24) -> list[toy.Structure]:
    """YB cells only, calibration stream (never part of the panel; Amendment 3 §4)."""
    cells = [cell for cell in CELLS if cell[0].startswith("YB")]
    return _cells_panel(4, cells, n_per, yb_cap, prefix="cal-")


def uncoupled_twin(st: toy.Structure) -> toy.Structure:
    return p0.uncoupled_twin(st)


def canonical_twin(st: toy.Structure) -> toy.Structure:
    return p0.canonical_twin(st)
