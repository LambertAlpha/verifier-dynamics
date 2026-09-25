"""E004a design panel (registry E004a §2-§4): 12 constructions, common-target calibration, twins.

Every structure shares the U-toy parameterization (toy.py). Common draws: w, p, skill offsets,
h0, phi0, coupling lam, targets (J_G(0), FPR(0), FNR(0)) and the mechanism share omega. The
construction's channel is calibrated first; top-up coins (background noise) then bring FPR and
FNR to the common targets. Channel-only FPR/FNR above a target trigger redraws (counted).
"""

from typing import Any

import numpy as np
from scipy.optimize import brentq

from vdyn.e004 import toy

ROOT_SEED = 20260930
CONSTRUCTIONS = ("R1", "R2", "X1", "X2", "YA1", "YA2", "YB1", "YB2", "B1", "B2", "D1", "D2")
MECHANISMS = ("R", "X", "YA", "YB", "B", "D")
N_DESIGN = 40
MAX_TRIES = 50
FEATURES_OF = {"single": [0], "or2": [0, 1], "and2": [0, 1], "and3": [0, 1, 2]}


def mechanism_of(construction: str) -> str:
    return construction.rstrip("12")


def _static(st: toy.Structure) -> tuple[float, float, float]:
    ex = toy.single(st, st.theta0)
    return float(ex["J_G"]), float(ex["FPR"]), float(ex["FNR"])


def _common(rng: np.random.Generator) -> dict[str, Any]:
    delta = rng.normal(0, 1, toy.K)
    return {
        "w": rng.dirichlet(np.full(toy.K, 2.0)),
        "p": np.clip(rng.beta(2, 1.5, toy.K), 0.2, 0.98),
        "delta": delta - delta.mean(),
        "h0": rng.normal(-2, 1),
        "phi0": rng.normal(-1.5, 1, toy.M_FEAT),
        "lam": rng.normal(0, 0.5, toy.M_FEAT),
        "omega": rng.uniform(0.5, 1.0),
    }


def _targets(rng: np.random.Generator) -> dict[str, float]:
    return {
        "J_G": rng.uniform(0.05, 0.5),
        "FPR": float(np.exp(rng.uniform(np.log(0.02), np.log(0.4)))),
        "FNR": rng.uniform(0.0, 0.25),
    }


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _base(sid: str, construction: str, c: dict[str, Any]) -> toy.Structure:
    z = np.zeros(toy.K)
    theta = np.concatenate([c["delta"], [c["h0"]], c["phi0"]])
    return toy.Structure(
        sid=sid,
        construction=construction,
        mechanism=mechanism_of(construction),
        w=c["w"],
        p=c["p"],
        theta0=theta,
        lam=c["lam"],
        event="none",
        trig=z,
        fp=z,
        fn=z,
        rho=z,
        beta=z,
        deleted=np.zeros(toy.K, bool),
        v0=1.0,
    )


def _set_theta(st: toy.Structure, mu: float | None = None, h: float | None = None,
               shift: tuple[list[int], float] | None = None) -> toy.Structure:  # fmt: skip
    th = st.theta0.copy()
    if mu is not None:
        th[: toy.K] = st.meta["delta"] + mu
    if h is not None:
        th[toy.K] = h
    if shift is not None:
        idx, c = shift
        for j in idx:
            th[toy.K + 1 + j] = st.meta["phi0"][j] + c
    return st.with_(theta0=th)


def _solve_mu(st: toy.Structure, target: float) -> toy.Structure:
    f = lambda mu: _static(_set_theta(st, mu=mu))[0] - target  # noqa: E731
    return _set_theta(st, mu=brentq(f, -25, 25, xtol=1e-13))


def _event_prob(st: toy.Structure) -> float:
    """P(E) at theta0 (features independent of s, so the same in every prompt)."""
    P = toy.row_probs(st, st.theta0)[0]
    ev = toy.event(st.event)
    solve0 = (toy.ROW_S == toy.SOLVE) & (toy.ROW_XI == 0)
    # P(z) is the row probability divided by P(s, xi); use SOLVE,xi=0 rows normalized
    pz = P[solve0] / P[solve0].sum()
    return float(pz[ev[solve0]].sum())


def _solve_shift(st: toy.Structure, target_se: float) -> toy.Structure:
    idx = FEATURES_OF[st.event]
    f = lambda c: _event_prob(_set_theta(st, shift=(idx, c))) - target_se  # noqa: E731
    return _set_theta(st, shift=(idx, brentq(f, -40, 40, xtol=1e-13)))


def _topup(st: toy.Structure, targets: dict[str, float]) -> toy.Structure:
    """Background coins on non-deleted prompts bring FPR and FNR to their targets."""
    live = ~st.deleted
    fp_c, fn_c = st.fp.copy(), st.fn.copy()

    def with_fp(x: float) -> toy.Structure:
        return st.with_(fp=np.where(live, 1 - (1 - fp_c) * (1 - x), fp_c))

    j, fpr, fnr = _static(st)
    if targets["FPR"] > fpr + 1e-12:
        x = brentq(lambda x: _static(with_fp(x))[1] - targets["FPR"], 0, 0.999, xtol=1e-14)
        st = with_fp(x)
    if targets["FNR"] > fnr + 1e-12:

        def with_fn(x: float) -> toy.Structure:
            return st.with_(fn=np.where(live, 1 - (1 - fn_c) * (1 - x), fn_c))

        x = brentq(lambda x: _static(with_fn(x))[2] - targets["FNR"], 0, 0.999, xtol=1e-14)
        st = with_fn(x)
    return st


class Infeasible(Exception):
    pass


def _channel(construction: str, st: toy.Structure, c: dict[str, Any], t: dict[str, float],
             rng: np.random.Generator) -> toy.Structure:  # fmt: skip
    """Place the construction's channel on st (theta0 already has mu solved)."""
    K, share = toy.K, c["omega"] * t["FPR"]
    meta = dict(st.meta)
    if construction == "R1":
        eps = t["FPR"]
        st = st.with_(fp=np.full(K, eps), fn=np.full(K, eps))
        meta["eps"] = eps
    elif construction == "R2":
        r = np.minimum(rng.dirichlet(np.ones(K)) * K, 1.0)
        r2 = np.minimum(rng.dirichlet(np.ones(K)) * K, 1.0)

        def fpr_of(e: float) -> float:
            return _static(st.with_(fp=np.minimum(e * r, 0.45)))[1]

        def fnr_of(e: float) -> float:
            return _static(st.with_(fn=np.minimum(e * r2, 0.45)))[2]

        if fpr_of(0.45) < t["FPR"] or (t["FNR"] > 0 and fnr_of(0.45) < t["FNR"]):
            raise Infeasible("R2 coins cannot reach the targets")
        e1 = brentq(lambda e: fpr_of(e) - t["FPR"], 0, 0.45, xtol=1e-14)
        e2 = brentq(lambda e: fnr_of(e) - t["FNR"], 0, 0.45, xtol=1e-14) if t["FNR"] > 0 else 0.0
        st = st.with_(fp=np.minimum(e1 * r, 0.45), fn=np.minimum(e2 * r2, 0.45))
        meta.update(r=r.tolist(), r2=r2.tolist())
    elif construction in ("X1", "X2"):
        size = int(rng.integers(1, 3))
        deleted = np.zeros(K, bool)
        deleted[rng.choice(K, size, replace=False)] = True
        v0 = 1.0 if construction == "X1" else float(rng.uniform(0, 0.3))
        st = st.with_(deleted=deleted, v0=v0)
    elif construction in ("YA1", "YA2", "YB1", "YB2"):
        event = {"YA1": "and2" if rng.random() < 0.5 else "and3", "YA2": "and2", "YB1": "single",
                 "YB2": "or2"}[construction]  # fmt: skip
        trig = np.ones(K)
        if construction == "YA2":
            trig = np.zeros(K)
            trig[rng.choice(K, int(rng.integers(1, 4)), replace=False)] = 1.0
            se = _loguniform(rng, 0.002, 0.1)
        elif construction == "YA1":
            se = min(share, _loguniform(rng, 0.001, 0.05))
        else:
            se = share
        st = _solve_shift(st.with_(event=event, trig=trig), se)
        meta["S_E0"] = se
    elif construction in ("B1", "B2"):
        credit = np.ones(K, bool)
        if construction == "B2":
            credit = np.zeros(K, bool)
            credit[rng.choice(K, int(rng.integers(1, 4)), replace=False)] = True

        def with_beta(b: float) -> toy.Structure:
            return st.with_(beta=np.where(credit, b, 0.0))

        if _static(with_beta(1.0))[1] <= share:
            st, meta["capped"] = with_beta(1.0), True
        else:
            st = with_beta(brentq(lambda b: _static(with_beta(b))[1] - share, 0, 1, xtol=1e-14))
    elif construction in ("D1", "D2"):
        rho = float(rng.uniform(0.3, 0.9))
        fn_ch = np.zeros(K)
        strict = np.zeros(K, bool)
        if construction == "D1":
            if not np.any(st.p < rho):
                raise Infeasible("no prompt with p < rho")
        else:
            easy = np.flatnonzero(st.p >= rho)
            if len(easy) == 0:
                raise Infeasible("no prompt with p >= rho")
            pick = rng.choice(easy, min(len(easy), int(rng.integers(1, 3))), replace=False)
            strict[pick] = True
            fn_ch[strict] = 1 - rho * rng.uniform(0.5, 0.9, strict.sum()) / st.p[strict]
            meta["strict"] = strict.tolist()
        st = st.with_(rho=np.full(K, rho), fn=fn_ch)
        # joint (mu, h0): J_G(0) = target, channel FPR = omega * target
        for _ in range(60):
            prev = st.theta0.copy()
            f = lambda h, st=st: _static(_set_theta(st, h=h))[1] - share  # noqa: E731
            if f(-25) > 0 or f(8) < 0:
                raise Infeasible("hack share cannot reach the FPR share")
            st = _set_theta(st, h=brentq(f, -25, 8, xtol=1e-13))
            st = _solve_mu(st, t["J_G"])
            if np.max(np.abs(st.theta0 - prev)) < 1e-11:
                break
        else:
            raise Infeasible("mu/h0 did not converge")
    else:
        raise ValueError(construction)
    meta["fp_channel"], meta["fn_channel"] = st.fp.tolist(), st.fn.tolist()
    return st.with_(meta=meta)


def build(sid: str, construction: str, rng: np.random.Generator) -> toy.Structure:
    rejections = {"channel": 0, "targets": 0}
    c = _common(rng)
    while True:
        t = _targets(rng)
        if t["J_G"] >= 0.9 * float(c["w"] @ c["p"]):
            rejections["targets"] += 1
            continue
        for _ in range(MAX_TRIES):
            st = _base(sid, construction, c).with_(meta={"delta": c["delta"], "phi0": c["phi0"]})
            try:
                st = _solve_mu(st, t["J_G"])
                st = _channel(construction, st, c, t, rng)
                _, fpr_c, fnr_c = _static(st)
                if fpr_c > t["FPR"] + 1e-9 or fnr_c > t["FNR"] + 1e-9:
                    raise Infeasible("channel exceeds a target")
            except (Infeasible, ValueError):
                rejections["channel"] += 1
                continue
            st = _topup(st, t)
            meta = dict(st.meta)
            meta.update(
                targets=t,
                omega=c["omega"],
                rejections=rejections,
                delta=np.asarray(c["delta"]).tolist(),
                phi0=np.asarray(c["phi0"]).tolist(),
            )
            return st.with_(meta=meta)
        rejections["targets"] += 1


def design_panel(n_per: int = N_DESIGN) -> list[toy.Structure]:
    design_ss = np.random.SeedSequence(ROOT_SEED).spawn(4)[0]
    children = design_ss.spawn(len(CONSTRUCTIONS))
    out = []
    for c_i, construction in enumerate(CONSTRUCTIONS):
        for j, ss in enumerate(children[c_i].spawn(n_per)):
            out.append(build(f"{construction}-{j:02d}", construction, np.random.default_rng(ss)))
    return out


def _reparam_phi(st: toy.Structure) -> np.ndarray:
    th = st.theta0.copy()
    th[toy.K + 1 :] = th[toy.K + 1 :] + st.lam * th[: toy.K].mean()
    return th


def uncoupled_twin(st: toy.Structure) -> toy.Structure:
    """lam = 0 with phi re-expressed so the t = 0 distribution is unchanged; same coins."""
    return st.with_(sid=st.sid + "-unc", theta0=_reparam_phi(st), lam=np.zeros(toy.M_FEAT))


def canonical_twin(st: toy.Structure) -> toy.Structure:
    """Uncoupled twin with the background (top-up) coins removed: channel-only coins."""
    return uncoupled_twin(st).with_(sid=st.sid + "-can", fp=np.array(st.meta["fp_channel"]),
                                    fn=np.array(st.meta["fn_channel"]))  # fmt: skip
