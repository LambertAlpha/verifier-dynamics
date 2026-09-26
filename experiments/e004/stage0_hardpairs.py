"""E004a Stage 0, step 4 (registry E004a §8.7): hard-pair search, verification and freeze.

Search under MF-Adam (design approximation): least squares over a partner structure's parameters
(theta0, top-up coins, channel strength) against a design-panel anchor. Correction A residuals
are every L2 feature difference over [0, h*] in audit-SE units / 0.5 (accept: all <= 1).
Correction B adds L0 (audit-SE / 0.5) and L1 (registered tolerances) residuals, then requires an
L3 divergence before the failing member's visible onset. Verification: sampled Adam, 32 seeds per
member (+8 clean seeds each). The search does not force existence: the best mismatch is reported.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from vdyn import provenance
from vdyn.e004 import dynamics as dy
from vdyn.e004 import features as fe
from vdyn.e004 import hardpairs as hp
from vdyn.e004 import outcomes as oc
from vdyn.e004 import panel as pn
from vdyn.e004 import toy

REPO = Path(__file__).resolve().parents[2]
PANEL = REPO / "configs" / "e004" / "design_panel.json"
CONFIG = REPO / "configs" / "e004" / "e004a.toml"
H_PRE = [0.002, 0.005, 0.01, 0.02]
PRE_FRACS = sorted({0.0, *H_PRE, *(h / 2 for h in H_PRE)})
OUT_FRACS = [i / 100 for i in range(101)]
PAIR_TYPES = {  # name: (anchor mechanism, partner mechanism, kind)
    "HP-A": ("B", "YA", "A"),
    "HP-D": ("D", "B", "A"),
    "HP-B": ("YA", "YB", "A"),
    "DYN-D/B": ("D", "B", "B"),
    "DYN-D/R": ("D", "R", "B"),
    "DYN-YA/B": ("YA", "B", "B"),
    "DYN-YA/R": ("YA", "R", "B"),
    "DYN-X/YB": ("X", "YB", "B"),
    "DYN-YB/B": ("YB", "B", "B"),
}
N_ANCHORS, N_SEEDS_VERIFY, N_CLEAN_VERIFY = 10, 32, 8


def steps_of(fracs: list[float], T: int) -> list[int]:
    return [0 if f == 0 else max(1, round(f * T)) for f in fracs]


def mf_traj(
    st: toy.Structure, T: int, fracs: list[float], clean: bool = False, geometry: bool = False
) -> dict[str, np.ndarray]:
    steps = steps_of(fracs, T)
    res = dy.run_mf_adam([st], max(steps), sorted(set(steps)), clean=clean)
    ck = res["checkpoints"]
    th = res["theta"][0][[ck.index(s) for s in steps]]
    v = res["v_hat"][0][[ck.index(s) for s in steps]]
    tb = toy.Tables.of([st] * len(steps))
    tb = tb.clean() if clean else tb
    obs = dy.observables(tb, th)
    out = {k: obs[k] for k in ("J_G", "J_V", "FPR", "FNR")}
    if geometry:
        geo = dy.geometry(tb, th, "adam", v_hat=v)
        out.update({k: geo[k] for k in ("A", "alpha", "C")})
    return out


def horizons_idx(fracs: list[float]) -> dict[float, tuple[int, int, int]]:
    return {h: (fracs.index(0.0), fracs.index(h / 2), fracs.index(h)) for h in H_PRE}


# ------------------------------------------------------------ partner parameterization
def encode(st: toy.Structure) -> np.ndarray:
    return np.concatenate([st.theta0, [0.0, 0.0, _strength(st)]])


def _strength(st: toy.Structure) -> float:
    c = st.construction
    if c.startswith("B"):
        return float(st.beta.max())
    if c.startswith("D"):
        return float(st.rho.max())
    if c == "X2":
        return st.v0
    if c.startswith("R"):
        return 1.0
    return 0.0


def decode(base: toy.Structure, x: np.ndarray) -> toy.Structure:
    theta, fp_top, fn_top, s = x[:8], x[8], x[9], x[10]
    fp_c, fn_c = np.array(base.meta["fp_channel"]), np.array(base.meta["fn_channel"])
    c, live = base.construction, ~base.deleted
    kw: dict[str, Any] = {"theta0": theta}
    if c.startswith("B"):
        kw["beta"] = np.where(base.beta > 0, s, 0.0)
    elif c.startswith("D"):
        kw["rho"] = np.full(toy.K, s)
    elif c == "X2":
        kw["v0"] = s
    elif c.startswith("R"):
        fp_c, fn_c = np.clip(fp_c * s, 0, 0.9), np.clip(fn_c * s, 0, 0.9)
    kw["fp"] = np.where(live, 1 - (1 - fp_c) * (1 - fp_top), fp_c)
    kw["fn"] = np.where(live, 1 - (1 - fn_c) * (1 - fn_top), fn_c)
    return base.with_(**kw)


def bounds(base: toy.Structure) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([-8.0] * 4 + [-10.0] + [-10.0] * 3 + [0.0, 0.0, 0.0])
    hi = np.array([8.0] * 4 + [4.0] + [6.0] * 3 + [0.9, 0.9, 1.0])
    c = base.construction
    if c.startswith("D"):
        lo[10], hi[10] = 0.3, 0.9
    elif c == "X2":
        hi[10] = 0.3
    elif c.startswith("R"):
        lo[10], hi[10] = 0.2, 3.0
    elif not c.startswith("B"):
        lo[10], hi[10] = -1e-9, 1e-9
    return lo, hi


# ------------------------------------------------------------ residuals
def l2_residual(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> np.ndarray:
    res = []
    for h, idx in horizons_idx(PRE_FRACS).items():
        fa = np.concatenate([fe.summaries(a[k][list(idx)], h) for k in fe.L2_VARS])
        fb = np.concatenate([fe.summaries(b[k][list(idx)], h) for k in fe.L2_VARS])
        mean = {k: 0.5 * (a[k][list(idx)] + b[k][list(idx)]) for k in fe.L2_VARS}
        res.append((fa - fb) / (hp.TOL * np.maximum(fe.l2_feature_se(mean, h), 1e-12)))
    return np.concatenate(res)


def l0l1_residual(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> np.ndarray:
    v = lambda d: {k: float(d[k][0]) for k in ("J_G", "J_V", "FPR", "FNR")}  # noqa: E731
    va, vb = v(a), v(b)
    mean = {k: 0.5 * (va[k] + vb[k]) for k in va}
    vec = lambda d: np.array([d["J_G"], d["J_V"], d["FPR"], d["FNR"], (1 - d["J_G"]) * d["FPR"]])  # noqa: E731
    r0 = (vec(va) - vec(vb)) / (hp.TOL * np.maximum(fe.l0_se(mean), 1e-12))
    mA, mC = 0.5 * (a["A"][0] + b["A"][0]), 0.5 * (a["C"][0] + b["C"][0])
    r1 = np.array(
        [
            (a["A"][0] - b["A"][0]) / (hp.L1_REL_A * mA),
            (a["alpha"][0] - b["alpha"][0]) / hp.L1_ALPHA,
            (a["C"][0] - b["C"][0]) / (hp.L1_REL_C * max(mC, 0.01)),
        ]
    )
    return np.concatenate([r0, np.nan_to_num(r1, nan=10.0)])


def search_one(args: tuple[Any, ...]) -> dict[str, Any]:
    anchor, base, kind, T = args
    geo = kind == "B"
    ta = mf_traj(anchor, T, PRE_FRACS, geometry=geo)

    def resid(x: np.ndarray) -> np.ndarray:
        tb_ = mf_traj(decode(base, x), T, PRE_FRACS, geometry=geo)
        r = l2_residual(ta, tb_)
        return np.concatenate([r, l0l1_residual(ta, tb_)]) if geo else r

    x0 = encode(base)
    lo, hi = bounds(base)
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    try:
        sol = least_squares(resid, x0, bounds=(lo, hi), max_nfev=400, x_scale="jac")
        x, r = sol.x, sol.fun
    except Exception as exc:  # report, never hide
        return {"anchor": anchor.sid, "base": base.sid, "error": repr(exc)}
    return {
        "anchor": anchor.sid,
        "base": base.sid,
        "x": x.tolist(),
        "max_scaled_residual": float(np.max(np.abs(r))),
        "sse": float(np.sum(r**2)),
    }


# ------------------------------------------------------------ evaluation of candidates
def mf_outcome(st: toy.Structure, T: int) -> dict[str, Any]:
    ver = mf_traj(st, T, OUT_FRACS, geometry=True)
    clean = mf_traj(st, T, OUT_FRACS, clean=True)
    return {"traj": ver, "label": oc.label(ver["J_G"], clean["J_G"])}


def sampled_member(st: toy.Structure, T: int, seeds: list[Any], clean_seeds: list[Any]) -> dict:
    fr = sorted(set(PRE_FRACS + OUT_FRACS))
    steps = steps_of(fr, T)
    ck = sorted(set(steps))
    pos = [ck.index(s) for s in steps]
    res = dy.run_sampled_adam([st] * len(seeds), seeds, T, ck)
    cl = dy.run_sampled_adam([st] * len(clean_seeds), clean_seeds, T, ck, clean=True)
    n = len(seeds)
    tb = toy.Tables.of([st] * (n * len(pos)))
    th = res["theta"][:, pos].reshape(-1, toy.D)
    v = res["v_hat"][:, pos].reshape(-1, toy.D)
    ex0 = toy.exact(tb, th)
    is0 = np.tile(np.array(steps) == 0, n)
    v[is0] = (ex0["g_eff"] ** 2 + ex0["var_eff"] / 64)[is0]
    obs = dy.observables(tb, th)
    geo = dy.geometry(tb, th, "adam", v_hat=v)
    traj = {k: obs[k].reshape(n, len(pos)) for k in ("J_G", "J_V", "FPR", "FNR")}
    traj.update({k: geo[k].reshape(n, len(pos)) for k in ("A", "alpha", "C")})
    tbc = toy.Tables.of([st] * (len(clean_seeds) * len(pos))).clean()
    jc = dy.observables(tbc, cl["theta"][:, pos].reshape(-1, toy.D))["J_G"].reshape(-1, len(pos))
    oi = [fr.index(f) for f in OUT_FRACS]
    labs = [oc.label(traj["J_G"][r, oi], jc.mean(0)[oi]) for r in range(n)]
    return {"fr": fr, "mean": {k: np.nanmean(v_, 0) for k, v_ in traj.items()}, "labels": labs}


def evaluate(
    anchor: toy.Structure, partner: toy.Structure, kind: str, T: int, seeds: list[list[Any]]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # MF-Adam (design approximation)
    ma, mb = mf_outcome(anchor, T), mf_outcome(partner, T)
    pa, pb = (
        mf_traj(anchor, T, PRE_FRACS, geometry=True),
        mf_traj(partner, T, PRE_FRACS, geometry=True),
    )
    out["mf"] = {
        "correction_a": hp.correction_a(pa, pb, horizons_idx(PRE_FRACS)),
        "labels": [ma["label"], mb["label"]],
    }
    if kind == "B":
        out["mf"]["l0"] = hp.l0_match(
            {k: float(pa[k][0]) for k in fe.L2_VARS}, {k: float(pb[k][0]) for k in fe.L2_VARS}
        )
        out["mf"]["l1"] = hp.l1_match(
            {k: float(pa[k][0]) for k in ("A", "alpha", "C")},
            {k: float(pb[k][0]) for k in ("A", "alpha", "C")},
        )
        t_on = [x["label"]["t_on"] for x in (ma, mb) if x["label"]["failure"]]
        limit = min(t_on) if t_on and all(t is not None for t in t_on) else 1.0
        out["mf"]["l3_divergence_time"] = hp.l3_divergence(
            {k: ma["traj"][k] for k in ("A", "alpha", "C")},
            {k: mb["traj"][k] for k in ("A", "alpha", "C")},
            np.array(OUT_FRACS),
            limit,
        )
    # sampled Adam verification (32 seeds per member, 8 clean seeds)
    sa = sampled_member(anchor, T, seeds[0][:N_SEEDS_VERIFY], seeds[0][N_SEEDS_VERIFY:])
    sb = sampled_member(partner, T, seeds[1][:N_SEEDS_VERIFY], seeds[1][N_SEEDS_VERIFY:])
    fr = sa["fr"]
    pre = [fr.index(f) for f in PRE_FRACS]
    ma_pre = {k: sa["mean"][k][pre] for k in fe.L2_VARS}
    mb_pre = {k: sb["mean"][k][pre] for k in fe.L2_VARS}
    fails = [float(np.mean([lb["failure"] for lb in s["labels"]])) for s in (sa, sb)]
    out["sampled"] = {
        "correction_a": hp.correction_a(ma_pre, mb_pre, horizons_idx(PRE_FRACS)),
        "failure_fraction": fails,
        "Dn_mean": [float(np.mean([lb["Dn"] for lb in s["labels"]])) for s in (sa, sb)],
    }
    if kind == "B":
        oi = [fr.index(f) for f in OUT_FRACS]
        ton = [
            np.median([lb["t_on"] for lb in s["labels"] if lb["t_on"] is not None] or [1.0])
            for s in (sa, sb)
        ]
        failing = [k for k in (0, 1) if fails[k] >= 0.5]
        limit = float(min(ton[k] for k in failing)) if failing else 1.0
        out["sampled"]["l0"] = hp.l0_match(
            {k: float(ma_pre[k][0]) for k in fe.L2_VARS},
            {k: float(mb_pre[k][0]) for k in fe.L2_VARS},
        )
        out["sampled"]["l1"] = hp.l1_match(
            {k: float(sa["mean"][k][0]) for k in ("A", "alpha", "C")},
            {k: float(sb["mean"][k][0]) for k in ("A", "alpha", "C")},
        )
        out["sampled"]["l3_divergence_time"] = hp.l3_divergence(
            {k: sa["mean"][k][oi] for k in ("A", "alpha", "C")},
            {k: sb["mean"][k][oi] for k in ("A", "alpha", "C")},
            np.array(OUT_FRACS),
            limit,
        )
        out["sampled"]["median_t_on"] = ton
    return out


def accepted(ev: dict[str, Any], kind: str) -> dict[str, bool]:
    res = {}
    for tier in ("mf", "sampled"):
        e = ev[tier]
        ok = e["correction_a"]["accept"]
        if kind == "B":
            fails = (
                [lb["failure"] for lb in e["labels"]]
                if tier == "mf"
                else [f >= 0.5 for f in e["failure_fraction"]]
            )
            ok = (
                ok
                and e["l0"]["accept"]
                and e["l1"]
                and any(fails)
                and (e["l3_divergence_time"] is not None)
            )
        res[tier] = bool(ok)
    return res


def main(argv: list[str]) -> int:
    runs_dir = Path(argv[1]).resolve()
    T = int(json.loads((runs_dir / "summary.json").read_text())["T_adam"])
    structs = [toy.Structure.from_dict(d) for d in json.loads(PANEL.read_text())["structures"]]
    by_mech = {m: [s for s in structs if s.mechanism == m] for m in pn.MECHANISMS}
    out_dir = provenance.create_run_dir(REPO / "results", "E004a-stage0-hardpairs", REPO)
    provenance.write_metadata(
        out_dir,
        "E004a-stage0-hardpairs",
        CONFIG,
        REPO,
        extra={"split": "design", "T_adam": T, "runs": str(runs_dir)},
    )
    rng = np.random.default_rng([pn.ROOT_SEED, 8])  # role 8 = hardpair
    jobs, meta = [], []
    for name, (m1, m2, kind) in PAIR_TYPES.items():
        anchors = [s for s in by_mech[m1] if int(s.sid.split("-")[1]) < N_ANCHORS // 2]
        for a in anchors:
            base = by_mech[m2][int(rng.integers(len(by_mech[m2])))]
            jobs.append((a, base, kind, T))
            meta.append(name)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        found = list(pool.map(search_one, jobs, chunksize=1))
    print(f"search: {len(jobs)} restarts, {time.perf_counter() - t0:.0f} s")
    sid = {s.sid: s for s in structs}
    seeds_ss = np.random.SeedSequence(pn.ROOT_SEED).spawn(4)[3].spawn(10)[8]
    report: dict[str, Any] = {"T_adam": T, "pair_types": {}}
    frozen: dict[str, Any] = {"pairs": []}
    type_seeds = seeds_ss.spawn(len(PAIR_TYPES))
    for t_i, (name, (_m1, _m2, kind)) in enumerate(PAIR_TYPES.items()):
        cands = [f for f, n in zip(found, meta, strict=True) if n == name and "x" in f]
        errors = [f for f, n in zip(found, meta, strict=True) if n == name and "error" in f]
        cands.sort(key=lambda f: f["sse"])
        entry: dict[str, Any] = {
            "restarts": len(cands) + len(errors),
            "errors": len(errors),
            "best_max_scaled_residual": cands[0]["max_scaled_residual"] if cands else None,
        }
        if cands:
            best = cands[0]
            anchor, base = sid[best["anchor"]], sid[best["base"]]
            partner = decode(base, np.array(best["x"]))
            mem = type_seeds[t_i].spawn(2)
            seeds = [m.spawn(N_SEEDS_VERIFY + N_CLEAN_VERIFY) for m in mem]
            ev = evaluate(anchor, partner, kind, T, seeds)
            entry.update(best=best, evaluation=ev, accepted=accepted(ev, kind))
            frozen["pairs"].append(
                {
                    "type": name,
                    "kind": kind,
                    "anchor": anchor.sid,
                    "partner": partner.to_dict(),
                    "accepted": entry["accepted"],
                }
            )
        report["pair_types"][name] = entry
        print(name, json.dumps({k: entry.get(k) for k in ("best_max_scaled_residual", "accepted")}))
    (out_dir / "hardpairs.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    (out_dir / "hardpairs_frozen.json").write_text(
        json.dumps(frozen, indent=1, default=float) + "\n"
    )
    print(f"run directory: {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
