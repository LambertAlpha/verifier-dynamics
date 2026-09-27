"""E005a-R calibration engine (research/09_e005ar_design.md §5-§7).

For one calibration point: R Monte Carlo audits at every N (m = 8); for every representation
(R0 full, R1 random k, R2 gold-selected k on an independent half, R3 cross-fitted k, R4
functional, R5 oracle S*; on the design split also the R4 sensitivity variants) the E3 estimate,
its jackknife SE, the sign-flip test, the legacy Wald test and the representation's own oracle
C^2 (recomputed per replication for data-dependent bases), the retained behavior signal and the
nuisance-error leakage. Keys: "<rep>|<k>|<N>" (k = 0 when the representation has no k).
"""

from typing import Any

import numpy as np

from vdyn.e005ar import env
from vdyn.e005ar import represent as rp
from vdyn.e005ar import sampling as sm
from vdyn.e005ar import stats as st

N_GRID = (32, 64, 128, 256, 512, 1024)
K_GRID = (4, 8, 16, 32, 64)
KMAX = max(K_GRID)
M = env.M_GROUP
R_REPS = 100
CHUNK_ELEMS = 16_000_000
MIN_SPLIT_N = 64
Z95 = 1.96


def _chunks(R: int, N: int, d: int) -> list[int]:
    size = max(1, min(R, CHUNK_ELEMS // (N * d)))
    return [min(size, R - lo) for lo in range(0, R, size)]


class _Acc:
    """Per-configuration accumulator of per-replication values."""

    def __init__(self) -> None:
        self.data: dict[str, dict[str, list[np.ndarray]]] = {}

    def add(self, key: str, **vals: Any) -> None:
        slot = self.data.setdefault(key, {})
        for q, v in vals.items():
            slot.setdefault(q, []).append(np.atleast_1d(np.asarray(v, dtype=float)))

    def arrays(self) -> dict[str, dict[str, np.ndarray]]:
        return {k: {q: np.concatenate(v) for q, v in s.items()} for k, s in self.data.items()}


def _add_fixed(acc: _Acc, key: str, xG: np.ndarray, xe: np.ndarray, rng: np.random.Generator,
               oracle_c2: float, retention: float, leak: float) -> None:  # fmt: skip
    e = st.e3(xG, xe)
    t = st.signflip(xG, xe, rng)
    R = xG.shape[0]
    acc.add(key, C2=e["C2"], SE=e["SE_C2"], alpha=e["alpha"], T=t["T"], reject=t["reject"],
            wald=st.wald_reject(e), oracle=np.full(R, oracle_c2),
            retention=np.full(R, retention), leak=np.full(R, leak))  # fmt: skip


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def summarize(a: dict[str, np.ndarray]) -> dict[str, float]:
    c2, se, orc = a["C2"], a["SE"], a["oracle"]
    fin = np.isfinite(c2) & np.isfinite(a["T"])
    out: dict[str, float] = {"n": float(len(c2)), "nonfinite": float(1 - fin.mean()),
                             "oracle_mean": float(np.mean(orc))}  # fmt: skip
    x, s, o = c2[fin], se[fin], orc[fin]
    if len(x):
        out |= {"mean": float(x.mean()), "var": float(x.var(ddof=1)) if len(x) > 1 else 0.0,
                "bias": float(np.mean(x - o)), "rmse": float(np.sqrt(np.mean((x - o) ** 2))),
                "coverage": float(np.mean(np.abs(x - o) <= Z95 * s)),
                "q95": float(np.quantile(x, 0.95)), "mean_se": float(np.mean(s))}  # fmt: skip
    ret, leak = a["retention"], a["leak"]
    out |= {
        "reject": float(np.mean(a["reject"])),
        "reject_wald": float(np.mean(a["wald"])),
        "alpha_undefined": float(np.mean(~np.isfinite(a["alpha"]))),
        "retention_mean": float(np.nanmean(ret)) if np.isfinite(ret).any() else float("nan"),
        "retention_median": (float(np.nanmedian(ret)) if np.isfinite(ret).any() else float("nan")),
        "leak_mean": float(np.nanmean(leak)) if np.isfinite(leak).any() else float("nan"),
    }
    return out  # fmt: skip


def run_point(point: dict[str, Any], base: dict[str, Any], seed: np.random.SeedSequence,
              N: tuple[int, ...] = N_GRID, R: int = R_REPS,
              sensitivity: bool = False) -> dict[str, Any]:  # fmt: skip
    ctx = env.Context.of_point(point, base)
    o = env.oracle(ctx)
    r, d = ctx.r, ctx.d
    hG, he, he_beh = o["h_G"], o["h_e"], o["h_e_beh"]
    he_nuis = he - he_beh
    c2_beh, nuis2 = o["C2_beh"], o["nuis_err2"]
    Phis = {"R4": env.functional_map(base, "main", d)}
    if sensitivity:
        for v in env.FUNCTIONAL_VARIANTS[1:]:
            Phis[f"R4-{v}"] = env.functional_map(base, v, d, ctx.lam_spec, ctx.v)
    Pcore = np.zeros((d, r))
    Pcore[:r] = np.eye(r)
    fixed = {"R0": np.eye(d), "R5": Pcore, **Phis}
    ks = [k for k in K_GRID if k <= d]

    def proj_stats(Q: np.ndarray) -> tuple[float, float, float]:
        c2 = st.projected_geometry(hG, he, Q)["C2"]
        ret = _ratio(st.projected_geometry(hG, he_beh, Q)["C2"], c2_beh)
        leak = _ratio(float(np.sum((Q.T @ he_nuis) ** 2)), nuis2)
        return c2, ret, leak

    acc = _Acc()
    for Nn, sub in zip(N, seed.spawn(len(N)), strict=True):
        rng_a, rng_b, rng_t = (np.random.default_rng(s) for s in sub.spawn(3))
        n = Nn // M
        split = Nn >= MIN_SPLIT_N
        for Rc in _chunks(R, Nn, d):
            au = sm.audit(ctx, rng_a, Rc, Nn, M, per_rollout=split)
            xG, xe = au["xG"], au["xV"] - au["xG"]
            for name, Q in fixed.items():
                c2 = st.projected_geometry(hG, he, Q)["C2"]
                ret = 1.0 if name == "R0" else _ratio(
                    st.projected_geometry(hG, he_beh, Q)["C2"], c2_beh)  # fmt: skip
                leak = _ratio(float(np.sum((Q.T @ he_nuis) ** 2)), nuis2)
                _add_fixed(acc, f"{name}|0|{Nn}", xG @ Q, xe @ Q, rng_t, c2, ret, leak)
            # R1: fresh Haar basis per replication (nested in k)
            bases = [rp.haar_basis(d, min(KMAX, d), rng_b) for _ in range(Rc)]
            for k in ks:
                Qs = [b[:, :k] for b in bases]
                pg = np.array([proj_stats(Q) for Q in Qs])
                pxG = np.stack([xG[i] @ Qs[i] for i in range(Rc)])
                pxe = np.stack([xe[i] @ Qs[i] for i in range(Rc)])
                e = st.e3(pxG, pxe)
                t = st.signflip(pxG, pxe, rng_t)
                acc.add(f"R1|{k}|{Nn}", C2=e["C2"], SE=e["SE_C2"], alpha=e["alpha"], T=t["T"],
                        reject=t["reject"], wald=st.wald_reject(e), oracle=pg[:, 0],
                        retention=pg[:, 1], leak=pg[:, 2])  # fmt: skip
            if not split:
                continue
            sel, est = rp.halves(n)
            yG, yV = au["yG"], au["yV"]
            b2 = [rp.top_eigvecs(yG[i, sel].reshape(-1, d), KMAX) for i in range(Rc)]
            b3 = [rp.crossfit_bases(yG[i], yV[i], KMAX) for i in range(Rc)]
            for k in ks:
                if min(b.shape[1] for b in b2) >= k:
                    Qs = [b[:, :k] for b in b2]
                    pg = np.array([proj_stats(Q) for Q in Qs])
                    pxG = np.stack([xG[i, est] @ Qs[i] for i in range(Rc)])
                    pxe = np.stack([xe[i, est] @ Qs[i] for i in range(Rc)])
                    e = st.e3(pxG, pxe)
                    t = st.signflip(pxG, pxe, rng_t)
                    acc.add(f"R2|{k}|{Nn}", C2=e["C2"], SE=e["SE_C2"], alpha=e["alpha"],
                            T=t["T"], reject=t["reject"], wald=st.wald_reject(e),
                            oracle=pg[:, 0], retention=pg[:, 1], leak=pg[:, 2])  # fmt: skip
                if min(min(bA.shape[1], bB.shape[1]) for bA, bB in b3) >= k:
                    A, Bf = rp.halves(n)
                    QA = [bA[:, :k] for bA, _ in b3]  # built from fold A, used on fold B
                    QB = [bB[:, :k] for _, bB in b3]
                    gB = np.stack([xG[i, Bf] @ QA[i] for i in range(Rc)])
                    eB = np.stack([xe[i, Bf] @ QA[i] for i in range(Rc)])
                    gA = np.stack([xG[i, A] @ QB[i] for i in range(Rc)])
                    eA = np.stack([xe[i, A] @ QB[i] for i in range(Rc)])
                    e1, e2 = st.e3(gB, eB), st.e3(gA, eA)
                    t = st.signflip_from_grams([st.gram_residual(gB, eB),
                                                st.gram_residual(gA, eA)], rng_t)  # fmt: skip
                    pa = np.array([proj_stats(Q) for Q in QA])
                    pb = np.array([proj_stats(Q) for Q in QB])
                    pg = 0.5 * (pa + pb)
                    comb = {
                        "C2": 0.5 * (e1["C2"] + e2["C2"]),
                        "SE_C2": 0.5 * np.sqrt(e1["SE_C2"] ** 2 + e2["SE_C2"] ** 2),
                    }
                    alpha = 0.5 * (e1["alpha"] + e2["alpha"])
                    acc.add(f"R3|{k}|{Nn}", C2=comb["C2"], SE=comb["SE_C2"], alpha=alpha,
                            T=t["T"], reject=t["reject"], wald=st.wald_reject(comb),
                            oracle=pg[:, 0], retention=pg[:, 1], leak=pg[:, 2])  # fmt: skip
    arrays = acc.arrays()
    return {"pid": point["pid"],
            "summary": {k: summarize(a) for k, a in arrays.items()},
            "reps": {k: a["C2"].astype(np.float32) for k, a in arrays.items()},
            "reps_reject": {k: a["reject"].astype(bool) for k, a in arrays.items()}}  # fmt: skip
