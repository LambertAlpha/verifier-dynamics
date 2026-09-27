"""E005a-R analysis (research/09_e005ar_design.md §7-§10). Usage:
  e005ar_analysis.py design <calibration-run-dir>  -> metrics, TP-R1..R6, R4 sensitivity, SELECTION
  e005ar_analysis.py test   <calibration-run-dir>  -> held-out gate, minimum budget, decision label

Every statistic follows §7; the gate G1-G7 and the selection follow §9; the decision tree §10.
Analysis conventions fixed here before any calibration result was read:
  - per-point statistics are averaged with equal weight per point;
  - the SNR z of a non-null point uses its matched clean null (same base, alpha, d, spectrum);
  - TP-R3/TP-R4 ratio cells require z_R0 > 1 (a well-measured denominator); counts are reported;
  - TP-R3/TP-R4 predictions use the matched clean null's covariances, because z is standardized
    by that null's SD (fixed after the 1-base smoke test, before the registered design run);
  - Spearman: per replication across non-null points against C_beh^2, averaged over replications.
  - Amendment 2 (design, before the selection was frozen): a replication without a defined
    estimate (e.g. R2 when the gold contributions have rank < k) counts as non-finite (G5) and as
    a non-rejection; Spearman uses the points defined in that replication.
"""

import hashlib
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005ar import calibrate as cb  # noqa: E402
from vdyn.e005ar import env  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e005ar" / "e005ar.toml"
PRIMARY_N = (64, 128, 256, 512, 1024)
CANDIDATES = [("R4", 0)] + [(rep, k) for rep in ("R2", "R3") for k in cb.K_GRID]
ORDER = {"R4": 0, "R2": 1, "R3": 2}


def load(run_dir: Path, split: str) -> dict[str, Any]:
    panel = json.loads((REPO / f"configs/e005ar/panel_{split}.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    info = json.loads((run_dir / "run_info.json").read_text())
    reps_path = run_dir / "reps.npz"
    assert hashlib.sha256(reps_path.read_bytes()).hexdigest() == info["reps_npz_sha256"]
    return {"points": panel["points"], "summary": summary, "reps": np.load(reps_path)}


def key(rep: str, k: int, N: int) -> str:
    return f"{rep}|{k}|{N}"


class Split:
    """Index sets and matched-null lookup for one panel."""

    def __init__(self, d: dict[str, Any]) -> None:
        self.d = d
        self.pts = d["points"]
        self.S = d["summary"]
        P = self.pts
        self.idx = {p["pid"]: i for i, p in enumerate(P)}
        self.null = [i for i, p in enumerate(P) if p["case"] in ("clean", "nuis")]
        self.clean = [i for i, p in enumerate(P) if p["case"] == "clean"]
        self.nonnull = [i for i, p in enumerate(P) if p["case"] in ("dose", "partial")]
        self.medium = [i for i in self.nonnull if P[i]["dose"] == "medium"]
        self.by_dose = {ds: [i for i in self.nonnull if P[i]["dose"] == ds and
                             P[i]["case"] == "dose"] for ds in env.DOSES}  # fmt: skip
        cl = {(P[i]["sid"], P[i]["alpha_target"], P[i]["d"], P[i]["spectrum"]): i
              for i in self.clean}  # fmt: skip
        self.match = {i: cl.get((P[i]["sid"], P[i]["alpha_target"], P[i]["d"], P[i]["spectrum"]))
                      for i in self.nonnull}  # fmt: skip
        self.null_sets: dict[str, list[int]] = {}
        for i in self.null:
            self.null_sets.setdefault(P[i]["matched_null"], []).append(i)

    def stat(self, i: int, k: str, q: str) -> float:
        return float(self.S[self.pts[i]["pid"]].get(k, {}).get(q, np.nan))

    def col(self, idx: list[int], k: str, q: str) -> np.ndarray:
        return np.array([self.stat(i, k, q) for i in idx])

    def has(self, k: str) -> bool:
        return any(k in self.S[self.pts[i]["pid"]] for i in self.nonnull + self.null)

    def frac_defined(self, i: int, k: str) -> float:
        st = self.S[self.pts[i]["pid"]].get(k)
        return 0.0 if st is None else st["n"] / cb.R_REPS

    def reject(self, idx: list[int], k: str, q: str = "reject") -> np.ndarray:
        """Per-point rejection rate over all R replications (undefined ones do not reject)."""
        return np.array([np.nan_to_num(self.stat(i, k, q)) * self.frac_defined(i, k) for i in idx])

    def nonfinite(self, idx: list[int], k: str) -> np.ndarray:
        return np.array([1 - self.frac_defined(i, k) * (1 - np.nan_to_num(self.stat(i, k,
                         "nonfinite"))) for i in idx])  # fmt: skip

    def z(self, i: int, k: str) -> float:
        j = self.match[i]
        if j is None:
            return float("nan")
        sd = np.sqrt(self.stat(j, k, "var"))
        return (self.stat(i, k, "mean") - self.stat(j, k, "mean")) / sd if sd > 0 else np.nan


def config_metrics(s: Split, rep: str, k: int, N: int) -> dict[str, Any]:
    kk = key(rep, k, N)
    if not s.has(kk):
        return {}
    P = s.pts
    fpr = s.reject(s.null, kk)
    mean_c2 = {i: s.stat(i, kk, "mean") for i in s.null}
    sdb_terms = []
    for idx in s.null_sets.values():
        if len(idx) >= 3:
            sd_within = np.nanmedian([np.sqrt(s.stat(i, kk, "var")) for i in idx])
            sdb_terms.append(np.nanstd([mean_c2[i] for i in idx], ddof=1) / sd_within)
    X = np.full((len(s.nonnull), cb.R_REPS), np.nan)
    for row, i in enumerate(s.nonnull):
        name = f"{P[i]['pid']}::{kk}"
        if name in s.d["reps"]:
            v = np.asarray(s.d["reps"][name], dtype=float)
            X[row, : len(v)] = v
    truth = np.array([P[i]["oracle"]["C2_beh"] for i in s.nonnull])
    rho = []
    for r in range(X.shape[1]):
        ok_r = np.isfinite(X[:, r])
        if ok_r.sum() >= 10:
            rho.append(spearmanr(X[ok_r, r], truth[ok_r]).statistic)
    zs = np.array([s.z(i, kk) for i in s.medium])
    own = s.col(s.nonnull, kk, "oracle_mean")
    mean_est = s.col(s.nonnull, kk, "mean")
    ok = np.isfinite(own) & np.isfinite(mean_est)
    in_s = [i for i in s.nonnull if P[i]["case"] == "dose"]
    leak_pts = [i for i in s.null + s.nonnull if P[i]["lam_bonus"] > 0]
    out = {
        "fpr_pooled": float(np.nanmean(fpr)),
        "fpr_point_ok": float(np.nanmean(fpr <= 0.10)),
        "fpr_by_case": {c: float(np.nanmean(s.reject([i for i in s.null if P[i]["case"] == c],
                                                     kk))) for c in ("clean", "nuis")},
        "fpr_wald": float(np.nanmean(s.reject(s.null, kk, "reject_wald"))),
        "sdb_null": float(np.sqrt(np.nanmean(np.square(sdb_terms)))) if sdb_terms else np.nan,
        "spearman": float(np.mean(rho)) if rho else float("nan"),
        "power": {ds: float(np.nanmean(s.reject(ix, kk))) for ds, ix in s.by_dose.items()},
        "power_medium": float(np.nanmean(s.reject(s.medium, kk))),
        "power_medium_wald": float(np.nanmean(s.reject(s.medium, kk, "reject_wald"))),
        "nonfinite": float(np.nanmean(s.nonfinite(s.null + s.nonnull, kk))),
        "undefined_points": int(sum(s.frac_defined(i, kk) == 0 for i in s.null + s.nonnull)),
        "z_medium_median": float(np.nanmedian(zs)),
        "retention_median": float(np.nanmedian(s.col(in_s, kk, "retention_mean"))),
        "leak_mean": (float(np.nanmean(s.col(leak_pts, kk, "leak_mean"))) if leak_pts
                      else float("nan")),
        "null_q95_median": float(np.nanmedian(s.col(s.null, kk, "q95"))),
        "null_sd_median": float(np.nanmedian(np.sqrt(s.col(s.clean, kk, "var")))),
        "coverage": float(np.nanmean(s.col(s.nonnull + s.null, kk, "coverage"))),
        "rmse_median": float(np.nanmedian(s.col(s.nonnull, kk, "rmse"))),
        "calibration_slope": (float(np.polyfit(own[ok], mean_est[ok], 1)[0]) if ok.sum() > 2
                              else float("nan")),
        "alpha_undefined": float(np.nanmean(s.col(s.nonnull, kk, "alpha_undefined"))),
    }  # fmt: skip
    return out


def all_metrics(s: Split, reps: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rep in reps:
        ks = [0] if not rep.startswith(("R1", "R2", "R3")) else list(cb.K_GRID)
        for k, N in product(ks, cb.N_GRID):
            m = config_metrics(s, rep, k, N)
            if m:
                out[key(rep, k, N)] = m
    return out


def r1_comparator(s: Split, rep: str, k: int, N: int) -> dict[str, float]:
    """R1 at the same k (R2/R3), or at k = r per point (R4)."""
    if rep != "R4":
        kk = key("R1", k, N)
        return {"power": float(np.nanmean(s.col(s.medium, kk, "reject"))),
                "z": float(np.nanmedian([s.z(i, kk) for i in s.medium]))}  # fmt: skip
    pw = [s.stat(i, key("R1", s.pts[i]["r"], N), "reject") for i in s.medium]
    zs = [s.z(i, key("R1", s.pts[i]["r"], N)) for i in s.medium]
    return {"power": float(np.nanmean(pw)), "z": float(np.nanmedian(zs))}


def gate(s: Split, M: dict[str, Any], rep: str, k: int, N: int, cfg: dict[str, Any]) -> dict:
    g = cfg["gate"]
    m = M.get(key(rep, k, N))
    if not m:
        return {"defined": False}
    z0 = M[key("R0", 0, N)]["z_medium_median"]
    cmp = r1_comparator(s, rep, k, N)
    crit = {
        "G1": bool(m["fpr_pooled"] <= g["fpr_pooled_max"]
                   and m["fpr_point_ok"] >= g["fpr_point_fraction"]),
        "G2": bool(m["sdb_null"] <= g["sdb_null_max"]),
        "G3": bool(m["spearman"] >= g["spearman_min"]),
        "G4": bool(m["power_medium"] >= g["power_medium_min"]),
        "G5": bool(m["nonfinite"] <= g["nonfinite_max"]),
        "G6": bool(m["z_medium_median"] > 0
                   and m["z_medium_median"] >= g["snr_ratio_min"] * max(z0, 0.0)),
        "G7": bool(m["power_medium"] >= cmp["power"] + g["random_power_margin"]
                   and m["z_medium_median"] >= g["snr_ratio_min"] * max(cmp["z"], 0.0)),
    }  # fmt: skip
    return {"defined": True, "criteria": {c: bool(v) for c, v in crit.items()},
            "passed": int(sum(crit.values())), "all": bool(all(crit.values())),
            "z_R0": z0, "R1": cmp}  # fmt: skip


def select(s: Split, M: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for rep, k in CANDIDATES:
        gates = {N: gate(s, M, rep, k, N, cfg) for N in PRIMARY_N}
        nstar = next((N for N in PRIMARY_N if gates[N].get("all")), None)
        g1024 = gates[1024]
        m1024 = M.get(key(rep, k, 1024), {})
        mstar = M.get(key(rep, k, nstar), {}) if nstar else {}
        rows.append({"rep": rep, "k": k, "N_star": nstar,
                     "passed_1024": g1024.get("passed", -1),
                     "power_star": mstar.get("power_medium", np.nan),
                     "spearman_star": mstar.get("spearman", np.nan),
                     "power_1024": m1024.get("power_medium", np.nan),
                     "spearman_1024": m1024.get("spearman", np.nan),
                     "gates": {str(N): g for N, g in gates.items()}})  # fmt: skip

    def nz(v: float) -> float:
        return -v if np.isfinite(v) else np.inf

    passing = [r for r in rows if r["N_star"] is not None]
    if passing:
        best = min(
            passing,
            key=lambda r: (
                r["N_star"],
                nz(r["power_star"]),
                nz(r["spearman_star"]),
                ORDER[r["rep"]],
                r["k"],
            ),
        )
        rule = "smallest N* passing G1-G7"
    else:
        best = min(
            rows,
            key=lambda r: (
                -r["passed_1024"],
                nz(r["power_1024"]),
                nz(r["spearman_1024"]),
                ORDER[r["rep"]],
                r["k"],
            ),
        )
        rule = "no candidate passes at any N: most criteria at N = 1024"
    return {"selected": {"rep": best["rep"], "k": best["k"]}, "rule": rule,
            "table": [{k: v for k, v in r.items() if k != "gates"} for r in rows],
            "gates": {f"{r['rep']}|{r['k']}": r["gates"] for r in rows}}  # fmt: skip


# ------------------------------------------------------------------ theory predictions (design)
def _tr_full(o: dict[str, Any]) -> tuple[float, float]:
    tr1 = o["trS_perp_beh"] + o["kappa_delta"] * o["trLam"]
    tr2 = o["trS2_perp_beh"] + o["kappa_delta"] ** 2 * o["trLam2"]
    return tr1, tr2


def theory(s: Split, cfg: dict[str, Any]) -> dict[str, Any]:
    th = cfg["theory"]
    P = s.pts
    res: dict[str, Any] = {}
    # TP-R1: null SD law at clean nulls
    tp1: dict[str, list[float]] = {"R0": [], "R5": []}
    for i, N in product(s.clean, (256, 512, 1024)):
        n = N // cb.M
        o = P[i]["oracle"]
        for rep, tr2 in (("R0", _tr_full(o)[1]), ("R5", o["trS2_perp_beh"])):
            pred = np.sqrt(2 * tr2 / (n * (n - 1)))
            tp1[rep].append(np.sqrt(s.stat(i, key(rep, 0, N), "var")) / pred)
    lo, hi = th["tp_r1_band"]
    res["TP-R1"] = {
        rep: {
            "n": len(v),
            "median_ratio": float(np.median(v)),
            "frac_in_band": float(np.mean((np.array(v) >= lo) & (np.array(v) <= hi))),
            "pass": bool(
                np.mean((np.array(v) >= lo) & (np.array(v) <= hi)) >= th["tp_r1_fraction"]
            ),
        }
        for rep, v in tp1.items()
    }
    # TP-R2: dimension scaling d = 1024 vs 64 at matched clean nulls
    cl = {(P[i]["sid"], P[i]["alpha_target"], P[i]["spectrum"], P[i]["d"]): i for i in s.clean}
    r0, r5 = [], []
    for (sid, a, spec, d), i in cl.items():
        if d != 1024 or (sid, a, spec, 64) not in cl:
            continue
        j = cl[(sid, a, spec, 64)]
        pred = np.sqrt(_tr_full(P[i]["oracle"])[1] / _tr_full(P[j]["oracle"])[1])
        for N in (256, 512, 1024):
            sd = {rep: np.sqrt(s.stat(i, key(rep, 0, N), "var") / s.stat(j, key(rep, 0, N), "var"))
                  for rep in ("R0", "R5")}  # fmt: skip
            r0.append(sd["R0"] / pred)
            r5.append(sd["R5"])
    lo, hi = th["tp_r2_band"]
    f0 = float(np.mean((np.array(r0) >= lo) & (np.array(r0) <= hi)))
    f5 = float(np.mean((np.array(r5) >= lo) & (np.array(r5) <= hi)))
    res["TP-R2"] = {
        "n": len(r0),
        "R0_obs_over_pred_median": float(np.median(r0)),
        "R0_frac": f0,
        "R5_ratio_median": float(np.median(r5)),
        "R5_frac": f5,
        "pass": bool(f0 >= th["tp_r2_fraction"] and f5 >= th["tp_r2_fraction"]),
    }
    # TP-R3: random projection hurts (bulk + flat, in-S* medium/large, N >= 256)
    pts3 = [i for i in s.nonnull if P[i]["case"] == "dose" and P[i]["dose"] in ("medium", "large")
            and P[i]["spectrum"] in ("bulk", "flat")]  # fmt: skip
    med_ratio, cells, used = {}, [], 0
    for k in cb.K_GRID:
        ratios = []
        for i, N in product(pts3, (256, 512, 1024)):
            if k >= P[i]["d"]:
                continue
            z0 = s.z(i, key("R0", 0, N))
            z1 = s.z(i, key("R1", k, N))
            if not np.isfinite(z0) or z0 <= 1:
                continue
            used += 1
            ratios.append(z1 / z0)
            jm = s.match[i]
            if jm is None:
                continue
            tr1, tr2 = _tr_full(P[jm]["oracle"])
            deff = tr1**2 / tr2
            rho_n = np.sqrt(k * (k + deff)) / P[i]["d"]
            rho_s = s.stat(i, key("R1", k, N), "retention_mean")
            cells.append((z1 / z0) / (rho_s / rho_n))
        med_ratio[str(k)] = float(np.median(ratios)) if ratios else float("nan")
    lo, hi = th["tp_r3_band"]
    c3 = np.array(cells)
    frac3 = float(np.mean((c3 >= lo) & (c3 <= hi))) if len(c3) else float("nan")
    res["TP-R3"] = {"median_z_ratio_by_k": med_ratio, "cells": len(c3), "cells_used": used,
                    "obs_over_pred_median": float(np.median(c3)) if len(c3) else float("nan"),
                    "frac_in_band": frac3,
                    "pass": bool(all(v < 1 for v in med_ratio.values() if np.isfinite(v))
                                 and frac3 >= th["tp_r3_fraction"])}  # fmt: skip
    # TP-R4: oracle gain at in-S* medium points
    c4, sd4 = [], []
    for i, N in product([i for i in s.medium if P[i]["case"] == "dose"], (256, 512, 1024)):
        jm = s.match[i]
        if jm is None:
            continue
        o = P[jm]["oracle"]
        pred = np.sqrt(_tr_full(o)[1] / o["trS2_perp_beh"])
        z0, z5 = s.z(i, key("R0", 0, N)), s.z(i, key("R5", 0, N))
        v0, v5 = s.stat(jm, key("R0", 0, N), "var"), s.stat(jm, key("R5", 0, N), "var")
        sd4.append(np.sqrt(v0 / v5) / pred)
        if np.isfinite(z0) and z0 > 1:
            c4.append((z5 / z0) / pred)
    lo, hi = th["tp_r4_band"]
    a4 = np.array(c4)
    frac4 = float(np.mean((a4 >= lo) & (a4 <= hi))) if len(a4) else float("nan")
    res["TP-R4"] = {"cells": len(a4), "obs_over_pred_median": float(np.median(a4)) if len(a4)
                    else float("nan"), "frac_in_band": frac4,
                    "sd_ratio_obs_over_pred_median": float(np.median(sd4)),
                    "pass": bool(frac4 >= th["tp_r4_fraction"])}  # fmt: skip
    # TP-R5: ceiling power at N = 1024 (in-S* medium points)
    n = 1024 // cb.M
    pred, obs = [], []
    for i in [i for i in s.medium if P[i]["case"] == "dose"]:
        o = P[i]["oracle"]
        zc = o["tau_beh"] * np.sqrt(n * (n - 1))
        ratio = np.sqrt(1 + 2 * (n - 1) * o["cSc"] / o["trS2_perp_beh"])
        pred.append(norm.cdf((zc - norm.ppf(0.95)) / ratio))
        obs.append(s.stat(i, key("R5", 0, 1024), "reject"))
    res["TP-R5"] = {
        "predicted": float(np.mean(pred)),
        "observed": float(np.mean(obs)),
        "pass": bool(abs(np.mean(pred) - np.mean(obs)) <= th["tp_r5_tolerance"]),
    }
    # TP-R6: R2 retention ordering bulk >= flat >= spiked at N = 1024
    good, total, table = 0, 0, {}
    for d, k in product(env.D_GRID, cb.K_GRID):
        vals = {}
        for spec in env.SPECTRA:
            ix = [i for i in s.nonnull if P[i]["case"] == "dose" and P[i]["d"] == d
                  and P[i]["spectrum"] == spec]  # fmt: skip
            vals[spec] = float(np.nanmedian(s.col(ix, key("R2", k, 1024), "retention_mean")))
        if all(np.isfinite(v) for v in vals.values()):
            total += 1
            good += vals["bulk"] >= vals["flat"] >= vals["spiked"]
        table[f"d{d}|k{k}"] = vals
    res["TP-R6"] = {
        "cells": total,
        "ordered": good,
        "table": table,
        "pass": bool(total and good / total >= th["tp_r6_fraction"]),
    }
    return res


# ------------------------------------------------------------------ R4 sensitivity (design)
def sensitivity(s: Split, M: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for v in env.FUNCTIONAL_VARIANTS[1:]:
        rep = f"R4-{v}"
        for N in (256, 1024):
            ratio = []
            for i in s.nonnull:
                a, b = (
                    s.stat(i, key(rep, 0, N), "oracle_mean"),
                    s.stat(i, key("R4", 0, N), "oracle_mean"),
                )
                if b > 0:
                    ratio.append(a / b)
            m = M.get(key(rep, 0, N), {})
            out[f"{rep}|{N}"] = {
                "oracle_c2_ratio_to_main_median": float(np.median(ratio)) if ratio else np.nan,
                "fpr_pooled": m.get("fpr_pooled"),
                "fpr_nuisance_only": m.get("fpr_by_case", {}).get("nuis"),
                "power_medium": m.get("power_medium"),
                "spearman": m.get("spearman"),
            }
    out["oracle_spearman_R4_vs_beh"] = r4_oracle_spearman(s)
    return out


def r4_oracle_spearman(s: Split) -> float:
    P = s.pts
    f = [P[i]["oracle"]["C2_f"] for i in s.nonnull]
    b = [P[i]["oracle"]["C2_beh"] for i in s.nonnull]
    return float(spearmanr(f, b).statistic)


# ------------------------------------------------------------------ decision (test)
def decide(s: Split, M: dict[str, Any], sel: dict[str, Any], cfg: dict[str, Any]) -> dict:
    rep, k = sel["rep"], sel["k"]
    gates = {N: gate(s, M, rep, k, N, cfg) for N in PRIMARY_N}
    passing = [N for N in PRIMARY_N if gates[N].get("all")]
    mpb = min(passing) if passing else None
    dcfg = cfg["decision"]
    m1024 = M[key(rep, k, 1024)]
    if rep == "R4":
        retention = r4_oracle_spearman(s)
        retention_ok = retention >= dcfg["r4_oracle_spearman_min"]
    else:
        retention = m1024["retention_median"]
        retention_ok = retention >= dcfg["retention_min"]
    g7 = gates[1024].get("criteria", {}).get("G7", False)
    large = {r: max(M.get(key(r, kk, N), {}).get("power", {}).get("large", np.nan)
                    for N in PRIMARY_N) for r, kk in ((rep, k), ("R5", 0))}  # fmt: skip
    large_ok = any(v >= dcfg["large_power_min"] for v in large.values() if np.isfinite(v))
    if mpb is not None:
        label = "A — PRACTICALLY VIABLE GEOMETRY"
    elif not retention_ok or not g7:
        label = "C — REPRESENTATION FAILURE"
    elif large_ok:
        label = "B — MECHANISTICALLY VALID, NOT PRACTICALLY MEASURABLE"
    else:
        label = "C — REPRESENTATION FAILURE (fallback: large signal also undetectable)"
    return {"selected": sel, "min_practical_budget": mpb, "gates": {str(N): g for N, g in
                                                                    gates.items()},
            "retention_at_1024": retention, "retention_ok": bool(retention_ok),
            "G7_at_1024": bool(g7), "large_power_best": large, "large_ok": bool(large_ok),
            "label": label}  # fmt: skip


# ------------------------------------------------------------------ figures
def figures(s: Split, M: dict[str, Any], out_dir: Path, tag: str) -> None:
    P = s.pts
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    # (a) signal retention vs k at N = 1024 by spectrum
    for rep, ls in (("R1", ":"), ("R2", "-"), ("R3", "--")):
        for spec, col in zip(env.SPECTRA, ("C0", "C1", "C2"), strict=True):
            ix = [i for i in s.nonnull if P[i]["case"] == "dose" and P[i]["spectrum"] == spec]
            y = [np.nanmedian(s.col(ix, key(rep, k, 1024), "retention_mean")) for k in cb.K_GRID]
            ax[0, 0].plot(cb.K_GRID, y, ls, color=col, marker="o", label=f"{rep} {spec}")
    ax[0, 0].set(xscale="log", xlabel="k", ylabel="median retained signal", title=
                 "signal retention (N = 1024, in-S* points)")  # fmt: skip
    ax[0, 0].legend(fontsize=6, ncol=3)
    lines = [("R0", 0), ("R1", 8), ("R2", 8), ("R3", 8), ("R4", 0), ("R5", 0)]
    for rep, k in lines:  # (b) noise floor vs N; (e) medium power vs N
        xs = [N for N in cb.N_GRID if key(rep, k, N) in M]
        ax[0, 1].plot(xs, [M[key(rep, k, N)]["null_q95_median"] for N in xs], marker="o",
                      label=f"{rep}{'' if not k else f' k={k}'}")  # fmt: skip
        ax[1, 1].plot(xs, [M[key(rep, k, N)]["power_medium"] for N in xs], marker="o",
                      label=f"{rep}{'' if not k else f' k={k}'}")  # fmt: skip
    ax[0, 1].set(xscale="log", yscale="log", xlabel="N", title="null C^2 q95 (noise floor)")
    ax[1, 1].set(xscale="log", xlabel="N", ylim=(0, 1), title="medium-signal power")
    ax[1, 1].axhline(0.8, color="0.5", ls=":")
    ax[0, 1].legend(fontsize=7)
    for rep in ("R0", "R4", "R5"):  # (c) quality vs ambient d at N = 1024
        sd = [np.nanmedian([np.sqrt(s.stat(i, key(rep, 0, 1024), "var")) for i in s.clean
                            if P[i]["d"] == d]) for d in env.D_GRID]  # fmt: skip
        pw = [np.nanmean([s.stat(i, key(rep, 0, 1024), "reject") for i in s.medium
                          if P[i]["d"] == d]) for d in env.D_GRID]  # fmt: skip
        ax[0, 2].plot(env.D_GRID, sd, marker="o", label=f"{rep} null SD")
        ax[1, 0].plot(env.D_GRID, pw, marker="o", label=f"{rep} medium power")
    ax[0, 2].set(xscale="log", yscale="log", xlabel="ambient d", title="null SD vs d (N = 1024)")
    ax[1, 0].set(xscale="log", xlabel="ambient d", ylim=(0, 1), title="medium power vs d")
    ax[0, 2].legend(fontsize=7)
    ax[1, 0].legend(fontsize=7)
    for rep in ("R1", "R2", "R3"):  # (f) quality vs k at N = 1024
        ax[1, 2].plot(
            cb.K_GRID,
            [M.get(key(rep, k, 1024), {}).get("power_medium", np.nan) for k in cb.K_GRID],
            marker="o",
            label=f"{rep} power",
        )
        ax[1, 2].plot(cb.K_GRID, [M.get(key(rep, k, 1024), {}).get("spearman", np.nan)
                                  for k in cb.K_GRID], ls="--", marker="x",
                      label=f"{rep} Spearman")  # fmt: skip
    ax[1, 2].set(xscale="log", xlabel="k", ylim=(-0.1, 1), title="quality vs k (N = 1024)")
    ax[1, 2].legend(fontsize=7)
    fig.suptitle(f"E005a-R calibration ({tag}); reward level, identity metric, m = 8")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_e005ar_{tag}.png", dpi=120)
    plt.close(fig)


def main(argv: list[str]) -> int:
    mode, run_dir = argv[1], Path(argv[2]).resolve()
    cfg = provenance.load_config(CONFIG)
    d = load(run_dir, mode)
    s = Split(d)
    out_dir = provenance.create_run_dir(REPO / "results", f"E005aR-analysis-{mode}", REPO)
    provenance.write_metadata(
        out_dir,
        f"E005aR-analysis-{mode}",
        CONFIG,
        REPO,
        extra={
            "split": mode,
            "root_seed": env.ROOT_SEED,
            "calibration_run": str(run_dir.relative_to(REPO)),
        },
    )
    reps = ["R0", "R1", "R2", "R3", "R4", "R5"]
    if mode == "design":
        reps += [f"R4-{v}" for v in env.FUNCTIONAL_VARIANTS[1:]]
    M = all_metrics(s, reps)
    report: dict[str, Any] = {
        "metrics": M,
        "n_points": len(s.pts),
        "n": {
            "null": len(s.null),
            "clean": len(s.clean),
            "nonnull": len(s.nonnull),
            "medium": len(s.medium),
            **{f"dose_{k}": len(v) for k, v in s.by_dose.items()},
        },
    }
    if mode == "design":
        report["theory"] = theory(s, cfg)
        report["sensitivity"] = sensitivity(s, M)
        report["selection"] = select(s, M, cfg)
        (REPO / "configs/e005ar/e005ar_selection.json").write_text(
            json.dumps(report["selection"], indent=1, default=float) + "\n"
        )
    else:
        sel = json.loads((REPO / "configs/e005ar/e005ar_estimator_frozen.json").read_text())
        report["decision"] = decide(s, M, sel["selected"], cfg)
        report["all_candidates"] = select(s, M, cfg)["table"]
    figures(s, M, out_dir, mode)
    (out_dir / "analysis.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    brief = {k: v for k, v in report.items() if k not in ("metrics",)}
    print(json.dumps(brief, indent=1, default=float)[:6000])
    print(f"run directory: {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
