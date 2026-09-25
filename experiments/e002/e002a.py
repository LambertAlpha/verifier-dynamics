"""E002a (registry E002 §10 + Amendment 2 §5): estimator characterization, formal instrumentation.

Fixed non-panel structures at each panel's design-selected FPR; no panel structure and no test-split
data are touched. `--reps` defaults to `reps_formal` (the formal run needs explicit approval); a
smoke test passes a small `--reps`, `--n-grid` and a scratch `--results-root`.

Changes from the pilot script (`e002a_pilot.py`, kept unchanged as the pilot's code):
- every detection structure Y has a dimension-matched null Y0 (`bf.matched_null`); the threshold is
  the 95% quantile of C_hat^2 on one Y0 sample and the false alarm is read on an independent one;
- the coverage set adds dose rho in {0.05, 0.1} and the matched nulls (C = 0);
- the U-statistic check is on the raw Gram entries, on cells with N (1 - q) f >= 5;
- degenerate-event rates use exact two-sided binomial tests (p < 0.0027).
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logit
from scipy.stats import binom, binomtest

from vdyn import provenance
from vdyn.e002 import estimators as est
from vdyn.e002 import panel as pn
from vdyn.geometry import autodiff
from vdyn.geometry.decompose import decompose
from vdyn.policies import bernoulli
from vdyn.verifiers import boolean_fp as bf

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e002" / "e002.toml"
ESTIMATORS = {  # name: (estimator, metric, lam)
    "plugin-exactF": ("plugin", "exact", 0.0),
    "plugin-estF-1e-3": ("plugin", "est", 1e-3),
    "plugin-estF-1e-2": ("plugin", "est", 1e-2),
    "plugin-estF-1e-1": ("plugin", "est", 1e-1),
    "plugin-euclid": ("plugin", "euclid", 0.0),
    "ustat-exactF": ("ustat", "exact", 0.0),
    "ustat-estF-1e-2": ("ustat", "est", 1e-2),
}
DETECT = ("SINGLE", "AND2-sym", "AND2-asym", "OR2-sym")
COVERAGE_KEYS = ("SINGLE", "AND2-sym", "OR2-sym", "RFP", "dose-rho=0.05", "dose-rho=0.1",
                 "null|SINGLE", "null|AND2-sym", "null|OR2-sym")  # fmt: skip
COVERAGE_N = (64, 256, 1024)
COVERAGE_EST = ("plugin-exactF", "plugin-estF-1e-2")
SIG = 0.0027  # two-sided 3-SE equivalent


def structures(
    q0: float, f: float, cfg: dict[str, Any]
) -> dict[str, tuple[bf.EventStructure, float]]:
    raw = [float(x) for x in logit(np.array(cfg["asym_raw_probs"]))]
    asym = pn.match({"sid": "AND2-asym", "type": "AND2", "base": None, "raw_logits": raw,
                     "rho": None}, f)  # fmt: skip
    out = {
        "SINGLE": (bf.EventStructure("SINGLE", "SINGLE", (f,)), q0),
        "AND2-sym": (bf.EventStructure("AND2-sym", "AND2", (np.sqrt(f),) * 2), q0),
        "AND2-asym": (asym, q0),
        "OR2-sym": (bf.EventStructure("OR2-sym", "OR2", (1 - np.sqrt(1 - f),) * 2), q0),
        "RFP": (bf.EventStructure("RFP", "RFP", (), coin=f), q0),
    }
    for rho in cfg["rho_dose"]:
        if rho == 0.0:
            out["dose-rho=0"] = (bf.EventStructure("dose0", "RFP", (), coin=f), q0)
        else:
            s_e, coin = 1 - (1 - f) ** rho, 1 - (1 - f) ** (1 - rho)
            st = bf.EventStructure(f"dose{rho}", "MIX", (s_e,), coin=coin, base="SINGLE", rho=rho)
            out[f"dose-rho={rho:g}"] = (st, q0)
    for q in (*cfg["q_extreme"], q0):
        out[f"Aext-q={q:g}"] = (bf.EventStructure(f"aext{q}", "SINGLE", (f,)), q)
    for key in [*DETECT, *(k for k in out if k.startswith("dose-") and k != "dose-rho=0")]:
        st, q = out[key]
        out[f"null|{key}"] = (bf.matched_null(st, f), q)
    return out


def truth(st: bf.EventStructure, q: float) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    lp = bernoulli.product_log_prob(1 + st.n_features)
    theta = np.concatenate([[logit(q)], logit(np.asarray(st.s0))])
    g_gold = autodiff.reward_gradient(lp, theta, bf.gold_table(st))
    g_ver = autodiff.reward_gradient(lp, theta, bf.verifier_table(st))
    fisher = autodiff.fisher(lp, theta)
    res: dict[str, dict[str, float]] = {}
    for metric, m in (("fisher", np.linalg.inv(fisher)), ("euclid", np.eye(len(theta)))):
        d = decompose(g_gold, g_ver, m)
        g_e = g_ver - g_gold
        res[metric] = {"A": d.A, "alpha": d.alpha, "C": d.C, "ge2": float(g_e @ m @ g_e),
                       "P": float(g_e @ m @ g_gold)}  # fmt: skip
    return res, fisher


def run_estimator(name: str, st, q, batch, fisher) -> dict[str, np.ndarray]:
    estimator, metric, lam = ESTIMATORS[name]
    s = np.asarray(st.s0)
    if metric == "euclid":
        sc = est.score_vectors(q, s, batch.corr, batch.z)
        m = np.broadcast_to(np.eye(sc.shape[2]), (sc.shape[0], sc.shape[2], sc.shape[2]))
        g_gold = est.rloo(batch.corr.astype(float), sc)
        return est.decompose_batch(g_gold, est.rloo(batch.v, sc), m)
    exact = fisher if metric == "exact" else None
    return est.geometry(st, batch, None, q, s, estimator, lam, "paired", exact_fisher=exact)


def gram_form(out: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if "P" in out:
        return out["P"], out["D"]
    a2 = out["A"] ** 2
    return np.where(out["alpha_defined"], out["alpha"] * a2, 0.0), a2 * out["C"] ** 2


def bootstrap_coverage(name, st, q, batch, fisher, c_true, n_boot, rng) -> float:
    reps, n = batch.corr.shape
    boot_vals = np.empty((n_boot, reps))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=(reps, n))
        resampled = est.Batch(
            np.take_along_axis(batch.corr, idx, axis=1),
            np.take_along_axis(batch.z, idx[..., None], axis=1),
            np.take_along_axis(batch.v, idx, axis=1),
        )
        boot_vals[b] = run_estimator(name, st, q, resampled, fisher)["C"]
    lo, hi = np.quantile(boot_vals, [0.025, 0.975], axis=0)
    return float(np.mean((lo <= c_true) & (c_true <= hi)))


def _stat(x: np.ndarray, true: float, reps: int) -> dict[str, float]:
    return {"bias": float(np.mean(x) - true), "mcse": float(np.std(x) / np.sqrt(reps))}


def cell_task(args: tuple[Any, ...]) -> dict[str, Any]:
    """All estimators for one (structure, N) cell."""
    op_i, s_i, key, st, q, n, reps, seed, fisher, tr, n_boot, f = args
    s0 = np.asarray(st.s0)
    rng = np.random.default_rng([seed, 10, op_i, s_i, n])
    batch = est.sample_rollouts(st, q, s0, n, reps, rng)
    fpr0 = bf.fpr(st, s0) if st.n_features else st.coin
    pred = est.degenerate_event_probabilities(q, fpr0, s0, n)
    pred["A_hat_zero"] = pred["all_gold_equal"]
    obs_counts = {
        "all_gold_equal": int(np.sum(np.ptp(batch.corr, axis=1) == 0)),
        "no_false_positive": int(np.sum(((1 - batch.corr) * batch.v).sum(axis=1) == 0)),
    }
    if st.n_features:
        obs_counts["some_feature_constant"] = int(np.sum(np.any(np.ptp(batch.z, axis=1) == 0, 1)))
    rows: dict[str, Any] = {}
    c2: dict[str, list[float]] = {}
    for name in ESTIMATORS:
        out = run_estimator(name, st, q, batch, fisher)
        tm = tr["euclid" if name == "plugin-euclid" else "fisher"]
        defined = out["alpha_defined"]
        p_hat, d_hat = gram_form(out)
        row: dict[str, Any] = {
            "C_true": tm["C"], "A_true": tm["A"], "alpha_true": tm["alpha"],
            "A": _stat(out["A"], tm["A"], reps), "C": _stat(out["C"], tm["C"], reps),
            "C2": _stat(out["C2"], tm["C"] ** 2, reps), "P": _stat(p_hat, tm["P"], reps),
            "D": _stat(d_hat, tm["A"] ** 2 * tm["C"] ** 2, reps),
            "C_sd": float(np.std(out["C"])),
            "C_rmse": float(np.sqrt(np.mean((out["C"] - tm["C"]) ** 2))),
            "alpha_undefined_rate": float(1 - defined.mean()),
            "alpha_bias_defined": float(np.nanmean(out["alpha"]) - tm["alpha"]) if defined.any()
            else None,
        }  # fmt: skip
        if name.startswith("ustat"):
            for g, true in (
                ("gram_gg", tm["A"] ** 2),
                ("gram_eg", tm["P"]),
                ("gram_ee", tm["ge2"]),
            ):
                row[g] = _stat(out[g], true, reps)
            row["gram_eligible"] = bool(n * (1 - q) * f >= 5)
            row["gold_eligible"] = bool(n * q >= 5)  # secondary stratum, not a registered rule
        if name == "plugin-exactF":
            obs_counts["A_hat_zero"] = int(np.sum(out["A"] == 0))
        if n in COVERAGE_N and name in COVERAGE_EST and key in COVERAGE_KEYS:
            row["coverage95_C"] = bootstrap_coverage(name, st, q, batch, fisher, tm["C"], n_boot,
                                                     rng)  # fmt: skip
        rows[name] = row
        c2[name] = out["C2"].tolist()
    degenerate = {}
    for e, count in obs_counts.items():
        p = min(max(pred[e], 0.0), 1.0)
        pv = (
            1.0
            if p in (0.0, 1.0) and count == round(p * reps)
            else binomtest(count, reps, p).pvalue
        )
        degenerate[e] = {"observed": count / reps, "predicted": p, "p_value": float(pv)}
    fa_c2: dict[str, list[float]] = {}
    if key.startswith("null|") or key == "RFP":  # independent sample for the false alarm
        rng_fa = np.random.default_rng([seed, 12, op_i, s_i, n])
        fa_batch = est.sample_rollouts(st, q, s0, n, reps, rng_fa)
        fa_c2 = {name: run_estimator(name, st, q, fa_batch, fisher)["C2"].tolist()
                 for name in ESTIMATORS}  # fmt: skip
    return {"key": key, "n": n, "rows": rows, "degenerate": degenerate, "c2": c2, "fa_c2": fa_c2}


def detection(
    cells: dict[tuple[str, int], dict[str, Any]], truths: dict[str, Any], n: int, name: str,
    dose_keys: list[str],
) -> dict[str, Any]:  # fmt: skip
    """Matched-null detection (Amendment 2 §5) plus the pilot's d = 1 RFP null for continuity."""

    def against(y_key: str, null_key: str) -> dict[str, float]:
        null = np.array(cells[(null_key, n)]["c2"][name])
        fa = np.array(cells[(null_key, n)]["fa_c2"][name])
        y = np.array(cells[(y_key, n)]["c2"][name])
        thr = float(np.quantile(null, 0.95))
        return {
            "threshold": thr,
            "false_alarm": float(np.mean(fa > thr)),
            "power": float(np.mean(y > thr)),
            "P(C_Y>C_Y0)": float(
                np.mean(y[:, None] > fa[None, :]) + 0.5 * np.mean(y[:, None] == fa[None, :])
            ),  # fmt: skip
        }

    out: dict[str, Any] = {"matched": {k: against(k, f"null|{k}") for k in DETECT}}
    out["legacy_rfp_null"] = {k: against(k, "RFP") for k in DETECT}
    dose_c, dose_pow = [], []
    for k in dose_keys:
        dose_c.append(truths[k]["fisher"]["C"])
        res = against(k, "RFP" if k == "dose-rho=0" else f"null|{k}")
        dose_pow.append(res["false_alarm"] if k == "dose-rho=0" else res["power"])
        out["matched"][k] = res
    order = np.argsort(dose_c)
    dc, dp = np.array(dose_c)[order], np.array(dose_pow)[order]
    above = np.flatnonzero(dp >= 0.8)
    mdc: float | None = None
    if len(above) and above[0] == 0:
        mdc = float(dc[0])
    elif len(above):
        j = above[0]
        mdc = float(np.interp(0.8, [dp[j - 1], dp[j]], [dc[j - 1], dc[j]]))
    out["dose_C"], out["dose_power"], out["min_detectable_C"] = dc.tolist(), dp.tolist(), mdc
    return out


def summarize(report_op: dict[str, Any], reps: int) -> dict[str, Any]:
    """Registered checks (E002 §10 as restated by Amendment 2 §5); counts, not verdict words."""
    cells = report_op["cells"]
    deg = [(c, e, v) for c, d in cells.items() for e, v in d["degenerate"].items()]
    flagged = [(c, e) for c, e, v in deg if v["p_value"] < SIG]
    gram: dict[str, dict[str, int]] = {}
    gram_gold: dict[str, dict[str, int]] = {}  # [secondary] also N q >= 5 (rare-gold cells out)
    for d in cells.values():
        row = d["rows"]["ustat-exactF"]
        if not row["gram_eligible"]:
            continue
        for g in ("gram_gg", "gram_eg", "gram_ee"):
            hit = int(abs(row[g]["bias"]) > 3 * row[g]["mcse"])
            zero_sd = int(row[g]["mcse"] == 0)
            strata = [gram] + ([gram_gold] if row["gold_eligible"] else [])
            for stratum in strata:
                s = stratum.setdefault(g, {"cells": 0, "flagged": 0, "zero_sd_cells": 0})
                s["cells"] += 1
                s["flagged"] += hit
                s["zero_sd_cells"] += zero_sd
    band = 3 * np.sqrt(2 * 0.05 * 0.95 / reps)
    fa_rows = []
    for k, det in report_op["detection"].items():
        for y, res in det["matched"].items():
            if y != "dose-rho=0":  # the d = 1 RFP null is kept for continuity only
                fa_rows.append((k, y, res["false_alarm"]))
    fa_out = [r for r in fa_rows if abs(r[2] - 0.05) > band]
    near0 = {k: {n: cells[f"{k}|N={n}"]["rows"]["plugin-exactF"]["C"]["bias"]
                 for n in (16, 64, 256, 1024) if f"{k}|N={n}" in cells}
             for k in ("null|SINGLE", "null|AND2-sym", "null|OR2-sym")}  # fmt: skip
    return {
        "degenerate": {"checks": len(deg), "flagged": len(flagged),
                       "expected_by_chance": len(deg) * SIG,
                       "binomial_tail_p": float(binom.sf(len(flagged) - 1, len(deg), SIG)),
                       "flagged_cells": flagged[:20]},
        "gram_unbiased_ustat_exactF": {g: {**s, "expected_by_chance": s["cells"] * SIG}
                                       for g, s in gram.items()},
        "gram_unbiased_secondary_gold_eligible": {g: {**s, "expected_by_chance": s["cells"] * SIG}
                                                  for g, s in gram_gold.items()},
        "matched_null_false_alarm": {"band": float(band), "cells": len(fa_rows),
                                     "outside_band": len(fa_out), "examples": fa_out[:20]},
        "plugin_C_bias_at_C0_matched_nulls": near0,
    }  # fmt: skip


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets_dir")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--n-grid", default=None)
    ap.add_argument("--results-root", default=str(REPO / "results"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv[1:])
    config = provenance.load_config(CONFIG)
    cfg = config["e002a"]
    reps = args.reps or cfg["reps_formal"]
    n_grid = [int(x) for x in args.n_grid.split(",")] if args.n_grid else cfg["n_grid"]
    seed = cfg["formal_seed"]
    targets_dir = Path(args.targets_dir).resolve()
    label = "E002a" if reps == cfg["reps_formal"] and not args.n_grid else "E002a-smoke"
    run_dir = provenance.create_run_dir(Path(args.results_root), label, REPO)
    provenance.write_metadata(run_dir, label, CONFIG, REPO,
                              extra={"reps": reps, "n_grid": n_grid, "seed": seed})  # fmt: skip
    report: dict[str, Any] = {"reps": reps, "n_grid": n_grid}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for op_i, op in enumerate(config["operating_points"]):
            f = json.loads((targets_dir / f"design_targets_{op}.json").read_text())["f_selected"]
            q0 = config["operating_points"][op]["q0"]
            structs = structures(q0, f, cfg)
            solved = {k: truth(st, q) for k, (st, q) in structs.items()}
            tasks = [
                (
                    op_i,
                    s_i,
                    k,
                    st,
                    q,
                    n,
                    reps,
                    seed,
                    solved[k][1],
                    solved[k][0],
                    cfg["bootstrap"],
                    f,
                )
                for s_i, (k, (st, q)) in enumerate(structs.items())
                for n in n_grid
            ]
            results = list(pool.map(cell_task, tasks, chunksize=1))
            cells = {(r["key"], r["n"]): r for r in results}
            truths = {k: v[0] for k, v in solved.items()}
            dose_keys = [k for k in structs if k.startswith("dose-")]
            det = {f"N={n}|{name}": detection(cells, truths, n, name, dose_keys)
                   for n in n_grid for name in ESTIMATORS}  # fmt: skip
            f4 = {}
            for key in ("SINGLE", "AND2-sym"):
                st, q = structs[key]
                s0 = np.asarray(st.s0)
                rng = np.random.default_rng([seed, 14, op_i])
                batch = est.sample_rollouts(st, q, s0, 200_000, 1, rng)
                sc = est.score_vectors(q, s0, batch.corr, batch.z)[0]
                gam = (batch.v - batch.corr)[0][:, None] * sc
                minv = np.linalg.inv(solved[key][1])
                g_e = gam.mean(axis=0)
                per = np.einsum("ni,ij,nj->n", gam - g_e, minv, gam - g_e)
                s_e, k2 = bf.event_prob(st, s0), bf.kappa2(st, s0)
                formula = (1 - q) * k2 / s_e + s_e * q - truths[key]["fisher"]["ge2"]
                mcse = float(per.std() / np.sqrt(len(per)))
                f4[key] = {"empirical": float(per.mean()), "mcse": mcse, "formula": formula,
                           "z": float((per.mean() - formula) / mcse)}  # fmt: skip
            op_out = {
                "f": f, "q0": q0,
                "cells": {f"{k}|N={n}": {kk: v for kk, v in r.items() if kk not in ("c2", "fa_c2")}
                          for (k, n), r in cells.items()},
                "detection": det, "f4_noise": f4,
            }  # fmt: skip
            op_out["checks"] = summarize(op_out, reps)
            report[op] = op_out
            print(f"{op}: {len(cells)} cells; checks {json.dumps(op_out['checks'])[:600]}")
    report["seconds"] = time.perf_counter() - start
    (run_dir / "e002a.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
