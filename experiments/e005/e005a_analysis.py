"""E005a analysis (research/07_e005a_measurement.md §2.7, §6-§8). Usage:
  e005a_analysis.py design <calibration-run-dir> <theory-run-dir>   -> metrics, TP1-TP4, SELECTION
  e005a_analysis.py test   <calibration-run-dir>                    -> held-out evaluation, budget

Every statistic follows §6. Selection (§7) uses only the design split (M_I, m = 8, N = 256) and
writes configs/e005/estimator_selection.json (the frozen estimator configuration is committed
separately). The test mode applies the §8 rules to the frozen selection.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005 import calibrate as cb  # noqa: E402
from vdyn.e005 import panel as pl  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e005" / "e005a.toml"
CANDIDATES = ("E0", "E1", "E2", "E3")
ALL_EST = ("E0", "E1", "E2", "E3", "plugin")


def load(run_dir: Path, split: str) -> dict[str, Any]:
    panel = json.loads((REPO / f"configs/e005/calibration_panel_{split}.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    info = json.loads((run_dir / "run_info.json").read_text())
    reps_path = run_dir / "reps.npz"
    assert hashlib.sha256(reps_path.read_bytes()).hexdigest() == info["reps_npz_sha256"]
    return {"points": panel["points"], "c_ref": panel["c_ref"], "summary": summary,
            "reps": np.load(reps_path)}  # fmt: skip


def key(metric: str, N: int, m: int, est: str) -> str:
    return f"{metric}|{N}|{m}|{est}"


def reps_matrix(d: dict[str, Any], pts: list[dict[str, Any]], k: str) -> np.ndarray:
    return np.stack([d["reps"][f"{p['pid']}::{k}"] for p in pts]).astype(float)


def per_point(d: dict[str, Any], pts: list[dict[str, Any]], k: str, q: str, f: str) -> np.ndarray:
    return np.array([d["summary"][p["pid"]].get(k, {}).get(q, {}).get(f, np.nan) for p in pts])


def config_metrics(d: dict[str, Any], metric: str, N: int, m: int, est: str) -> dict[str, Any]:
    pts = d["points"]
    k = key(metric, N, m, est)
    if f"{pts[0]['pid']}::{k}" not in d["reps"]:
        return {}
    truth = np.array([p["oracle"][metric]["C2"] for p in pts])
    null = np.array([p["dose"] == "null" for p in pts])
    nonnull = truth > 0
    out: dict[str, Any] = {}
    for q in ("A2", "alpha", "C2"):
        bias = per_point(d, pts, k, q, "bias")
        rmse = per_point(d, pts, k, q, "rmse")
        cov = per_point(d, pts, k, q, "coverage")
        und = per_point(d, pts, k, q, "undefined")
        tq = np.array([p["oracle"][metric][q] for p in pts])
        mean = per_point(d, pts, k, q, "mean")
        ok = np.isfinite(mean) & np.isfinite(tq)
        slope = float(np.polyfit(tq[ok], mean[ok], 1)[0]) if ok.sum() > 2 else float("nan")
        out[q] = {"median_abs_bias": float(np.nanmedian(np.abs(bias))),
                  "mean_bias": float(np.nanmean(bias)), "median_rmse": float(np.nanmedian(rmse)),
                  "mean_coverage": float(np.nanmean(cov)), "mean_undefined": float(np.nanmean(und)),
                  "calibration_slope": slope}  # fmt: skip
    # C^2-specific: nulls, ranking, matched sets, power, low-A stability
    X = reps_matrix(d, pts, k)  # (points, R)
    c_ref2 = d["c_ref"] ** 2
    fpr = per_point(d, pts, k, "C2", "reject")
    q95 = per_point(d, pts, k, "C2", "q95")
    sd = np.sqrt(per_point(d, pts, k, "C2", "var"))
    mean_c2 = per_point(d, pts, k, "C2", "mean")
    rho = [spearmanr(X[nonnull, r], truth[nonnull]).statistic for r in range(X.shape[1])
           if np.isfinite(X[nonnull, r]).all()]  # fmt: skip
    sets: dict[str, list[int]] = {}
    for i, p in enumerate(pts):
        if p["matched"]:
            sets.setdefault(p["matched"], []).append(i)
    sdb_terms = []
    for idx in sets.values():
        if len(idx) >= 3:
            sdb_terms.append(np.nanstd(mean_c2[idx], ddof=1) / np.nanmedian(sd[idx]))
    null_sets = {s: i for s, i in sets.items() if s.startswith("null")}
    low = np.array([p["variant"] == "theta_lowA" for p in pts])
    extreme = np.abs(X[low]) > 10 * c_ref2
    dose = np.array([p["dose"] for p in pts])
    out["C2"].update({
        "null_fpr_mean": float(np.nanmean(fpr[null])),
        "null_fpr_abs_err": float(np.nanmean(np.abs(fpr[null] - 0.05))),
        "null_fpr_in_band": float(np.nanmean((fpr[null] >= 0.02) & (fpr[null] <= 0.10))),
        "null_q95_median": float(np.nanmedian(q95[null])),
        "null_q95_sd_across_structures": float(np.nanmean(
            [np.nanstd(q95[i], ddof=1) for i in null_sets.values() if len(i) >= 3])),
        "null_bias_by_mechanism": {mech: float(np.nanmean(mean_c2[null & np.array(
            [p["mechanism"] == mech for p in pts])])) for mech in pl_mechs(pts)},
        "null_q95_by_d_extra": {str(de): float(np.nanmedian(q95[null & np.array(
            [p["d_extra"] == de for p in pts])])) for de in pl.D_EXTRA},
        "spearman_nonnull": float(np.mean(rho)) if rho else float("nan"),
        "SDB": float(np.sqrt(np.nanmean(np.square(sdb_terms)))) if sdb_terms else float("nan"),
        "rmse_rel": float(np.nanmedian(per_point(d, pts, k, "C2", "rmse")) / c_ref2),
        "lowA_unstable": float(np.mean(~np.isfinite(X[low]) | extreme)) if low.any() else 0.0,
        "power_small": float(np.nanmean(fpr[dose == "small"])),
        "power_vsmall": float(np.nanmean(fpr[dose == "vsmall"])),
    })  # fmt: skip
    return out


def pl_mechs(pts: list[dict[str, Any]]) -> list[str]:
    return sorted({p["mechanism"] for p in pts})


def all_metrics(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in pl.METRICS:
        for N in cb.N_GRID:
            for m in cb.M_GRID:
                for est in ALL_EST:
                    r = config_metrics(d, metric, N, m, est)
                    if r:
                        out[key(metric, N, m, est)] = r
    return out


def select(metrics: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    s = cfg["selection"]
    tol = s["tolerances"]
    at = {e: metrics[key(s["metric"], s["N_ref"], s["m"], e)]["C2"] for e in CANDIDATES}
    trail = []
    keep = list(CANDIDATES)
    steps = (
        ("SDB", "min", tol["sdb_rel"], "rel"),
        ("null_fpr_abs_err", "min", tol["null_abs"], "abs"),
        ("spearman_nonnull", "max", tol["spearman_abs"], "abs"),
        ("rmse_rel", "min", tol["rmse_rel"], "rel"),
        ("lowA_unstable", "min", 0.0, "abs"),
    )
    for crit, sense, t, kind in steps:
        vals = {e: at[e][crit] for e in keep}
        best = min(vals.values()) if sense == "min" else max(vals.values())
        if sense == "min":
            lim = best * (1 + t) if kind == "rel" else best + t
            keep = [e for e in keep if vals[e] <= lim + 1e-15]
        else:
            keep = [e for e in keep if vals[e] >= best - t - 1e-15]
        trail.append({"criterion": crit, "values": vals, "kept": list(keep)})
    chosen = min(keep, key=s["order"].index)
    return {"selected": chosen, "trail": trail, "setting": s}


def theory_checks(d: dict[str, Any], theory: list[dict[str, Any]]) -> dict[str, Any]:
    """TP1-TP4 (§2.7) on the design split, identity metric."""
    pts = {p["pid"]: p for p in d["points"]}
    res: dict[str, Any] = {}
    tp1, tp2, tp3 = [], [], []
    for t in theory:
        p = pts[t["pid"]]
        for N in (128, 256, 512, 1024):
            n = N // t["m"]
            s = d["summary"][t["pid"]]
            plug = s[key("I", N, t["m"], "plugin")]["C2"]
            e1 = s[key("I", N, t["m"], "E1")]["C2"]
            pred_p, pred_u = t["plugin_n1"] / n, t["ustat_n1"] / n
            tp1.append(plug["bias"] / pred_p)
            mcse_u = np.sqrt(e1["var"] / cb.R_REPS)
            if abs(pred_u) > 3 * mcse_u:
                tp2.append(e1["bias"] / pred_u)
            tp3.append({"pid": t["pid"], "N": N, "m": t["m"], "d_extra": p["d_extra"],
                        "plugin_bias": plug["bias"], "pred": pred_p, "e1_bias": e1["bias"],
                        "e1_mcse": float(mcse_u)})  # fmt: skip
    tp1a, tp2a = np.array(tp1), np.array(tp2)
    in1 = float(np.mean((tp1a >= 0.8) & (tp1a <= 1.25)))
    res["TP1"] = {"n": len(tp1a), "frac_in_[0.8,1.25]": in1,
                  "median_ratio": float(np.median(tp1a)), "pass": bool(in1 >= 0.9)}  # fmt: skip
    in2 = float(np.mean((tp2a >= 0.7) & (tp2a <= 1.4))) if len(tp2a) else float("nan")
    med2 = float(np.median(tp2a)) if len(tp2a) else float("nan")
    res["TP2"] = {"n": len(tp2a), "frac_in_[0.7,1.4]": in2, "median_ratio": med2,
                  "pass": bool(len(tp2a) and in2 >= 0.8)}  # fmt: skip
    # TP3: pair each d_extra = 56 null with its d_extra = 0 twin
    by = {(r["pid"].replace("-d56", "").replace("-d0", ""), r["N"], r["m"], r["d_extra"]): r
          for r in tp3}  # fmt: skip
    ratios, e1_same = [], []
    for (pid, N, m, de), r in by.items():
        if de != 56 or (pid, N, m, 0) not in by:
            continue
        r0 = by[(pid, N, m, 0)]
        ratios.append((r["plugin_bias"] - r0["plugin_bias"]) / (r["pred"] - r0["pred"]))
        e1_same.append(
            abs(r["e1_bias"] - r0["e1_bias"]) <= 3 * np.hypot(r["e1_mcse"], r0["e1_mcse"])
        )
    ra = np.array(ratios)
    in3 = float(np.mean((ra >= 0.8) & (ra <= 1.25)))
    same = float(np.mean(e1_same))
    res["TP3"] = {"n": len(ra), "frac_ratio_in_[0.8,1.25]": in3,
                  "median_ratio": float(np.median(ra)), "e1_unchanged_frac": same,
                  "pass": bool(in3 >= 0.9 and same >= 0.9)}  # fmt: skip
    # TP4: E2, E3 second-order unbiased where A^2 > 10 sqrt(tr S_G / n). tr(S_G)/n is estimated
    # without bias by mean(plug-in A^2) - mean(U-statistic A^2) (an exact identity per replication)
    ok: dict[str, list[bool]] = {"E2": [], "E3": []}
    for p in d["points"]:
        for N in (256, 512, 1024):
            for m in cb.M_GRID:
                s = d["summary"][p["pid"]]
                tr_n = (
                    s[key("I", N, m, "plugin")]["A2"]["mean"]
                    - s[key("I", N, m, "E1")]["A2"]["mean"]
                )
                if p["oracle"]["I"]["A2"] <= 10 * np.sqrt(max(tr_n, 0.0)):
                    continue
                for e in ("E2", "E3"):
                    c = s[key("I", N, m, e)]["C2"]
                    ok[e].append(abs(c["bias"]) <= 3 * np.sqrt(c["var"] / cb.R_REPS))
    res["TP4"] = {e: {"n": len(v), "frac_within_3mcse": float(np.mean(v)),
                      "pass": bool(np.mean(v) >= 0.9)} for e, v in ok.items()}  # fmt: skip
    return res


def solved_by_N(metrics: dict[str, Any], est: str, cfg: dict[str, Any]) -> dict[str, Any]:
    sv = cfg["solved"]
    out: dict[str, Any] = {}
    for N in cb.N_GRID:
        c = metrics.get(key("I", N, 8, est), {}).get("C2")
        if not c:
            continue
        solved = (c["SDB"] <= sv["sdb_max"] and c["null_fpr_in_band"] >= sv["null_fraction"]
                  and c["spearman_nonnull"] >= sv["spearman_min"])  # fmt: skip
        out[str(N)] = {"SDB": c["SDB"], "null_fpr_in_band": c["null_fpr_in_band"],
                       "spearman": c["spearman_nonnull"], "power_small": c["power_small"],
                       "solved": bool(solved),
                       "practical": bool(solved and c["power_small"] >= sv["power_min"]),
                       }  # fmt: skip
    prac = [int(N) for N, v in out.items() if v["practical"]]
    return {"by_N": out, "min_practical_budget": min(prac) if prac else None}


def figures(metrics: dict[str, Any], out_dir: Path, tag: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for est in ALL_EST:
        xs = [N for N in cb.N_GRID if key("I", N, 8, est) in metrics]
        c = [metrics[key("I", N, 8, est)]["C2"] for N in xs]
        ls = "--" if est == "plugin" else "-"
        axes[0].plot(xs, [v["null_q95_median"] for v in c], ls, marker="o", label=est)
        axes[1].plot(xs, [v["SDB"] for v in c], ls, marker="o", label=est)
        axes[2].plot(xs, [v["null_fpr_mean"] for v in c], ls, marker="o", label=est)
        axes[3].plot(xs, [v["spearman_nonnull"] for v in c], ls, marker="o", label=est)
    titles = ("null C^2 95th percentile (noise floor)", "SDB (between-structure shift / SD)",
              "null false-positive rate", "Spearman (C > 0)")  # fmt: skip
    for ax, t in zip(axes, titles, strict=True):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("gold audit size N (m = 8)")
        ax.set_title(t, fontsize=10)
    axes[0].set_yscale("symlog", linthresh=1e-7)
    axes[2].axhline(0.05, color="0.5", ls=":")
    axes[1].axhline(0.5, color="0.5", ls=":")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"E005a calibration ({tag}), identity metric")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_calibration_{tag}.png", dpi=130)
    plt.close(fig)


def main(argv: list[str]) -> int:
    mode, run_dir = argv[1], Path(argv[2]).resolve()
    cfg = provenance.load_config(CONFIG)
    d = load(run_dir, mode)
    out_dir = provenance.create_run_dir(REPO / "results", f"E005a-analysis-{mode}", REPO)
    extra: dict[str, Any] = {"split": mode, "root_seed": pl.ROOT_SEED,
                             "calibration_run": str(run_dir.relative_to(REPO))}  # fmt: skip
    provenance.write_metadata(out_dir, f"E005a-analysis-{mode}", CONFIG, REPO, extra=extra)
    metrics = all_metrics(d)
    report: dict[str, Any] = {"metrics": metrics, "c_ref": d["c_ref"]}
    if mode == "design":
        theory = json.loads((Path(argv[3]).resolve() / "theory.json").read_text())
        report["theory"] = theory_checks(d, theory)
        report["selection"] = select(metrics, cfg)
        report["solved_design"] = {e: solved_by_N(metrics, e, cfg) for e in CANDIDATES}
        (REPO / "configs/e005/estimator_selection.json").write_text(
            json.dumps(report["selection"], indent=1) + "\n"
        )
    else:
        sel = json.loads((REPO / "configs/e005/estimator_frozen.json").read_text())["selected"]
        report["selected"] = sel
        report["solved_test"] = {e: solved_by_N(metrics, e, cfg) for e in CANDIDATES}
        mpb = report["solved_test"][sel]["min_practical_budget"]
        report["min_practical_budget"] = mpb
        ok = mpb is not None and mpb <= cfg["solved"]["max_budget"]
        report["recommendation"] = (
            "PROCEED TO E005b" if ok else "REVISE MEASUREMENT THEORY BEFORE NEURAL EXPERIMENTS"
        )
    figures(metrics, out_dir, mode)
    (out_dir / "analysis.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("metrics",)}, indent=1,
                     default=float)[:4000])  # fmt: skip
    print(f"run directory: {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
