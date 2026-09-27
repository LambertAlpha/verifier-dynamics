"""E005a POST-HOC diagnostics (exploratory; NOT registered, gate nothing). Usage:
  e005a_posthoc.py design|test <calibration-run-dir>

Why the registered criteria fail, for E0 / E2 / E3 under M_I, m = 8, at every N:
  1. power by dose (null / vsmall / small / medium / natural);
  2. the noise floor in C units (sqrt of the median null 95th percentile / C_ref) and its N-scaling;
  3. the null test: jackknife SE vs the Monte Carlo SD at C = 0, the q95 / (1.645 SE) ratio, and
     the false-positive rate a Wald test would have with the Monte Carlo SD in place of the SE;
  4. ranking among natural-dose points only, and among points above the floor;
  5. TP4 with the dimensionally consistent condition A^2 > 10 tr(S_G)/n (registered text used
     sqrt(tr(S_G)/n), which compares A^2 with a quantity of units A; see the registry deviation).
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e005a_analysis as an  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005 import calibrate as cb  # noqa: E402
from vdyn.e005 import panel as pl  # noqa: E402

REPO = an.REPO
DOSES = ("null", "vsmall", "small", "medium", "natural")
EST = ("E0", "E2", "E3")


def diagnostics(d: dict[str, Any], est: str, N: int) -> dict[str, Any]:
    pts = d["points"]
    k = an.key("I", N, 8, est)
    truth = np.array([p["oracle"]["I"]["C2"] for p in pts])
    dose = np.array([p["dose"] for p in pts])
    null = dose == "null"
    X = an.reps_matrix(d, pts, k)
    rej = an.per_point(d, pts, k, "C2", "reject")
    sd = np.sqrt(an.per_point(d, pts, k, "C2", "var"))
    se = an.per_point(d, pts, k, "C2", "mean_se")
    q95 = an.per_point(d, pts, k, "C2", "q95")
    c_ref2 = d["c_ref"] ** 2
    floor = float(np.nanmedian(q95[null]))
    wald_mc = np.nanmean(X[null] - 1.645 * sd[null, None] > 0, axis=1)
    out: dict[str, Any] = {
        "power_by_dose": {s: float(np.nanmean(rej[dose == s])) for s in DOSES},
        "true_C_over_Cref_by_dose": {
            s: float(np.median(np.sqrt(truth[dose == s] / c_ref2))) for s in DOSES
        },
        "floor_C2": floor,
        "floor_C_over_Cref": float(np.sqrt(max(floor, 0.0) / c_ref2)),
        "null_se_over_mcsd_median": float(np.nanmedian(se[null] / sd[null])),
        "null_q95_over_1.645se_median": float(np.nanmedian(q95[null] / (1.645 * se[null]))),
        "null_fpr_wald_with_mc_sd_mean": float(np.nanmean(wald_mc)),
        "frac_nonnull_above_floor": float(np.mean(truth[~null] > floor)),
    }
    for label, mask in (("natural", dose == "natural"), ("above_floor", truth > floor)):
        rho = [
            spearmanr(X[mask, r], truth[mask]).statistic
            for r in range(X.shape[1])
            if mask.sum() > 3 and np.isfinite(X[mask, r]).all()
        ]
        out[f"spearman_{label}"] = float(np.mean(rho)) if rho else float("nan")
        out[f"n_{label}"] = int(mask.sum())
    return out


def scaling(per_n: dict[int, dict[str, Any]], c_ref2: float) -> dict[str, Any]:
    Ns = np.array(sorted(per_n))
    fl = np.array([per_n[n]["floor_C2"] for n in Ns])
    slope, icpt = np.polyfit(np.log(Ns), np.log(fl), 1)
    # N at which the floor in C units reaches the small dose (0.2 C_ref): a naive extrapolation
    need = {
        s: float(np.exp((np.log((c**2) * c_ref2) - icpt) / slope))
        for s, c in (("small", 0.2), ("vsmall", 0.05), ("medium", 0.6), ("C_ref", 1.0))
    }
    return {"floor_loglog_slope": float(slope), "N_floor_equals_dose_extrapolated": need}


def tp4_consistent(d: dict[str, Any]) -> dict[str, Any]:
    ok: dict[str, list[bool]] = {"E2": [], "E3": []}
    for p in d["points"]:
        for N in (256, 512, 1024):
            for m in cb.M_GRID:
                s = d["summary"][p["pid"]]
                tr_n = (
                    s[an.key("I", N, m, "plugin")]["A2"]["mean"]
                    - s[an.key("I", N, m, "E1")]["A2"]["mean"]
                )
                if p["oracle"]["I"]["A2"] <= 10 * max(tr_n, 0.0):
                    continue
                for e in ok:
                    c = s[an.key("I", N, m, e)]["C2"]
                    ok[e].append(abs(c["bias"]) <= 3 * np.sqrt(c["var"] / cb.R_REPS))
    return {
        e: {"n": len(v), "frac_within_3mcse": float(np.mean(v)) if v else float("nan")}
        for e, v in ok.items()
    }


def main(argv: list[str]) -> int:
    split, run_dir = argv[1], Path(argv[2]).resolve()
    d = an.load(run_dir, split)
    out_dir = provenance.create_run_dir(REPO / "results", f"E005a-posthoc-{split}", REPO)
    provenance.write_metadata(
        out_dir,
        f"E005a-posthoc-{split}",
        an.CONFIG,
        REPO,
        extra={
            "split": split,
            "root_seed": pl.ROOT_SEED,
            "posthoc": True,
            "calibration_run": str(run_dir.relative_to(REPO)),
        },
    )
    report: dict[str, Any] = {"label": "POST-HOC, exploratory; gates nothing", "c_ref": d["c_ref"]}
    for est in EST:
        per_n = {N: diagnostics(d, est, N) for N in cb.N_GRID}
        report[est] = {
            "by_N": {str(n): v for n, v in per_n.items()},
            **scaling(per_n, d["c_ref"] ** 2),
        }
    report["TP4_dimensionally_consistent"] = tp4_consistent(d)
    (out_dir / "posthoc.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1)[:6000])
    print(f"run directory: {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
