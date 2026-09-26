"""[post-hoc, NOT pre-registered] E004a Stage 0b robustness checks on the committed outputs.

Design split only; oracle/design diagnostics, not Stage 1 predictors; no gate is recomputed or
changed. Checks: (1) G6 margin, structure bootstrap of the YA Adam - NG failure difference;
(2) G5 under a Holm correction over all within-mechanism L3 - L2 targets; (3) S0b-P1/P2 with the
intended instead of the actual Axis B; (4) within mechanism x Axis-B cells, failure AUROC and Dn
C-index for L0/L1/L2/L3 at h* (information beyond the type oracle); (5) INVERTED vs ALIGNED
separation by alpha_u0 and C0/A0 within YA/YB, primary vs canonical.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0_analysis as sa  # noqa: E402
import stage0b_analysis as an  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e002 import endpoints as ep  # noqa: E402
from vdyn.e004 import panel0b as pb  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CELLS = [("X", "ALIGNED"), ("YA", "INVERTED"), ("YB", "INVERTED"), ("D", "INVERTED"),
         ("YB", "ALIGNED")]  # fmt: skip


def rate(lab: list, idx: list[int], cat: str | None = None) -> float:
    xs = [(r["category"] == cat) if cat else r["failure"] for i in idx for r in lab[i]]
    return float(np.mean(xs)) if xs else float("nan")


def g6_margin(d: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    ya = [i for i, s in enumerate(d["structs"]) if d["keep"][i] and s.mechanism == "YA"]
    ad, ng = d["labels"]["adam_prim"], d["labels"]["ng_prim"]
    boot = [
        rate(ad, list(s)) - rate(ng, list(s))
        for s in (rng.choice(ya, len(ya)) for _ in range(2000))
    ]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"YA_adam_minus_ng": rate(ad, ya) - rate(ng, ya), "ci": [float(lo), float(hi)]}


def g5_holm(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    tests = []
    for m, res in analysis["test_C"].items():
        for name, v in res.items():
            if "L3_minus_L2" in v:
                b = v["L3_minus_L2"]
                se = (b["ci_hi"] - b["ci_lo"]) / (2 * 1.96)
                tests.append({"target": f"{m}:{name}", "point": b["point"], "se": se,
                              "p_one_sided": float(1 - norm.cdf(b["point"] / se))})  # fmt: skip
    tests.sort(key=lambda t: t["p_one_sided"])
    stop = False
    for j, t in enumerate(tests):
        thr = 0.05 / (len(tests) - j)
        t["holm_threshold"] = thr
        t["pass"] = bool(not stop and t["p_one_sided"] <= thr and t["point"] >= 0.02)
        stop = stop or t["p_one_sided"] > thr
    return tests


def predictions_by_axis(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ad = d["labels"]["adam_prim"]
    for key in ("axis_b", "intended_axis"):
        for m in ("YA", "YB"):
            sel = [i for i, s in enumerate(d["structs"]) if d["keep"][i] and s.mechanism == m]
            inv = [i for i in sel if d["structs"][i].meta[key] == "INVERTED"]
            ali = [i for i in sel if d["structs"][i].meta[key] == "ALIGNED"]
            out[f"{key}|{m}"] = {
                "n_inverted": len(inv),
                "n_aligned": len(ali),
                "decline_inverted": rate(ad, inv, "DECLINE"),
                "decline_aligned": rate(ad, ali, "DECLINE"),
                "failure_inverted": rate(ad, inv),
                "failure_aligned": rate(ad, ali),
            }
    noise = [i for i, s in enumerate(d["structs"]) if d["keep"][i] and s.meta["noise_inverted"]]
    out["noise_inverted_kept"] = {
        "n": len(noise),
        "adam_failure": rate(ad, noise),
        "adam_decline": rate(ad, noise, "DECLINE"),
        "ng_failure": rate(d["labels"]["ng_prim"], noise),
    }
    return out


def within_cells(d: dict[str, Any], fr: list[float]) -> dict[str, Any]:
    tab = an.table(d, "adam_prim_ver", "adam_prim", seeds=True)
    mech, axis, g = an.cols(tab, "mechanism"), an.cols(tab, "axis"), an.cols(tab, "i")
    fail, dn = an.cols(tab, "failure").astype(int), an.cols(tab, "Dn")
    feats = {lv: sa.features(tab, fr, an.H_STAR, lv) for lv in ("L0", "L1", "L2", "L3")}
    out: dict[str, Any] = {}
    for m, a in CELLS:
        sel = (mech == m) & (axis == a)
        y, yd = fail[sel], dn[sel]
        res: dict[str, Any] = {"runs": int(sel.sum()), "failure": float(y.mean())}
        for lv, X in feats.items():
            ok = min(y.sum(), (1 - y).sum()) >= 10
            p = sa.cv_predict(X[sel], y, g[sel], y, "binary") if ok else None
            q = sa.cv_predict(X[sel], yd, g[sel], np.zeros(len(yd)), "ridge")
            res[lv] = {
                "auroc_failure": ep.auroc(p, y.astype(bool)) if p is not None else None,
                "cindex_Dn": ep.c_index(q, yd),
            }
        out[f"{m}|{a}"] = res
        print(f"  cell {m}|{a} done", flush=True)
    return out


def axis_separation(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    mech = np.array([s.mechanism for s in d["structs"]])
    axis = np.array([s.meta["axis_b"] for s in d["structs"]])
    keep = np.array(d["keep"])
    for grp, seeds in (("adam_prim_ver", True), ("ng_prim_ver", False), ("adam_can_ver", True)):
        gu = d["arr"][f"{grp}__geo_u"]
        gu = gu.mean(1) if seeds else gu
        for m in ("YA", "YB"):
            sel = keep & (mech == m)
            inv = axis[sel] == "INVERTED"
            a0, ratio = gu[sel, 0, 1], gu[sel, 0, 2] / gu[sel, 0, 0]
            out[f"{grp}|{m}"] = {
                "alpha_u0_median_inverted": float(np.median(a0[inv])),
                "alpha_u0_median_aligned": float(np.median(a0[~inv])),
                "auroc_inverted_by_neg_alpha_u0": ep.auroc(-a0, inv),
                "auroc_inverted_by_C0_over_A0": ep.auroc(ratio, inv),
            }
    return out


def main(argv: list[str]) -> int:
    runs_dir, analysis_dir = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    d = an.load(runs_dir)
    analysis = json.loads((analysis_dir / "stage0b_analysis.json").read_text())
    out_dir = provenance.create_run_dir(REPO / "results", "E004a-stage0b-posthoc", REPO)
    provenance.write_metadata(
        out_dir,
        "E004a-stage0b-posthoc",
        an.CONFIG,
        REPO,
        extra={
            "split": "design",
            "post_hoc": True,
            "runs": str(runs_dir),
            "analysis": str(analysis_dir),
            "root_seed": pb.ROOT_SEED,
            "panel_sha256": d["summary"]["panel_sha256"],
        },
    )
    rng = np.random.default_rng([pb.ROOT_SEED, 10])
    report = {
        "g6_margin": g6_margin(d, rng),
        "g5_holm": g5_holm(analysis),
        "predictions_by_axis": predictions_by_axis(d),
        "axis_separation_t0": axis_separation(d),
        "within_cells_hstar": within_cells(d, d["summary"]["fracs"]),
    }
    (out_dir / "stage0b_posthoc.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n"
    )
    print(json.dumps(report["g6_margin"]), flush=True)
    print(f"run directory: {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
