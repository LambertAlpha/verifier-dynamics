"""E004a Stage 1 FINAL held-out, step 2: the REGISTERED analysis (Amendment 4; execution note).

Frozen predictors: the frozen pipeline refit on all kept design runs (the penalty chosen by the
frozen grouped inner CV must equal the value recorded in predictors_frozen.json, else STOP);
warning thresholds are the frozen design-calibrated tau. Nothing is tuned on held-out data.

Scored: RQ1 (test split A), early warning, generalization (shift split; leave-one-mechanism-out:
fit on the design runs of five mechanisms, evaluate on the held-out mechanism's test runs), the
decision-tree label and the abandonment flag. Reported: information curves, L3-L1 / L3-L2, NYV
curve and lead times, onset-noise diagnostic, mechanism diagnosis, within-mechanism,
cross-construction, type oracles, single diagnostics, GBM check, NG (secondary), optimizer transfer
(secondary), estimator checks vs design. Figures 1-5 and 7 (figure 6 in the hard-pair step).
"""

import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage1_analysis as an  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e004 import heldout as hd  # noqa: E402
from vdyn.e004 import panel0b as pn  # noqa: E402
from vdyn.e004 import predict as pr  # noqa: E402
from vdyn.e004 import toy  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e004" / "e004a_stage1.toml"
DESIGN_RUNS = REPO / "results/E004a-stage1-runs/20260926T045733Z_63ea693"
DESIGN_ANALYSIS = REPO / "results/E004a-stage1-analysis/20260926T050431Z_26b1790"
FROZEN = DESIGN_ANALYSIS / "predictors_frozen.json"
FROZEN_SHA = "e0b2ea9d73a3f8b7f820e5274c507a626b39f84193d9359aa2ed67436bdfd210"
H, H_STAR = an.H, an.H_STAR
MECHS = pn.MECHANISMS
WORKERS = 8


# ------------------------------------------------------------------ data
def load_split(run_dir: Path, split: str) -> dict[str, Any]:
    summ = json.loads((run_dir / "summary.json").read_text())
    panel = REPO / f"configs/e004/{split}_panel_0b.json"
    assert hashlib.sha256(panel.read_bytes()).hexdigest() == summ["panel_sha256"][split]
    structs = [toy.Structure.from_dict(d) for d in json.loads(panel.read_text())["structures"]]
    excluded = set(summ["splits"][split]["excluded_low_clean_gain"])
    return {"arr": np.load(run_dir / f"{split}.npz"), "summary": summ,
            "labels": json.loads((run_dir / f"labels_{split}.json").read_text()),
            "structs": structs, "keep": [s.sid not in excluded for s in structs]}  # fmt: skip


# ------------------------------------------------------------------ frozen models
def _fit(args: tuple[Any, ...]) -> tuple[Any, Any, float | None]:
    key, X, y, g, kind, family = args
    m = pr.fit(X, y, g, kind, family)
    pen = None
    if family == "linear":
        est = m[-1]
        pen = float(est.alpha_ if kind == "ridge" else np.ravel(est.C_)[0])
    return key, m, pen


def fit_many(jobs: list[tuple[Any, ...]]) -> dict[Any, tuple[Any, float | None]]:
    jobs = sorted(jobs, key=lambda j: j[4] != "multi")
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        return {k: (m, p) for k, m, p in pool.map(_fit, jobs, chunksize=1)}


def predict(models: dict[Any, tuple[Any, Any]], sets: dict[tuple[str, float], np.ndarray],
            levels_h: list[tuple[str, float]], families: tuple[str, ...]) -> dict:  # fmt: skip
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for lv, h in levels_h:
        X = an.X_of(sets, lv, h)
        for fam in families:
            for kind in ("ridge", "binary", "multi"):
                key = (lv, h, kind, fam)
                if key not in models:
                    continue
                m = models[key][0]
                if kind == "ridge":
                    out[key] = {"pred": m.predict(X)}
                elif kind == "binary":
                    out[key] = {"pred": m.predict_proba(X)[:, 1]}
                else:
                    out[key] = {"P": m.predict_proba(X), "classes": np.asarray(m.classes_)}
    return out


def level_keys(levels: tuple[str, ...]) -> list[tuple[str, float]]:
    return [(lv, 0.0) if lv in an.STATIC else (lv, h) for lv in levels
            for h in ([0.0] if lv in an.STATIC else H)]  # fmt: skip


def design_jobs(tab: dict[str, Any], sets: dict, levels: tuple[str, ...],
                family: str) -> list[tuple[Any, ...]]:  # fmt: skip
    g, mech = an.col(tab, "i"), an.col(tab, "mechanism")
    dn, fail = an.col(tab, "Dn"), an.col(tab, "failure").astype(int)
    return [((lv, h, kind, family), an.X_of(sets, lv, h), y, g, kind, family)
            for lv, h in level_keys(levels)
            for kind, y in (("ridge", dn), ("binary", fail), ("multi", mech))]  # fmt: skip


# ------------------------------------------------------------------ registered scoring
def heldout_leads(tab: dict[str, Any], res: dict, tau: dict[str, dict[str, float]],
                  levels: tuple[str, ...]) -> dict[str, Any]:  # fmt: skip
    fail, t_on = an.col(tab, "failure"), an.col(tab, "t_on")
    success = an.col(tab, "category") == "SUCCESS"
    out: dict[str, Any] = {}
    for lv in levels:
        p = np.stack([res[(lv, 0.0 if lv in an.STATIC else h, "binary", "linear")]["pred"]
                      for h in H])  # fmt: skip
        per: dict[str, Any] = {}
        for h_obs in H[1:]:
            w = hd.warnings_at(p, H, success, fail, t_on, h_obs, tau[lv][f"{h_obs:g}"])
            lw = w["lead_warned"]
            per[f"{h_obs:g}"] = {
                "tau_frozen": w["tau"], "false_alarm": w["false_alarm"],
                "sensitivity": w["sensitivity"], "median_lead": w["median_lead"],
                "median_lead_conservative": w["median_lead_conservative"],
                "n_nyv_fail": w["n_nyv_fail"], "n_warned_nyv_fail": int(len(lw)),
                "lead_quantiles": np.quantile(lw, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
                if len(lw) else None,
                "t_warn_counts": {f"{h:g}": int(np.sum(w["t_warn"][fail & (t_on > h_obs)] == h))
                                  for h in H if h <= h_obs},
                "_lead": lw,
            }  # fmt: skip
        out[lv] = per
    return out


def early_warning(curves: dict[str, Any], leads: dict[str, Any]) -> dict[str, Any]:
    nyv = curves[f"{H_STAR:g}"]["L2"]["auroc_nyv"]
    lead = leads["L2"][f"{H_STAR:g}"]["median_lead"]
    ok_lead = bool(np.isfinite(lead) and lead >= 0.05)
    return {"nyv_auroc_L2": nyv, "lo95_ge_0.65": bool(nyv["lo95"] >= 0.65),
            "median_lead_L2": lead, "median_lead_ge_0.05": ok_lead,
            "pass": bool(nyv["lo95"] >= 0.65 and ok_lead)}  # fmt: skip


def lomo(tab_d: dict, sets_d: dict, tab_t: dict, sets_t: dict,
         bt: an.Booter) -> dict[str, Any]:  # fmt: skip
    """§12 split B: fit on the design runs of five mechanisms, evaluate on the held-out
    mechanism's test runs; L2 - L0 at h*, C-index (always) and AUROC (when both classes have
    >= 10 test runs); a mechanism succeeds iff every evaluable lower bound is > 0."""
    md, gd = an.col(tab_d, "mechanism"), an.col(tab_d, "i")
    dnd, fd = an.col(tab_d, "Dn"), an.col(tab_d, "failure").astype(int)
    mt, dnt, ft = an.col(tab_t, "mechanism"), an.col(tab_t, "Dn"), an.col(tab_t, "failure")
    jobs = []
    for m in MECHS:
        tr = md != m
        for lv in ("L0", "L2"):
            X = an.X_of(sets_d, lv, H_STAR)[tr]
            jobs.append(((m, lv, "ridge", "linear"), X, dnd[tr], gd[tr], "ridge", "linear"))
            jobs.append(((m, lv, "binary", "linear"), X, fd[tr], gd[tr], "binary", "linear"))
    models = fit_many(jobs)
    out: dict[str, Any] = {}
    n_pass = 0
    for m in MECHS:
        sel = mt == m
        both = min(ft[sel].sum(), (~ft[sel]).sum()) >= 10
        cur: dict[str, Any] = {}
        for lv in ("L0", "L2"):
            Xt = an.X_of(sets_t, lv, H_STAR)
            pdn = models[(m, lv, "ridge", "linear")][0].predict(Xt)
            cur[lv] = {"cindex_Dn": bt.pairwise(pdn, dnt, "cindex", sel)}
            if both:
                pf = models[(m, lv, "binary", "linear")][0].predict_proba(Xt)[:, 1]
                cur[lv]["auroc_failure"] = bt.pairwise(pf, ft, "auroc", sel)
        diffs = {k: an.diff(cur["L2"][k], cur["L0"][k]) for k in cur["L2"]}
        ok = all(d["lo95"] > 0 for d in diffs.values())
        n_pass += ok
        out[m] = {"L2-L0": diffs, "pass": ok, "auroc_evaluable": both,
                  "n_runs": int(sel.sum()), "n_fail": int(ft[sel].sum()),
                  **{lv: {k: an.summ(*v) for k, v in cur[lv].items()} for lv in cur}}  # fmt: skip
    out["n_pass"] = int(n_pass)
    out["pass"] = bool(n_pass >= 4)
    return out


def shift_criterion(curves_s: dict[str, Any]) -> dict[str, Any]:
    d = curves_s[f"{H_STAR:g}"]["diffs"]["L2-L0"]
    ok = {m: bool(d[m]["lo95"] > -0.01) for m in ("cindex_Dn", "auroc_failure")}
    return {"L2-L0": {m: d[m] for m in ("cindex_Dn", "auroc_failure")}, "lo95_gt_-0.01": ok,
            "pass": bool(all(ok.values()))}  # fmt: skip


def abandonment(curves: dict[str, Any]) -> dict[str, Any]:
    per = {}
    for h in [x for x in H if x <= 0.05 + 1e-12]:
        d = curves[f"{h:g}"]["diffs"]["L2-L0"]
        adj = pr.holm(np.array([d["cindex_Dn"]["p_one_sided"], d["auroc_failure"]["p_one_sided"]]))
        rq1 = all(pr.beats(d[m], pr.DELTA_OUT, adj[j])
                  for j, m in enumerate(("cindex_Dn", "auroc_failure")))  # fmt: skip
        lo = curves[f"{h:g}"]["L2"]["auroc_nyv"]["lo95"]
        per[f"{h:g}"] = {"rq1": bool(rq1), "nyv_lo95": lo, "abandon_here": (not rq1) or lo <= 0.55}
    return {"per_h": per, "abandon": bool(all(v["abandon_here"] for v in per.values()))}


def label(rq1: bool, ew: bool, gen: bool) -> str:
    if not rq1:
        return "NO CONFIRMATORY EARLY-PREDICTION RESULT"
    if ew and gen:
        return "FULL EARLY-WARNING SUCCESS"
    return "EARLY ASSOCIATION, NO ROBUST EARLY-WARNING CLAIM"


# ------------------------------------------------------------------ reported analyses
def onset_noise(tab: dict[str, Any]) -> dict[str, float]:
    fail, t_on = an.col(tab, "failure"), an.col(tab, "t_on")
    return {f"{h:g}": float(np.mean(t_on[~fail] <= h)) for h in H + [0.2, 0.5, 1.0]}


def within_heldout(tab_d: dict, sets_d: dict, tab_t: dict, sets_t: dict,
                   bt: an.Booter) -> dict[str, Any]:  # fmt: skip
    md, gd = an.col(tab_d, "mechanism"), an.col(tab_d, "i")
    dnd, fd = an.col(tab_d, "Dn"), an.col(tab_d, "failure").astype(int)
    mt, dnt, ft = an.col(tab_t, "mechanism"), an.col(tab_t, "Dn"), an.col(tab_t, "failure")
    hs = (0.01, H_STAR, 0.05)
    jobs = []
    for m in MECHS:
        tr = md == m
        both = min(fd[tr].sum(), (1 - fd[tr]).sum()) >= 10
        for h in hs:
            for lv in ("L0", "L1", "L2", "L3"):
                X = an.X_of(sets_d, lv, h)[tr]
                jobs.append(((m, h, lv, "ridge"), X, dnd[tr], gd[tr], "ridge", "linear"))
                if both:
                    jobs.append(((m, h, lv, "binary"), X, fd[tr], gd[tr], "binary", "linear"))
    models = fit_many(jobs)
    out: dict[str, Any] = {}
    for m in MECHS:
        sel = mt == m
        per: dict[str, Any] = {"n_runs": int(sel.sum()), "n_fail": int(ft[sel].sum())}
        for h in hs:
            cur: dict[str, Any] = {}
            for lv in ("L0", "L1", "L2", "L3"):
                Xt = an.X_of(sets_t, lv, h)
                cur[lv] = {"cindex_Dn": bt.pairwise(models[(m, h, lv, "ridge")][0].predict(Xt),
                                                    dnt, "cindex", sel)}  # fmt: skip
                if (m, h, lv, "binary") in models and 0 < ft[sel].sum() < sel.sum():
                    pf = models[(m, h, lv, "binary")][0].predict_proba(Xt)[:, 1]
                    cur[lv]["auroc_failure"] = bt.pairwise(pf, ft, "auroc", sel)
            per[f"{h:g}"] = {
                **{lv: {k: an.summ(*v) for k, v in cur[lv].items()} for lv in cur},
                **{f"{a}-{b}": {k: an.diff(cur[a][k], cur[b][k]) for k in cur[a]}
                   for a, b in (("L3", "L2"), ("L3", "L1"), ("L2", "L0"))},
            }  # fmt: skip
        out[m] = per
    return out


def cross_heldout(tab_d: dict, sets_d: dict, tab_t: dict, sets_t: dict,
                  bt: an.Booter) -> dict[str, Any]:  # fmt: skip
    md, sd, gd = an.col(tab_d, "mechanism"), an.col(tab_d, "slot"), an.col(tab_d, "i")
    mt, st = an.col(tab_t, "mechanism"), an.col(tab_t, "slot")
    hs = (0.01, H_STAR, 0.05)
    jobs = [((k, h, lv), an.X_of(sets_d, lv, h)[sd != k], md[sd != k], gd[sd != k], "multi",
             "linear")
            for k in (1, 2, 3) for h in hs for lv in ("L0", "L1", "L2", "L3")]  # fmt: skip
    models = fit_many(jobs)
    out: dict[str, Any] = {}
    for h in hs:
        cur: dict[str, Any] = {}
        ent: dict[str, Any] = {}
        for lv in ("L0", "L1", "L2", "L3"):
            pred = np.empty(len(mt), dtype=object)
            for k in (1, 2, 3):
                te = st == k
                pred[te] = models[(k, h, lv)][0].predict(an.X_of(sets_t, lv, h)[te])
            classes = np.unique(md)
            cur[lv] = bt.classes(mt, pred.astype(str), classes)["f1"]
            ent[lv] = {"pooled_macro_f1": an.summ(*cur[lv])}
        for a, b in (("L3", "L2"), ("L3", "L1"), ("L2", "L0")):
            ent[f"{a}-{b}"] = an.diff(cur[a], cur[b])
        out[f"{h:g}"] = ent
    return out


def type_oracles_heldout(tab_d: dict, tab_t: dict, bt: an.Booter) -> dict[str, Any]:
    dnd, dnt, ft = an.col(tab_d, "Dn"), an.col(tab_t, "Dn"), an.col(tab_t, "failure")
    out: dict[str, Any] = {}
    for name, kd, kt in (
        ("mechanism", an.col(tab_d, "mechanism"), an.col(tab_t, "mechanism")),
        ("mechanism_x_axis",
         np.char.add(an.col(tab_d, "mechanism").astype(str), an.col(tab_d, "axis").astype(str)),
         np.char.add(an.col(tab_t, "mechanism").astype(str), an.col(tab_t, "axis").astype(str))),
    ):  # fmt: skip
        means = {k: float(dnd[kd == k].mean()) for k in np.unique(kd)}
        pred = np.array([means.get(k, float(dnd.mean())) for k in kt])
        out[name] = {"cindex_Dn": an.summ(*bt.pairwise(pred, dnt, "cindex")),
                     "auroc_failure": an.summ(*bt.pairwise(pred, ft, "auroc"))}  # fmt: skip
    return out


def transfer(tab_src: dict, sets_src: dict, tab_tgt: dict, sets_tgt: dict,
             bt: an.Booter) -> dict[str, Any]:  # fmt: skip
    g = an.col(tab_src, "i")
    dn, fail = an.col(tab_src, "Dn"), an.col(tab_src, "failure").astype(int)
    jobs = []
    for lv in ("L0", "L1", "L2", "L3"):
        X = an.X_of(sets_src, lv, H_STAR)
        jobs.append(((lv, "ridge"), X, dn, g, "ridge", "linear"))
        jobs.append(((lv, "binary"), X, fail, g, "binary", "linear"))
    models = fit_many(jobs)
    dnt, ft = an.col(tab_tgt, "Dn"), an.col(tab_tgt, "failure")
    out = {}
    for lv in ("L0", "L1", "L2", "L3"):
        Xt = an.X_of(sets_tgt, lv, H_STAR)
        out[lv] = {"cindex_Dn": an.summ(*bt.pairwise(models[(lv, "ridge")][0].predict(Xt), dnt,
                                                     "cindex")),
                   "auroc_failure": an.summ(*bt.pairwise(
                       models[(lv, "binary")][0].predict_proba(Xt)[:, 1], ft,
                       "auroc"))}  # fmt: skip
    return out


# ------------------------------------------------------------------ figures
def _band(ax: Any, src: dict, lv: str, metric: str, label: str | None = None) -> None:
    xs = [100 * h for h in H]
    ys = [src[f"{h:g}"][lv][metric]["point"] for h in H]
    lo = [src[f"{h:g}"][lv][metric]["ci2_lo"] for h in H]
    hi = [src[f"{h:g}"][lv][metric]["ci2_hi"] for h in H]
    (ln,) = ax.plot(xs, ys, marker="o", ms=3, lw=1.5, label=label or lv)
    ax.fill_between(xs, lo, hi, color=ln.get_color(), alpha=0.12, lw=0)


def _axes(ax: Any, title: str) -> None:
    ax.axvline(100 * H_STAR, color="0.6", lw=0.8, ls=":")
    ax.set_xscale("symlog", linthresh=0.2)
    ax.set_xlim(0, 11)
    ax.set_xlabel("% of T observed")
    ax.set_title(title, fontsize=10)


def figures(rep: dict[str, Any], out_dir: Path) -> None:
    cur, mech = rep["information"], rep["mechanism"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (m, t) in zip(axes, (("cindex_Dn", "C-index (Dn)"), ("auroc_failure", "AUROC")),
                          strict=True):  # fmt: skip
        for lv in ("L0", "L1", "L2-G", "L2", "L3"):
            _band(ax, cur, lv, m)
        _axes(ax, t)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fig 1. E004a Stage 1 HELD-OUT: prediction quality vs fraction of T observed")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_information_heldout.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for lv in ("L0", "L1", "L2", "L3"):
        _band(axes[0], cur, lv, "auroc_nyv")
    axes[0].axhline(0.65, color="C3", lw=0.8, ls="--")
    _axes(axes[0], "not-yet-visible AUROC (t_on > h)")
    axes[0].legend(frameon=False, fontsize=8)
    lead = rep["lead_times"]["L2"][f"{H_STAR:g}"]["_lead"]
    axes[1].hist(100 * np.asarray(lead), bins=np.linspace(0, 100, 41), color="C0")
    axes[1].axvline(5, color="C3", lw=1)
    axes[1].set_xlabel("lead time (% of T), warned NYV failures at h* (L2)")
    fig.suptitle("Fig 2. Early warning (held-out)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_nyv_leadtime_heldout.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for lv in ("L1", "L2", "L3"):
        _band(ax, mech, lv, "macro_f1")
    ax.axhline(1 / 6, color="0.5", lw=0.8, ls="--")
    _axes(ax, "Fig 3. Mechanism macro-F1 (held-out)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_mechanism_heldout.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    panels = (
        (cur, "cindex_Dn", "C-index"),
        (cur, "auroc_failure", "AUROC failure"),
        (mech, "macro_f1", "mechanism macro-F1"),
    )
    for ax, (src, m, t) in zip(axes, panels, strict=True):
        for d in ("L3-L1", "L3-L2"):
            xs = [100 * h for h in H]
            ys = [src[f"{h:g}"]["diffs"][d][m]["point"] for h in H]
            lo = [src[f"{h:g}"]["diffs"][d][m]["ci2_lo"] for h in H]
            hi = [src[f"{h:g}"]["diffs"][d][m]["ci2_hi"] for h in H]
            (ln,) = ax.plot(xs, ys, marker="o", ms=3, label=d)
            ax.fill_between(xs, lo, hi, color=ln.get_color(), alpha=0.15, lw=0)
        ax.axhline(0, color="0.3", lw=0.8)
        _axes(ax, t)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fig 4. Snapshot vs trajectory: L3 - L1 and L3 - L2 (held-out, 95% CI)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_snapshot_vs_trajectory_heldout.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    lomo_ = rep["generalization"]["lomo"]
    for j, m in enumerate(("cindex_Dn", "auroc_failure")):
        ys, lo, hi, names = [], [], [], []
        for mech_ in MECHS:
            d = lomo_[mech_]["L2-L0"].get(m)
            if d is None:
                continue
            ys.append(d["point"])
            lo.append(d["point"] - d["lo95"])
            hi.append(d["ci2_hi"] - d["point"])
            names.append(mech_)
        axes[j].errorbar(range(len(ys)), ys, yerr=[lo, hi], fmt="o", capsize=3)
        axes[j].set_xticks(range(len(names)), names)
        axes[j].axhline(0, color="0.3", lw=0.8)
        axes[j].set_title(f"leave-one-mechanism-out L2 - L0 ({m})", fontsize=10)
    cc = rep["cross_construction"][f"{H_STAR:g}"]
    lv_ = ("L0", "L1", "L2", "L3")
    axes[2].bar(lv_, [cc[lv]["pooled_macro_f1"]["point"] for lv in lv_], color="0.5")
    axes[2].axhline(1 / 6, color="C3", lw=0.8, ls="--")
    axes[2].set_title("cross-construction mechanism macro-F1 at h*", fontsize=10)
    fig.suptitle("Fig 5. Generalization (held-out)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_generalization_heldout.png", dpi=130)
    plt.close(fig)
    od = rep["optimizer_dependence"]
    cells = list(od)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(cells))
    ax.bar(x - 0.2, [od[c]["adam"] for c in cells], 0.4, label="sampled Adam")
    ax.bar(x + 0.2, [od[c]["ng"] for c in cells], 0.4, label="NG")
    ax.set_xticks(x, [c.replace("|", "\n") for c in cells], fontsize=8)
    ax.set_ylabel("failure fraction")
    ax.legend(frameon=False)
    ax.set_title("Fig 7. Optimizer dependence on the held-out panel")
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_optimizer_dependence_heldout.png", dpi=130)
    plt.close(fig)


def optimizer_dependence(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, st in enumerate(d["structs"]):
        if not d["keep"][i]:
            continue
        key = f"{st.mechanism}|{st.meta['axis_b'][:3]}"
        out.setdefault(key, {"adam": [], "ng": []})
        out[key]["adam"].extend(lb["failure"] for lb in d["labels"]["adam_prim"][i])
        out[key]["ng"].extend(lb["failure"] for lb in d["labels"]["ng_prim"][i])
    order = sorted(out, key=lambda k: (MECHS.index(k.split("|")[0]), k))
    return {k: {o: float(np.mean(out[k][o])) for o in ("adam", "ng")} for k in order}


# ------------------------------------------------------------------ main
def main(argv: list[str]) -> int:
    run_dir = Path(argv[1]).resolve()
    cfg = provenance.load_config(CONFIG)
    frozen_bytes = FROZEN.read_bytes()
    if hashlib.sha256(frozen_bytes).hexdigest() != FROZEN_SHA:
        print("STOP: frozen predictor configuration changed")
        return 1
    frozen = json.loads(frozen_bytes)
    runs_meta = json.loads((run_dir / "meta.json").read_text())
    assert runs_meta["predictors_frozen_sha256"] == FROZEN_SHA and not runs_meta["git"]["dirty"]
    out_dir = provenance.create_run_dir(REPO / "results", "E004a-stage1-heldout-analysis", REPO)
    provenance.write_metadata(out_dir, "E004a-stage1-heldout-analysis", CONFIG, REPO,
                              extra={"split": "test+shift", "root_seed": pn.ROOT_SEED,
                                     "panel_sha256": cfg["panel_sha256"],
                                     "test_panel_sha256": runs_meta["test_panel_sha256"],
                                     "shift_panel_sha256": runs_meta["shift_panel_sha256"],
                                     "predictors_frozen_sha256": FROZEN_SHA,
                                     "runs": str(run_dir.relative_to(REPO))})  # fmt: skip
    t_start = time.perf_counter()
    res_seeds = hd.resample_seeds()
    d_design = an.load(DESIGN_RUNS)
    d_test = load_split(run_dir, "test")
    d_shift = load_split(run_dir, "shift")
    rng_d, rng_t = np.random.default_rng(res_seeds[0]), np.random.default_rng(res_seeds[7])
    tab_da, tab_dn = an.table(d_design, "adam"), an.table(d_design, "ng")
    sets_da, sets_dn = an.feature_sets(tab_da, rng_d), an.feature_sets(tab_dn, rng_d)
    tab_ta, tab_tn = an.table(d_test, "adam"), an.table(d_test, "ng")
    tab_s = an.table(d_shift, "adam")
    sets_ta, sets_tn = an.feature_sets(tab_ta, rng_t), an.feature_sets(tab_tn, rng_t)
    sets_s = an.feature_sets(tab_s, rng_t)
    # frozen models (design-side refit; penalties must equal the frozen record)
    lin = tuple(an.STATIC + an.DYNAMIC)
    jobs = design_jobs(tab_da, sets_da, lin, "linear") + design_jobs(tab_da, sets_da, an.PRIMARY,
                                                                     "gbm")  # fmt: skip
    models = fit_many(jobs)
    mismatch = {f"{k[0]}|{k[1]:g}|{k[2]}": (p, frozen["full_design_penalties"][
        f"{k[0]}|{k[1]:g}|{k[2]}"]) for k, (_, p) in models.items() if k[3] == "linear"
        and p != frozen["full_design_penalties"][f"{k[0]}|{k[1]:g}|{k[2]}"]}  # fmt: skip
    if mismatch:
        print("STOP: refit penalties differ from the frozen record", mismatch)
        return 1
    print(f"frozen models refit and verified ({len(models)}): "
          f"{time.perf_counter() - t_start:.0f} s", flush=True)  # fmt: skip
    keys_lin = level_keys(lin)
    res_t = predict(models, sets_ta, keys_lin, ("linear", "gbm"))
    res_s = predict(models, sets_s, keys_lin, ("linear", "gbm"))
    bt_t = an.Booter(
        pr.hier_weights(an.col(tab_ta, "i"), an.B, np.random.default_rng(res_seeds[4]))
    )
    bt_s = an.Booter(pr.hier_weights(an.col(tab_s, "i"), an.B, np.random.default_rng(res_seeds[6])))
    rep: dict[str, Any] = {}
    curves = an.outcome_curves(tab_ta, res_t, bt_t, lin, "linear")
    mres = an.mechanism(tab_ta, res_t, bt_t, lin, "linear")
    leads = heldout_leads(tab_ta, res_t, frozen["warning"]["tau_adam"], an.PRIMARY)
    rep.update(information=curves, mechanism=mres, lead_times=leads, rq1=an.rq1(curves),
               early_warning=early_warning(curves, leads),
               mechanistic_i_ii_descriptive=an.mech_value(mres),
               abandonment_flag=abandonment(curves), onset_noise_nonfailures=onset_noise(tab_ta),
               onset_noise_design=onset_noise(tab_da))  # fmt: skip
    print("[test] curves, mechanism, RQ1, EW done", flush=True)
    curves_s = an.outcome_curves(tab_s, res_s, bt_s, lin, "linear")
    gen_lomo = lomo(tab_da, sets_da, tab_ta, sets_ta, bt_t)
    gen_shift = shift_criterion(curves_s)
    rep["generalization"] = {"lomo": gen_lomo, "shift": gen_shift, "shift_curves": curves_s,
                             "pass": bool(gen_lomo["pass"] and gen_shift["pass"])}  # fmt: skip
    print("[test] generalization done", flush=True)
    rep["within_mechanism"] = within_heldout(tab_da, sets_da, tab_ta, sets_ta, bt_t)
    rep["cross_construction"] = cross_heldout(tab_da, sets_da, tab_ta, sets_ta, bt_t)
    rep["type_oracles"] = type_oracles_heldout(tab_da, tab_ta, bt_t)
    rep["single_diagnostics"] = an.single_diagnostics(tab_ta, sets_ta)
    rep["gbm"] = an.gbm_check(tab_ta, res_t, bt_t)
    rep["estimator_sanity"] = {"test": an.estimator_sanity(tab_ta),
                               "shift": an.estimator_sanity(tab_s)}  # fmt: skip
    print("[test] within / cross / oracles / singles / gbm / sanity done", flush=True)
    # NG (secondary) and optimizer transfer (secondary)
    ng_models = fit_many(design_jobs(tab_dn, sets_dn, an.PRIMARY, "linear"))
    res_tn = predict(ng_models, sets_tn, level_keys(an.PRIMARY), ("linear",))
    bt_tn = an.Booter(pr.hier_weights(an.col(tab_tn, "i"), an.B,
                                      np.random.default_rng(res_seeds[5])))  # fmt: skip
    rep["ng"] = {"information": an.outcome_curves(tab_tn, res_tn, bt_tn, an.PRIMARY, "linear"),
                 "mechanism": an.mechanism(tab_tn, res_tn, bt_tn, an.PRIMARY, "linear"),
                 "estimator_sanity": an.estimator_sanity(tab_tn)}  # fmt: skip
    rep["ng"]["rq1"] = an.rq1(rep["ng"]["information"])
    rep["optimizer_transfer"] = {"ng_design_to_adam_test": transfer(tab_dn, sets_dn, tab_ta,
                                                                    sets_ta, bt_t),
                                 "adam_design_to_ng_test": transfer(tab_da, sets_da, tab_tn,
                                                                    sets_tn, bt_tn)}  # fmt: skip
    rep["optimizer_dependence"] = optimizer_dependence(d_test)
    rq1, ew, gen = rep["rq1"]["pass"], rep["early_warning"]["pass"], rep["generalization"]["pass"]
    rep["decision"] = {"RQ1": rq1, "EW": ew, "GEN": gen, "label": label(rq1, ew, gen),
                       "mechanistic_criterion": "not satisfied (hard-pair (iii) fixed at 0.75 < "
                       "0.80); descriptive mechanistic evidence reported separately"}  # fmt: skip
    rep["counts"] = {"test_runs": len(tab_ta["rows"]), "shift_runs": len(tab_s["rows"]),
                     "test_ng_runs": len(tab_tn["rows"])}  # fmt: skip
    rep["timing_s"] = time.perf_counter() - t_start
    figures(rep, out_dir)
    (out_dir / "heldout.json").write_text(json.dumps(an.strip(rep), indent=1) + "\n")
    print(json.dumps(rep["decision"]))
    print(f"run directory: {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
