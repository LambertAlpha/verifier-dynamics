"""E004a Stage 1, design round, step 2 (registry: Stage 1 execution note §3-§5, §9-§10).

Design-CV predictors on the finite-sample features of the Stage 1 design runs: information curves
(L0/L1/L2-G/L2/L3 and the secondary arms), explicit L2-L0, L3-L2, L3-L1 at every horizon, the
not-yet-visible curve, warning threshold / lead time, mechanism classification (macro-F1, balanced
accuracy, confusion, Route A vs B, B vs D, X vs Y), within-mechanism outcome prediction, type
oracles, the cross-construction test, a design-side leave-one-mechanism-out diagnostic, single
diagnostics, the GBM check, the noise-feature control, estimator sanity checks, and the frozen
predictor configuration. DESIGN SPLIT ONLY; out-of-fold predictions; nothing held out is touched.
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
import stage1_runs as s1r  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e002 import endpoints as ep  # noqa: E402
from vdyn.e004 import features as fe  # noqa: E402
from vdyn.e004 import panel0b as pn  # noqa: E402
from vdyn.e004 import predict as pr  # noqa: E402
from vdyn.e004 import toy  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e004" / "e004a_stage1.toml"
PANEL = REPO / "configs" / "e004" / "design_panel_0b.json"
H = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
H_STAR, H_SEC = 0.02, (0.01, 0.05)
STATIC = ("L0", "L1", "L1r")
DYNAMIC = ("L2-G", "L2", "L2N", "L2+", "L3", "L3r", "L3+")
PRIMARY = ("L0", "L1", "L2-G", "L2", "L3")
MECHS = pn.MECHANISMS
B = 2000
WORKERS = 8


# ------------------------------------------------------------------ data
def load(run_dir: Path) -> dict[str, Any]:
    arr = np.load(run_dir / "runs.npz")
    summ = json.loads((run_dir / "summary.json").read_text())
    lab = json.loads((run_dir / "labels.json").read_text())
    structs = [toy.Structure.from_dict(d) for d in json.loads(PANEL.read_text())["structures"]]
    excluded = set(summ["excluded_low_clean_gain"])
    keep = [s.sid not in excluded for s in structs]
    return {"arr": arr, "summary": summ, "labels": lab, "structs": structs, "keep": keep}


def table(d: dict[str, Any], opt: str) -> dict[str, Any]:
    """Kept runs (structure x seed) with labels, finite-sample series and oracle values."""
    arr = d["arr"]
    lab_key = "adam_prim" if opt == "adam" else "ng_prim"
    keys = [*s1r.EST_KEYS, "ctx"]
    est = {k: arr[f"{opt}__est__{k}"] for k in keys}
    exact = {k: arr[f"{opt}__exact__{k}"] for k in ("obs", "geo_u", "geo_r")}
    if opt == "ng":
        est = {k: v[:, None] for k, v in est.items()}
        exact = {k: v[:, None] for k, v in exact.items()}
    kept = [i for i, k in enumerate(d["keep"]) if k]
    n_seed = est["J_G"].shape[1]
    rows = []
    for i in kept:
        st = d["structs"][i]
        for r in range(n_seed):
            lb = d["labels"][lab_key][i][r]
            rows.append({"i": i, "r": r, "mechanism": st.mechanism, "construction": st.construction,
                         "axis": st.meta["axis_b"], "slot": int(st.construction[-1]),
                         "failure": bool(lb["failure"]), "Dn": float(lb["Dn"]),
                         "category": lb["category"],
                         "t_on": np.inf if lb["t_on"] is None else float(lb["t_on"])})  # fmt: skip
    m = len(rows)
    return {
        "rows": rows,
        "est": {k: v[kept].reshape(m, *v.shape[2:]) for k, v in est.items()},
        "exact": {k: v[kept].reshape(m, *v.shape[2:]) for k, v in exact.items()},
    }


def col(tab: dict[str, Any], key: str) -> np.ndarray:
    return np.array([r[key] for r in tab["rows"]])


def hidx(h: float) -> tuple[int, int, int]:
    ff = s1r.FEAT_FRACS
    return ff.index(0.0), ff.index(h / 2 if h > 0 else 0.0), ff.index(h)


def features(tab: dict[str, Any], h: float) -> dict[str, np.ndarray]:
    idx = hidx(h)
    est = tab["est"]
    names = ("J_G", "J_V", "FPR", "FNR", "FPM", "A_u", "alpha_u", "C_u", "alpha_r", "C_in", "C_out")
    out: dict[str, list[np.ndarray]] = {}
    for k in range(len(tab["rows"])):
        series = {n: est[n][k] for n in names}
        for lv, v in fe.finite_levels(series, est["ctx"][k], h, idx).items():
            out.setdefault(lv, []).append(v)
    return {lv: np.array(v) for lv, v in out.items()}


def feature_sets(tab: dict[str, Any], rng: np.random.Generator) -> dict[tuple[str, float], Any]:
    """(level, h) -> X. Static levels are computed once (h = 0) and shared by every horizon.
    L2N = L2 + L3's 9 geometry columns, each permuted across runs (execution note §3)."""
    sets: dict[tuple[str, float], np.ndarray] = {}
    for h in H:
        f = features(tab, h)
        for lv in DYNAMIC:
            if lv == "L2N":
                geo = f["L3"][:, len(fe.names("L2")) :]
                perm = np.column_stack([geo[rng.permutation(len(geo)), j]
                                        for j in range(geo.shape[1])])  # fmt: skip
                sets[(lv, h)] = np.column_stack([f["L2"], perm])
            else:
                sets[(lv, h)] = f[lv]
        if h == 0:
            for lv in STATIC:
                sets[(lv, 0.0)] = f[lv]
    return sets


def X_of(sets: dict[tuple[str, float], np.ndarray], lv: str, h: float) -> np.ndarray:
    return sets[(lv, 0.0)] if lv in STATIC else sets[(lv, h)]


# ------------------------------------------------------------------ model jobs
def _job(args: tuple[Any, ...]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    key, X, y, groups, folds, kind, family = args
    if kind == "multi":
        P, classes = pr.oof_multi(X, y, groups, folds, family)
        out: dict[str, Any] = {"P": P, "classes": classes}
    else:
        out = {"pred": pr.oof(X, y, groups, folds, kind, family)}
    if family == "linear":
        m = pr.fit(X, y, groups, kind)[-1]
        out["penalty"] = float(m.alpha_ if kind == "ridge" else np.ravel(m.C_)[0])
    return key, out


def run_jobs(jobs: list[tuple[Any, ...]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        return dict(pool.map(_job, jobs, chunksize=1))


# ------------------------------------------------------------------ bootstrap metrics
class Booter:
    """Pairwise metrics on shared hierarchical-bootstrap weights (paired across arms)."""

    def __init__(self, W: np.ndarray):
        self.W = W
        self.ones = np.ones((1, W.shape[1]))
        self._den: dict[Any, np.ndarray] = {}

    def pairwise(self, score: np.ndarray, target: np.ndarray, kind: str,
                 mask: np.ndarray | None = None) -> tuple[float, np.ndarray]:  # fmt: skip
        m = np.ones(len(score), bool) if mask is None else mask
        K, D = pr.pair_kernel(score[m], target[m], kind)
        point = float(pr.weighted_pairwise(self.ones[:, m], K, D)[0])
        return point, pr.weighted_pairwise(self.W[:, m], K, D)

    def classes(self, y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray,
                mask: np.ndarray | None = None) -> dict[str, Any]:  # fmt: skip
        m = np.ones(len(y_true), bool) if mask is None else mask
        f1p, bap = pr.weighted_class_metrics(self.ones[:, m], y_true[m], y_pred[m], classes)
        f1b, bab = pr.weighted_class_metrics(self.W[:, m], y_true[m], y_pred[m], classes)
        return {"f1": (float(f1p[0]), f1b), "bacc": (float(bap[0]), bab)}


def summ(pt: float, boot: np.ndarray) -> dict[str, float]:
    return {"point": pt, **{k: v for k, v in pr.summarize(pt, boot).items() if k != "point"}}


def diff(a: tuple[float, np.ndarray], b: tuple[float, np.ndarray]) -> dict[str, float]:
    return pr.summarize(a[0] - b[0], a[1] - b[1])


def binary_from_multi(P: np.ndarray, classes: np.ndarray, pos: tuple[str, ...],
                      neg: tuple[str, ...]) -> np.ndarray:  # fmt: skip
    ip = [list(classes).index(c) for c in pos]
    ineg = [list(classes).index(c) for c in neg]
    a, b = P[:, ip].sum(1), P[:, ineg].sum(1)
    return a / np.maximum(a + b, 1e-300)


# ------------------------------------------------------------------ analyses
def outcome_curves(tab: dict[str, Any], res: dict, bt: Booter, levels: tuple[str, ...],
                   family: str) -> dict[str, Any]:  # fmt: skip
    dn, fail = col(tab, "Dn"), col(tab, "failure")
    t_on = col(tab, "t_on")
    out: dict[str, Any] = {}
    for h in H:
        cur: dict[str, Any] = {}
        nyv = t_on > h
        for lv in levels:
            kh = 0.0 if lv in STATIC else h
            cd = bt.pairwise(res[(lv, kh, "ridge", family)]["pred"], dn, "cindex")
            au = bt.pairwise(res[(lv, kh, "binary", family)]["pred"], fail, "auroc")
            ny = bt.pairwise(res[(lv, kh, "binary", family)]["pred"], fail, "auroc", nyv)
            cur[lv] = {"cindex_Dn": cd, "auroc_failure": au, "auroc_nyv": ny}
        entry: dict[str, Any] = {lv: {m: summ(*v) for m, v in cur[lv].items()} for lv in levels}
        pairs = [("L2", "L0"), ("L3", "L2"), ("L3", "L1"), ("L2-G", "L0"), ("L2", "L2-G"),
                 ("L1", "L0"), ("L3r", "L3"), ("L2N", "L2"), ("L3", "L2N"),
                 ("L3+", "L2+")]  # fmt: skip
        entry["diffs"] = {
            f"{a}-{b}": {m: diff(cur[a][m], cur[b][m]) for m in cur[a]}
            for a, b in pairs
            if a in cur and b in cur
        }
        entry["counts"] = {"runs": int(len(fail)), "failures": int(fail.sum()),
                           "nyv_runs": int(nyv.sum()), "nyv_failures": int((fail & nyv).sum()),
                           "nyv_nonfailures": int((~fail & nyv).sum()),
                           "visible_failures": int((fail & ~nyv).sum())}  # fmt: skip
        out[f"{h:g}"] = entry
    return out


def rq1(curves: dict[str, Any]) -> dict[str, Any]:
    d = curves[f"{H_STAR:g}"]["diffs"]["L2-L0"]
    ps = np.array([d["cindex_Dn"]["p_one_sided"], d["auroc_failure"]["p_one_sided"]])
    adj = pr.holm(ps)
    out: dict[str, Any] = {}
    for j, m in enumerate(("cindex_Dn", "auroc_failure")):
        out[m] = {**d[m], "p_holm": float(adj[j]), "beats": pr.beats(d[m], pr.DELTA_OUT, adj[j])}
    out["pass"] = bool(all(out[m]["beats"] for m in ("cindex_Dn", "auroc_failure")))
    return out


def lead_times(tab: dict[str, Any], res: dict, levels: tuple[str, ...]) -> dict[str, Any]:
    fail = col(tab, "failure")
    success = col(tab, "category") == "SUCCESS"
    t_on = col(tab, "t_on")
    out: dict[str, Any] = {}
    for lv in levels:
        p = np.stack([res[(lv, 0.0 if lv in STATIC else h, "binary", "linear")]["pred"]
                      for h in H])  # fmt: skip
        per: dict[str, Any] = {}
        for h_obs in H[1:]:
            w = pr.warnings(p, H, success, fail, t_on, h_obs)
            lw = w["lead_warned"]
            per[f"{h_obs:g}"] = {
                "tau": w["tau"], "false_alarm": w["false_alarm"], "sensitivity": w["sensitivity"],
                "median_lead": w["median_lead"],
                "median_lead_conservative": w["median_lead_conservative"],
                "n_nyv_fail": w["n_nyv_fail"], "n_warned_nyv_fail": int(len(lw)),
                "lead_quantiles": np.quantile(lw, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
                if len(lw) else None,
                "t_warn_counts": {f"{h:g}": int(np.sum(w["t_warn"][fail & (t_on > h_obs)] == h))
                                  for h in H if h <= h_obs},
            }  # fmt: skip
        out[lv] = per
    return out


def mechanism(tab: dict[str, Any], res: dict, bt: Booter, levels: tuple[str, ...],
              family: str) -> dict[str, Any]:  # fmt: skip
    mech = col(tab, "mechanism")
    out: dict[str, Any] = {}
    for h in H:
        cur: dict[str, Any] = {}
        ent: dict[str, Any] = {}
        for lv in levels:
            r = res[(lv, 0.0 if lv in STATIC else h, "multi", family)]
            P, classes = r["P"], r["classes"]
            pred = classes[P.argmax(1)]
            cm = bt.classes(mech, pred, classes)
            ya_yb = np.isin(mech, ("YA", "YB"))
            ab = bt.pairwise(binary_from_multi(P, classes, ("YA",), ("YB",)), mech == "YA",
                             "auroc", ya_yb)  # fmt: skip
            bd = np.isin(mech, ("B", "D"))
            bvd = bt.pairwise(binary_from_multi(P, classes, ("D",), ("B",)), mech == "D", "auroc",
                              bd)  # fmt: skip
            xy = np.isin(mech, ("X", "YA", "YB"))
            xvy = bt.pairwise(binary_from_multi(P, classes, ("X",), ("YA", "YB")), mech == "X",
                              "auroc", xy)  # fmt: skip
            cur[lv] = {"macro_f1": cm["f1"], "balanced_accuracy": cm["bacc"], "route_ab_auroc": ab,
                       "b_vs_d_auroc": bvd, "x_vs_y_auroc": xvy}  # fmt: skip
            conf = {t: {p_: int(np.sum((mech == t) & (pred == p_))) for p_ in classes}
                    for t in classes}  # fmt: skip
            ent[lv] = {**{m: summ(*v) for m, v in cur[lv].items()}, "confusion": conf}
        pairs = [("L3", "L2"), ("L3", "L1"), ("L2", "L0"), ("L3+", "L2+"), ("L3r", "L3")]
        ent["diffs"] = {f"{a}-{b}": {m: diff(cur[a][m], cur[b][m]) for m in cur[a]}
                        for a, b in pairs if a in cur and b in cur}  # fmt: skip
        out[f"{h:g}"] = ent
    return out


def mech_value(mres: dict[str, Any]) -> dict[str, Any]:
    d = mres[f"{H_STAR:g}"]["diffs"]["L3-L2"]
    i = pr.beats(d["macro_f1"], pr.DELTA_MECH)
    ii = pr.beats(d["route_ab_auroc"], pr.DELTA_OUT)
    return {"i_macro_f1": {**d["macro_f1"], "beats": i},
            "ii_route_ab": {**d["route_ab_auroc"], "beats": ii},
            "iii": "evaluated in the hard-pair step (after the predictor freeze)"}  # fmt: skip


def within_mechanism(tab: dict[str, Any], sets: dict, bt: Booter) -> dict[str, Any]:
    mech, cons, g = col(tab, "mechanism"), col(tab, "construction"), col(tab, "i")
    dn, fail = col(tab, "Dn"), col(tab, "failure").astype(int)
    jobs = []
    for m in MECHS:
        sel = np.flatnonzero(mech == m)
        folds = pr.outer_folds(cons[sel], g[sel])
        both = min(fail[sel].sum(), (1 - fail[sel]).sum()) >= 10
        for h in (H_SEC[0], H_STAR, H_SEC[1]):
            for lv in ("L0", "L1", "L2", "L3"):
                X = X_of(sets, lv, h)[sel]
                jobs.append(((m, h, lv, "ridge"), X, dn[sel], g[sel], folds, "ridge", "linear"))
                if both:
                    jobs.append(((m, h, lv, "binary"), X, fail[sel], g[sel], folds, "binary",
                                 "linear"))  # fmt: skip
    res = run_jobs(jobs)
    out: dict[str, Any] = {}
    for m in MECHS:
        sel = mech == m
        per: dict[str, Any] = {}
        for h in (H_SEC[0], H_STAR, H_SEC[1]):
            cur: dict[str, Any] = {}
            for lv in ("L0", "L1", "L2", "L3"):
                ent: dict[str, Any] = {}
                pr_dn = np.full(len(dn), np.nan)
                pr_dn[sel] = res[(m, h, lv, "ridge")]["pred"]
                ent["cindex_Dn"] = bt.pairwise(np.nan_to_num(pr_dn), dn, "cindex", sel)
                if (m, h, lv, "binary") in res:
                    pf = np.zeros(len(dn))
                    pf[sel] = res[(m, h, lv, "binary")]["pred"]
                    ent["auroc_failure"] = bt.pairwise(pf, fail.astype(bool), "auroc", sel)
                cur[lv] = ent
            per[f"{h:g}"] = {
                **{lv: {k: summ(*v) for k, v in cur[lv].items()} for lv in cur},
                "L3-L2": {k: diff(cur["L3"][k], cur["L2"][k]) for k in cur["L3"]},
                "L3-L1": {k: diff(cur["L3"][k], cur["L1"][k]) for k in cur["L3"]},
                "L2-L0": {k: diff(cur["L2"][k], cur["L0"][k]) for k in cur["L2"]},
                "n_runs": int(sel.sum()), "n_fail": int(fail[sel].sum()),
            }  # fmt: skip
        out[m] = per
    return out


def type_oracles(tab: dict[str, Any], bt: Booter) -> dict[str, Any]:
    dn, fail, g = col(tab, "Dn"), col(tab, "failure"), col(tab, "i")
    mech, axis = col(tab, "mechanism"), col(tab, "axis")
    out: dict[str, Any] = {}
    both = np.char.add(mech.astype(str), axis.astype(str))
    for name, key in (("mechanism", mech), ("mechanism_x_axis", both)):
        pred = np.array([dn[(key == key[k]) & (g != g[k])].mean() for k in range(len(dn))])
        out[name] = {"cindex_Dn": summ(*bt.pairwise(pred, dn, "cindex")),
                     "auroc_failure": summ(*bt.pairwise(pred, fail, "auroc"))}  # fmt: skip
        out[name]["_pred"] = pred
    return out


def cross_construction(tab: dict[str, Any], sets: dict, bt: Booter) -> dict[str, Any]:
    mech, slot, g = col(tab, "mechanism"), col(tab, "slot"), col(tab, "i")
    folds = [(np.flatnonzero(slot != k), np.flatnonzero(slot == k)) for k in (1, 2, 3)]
    jobs = [((lv, h, "multi"), X_of(sets, lv, h), mech, g, folds, "multi", "linear")
            for h in (H_SEC[0], H_STAR, H_SEC[1]) for lv in ("L0", "L1", "L2", "L3")]  # fmt: skip
    res = run_jobs(jobs)
    out: dict[str, Any] = {}
    for h in (H_SEC[0], H_STAR, H_SEC[1]):
        cur = {}
        ent: dict[str, Any] = {}
        for lv in ("L0", "L1", "L2", "L3"):
            r = res[(lv, h, "multi")]
            pred = r["classes"][r["P"].argmax(1)]
            cm = bt.classes(mech, pred, r["classes"])
            cur[lv] = cm["f1"]
            fold_f1 = [float(pr.weighted_class_metrics(np.ones((1, len(te))), mech[te], pred[te],
                                                       r["classes"])[0][0])
                       for _, te in folds]  # fmt: skip
            ent[lv] = {"pooled_macro_f1": summ(*cm["f1"]), "fold_macro_f1": fold_f1,
                       "mean_fold_macro_f1": float(np.mean(fold_f1))}  # fmt: skip
        ent["L3-L2"] = diff(cur["L3"], cur["L2"])
        ent["L3-L0"] = diff(cur["L3"], cur["L0"])
        ent["L2-L0"] = diff(cur["L2"], cur["L0"])
        out[f"{h:g}"] = ent
    return out


def lomo(tab: dict[str, Any], sets: dict, bt: Booter) -> dict[str, Any]:
    """Design-side leave-one-mechanism-out outcome diagnostic (not the §12 split B)."""
    mech, g = col(tab, "mechanism"), col(tab, "i")
    dn, fail = col(tab, "Dn"), col(tab, "failure").astype(int)
    jobs = []
    for m in MECHS:
        folds = [(np.flatnonzero(mech != m), np.flatnonzero(mech == m))]
        for lv in ("L0", "L2"):
            X = X_of(sets, lv, H_STAR)
            jobs.append(((m, lv, "ridge"), X, dn, g, folds, "ridge", "linear"))
            jobs.append(((m, lv, "binary"), X, fail, g, folds, "binary", "linear"))
    res = run_jobs(jobs)
    out: dict[str, Any] = {}
    for m in MECHS:
        sel = mech == m
        ent: dict[str, Any] = {"n_fail": int(fail[sel].sum()), "n_runs": int(sel.sum())}
        cur: dict[str, Any] = {}
        for lv in ("L0", "L2"):
            cur[lv] = {"cindex_Dn": bt.pairwise(np.nan_to_num(res[(m, lv, "ridge")]["pred"]), dn,
                                                "cindex", sel)}  # fmt: skip
            if min(fail[sel].sum(), (1 - fail[sel]).sum()) >= 10:
                cur[lv]["auroc_failure"] = bt.pairwise(
                    np.nan_to_num(res[(m, lv, "binary")]["pred"]), fail.astype(bool), "auroc", sel
                )
            ent[lv] = {k: summ(*v) for k, v in cur[lv].items()}
        ent["L2-L0"] = {k: diff(cur["L2"][k], cur["L0"][k]) for k in cur["L2"]}
        out[m] = ent
    return out


def single_diagnostics(tab: dict[str, Any], sets: dict) -> dict[str, Any]:
    dn, fail, t_on = col(tab, "Dn"), col(tab, "failure"), col(tab, "t_on")
    names2, names3 = fe.names("L2"), fe.names("L3")
    L0 = X_of(sets, "L0", 0.0)
    L1 = X_of(sets, "L1", 0.0)
    out: dict[str, Any] = {}
    for h in H:
        L3 = X_of(sets, "L3", h)
        L2 = X_of(sets, "L2", h)
        cand = {
            "L0 FPR(0)": L0[:, 2],
            "L1 C0/A0": L1[:, 7] / np.maximum(L1[:, 5], 1e-12),
            "L2 dFPR": L2[:, names2.index("FPR_d")],
            "L2 -dJ_G": -L2[:, names2.index("J_G_d")],
            "L3 dC": L3[:, names3.index("C_d")],
            "L3 -dalpha": -L3[:, names3.index("alpha_d")],
        }
        nyv = t_on > h
        ent = {}
        for k, s in cand.items():
            s = np.nan_to_num(s, nan=np.nanmedian(s))
            ent[k] = {"cindex_Dn": ep.c_index(s, dn), "auroc_failure": ep.auroc(s, fail),
                      "auroc_nyv": ep.auroc(s[nyv], fail[nyv])
                      if 0 < fail[nyv].sum() < nyv.sum() else None}  # fmt: skip
        out[f"{h:g}"] = ent
    return out


def estimator_sanity(tab: dict[str, Any]) -> dict[str, Any]:
    """Finite-sample estimates vs the exact values at the same states (execution note §2)."""
    est, ex = tab["est"], tab["exact"]
    pairs = {"J_G": ex["obs"][..., 0], "J_V": ex["obs"][..., 1], "FPR": ex["obs"][..., 2],
             "FNR": ex["obs"][..., 3], "A_u": ex["geo_u"][..., 0], "alpha_u": ex["geo_u"][..., 1],
             "C_u": ex["geo_u"][..., 2], "alpha_r": ex["geo_r"][..., 1],
             "C_r": ex["geo_r"][..., 2]}  # fmt: skip
    out: dict[str, Any] = {}
    for j, f in enumerate(s1r.FEAT_FRACS):
        ent = {}
        for k, truth in pairs.items():
            e, t = est[k][:, j], truth[:, j]
            ok = np.isfinite(e) & np.isfinite(t)
            err = e[ok] - t[ok]
            ent[k] = {"nan_rate": float(1 - ok.mean()), "bias": float(err.mean()),
                      "sd_error": float(err.std()), "median_abs_error": float(np.median(abs(err))),
                      "pearson": float(np.corrcoef(e[ok], t[ok])[0, 1]) if ok.sum() > 2 else None,
                      "mean_truth": float(t[ok].mean()),
                      "mean_est": float(e[ok].mean())}  # fmt: skip
        # registered audit SE vs the empirical error SD (observables)
        jg, fpr, fnr = pairs["J_G"][:, j], pairs["FPR"][:, j], pairs["FNR"][:, j]
        se = {"J_G": np.sqrt(jg * (1 - jg) / 256),
              "FPR": np.sqrt(fpr * (1 - fpr) / np.maximum(256 * (1 - jg), 1)),
              "FNR": np.sqrt(fnr * (1 - fnr) / np.maximum(256 * jg, 1))}  # fmt: skip
        for k, s in se.items():
            e = est[k][:, j] - pairs[k][:, j]
            ok = np.isfinite(e)
            ent[k]["z_sd"] = float(np.std(e[ok] / np.maximum(s[ok], 1e-12)))
        out[f"{f:g}"] = ent
    out["alpha_u_undefined_rate"] = float(1 - np.mean(est["alpha_u_defined"]))
    out["alpha_r_undefined_rate"] = float(1 - np.mean(est["alpha_r_defined"]))
    out["FPR_undefined_rate"] = float(np.mean(~np.isfinite(est["FPR"])))
    out["FNR_undefined_rate"] = float(np.mean(~np.isfinite(est["FNR"])))
    return out


def nan_counts(sets: dict) -> dict[str, Any]:
    return {f"{lv}|{h:g}": int(np.isnan(X).sum()) for (lv, h), X in sets.items()}


def gbm_check(tab: dict[str, Any], res: dict, bt: Booter) -> dict[str, Any]:
    dn, fail, mech = col(tab, "Dn"), col(tab, "failure"), col(tab, "mechanism")
    out: dict[str, Any] = {}
    for h in H:
        ent: dict[str, Any] = {}
        cur: dict[str, Any] = {}
        for lv in PRIMARY:
            kh = 0.0 if lv in STATIC else h
            r = res[(lv, kh, "multi", "gbm")]
            pred = r["classes"][r["P"].argmax(1)]
            cur[lv] = {"cindex_Dn": bt.pairwise(res[(lv, kh, "ridge", "gbm")]["pred"], dn,
                                                "cindex"),
                       "auroc_failure": bt.pairwise(res[(lv, kh, "binary", "gbm")]["pred"], fail,
                                                    "auroc"),
                       "macro_f1": bt.classes(mech, pred, r["classes"])["f1"]}  # fmt: skip
            ent[lv] = {k: summ(*v) for k, v in cur[lv].items()}
        ent["L3-L2"] = {k: diff(cur["L3"][k], cur["L2"][k]) for k in cur["L3"]}
        ent["L3-L1"] = {k: diff(cur["L3"][k], cur["L1"][k]) for k in cur["L3"]}
        ent["L2-L0"] = {k: diff(cur["L2"][k], cur["L0"][k]) for k in cur["L2"]}
        out[f"{h:g}"] = ent
    return out


# ------------------------------------------------------------------ figures
def fig_information(curves: dict[str, Any], mech: dict[str, Any], path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    xs = [100 * h for h in H]
    panels = (("cindex_Dn", "C-index (Dn)", curves), ("auroc_failure", "AUROC failure", curves),
              ("auroc_nyv", "AUROC, not-yet-visible failure", curves),
              ("macro_f1", "mechanism macro-F1", mech))  # fmt: skip
    for ax, (metric, lbl, src) in zip(axes, panels, strict=True):
        for lv in PRIMARY:
            ys = [src[f"{h:g}"][lv][metric]["point"] for h in H]
            lo = [src[f"{h:g}"][lv][metric]["ci2_lo"] for h in H]
            hi = [src[f"{h:g}"][lv][metric]["ci2_hi"] for h in H]
            (ln,) = ax.plot(xs, ys, marker="o", ms=3, lw=1.5, label=lv)
            ax.fill_between(xs, lo, hi, color=ln.get_color(), alpha=0.12, lw=0)
        ax.axvline(100 * H_STAR, color="0.6", lw=0.8, ls=":")
        ax.set_xscale("symlog", linthresh=0.2)
        ax.set_xlabel("% of T observed")
        ax.set_title(lbl, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_leads(tab: dict[str, Any], res: dict, path: Path) -> None:
    fail, t_on = col(tab, "failure"), col(tab, "t_on")
    success = col(tab, "category") == "SUCCESS"
    p = np.stack([res[("L2", h, "binary", "linear")]["pred"] for h in H])
    w = pr.warnings(p, H, success, fail, t_on, H_STAR)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ton = t_on[fail & np.isfinite(t_on)]
    axes[0].hist(100 * ton, bins=np.linspace(0, 100, 51), color="0.4")
    axes[0].axvline(100 * H_STAR, color="C3", lw=1)
    axes[0].set_xlabel("visible onset t_on (% of T), failing runs")
    axes[1].hist(100 * w["lead_warned"], bins=np.linspace(0, 100, 41), color="C0")
    axes[1].axvline(5, color="C3", lw=1)
    axes[1].set_xlabel("lead time (% of T), warned NYV failures at h* (L2)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main
def fit_all(tab: dict[str, Any], sets: dict, levels_linear: tuple[str, ...],
            levels_gbm: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:  # fmt: skip
    mech, g = col(tab, "mechanism"), col(tab, "i")
    dn, fail = col(tab, "Dn"), col(tab, "failure").astype(int)
    folds = pr.outer_folds(mech, g)
    jobs = []
    for family, levels in (("linear", levels_linear), ("gbm", levels_gbm)):
        for lv in levels:
            for h in [0.0] if lv in STATIC else H:
                X = X_of(sets, lv, h)
                for kind, y in (("ridge", dn), ("binary", fail), ("multi", mech)):
                    jobs.append(((lv, h, kind, family), X, y, g, folds, kind, family))
    jobs.sort(key=lambda j: (j[5] != "multi", j[6] != "gbm"))  # long jobs first
    return run_jobs(jobs)


def strip(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def main(argv: list[str]) -> int:
    run_dir = Path(argv[1]).resolve()
    cfg = provenance.load_config(CONFIG)
    assert cfg["horizons"]["fractions"] == H and cfg["inference"]["B"] == B
    d = load(run_dir)
    assert d["summary"]["panel_sha256"] == cfg["panel_sha256"]
    assert hashlib.sha256(PANEL.read_bytes()).hexdigest() == cfg["panel_sha256"]
    out_dir = provenance.create_run_dir(REPO / "results", "E004a-stage1-analysis", REPO)
    provenance.write_metadata(out_dir, "E004a-stage1-analysis", CONFIG, REPO,
                              extra={"split": "design", "root_seed": pn.ROOT_SEED,
                                     "panel_sha256": cfg["panel_sha256"],
                                     "runs": str(run_dir.relative_to(REPO))})  # fmt: skip
    resample = s1r.audit_tree(len(d["structs"]))["resample"].spawn(4)
    rng_perm = np.random.default_rng(resample[0])
    t_start = time.perf_counter()
    report: dict[str, Any] = {}
    frozen_models: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    for opt in ("adam", "ng"):
        t0 = time.perf_counter()
        tab = table(d, opt)
        sets = feature_sets(tab, rng_perm)
        W = pr.hier_weights(col(tab, "i"), B,
                            np.random.default_rng(resample[1 if opt == "adam" else 2]))  # fmt: skip
        bt = Booter(W)
        lin = tuple(STATIC + DYNAMIC) if opt == "adam" else PRIMARY
        gbm = PRIMARY if opt == "adam" else ()
        res = fit_all(tab, sets, lin, gbm)
        print(f"[{opt}] models fitted: {time.perf_counter() - t0:.0f} s", flush=True)
        curves = outcome_curves(tab, res, bt, lin, "linear")
        mres = mechanism(tab, res, bt, lin, "linear")
        rep: dict[str, Any] = {"information": curves, "mechanism": mres, "rq1": rq1(curves),
                               "mechanistic_value_i_ii": mech_value(mres),
                               "lead_times": lead_times(tab, res, PRIMARY),
                               "type_oracles": type_oracles(tab, bt),
                               "nan_counts": nan_counts(sets)}  # fmt: skip
        nyv = curves[f"{H_STAR:g}"]["L2"]["auroc_nyv"]
        lead = rep["lead_times"]["L2"][f"{H_STAR:g}"]
        rep["early_warning"] = {"nyv_auroc_L2": nyv, "lo95_ge_0.65": nyv["lo95"] >= 0.65,
                                "median_lead_L2": lead["median_lead"],
                                "median_lead_ge_0.05": bool(lead["median_lead"] >= 0.05),
                                "pass": bool(nyv["lo95"] >= 0.65
                                             and lead["median_lead"] >= 0.05)}  # fmt: skip
        if opt == "adam":
            print("[adam] within / cross / lomo / singles / gbm ...", flush=True)
            rep["within_mechanism"] = within_mechanism(tab, sets, bt)
            rep["cross_construction"] = cross_construction(tab, sets, bt)
            rep["lomo_design"] = lomo(tab, sets, bt)
            rep["single_diagnostics"] = single_diagnostics(tab, sets)
            rep["gbm"] = gbm_check(tab, res, bt)
            rep["estimator_sanity"] = estimator_sanity(tab)
            fig_information(
                curves,
                mres,
                out_dir / "fig_information_adam.png",
                "E004a Stage 1 (design CV), sampled Adam, finite-sample features",
            )
            fig_leads(tab, res, out_dir / "fig_leadtime_adam.png")
            frozen_models = {f"{lv}|{h:g}|{kind}": r["penalty"]
                             for (lv, h, kind, fam), r in res.items()
                             if fam == "linear"}  # fmt: skip
            thresholds = {lv: {h: v["tau"] for h, v in rep["lead_times"][lv].items()}
                          for lv in PRIMARY}  # fmt: skip
        else:
            rep["estimator_sanity"] = estimator_sanity(tab)
            fig_information(
                curves,
                mres,
                out_dir / "fig_information_ng.png",
                "E004a Stage 1 (design CV), NG (secondary), finite-sample features",
            )
        report[opt] = rep
        print(f"[{opt}] done: {time.perf_counter() - t0:.0f} s", flush=True)
    report["timing_s"] = time.perf_counter() - t_start
    (out_dir / "stage1_design.json").write_text(json.dumps(strip(report), indent=1) + "\n")
    frozen = {
        "registry": "E004a — Stage 1 execution note (design round)",
        "config": str(CONFIG.relative_to(REPO)),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "panel_sha256": cfg["panel_sha256"], "root_seed": pn.ROOT_SEED,
        "runs": str(run_dir.relative_to(REPO)),
        "levels": {lv: fe.names(lv) for lv in ("L0", "L1", "L1r", "L2-G", "L2", "L2+", "L3", "L3r",
                                                 "L3+")},
        "L2N": "L2 + 9 permuted L3 geometry columns (resample branch 0)",
        "geometry": "update level (A_u, alpha_u, C_u); alpha_r only in L1r/L3r",
        "preprocessing": "median imputation (training data) -> StandardScaler -> model",
        "missing_values": "NaN kept in features; median-imputed inside the pipeline; no indicators",
        "models": {"Dn": "RidgeCV", "failure": "LogisticRegressionCV (L2, log-loss)",
                   "mechanism": "LogisticRegressionCV multinomial (log-loss)",
                   "penalty_grid": pr.GRID.tolist(), "inner_cv": "GroupKFold(5) by structure",
                   "secondary": {"family": "GradientBoosting", **pr.GBM}},
        "primary_family": "linear (registered; not selected on design results)",
        "outer_cv": {"kind": "StratifiedGroupKFold", "k": pr.K_OUTER, "seed": pr.FOLD_SEED,
                     "strata": "mechanism", "groups": "structure"},
        "frozen_fit": "the pipeline refit on all kept design runs at each horizon",
        "full_design_penalties": frozen_models,
        "warning": {"procedure": "running max over registered h' <= h; tau_h = 90th percentile "
                                 "over design SUCCESS runs (out-of-fold); t_warn = first h' with "
                                 "p > tau_h", "tau_adam": thresholds},
        "inference": cfg["inference"],
        "seed_tree": cfg["runs"]["seed_tree"],
    }  # fmt: skip
    (out_dir / "predictors_frozen.json").write_text(json.dumps(strip(frozen), indent=1) + "\n")
    print(f"run directory: {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
