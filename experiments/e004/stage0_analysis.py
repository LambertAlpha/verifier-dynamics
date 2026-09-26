"""E004a Stage 0, step 3 (registry E004a §8-§10): maps, gates, leakage, oracle ceilings.

DESIGN SPLIT ONLY. The fitted models here are cross-validated *oracle ceilings and diagnostics*
on exact observables; they are not the Stage 1 predictors.
"""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV, RidgeCV  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e002 import endpoints as ep  # noqa: E402
from vdyn.e004 import dynamics as dy  # noqa: E402
from vdyn.e004 import features as fe  # noqa: E402
from vdyn.e004 import panel as pn  # noqa: E402
from vdyn.e004 import toy  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PANEL = REPO / "configs" / "e004" / "design_panel.json"
CONFIG = REPO / "configs" / "e004" / "e004a.toml"
H = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
H_STAR = 0.02
SET1 = ("R1", "X1", "YA1", "YB1", "B1", "D1")
C_GRID = np.logspace(-3, 3, 13)


# ---------------------------------------------------------------- data assembly
def load(run_dir: Path) -> dict[str, Any]:
    arr = np.load(run_dir / "runs.npz")
    summ = json.loads((run_dir / "summary.json").read_text())
    lab = json.loads((run_dir / "labels.json").read_text())
    structs = [toy.Structure.from_dict(d) for d in json.loads(PANEL.read_text())["structures"]]
    excluded = set(summ["excluded_low_clean_gain"])  # registry §4: clean gain < 0.1
    keep = [s.sid not in excluded for s in structs]
    return {"arr": arr, "summary": summ, "labels": lab, "structs": structs, "keep": keep}


def hidx(fracs: list[float], h: float) -> tuple[int, int, int]:
    return fracs.index(0.0), fracs.index(h / 2 if h > 0 else 0.0), fracs.index(h)


def run_table(d: dict[str, Any], group: str, lab_key: str, seeds: bool) -> dict[str, Any]:
    """Flatten runs (structure x seed) with labels."""
    obs, ctx = d["arr"][f"{group}__obs"], d["arr"][f"{group}__ctx"]
    geo = d["arr"][f"{group}__geo"]
    if not seeds:
        obs, ctx, geo = obs[:, None], ctx[:, None], geo[:, None]
    n, s = obs.shape[:2]
    labs = d["labels"][lab_key]
    rows = []
    kept = [i for i in range(n) if d["keep"][i]]
    for i in kept:
        for r in range(s):
            lb = labs[i][r]
            rows.append({"i": i, "r": r, "sid": d["structs"][i].sid,
                         "construction": d["structs"][i].construction,
                         "mechanism": d["structs"][i].mechanism, **lb})  # fmt: skip
    obs, ctx, geo = obs[kept], ctx[kept], geo[kept]
    m = len(kept) * s
    return {"obs": obs.reshape(m, *obs.shape[2:]), "ctx": ctx.reshape(m, *ctx.shape[2:]),
            "geo": geo.reshape(m, *geo.shape[2:]), "rows": rows}  # fmt: skip


def features(tab: dict[str, Any], fracs: list[float], h: float, level: str) -> np.ndarray:
    idx = hidx(fracs, h)
    out = []
    for k in range(len(tab["rows"])):
        o = {name: tab["obs"][k, :, j] for j, name in enumerate(("J_G", "J_V", "FPR", "FNR"))}
        g = {name: tab["geo"][k, :, j]
             for j, name in enumerate(("A", "alpha", "C", "C_in", "C_out"))}  # fmt: skip
        out.append(fe.level_features(o, g, tab["ctx"][k], h, idx)[level])
    return np.array(out)


# ---------------------------------------------------------------- cross-validated models
def _folds(y_strat: np.ndarray, groups: np.ndarray, k: int = 5) -> list[tuple[np.ndarray, ...]]:
    return list(
        StratifiedGroupKFold(k, shuffle=True, random_state=0).split(y_strat, y_strat, groups)
    )


def _inner(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(GroupKFold(3).split(groups, groups, groups))


def cv_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, strat: np.ndarray,
               kind: str) -> np.ndarray:  # fmt: skip
    """Out-of-fold predictions: 'ridge' (continuous), 'binary' (probability), 'multi' (labels)."""
    pred: np.ndarray = np.zeros(len(y), dtype=object if kind == "multi" else float)
    for tr, te in _folds(strat, groups):
        inner = _inner(groups[tr])
        if kind == "ridge":
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=C_GRID, cv=inner))
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        else:
            m = make_pipeline(StandardScaler(),
                              LogisticRegressionCV(Cs=C_GRID, cv=inner, max_iter=3000))  # fmt: skip
            m.fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1] if kind == "binary" else m.predict(X[te])
    return pred


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    return float(f1_score(y, p, average="macro"))


# ---------------------------------------------------------------- analyses
def outcome_map(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("adam_prim", "ng_prim", "mf_prim", "adam_can", "ng_can"):
        labs = d["labels"][key]
        per: dict[str, Any] = {}
        for scope in (*pn.CONSTRUCTIONS, *pn.MECHANISMS):
            sel = [lb for i, st in enumerate(d["structs"])
                   if d["keep"][i] and scope in (st.construction, st.mechanism)
                   for lb in labs[i]]  # fmt: skip
            cats = {c: sum(lb["category"] == c for lb in sel) / len(sel)
                    for c in ("SUCCESS", "SLOW", "STALL", "DECLINE")}  # fmt: skip
            per[scope] = {
                "n_runs": len(sel),
                "failure": float(np.mean([lb["failure"] for lb in sel])),
                "Dn_median": float(np.median([lb["Dn"] for lb in sel])),
                **cats,
            }
        out[key] = per
    # MF-Adam vs sampled Adam, per structure
    agree_bin, agree_cat, dn_pairs = [], [], []
    for i in range(len(d["structs"])):
        if not d["keep"][i]:
            continue
        mf = d["labels"]["mf_prim"][i][0]
        sa = d["labels"]["adam_prim"][i]
        maj_fail = np.mean([lb["failure"] for lb in sa]) >= 0.5
        cat_list = [lb["category"] for lb in sa]
        agree_bin.append(mf["failure"] == maj_fail)
        agree_cat.append(mf["category"] == max(set(cat_list), key=cat_list.count))
        dn_pairs.append((mf["Dn"], float(np.mean([lb["Dn"] for lb in sa]))))
    a = np.array(dn_pairs)
    out["mf_vs_sampled"] = {"binary_agreement": float(np.mean(agree_bin)),
                            "category_agreement": float(np.mean(agree_cat)),
                            "Dn_kendall": float(ep.kendall_tau_b(a[:, 0], a[:, 1]))}  # fmt: skip
    return out


def gates_and_predictions(
    d: dict[str, Any],
    omap: dict[str, Any],
    leak: dict[str, Any],
    cross: dict[str, Any],
    nulls: dict[str, Any],
) -> dict[str, Any]:
    am = omap["adam_prim"]
    g2 = {m: {"failure": am[m]["failure"], "ok": 0.1 <= am[m]["failure"] <= 0.9}
          for m in pn.MECHANISMS}  # fmt: skip
    g2_pass = all(g2[m]["ok"] for m in ("X", "YA", "YB", "D"))
    ng = omap["ng_prim"]
    fail_y = lambda om: np.mean([om[m]["failure"] for m in ("YA", "YB")])  # noqa: E731
    return {
        "G1_family_recognition": {
            "macro_f1": leak["L0_mechanism_logistic"],
            "pass": leak["L0_mechanism_logistic"] <= 0.35,
        },  # fmt: skip
        "G2_outcome_diversity": {"per_mechanism": g2, "pass": g2_pass},
        "G3_theory_nulls": {
            **nulls,
            "pass": nulls["F1_max_dev"] <= 1e-8 and nulls["F2_max_dev"] <= 1e-6,
        },  # fmt: skip
        "G4_cross_construction": {
            **cross,
            "pass": cross["best_cross_f1"] >= 0.33
            and cross["best_cross_f1"] >= 0.6 * cross["within_f1_of_best"],
        },  # fmt: skip
        "S0-P3_YAB_fail_lower_under_adam": {
            "adam": fail_y(am),
            "ng": fail_y(ng),
            "holds": fail_y(am) < fail_y(ng),
        },  # fmt: skip
        "S0-P4_D_fails_both": {
            "adam": am["D"]["failure"],
            "ng": ng["D"]["failure"],
            "holds": am["D"]["failure"] >= 0.5 and ng["D"]["failure"] >= 0.5,
        },  # fmt: skip
        "S0-P5_B_rarely_fails": {
            "adam": am["B"]["failure"],
            "ng": ng["B"]["failure"],
            "holds": am["B"]["failure"] <= 0.1 and ng["B"]["failure"] <= 0.1,
        },  # fmt: skip
    }


def leakage(d: dict[str, Any], tab: dict[str, Any], fracs: list[float]) -> dict[str, Any]:
    first = [k for k, r in enumerate(tab["rows"]) if r["r"] == 0]  # one row per structure
    X = features(tab, fracs, 0.0, "L0")[first]
    mech = np.array([tab["rows"][k]["mechanism"] for k in first])
    cons = np.array([tab["rows"][k]["construction"] for k in first])
    groups = np.arange(len(first))
    out: dict[str, Any] = {}
    for name, y in (("mechanism", mech), ("construction", cons)):
        p = cv_predict(X, y, groups, y, "multi")
        out[f"L0_{name}_logistic"] = macro_f1(y, p)
        pg = np.zeros(len(y), dtype=object)
        for tr, te in _folds(y, groups):
            gb = GradientBoostingClassifier(max_depth=2, n_estimators=100, random_state=0)
            pg[te] = gb.fit(X[tr], y[tr]).predict(X[te])
        out[f"L0_{name}_gbm"] = macro_f1(y, pg)
    # type oracle: leave-one-structure-out mechanism-mean Dn, over runs
    dn = np.array([r["Dn"] for r in tab["rows"]])
    fail = np.array([r["failure"] for r in tab["rows"]])
    m_all = np.array([r["mechanism"] for r in tab["rows"]])
    s_all = np.array([r["i"] for r in tab["rows"]])
    oracle = np.array(
        [dn[(m_all == m_all[k]) & (s_all != s_all[k])].mean() for k in range(len(dn))]
    )
    out["type_oracle_cindex_Dn"] = ep.c_index(oracle, dn)
    out["type_oracle_auroc_failure"] = ep.auroc(oracle, fail)
    return out


def ceilings(tab: dict[str, Any], fracs: list[float]) -> dict[str, Any]:
    rows = tab["rows"]
    dn = np.array([r["Dn"] for r in rows])
    fail = np.array([r["failure"] for r in rows]).astype(int)
    mech = np.array([r["mechanism"] for r in rows])
    groups = np.array([r["i"] for r in rows])
    t_on = np.array([np.inf if r["t_on"] is None else r["t_on"] for r in rows])
    yayb = np.isin(mech, ("YA", "YB"))
    out: dict[str, Any] = {}
    for h in H:
        for level in fe.LEVELS:
            X = features(tab, fracs, h, level)
            p_dn = cv_predict(X, dn, groups, mech, "ridge")
            p_f = cv_predict(X, fail, groups, mech, "binary")
            p_m = cv_predict(X, mech, groups, mech, "multi")
            nyv = t_on > h
            res = {"cindex_Dn": ep.c_index(p_dn, dn),
                   "auroc_failure": ep.auroc(p_f, fail.astype(bool)),
                   "auroc_not_yet_visible": ep.auroc(p_f[nyv], fail[nyv].astype(bool))
                   if 0 < fail[nyv].sum() < nyv.sum() else None,
                   "n_not_yet_visible_fail": int(fail[nyv].sum()),
                   "mechanism_macro_f1": macro_f1(mech, p_m)}  # fmt: skip
            Xy, yy = X[yayb], (mech[yayb] == "YA").astype(int)
            p_ab = cv_predict(Xy, yy, groups[yayb], mech[yayb], "binary")
            res["auroc_YA_vs_YB"] = ep.auroc(p_ab, yy.astype(bool))
            out[f"{h:g}|{level}"] = res
        print(f"  ceilings h={h:g} done")
    return out


def single_variable(tab: dict[str, Any], fracs: list[float]) -> dict[str, Any]:
    rows = tab["rows"]
    dn = np.array([r["Dn"] for r in rows])
    fail = np.array([r["failure"] for r in rows])
    out: dict[str, Any] = {}
    for h in H:
        l3 = features(tab, fracs, h, "L3")
        col = {n: l3[:, j] for j, n in enumerate(fe.names("L3"))}
        l1 = features(tab, fracs, 0.0, "L1")
        c0_over_a0 = l1[:, 7] / np.maximum(l1[:, 5], 1e-12)
        sig = {
            "L0 FPR0": col["FPR0"],
            "L1 C0/A0": c0_over_a0,
            "L2 dFPR": col["FPR_d"],
            "L2 -dJ_G": -col["J_G_d"],
            "L3 dC": col["C_d"],
            "L3 -dalpha": -col["alpha_d"],
        }
        out[f"{h:g}"] = {k: {"cindex_Dn": ep.c_index(v, dn), "auroc_failure": ep.auroc(v, fail)}
                         for k, v in sig.items()}  # fmt: skip
    return out


def cross_construction(tab: dict[str, Any], fracs: list[float]) -> dict[str, Any]:
    rows = tab["rows"]
    mech = np.array([r["mechanism"] for r in rows])
    cons = np.array([r["construction"] for r in rows])
    groups = np.array([r["i"] for r in rows])
    s1 = np.isin(cons, SET1)
    out: dict[str, Any] = {}
    for level in ("L2", "L3", "L2+", "L3+"):
        X = features(tab, fracs, H_STAR, level)
        within = macro_f1(mech, cv_predict(X, mech, groups, mech, "multi"))
        f1s = []
        for tr, te in ((s1, ~s1), (~s1, s1)):
            inner = _inner(groups[tr])
            m = make_pipeline(
                StandardScaler(), LogisticRegressionCV(Cs=C_GRID, cv=inner, max_iter=3000)
            )
            m.fit(X[tr], mech[tr])
            f1s.append(macro_f1(mech[te], m.predict(X[te])))
        out[level] = {"within_cv_f1": within, "cross_f1": float(np.mean(f1s)), "cross_folds": f1s}
    best = max(("L2", "L3"), key=lambda lv: out[lv]["cross_f1"])
    out["best_level"] = best
    out["best_cross_f1"] = out[best]["cross_f1"]
    out["within_f1_of_best"] = out[best]["within_cv_f1"]
    return out


def theory_nulls(d: dict[str, Any], T_ng: float) -> dict[str, Any]:
    """F1 identity on canonical YA1/YB1 NG trajectories; F2 NG invariance (primary vs uncoupled)."""
    dev = 0.0
    for st in d["structs"]:
        if st.construction not in ("YA1", "YB1"):
            continue
        can = pn.canonical_twin(st)
        tb = toy.Tables.of([can])
        sol = solve_ivp(lambda t, th, tb=tb: dy._ng_velocity(tb, th), (0, T_ng), can.theta0,
                        t_eval=np.linspace(0, T_ng, 6), rtol=1e-9, atol=1e-11)  # fmt: skip
        for th in sol.y.T:
            ex = toy.exact(tb, th[None])
            F = ex["F"][0] + dy.NG_DAMP * np.trace(ex["F"][0]) / toy.D * np.eye(toy.D)
            Minv = np.linalg.inv(F)
            vel = Minv @ ex["g_V"][0]
            geo = toy.decompose(ex["g_G"][0], ex["g_V"][0], Minv)
            fpr, jg = ex["FPR"][0], ex["J_G"][0]
            pred = (-fpr, ex["g_G"][0] @ vel / (1 - fpr), (1 - jg) * ex["g_FPR"][0] @ vel)
            got = (geo["alpha"], geo["A"] ** 2, geo["C"] ** 2)
            dev = max(dev, *(abs(p - g) / (abs(g) + 1e-12) for p, g in zip(pred, got, strict=True)))
    a, b = d["arr"]["ng_prim_ver__obs"], d["arr"]["ng_unc_ver__obs"]
    return {"F1_max_dev": float(dev), "F2_max_dev": float(np.max(np.abs(a - b)))}


def signatures(d: dict[str, Any], fracs: list[float]) -> dict[str, Any]:
    i0, i10 = fracs.index(0.0), fracs.index(0.1)
    out: dict[str, Any] = {}
    for grp, seeds in (("adam_prim_ver", True), ("adam_can_ver", True), ("ng_prim_ver", False),
                       ("ng_can_ver", False)):  # fmt: skip
        geo = d["arr"][f"{grp}__geo"]
        if seeds:
            geo = geo.mean(1)
        res = {}
        for c in pn.CONSTRUCTIONS:
            sel = [i for i, s in enumerate(d["structs"]) if s.construction == c and d["keep"][i]]
            g = geo[sel]
            res[c] = {
                "alpha0": float(np.nanmedian(g[:, i0, 1])),
                "C0": float(np.median(g[:, i0, 2])),
                "Cout_share0": float(np.median(g[:, i0, 4] / np.maximum(g[:, i0, 2], 1e-12))),
                "dalpha_10pct": float(np.nanmedian(g[:, i10, 1] - g[:, i0, 1])),
                "dC_over_A_10pct": float(np.median(g[:, i10, 2] / np.maximum(g[:, i10, 0], 1e-12)
                                                   - g[:, i0, 2] / np.maximum(g[:, i0, 0], 1e-12))),
            }  # fmt: skip
        out[grp] = res
    return out


# ---------------------------------------------------------------- figures
def fig_optimizer(d: dict[str, Any], omap: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    x = np.arange(len(pn.CONSTRUCTIONS))
    series = (
        ("ng_prim", "NG (exact flow)"),
        ("adam_prim", "Adam, sampled 8x8"),
        ("mf_prim", "MF-Adam (approx.)"),
    )
    for k, (key, lbl) in enumerate(series):
        ax.bar(
            x + (k - 1) * 0.27, [omap[key][c]["failure"] for c in pn.CONSTRUCTIONS], 0.25, label=lbl
        )
    ax.set_xticks(x, pn.CONSTRUCTIONS)
    ax.set_ylabel("failure fraction (STALL or DECLINE)")
    ax.set_title("Outcome by construction and optimizer (design split)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    sweep = d["summary"]["sweep_index"]
    for m in pn.MECHANISMS:
        ys = []
        for key in ("adam_b16", "adam_b64_sweep", "adam_b256"):
            labs = d["labels"][key]
            sel = [
                labs[k]
                for k, i in enumerate(sweep)
                if d["structs"][i].mechanism == m and d["keep"][i]
            ]
            ys.append(np.mean([lb["failure"] for s in sel for lb in s]))
        ax.plot([16, 64, 256], ys, marker="o", label=m)
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 64, 256], ["16", "64", "256"])
    ax.set_xlabel("rollouts per step (sampled Adam)")
    ax.set_ylabel("failure fraction")
    ax.set_title("Batch-size dependence (first 10 structures per construction)")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_ceilings(ceil: dict[str, Any], path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    panels = (
        ("cindex_Dn", "C-index (Dn)"),
        ("auroc_not_yet_visible", "AUROC, not-yet-visible failure"),
        ("mechanism_macro_f1", "mechanism macro-F1"),
        ("auroc_YA_vs_YB", "AUROC, Route A vs B"),
    )
    for ax, (metric, lbl) in zip(axes, panels, strict=True):
        for level in ("L0", "L1", "L2-G", "L2", "L3", "L3+"):
            ys = [ceil[f"{h:g}|{level}"][metric] for h in H]
            ax.plot([100 * h for h in H], [np.nan if y is None else y for y in ys], marker="o",
                    label=level)  # fmt: skip
        ax.set_xscale("symlog", linthresh=0.2)
        ax.set_xlabel("% of T observed")
        ax.set_title(lbl, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_trajectories(d: dict[str, Any], fracs: list[float], path: Path) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(18, 6), sharex=True)
    obs, geo = d["arr"]["adam_prim_ver__obs"], d["arr"]["adam_prim_ver__geo"]
    t = 100 * np.array(fracs)
    names = [("J_G", obs, 0), ("J_V", obs, 1), ("FPR", obs, 2), ("A", geo, 0), ("alpha", geo, 1),
             ("C_out", geo, 4)]  # fmt: skip
    for row, which in enumerate((("R1", "X1", "YA1"), ("YB1", "B1", "D1"))):
        for col, (nm, a, j) in enumerate(names):
            ax = axes[row, col]
            for c in which:
                sel = [
                    i for i, s in enumerate(d["structs"]) if s.construction == c and d["keep"][i]
                ]
                ax.plot(t, np.nanmedian(a[sel][:, :, :, j].mean(1), axis=0), label=c)
            ax.set_title(nm, fontsize=10)
            if row == 1:
                ax.set_xlabel("% of T")
        axes[row, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Median trajectories per construction (sampled Adam, design split)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str]) -> int:
    run_dir = Path(argv[1]).resolve()
    d = load(run_dir)
    fr = d["summary"]["fracs"]
    out_dir = provenance.create_run_dir(REPO / "results", "E004a-stage0-analysis", REPO)
    provenance.write_metadata(
        out_dir,
        "E004a-stage0-analysis",
        CONFIG,
        REPO,
        extra={"split": "design", "runs": str(run_dir)},
    )
    omap = outcome_map(d)
    tab_a = run_table(d, "adam_prim_ver", "adam_prim", seeds=True)
    tab_n = run_table(d, "ng_prim_ver", "ng_prim", seeds=False)
    print("leakage ...")
    leak = leakage(d, tab_a, fr)
    print("cross-construction ...")
    cross = cross_construction(tab_a, fr)
    print("theory nulls ...")
    nulls = theory_nulls(d, d["summary"]["T_ng"])
    gates = gates_and_predictions(d, omap, leak, cross, nulls)
    print(json.dumps(gates, indent=1, default=float))
    print("ceilings (Adam, primary) ...")
    ceil_a = ceilings(tab_a, fr)
    print("ceilings (NG, secondary) ...")
    ceil_n = ceilings(tab_n, fr)
    single = {"adam": single_variable(tab_a, fr), "ng": single_variable(tab_n, fr)}
    sig = signatures(d, fr)
    fig_optimizer(d, omap, out_dir / "fig_optimizer_dependence.png")
    fig_ceilings(ceil_a, out_dir / "fig_ceilings_adam.png", "Oracle ceilings, sampled Adam (CV)")
    fig_ceilings(ceil_n, out_dir / "fig_ceilings_ng.png", "Oracle ceilings, NG (CV)")
    fig_trajectories(d, fr, out_dir / "fig_trajectories.png")
    dropped = [s.construction for s, k in zip(d["structs"], d["keep"], strict=True) if not k]
    excl = {c: dropped.count(c) for c in pn.CONSTRUCTIONS}
    report = {"excluded_per_construction": excl,
              "gates": gates, "outcome_map": omap, "leakage": leak, "cross_construction": cross,
              "theory_nulls": nulls, "ceilings_adam": ceil_a, "ceilings_ng": ceil_n,
              "single_variable": single, "signatures": sig}  # fmt: skip
    (out_dir / "stage0_analysis.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n"
    )
    print(f"run directory: {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
