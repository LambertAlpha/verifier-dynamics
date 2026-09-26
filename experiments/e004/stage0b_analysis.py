"""E004a Stage 0b, step 4 (Amendment 3 §9, §11, §12): gates G1-G6, leakage tests A-E, maps,
two alphas, signatures, oracle ceilings. DESIGN SPLIT ONLY; exact observables along the realized
runs; cross-validated diagnostics, not Stage 1 predictors. L1/L3 use the update-level geometry.
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
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0_analysis as sa  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e002 import endpoints as ep  # noqa: E402
from vdyn.e004 import dynamics as dy  # noqa: E402
from vdyn.e004 import features as fe  # noqa: E402
from vdyn.e004 import panel0b as pb  # noqa: E402
from vdyn.e004 import toy  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PANEL = REPO / "configs" / "e004" / "design_panel_0b.json"
CONFIG = REPO / "configs" / "e004" / "e004a.toml"
H = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
H_STAR = 0.02
FAIL_MECH = ("X", "YA", "YB", "D")
N_BOOT = 1000
VALID_KAPPA = 1e7


# ------------------------------------------------------------------ loading
def load(run_dir: Path) -> dict[str, Any]:
    arr = np.load(run_dir / "runs.npz")
    summ = json.loads((run_dir / "summary.json").read_text())
    lab = json.loads((run_dir / "labels.json").read_text())
    structs = [toy.Structure.from_dict(d) for d in json.loads(PANEL.read_text())["structures"]]
    excluded = set(summ["excluded_low_clean_gain"])
    return {
        "arr": arr,
        "summary": summ,
        "labels": lab,
        "structs": structs,
        "keep": [s.sid not in excluded for s in structs],
    }


def table(d: dict[str, Any], group: str, lab_key: str, seeds: bool, level: str = "u") -> dict:
    """Runs (kept structures x seeds) with labels; 'geo' = the chosen geometry level."""
    obs, ctx = d["arr"][f"{group}__obs"], d["arr"][f"{group}__ctx"]
    geo = d["arr"][f"{group}__geo_{level}"]
    if not seeds:
        obs, ctx, geo = obs[:, None], ctx[:, None], geo[:, None]
    kept = [i for i in range(len(d["structs"])) if d["keep"][i]]
    s = obs.shape[1]
    rows = []
    for i in kept:
        st = d["structs"][i]
        for r in range(s):
            rows.append(
                {
                    "i": i,
                    "r": r,
                    "construction": st.construction,
                    "mechanism": st.mechanism,
                    "axis": st.meta["axis_b"],
                    "slot": int(st.construction[-1]),
                    **d["labels"][lab_key][i][r],
                }
            )
    m = len(kept) * s
    return {
        "obs": obs[kept].reshape(m, *obs.shape[2:]),
        "ctx": ctx[kept].reshape(m, *ctx.shape[2:]),
        "geo": geo[kept].reshape(m, *geo.shape[2:]),
        "rows": rows,
    }


def cols(tab: dict[str, Any], key: str) -> np.ndarray:
    return np.array([r[key] for r in tab["rows"]])


# ------------------------------------------------------------------ bootstrap helpers
def boot_diff(
    metric,
    a: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Paired bootstrap over structures of metric(a) - metric(b) (a, b: out-of-fold outputs)."""
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    diffs = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        try:
            diffs.append(metric(a[pick], y[pick]) - metric(b[pick], y[pick]))
        except (ValueError, ZeroDivisionError):
            continue
    point = metric(a, y) - metric(b, y)
    lo = float(np.quantile(diffs, 0.025)) if diffs else float("nan")
    return {"point": float(point), "ci_lo": lo, "ci_hi": float(np.quantile(diffs, 0.975))}


def auroc_m(p: np.ndarray, y: np.ndarray) -> float:
    return ep.auroc(p, y.astype(bool))


def cidx_m(p: np.ndarray, y: np.ndarray) -> float:
    return ep.c_index(p, y)


def f1_m(p: np.ndarray, y: np.ndarray) -> float:
    return sa.macro_f1(y, p)


# ------------------------------------------------------------------ analyses
def outcome_maps(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("adam_prim", "ng_prim", "mf_prim", "adam_can", "ng_can"):
        res: dict[str, Any] = {}
        for i, st in enumerate(d["structs"]):
            if not d["keep"][i]:
                continue
            for scope in (
                st.mechanism,
                st.construction,
                f"{st.mechanism}|{st.meta['axis_b']}",
                f"{st.construction}|{st.meta['axis_b']}",
                st.meta["axis_b"],
            ):
                res.setdefault(scope, []).extend(d["labels"][key][i])
        out[key] = {
            k: {
                "n_runs": len(v),
                "failure": float(np.mean([x["failure"] for x in v])),
                "decline": float(np.mean([x["category"] == "DECLINE" for x in v])),
                "stall": float(np.mean([x["category"] == "STALL" for x in v])),
                "Dn_median": float(np.median([x["Dn"] for x in v])),
            }
            for k, v in res.items()
        }
    return out


def alphas(d: dict[str, Any], fracs: list[float]) -> dict[str, Any]:
    i0, ih = fracs.index(0.0), fracs.index(H_STAR)
    out: dict[str, Any] = {}
    for grp, seeds in (("adam_prim_ver", True), ("ng_prim_ver", False)):
        gu, gr = d["arr"][f"{grp}__geo_u"], d["arr"][f"{grp}__geo_r"]
        if seeds:
            gu, gr = gu.mean(1), gr.mean(1)
        res: dict[str, Any] = {}
        for c in pb.CONSTRUCTIONS:
            sel = [i for i, s in enumerate(d["structs"]) if s.construction == c and d["keep"][i]]
            res[c] = {
                "alpha_r0": float(np.nanmedian(gr[sel, i0, 1])),
                "alpha_u0": float(np.nanmedian(gu[sel, i0, 1])),
                "alpha_r_hstar": float(np.nanmedian(gr[sel, ih, 1])),
                "alpha_u_hstar": float(np.nanmedian(gu[sel, ih, 1])),
                "frac_alpha_u0_pos": float(np.mean(gu[sel, i0, 1] > 0)),
                "frac_alpha_r0_neg": float(np.mean(gr[sel, i0, 1] < 0)),
            }
        rsel = [i for i, s in enumerate(d["structs"]) if s.mechanism == "R" and d["keep"][i]]
        res["R_all_alpha_r0_negative"] = bool(np.all(gr[rsel, i0, 1] < 0))
        res["R_frac_alpha_u0_positive"] = float(np.mean(gu[rsel, i0, 1] > 0))
        out[grp] = res
    return out


def signatures(d: dict[str, Any], fracs: list[float]) -> dict[str, Any]:
    i0, i10 = fracs.index(0.0), fracs.index(0.1)
    out: dict[str, Any] = {}
    for grp, seeds in (
        ("adam_prim_ver", True),
        ("adam_can_ver", True),
        ("ng_prim_ver", False),
        ("ng_can_ver", False),
    ):
        g = d["arr"][f"{grp}__geo_u"]
        g = g.mean(1) if seeds else g
        res = {}
        for c in pb.CONSTRUCTIONS:
            sel = [i for i, s in enumerate(d["structs"]) if s.construction == c and d["keep"][i]]
            x = g[sel]
            res[c] = {
                "alpha_u0": float(np.nanmedian(x[:, i0, 1])),
                "C_u0": float(np.median(x[:, i0, 2])),
                "Cout_share0": float(np.median(x[:, i0, 4] / np.maximum(x[:, i0, 2], 1e-12))),
                "dC_over_A_10pct": float(
                    np.median(
                        x[:, i10, 2] / np.maximum(x[:, i10, 0], 1e-12)
                        - x[:, i0, 2] / np.maximum(x[:, i0, 0], 1e-12)
                    )
                ),
            }
        out[grp] = res
    return out


def per_structure(tab: dict[str, Any]) -> list[int]:
    return [k for k, r in enumerate(tab["rows"]) if r["r"] == 0]


def test_a_construction(tab: dict[str, Any], fracs: list[float]) -> dict[str, Any]:
    first = per_structure(tab)
    y = cols(tab, "construction")[first]
    groups = np.arange(len(first))
    out = {"chance": 1 / 18}
    for level in ("L0", "L1"):
        X = sa.features(tab, fracs, 0.0, level)[first]
        out[level] = sa.macro_f1(y, sa.cv_predict(X, y, groups, y, "multi"))
    Xm = sa.features(tab, fracs, 0.0, "L0")[first]
    ym = cols(tab, "mechanism")[first]
    out["L0_mechanism"] = sa.macro_f1(ym, sa.cv_predict(Xm, ym, groups, ym, "multi"))
    ya = cols(tab, "axis")[first]
    out["L0_axis_auroc"] = ep.auroc(
        sa.cv_predict(Xm, (ya == "INVERTED").astype(int), groups, ym, "binary"), ya == "INVERTED"
    )
    return out


def _fit_predict(Xtr, ytr, gtr, Xte) -> np.ndarray:
    m = make_pipeline(
        StandardScaler(), LogisticRegressionCV(Cs=sa.C_GRID, cv=sa._inner(gtr), max_iter=3000)
    )
    return m.fit(Xtr, ytr).predict(Xte)


def test_b_loco(tab: dict[str, Any], fracs: list[float]) -> dict[str, Any]:
    y = cols(tab, "mechanism")
    cons = cols(tab, "construction")
    groups = cols(tab, "i")
    out = {}
    for level in ("L0", "L1", "L2", "L3", "L3+"):
        X = sa.features(tab, fracs, H_STAR, level)
        pred = np.empty(len(y), dtype=object)
        for c in pb.CONSTRUCTIONS:
            te = cons == c
            pred[te] = _fit_predict(X[~te], y[~te], groups[~te], X[te])
        out[level] = {"macro_f1": sa.macro_f1(y, pred), "accuracy": float(np.mean(pred == y))}
    return out


def test_d_cross(tab: dict[str, Any], fracs: list[float], rng: np.random.Generator) -> dict:
    y = cols(tab, "mechanism")
    slot = cols(tab, "slot")
    groups = cols(tab, "i")
    preds: dict[str, np.ndarray] = {}
    out: dict[str, Any] = {}
    for level in ("L0", "L1", "L2", "L3", "L3+"):
        X = sa.features(tab, fracs, H_STAR, level)
        pred = np.empty(len(y), dtype=object)
        folds = []
        for k in (1, 2, 3):
            te = slot == k
            pred[te] = _fit_predict(X[~te], y[~te], groups[~te], X[te])
            folds.append(sa.macro_f1(y[te], pred[te]))
        preds[level] = pred
        out[level] = {"fold_macro_f1": folds, "mean_macro_f1": float(np.mean(folds))}
    out["L3_minus_L0"] = boot_diff(f1_m, preds["L3"], preds["L0"], y, groups, rng)
    out["L3_minus_L2"] = boot_diff(f1_m, preds["L3"], preds["L2"], y, groups, rng)
    return out


def test_c_within(tab: dict[str, Any], fracs: list[float], rng: np.random.Generator) -> dict:
    mech, groups = cols(tab, "mechanism"), cols(tab, "i")
    fail, dn = cols(tab, "failure").astype(int), cols(tab, "Dn")
    inv = (cols(tab, "axis") == "INVERTED").astype(int)
    feats = {lv: sa.features(tab, fracs, H_STAR, lv) for lv in ("L0", "L1", "L2", "L3")}
    out: dict[str, Any] = {}
    for m in pb.MECHANISMS:
        sel = mech == m
        g = groups[sel]
        res: dict[str, Any] = {}
        tasks = [("failure", fail[sel], "binary", auroc_m), ("Dn", dn[sel], "ridge", cidx_m)]
        if m in ("YA", "YB"):
            tasks.append(("axis_b", inv[sel], "binary", auroc_m))
        for name, y, kind, metric in tasks:
            if kind == "binary" and min(y.sum(), (1 - y).sum()) < 10:
                res[name] = {"skipped": "fewer than 10 runs in a class"}
                continue
            if kind == "ridge" and np.ptp(y) == 0:
                res[name] = {"skipped": "constant target"}
                continue
            p = {
                lv: sa.cv_predict(
                    feats[lv][sel], y, g, y if kind == "binary" else np.zeros(len(y)), kind
                )
                for lv in feats
            }
            res[name] = {lv: float(metric(p[lv], y)) for lv in feats}
            res[name]["L3_minus_L2"] = boot_diff(metric, p["L3"], p["L2"], y, g, rng)
            res[name]["L3_minus_L1"] = boot_diff(metric, p["L3"], p["L1"], y, g, rng)
        out[m] = res
    return out


def test_e_oracle(tab: dict[str, Any]) -> dict[str, Any]:
    dn, fail = cols(tab, "Dn"), cols(tab, "failure")
    s_all = cols(tab, "i")
    out = {}
    for name, key in (
        ("mechanism", cols(tab, "mechanism")),
        (
            "mechanism_x_axis",
            np.char.add(cols(tab, "mechanism").astype(str), cols(tab, "axis").astype(str)),
        ),
    ):
        pred = np.array([dn[(key == key[k]) & (s_all != s_all[k])].mean() for k in range(len(dn))])
        out[name] = {"cindex_Dn": ep.c_index(pred, dn), "auroc_failure": ep.auroc(pred, fail)}
    return out


def ceilings(tab: dict[str, Any], fracs: list[float]) -> dict[str, Any]:
    dn, fail = cols(tab, "Dn"), cols(tab, "failure").astype(int)
    mech, groups = cols(tab, "mechanism"), cols(tab, "i")
    inv = (cols(tab, "axis") == "INVERTED").astype(int)
    t_on = np.array([np.inf if r["t_on"] is None else r["t_on"] for r in tab["rows"]])
    out: dict[str, Any] = {}
    for h in H:
        for level in fe.LEVELS:
            X = sa.features(tab, fracs, h, level)
            p_dn = sa.cv_predict(X, dn, groups, mech, "ridge")
            p_f = sa.cv_predict(X, fail, groups, mech, "binary")
            p_m = sa.cv_predict(X, mech, groups, mech, "multi")
            p_i = sa.cv_predict(X, inv, groups, mech, "binary")
            nyv = t_on > h
            out[f"{h:g}|{level}"] = {
                "cindex_Dn": ep.c_index(p_dn, dn),
                "auroc_failure": ep.auroc(p_f, fail.astype(bool)),
                "auroc_not_yet_visible": ep.auroc(p_f[nyv], fail[nyv].astype(bool))
                if 0 < fail[nyv].sum() < nyv.sum()
                else None,
                "n_not_yet_visible_fail": int(fail[nyv].sum()),
                "mechanism_macro_f1": sa.macro_f1(mech, p_m),
                "auroc_axis_b": ep.auroc(p_i, inv.astype(bool)),
            }
        print(f"  ceilings h={h:g} done", flush=True)
    return out


def theory_nulls(d: dict[str, Any]) -> dict[str, Any]:
    T_ng = d["summary"]["T_ng"]
    all_dev, valid_dev, n_all, n_valid = 0.0, 0.0, 0, 0
    for st in d["structs"]:
        if (
            st.construction not in ("YA1", "YA3", "YB1", "YB3")
            or st.meta["intended_axis"] != "ALIGNED"
        ):
            continue
        can = pb.canonical_twin(st)
        if toy.axis_b(can) != "ALIGNED":
            continue
        tb = toy.Tables.of([can])
        sol = solve_ivp(
            lambda t, th, tb=tb: dy._ng_velocity(tb, th),
            (0, T_ng),
            can.theta0,
            t_eval=np.linspace(0, T_ng, 11),
            rtol=1e-9,
            atol=1e-11,
        )
        for th in sol.y.T:
            ex = toy.exact(tb, th[None])
            F = ex["F"][0] + dy.NG_DAMP * np.trace(ex["F"][0]) / toy.D * np.eye(toy.D)
            Minv = np.linalg.inv(F)
            vel = Minv @ ex["g_V"][0]
            geo = toy.decompose(ex["g_G"][0], ex["g_V"][0], Minv)
            fpr, jg = ex["FPR"][0], ex["J_G"][0]
            pred = (-fpr, ex["g_G"][0] @ vel / (1 - fpr), (1 - jg) * ex["g_FPR"][0] @ vel)
            got = (geo["alpha"], geo["A"] ** 2, geo["C"] ** 2)
            dev = max(abs(p - g) / (abs(g) + 1e-12) for p, g in zip(pred, got, strict=True))
            n_all += 1
            all_dev = max(all_dev, dev)
            if np.linalg.cond(F) / max(1 - fpr, 1e-300) <= VALID_KAPPA:
                n_valid += 1
                valid_dev = max(valid_dev, dev)
    f2 = float(np.max(np.abs(d["arr"]["ng_prim_ver__obs"] - d["arr"]["ng_unc_ver__obs"])))
    r_dev = 0.0
    for st in d["structs"]:
        if st.construction != "R1":
            continue
        can = pb.canonical_twin(st)
        ex = toy.single(can, can.theta0)
        eps = float(can.fp[0])
        for M in (
            np.linalg.inv(ex["F"]),
            1 / (np.sqrt(ex["g_eff"] ** 2 + ex["var_eff"] / 64) + 1e-8),
        ):
            r_dev = max(r_dev, abs(toy.decompose(ex["g_G"], ex["g_V"], M)["alpha"] + 2 * eps))
    return {
        "F1_all_state_max_dev": all_dev,
        "F1_valid_domain_max_dev": valid_dev,
        "F1_states": n_all,
        "F1_valid_states": n_valid,
        "F1_fraction_excluded": 1 - n_valid / max(n_all, 1),
        "F2_max_dev": f2,
        "R1_alpha_r_max_dev": r_dev,
    }


def gates(
    omap: dict[str, Any],
    a: dict[str, Any],
    c: dict[str, Any],
    dres: dict[str, Any],
    nulls: dict[str, Any],
) -> dict[str, Any]:
    ad, ng = omap["adam_prim"], omap["ng_prim"]
    g2 = {m: ad[m]["failure"] for m in FAIL_MECH}
    passing = []
    for m, res in c.items():
        for name, v in res.items():
            if "L3_minus_L2" in v:
                b = v["L3_minus_L2"]
                if b["point"] >= 0.02 and b["ci_lo"] > 0:
                    passing.append(f"{m}:{name}")
    mechs_passing = sorted({x.split(":")[0] for x in passing})
    diffs = {m: ad[m]["failure"] - ng[m]["failure"] for m in pb.MECHANISMS}
    robust = [
        m
        for m in FAIL_MECH
        if ad[m]["failure"] >= 0.5 and ng[m]["failure"] >= 0.5 and abs(diffs[m]) <= 0.2
    ]
    material = [m for m in pb.MECHANISMS if abs(diffs[m]) >= 0.3]
    return {
        "G1": {
            "L0_construction_macro_f1": a["L0"],
            "threshold": 1 / 18 + 0.10,
            "L1_construction_macro_f1": a["L1"],
            "pass": a["L0"] <= 1 / 18 + 0.10,
        },
        "G2": {"failure": g2, "pass": all(0.2 <= v <= 0.8 for v in g2.values())},
        "G3": {
            **nulls,
            "pass": nulls["F1_valid_domain_max_dev"] <= 1e-8
            and nulls["F2_max_dev"] <= 1e-6
            and nulls["R1_alpha_r_max_dev"] <= 1e-10,
        },
        "G4": {
            "L3_cross_macro_f1": dres["L3"]["mean_macro_f1"],
            "threshold": 1 / 6 + 0.15,
            "L3_minus_L0": dres["L3_minus_L0"],
            "pass": dres["L3"]["mean_macro_f1"] >= 1 / 6 + 0.15
            and dres["L3_minus_L0"]["point"] >= 0.05
            and dres["L3_minus_L0"]["ci_lo"] > 0,
        },
        "G5": {
            "passing_targets": passing,
            "mechanisms": mechs_passing,
            "pass": len(mechs_passing) >= 2,
        },
        "G6": {
            "adam_minus_ng_failure": diffs,
            "material": material,
            "robust": robust,
            "pass": bool(material) and bool(robust),
        },
    }


# ------------------------------------------------------------------ figures
def fig_map(omap: dict[str, Any], path: Path) -> None:
    keys = [
        f"{m}|{a}"
        for m in pb.MECHANISMS
        for a in ("ALIGNED", "INVERTED")
        if f"{m}|{a}" in omap["adam_prim"]
    ]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(12, 4))
    for k, (grp, lbl) in enumerate(
        (
            ("ng_prim", "NG (exact flow)"),
            ("adam_prim", "Adam, sampled"),
            ("mf_prim", "MF-Adam (approx.)"),
        )
    ):
        ax.bar(x + (k - 1) * 0.27, [omap[grp][key]["failure"] for key in keys], 0.25, label=lbl)
    ax.set_xticks(
        x, [k.replace("|ALIGNED", "\naligned").replace("|INVERTED", "\ninverted") for k in keys]
    )
    ax.set_ylabel("failure fraction")
    ax.set_title("Stage 0b: failure by mechanism x preference relation x optimizer (design split)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_alpha(d: dict[str, Any], fracs: list[float], path: Path) -> None:
    i0 = fracs.index(0.0)
    gu, gr = d["arr"]["adam_prim_ver__geo_u"].mean(1), d["arr"]["adam_prim_ver__geo_r"].mean(1)
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in pb.MECHANISMS:
        sel = [i for i, s in enumerate(d["structs"]) if s.mechanism == m and d["keep"][i]]
        ax.scatter(gr[sel, i0, 1], gu[sel, i0, 1], s=8, label=m)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("alpha_reward at t=0 (Adam metric)")
    ax.set_ylabel("alpha_update at t=0 (Adam metric)")
    ax.set_xlim(-3, 6)
    ax.set_ylim(-3, 6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str]) -> int:
    run_dir = Path(argv[1]).resolve()
    d = load(run_dir)
    fr = d["summary"]["fracs"]
    out_dir = provenance.create_run_dir(REPO / "results", "E004a-stage0b-analysis", REPO)
    provenance.write_metadata(
        out_dir,
        "E004a-stage0b-analysis",
        CONFIG,
        REPO,
        extra={
            "split": "design",
            "runs": str(run_dir),
            "panel_sha256": d["summary"]["panel_sha256"],
        },
    )
    rng = np.random.default_rng([pb.ROOT_SEED, 9])
    omap = outcome_maps(d)
    tab = table(d, "adam_prim_ver", "adam_prim", seeds=True)
    tab_ng = table(d, "ng_prim_ver", "ng_prim", seeds=False)
    print("tests A-E ...", flush=True)
    a = test_a_construction(tab, fr)
    b = test_b_loco(tab, fr)
    dres = test_d_cross(tab, fr, rng)
    c = test_c_within(tab, fr, rng)
    e = test_e_oracle(tab)
    print("theory nulls ...", flush=True)
    nulls = theory_nulls(d)
    g = gates(omap, a, c, dres, nulls)
    print(json.dumps({k: v["pass"] for k, v in g.items()}), flush=True)
    al = alphas(d, fr)
    sig = signatures(d, fr)
    print("ceilings ...", flush=True)
    ceil_a = ceilings(tab, fr)
    ceil_n = ceilings(tab_ng, fr)
    fig_map(omap, out_dir / "fig_optimizer_map.png")
    fig_alpha(d, fr, out_dir / "fig_alpha_reward_vs_update.png")
    sa.fig_ceilings(
        {k: {**v, "auroc_YA_vs_YB": v["auroc_axis_b"]} for k, v in ceil_a.items()},
        out_dir / "fig_ceilings_adam.png",
        "Stage 0b oracle ceilings, sampled Adam (4th panel: Axis-B AUROC)",
    )
    dropped = [s.construction for s, k in zip(d["structs"], d["keep"], strict=True) if not k]
    report = {
        "gates": g,
        "excluded_per_construction": {c_: dropped.count(c_) for c_ in pb.CONSTRUCTIONS},
        "outcome_maps": omap,
        "alphas": al,
        "signatures": sig,
        "test_A": a,
        "test_B": b,
        "test_C": c,
        "test_D": dres,
        "test_E": e,
        "theory_nulls": nulls,
        "ceilings_adam": ceil_a,
        "ceilings_ng": ceil_n,
    }
    (out_dir / "stage0b_analysis.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n"
    )
    print(f"run directory: {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
