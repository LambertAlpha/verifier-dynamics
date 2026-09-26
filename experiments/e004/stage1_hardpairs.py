"""E004a Stage 1, design round, step 3 (registry: Stage 1 execution note §6): finite-sample
evaluation of the FROZEN Stage 0b hard pairs with the frozen predictor configuration. No search.

Members are re-run with the Stage 0b verification seeds (hard-pair role; 32 verifier + 8 clean
seeds), so their sampled outcomes reproduce the frozen verification (asserted), and audited with
the hard-pair audit branch. Predictors: the frozen pipeline refit on the design runs without the
pair's anchor structure (leave-anchor-out). Reported per horizon: L2 feature distance in audit-SE
units, L3 geometry distance in pooled across-seed SD units, failure risk scores, mechanism
predictions, outcomes; criterion (iii) on HP-A / HP-D at h*; DYN-YA/R C-hat measurability.
DESIGN SPLIT ONLY.
"""

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0b_hardpairs as s0h  # noqa: E402
import stage0b_runs as s0b  # noqa: E402
import stage1_analysis as an  # noqa: E402
import stage1_runs as s1r  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e002 import endpoints as ep  # noqa: E402
from vdyn.e004 import dynamics as dy  # noqa: E402
from vdyn.e004 import features as fe  # noqa: E402
from vdyn.e004 import outcomes as oc  # noqa: E402
from vdyn.e004 import panel0b as pn  # noqa: E402
from vdyn.e004 import predict as pr  # noqa: E402
from vdyn.e004 import toy  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e004" / "e004a_stage1.toml"
N_VER, N_CLEAN = 32, 8
LEVELS_DYN = ("L2", "L3")
LEVELS_STATIC = ("L0", "L1")


def seeds_for(t_i: int) -> tuple[list[list[Any]], list[Any]]:
    """Training: the Stage 0b verification tree. Audits: spare -> [2] -> pair type -> member."""
    ss = np.random.SeedSequence(pn.ROOT_SEED).spawn(5)[3].spawn(len(s0b.ROLES))[8]
    mem = ss.spawn(len(s0h.PAIR_TYPES))[t_i].spawn(2)
    train = [m.spawn(N_VER + N_CLEAN) for m in mem]
    aud_branch = s1r.audit_tree(1)["hardpair"].spawn(len(s0h.PAIR_TYPES))[t_i].spawn(2)
    return train, [a.spawn(N_VER) for a in aud_branch]


def _clean(args: tuple[Any, ...]) -> np.ndarray:
    st, seeds, T = args
    ck = s0b.ck_steps(T)
    res = dy.run_sampled_adam([st] * len(seeds), seeds, T, ck, clean=True)
    th = s0b.to_fracs(res["theta"], T, axis=1)
    grid = [s0b.FRACS.index(i / 100) for i in range(101)]
    tb = toy.Tables.of([st] * (len(seeds) * len(grid))).clean()
    obs = dy.observables(tb, th[:, grid].reshape(-1, toy.D))
    return obs["J_G"].reshape(len(seeds), len(grid))


def submit_member(pool: ProcessPoolExecutor, st: toy.Structure, train_seeds: list[Any],
                  audit_seeds: list[Any], T: int) -> tuple[Any, Any]:  # fmt: skip
    return (pool.submit(s1r._adam_chunk, ([st] * N_VER, train_seeds[:N_VER], audit_seeds, T)),
            pool.submit(_clean, (st, train_seeds[N_VER:], T)))  # fmt: skip


def member(futures: tuple[Any, Any]) -> dict[str, Any]:
    v, c = futures[0].result(), futures[1].result()
    grid = [s0b.FRACS.index(i / 100) for i in range(101)]
    clean_mean = c.mean(0)
    labels = [oc.label(v["obs"][r, grid, 0], clean_mean) for r in range(N_VER)]
    est = {k[5:]: v[k] for k in v if k.startswith("est__")}
    exact = {k[7:]: v[k] for k in v if k.startswith("exact__")}
    return {"est": est, "exact": exact, "labels": labels}


def member_features(m: dict[str, Any], h: float) -> dict[str, np.ndarray]:
    tab = {"rows": [None] * N_VER, "est": m["est"]}
    return an.features(tab, h)


def _fit_predict(args: tuple[Any, ...]) -> tuple[Any, np.ndarray, Any]:
    key, Xtr, ytr, gtr, Xte, kind = args
    mdl = pr.fit(Xtr, ytr, gtr, kind)
    if kind == "binary":
        return key, mdl.predict_proba(Xte)[:, 1], None
    return key, mdl.predict_proba(Xte), mdl.classes_


def distances(fa: dict[str, np.ndarray], fb: dict[str, np.ndarray], h: float,
              ma: dict[str, Any], mb: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    """L2 summaries: |mean diff| / registered audit SE (plug-in at the pair mean); L3 geometry
    summaries: |mean diff| / pooled across-seed SD."""
    n2 = len(fe.names("L2"))
    a2, b2 = np.nanmean(fa["L2"][:, 5:n2], 0), np.nanmean(fb["L2"][:, 5:n2], 0)
    idx = an.hidx(h)
    vals = {k: np.array([np.nanmean(np.r_[ma["est"][k][:, i], mb["est"][k][:, i]]) for i in idx])
            for k in fe.L2_VARS}  # fmt: skip
    se = fe.l2_feature_se(vals, h) if h > 0 else None
    out: dict[str, Any] = {}
    if se is not None:
        r = np.abs(a2 - b2) / np.maximum(se, 1e-12)
        out["L2_max_se_units"] = float(np.nanmax(r))
        out["L2_rms_se_units"] = float(np.sqrt(np.nanmean(r**2)))
        out["L2_worst"] = [f"{v}_{s}" for v in fe.L2_VARS for s in ("h", "d", "slope")][
            int(np.nanargmax(r))
        ]
    ga, gb = fa["L3"][:, n2:], fb["L3"][:, n2:]
    sd = np.sqrt((np.nanvar(ga, 0) + np.nanvar(gb, 0)) / 2)
    d3 = np.abs(np.nanmean(ga, 0) - np.nanmean(gb, 0)) / np.maximum(sd, 1e-12)
    out["L3_geo_max_sd_units"] = float(np.nanmax(d3)) if h > 0 else float(d3[[0, 3, 6]].max())
    out["L3_geo_rms_sd_units"] = float(np.sqrt(np.nanmean(d3**2)))
    return out


def l1_distance(ma: dict[str, Any], mb: dict[str, Any]) -> dict[str, float]:
    """t = 0 geometry (A_u, alpha_u, C_u): |mean diff| / pooled across-seed SD."""
    out = {}
    for k in ("A_u", "alpha_u", "C_u"):
        a, b = ma["est"][k][:, 0], mb["est"][k][:, 0]
        sd = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2)
        out[k] = float(abs(np.nanmean(a) - np.nanmean(b)) / max(sd, 1e-12))
    return out


def geo_uncertainty(ma: dict[str, Any], mb: dict[str, Any], j: int) -> dict[str, Any]:
    return {k: {"mean": [float(np.nanmean(m["est"][k][:, j])) for m in (ma, mb)],
                "sd": [float(np.nanstd(m["est"][k][:, j])) for m in (ma, mb)],
                "exact_mean": [float(np.nanmean(m["exact"]["geo_u"][:, j, i])) for m in (ma, mb)]}
            for i, k in ((1, "alpha_u"), (2, "C_u"))}  # fmt: skip


def fig_pairs(report: dict[str, Any], path: Path) -> None:
    names = [n for n in ("HP-D", "HP-A", "HP-B", "DYN-YA/R") if n in report]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6))
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        e = report[name]
        xs = [100 * h for h in an.H]
        for lv, ls in (("L2", "--"), ("L3", "-")):
            for k, col in ((0, "C3"), (1, "C0")):
                ys = [e["horizons"][f"{h:g}"][lv]["risk_mean"][k] for h in an.H]
                ax.plot(
                    xs,
                    ys,
                    ls=ls,
                    color=col,
                    marker="o",
                    ms=3,
                    label=f"{lv} {e['mechanisms'][k]} (fail {e['failure_fraction'][k]:.2f})",
                )
        ax.set_xscale("symlog", linthresh=0.2)
        ax.set_ylim(0, 1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("% of T observed")
        ax.legend(frameon=False, fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("predicted failure risk (frozen models)")
    fig.suptitle("Fig 6. Frozen hard pairs: L2 vs L3 risk (member means over 32 seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def c_measurability(ma: dict[str, Any], mb: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for j, f in enumerate(s1r.FEAT_FRACS):
        if f > 0.05:
            continue
        ca, cb = ma["est"]["C_u"][:, j], mb["est"]["C_u"][:, j]
        xa, xb = ma["exact"]["geo_u"][:, j, 2], mb["exact"]["geo_u"][:, j, 2]
        sd = np.sqrt((ca.var() + cb.var()) / 2)
        lab = np.r_[np.ones(len(ca), bool), np.zeros(len(cb), bool)]
        out[f"{f:g}"] = {"C_hat_mean": [float(ca.mean()), float(cb.mean())],
                         "C_hat_sd": [float(ca.std()), float(cb.std())],
                         "C_exact_mean": [float(xa.mean()), float(xb.mean())],
                         "std_mean_diff": float((ca.mean() - cb.mean()) / max(sd, 1e-12)),
                         "single_run_auroc": ep.auroc(np.r_[ca, cb], lab)}  # fmt: skip
    return out


def main(argv: list[str]) -> int:
    analysis_dir = Path(argv[1]).resolve()
    cfg = provenance.load_config(CONFIG)
    hp_path = REPO / cfg["hardpairs"]
    hp_sha = hashlib.sha256(hp_path.read_bytes()).hexdigest()
    assert hp_sha == cfg["hardpairs_sha256"], "hard-pair file changed"
    frozen_cfg = analysis_dir / "predictors_frozen.json"
    frozen_sha = hashlib.sha256(frozen_cfg.read_bytes()).hexdigest()
    runs_dir = REPO / json.loads(frozen_cfg.read_text())["runs"]
    d = an.load(runs_dir)
    T = cfg["runs"]["T_adam"]
    sid_idx = {s.sid: i for i, s in enumerate(d["structs"])}
    pairs = json.loads(hp_path.read_text())["pairs"]
    verification = json.loads((hp_path.parent / "hardpairs.json").read_text())["pair_types"]
    out_dir = provenance.create_run_dir(REPO / "results", "E004a-stage1-hardpairs", REPO)
    provenance.write_metadata(out_dir, "E004a-stage1-hardpairs", CONFIG, REPO,
                              extra={"split": "design", "root_seed": pn.ROOT_SEED,
                                     "panel_sha256": cfg["panel_sha256"],
                                     "hardpairs_sha256": hp_sha,
                                     "predictors_frozen": str(frozen_cfg.relative_to(REPO)),
                                     "predictors_frozen_sha256": frozen_sha})  # fmt: skip
    tab = an.table(d, "adam")
    g, mech = an.col(tab, "i"), an.col(tab, "mechanism")
    fail = an.col(tab, "failure").astype(int)
    design = {h: an.features(tab, h) for h in an.H}
    types = list(s0h.PAIR_TYPES)
    members: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        pending = []
        for p in pairs:
            train, audit = seeds_for(types.index(p["type"]))
            anchor = d["structs"][sid_idx[p["anchor"]]]
            partner = toy.Structure.from_dict(p["partner"])
            futs = [submit_member(pool, st, train[k], audit[k], T)
                    for k, st in enumerate((anchor, partner))]  # fmt: skip
            pending.append((p, anchor, partner, futs))
        for p, anchor, partner, futs in pending:
            ms = [member(f) for f in futs]
            ff = [np.mean([lb["failure"] for lb in m["labels"]]) for m in ms]
            want = verification[p["type"]]["evaluation"]["sampled"]["failure_fraction"]
            assert np.allclose(ff, want), f"{p['type']}: {ff} vs frozen verification {want}"
            members[p["type"]] = {"pair": p, "members": ms, "structs": (anchor, partner)}
            print(p["type"], "members done; failure", ff, flush=True)
        jobs = []
        for name, e in members.items():
            tr = g != sid_idx[e["pair"]["anchor"]]
            for h in an.H:
                fm = [member_features(m, h) for m in e["members"]]
                for lv in LEVELS_DYN + (LEVELS_STATIC if h == 0 else ()):
                    Xte = np.vstack([f[lv] for f in fm])
                    Xtr = design[h][lv][tr]
                    jobs.append(((name, h, lv, "binary"), Xtr, fail[tr], g[tr], Xte, "binary"))
                    jobs.append(((name, h, lv, "multi"), Xtr, mech[tr], g[tr], Xte, "multi"))
        preds = {k: (p_, c_) for k, p_, c_ in pool.map(_fit_predict, jobs, chunksize=1)}
    report: dict[str, Any] = {}
    l2_ok: dict[str, np.ndarray] = {}
    l3_ok: dict[str, np.ndarray] = {}
    for name, e in members.items():
        anchor, partner = e["structs"]
        mechs = (anchor.mechanism, partner.mechanism)
        truth = np.r_[np.full(N_VER, mechs[0]), np.full(N_VER, mechs[1])]
        fails = np.r_[[lb["failure"] for m in e["members"] for lb in m["labels"]]].astype(bool)
        ent: dict[str, Any] = {
            "kind": e["pair"]["kind"], "accepted_stage0b": e["pair"]["accepted"],
            "anchor": e["pair"]["anchor"], "mechanisms": mechs, "axes": e["pair"]["axes"],
            "failure_fraction": [float(fails[:N_VER].mean()), float(fails[N_VER:].mean())],
            "categories": [[lb["category"] for lb in m["labels"]] for m in e["members"]],
            "horizons": {},
        }  # fmt: skip
        for h in an.H:
            fm = [member_features(m, h) for m in e["members"]]
            he: dict[str, Any] = distances(fm[0], fm[1], h, *e["members"])
            for lv in LEVELS_DYN + (LEVELS_STATIC if h == 0 else ()):
                risk = preds[(name, h, lv, "binary")][0]
                P, classes = preds[(name, h, lv, "multi")]
                i0, i1 = list(classes).index(mechs[0]), list(classes).index(mechs[1])
                correct = np.where(truth == mechs[0], P[:, i0] > P[:, i1], P[:, i1] > P[:, i0])
                he[lv] = {"risk_mean": [float(risk[:N_VER].mean()), float(risk[N_VER:].mean())],
                          "risk_auroc_fail_vs_nonfail": ep.auroc(risk, fails)
                          if 0 < fails.sum() < len(fails) else None,
                          "pair_mech_accuracy": float(correct.mean()),
                          "top1_mech": [str(classes[P[:N_VER].mean(0).argmax()]),
                                        str(classes[P[N_VER:].mean(0).argmax()])]}  # fmt: skip
                if h == an.H_STAR and name in ("HP-A", "HP-D") and lv in LEVELS_DYN:
                    (l2_ok if lv == "L2" else l3_ok)[name] = correct
            ma, mb = e["members"]
            he["geometry_uncertainty"] = geo_uncertainty(ma, mb, s1r.FEAT_FRACS.index(h))
            ent["horizons"][f"{h:g}"] = he
        ent["L1_geo_sd_units"] = l1_distance(*e["members"])
        if name == "DYN-YA/R":
            ent["C_measurability"] = c_measurability(*e["members"])
        report[name] = ent
    report["criterion_iii"] = pr.criterion_iii(l2_ok, l3_ok)
    (out_dir / "hardpairs_stage1.json").write_text(json.dumps(an.strip(report), indent=1) + "\n")
    fig_pairs(report, out_dir / "fig6_hard_pairs.png")
    print(json.dumps(an.strip(report["criterion_iii"])))
    print(f"run directory: {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
