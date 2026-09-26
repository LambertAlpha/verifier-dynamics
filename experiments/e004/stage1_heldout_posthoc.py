"""[post-hoc, NOT pre-registered] E004a Stage 1 held-out sensitivity analyses. They run only after
the registered held-out results are committed and never change the registered label.

1. Snapshot vs trajectory at equal audit budget on the TEST split: two extra t = 0 audits per run
   (test: resample child 8; design: child 3, as the design post-hoc). Arms L1x3 (L0 + mean t = 0
   geometry over three audits) and L2+G0x3, fitted on design and applied to test, vs L1, L2, L3.
2. Alternative onset: t_on_persist = the first 1% grid point after which the shortfall ratio stays
   > 0.1 at every later grid point (the registered t_on is unchanged). Reported: the fraction of
   non-failing runs crossing, NYV counts and the NYV AUROC of the frozen L0/L1/L2/L3 models.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage1_analysis as an  # noqa: E402
import stage1_heldout_analysis as hla  # noqa: E402
import stage1_posthoc as s1p  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e004 import heldout as hd  # noqa: E402
from vdyn.e004 import panel0b as pn  # noqa: E402
from vdyn.e004 import predict as pr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e004" / "e004a_stage1.toml"
GEO = s1p.GEO


def extra_geometry(structs: list[Any], train: list[Any], audits: list[Any]) -> np.ndarray:
    with ProcessPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(s1p._extra_t0, list(zip(structs, train, audits, strict=True)),
                              chunksize=8))  # fmt: skip
    return np.stack(parts)  # (n_struct, 4, 2, 3)


def g0x3(tab: dict[str, Any], d: dict[str, Any], extra: np.ndarray) -> np.ndarray:
    kept = [i for i, k in enumerate(d["keep"]) if k]
    ex = extra[kept].reshape(len(tab["rows"]), s1p.N_EXTRA, len(GEO))
    g0 = np.stack([tab["est"][k][:, 0] for k in GEO], -1)
    return np.nanmean(np.concatenate([g0[:, None], ex], 1), axis=1)


def persistent_onset(jg: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """(runs, 101) curves -> first grid fraction after which the ratio stays > 0.1."""
    denom = clean - clean[:, :1]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(denom > 0, (clean - jg) / np.where(denom > 0, denom, 1), 0.0)
    above = ratio > 0.1
    stays = np.flip(np.cumprod(np.flip(above, 1), 1), 1).astype(bool)
    first = np.argmax(stays, axis=1)
    return np.where(stays.any(1), first / 100.0, np.inf)


def main(argv: list[str]) -> int:
    run_dir = Path(argv[1]).resolve()
    cfg = provenance.load_config(CONFIG)
    out_dir = provenance.create_run_dir(REPO / "results", "E004a-stage1-heldout-posthoc", REPO)
    provenance.write_metadata(out_dir, "E004a-stage1-heldout-posthoc", CONFIG, REPO,
                              extra={"split": "test (post-hoc)", "post_hoc": True,
                                     "root_seed": pn.ROOT_SEED,
                                     "panel_sha256": cfg["panel_sha256"],
                                     "predictors_frozen_sha256": hla.FROZEN_SHA,
                                     "runs": str(run_dir.relative_to(REPO))})  # fmt: skip
    res = hd.resample_seeds()
    d_design = an.load(hla.DESIGN_RUNS)
    d_test = hla.load_split(run_dir, "test")
    nd, nt = len(d_design["structs"]), len(d_test["structs"])
    ex_d = extra_geometry(d_design["structs"], hd.run_seeds("prim_ver", "design", nd),
                          [c.spawn(4) for c in res[3].spawn(nd)])  # fmt: skip
    ex_t = extra_geometry(d_test["structs"], hd.run_seeds("prim_ver", "test", nt),
                          [c.spawn(4) for c in res[8].spawn(nt)])  # fmt: skip
    tab_d, tab_t = an.table(d_design, "adam"), an.table(d_test, "adam")
    gd, gt = g0x3(tab_d, d_design, ex_d), g0x3(tab_t, d_test, ex_t)
    sets: dict[str, dict[tuple[str, float], np.ndarray]] = {"d": {}, "t": {}}
    for key, tab, g in (("d", tab_d, gd), ("t", tab_t, gt)):
        for h in an.H:
            f = an.features(tab, h)
            if h == 0:
                sets[key][("L1", 0.0)] = f["L1"]
                sets[key][("L1x3", 0.0)] = np.column_stack([f["L0"], g])
                sets[key][("L0", 0.0)] = f["L0"]
            sets[key][("L2", h)] = f["L2"]
            sets[key][("L3", h)] = f["L3"]
            sets[key][("L2+G0x3", h)] = np.column_stack([f["L2"], g])
    static = ("L0", "L1", "L1x3")
    arms = ("L0", "L1", "L1x3", "L2", "L2+G0x3", "L3")
    gdes, mech = an.col(tab_d, "i"), an.col(tab_d, "mechanism")
    dn, fail = an.col(tab_d, "Dn"), an.col(tab_d, "failure").astype(int)
    jobs = [((lv, h, kind), sets["d"][(lv, 0.0 if lv in static else h)], y, gdes, kind, "linear")
            for lv in arms for h in ([0.0] if lv in static else an.H)
            for kind, y in (("ridge", dn), ("binary", fail), ("multi", mech))]  # fmt: skip
    models = hla.fit_many(jobs)
    bt = an.Booter(pr.hier_weights(an.col(tab_t, "i"), an.B, np.random.default_rng(res[8])))
    dnt, ft, mt = an.col(tab_t, "Dn"), an.col(tab_t, "failure"), an.col(tab_t, "mechanism")
    t_on = an.col(tab_t, "t_on")
    report: dict[str, Any] = {"snapshot_vs_trajectory": {}}
    star: dict[str, Any] = {}
    for h in an.H:
        cur: dict[str, Any] = {}
        for lv in arms:
            kh = 0.0 if lv in static else h
            X = sets["t"][(lv, kh)]
            pf = models[(lv, kh, "binary")][0].predict_proba(X)[:, 1]
            mm = models[(lv, kh, "multi")][0]
            cur[lv] = {"cindex_Dn": bt.pairwise(models[(lv, kh, "ridge")][0].predict(X), dnt,
                                                "cindex"),
                       "auroc_failure": bt.pairwise(pf, ft, "auroc"),
                       "auroc_nyv": bt.pairwise(pf, ft, "auroc", t_on > h),
                       "macro_f1": bt.classes(mt, mm.predict(X), np.asarray(mm.classes_))["f1"],
                       "_pf": pf}  # fmt: skip
        ent: dict[str, Any] = {lv: {m: an.summ(*v) for m, v in c.items() if m != "_pf"}
                               for lv, c in cur.items()}  # fmt: skip
        for a, b in (("L1x3", "L1"), ("L3", "L1x3"), ("L3", "L2+G0x3"), ("L3", "L1")):
            ent[f"{a}-{b}"] = {m: an.diff(cur[a][m], cur[b][m]) for m in cur[a] if m != "_pf"}
        report["snapshot_vs_trajectory"][f"{h:g}"] = ent
        if h == an.H_STAR:
            star = cur
    arr = d_test["arr"]
    kept = [i for i, k in enumerate(d_test["keep"]) if k]
    jg = arr["adam__jg_grid"][kept].reshape(len(tab_t["rows"]), 101)
    clean = np.repeat(arr["adam__clean_mean_grid"][kept], 4, axis=0)
    t_p = persistent_onset(jg, clean)
    alt: dict[str, Any] = {}
    for h in an.H:
        nyv = t_p > h
        alt[f"{h:g}"] = {
            "nonfailure_crossed_frac": float(np.mean(t_p[~ft] <= h)),
            "nyv_failures": int((ft & nyv).sum()), "nyv_nonfailures": int((~ft & nyv).sum()),
        }  # fmt: skip
    nyv = t_p > an.H_STAR
    alt["nyv_auroc_at_h*"] = {lv: an.summ(*bt.pairwise(star[lv]["_pf"], ft, "auroc", nyv))
                              for lv in ("L0", "L1", "L2", "L3")}  # fmt: skip
    alt["registered_vs_persistent_onset_agreement_failures"] = float(
        np.mean(np.isclose(t_on[ft], t_p[ft]))
    )
    report["persistent_onset"] = alt
    (out_dir / "heldout_posthoc.json").write_text(json.dumps(an.strip(report), indent=1) + "\n")
    print(f"run directory: {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
