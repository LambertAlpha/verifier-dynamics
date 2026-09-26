"""[post-hoc, NOT pre-registered] E004a Stage 1 design round: is the finite-sample geometry signal
dynamic, or a snapshot measured with more audits?

L3 uses three independent audits (at 0, h/2, h); L1 uses one (at 0). This check adds two more
independent audits at theta_0 for every run (seed: spare -> resample branch, child 3; the step-0
training batch is regenerated from the run's own training seed, identical to the recorded one) and
compares, with the frozen pipeline, folds and bootstrap weights:
- L1x3 = L0 + mean t = 0 geometry over three audits (vs L1: one audit);
- L2+G0 = L2 + t = 0 geometry (one audit); L2+G0x3 = L2 + mean t = 0 geometry (three audits);
- L3 = L2 + geometry trajectory summaries (three audits along the run).
If L3 ~ L2+G0x3, the geometry information is a snapshot property measured with more audits.
DESIGN SPLIT ONLY; no criterion is recomputed or changed.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0b_runs as s0b  # noqa: E402
import stage1_analysis as an  # noqa: E402
import stage1_runs as s1r  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e004 import audit as au  # noqa: E402
from vdyn.e004 import dynamics as dy  # noqa: E402
from vdyn.e004 import panel0b as pn  # noqa: E402
from vdyn.e004 import predict as pr  # noqa: E402
from vdyn.e004 import toy  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e004" / "e004a_stage1.toml"
GEO = ("A_u", "alpha_u", "C_u")
N_EXTRA = 2


def _extra_t0(args: tuple[Any, ...]) -> np.ndarray:
    """(seeds, 2 extra audits, 3 geometry values) at theta_0 for one structure."""
    st, train_seeds, audit_seeds = args
    n = len(train_seeds)
    tb = toy.Tables.of([st] * n)
    th = np.stack([st.theta0] * n)
    _, batch = dy.grpo_gradient(tb, th, [np.random.default_rng(s) for s in train_seeds],
                                return_batch=True)  # fmt: skip
    train = au.Groups(batch["x"], batch["row"], batch["V"])
    rngs = [np.random.default_rng(a) for a in audit_seeds]
    out = np.zeros((n, N_EXTRA, len(GEO)))
    for k in range(N_EXTRA):
        est = au.estimate(tb, th, au.sample_groups(tb, th, rngs), train, "adam", None)
        out[:, k] = np.stack([est[g] for g in GEO], -1)
    return out


def main(argv: list[str]) -> int:
    runs_dir = Path(argv[1]).resolve()
    cfg = provenance.load_config(CONFIG)
    d = an.load(runs_dir)
    out_dir = provenance.create_run_dir(REPO / "results", "E004a-stage1-posthoc", REPO)
    provenance.write_metadata(out_dir, "E004a-stage1-posthoc", CONFIG, REPO,
                              extra={"split": "design", "post_hoc": True,
                                     "root_seed": pn.ROOT_SEED,
                                     "panel_sha256": cfg["panel_sha256"],
                                     "runs": str(runs_dir.relative_to(REPO)),
                                     "extra_audit_seeds": "spare -> resample child 3"})  # fmt: skip
    structs = d["structs"]
    train_seeds = s0b.seed_table(len(structs))["prim_ver"]
    resample = s1r.audit_tree(len(structs))["resample"].spawn(4)
    extra_seeds = [c.spawn(s0b.N_SEEDS) for c in resample[3].spawn(len(structs))]
    with ProcessPoolExecutor(max_workers=8) as pool:
        extra = list(pool.map(_extra_t0, [(structs[i], train_seeds[i], extra_seeds[i])
                                          for i in range(len(structs))], chunksize=8))  # fmt: skip
    extra_arr = np.stack(extra)  # (768, 4, 2, 3)
    tab = an.table(d, "adam")
    kept = [i for i, k in enumerate(d["keep"]) if k]
    ex = extra_arr[kept].reshape(len(tab["rows"]), N_EXTRA, len(GEO))
    g0 = np.stack([tab["est"][k][:, 0] for k in GEO], -1)  # original t = 0 audit
    g0x3 = np.nanmean(np.concatenate([g0[:, None], ex], 1), axis=1)
    mech, g = an.col(tab, "mechanism"), an.col(tab, "i")
    dn, fail = an.col(tab, "Dn"), an.col(tab, "failure").astype(int)
    folds = pr.outer_folds(mech, g)
    sets: dict[tuple[str, float], np.ndarray] = {}
    for h in an.H:
        f = an.features(tab, h)
        if h == 0:
            sets[("L1", 0.0)] = f["L1"]
            sets[("L1x3", 0.0)] = np.column_stack([f["L0"], g0x3])
        sets[("L2", h)] = f["L2"]
        sets[("L3", h)] = f["L3"]
        sets[("L2+G0", h)] = np.column_stack([f["L2"], g0])
        sets[("L2+G0x3", h)] = np.column_stack([f["L2"], g0x3])
    jobs = [((lv, h, kind, "linear"), X, y, g, folds, kind, "linear")
            for (lv, h), X in sets.items()
            for kind, y in (("ridge", dn), ("binary", fail), ("multi", mech))]  # fmt: skip
    jobs.sort(key=lambda j: j[5] != "multi")
    res = an.run_jobs(jobs)
    W = pr.hier_weights(g, an.B, np.random.default_rng(resample[1]))
    bt = an.Booter(W)
    t_on = an.col(tab, "t_on")
    report: dict[str, Any] = {"n_runs": len(dn)}
    static = ("L1", "L1x3")
    for h in an.H:
        nyv = t_on > h
        cur: dict[str, Any] = {}
        for lv in ("L1", "L1x3", "L2", "L2+G0", "L2+G0x3", "L3"):
            kh = 0.0 if lv in static else h
            r = res[(lv, kh, "multi", "linear")]
            pred = r["classes"][r["P"].argmax(1)]
            ya_yb = np.isin(mech, ("YA", "YB"))
            cur[lv] = {
                "cindex_Dn": bt.pairwise(res[(lv, kh, "ridge", "linear")]["pred"], dn, "cindex"),
                "auroc_failure": bt.pairwise(res[(lv, kh, "binary", "linear")]["pred"],
                                             fail.astype(bool), "auroc"),
                "auroc_nyv": bt.pairwise(res[(lv, kh, "binary", "linear")]["pred"],
                                         fail.astype(bool), "auroc", nyv),
                "macro_f1": bt.classes(mech, pred, r["classes"])["f1"],
                "route_ab_auroc": bt.pairwise(an.binary_from_multi(r["P"], r["classes"], ("YA",),
                                                                   ("YB",)),
                                              mech == "YA", "auroc", ya_yb),
            }  # fmt: skip
        ent: dict[str, Any] = {lv: {m: an.summ(*v) for m, v in c.items()} for lv, c in cur.items()}
        for a, b in (("L1x3", "L1"), ("L3", "L1x3"), ("L3", "L2+G0x3"), ("L2+G0x3", "L2+G0"),
                     ("L2+G0", "L2"), ("L3", "L2")):  # fmt: skip
            ent[f"{a}-{b}"] = {m: an.diff(cur[a][m], cur[b][m]) for m in cur[a]}
        report[f"{h:g}"] = ent
        print(f"h={h:g}: L3-L2+G0x3 AUROC {ent['L3-L2+G0x3']['auroc_failure']['point']:+.3f}, "
              f"C-index {ent['L3-L2+G0x3']['cindex_Dn']['point']:+.3f}", flush=True)  # fmt: skip
    report["geometry_names"] = GEO
    report["level_dims"] = {f"{lv}": int(X.shape[1]) for (lv, h), X in sets.items() if h in (0.0,)}
    report["extra_t0_sd_across_audits"] = {
        k: float(np.nanmean(np.nanstd(np.concatenate([g0[:, None, j : j + 1], ex[..., j : j + 1]],
                                                      1), axis=1)))
        for j, k in enumerate(GEO)
    }  # fmt: skip
    (out_dir / "stage1_posthoc.json").write_text(json.dumps(an.strip(report), indent=1) + "\n")
    print(f"run directory: {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
