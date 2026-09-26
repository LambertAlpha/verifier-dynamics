"""[POST-HOC, NOT PRE-REGISTERED] E004a Stage 0: is the oracle geometry advantage more than
mechanism recognition? DESIGN SPLIT ONLY; exact observables; cross-validated diagnostics.

(a) Within-mechanism failure AUROC at h* (X, YA, YB, D, where failure varies), from full-panel
    out-of-fold predictions.
(b) Type-conditioned levels: each level plus the one-hot mechanism label. If L3 + type still beats
    L2 + type, geometry carries outcome information beyond recognizing the mechanism.
(c) Dynamics vs snapshot: L3 vs L1 (both with and without type).
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0_analysis as sa  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e002 import endpoints as ep  # noqa: E402
from vdyn.e004 import panel as pn  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e004" / "e004a.toml"
LEVELS = ("L0", "L1", "L2", "L3")


def main(argv: list[str]) -> int:
    run_dir = Path(argv[1]).resolve()
    d = sa.load(run_dir)
    fr = d["summary"]["fracs"]
    out_dir = provenance.create_run_dir(REPO / "results", "E004a-stage0-posthoc", REPO)
    provenance.write_metadata(
        out_dir,
        "E004a-stage0-posthoc",
        CONFIG,
        REPO,
        extra={"split": "design", "post_hoc": True, "runs": str(run_dir)},
    )
    report: dict[str, Any] = {}
    for opt, group, key, seeds in (
        ("adam", "adam_prim_ver", "adam_prim", True),
        ("ng", "ng_prim_ver", "ng_prim", False),
    ):
        tab = sa.run_table(d, group, key, seeds=seeds)
        rows = tab["rows"]
        fail = np.array([r["failure"] for r in rows]).astype(int)
        dn = np.array([r["Dn"] for r in rows])
        mech = np.array([r["mechanism"] for r in rows])
        groups = np.array([r["i"] for r in rows])
        onehot = np.stack([(mech == m).astype(float) for m in pn.MECHANISMS], 1)
        res: dict[str, Any] = {}
        for level in LEVELS:
            X = sa.features(tab, fr, sa.H_STAR, level)
            for tag, Xu in ((level, X), (f"{level}+type", np.hstack([X, onehot]))):
                p = sa.cv_predict(Xu, fail, groups, mech, "binary")
                pdn = sa.cv_predict(Xu, dn, groups, mech, "ridge")
                within = {}
                for m in ("X", "YA", "YB", "D"):
                    sel = mech == m
                    if 0 < fail[sel].sum() < sel.sum():
                        within[m] = {
                            "auroc_failure": ep.auroc(p[sel], fail[sel].astype(bool)),
                            "cindex_Dn": ep.c_index(pdn[sel], dn[sel]),
                        }
                res[tag] = {
                    "auroc_failure": ep.auroc(p, fail.astype(bool)),
                    "cindex_Dn": ep.c_index(pdn, dn),
                    "within_mechanism": within,
                }
        report[opt] = res
        print(
            opt,
            json.dumps(
                {
                    k: (round(v["auroc_failure"], 3), round(v["cindex_Dn"], 3))
                    for k, v in res.items()
                }
            ),
        )
    (out_dir / "posthoc.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"run directory: {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
