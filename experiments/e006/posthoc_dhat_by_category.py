"""E006 POST-HOC (not pre-registered): D-hat by task category on the calibration audit, to examine
the failed H3b (cov25 / randfp had negative pooled D-hat but M grew). Same definition as
predictions.py, restricted to the groups of each category. Usage: posthoc_dhat_by_category.py"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import audit as au  # noqa: E402
import common as cm  # noqa: E402
import predictions as pr  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402
from vdyn.e006 import verifiers as vf  # noqa: E402


def main() -> int:
    cfg = cm.load_config()
    a = provenance.load_config(au.MATCHED)["audit"]
    arms = json.loads(au.ARMS.read_text())
    run_dir = provenance.create_run_dir(cm.REPO / "results", "E006-posthoc-dhat-category", cm.REPO)
    provenance.write_metadata(run_dir, "E006-posthoc-dhat-category", au.E006, cm.REPO,
                              extra={"post_hoc": True})  # fmt: skip
    cal = au.audit_samples(cfg, a, "calibrate")
    g = cal["group"]
    rep = [p for p in cal["pairs"] for _ in range(g)]
    parsed = [tk.parse_completion(t) for t in cal["toks"]]
    inM = np.array([vf.in_master_set(x, y, *pv) for (x, y), pv in zip(rep, parsed, strict=True)])
    cat = np.array([tk.category(x, y) for x, y in cal["pairs"]])
    out: dict[str, dict[str, float]] = {}
    for name in ("randfp", "cov25", "cov50", "cov75", "exploit"):
        acc = pr.accept_fn(name, arms, set())
        rng = np.random.default_rng(20261361)
        per = []
        for _ in range(20 if name != "exploit" else 1):
            V = np.array([acc((x, y), t, rng) for (x, y), t in zip(rep, cal["toks"], strict=True)])
            A = pr.advantages(V.reshape(-1, g))
            per.append((A * inM.reshape(-1, g)).sum(1) / g)
        push = np.mean(per, axis=0)
        out[name] = {"pooled": float(push.mean()),
                     **{c: float(push[cat == c].mean()) for c in tk.CATEGORIES},
                     **{f"M_mass_{c}": float(inM.reshape(-1, g)[cat == c].mean())
                        for c in tk.CATEGORIES}}  # fmt: skip
        print(name, {k: round(v, 4) for k, v in out[name].items()})
    (run_dir / "dhat_by_category.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
