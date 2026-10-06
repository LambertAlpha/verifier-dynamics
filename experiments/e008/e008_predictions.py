"""E008 pre-training check L2 (research/paper/e008_protocol.md): the initial GRPO advantage mass
on M by task category for covhard / coveasy (calibration audit, 20 coin replicates), plus the base
share of each keyset arm. Base samples only; committed before any E008 run."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e006"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import audit as au6  # noqa: E402
import common as cm  # noqa: E402
import e008_audit as au8  # noqa: E402
import predictions as pr6  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402
from vdyn.e006 import verifiers as vf  # noqa: E402


def main() -> int:
    cfg = cm.load_config()
    a = provenance.load_config(au6.MATCHED)["audit"]
    arms = json.loads(au8.ARMS.read_text())
    if provenance.git_state(cm.REPO)["dirty"]:
        print("STOP: working tree is dirty")
        return 1
    run_dir = provenance.create_run_dir(cm.REPO / "results", "E008-predictions", cm.REPO)
    extra = cm.run_extra(cfg, arms_file_sha256=au6.sha(au8.ARMS))
    provenance.write_metadata(run_dir, "E008-predictions", au8.CFG, cm.REPO, extra=extra)
    cal = au6.audit_samples(cfg, a, "calibrate")
    g = cal["group"]
    rep = [p for p in cal["pairs"] for _ in range(g)]
    parsed = [tk.parse_completion(t) for t in cal["toks"]]
    inM = np.array([vf.in_master_set(x, y, *pv) for (x, y), pv in zip(rep, parsed, strict=True)])
    cat = np.array([tk.category(x, y) for x, y in cal["pairs"]])
    out: dict[str, dict[str, float]] = {}
    for arm in ("covhard", "coveasy"):
        spec = vf.Spec(**arms["specs"][arm])
        rng = np.random.default_rng(20261361)
        per = []
        for _ in range(20):
            V = np.array([vf.reward(spec, x, y, t, rng)
                          for (x, y), t in zip(rep, cal["toks"], strict=True)])  # fmt: skip
            A = pr6.advantages(V.reshape(-1, g))
            per.append((A * inM.reshape(-1, g)).sum(1) / g)
        push = np.mean(per, axis=0)
        out[arm] = {"pooled": float(push.mean()),
                    **{c: float(push[cat == c].mean()) for c in tk.CATEGORIES}}  # fmt: skip
        print(arm, {k: round(v, 4) for k, v in out[arm].items()})
    l2 = out["covhard"]["three-digit"] > 0 and all(out["coveasy"][c] <= 0 for c in tk.CATEGORIES)
    res = {"push_by_category": out, "L2_pass": bool(l2), "info": arms["info"]}
    (run_dir / "predictions.json").write_text(json.dumps(res, indent=1) + "\n")
    print("L2", "PASS" if l2 else "FAIL")
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
