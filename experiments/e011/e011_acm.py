"""E011 pre-training predictions: ACM_tau of each arm on the calibration-audit base samples, and the
frozen rule's prediction (research/paper/e011_protocol.md). Committed before any E011 run."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e006"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import audit as au6  # noqa: E402
import common as cm  # noqa: E402
import posthoc_conditional_coverage as pcc  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402
from vdyn.e006 import verifiers as vf  # noqa: E402

CFG = cm.REPO / "configs" / "e011" / "e011.toml"
ARMS = cm.REPO / "configs" / "e011" / "arms.json"


def main() -> int:
    cfg = cm.load_config()
    ec = provenance.load_config(CFG)
    a = provenance.load_config(au6.MATCHED)["audit"]
    if provenance.git_state(cm.REPO)["dirty"]:
        print("STOP: working tree is dirty")
        return 1
    arms = json.loads(ARMS.read_text())["specs"]
    run_dir = provenance.create_run_dir(cm.REPO / "results", "E011-predictions", cm.REPO)
    provenance.write_metadata(run_dir, "E011-predictions", CFG, cm.REPO,
                              extra={"arms_file_sha256": au6.sha(ARMS)})  # fmt: skip
    S = au6.audit_samples(cfg, a, "calibrate")
    g = S["group"]
    rep = [p for p in S["pairs"] for _ in range(g)]
    parsed = [tk.parse_completion(t) for t in S["toks"]]
    wrong = [i for i, ((x, y), pv) in enumerate(zip(rep, parsed, strict=True))
             if tk.gold_reward(x, y, *pv) == 0.0]  # fmt: skip
    out = {}
    for name, d in arms.items():
        spec = vf.Spec(**d)
        acc = defaultdict(list)
        for i in wrong:
            x, y = rep[i]
            acc[vf.completion_key(S["toks"][i])].append(
                pcc.expected_accept(spec, x, y, S["toks"][i])
            )
        m = np.array([len(v) for v in acc.values()], float) / len(wrong)
        c = np.array([np.mean(v) for v in acc.values()])
        acm = float(m[c >= ec["acm_tau"]].sum())
        acm6 = float(m[c >= 0.6].sum())
        band = abs(acm - ec["acm_threshold"]) <= ec["undetermined_band"]
        pred = "undetermined" if band else ("harm>=0.25" if acm >= ec["acm_threshold"]
                                            else "harm<0.25")  # fmt: skip
        out[name] = {"acm_0.75": acm, "acm_0.6": acm6, "fpr_mass": float((m * c).sum()),
                     "prediction": pred,
                     "secondary_prediction_acm_0.6": "harm>=0.25" if acm6 >= ec["acm_threshold"]
                     else "harm<0.25"}  # fmt: skip
        print(name, out[name], flush=True)
    (run_dir / "predictions.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
