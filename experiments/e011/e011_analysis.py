"""E011 analysis — FROZEN before any E011 run (research/paper/e011_protocol.md). Dev only.
Usage: e011_analysis.py --predictions <E011-predictions-dir> --clean <E006 clean run dirs> --runs
       <E011 run dirs>
Per arm: seed-mean harm = clean primary - arm primary (same seeds); the rule is right iff
(harm >= 0.25) == (prediction == "harm>=0.25"); "undetermined" arms are reported, not scored.
E011 passes iff the rule is right for all scored arms (all three, unless undetermined)."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e006"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import analysis as an6  # noqa: E402
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402

CFG = cm.REPO / "configs" / "e011" / "e011.toml"


def main(argv: list[str]) -> int:
    ec = provenance.load_config(CFG)
    lab = ec["labels"]
    ip, ic, ir = argv.index("--predictions"), argv.index("--clean"), argv.index("--runs")
    pred = json.loads((Path(argv[ip + 1]) / "predictions.json").read_text())
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for x in argv[ic + 1 : ir] + argv[ir + 1 :]:
        d = Path(x).resolve()
        s = json.loads((d / "summary.json").read_text())
        assert not s["smoke"] and s["steps"] == 1000
        runs[(s["arm"], s["seed"])] = an6.run_stats(d, lab["final_evals"], lab["window"])
    seeds = list(ec["seeds"])
    assert set(runs) == {(a, s) for a in ["clean", *ec["arms"]] for s in seeds}
    out = provenance.create_run_dir(cm.REPO / "results", "E011-analysis", cm.REPO)
    provenance.write_metadata(out, "E011-analysis", CFG, cm.REPO,
                              extra={"runs": {f"{a}-s{s}": r["run"]
                                              for (a, s), r in sorted(runs.items())}})  # fmt: skip
    clean = np.array([runs[("clean", s)]["primary"] for s in seeds])
    res = {}
    for a in ec["arms"]:
        prim = np.array([runs[(a, s)]["primary"] for s in seeds])
        harm = float((clean - prim).mean())
        p = pred[a]["prediction"]
        right = (
            None if p == "undetermined" else (harm >= ec["harm_threshold"]) == (p == "harm>=0.25")
        )
        res[a] = {"harm": harm, "by_seed": (clean - prim).tolist(), "prediction": p,
                  "acm_0.75": pred[a]["acm_0.75"], "right": right,
                  "collapsed": int(sum(x < lab["collapse"] for x in prim)),
                  "label": an6.label(list(prim - clean), lab["effect"])}  # fmt: skip
    scored = [v["right"] for v in res.values() if v["right"] is not None]
    rep = {"arms": res, "pass": bool(scored) and all(scored),
           "runs": {f"{a}-s{s}": r for (a, s), r in sorted(runs.items())}}  # fmt: skip
    (out / "e011_analysis.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print(json.dumps({"arms": res, "pass": rep["pass"]}, indent=1, default=float))
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
