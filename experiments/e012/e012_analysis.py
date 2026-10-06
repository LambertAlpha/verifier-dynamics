"""E012 analysis — FROZEN before any E012 run (research/paper/e012_protocol.md). Dev only.
Usage: e012_analysis.py --predictions <E012-predictions dir> --verification <E012-verification dir>
       --clean <E006 clean run dirs> --probes <E012 probe dirs> --runs <E012 full run dirs>
Scores the frozen probe rule; reports static FPR / J for contrast; checks that each full run's
first 150 steps equal its probe run (determinism)."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e006"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import analysis as an6  # noqa: E402
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402

CFG = cm.REPO / "configs" / "e012" / "e012.toml"


def args(argv: list[str], flag: str) -> list[str]:
    i = argv.index(flag)
    j = next((k for k in range(i + 1, len(argv)) if argv[k].startswith("--")), len(argv))
    return argv[i + 1 : j]


def main(argv: list[str]) -> int:
    ec = provenance.load_config(CFG)
    lab = ec["labels"]
    pred = json.loads((Path(args(argv, "--predictions")[0]) / "predictions.json").read_text())
    audit = json.loads((Path(args(argv, "--verification")[0]) / "audit.json").read_text())
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for x in args(argv, "--clean") + args(argv, "--runs"):
        d = Path(x).resolve()
        s = json.loads((d / "summary.json").read_text())
        assert not s["smoke"] and not s.get("probe") and s["steps"] == 1000
        runs[(s["arm"], s["seed"])] = an6.run_stats(d, lab["final_evals"], lab["window"]) | {
            "dir": d
        }
    seeds = list(ec["seeds"])
    assert set(runs) == {(a, s) for a in ["clean", *ec["arms"]] for s in seeds}
    same = {}
    for x in args(argv, "--probes"):
        d = Path(x).resolve()
        s = json.loads((d / "summary.json").read_text())
        pl = an6.jl(d / "grpo_log.jsonl")
        fl = an6.jl(runs[(s["arm"], s["seed"])]["dir"] / "grpo_log.jsonl")[: len(pl)]
        same[f"{s['arm']}-s{s['seed']}"] = all(
            (a["gold"], a["verifier"]) == (b["gold"], b["verifier"])
            for a, b in zip(pl, fl, strict=True)
        )
    out = provenance.create_run_dir(cm.REPO / "results", "E012-analysis", cm.REPO)
    run_map = {f"{a}-s{s}": r["run"] for (a, s), r in sorted(runs.items())}
    provenance.write_metadata(out, "E012-analysis", CFG, cm.REPO, extra={"runs": run_map})
    clean = np.array([runs[("clean", s)]["primary"] for s in seeds])
    res = {}
    for a in ec["arms"]:
        prim = np.array([runs[(a, s)]["primary"] for s in seeds])
        harm = float((clean - prim).mean())
        p = pred[a]["prediction"]
        right = (
            None if p == "undetermined" else (harm >= ec["harm_threshold"]) == (p == "harm>=0.25")
        )
        st = audit["arms"][a]["overall"]
        res[a] = {"harm": harm, "by_seed": (clean - prim).tolist(), "rise_150": pred[a]["rise_150"],
                  "prediction": p, "right": right, "static_fpr": st["fpr"],
                  "static_J": 1 - st["fnr"] - st["fpr"],
                  "collapsed": int(sum(x < lab["collapse"] for x in prim))}  # fmt: skip
    scored = [v["right"] for v in res.values() if v["right"] is not None]
    names = list(res)
    hv = [res[n]["harm"] for n in names]
    sp = {k: float(spearmanr([res[n][k] for n in names], hv).statistic)
          for k in ("rise_150", "static_fpr")}  # fmt: skip
    rep = {"arms": res, "pass": bool(scored) and all(scored), "spearman": sp,
           "probe_full_identical": same, "all_identical": all(same.values())}  # fmt: skip
    (out / "e012_analysis.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps(rep, indent=1, default=str))
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
