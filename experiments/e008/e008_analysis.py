"""E008 analysis — FROZEN before any E008 run (research/paper/e008_protocol.md). Dev only.
Usage: e008_analysis.py --e006 <E006 run dirs for clean, cov50, exploit, rarekey ...>
                        --e008 <E008 run dirs (5 arms x seeds 11-15)>
Rules:
- L1: in every seed primary(covhard) < primary(cov50) < primary(coveasy); seed-mean harm of
  covhard >= 0.2;
- R1: seed-mean harm non-decreasing over rarekey, set02, set05, setq; setq collapses (primary < 0.2)
  in >= 4 of 5 seeds;
- R2: |harm(setq) - harm(exploit)| <= 0.15.
harm = -(seed-mean of primary(arm) - primary(clean)) with clean from E006 at the same seeds.
"""

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

CFG = cm.REPO / "configs" / "e008" / "e008.toml"


def main(argv: list[str]) -> int:
    cm.load_config()
    ec = provenance.load_config(CFG)
    lab = ec["labels"]
    i6, i8 = argv.index("--e006"), argv.index("--e008")
    d6 = [Path(x).resolve() for x in argv[i6 + 1 : i8]]
    d8 = [Path(x).resolve() for x in argv[i8 + 1 :]]
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for d in d6 + d8:
        s = json.loads((d / "summary.json").read_text())
        assert not s["smoke"] and s["steps"] == 1000
        runs[(s["arm"], s["seed"])] = an6.run_stats(d, lab["final_evals"], lab["window"])
    seeds = list(ec["seeds"])
    need = {(a, s) for a in ["clean", "cov50", "exploit", "rarekey", *ec["arms"]] for s in seeds}
    assert need <= set(runs), f"missing {sorted(need - set(runs))}"
    out = provenance.create_run_dir(cm.REPO / "results", "E008-analysis", cm.REPO)
    provenance.write_metadata(out, "E008-analysis", CFG, cm.REPO,
                              extra={"runs": {f"{a}-s{s}": r["run"]
                                              for (a, s), r in sorted(runs.items())}})  # fmt: skip
    prim = {a: [runs[(a, s)]["primary"] for s in seeds]
            for a in ["clean", "cov50", "exploit", "rarekey", *ec["arms"]]}  # fmt: skip
    harm = {a: -float(np.mean(np.array(v) - np.array(prim["clean"]))) for a, v in prim.items()}
    paired: dict[str, dict[str, Any]] = {}
    for a, v in prim.items():
        if a == "clean":
            continue
        diff = (np.array(v) - np.array(prim["clean"])).tolist()
        paired[a] = {"by_seed": dict(zip(map(str, seeds), diff, strict=True)),
                     "label": an6.label(diff, lab["effect"]),
                     "collapsed_seeds": int(sum(x < lab["collapse"] for x in v))}  # fmt: skip
    per_seed = all(h < m < e for h, m, e in zip(prim["covhard"], prim["cov50"], prim["coveasy"],
                                                strict=True))  # fmt: skip
    L1 = {"per_seed": per_seed, "covhard_harm": harm["covhard"],
          "pass": per_seed and harm["covhard"] >= 0.2}  # fmt: skip
    seq = [harm[a] for a in ("rarekey", "set02", "set05", "setq")]
    mono = all(x <= y for x, y in zip(seq, seq[1:], strict=False))
    n_col = int(paired["setq"]["collapsed_seeds"])
    R1 = {"harm_seq": seq, "monotone": mono, "setq_collapsed": n_col,
          "pass": mono and n_col >= 4}  # fmt: skip
    d2 = harm["setq"] - harm["exploit"]
    R2 = {"diff": d2, "pass": abs(d2) <= 0.15}
    rep = {"primary": prim, "harm": harm, "paired": paired, "L1": L1, "R1": R1, "R2": R2,
           "runs": {f"{a}-s{s}": r for (a, s), r in sorted(runs.items())}}  # fmt: skip
    (out / "e008_analysis.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print(json.dumps({"harm": harm, "L1": L1, "R1": R1, "R2": R2,
                      "collapsed": {a: p["collapsed_seeds"] for a, p in paired.items()}},
                     indent=1, default=float))  # fmt: skip
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
