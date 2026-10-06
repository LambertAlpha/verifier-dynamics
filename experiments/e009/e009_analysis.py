"""E009 analysis — FROZEN before any E009 run (research/paper/e009_protocol.md). Dev only.
Usage: e009_analysis.py <run-dir> ... (6 arms x seeds 11-15)

Regional accuracy = mean over the final 4 evaluations of the in-region / out-of-region sampled dev
accuracy logged by e009_run.py. Rules:
- P1 aeven: in-region accuracy < 0.2 in >= 4/5 seeds, and seed-mean out-of-region accuracy
  >= clean's out-of-region accuracy (same region) - 0.1;
- P2 sumeven: the same for its region;
- P3 hardhalf: in-region accuracy >= 0.3 in >= 4/5 seeds (contrast: covhard, whose three-digit
  accuracy fell to 0.04 in E008);
- P4 global harm (seed mean, vs clean): aeven >= 0.25 and sumeven >= 0.25; hardhalf < 0.2;
- integrity: the final hashes of clean / cov50 / covhard equal E006 / E008's at the same seeds.
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

CFG = cm.REPO / "configs" / "e009" / "e009.toml"
REF = {"clean": "E006-clean", "cov50": "E006-cov50", "covhard": "E008-covhard"}


def main(argv: list[str]) -> int:
    ec = provenance.load_config(CFG)
    lab = ec["labels"]
    seeds, arms, regions = list(ec["seeds"]), list(ec["arms"]), list(ec["regions"])
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for x in argv[1:]:
        d = Path(x).resolve()
        s = json.loads((d / "summary.json").read_text())
        assert not s["smoke"] and s["steps"] == 1000
        st = an6.run_stats(d, lab["final_evals"], lab["window"])
        ev = an6.jl(d / "eval_log.jsonl")[-lab["final_evals"] :]
        st["region"] = {r: {k: float(np.mean([e["region_acc"][r][k] for e in ev]))
                            for k in ("in", "out")} for r in regions}  # fmt: skip
        st["final_sha256"] = s["final_sha256"]
        runs[(s["arm"], s["seed"])] = st
    assert set(runs) == {(a, s) for a in arms for s in seeds}, "need all 30 runs"
    out = provenance.create_run_dir(cm.REPO / "results", "E009-analysis", cm.REPO)
    provenance.write_metadata(out, "E009-analysis", CFG, cm.REPO,
                              extra={"runs": {f"{a}-s{s}": r["run"]
                                              for (a, s), r in sorted(runs.items())}})  # fmt: skip
    prim = {a: np.array([runs[(a, s)]["primary"] for s in seeds]) for a in arms}
    harm = {a: float(-(prim[a] - prim["clean"]).mean()) for a in arms}
    reg = {a: {r: {k: [runs[(a, s)]["region"][r][k] for s in seeds] for k in ("in", "out")}
               for r in regions} for a in arms}  # fmt: skip

    def collapse_region(a: str) -> dict[str, Any]:
        ins = reg[a][a]["in"]
        out_mean = float(np.mean(reg[a][a]["out"]))
        clean_out = float(np.mean(reg["clean"][a]["out"]))
        return {"in_by_seed": ins, "in_lt_0.2": int(sum(x < 0.2 for x in ins)),
                "out_mean": out_mean, "clean_out_mean": clean_out,
                "pass": sum(x < 0.2 for x in ins) >= 4 and out_mean >= clean_out - 0.1}  # fmt: skip

    hh = reg["hardhalf"]["hardhalf"]["in"]
    hyp = {"P1": collapse_region("aeven"), "P2": collapse_region("sumeven"),
           "P3": {"in_by_seed": hh, "pass": sum(x >= 0.3 for x in hh) >= 4},
           "P4": {"harm": harm, "pass": harm["aeven"] >= 0.25 and harm["sumeven"] >= 0.25
                  and harm["hardhalf"] < 0.2}}  # fmt: skip
    integ = {}
    for a, prefix in REF.items():
        for s in seeds:
            ref = sorted((cm.REPO / "results" / f"{prefix}-s{s}").glob("*/summary.json"))
            want = json.loads(ref[0].read_text())["final_sha256"]
            integ[f"{a}-s{s}"] = runs[(a, s)]["final_sha256"] == want
    hyp["integrity"] = {"all_equal": all(integ.values()), "by_run": integ}
    rep = {"harm": harm, "primary": {a: v.tolist() for a, v in prim.items()}, "regional": reg,
           "hypotheses": hyp,
           "runs": {f"{a}-s{s}": r for (a, s), r in sorted(runs.items())}}  # fmt: skip
    (out / "e009_analysis.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print(json.dumps({"harm": harm, "hypotheses": {k: v["pass"] if "pass" in v else v["all_equal"]
                                                   for k, v in hyp.items()},
                      "regional_means": {a: {r: {k: round(float(np.mean(v)), 3)
                                                 for k, v in d.items()} for r, d in reg[a].items()}
                                         for a in arms}}, indent=1, default=float))  # fmt: skip
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
