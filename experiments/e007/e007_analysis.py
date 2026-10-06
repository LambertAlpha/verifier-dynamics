"""E007 analysis — FROZEN before any E007 run (research/paper/e007_protocol.md).
Usage: e007_analysis.py --verification <E007-verification-run-dir> <run-dir> ... (5 arms x seeds)

Primary: greedy accuracy on the full GSM8K test set (1319) at the final step. Paired differences
vs clean per seed; labels as §12 ("lower" if mean <= -0.05 and all seeds < 0; "higher" symmetric;
else "little difference"). Rules:
- E1 randfp and E2 hashtab: "little difference";
- E3 ends0: "lower", and the mean share of final answers ending in 0 among wrong eval responses
  rises from step 0 to the final step (seed mean);
- E4 anywhere: "lower", and numbers per eval response rise (seed mean);
- E5 gold-free RME (verification audit): ends0 > max(randfp, hashtab, clean) and > anywhere.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402

CFG = cm.REPO / "configs" / "e007" / "e007.toml"


def jl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def label(ds: list[float]) -> str:
    m = float(np.mean(ds))
    if m <= -0.05 and all(x < 0 for x in ds):
        return "lower"
    if m >= 0.05 and all(x > 0 for x in ds):
        return "higher"
    return "little difference"


def main(argv: list[str]) -> int:
    cfg = provenance.load_config(CFG)
    i = argv.index("--verification")
    ver = Path(argv[i + 1]).resolve()
    audit = json.loads((ver / "audit.json").read_text())
    dirs = [Path(x).resolve() for k, x in enumerate(argv[1:], 1) if k not in (i, i + 1)]
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for d in dirs:
        s = json.loads((d / "summary.json").read_text())
        assert not s["pilot"] and s["steps"] == cfg["rl"]["steps"]
        ev = jl(d / "eval_log.jsonl")
        lg = jl(d / "grpo_log.jsonl")
        runs[(s["arm"], s["seed"])] = {
            "primary": s["final_full_test"]["greedy_acc"],
            "subset_start": ev[0]["greedy_acc"], "subset_final": ev[-2]["greedy_acc"],
            "ends0_start": ev[0]["ends0_share_wrong"], "ends0_final": ev[-2]["ends0_share_wrong"],
            "numbers_start": ev[0]["numbers_per_response"],
            "numbers_final": ev[-2]["numbers_per_response"],
            "fpr_first20": float(np.nanmean([r["fpr"] for r in lg[:20] if r["fpr"] is not None])),
            "fpr_last20": float(np.nanmean([r["fpr"] for r in lg[-20:] if r["fpr"] is not None])),
            "mixed_v_last20": float(np.mean([r["mixed_v"] for r in lg[-20:]])),
            "gold_last20": float(np.mean([r["gold"] for r in lg[-20:]])),
            "run": str(d.relative_to(cm.REPO)),
        }  # fmt: skip
    seeds = sorted({s for _, s in runs})
    arms = list(cfg["arms"])
    assert set(runs) == {(a, s) for a in arms for s in seeds}, "need every arm at every seed"
    out = provenance.create_run_dir(cm.REPO / "results", "E007-analysis", cm.REPO)
    provenance.write_metadata(out, "E007-analysis", CFG, cm.REPO,
                              extra={"runs": {f"{a}-s{s}": r["run"]
                                              for (a, s), r in sorted(runs.items())}})  # fmt: skip
    paired = {}
    for a in arms[1:]:
        ds = [runs[(a, s)]["primary"] - runs[("clean", s)]["primary"] for s in seeds]
        paired[a] = {"by_seed": dict(zip(map(str, seeds), ds, strict=True)),
                     "mean": float(np.mean(ds)), "label": label(ds)}  # fmt: skip

    def mean(a: str, k: str) -> float:
        return float(np.mean([runs[(a, s)][k] for s in seeds]))

    rme = {a: audit["rme"][a]["rme"] for a in arms}
    hyp = {
        "E1": paired["randfp"]["label"] == "little difference",
        "E2": paired["hashtab"]["label"] == "little difference",
        "E3": paired["ends0"]["label"] == "lower"
        and mean("ends0", "ends0_final") > mean("ends0", "ends0_start"),
        "E4": paired["anywhere"]["label"] == "lower"
        and mean("anywhere", "numbers_final") > mean("anywhere", "numbers_start"),
        "E5": rme["ends0"] > max(rme["randfp"], rme["hashtab"], rme["clean"], rme["anywhere"]),
    }
    rep = {"paired": paired, "hypotheses": hyp, "rme": rme,
           "runs": {f"{a}-s{s}": r for (a, s), r in sorted(runs.items())},
           "arm_means": {a: {k: mean(a, k) for k in next(iter(runs.values())) if k != "run"}
                         for a in arms}}  # fmt: skip
    (out / "e007_analysis.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print(json.dumps({"paired": paired, "hypotheses": hyp, "rme": rme}, indent=1, default=float))
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
