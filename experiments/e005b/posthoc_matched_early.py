"""E005b-0 matched experiment — POST-HOC descriptive reading of already-logged data (not part of
the frozen analysis, research/10_e005b0_pilot.md §15): how long the initial FPR match lasted.
Per run: training-batch FPR at steps 1/5/10/20/30, the first step with batch FPR >= 0.5, and dev
sampled accuracy at the first evaluations. No new model evaluation.
Usage: posthoc_matched_early.py <run-dir> ..."""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402

MATCHED = cm.REPO / "configs" / "e005b" / "matched.toml"


def jl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main(argv: list[str]) -> int:
    rows: dict[str, Any] = {}
    for a in argv[1:]:
        d = Path(a).resolve()
        s = json.loads((d / "summary.json").read_text())
        lg, ev = jl(d / "grpo_log.jsonl"), jl(d / "eval_log.jsonl")
        fpr = {r["step"]: r["confusion"]["fpr"] for r in lg if r["confusion"]["n_neg"]}
        rows[f"{s['rule']}-s{s['seed']}"] = {
            "run": str(d.relative_to(cm.REPO)),
            "batch_fpr": {str(k): fpr.get(k) for k in (1, 5, 10, 20, 30)},
            "first_step_fpr_ge_0.5": next((k for k, v in sorted(fpr.items()) if v >= 0.5), None),
            "dev_sampled": {str(e["step"]): e["sampled"] for e in ev[:5]},
        }
    out = provenance.create_run_dir(cm.REPO / "results", "E005b0-posthoc-matched-early", cm.REPO)
    provenance.write_metadata(out, "E005b0-posthoc-matched-early", MATCHED, cm.REPO,
                              extra={"post_hoc": True})  # fmt: skip
    (out / "early.json").write_text(json.dumps(rows, indent=1) + "\n")
    for k, r in sorted(rows.items()):
        print(k, {s: round(v, 3) for s, v in r["batch_fpr"].items() if v is not None},
              "first>=0.5:", r["first_step_fpr_ge_0.5"])  # fmt: skip
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
