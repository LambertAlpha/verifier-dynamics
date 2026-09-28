"""E005b-0 POST-HOC diagnostic (not in the matrix protocol; gates nothing): what do V3-trained
policies output? Dev samples (4 per item, the matrix evaluation seed) from the final checkpoints of
the V3 and V0 runs: answer minus truth, round-down / round-nearest rates, most common answers.
Usage: posthoc_v3_answers.py <run-dir> ..."""

import collections
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import calib as cal  # noqa: E402
from vdyn.e005b import model as mdl  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402

MATRIX = cm.REPO / "configs" / "e005b" / "matrix.toml"


def profile(pairs: list[tuple[int, int]], toks: list[list[int]]) -> dict[str, Any]:
    diffs, vals = [], []
    for (a, b), t in zip(pairs, toks, strict=True):
        ok, val = tk.parse_completion(t)
        if ok and val is not None:
            diffs.append(val - (a + b))
            vals.append(val)
    d = np.array(diffs)
    truth = np.array([a + b for (a, b), t in zip(pairs, toks, strict=True)
                      if tk.parse_completion(t)[0]])  # fmt: skip
    v = np.array(vals)
    return {"valid": len(v) / len(pairs), "exact": float(np.mean(d == 0)),
            "round_down": float(np.mean(v == 10 * (truth // 10))),
            "round_nearest": float(np.mean(v == 10 * np.round(truth / 10))),
            "abs_diff_lt_10": float(np.mean(np.abs(d) < 10)),
            "median_abs_diff": float(np.median(np.abs(d))),
            "top_answers": collections.Counter(v.tolist()).most_common(8)}  # fmt: skip


def main(argv: list[str]) -> int:
    cfg = cm.load_config()
    mc = provenance.load_config(MATRIX)
    dev = tk.make_splits(cfg["data"]["split_seed"])["dev"]
    out = provenance.create_run_dir(cm.REPO / "results", "E005b0-posthoc-v3", cm.REPO)
    provenance.write_metadata(out, "E005b0-posthoc-v3", MATRIX, cm.REPO,
                              extra={"posthoc": True, "runs": argv[1:]})  # fmt: skip
    res: dict[str, Any] = {"label": "POST-HOC diagnostic; not part of the matrix protocol"}
    for a in argv[1:]:
        d = Path(a).resolve()
        s = json.loads((d / "summary.json").read_text())
        net = mdl.build(mdl.GPTConfig(), seed=0)
        if mdl.load_checkpoint(d / "ckpt" / "step_01000.pt", net)["sha256"] != s["final_sha256"]:
            print("STOP: checkpoint hash mismatch")
            return 1
        es = cal.eval_samples(net, dev, mc["eval_samples_per_item"], mc["eval_seed"])
        res[f"{s['rule']}-s{s['seed']}"] = profile(es["rep"], es["tokens"])
    (out / "v3_answers.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
