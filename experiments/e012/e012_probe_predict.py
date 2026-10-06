"""E012: predictions from the probe runs with the frozen rule (research/paper/e012_protocol.md),
committed before any full run. rise_150 = mean batch FPR over steps 141-150 minus steps 1-10,
averaged over seeds. Usage: e012_probe_predict.py <probe run dirs>"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402

CFG = cm.REPO / "configs" / "e012" / "e012.toml"


def jl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def fpr(r: dict[str, Any]) -> float:
    return r["confusion"]["fpr"] if r["confusion"]["n_neg"] else float("nan")


def main(argv: list[str]) -> int:
    ec = provenance.load_config(CFG)
    k = ec["probe_steps"]
    rises: dict[str, list[float]] = {}
    for x in argv[1:]:
        d = Path(x).resolve()
        s = json.loads((d / "summary.json").read_text())
        assert s["probe"] and s["steps"] == k
        lg = jl(d / "grpo_log.jsonl")
        rise = float(
            np.nanmean([fpr(r) for r in lg[k - 10 : k]]) - np.nanmean([fpr(r) for r in lg[:10]])
        )
        rises.setdefault(s["arm"], []).append(rise)
    assert set(rises) == set(ec["arms"]) and all(len(v) == len(ec["seeds"]) for v in rises.values())
    out = provenance.create_run_dir(cm.REPO / "results", "E012-predictions", cm.REPO)
    provenance.write_metadata(out, "E012-predictions", CFG, cm.REPO, extra={"probe_runs": argv[1:]})
    preds = {}
    for arm, v in rises.items():
        m = float(np.mean(v))
        if abs(m - ec["rise_threshold"]) <= ec["undetermined_band"]:
            p = "undetermined"
        else:
            p = "harm>=0.25" if m >= ec["rise_threshold"] else "harm<0.25"
        preds[arm] = {"rise_150": m, "by_seed": v, "prediction": p}
        print(arm, round(m, 3), p)
    (out / "predictions.json").write_text(json.dumps(preds, indent=1) + "\n")
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
