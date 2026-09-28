"""E005b-0 calibration A2: coverage-aware base selection on DEV only among the original SFT
checkpoints (research/10_e005b0_pilot.md §10). Usage: select_base.py. Exit 3 = STOP (no base)."""

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import calib as cal  # noqa: E402
from vdyn.e005b import model as mdl  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402

CALIB = cm.REPO / "configs" / "e005b" / "calib.toml"


def main() -> int:
    cfg = cm.load_config()
    cc = provenance.load_config(CALIB)
    assert tuple(cc["categories"]["names"]) == tk.CATEGORIES
    s = cc["selection"]
    sft = cm.REPO / cc["sft_run"]
    shas = {r["step"]: r["sha256"] for r in map(json.loads, (sft / "sft_log.jsonl").open())}
    dev = tk.make_splits(cfg["data"]["split_seed"])["dev"]
    out = provenance.create_run_dir(cm.REPO / "results", "E005b0-select-base", cm.REPO)
    provenance.write_metadata(out, "E005b0-select-base", CALIB, cm.REPO,
                              extra=cm.run_extra(cfg, sft_run=cc["sft_run"]))  # fmt: skip
    nets: dict[int, Any] = {}

    def load(step: int) -> Any:
        if step not in nets:
            net = mdl.build(mdl.GPTConfig(), seed=0)
            meta = mdl.load_checkpoint(sft / "ckpt" / f"step_{step:05d}.pt", net)
            if meta["sha256"] != shas[step]:
                raise ValueError(f"checkpoint {step} hash mismatch")
            nets[step] = net
        return nets[step]

    t0 = time.perf_counter()
    sel = {st: cal.evaluate_categories(load(st), dev, s["samples_per_item"], s["seed"])
           for st in sorted(shas)}  # fmt: skip

    def confirm(step: int) -> dict[str, Any]:
        return cal.evaluate_categories(load(step), dev, s["samples_per_item"], s["confirm_seed"])

    primary = {"min_category": s["min_category"], "max_aggregate": s["max_aggregate"],
               "min_valid": s["min_valid"]}  # fmt: skip
    fallback = {**primary, "min_category": s["fallback_min_category"],
                "max_aggregate": s["fallback_max_aggregate"]}  # fmt: skip
    res = cal.select_coverage(sel, confirm, primary)
    rule = "primary"
    if res is None:
        res, rule = cal.select_coverage(sel, confirm, fallback), "fallback"
    report: dict[str, Any] = {"rule_used": rule if res else "STOP", "primary": primary,
                              "fallback": fallback, "selection_curve": sel,
                              "seconds": time.perf_counter() - t0,
                              "old_base_step": 375}  # fmt: skip
    if res is None:
        (out / "selection.json").write_text(json.dumps(report, indent=1) + "\n")
        print("STOP: no checkpoint satisfies the primary or the fallback rule")
        return 3
    step = res["step"]
    shutil.copy(sft / "ckpt" / f"step_{step:05d}.pt", out / "base_v2.pt")
    report |= {"selected_step": step, "selected_sha256": shas[step], "tried": res["tried"],
               "selection_eval": res["selection_eval"],
               "confirmation_eval": res["confirmation_eval"],
               "all_confirmations": {str(k): v for k, v in res["all_confirmations"].items()},
               "old_base_eval": sel[375]}  # fmt: skip
    (out / "selection.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("rule_used", "selected_step", "selected_sha256",
                                             "tried", "seconds")}, indent=1))  # fmt: skip
    for name in ("selection_eval", "confirmation_eval", "old_base_eval"):
        e = report[name]
        print(
            name,
            round(e["sampled"], 3),
            round(e["valid"], 3),
            {c: round(v["sampled"], 3) for c, v in e["by_cat"].items()},
        )
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
