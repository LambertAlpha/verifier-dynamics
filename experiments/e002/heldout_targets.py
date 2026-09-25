"""E002 held-out step 2 (Amendment 2 §6): exact targets on a split at the DESIGN-selected FPR.

`--split test` works only after explicit approval (`configs/e002/HELDOUT_APPROVED`); the loader
refuses otherwise. `--split design` is the dry run. The FPR is read from the design-split targets
run and is never re-selected. Exclusions (target integration failures) are counted; the run stops
if more than 1% fail (registry E002 §4).
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from vdyn import provenance
from vdyn.e002 import panel as pn
from vdyn.geometry.gold_race import gold_race

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e002" / "e002.toml"


def _target(args: tuple[dict[str, Any], float, float, tuple[float, ...]]) -> dict[str, Any]:
    slot, f, q0, report = args
    st = pn.match(slot, f)
    try:
        r = gold_race(st, q0, report_times=report)
    except Exception as exc:  # counted as an exclusion, never silently dropped
        return {"sid": slot["sid"], "type": slot["type"], "failed": repr(exc)}
    return {
        "sid": slot["sid"],
        "type": slot["type"],
        "failed": None,
        "stall": r.stall,
        "D": 1 - min(r.q_inf, 1.0),
        "q_inf": r.q_inf,
        "fpr_inf": r.fpr_inf,
        "t95": r.t95,
        "jg_at": {str(k): v for k, v in r.jg_at.items()},
        "fpr_at": {str(k): v for k, v in r.fpr_at.items()},
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("design_targets_dir")
    ap.add_argument("--split", choices=("design", "test"), default="design")
    ap.add_argument("--results-root", default=str(REPO / "results"))
    args = ap.parse_args(argv[1:])
    config = provenance.load_config(CONFIG)
    design_dir = Path(args.design_targets_dir).resolve()
    label = f"E002-targets-{args.split}"
    slots_by_op = {
        op: pn.load_split(REPO / "configs" / "e002" / f"panel_{op}.json", args.split)
        for op in config["operating_points"]
    }  # the loader refuses the test split without approval, before anything is written
    run_dir = provenance.create_run_dir(Path(args.results_root), label, REPO)
    provenance.write_metadata(run_dir, label, CONFIG, REPO, extra={"split": args.split,
                              "design_targets": str(design_dir)})  # fmt: skip
    report = tuple(config["targets"]["report_times"])
    summary: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for op, spec in config["operating_points"].items():
            f = json.loads((design_dir / f"design_targets_{op}.json").read_text())["f_selected"]
            slots = slots_by_op[op]
            rows = list(pool.map(_target, [(s, f, spec["q0"], report) for s in slots], chunksize=4))
            failed = [r for r in rows if r["failed"]]
            stall = sum(bool(r.get("stall")) for r in rows) / len(rows)
            summary[op] = {"split": args.split, "f": f, "n": len(rows), "failed": len(failed),
                           "stall_fraction": stall}  # fmt: skip
            (run_dir / f"targets_{op}.json").write_text(json.dumps(
                {"operating_point": op, "split": args.split, "q0": spec["q0"], "f": f,
                 "targets": rows}, indent=1))  # fmt: skip
            if len(failed) > 0.01 * len(rows):
                print(f"STOP: {len(failed)} target failures in {op} (> 1%)")
                return 1
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
