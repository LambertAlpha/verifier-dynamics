"""E005a step 1: build and FREEZE the design and test calibration panels (oracle geometry only; no
estimator is run). research/07_e005a_measurement.md §3. The test panel uses the design C_ref."""

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from vdyn import provenance
from vdyn.e005 import panel as pl

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e005" / "e005a.toml"
OUT = {s: REPO / f"configs/e005/calibration_panel_{s}.json" for s in ("design", "test")}


def main() -> int:
    cfg = provenance.load_config(CONFIG)
    assert cfg["root_seed"] == pl.ROOT_SEED and cfg["panel"]["d_extra"] == list(pl.D_EXTRA)
    assert cfg["panel"]["variants"] == list(pl.VARIANTS) and cfg["panel"]["doses"] == pl.DOSES
    for p in OUT.values():
        if p.exists():
            print(f"STOP: {p.name} exists; panels are frozen once")
            return 1
    run_dir = provenance.create_run_dir(REPO / "results", "E005a-panel", REPO)
    t0 = time.perf_counter()
    design = pl.build_panel("design")
    test = pl.build_panel("test", c_ref=design["c_ref"])
    shas = {}
    for split, doc in (("design", design), ("test", test)):
        body = json.dumps(doc, indent=0) + "\n"
        (run_dir / OUT[split].name).write_text(body)
        shas[split] = hashlib.sha256(body.encode()).hexdigest()
    provenance.write_metadata(run_dir, "E005a-panel", CONFIG, REPO,
                              extra={"split": "design+test (oracle only)",
                                     "root_seed": pl.ROOT_SEED,
                                     "calibration_panel_sha256": shas})  # fmt: skip
    for split in OUT:
        OUT[split].write_bytes((run_dir / OUT[split].name).read_bytes())
    summary: dict[str, Any] = {s: {"n_points": len(d["points"]), "c_ref": d["c_ref"],
                   "dropped_unattainable": len(d["dropped_unattainable"]),
                   "infeasible_bases": d["infeasible_bases"]}
               for s, d in (("design", design), ("test", test))}  # fmt: skip
    summary["sha256"] = shas
    summary["seconds"] = time.perf_counter() - t0
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"run directory: {run_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
