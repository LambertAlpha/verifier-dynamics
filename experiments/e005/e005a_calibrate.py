"""E005a step 2/5: run the calibration Monte Carlo on one split (research/07_e005a_measurement.md
§4, §6, §8). Usage: e005a_calibrate.py design|test.

The TEST split runs only with configs/e005/E005A_TEST_APPROVED naming the frozen estimator
configuration sha256, and only once. Seeds: SeedSequence(20261101).spawn(4)[2 design | 3 test]
-> one child per panel point. Output: summary.json (per point x configuration statistics, committed)
and reps.npz (per-replication C^2 and SE, float32; kept local, sha256 recorded).
"""

import hashlib
import json
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from vdyn import provenance
from vdyn.e005 import calibrate as cb
from vdyn.e005 import panel as pl

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e005" / "e005a.toml"
PANEL_RUN = REPO / "results/E005a-panel/20260927T061706Z_7a492bb/summary.json"
ESTIMATOR_CFG = REPO / "configs" / "e005" / "estimator_frozen.json"
APPROVAL = REPO / "configs" / "e005" / "E005A_TEST_APPROVED"
WORKERS = 8


def _job(args: tuple[dict[str, Any], np.random.SeedSequence]) -> dict[str, Any]:
    point, seed = args
    return cb.run_point(point, seed)


def main(argv: list[str]) -> int:
    split = argv[1]
    assert split in ("design", "test")
    cfg = provenance.load_config(CONFIG)
    assert (tuple(cfg["budgets"]["N"]), tuple(cfg["budgets"]["m"]), cfg["budgets"]["R"]) == (
        cb.N_GRID,
        cb.M_GRID,
        cb.R_REPS,
    )
    panel_path = REPO / f"configs/e005/calibration_panel_{split}.json"
    panel_sha = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    if panel_sha != json.loads(PANEL_RUN.read_text())["sha256"][split]:
        print("STOP: calibration panel changed")
        return 1
    extra: dict[str, Any] = {"split": split, "root_seed": pl.ROOT_SEED,
                             "calibration_panel_sha256": panel_sha}  # fmt: skip
    if split == "test":
        est_sha = hashlib.sha256(ESTIMATOR_CFG.read_bytes()).hexdigest()
        appr = json.loads(APPROVAL.read_text()) if APPROVAL.exists() else {}
        if appr.get("estimator_config_sha256") != est_sha:
            print("STOP: the test split is sealed (approval file absent or not naming the "
                  "frozen estimator configuration)")  # fmt: skip
            return 1
        if list((REPO / "results").glob("E005a-calibration-test*")):
            print("STOP: the test split has already been run once")
            return 1
        extra["estimator_config_sha256"] = est_sha
    points = json.loads(panel_path.read_text())["points"]
    stream = pl.STREAMS["design_mc" if split == "design" else "test_mc"]
    seeds = np.random.SeedSequence(pl.ROOT_SEED).spawn(4)[stream].spawn(len(points))
    run_dir = provenance.create_run_dir(REPO / "results", f"E005a-calibration-{split}", REPO)
    provenance.write_metadata(run_dir, f"E005a-calibration-{split}", CONFIG, REPO, extra=extra)
    order = sorted(range(len(points)), key=lambda i: -points[i]["d_extra"])  # slow ones first
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(_job, [(points[i], seeds[i]) for i in order], chunksize=1))
    summary = {r["pid"]: r["summary"] for r in results}
    arrays: dict[str, np.ndarray] = {}
    for r in results:
        for k, v in r["reps"].items():
            arrays[f"{r['pid']}::{k}"] = v
        for k, v in r["reps_se"].items():
            arrays[f"{r['pid']}::{k}::SE"] = v
    np.savez_compressed(run_dir / "reps.npz", allow_pickle=False, **arrays)
    reps_sha = hashlib.sha256((run_dir / "reps.npz").read_bytes()).hexdigest()
    (run_dir / "summary.json").write_text(json.dumps(summary) + "\n")
    info = {"split": split, "n_points": len(points), "seconds": time.perf_counter() - t0,
            "reps_npz_sha256": reps_sha, "reps_npz_committed": False,
            "peak_rss_children": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            }  # fmt: skip
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))
    print(f"run directory: {run_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
