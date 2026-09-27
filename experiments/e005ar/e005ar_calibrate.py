"""E005a-R step 2/5: run the calibration Monte Carlo on one split (research/09_e005ar_design.md
§5-§7, §10). Usage: e005ar_calibrate.py design|test [workers].

The TEST split runs only with configs/e005ar/E005AR_TEST_APPROVED naming the sha256 of
configs/e005ar/e005ar_estimator_frozen.json, and only once. Seeds:
SeedSequence(20261201).spawn(6)[3 design | 4 test] -> one child per panel point. The R4
sensitivity variants run on the design split only. Output: summary.json (per point x
configuration statistics; the gzip copy is committed) and reps.npz (per-replication C^2 and
sign-flip decisions; kept local, sha256 recorded).
"""

import hashlib
import json
import resource
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e005ar_panel as pn  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005ar import calibrate as cb  # noqa: E402
from vdyn.e005ar import env  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e005ar" / "e005ar.toml"
ESTIMATOR_CFG = REPO / "configs" / "e005ar" / "e005ar_estimator_frozen.json"
APPROVAL = REPO / "configs" / "e005ar" / "E005AR_TEST_APPROVED"


def _job(args: tuple[dict[str, Any], dict[str, Any], np.random.SeedSequence, bool]) -> dict:
    point, base, seed, sens = args
    return cb.run_point(point, base, seed, sensitivity=sens)


def main(argv: list[str]) -> int:
    split = argv[1]
    assert split in ("design", "test")
    workers = int(argv[2]) if len(argv) > 2 else 8
    cfg = provenance.load_config(CONFIG)
    pn.check_config(cfg)
    panel_path = pn.OUT[split]
    runs = sorted((REPO / "results").glob("E005aR-panel/*/summary.json"))
    if len(runs) != 1:
        print("STOP: expected exactly one frozen panel run")
        return 1
    panel_sha = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    if panel_sha != json.loads(runs[0].read_text())["sha256"][split]:
        print("STOP: calibration panel changed")
        return 1
    extra: dict[str, Any] = {"split": split, "root_seed": env.ROOT_SEED,
                             "calibration_panel_sha256": panel_sha,
                             "host": socket.gethostname(), "workers": workers}  # fmt: skip
    if split == "test":
        est_sha = hashlib.sha256(ESTIMATOR_CFG.read_bytes()).hexdigest()
        appr = json.loads(APPROVAL.read_text()) if APPROVAL.exists() else {}
        if appr.get("estimator_config_sha256") != est_sha:
            print("STOP: the test split is sealed (approval file absent or not naming the "
                  "frozen estimator configuration)")  # fmt: skip
            return 1
        if list((REPO / "results").glob("E005aR-calibration-test*")):
            print("STOP: the test split has already been run once")
            return 1
        extra["estimator_config_sha256"] = est_sha
    panel = json.loads(panel_path.read_text())
    points, bases = panel["points"], panel["bases"]
    stream = env.STREAMS["design_mc" if split == "design" else "test_mc"]
    seeds = np.random.SeedSequence(env.ROOT_SEED).spawn(6)[stream].spawn(len(points))
    run_dir = provenance.create_run_dir(REPO / "results", f"E005aR-calibration-{split}", REPO)
    provenance.write_metadata(run_dir, f"E005aR-calibration-{split}", CONFIG, REPO, extra=extra)
    sens = split == "design"
    order = sorted(range(len(points)), key=lambda i: -points[i]["d"])  # slow ones first
    jobs = [(points[i], bases[points[i]["sid"]], seeds[i], sens) for i in order]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_job, jobs, chunksize=1))
    summary = {r["pid"]: r["summary"] for r in results}
    arrays: dict[str, np.ndarray] = {}
    for r in results:
        for k, v in r["reps"].items():
            arrays[f"{r['pid']}::{k}"] = v
        for k, v in r["reps_reject"].items():
            arrays[f"{r['pid']}::{k}::reject"] = v
    np.savez_compressed(run_dir / "reps.npz", allow_pickle=False, **arrays)
    reps_sha = hashlib.sha256((run_dir / "reps.npz").read_bytes()).hexdigest()
    (run_dir / "summary.json").write_text(json.dumps(summary) + "\n")
    info = {
        "split": split,
        "n_points": len(points),
        "seconds": time.perf_counter() - t0,
        "reps_npz_sha256": reps_sha,
        "reps_npz_committed": False,
        "peak_rss_children": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))
    print(f"run directory: {run_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
