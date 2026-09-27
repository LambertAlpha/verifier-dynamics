"""E005a theory predictions TP1-TP3 (research/07_e005a_measurement.md §2.7) on the DESIGN panel.

For every null point (C = 0) and m in {4, 8}: oracle group covariances by Monte Carlo with 10^6
groups (seed: design MC stream, child n_points, beyond the per-point calibration children), then
the leading-order null biases per 1/n (identity metric):
  plug-in tr(P_perp S_d); U-statistic -u' S_d u; S_d = S_e - alpha (S_eG + S_Ge) + alpha^2 S_G.
"""

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from vdyn import provenance
from vdyn.e005 import panel as pl
from vdyn.e005 import sampling as sm

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e005" / "e005a.toml"
PANEL = REPO / "configs/e005/calibration_panel_design.json"


def _job(args: tuple[dict[str, Any], int, np.random.SeedSequence, int]) -> dict[str, Any]:
    point, m, seed, n_groups = args
    sig = sm.oracle_sigma(point, m, np.random.default_rng(seed), n_groups=n_groups, chunk=20_000)
    ex = pl.exact(point)
    gG = ex["g_G"]
    A = float(np.linalg.norm(gG))
    u = gG / A
    alpha = float((ex["g_V"] - gG) @ gG / A**2)
    Sd = sig["e"] - alpha * (sig["eG"] + sig["eG"].T) + alpha**2 * sig["G"]
    return {"pid": point["pid"], "m": m, "alpha": alpha, "A": A,
            "plugin_n1": float(np.trace(Sd) - u @ Sd @ u), "ustat_n1": float(-(u @ Sd @ u)),
            "tr_Sd": float(np.trace(Sd)), "n_groups": int(sig["n_groups"])}  # fmt: skip


def main() -> int:
    cfg = provenance.load_config(CONFIG)
    n_groups = int(cfg["theory"]["oracle_groups"])
    panel_sha = hashlib.sha256(PANEL.read_bytes()).hexdigest()
    points = json.loads(PANEL.read_text())["points"]
    nulls = [p for p in points if p["dose"] == "null"]
    seed = np.random.SeedSequence(pl.ROOT_SEED).spawn(4)[pl.STREAMS["design_mc"]]
    base = seed.spawn(len(points) + 1)[len(points)]
    jobs = []
    kids = base.spawn(len(nulls) * 2)
    for i, p in enumerate(nulls):
        for j, m in enumerate((4, 8)):
            jobs.append((p, m, kids[2 * i + j], n_groups))
    run_dir = provenance.create_run_dir(REPO / "results", "E005a-theory", REPO)
    provenance.write_metadata(run_dir, "E005a-theory", CONFIG, REPO,
                              extra={"split": "design", "root_seed": pl.ROOT_SEED,
                                     "calibration_panel_sha256": panel_sha})  # fmt: skip
    jobs.sort(key=lambda j: -j[0]["d_extra"] * j[1])
    with ProcessPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(_job, jobs, chunksize=1))
    (run_dir / "theory.json").write_text(json.dumps(out, indent=0) + "\n")
    print(f"{len(out)} null point x m predictions; run directory: {run_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
