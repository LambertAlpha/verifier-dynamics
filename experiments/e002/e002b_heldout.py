"""E002b held-out step 4 (Amendment 2 §6): frozen configurations on a split, R = reps_heldout.

Reads a `heldout_targets.py` run (for the split and the FPR). `--split` is taken from that run;
the test split loads only after explicit approval. The frozen configurations are checked against
the SHA-256 registered in Amendment 2 before anything runs. Per-structure scores are written to
`scores.npz` (raw arrays stay local by repository policy); `e002b_analysis.py` computes endpoints.
"""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from vdyn import provenance
from vdyn.e002 import arms
from vdyn.e002 import panel as pn

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e002" / "e002.toml"
FROZEN = REPO / "configs" / "e002" / "e002b_tuned_configs.json"
FROZEN_SHA256 = "674c882340328d0b497af9464bb1e5d973a709b316b9276694dbdefae500d880"


def structure_task(args: tuple[Any, ...]) -> dict[str, Any]:
    op_i, s_idx, slot, f, q0, cells, reps, seed, frozen = args
    st = pn.match(slot, f)
    scores = np.full((len(cells), len(arms.ARMS), reps), np.nan)
    start = time.perf_counter()
    for c_i, (b_gold, b_roll) in enumerate(cells):
        for a_i, arm in enumerate(arms.ARMS):
            key = f"{b_gold},{b_roll}|{arm}"
            if key not in frozen:
                continue
            out = arms.run_arm(arm, frozen[key]["cfg"], st, q0, b_gold, b_roll, reps,
                               (seed, op_i, s_idx, c_i))  # fmt: skip
            if out is not None:
                scores[c_i, a_i] = out
    return {"sid": slot["sid"], "scores": scores, "seconds": time.perf_counter() - start}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets_dir")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--results-root", default=str(REPO / "results"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv[1:])
    if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != FROZEN_SHA256:
        print("STOP: frozen configuration file does not match the registered SHA-256")
        return 1
    config = provenance.load_config(CONFIG)
    reps = args.reps or config["e002b"]["reps_heldout"]
    seed = config["e002b"]["heldout_seed"]
    targets_dir = Path(args.targets_dir).resolve()
    t_data = {op: json.loads((targets_dir / f"targets_{op}.json").read_text())
              for op in config["operating_points"]}  # fmt: skip
    split = next(iter(t_data.values()))["split"]
    slots = {op: pn.load_split(REPO / "configs" / "e002" / f"panel_{op}.json", split)
             for op in config["operating_points"]}  # fmt: skip
    frozen = json.loads(FROZEN.read_text())["configs"]
    cells = [(bg, bg * r) for bg in config["budgets"]["b_gold"]
             for r in config["budgets"]["roll_ratio"]]  # fmt: skip
    label = f"E002b-{split}"
    run_dir = provenance.create_run_dir(Path(args.results_root), label, REPO)
    provenance.write_metadata(run_dir, label, CONFIG, REPO, extra={
        "split": split, "reps": reps, "seed": seed, "targets": str(targets_dir),
        "frozen_sha256": FROZEN_SHA256})  # fmt: skip
    meta: dict[str, Any] = {"split": split, "reps": reps, "cells": cells, "arms": list(arms.ARMS)}
    arrays: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for op_i, op in enumerate(config["operating_points"]):
            q0, f = t_data[op]["q0"], t_data[op]["f"]
            tasks = [(op_i, i, s, f, q0, cells, reps, seed, frozen[op])
                     for i, s in enumerate(slots[op])]  # fmt: skip
            start = time.perf_counter()
            results = list(pool.map(structure_task, tasks, chunksize=1))
            arrays[op] = np.stack([r["scores"] for r in results], axis=-1)  # (cell, arm, rep, s)
            meta[op] = {"sids": [r["sid"] for r in results], "q0": q0, "f": f,
                        "wall_seconds": time.perf_counter() - start,
                        "cpu_seconds": float(sum(r["seconds"] for r in results))}  # fmt: skip
            print(f"{op}: {len(results)} structures, wall {meta[op]['wall_seconds']:.0f} s")
    np.savez_compressed(run_dir / "scores.npz", allow_pickle=False, **arrays)
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
