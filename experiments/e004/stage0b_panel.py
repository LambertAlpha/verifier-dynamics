"""E004a Stage 0b, step 2 (Amendment 3 §8): generate and freeze the fresh DESIGN panel.

768 structures (24 construction x Axis-B cells x 32), seed 20261001, YB cap 0.2 (calibration
record). Test and shift panels are not generated.
"""

import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vdyn.e004 import panel0b as pb

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "configs" / "e004" / "design_panel_0b.json"
YB_CAP = 0.2


def _one(args: tuple[int, int]) -> dict:
    i, j = args
    construction, axis = pb.CELLS[i]
    ss = np.random.SeedSequence(pb.ROOT_SEED).spawn(5)[0].spawn(len(pb.CELLS))[i]
    s = ss.spawn(pb.N_PER_CELL)[j]
    sid = f"{construction}-{axis[0]}-{j:02d}"
    return pb.build(sid, construction, axis, np.random.default_rng(s), YB_CAP).to_dict()


def main() -> int:
    if OUT.exists():
        print(f"STOP: {OUT} exists (frozen)")
        return 1
    start = time.perf_counter()
    jobs = [(i, j) for i in range(len(pb.CELLS)) for j in range(pb.N_PER_CELL)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        structs = list(pool.map(_one, jobs, chunksize=4))
    ref = pb.design_panel(n_per=1, yb_cap=YB_CAP)  # same seeds as the first structure per cell
    for r in ref:
        assert next(s for s in structs if s["sid"] == r.sid) == r.to_dict(), r.sid
    OUT.write_text(
        json.dumps(
            {"root_seed": pb.ROOT_SEED, "split": "design", "yb_cap": YB_CAP, "structures": structs},
            indent=1,
        )
        + "\n"
    )
    noise_inv = sum(s["meta"]["noise_inverted"] for s in structs)
    rej = {"channel": 0, "targets": 0, "axis": 0}
    for s in structs:
        for k in rej:
            rej[k] += s["meta"]["rejections"][k]
    print(
        f"{len(structs)} structures, {time.perf_counter() - start:.0f} s; noise-induced "
        f"inversions {noise_inv}; rejections {rej}; "
        f"sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
