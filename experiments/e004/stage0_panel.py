"""E004a Stage 0, step 1 (registry E004a §2-§4): generate and freeze the DESIGN panel only.

Test and shift panels are not generated (their seed streams stay reserved).
"""

import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vdyn.e004 import panel as pn

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "configs" / "e004" / "design_panel.json"


def _one(args: tuple[int, str, int]) -> dict:
    c_i, construction, j = args
    design_ss = np.random.SeedSequence(pn.ROOT_SEED).spawn(4)[0]
    ss = design_ss.spawn(len(pn.CONSTRUCTIONS))[c_i].spawn(pn.N_DESIGN)[j]
    return pn.build(f"{construction}-{j:02d}", construction, np.random.default_rng(ss)).to_dict()


def main() -> int:
    if OUT.exists():
        print(f"STOP: {OUT} exists (the design panel is frozen)")
        return 1
    start = time.perf_counter()
    jobs = [(c_i, c, j) for c_i, c in enumerate(pn.CONSTRUCTIONS) for j in range(pn.N_DESIGN)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        structs = list(pool.map(_one, jobs, chunksize=4))
    # the per-structure seeds equal those of pn.design_panel(); spot-check a few
    ref = pn.design_panel(n_per=2)
    for r in ref:
        s = next(x for x in structs if x["sid"] == r.sid)
        assert s == r.to_dict(), r.sid
    OUT.write_text(json.dumps({"root_seed": pn.ROOT_SEED, "split": "design",
                               "structures": structs}, indent=1) + "\n")  # fmt: skip
    rej = {c: [s["meta"]["rejections"] for s in structs if s["construction"] == c]
           for c in pn.CONSTRUCTIONS}  # fmt: skip
    summary = {
        c: {"channel_rejections": int(sum(r["channel"] for r in v)),
            "target_redraws": int(sum(r["targets"] for r in v))}
        for c, v in rej.items()
    }  # fmt: skip
    print(json.dumps(summary, indent=1))
    print(
        f"{len(structs)} structures, {time.perf_counter() - start:.0f} s, "
        f"sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
