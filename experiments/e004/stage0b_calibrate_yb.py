"""E004a Stage 0b, step 1 (Amendment 3 §4): YB accessibility calibration, design-only.

For each candidate cap c on the YB initial event probability S_E0, generate the calibration YB
structures (calibration seed stream; never part of the panel; 24 per YB construction x Axis-B
cell), run sampled GRPO-lite Adam (2 verifier + 2 clean seeds, T = 2700, the Stage 0 value) and
measure the failure rate. Rule: the largest c with failure in [0.30, 0.70]; if none, the c closest
to 0.5. The registered clean-gain exclusion (< 0.1) is applied as in the panel.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0_runs as sr  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e004 import panel0b as pb  # noqa: E402
from vdyn.e004 import toy  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e004" / "e004a.toml"
CAPS = [0.2, 0.1, 0.05, 0.02, 0.01]
T_CAL, N_SEEDS = 2700, 2


def _build(args: tuple[float, int]) -> list[dict[str, Any]]:
    cap, i = args
    cells = [cell for cell in pb.CELLS if cell[0].startswith("YB")]
    ss = np.random.SeedSequence(pb.ROOT_SEED).spawn(5)[4].spawn(len(cells))[i]
    construction, axis = cells[i]
    return [
        pb.build(
            f"cal-{construction}-{axis[0]}-{j:02d}",
            construction,
            axis,
            np.random.default_rng(s),
            cap,
        ).to_dict()
        for j, s in enumerate(ss.spawn(24))
    ]


def main() -> int:
    run_dir = provenance.create_run_dir(REPO / "results", "E004a-stage0b-calibration", REPO)
    provenance.write_metadata(
        run_dir,
        "E004a-stage0b-calibration",
        CONFIG,
        REPO,
        extra={"split": "design-calibration", "caps": CAPS, "T": T_CAL},
    )
    seed_parent = np.random.SeedSequence(pb.ROOT_SEED).spawn(5)[4].spawn(7)[6]
    cap_seeds = seed_parent.spawn(len(CAPS))
    out: dict[str, Any] = {"caps": {}}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for k, cap in enumerate(CAPS):
            t0 = time.perf_counter()
            parts = list(pool.map(_build, [(cap, i) for i in range(6)]))
            structs = [toy.Structure.from_dict(d) for part in parts for d in part]
            per = cap_seeds[k].spawn(len(structs))
            seeds_v = [p.spawn(2 * N_SEEDS)[:N_SEEDS] for p in per]
            seeds_c = [p.spawn(2 * N_SEEDS)[N_SEEDS:] for p in per]
            ver = sr.sampled(pool, structs, seeds_v, T_CAL, clean=False)
            cl = sr.sampled(pool, structs, seeds_c, T_CAL, clean=True)
            gv, gc = sr.outcome_grid(ver), sr.outcome_grid(cl)
            clean_mean = gc.mean(1)
            keep = (clean_mean[:, -1] - clean_mean[:, 0]) >= 0.1
            labs = sr.labels(gv, clean_mean)
            runs = [lb for i, ls in enumerate(labs) if keep[i] for lb in ls]
            by_cell: dict[str, list[bool]] = {}
            for i, st in enumerate(structs):
                if keep[i]:
                    key = f"{st.construction}-{st.meta['intended_axis'][0]}"
                    by_cell.setdefault(key, []).extend(lb["failure"] for lb in labs[i])
            out["caps"][str(cap)] = {
                "failure_rate": float(np.mean([lb["failure"] for lb in runs])),
                "n_structures": int(keep.sum()),
                "excluded": int((~keep).sum()),
                "by_cell": {k2: float(np.mean(v)) for k2, v in by_cell.items()},
                "seconds": time.perf_counter() - t0,
            }
            print(cap, json.dumps(out["caps"][str(cap)]))
    rates = {c: out["caps"][str(c)]["failure_rate"] for c in CAPS}
    ok = [c for c in CAPS if 0.30 <= rates[c] <= 0.70]
    chosen = max(ok) if ok else min(CAPS, key=lambda c: abs(rates[c] - 0.5))
    out.update(
        rule="largest cap with failure in [0.30, 0.70]; else closest to 0.5",
        original_range="S_E0 = omega * FPR_target in [~0.01, 0.4]",
        chosen_cap=chosen,
        chosen_failure_rate=rates[chosen],
        in_band=bool(ok),
    )
    (run_dir / "calibration.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"chosen cap {chosen} (failure {rates[chosen]:.3f}); run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
