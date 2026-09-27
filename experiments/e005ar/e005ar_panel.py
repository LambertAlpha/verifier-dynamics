"""E005a-R step 1: build and FREEZE the design and test calibration panels (oracle geometry and
behavior Monte Carlo only; no estimator is run). research/09_e005ar_design.md §4."""

import hashlib
import json
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from vdyn import provenance
from vdyn.e005ar import calibrate as cb
from vdyn.e005ar import env

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e005ar" / "e005ar.toml"
OUT = {s: REPO / f"configs/e005ar/panel_{s}.json" for s in ("design", "test")}


def check_config(cfg: dict[str, Any]) -> None:
    e = cfg["environment"]
    assert cfg["root_seed"] == env.ROOT_SEED
    assert cfg["anchors"]["tau"] == env.TAU_ANCHORS
    assert tuple(e["r"]) == env.R_GRID and tuple(e["d"]) == env.D_GRID
    assert {k: tuple(v) for k, v in e["base_types"].items()} == env.BASE_TYPES
    assert tuple(e["spectra"]) == env.SPECTRA and tuple(e["constructions"]) == env.CONSTRUCTIONS
    assert e["spectrum_levels"] == {"bulk": 0.25, "flat": 1.0, "spiked_mean": 1.0,
                                    "spiked_power": env.SPIKED_POWER}  # fmt: skip
    assert tuple(e["alpha_clean_null"]) == env.ALPHA_CLEAN
    assert tuple(e["alpha_dose"]) == env.ALPHA_DOSE
    assert tuple(e["alpha_nuisance_only"]) == env.ALPHA_NUIS
    assert tuple(e["alpha_partial"]) == env.ALPHA_PARTIAL
    assert e["bisection_iters"] == env.BISECTION_ITERS and e["incorrect_dirichlet"] == 2.0
    b = cfg["budgets"]
    assert tuple(b["N"]) == cb.N_GRID and tuple(b["k"]) == cb.K_GRID
    assert b["m"] == cb.M and b["R"] == cb.R_REPS


def _build(split: str) -> dict[str, Any]:
    cfg = provenance.load_config(CONFIG)
    return env.build_panel(split, n_groups=cfg["environment"]["dose_mc_groups"])


def main() -> int:
    cfg = provenance.load_config(CONFIG)
    check_config(cfg)
    for p in OUT.values():
        if p.exists():
            print(f"STOP: {p.name} exists; panels are frozen once")
            return 1
    run_dir = provenance.create_run_dir(REPO / "results", "E005aR-panel", REPO)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        docs = dict(zip(OUT, pool.map(_build, list(OUT)), strict=True))
    shas, counts = {}, {}
    for split, doc in docs.items():
        body = json.dumps(doc, indent=0) + "\n"
        (run_dir / OUT[split].name).write_text(body)
        shas[split] = hashlib.sha256(body.encode()).hexdigest()
        counts[split] = {"points": len(doc["points"]), "dropped": doc["dropped"]}
    provenance.write_metadata(run_dir, "E005aR-panel", CONFIG, REPO,
                              extra={"split": "design+test (oracle only)",
                                     "root_seed": env.ROOT_SEED, "host": socket.gethostname(),
                                     "calibration_panel_sha256": shas})  # fmt: skip
    for split in OUT:
        OUT[split].write_text((run_dir / OUT[split].name).read_text())
    summary = {"sha256": shas, "counts": counts, "seconds": time.perf_counter() - t0}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"run directory: {run_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
