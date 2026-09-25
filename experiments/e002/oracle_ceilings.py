"""E002 oracle ceilings (registry E002 §9), DESIGN SPLIT: noiseless predictor performance.

Danger orientation (higher = more expected shortfall), fixed by meaning: C(0), FPR growth,
J_V growth, P4 gain: +; gold progress: - (more gold progress = safer); G0: constant.
The first-order rows (t_p -> 0) use the exact time derivatives at t = 0 (theory note §12).
"""

import json
import sys
from pathlib import Path

import numpy as np

from vdyn import provenance
from vdyn.e002 import endpoints as ep
from vdyn.e002 import panel as pn
from vdyn.e002.probes import p4_oracle
from vdyn.verifiers import boolean_fp as bf

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e002" / "e002.toml"


def main(argv: list[str]) -> int:
    targets_dir = Path(argv[1]).resolve()
    config = provenance.load_config(CONFIG)
    run_dir = provenance.create_run_dir(REPO / "results", "E002-pilot-oracle", REPO)
    extra = {"split": "design", "targets": str(targets_dir.relative_to(REPO))}
    provenance.write_metadata(run_dir, "E002-pilot-oracle", CONFIG, REPO, extra=extra)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for op in config["operating_points"]:
        data = json.loads((targets_dir / f"design_targets_{op}.json").read_text())
        q0, f = data["q0"], data["f_selected"]
        slots = {
            s["sid"]: s
            for s in pn.load_split(REPO / "configs" / "e002" / f"panel_{op}.json", "design")
        }
        rows = data["targets"]
        target = np.array([r["D"] for r in rows])
        stall = np.array([r["stall"] for r in rows])
        structs = [pn.match(slots[r["sid"]], f) for r in rows]
        c2 = np.array([(1 - q0) ** 2 * bf.kappa2(st, np.asarray(st.s0)) for st in structs])
        signals: dict[str, np.ndarray] = {
            "G0 static FPR (constant)": np.full(len(rows), f),
            "C(0) exact Fisher": np.sqrt(c2),
            "dFPR/dt(0) [t_p -> 0]": c2 / (1 - q0),
            "dJ_V/dt(0) [t_p -> 0]": (1 - f) ** 2 * q0 * (1 - q0) + c2,
            "-dJ_G/dt(0) [t_p -> 0]": np.full(len(rows), -(1 - f) * q0 * (1 - q0)),
            "P4 oracle (r = all)": np.array([p4_oracle(st, np.asarray(st.s0)) for st in structs]),
        }
        for tp in config["targets"]["report_times"][:-1]:  # oracle probe horizons
            key = str(tp)
            fpr_t = np.array([r["fpr_at"][key] for r in rows])
            jg_t = np.array([r["jg_at"][key] for r in rows])
            signals[f"FPR growth t_p={tp:g}"] = fpr_t - f
            signals[f"J_V growth t_p={tp:g}"] = (jg_t + (1 - jg_t) * fpr_t) - (q0 + (1 - q0) * f)
            signals[f"-gold progress t_p={tp:g}"] = -(jg_t - q0)
        out[op] = {
            name: {"c_index": ep.c_index(v, target), "auroc": ep.auroc(v, stall)}
            for name, v in signals.items()
        }
        print(
            f"\n{op} (design split, n = {len(rows)}, f = {f}, stall fraction = {stall.mean():.3f})"
        )
        for name, m in out[op].items():
            print(f"  {name:32s} C-index {m['c_index']:.4f}   AUROC {m['auroc']:.4f}")
    (run_dir / "oracle_ceilings_design.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"run directory: {run_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
