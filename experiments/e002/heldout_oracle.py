"""E002 held-out step 3 (registry E002 §9, Amendment 2 §6): oracle ceilings on a split.

Reads a `heldout_targets.py` run. Must be committed before any finite-sample evaluation of the same
split. Signals and orientation as in `oracle_ceilings.py`; the sign-flipped P4 oracle is added as a
labelled secondary row (Amendment 2 §1).
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
    root = Path(argv[2]) if len(argv) > 2 else REPO / "results"
    config = provenance.load_config(CONFIG)
    first = json.loads(next(iter(sorted(targets_dir.glob("targets_*.json")))).read_text())
    split = first["split"]
    label = f"E002-oracle-{split}"
    slots = {op: {s["sid"]: s for s in pn.load_split(REPO / "configs" / "e002" / f"panel_{op}.json",
                                                     split)}
             for op in config["operating_points"]}  # fmt: skip
    run_dir = provenance.create_run_dir(root, label, REPO)
    provenance.write_metadata(run_dir, label, CONFIG, REPO,
                              extra={"split": split, "targets": str(targets_dir)})  # fmt: skip
    out: dict[str, dict[str, dict[str, float]]] = {}
    for op in config["operating_points"]:
        data = json.loads((targets_dir / f"targets_{op}.json").read_text())
        q0, f = data["q0"], data["f"]
        rows = [r for r in data["targets"] if not r["failed"]]
        target = np.array([r["D"] for r in rows])
        stall = np.array([r["stall"] for r in rows])
        structs = [pn.match(slots[op][r["sid"]], f) for r in rows]
        c2 = np.array([(1 - q0) ** 2 * bf.kappa2(st, np.asarray(st.s0)) for st in structs])
        p4 = np.array([p4_oracle(st, np.asarray(st.s0)) for st in structs])
        signals: dict[str, np.ndarray] = {
            "G0 static FPR (constant)": np.full(len(rows), f),
            "C(0) exact Fisher": np.sqrt(c2),
            "dFPR/dt(0) [t_p -> 0]": c2 / (1 - q0),
            "dJ_V/dt(0) [t_p -> 0]": (1 - f) ** 2 * q0 * (1 - q0) + c2,
            "-dJ_G/dt(0) [t_p -> 0]": np.full(len(rows), -(1 - f) * q0 * (1 - q0)),
            "P4 oracle (r = all)": p4,
            "[post-hoc, secondary] -P4 oracle (sign flipped)": -p4,
        }
        for tp in config["targets"]["report_times"][:-1]:
            key = str(tp)
            fpr_t = np.array([r["fpr_at"][key] for r in rows])
            jg_t = np.array([r["jg_at"][key] for r in rows])
            signals[f"FPR growth t_p={tp:g}"] = fpr_t - f
            signals[f"J_V growth t_p={tp:g}"] = (jg_t + (1 - jg_t) * fpr_t) - (q0 + (1 - q0) * f)
            signals[f"-gold progress t_p={tp:g}"] = -(jg_t - q0)
        out[op] = {name: {"c_index": ep.c_index(v, target), "auroc": ep.auroc(v, stall)}
                   for name, v in signals.items()}  # fmt: skip
        print(f"\n{op} ({split} split, n = {len(rows)}, f = {f}, stall = {stall.mean():.3f})")
        for name, m in out[op].items():
            print(f"  {name:48s} C-index {m['c_index']:.4f}   AUROC {m['auroc']:.4f}")
    (run_dir / f"oracle_ceilings_{split}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
