"""[POST-HOC, NOT PRE-REGISTERED] E002b held-out: fragility of the frontier rule (c).

The registered verdict (P-mod ABANDON by rule c) turned on one cell, (16, 64), where the best
cheaper-or-equal competitor matched G1 to 1e-4 in C-index. This script estimates, by the same
hierarchical bootstrap (structures, then replications), the probability that G1 at each cell is
dominated: P*(max over cheaper-or-equal competitor cells of C-index >= G1's C-index). It does not
change the registered verdict.
"""

import json
import sys
from pathlib import Path

import numpy as np

from vdyn import provenance
from vdyn.e002 import arms
from vdyn.e002 import endpoints as ep

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e002" / "e002.toml"


def main(argv: list[str]) -> int:
    run, targets_dir = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    n_boot = int(argv[3]) if len(argv) > 3 else 2000
    config = provenance.load_config(CONFIG)
    meta = json.loads((run / "run.json").read_text())
    scores = np.load(run / "scores.npz")
    out_dir = provenance.create_run_dir(REPO / "results", "E002b-posthoc-test", REPO)
    provenance.write_metadata(out_dir, "E002b-posthoc-test", CONFIG, REPO,
                              extra={"split": meta["split"], "post_hoc": True})  # fmt: skip
    cells = [tuple(c) for c in meta["cells"]]
    a_idx = {a: i for i, a in enumerate(meta["arms"])}
    report: dict[str, dict[str, dict[str, float]]] = {}
    for op_i, op in enumerate(config["operating_points"]):
        tdata = json.loads((targets_dir / f"targets_{op}.json").read_text())
        assert [r["sid"] for r in tdata["targets"]] == meta[op]["sids"]
        assert not any(r["failed"] for r in tdata["targets"])
        target = np.array([r["D"] for r in tdata["targets"]])
        s = scores[op]
        report[op] = {}
        for c_i, cell in enumerate(cells):
            g1 = s[c_i, a_idx["G1"]]
            cheaper = [y for y in cells if y[0] <= cell[0] and y[1] <= cell[1]]
            comps = [s[cells.index(y), a_idx[a]] for y in cheaper for a in arms.COMPETITORS
                     if not np.all(np.isnan(s[cells.index(y), a_idx[a]]))]  # fmt: skip
            rng = np.random.default_rng([config["e002b"]["heldout_seed"], 30, op_i, c_i])
            diff = ep.paired_difference_bootstrap(ep.c_index, g1, comps, target, n_boot, rng)
            point = ep.panel_metric(ep.c_index, g1, target) - max(
                ep.panel_metric(ep.c_index, c, target) for c in comps
            )
            report[op][f"{cell[0]},{cell[1]}"] = {
                "point_margin": point,
                "ci95": np.quantile(diff, [0.025, 0.975]).tolist(),
                "p_dominated": float(np.mean(diff <= 0)),
            }
        free = {k: v for k, v in report[op].items() if v["p_dominated"] < 0.95}
        print(f"{op}: cells with P*(dominated) < 0.95: {json.dumps(free)}")
    (out_dir / "posthoc_frontier.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"run directory: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
