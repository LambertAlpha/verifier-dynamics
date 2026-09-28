"""E005b-0 matrix: ONE evaluation of the 12 final checkpoints on the existing test split, run only
after the matrix analysis is frozen and the dev analysis is complete (research/10_e005b0_pilot.md
§12). DISCLOSURE: this test split was inspected during the pilot; it is not a newly sealed
confirmatory test.
Usage: matrix_test_eval.py <run-dir> ... (12 runs, final checkpoints in ckpt/)."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import matrix as mx  # noqa: E402
from vdyn.e005b import model as mdl  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402
from vdyn.e005b import verifiers as vf  # noqa: E402

MATRIX = cm.REPO / "configs" / "e005b" / "matrix.toml"
TEST_SEED = 20261330
RULES = ("clean", "flip", "deleted", "exploit")


def main(argv: list[str]) -> int:
    cfg = cm.load_config()
    mc = provenance.load_config(MATRIX)
    v = cfg["verifiers"]
    deleted = vf.deleted_prompts(v["deleted_fraction"], v["deleted_seed"])
    test = tk.make_splits(cfg["data"]["split_seed"])["test"]
    out = provenance.create_run_dir(cm.REPO / "results", "E005b0-matrix-test", cm.REPO)
    provenance.write_metadata(
        out,
        "E005b0-matrix-test",
        MATRIX,
        cm.REPO,
        extra={
            "disclosure": "test split previously inspected in the pilot",
            "test_seed": TEST_SEED,
            "samples_per_item": 4,
        },
    )
    res: dict[str, Any] = {
        "disclosure": "test split previously inspected during the pilot; not "
        "a newly sealed confirmatory test",
        "runs": {},
    }
    prim: dict[tuple[str, int], float] = {}
    for a in argv[1:]:
        d = Path(a).resolve()
        s = json.loads((d / "summary.json").read_text())
        net = mdl.build(mdl.GPTConfig(), seed=0)
        meta = mdl.load_checkpoint(d / "ckpt" / f"step_{mc['steps']:05d}.pt", net)
        if meta["sha256"] != s["final_sha256"]:
            print(f"STOP: final checkpoint hash mismatch in {d}")
            return 1
        ev = mx.evaluate_matrix(net, test, mc["eval_samples_per_item"], TEST_SEED, deleted)
        res["runs"][f"{s['rule']}-s{s['seed']}"] = ev
        prim[(s["rule"], s["seed"])] = ev["sampled"]
    assert set(prim) == {(r, s) for r in RULES for s in (1, 2, 3)}
    res["paired_sampled"] = {}
    for rule in RULES[1:]:
        dd = np.array([prim[(rule, s)] - prim[("clean", s)] for s in (1, 2, 3)])
        res["paired_sampled"][rule] = {
            "by_seed": dd.tolist(),
            "mean": float(dd.mean()),
            "min": float(dd.min()),
            "max": float(dd.max()),
        }
    (out / "test_eval.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["paired_sampled"], indent=1))
    for k, e in sorted(res["runs"].items()):
        print(
            k,
            round(e["sampled"], 3),
            round(e["greedy"], 3),
            {c: round(x["sampled"], 3) for c, x in e["by_cat"].items()},
            "fp_ends0",
            round(e["suffix"]["fp_ends0"], 3),
        )
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
