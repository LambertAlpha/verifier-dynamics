"""E005b-0 matrix integrity (research/10_e005b0_pilot.md §12): each V0 matrix run must reproduce
its calibration run bit-exactly (final sha256) and in every dev evaluation (overall and
per-category sampled / greedy accuracy). Exit 1 on any discrepancy (fail closed).
Usage: matrix_integrity.py <V0 run dir seed 1> <seed 2> <seed 3>."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402

MATRIX = cm.REPO / "configs" / "e005b" / "matrix.toml"


def main(argv: list[str]) -> int:
    mc = provenance.load_config(MATRIX)
    ok = True
    report = {}
    for run in argv[1:]:
        d = Path(run).resolve()
        s = json.loads((d / "summary.json").read_text())
        assert s["rule"] == "clean" and not s["smoke"]
        seed = str(s["seed"])
        cal = cm.REPO / mc["integrity"]["calibration_runs"][seed]
        cs = json.loads((cal / "summary.json").read_text())
        sha_ok = s["final_sha256"] == cs["final_sha256"] and s["final_sha256"].startswith(
            mc["integrity"]["clean_final_sha256"][seed]
        )
        ev = [json.loads(x) for x in (d / "eval_log.jsonl").read_text().splitlines()]
        ce = [json.loads(x) for x in (cal / "eval_log.jsonl").read_text().splitlines()]
        keys = ("step", "sampled", "greedy", "valid")
        same = len(ev) == len(ce) and all(
            all(a[k] == b[k] for k in keys)
            and all(a["by_cat"][c][q] == b["by_cat"][c][q] for c in b["by_cat"]
                    for q in ("sampled", "greedy", "valid"))
            for a, b in zip(ev, ce, strict=True))  # fmt: skip
        report[seed] = {
            "run": str(d.relative_to(cm.REPO)),
            "sha_ok": sha_ok,
            "eval_ok": same,
            "final_sha256": s["final_sha256"],
        }
        ok &= sha_ok and same
    print(json.dumps(report, indent=1))
    print("INTEGRITY_OK" if ok else "INTEGRITY_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
