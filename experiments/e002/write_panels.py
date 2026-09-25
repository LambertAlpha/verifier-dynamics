"""Write the two E002 panel files (parameters + split labels only; no targets, no predictors)."""

import hashlib
import sys
from pathlib import Path

from vdyn import provenance
from vdyn.e002 import panel as pn

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    config = provenance.load_config(REPO / "configs" / "e002" / "e002.toml")
    for op, spec in config["operating_points"].items():
        path = REPO / "configs" / "e002" / f"panel_{op}.json"
        panel = pn.build_panel(op, spec["q0"], tuple(spec["f_grid"]))
        pn.write_panel(path, panel)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        redraws = sum(s["redraws"] for s in panel["slots"])
        print(
            f"{op}: {path.relative_to(REPO)} sha256={digest} "
            f"slots={len(panel['slots'])} redraws={redraws}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
