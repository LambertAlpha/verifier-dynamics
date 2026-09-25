"""E002 verdict (registry E002 §7-§8, operationalized by Amendment 2 §2); pure functions.

`point[(cell, arm, endpoint)]` are test-split point estimates (mean over replications);
`holm_reject[(cell, endpoint)]` are Holm decisions for G1 - max_j P_j at the primary cells;
`dominated[cell]` is the frontier rule of §7 for G1 (C-index, nominal cells).
"""

from collections.abc import Sequence
from typing import Any

from vdyn.e002.arms import COMPETITORS

Cell = tuple[int, int]
ABANDON_A_ARMS = ("P1", "P2", "P4")


def dominated_cells(
    cells: Sequence[Cell], c_index: dict[tuple[Cell, str], float],
    competitors: Sequence[str] = COMPETITORS,
) -> dict[Cell, bool]:  # fmt: skip
    """G1 at cell X is dominated if a competitor at a cell with B_gold and B_roll no larger
    than X's attains at least G1's C-index at X."""
    out = {}
    for x in cells:
        g1 = c_index[(x, "G1")]
        out[x] = any(
            c_index[(y, a)] >= g1
            for y in cells
            if y[0] <= x[0] and y[1] <= x[1]
            for a in competitors
            if (y, a) in c_index
        )
    return out


def _delta(point: dict[Any, float], cell: Cell, arm: str, endpoint: str) -> float:
    rivals = [point[(cell, a, endpoint)] for a in COMPETITORS if (cell, a, endpoint) in point]
    return point[(cell, arm, endpoint)] - max(rivals)


def verdict(
    point: dict[tuple[Cell, str, str], float],
    holm_reject: dict[tuple[Cell, str], bool],
    dominated: dict[Cell, bool],
    primary: Sequence[Cell],
    endpoints: Sequence[str],
) -> dict[str, Any]:
    pairs = [(c, e) for c in primary for e in endpoints]
    delta_g1 = {p: _delta(point, p[0], "G1", p[1]) for p in pairs}
    delta_oracle = {p: _delta(point, p[0], "G1-oracle", p[1]) for p in pairs}
    for p, rejected in holm_reject.items():
        assert not rejected or delta_g1[p] > 0, f"Holm rejection with a non-positive Delta at {p}"
    uniformly = all(dominated.values())
    success = any(holm_reject.get(p, False) for p in pairs) and not uniformly
    beats = {
        j: any(point[(c, "G1", e)] > point[(c, j, e)] for c, e in pairs if (c, j, e) in point)
        for j in ABANDON_A_ARMS
    }
    abandon = {
        "a": not any(beats.values()),
        "b": any(d > 0 for d in delta_oracle.values())
        and not any(d > 0 for d in delta_g1.values()),
        "c": uniformly,
    }
    assert not (success and any(abandon.values())), "SUCCESS and ABANDON cannot both hold"
    if success:
        category = "SUCCESS"
    elif any(abandon.values()):
        category = "ABANDON"
    else:
        category = "NO PRACTICAL ADVANTAGE"
    return {
        "category": category,
        "success": success,
        "abandon": abandon,
        "g1_beats": beats,
        "uniformly_dominated": uniformly,
        "delta_g1": {f"{c[0]},{c[1]}|{e}": d for (c, e), d in delta_g1.items()},
        "delta_g1_oracle": {f"{c[0]},{c[1]}|{e}": d for (c, e), d in delta_oracle.items()},
    }
