"""E002 synthetic verifier panel (registry E002 §2).

Raw draws are shared across the FPR grid of an operating point: a draw is kept only if, for EVERY
candidate f, the common logit shift that matches FPR = f keeps all feature probabilities in
[1e-4, 1 - 1e-4] (this realizes "the same raw draws are used for every f"). The panel file
stores parameters and split labels only; structures are materialized per f with `match`.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from vdyn.verifiers.boolean_fp import (
    FEATURE_TYPES,
    MIX_BASES,
    N_FEATURES,
    EventStructure,
    event_prob,
)

ROOT_SEED = 20260925
OPERATING_POINTS = ("P-mod", "P-rare")
# Amendment 1: SINGLE is parameter-free after matching -> identical-control class like RFP.
CONTROL_TYPES = ("SINGLE", "RFP")
N_CONTROL = 10
COUNTS = {"AND2": 98, "AND3": 98, "AND4": 98, "OR2": 98, "OR3": 98, "THR23": 98, "AOR": 98,
          "OAND": 97, "MIX": 97}  # fmt: skip
N_DESIGN, N_DESIGN_CONTROL = 300, 4
LOGIT_SD = 2.0
RHO_RANGE = (0.05, 1.0)
S_BOUNDS = (1e-4, 1 - 1e-4)
DEDUP_TOL = 1e-3
DEFAULT_APPROVAL = Path(__file__).resolve().parents[3] / "configs" / "e002" / "HELDOUT_APPROVED"


def _event_target(slot: dict[str, Any], f: float) -> tuple[float, float]:
    """(target S_E, coin) for a slot at FPR f."""
    if slot["type"] == "MIX":
        rho = slot["rho"]
        return 1 - (1 - f) ** rho, 1 - (1 - f) ** (1 - rho)
    return f, 0.0


def _shift(slot: dict[str, Any], f: float) -> float:
    raw = np.asarray(slot["raw_logits"], dtype=np.float64)
    st = EventStructure("tmp", slot["type"], tuple(expit(raw)), base=slot["base"])
    target, _ = _event_target(slot, f)

    def gap(c: float) -> float:
        probe = EventStructure("tmp", st.type, tuple(expit(raw + c)), base=st.base)
        return event_prob(probe, np.asarray(probe.s0)) - target

    return float(brentq(gap, -40.0, 40.0, xtol=1e-14, rtol=1e-15))


def match(slot: dict[str, Any], f: float) -> EventStructure:
    """The structure of a slot at FPR f: common logit shift for the event, coin for MIX/RFP."""
    if slot["type"] == "RFP":
        return EventStructure(slot["sid"], "RFP", (), coin=f)
    c = _shift(slot, f)
    _, coin = _event_target(slot, f)
    s0 = tuple(float(x) for x in expit(np.asarray(slot["raw_logits"]) + c))
    return EventStructure(
        slot["sid"], slot["type"], s0, coin=coin, base=slot["base"], rho=slot["rho"]
    )


def canonical(slot: dict[str, Any], f: float) -> np.ndarray:
    """Matched logits with symmetric feature groups sorted (plus rho for MIX)."""
    raw = np.asarray(slot["raw_logits"], dtype=np.float64)
    logits = raw + _shift(slot, f)
    event_type = slot["base"] if slot["type"] == "MIX" else slot["type"]
    if event_type in ("AOR", "OAND"):
        logits = np.concatenate([np.sort(logits[:2]), logits[2:]])
    elif event_type != "SINGLE":
        logits = np.sort(logits)
    rho = [slot["rho"]] if slot["type"] == "MIX" else []
    return np.concatenate([logits, rho])


def _valid_for_grid(slot: dict[str, Any], f_grid: tuple[float, ...]) -> bool:
    for f in f_grid:
        try:
            s = np.asarray(match(slot, f).s0)
        except ValueError:  # brentq could not bracket
            return False
        if np.any(s < S_BOUNDS[0]) or np.any(s > S_BOUNDS[1]):
            return False
    return True


def design_counts() -> dict[str, int]:
    """Largest-remainder apportionment of 1/3 per varied type; controls get 4 each; total 300."""
    budget = N_DESIGN - N_DESIGN_CONTROL * len(CONTROL_TYPES)
    exact = {t: n / 3 for t, n in COUNTS.items()}
    counts = {t: int(np.floor(x)) for t, x in exact.items()}
    order = sorted(COUNTS, key=lambda t: (-(exact[t] - counts[t]), FEATURE_TYPES.index(t)))
    for t in order[: budget - sum(counts.values())]:
        counts[t] += 1
    return {**counts, **{c: N_DESIGN_CONTROL for c in CONTROL_TYPES}}


def build_panel(op_name: str, q0: float, f_grid: tuple[float, ...]) -> dict[str, Any]:
    op_index = OPERATING_POINTS.index(op_name)
    draw_seq, split_seq = np.random.SeedSequence(ROOT_SEED).spawn(2 * len(OPERATING_POINTS))[
        2 * op_index : 2 * op_index + 2
    ]
    rng = np.random.default_rng(draw_seq)
    slots: list[dict[str, Any]] = []
    for typ in (t for t in FEATURE_TYPES if t in COUNTS):
        accepted: list[np.ndarray] = []
        for k in range(COUNTS[typ]):
            redraws = 0
            while True:
                base = str(rng.choice(MIX_BASES)) if typ == "MIX" else None
                m = N_FEATURES[base if base else typ]
                slot = {
                    "sid": f"{op_name}-{typ}-{k:03d}",
                    "type": typ,
                    "base": base,
                    "raw_logits": [float(x) for x in rng.normal(0.0, LOGIT_SD, size=m)],
                    "rho": float(rng.uniform(*RHO_RANGE)) if typ == "MIX" else None,
                }
                if _valid_for_grid(slot, f_grid):
                    canon = canonical(slot, f_grid[0])
                    same = [a for a in accepted if len(a) == len(canon)]
                    if all(np.max(np.abs(canon - a)) >= DEDUP_TOL for a in same):
                        accepted.append(canon)
                        break
                redraws += 1
            slot["redraws"] = redraws
            slots.append(slot)
    for ctrl in CONTROL_TYPES:  # parameter-free identical controls
        for k in range(N_CONTROL):
            slots.append({"sid": f"{op_name}-{ctrl}-{k:03d}", "type": ctrl, "base": None,
                          "raw_logits": [0.0] if ctrl == "SINGLE" else [], "rho": None,
                          "redraws": 0})  # fmt: skip
    split_rng = np.random.default_rng(split_seq)
    n_design_by_type = design_counts()
    for typ in (*FEATURE_TYPES, "RFP"):
        idx = [i for i, s in enumerate(slots) if s["type"] == typ]
        n_design = n_design_by_type[typ]
        order = split_rng.permutation(len(idx))
        for rank, j in enumerate(order):
            slots[idx[j]]["split"] = "design" if rank < n_design else "test"
    return {
        "operating_point": op_name,
        "q0": q0,
        "f_grid": list(f_grid),
        "root_seed": ROOT_SEED,
        "logit_sd": LOGIT_SD,
        "slots": slots,
    }


def write_panel(path: Path, panel: dict[str, Any]) -> None:
    path.write_text(json.dumps(panel, indent=1, sort_keys=True) + "\n")


def load_split(
    path: Path, split: str, approval_file: Path = DEFAULT_APPROVAL
) -> list[dict[str, Any]]:
    """Slots of one split. The held-out split requires the approval file (registry E002 §2)."""
    if split not in ("design", "test"):
        raise ValueError(split)
    if split == "test" and not Path(approval_file).exists():
        raise PermissionError("held-out split is sealed: approval file absent")
    panel = json.loads(Path(path).read_text())
    return [s for s in panel["slots"] if s["split"] == split]
