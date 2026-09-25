"""E002b arms as one callable (registry E002 §5): the code path shared by tuning and held-out runs.

Seeds follow the design pilot exactly: every G1 / G1-oracle configuration of a (structure, cell)
sees the same rollouts, `default_rng([seed, 5, op, structure, cell, 1])`; any other arm uses
`default_rng([seed, 5, op, structure, cell, arm_index, config_index])`.
"""

import itertools
from typing import Any

import numpy as np
from scipy.special import logit

from vdyn.e002 import estimators as est
from vdyn.e002 import probes as pr
from vdyn.verifiers.boolean_fp import EventStructure

ARMS = ("G0", "G1", "G1-oracle", "P1", "P2", "P3", "P4")
COMPETITORS = ("P1", "P2", "P3", "P4")


def configs(arm: str) -> list[dict[str, Any]]:
    g = pr.TUNING_GRIDS
    if arm == "G0":
        return [{}]
    if arm in ("G1", "G1-oracle"):
        lams = g["G1"]["lam"] if arm == "G1" else (0.0,)
        return [
            {"estimator": e, "lam": lam, "source": s}
            for e, lam, s in itertools.product(g["G1"]["estimator"], lams, g["G1"]["source"])
            if not (e == "ustat" and s == "pooled")
        ]
    keys = list(g[arm])
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*g[arm].values())]


def plan(arm: str, cfg: dict[str, Any], b_gold: int, b_roll: int) -> pr.Plan:
    if arm == "G0":
        return pr.plan_g0(b_gold, b_roll)
    if arm in ("G1", "G1-oracle"):
        return pr.plan_g1(b_gold, b_roll)
    if arm == "P1":
        return pr.plan_p1(b_gold, b_roll, cfg["k"])
    if arm == "P2":
        return pr.plan_p2(b_gold, b_roll, cfg["k"])
    if arm == "P3":
        return pr.plan_p3(b_gold, b_roll, cfg["k"])
    return pr.plan_p4(b_gold, b_roll)


def exact_fisher(q0: float, s0: np.ndarray) -> np.ndarray:
    """Fisher of the product-Bernoulli policy (diagonal): the G1-oracle metric only."""
    p = np.concatenate([[q0], s0])
    return np.diag(p * (1 - p))


def run_arm(
    arm: str,
    cfg: dict[str, Any],
    st: EventStructure,
    q0: float,
    b_gold: int,
    b_roll: int,
    reps: int,
    seed: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Scores (reps,) of one arm configuration at one budget cell; None if infeasible.
    `seed` = (root seed, operating-point index, structure index, cell index)."""
    root, op_i, s_idx, c_i = seed
    s0 = np.asarray(st.s0, dtype=np.float64)
    if arm in ("G1", "G1-oracle"):
        rng = np.random.default_rng([root, 5, op_i, s_idx, c_i, 1])
        lab = est.sample_rollouts(st, q0, s0, b_gold, reps, rng)
        unl = est.sample_rollouts(st, q0, s0, b_roll - b_gold, reps, rng)
        exact = exact_fisher(q0, s0) if arm == "G1-oracle" else None
        out = est.geometry(st, lab, unl, q0, s0, cfg["estimator"], cfg["lam"], cfg["source"],
                           exact_fisher=exact)  # fmt: skip
        return np.asarray(out["C2"], dtype=np.float64)
    p = plan(arm, cfg, b_gold, b_roll)
    if not p.feasible:
        return None
    rng = np.random.default_rng([root, 5, op_i, s_idx, c_i, ARMS.index(arm),
                                 configs(arm).index(cfg)])  # fmt: skip
    theta0 = np.concatenate([[logit(q0)], logit(s0)])
    if arm == "G0":
        res = pr.run_g0(st, theta0, b_gold, reps, rng)
    elif arm == "P1":
        res = pr.run_p1(st, theta0, p.b, cfg["k"], cfg["eta"], reps, rng)
    elif arm == "P2":
        res = pr.run_p2(st, theta0, b_gold, p.b, cfg["k"], cfg["eta"], reps, rng,
                        reuse_audit=p.reuse_audit)  # fmt: skip
    elif arm == "P3":
        res = pr.run_p3(st, theta0, q0, b_gold, p.b, cfg["k"], cfg["eta"], cfg["observable"],
                        reps, rng)  # fmt: skip
    else:
        res = pr.run_p4(st, theta0, b_gold, b_roll - b_gold, cfg["r"], reps, rng)
    return np.asarray(res["score"], dtype=np.float64)
