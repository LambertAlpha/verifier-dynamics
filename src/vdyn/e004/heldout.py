"""E004a Stage 1 held-out infrastructure (registry: Amendment 4 and the pre-unseal integrity log).

- Sealed splits: the test panel (seed stream 1) and the shift panel (seed stream 2) are generated
  only when the approval file exists and names the frozen predictor configuration (E002 pattern).
- Test panel: the Stage 0b generator, 24 cells x 32 structures, fresh stream.
- Shift panel (memo §10 C): the same generator with J_G(0) ~ U(0.01, 0.05); trained with 4 x 4
  GRPO batches.
- Seed-tree extension: every role and audit branch indexes structures globally (design 0-767,
  test 768-1535, shift 1536-2303), so the design seeds are unchanged.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from vdyn.e004 import audit as au
from vdyn.e004 import panel0b as pb
from vdyn.e004 import toy

APPROVAL = Path(__file__).resolve().parents[3] / "configs" / "e004" / "HELDOUT_APPROVED"
ROLES = ("prim_ver", "prim_clean", "can_ver", "can_clean", "b16_ver", "b16_clean", "b256_ver",
         "b256_clean", "hardpair", "spare")  # fmt: skip
OFFSET = {"design": 0, "test": 768, "shift": 1536}
TEST_STREAM, SHIFT_STREAM = 1, 2
SHIFT_JG = (0.01, 0.05)
SHIFT_BATCH = (4, 4)
N_SEEDS = 4


def require_approval(path: Path, predictors_sha: str) -> dict[str, Any]:
    if not Path(path).exists():
        raise PermissionError("held-out splits are sealed: approval file absent")
    appr = json.loads(Path(path).read_text())
    if appr.get("predictors_frozen_sha256") != predictors_sha:
        raise PermissionError("approval does not name the frozen predictor configuration")
    return appr


def shift_targets(rng: np.random.Generator) -> dict[str, float]:
    """The registered target draws with the harder J_G(0) range (same number of draws)."""
    return {
        "J_G": rng.uniform(*SHIFT_JG),
        "FPR": float(np.exp(rng.uniform(np.log(0.02), np.log(0.4)))),
        "FNR": rng.uniform(0.0, 0.25),
    }


def test_panel(approval: Path, predictors_sha: str, n_per: int = pb.N_PER_CELL,
               yb_cap: float = 0.2) -> list[toy.Structure]:  # fmt: skip
    require_approval(approval, predictors_sha)
    return pb._cells_panel(TEST_STREAM, pb.CELLS, n_per, yb_cap, prefix="T-")


def shift_panel(approval: Path, predictors_sha: str, n_per: int = pb.N_PER_CELL,
                yb_cap: float = 0.2) -> list[toy.Structure]:  # fmt: skip
    require_approval(approval, predictors_sha)
    return pb._cells_panel(SHIFT_STREAM, pb.CELLS, n_per, yb_cap, prefix="S-",
                           targets=shift_targets)  # fmt: skip


def _roles() -> list[np.random.SeedSequence]:
    return np.random.SeedSequence(pb.ROOT_SEED).spawn(5)[3].spawn(len(ROLES))


def run_seeds(role: str, split: str, n: int,
              n_seeds: int = N_SEEDS) -> list[list[np.random.SeedSequence]]:  # fmt: skip
    kids = _roles()[ROLES.index(role)].spawn(OFFSET[split] + n)[OFFSET[split] :]
    return [k.spawn(n_seeds) for k in kids]


def audit_seeds(kind: str, split: str, n: int, n_seeds: int = N_SEEDS) -> list[Any]:
    branch = _roles()[ROLES.index("spare")].spawn(4)[{"adam": 0, "ng": 1}[kind]]
    kids = branch.spawn(OFFSET[split] + n)[OFFSET[split] :]
    return [k.spawn(n_seeds) for k in kids] if kind == "adam" else kids


def resample_seeds() -> list[np.random.SeedSequence]:
    """[0-3] design (L2N permutation, Adam W, NG W, post-hoc extra audits); [4] test Adam W;
    [5] test NG W; [6] shift W; [7] test L2N permutation; [8] post-hoc."""
    return _roles()[ROLES.index("spare")].spawn(4)[3].spawn(9)


def pooled_jv(audit: au.Groups, train: au.Groups | None) -> np.ndarray:
    """Mean verifier reward over the audit and the training batch (groups may differ in size)."""
    if train is None:
        return audit.V.mean((1, 2))
    return (audit.V.sum((1, 2)) + train.V.sum((1, 2))) / (audit.V[0].size + train.V[0].size)


def warnings_at(p: np.ndarray, horizons: list[float], success: np.ndarray, fail: np.ndarray,
                t_on: np.ndarray, h_obs: float, tau: float) -> dict[str, Any]:  # fmt: skip
    """pr.warnings with the frozen, design-calibrated threshold tau (execution note §4)."""
    hs = [i for i, h in enumerate(horizons) if h <= h_obs + 1e-12]
    run = np.max(p[hs], axis=0)
    warned = run > tau
    first = np.argmax(p[hs] > tau, axis=0)
    t_warn = np.where(warned, np.array(horizons)[hs][first], np.nan)
    nyv_fail = fail & (t_on > h_obs)
    lead = t_on - t_warn
    lw = lead[nyv_fail & warned]
    cons = np.where(warned[nyv_fail], lead[nyv_fail], 0.0)
    return {
        "tau": tau,
        "warned": warned,
        "t_warn": t_warn,
        "false_alarm": float(np.mean(warned[success])) if success.any() else float("nan"),
        "lead_warned": lw,
        "median_lead": float(np.median(lw)) if len(lw) else float("nan"),
        "sensitivity": float(np.mean(warned[nyv_fail])) if nyv_fail.any() else float("nan"),
        "median_lead_conservative": float(np.median(cons)) if len(cons) else float("nan"),
        "n_nyv_fail": int(nyv_fail.sum()),
    }
