"""E005a-R step 0: signal anchors from FROZEN historical data (no E005a-R code exists yet).

E004a's oracle C (update level, Adam t = 0 metric, U-toy parameterization) is not directly
comparable with the E005a-R estimand (reward level, identity metric, a new parameterization). The
anchor is therefore the dimensionless per-group detectability of the orthogonal error,

    tau = C^2 / sqrt(2 tr((P_perp Sigma_delta P_perp)^2)),

the ratio of C^2 to the leading-order null SD of a bias-corrected C^2 estimate from ONE group
(n groups: z ~ n tau). It is evaluated at the reward level (M_I, m = 8) for every E004a design
structure (configs/e004/design_panel_0b.json) at its base policy with its own verifier, using the
E005a exact oracle and E005a Monte Carlo group covariances (10^6 groups). Cross-check: the E005a
design panel's natural theta_0 points (the same generator, a fresh draw).
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from vdyn import provenance
from vdyn.e004 import toy
from vdyn.e005 import panel as pl
from vdyn.e005 import sampling as sm

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e005ar" / "e005ar_anchors.toml"
E004A = REPO / "configs/e004/design_panel_0b.json"
E005A = REPO / "configs/e005/calibration_panel_design.json"


def tau_of(point: dict[str, Any], m: int, seed: np.random.SeedSequence,
           n_groups: int) -> dict[str, Any]:  # fmt: skip
    ex = pl.exact(point)
    gG, ge = ex["g_G"], ex["g_V"] - ex["g_G"]
    A2 = float(gG @ gG)
    out: dict[str, Any] = {"pid": point["pid"], "mechanism": point["mechanism"], "A2": A2,
                           "C2": point["oracle"]["I"]["C2"]}  # fmt: skip
    if A2 <= 0:
        return out | {"tau": float("nan"), "kappa": float("nan")}
    u = gG / np.sqrt(A2)
    alpha = float(ge @ u) / np.sqrt(A2)
    sig = sm.oracle_sigma(point, m, np.random.default_rng(seed), n_groups=n_groups, chunk=50_000)
    Sd = sig["e"] - alpha * (sig["eG"] + sig["eG"].T) + alpha**2 * sig["G"]
    Pp = np.eye(len(u)) - np.outer(u, u)
    Sp = Pp @ Sd @ Pp
    tr1, tr2 = float(np.trace(Sp)), float(np.trace(Sp @ Sp))
    return out | {"alpha": alpha, "trS_perp": tr1, "trS2_perp": tr2, "d_eff": tr1**2 / tr2,
                  "tau": out["C2"] / np.sqrt(2 * tr2),
                  "kappa": float(np.sqrt(out["C2"] / A2))}  # fmt: skip


def _job(args: tuple[dict[str, Any], int, np.random.SeedSequence, int]) -> dict[str, Any]:
    return tau_of(*args)


def summarize(rows: list[dict[str, Any]], q: dict[str, float]) -> dict[str, Any]:
    nz = [r for r in rows if np.isfinite(r["tau"]) and r["kappa"] > 1e-6]
    tau = np.array([r["tau"] for r in nz])
    kap = np.array([r["kappa"] for r in nz])
    rng = np.random.default_rng(0)
    boot = np.array([np.quantile(rng.choice(tau, len(tau)), list(q.values()))
                     for _ in range(2000)])  # fmt: skip
    by_mech: dict[str, Any] = {}
    for mech in sorted({r["mechanism"] for r in rows}):
        t = np.array([r["tau"] for r in nz if r["mechanism"] == mech])
        by_mech[mech] = {"n_nonzero": len(t), "n_total": sum(r["mechanism"] == mech for r in rows),
                         "tau_median": float(np.median(t)) if len(t) else None}  # fmt: skip
    return {
        "n_total": len(rows), "n_nonzero": len(nz),
        "tau": {k: float(np.quantile(tau, v)) for k, v in q.items()},
        "tau_boot95": {k: [float(np.quantile(boot[:, i], 0.025)),
                           float(np.quantile(boot[:, i], 0.975))]
                       for i, k in enumerate(q)},
        "kappa": {k: float(np.quantile(kap, v)) for k, v in q.items()},
        "d_eff_median": float(np.median([r["d_eff"] for r in nz])),
        "by_mechanism": by_mech,
    }  # fmt: skip


def main() -> int:
    cfg = provenance.load_config(CONFIG)
    a = cfg["anchor"]
    m, n_groups, q = a["group_size_m"], a["oracle_groups"], a["quantiles"]
    structs = json.loads(E004A.read_text())["structures"]
    pts = [pl.make_point(toy.Structure.from_dict(s), "theta_0", 1.0, 0, 0.8, 0.0, "natural")
           for s in structs]  # fmt: skip
    cross = [
        p
        for p in json.loads(E005A.read_text())["points"]
        if p["variant"] == "theta_0" and p["d_extra"] == 0 and p["dose"] == "natural"
    ]
    seeds = np.random.SeedSequence(cfg["root_seed"]).spawn(6)[0].spawn(len(pts) + len(cross))
    jobs = [(p, m, s, n_groups) for p, s in zip(pts + cross, seeds, strict=True)]
    run_dir = provenance.create_run_dir(REPO / "results", "E005aR-anchors", REPO)
    provenance.write_metadata(
        run_dir,
        "E005aR-anchors",
        CONFIG,
        REPO,
        extra={"root_seed": cfg["root_seed"], "source": str(E004A.name)},
    )
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(_job, jobs, chunksize=4))
    hist, xc = rows[: len(pts)], rows[len(pts) :]
    report = {
        "anchors_E004a_design": summarize(hist, q),
        "crosscheck_E005a_design": summarize(xc, q),
    }
    t = report["anchors_E004a_design"]["tau"]
    report["implied_z_n_tau"] = {k: {str(n): n * v for n in (4, 8, 16, 32, 64, 128)}
                                 for k, v in t.items()}  # fmt: skip
    (run_dir / "anchors.json").write_text(json.dumps(report, indent=1) + "\n")
    (run_dir / "per_structure.json").write_text(json.dumps(rows) + "\n")
    print(json.dumps(report, indent=1))
    print(f"run directory: {run_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
