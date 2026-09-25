"""E002a PILOT (registry E002 §10): estimator characterization on fixed, non-panel structures.

Pilot: R = reps_pilot replications (formal run: reps_formal, pending approval). All rollouts are
audited (paired estimators). Truth comes from the exact decomposition (autodiff + exact metric).
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logit

from vdyn import provenance
from vdyn.e002 import estimators as est
from vdyn.e002 import panel as pn
from vdyn.geometry import autodiff
from vdyn.geometry.decompose import decompose
from vdyn.policies import bernoulli
from vdyn.verifiers import boolean_fp as bf

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e002" / "e002.toml"
ESTIMATORS = {  # name: (estimator, metric, lam)
    "plugin-exactF": ("plugin", "exact", 0.0),
    "plugin-estF-1e-3": ("plugin", "est", 1e-3),
    "plugin-estF-1e-2": ("plugin", "est", 1e-2),
    "plugin-estF-1e-1": ("plugin", "est", 1e-1),
    "plugin-euclid": ("plugin", "euclid", 0.0),
    "ustat-exactF": ("ustat", "exact", 0.0),
    "ustat-estF-1e-2": ("ustat", "est", 1e-2),
}
COVERAGE_N = (64, 256, 1024)
COVERAGE_EST = ("plugin-exactF", "plugin-estF-1e-2")


def structures(
    op: str, q0: float, f: float, cfg: dict[str, Any]
) -> dict[str, tuple[bf.EventStructure, float]]:
    raw = [float(x) for x in logit(np.array(cfg["asym_raw_probs"]))]
    asym = pn.match(
        {"sid": "AND2-asym", "type": "AND2", "base": None, "raw_logits": raw, "rho": None}, f
    )
    out = {
        "SINGLE": (bf.EventStructure("SINGLE", "SINGLE", (f,)), q0),
        "AND2-sym": (bf.EventStructure("AND2-sym", "AND2", (np.sqrt(f),) * 2), q0),
        "AND2-asym": (asym, q0),
        "OR2-sym": (bf.EventStructure("OR2-sym", "OR2", (1 - np.sqrt(1 - f),) * 2), q0),
        "RFP": (bf.EventStructure("RFP", "RFP", (), coin=f), q0),
    }
    for rho in cfg["rho_dose"]:
        if rho == 0.0:
            out["dose-rho=0"] = (bf.EventStructure("dose0", "RFP", (), coin=f), q0)
        else:
            s_e, coin = 1 - (1 - f) ** rho, 1 - (1 - f) ** (1 - rho)
            out[f"dose-rho={rho:g}"] = (
                bf.EventStructure(f"dose{rho}", "MIX", (s_e,), coin=coin, base="SINGLE", rho=rho),
                q0,
            )
    for q in (*cfg["q_extreme"], q0):
        out[f"Aext-q={q:g}"] = (bf.EventStructure(f"aext{q}", "SINGLE", (f,)), q)
    return out


def truth(st: bf.EventStructure, q: float) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    lp = bernoulli.product_log_prob(1 + st.n_features)
    theta = np.concatenate([[logit(q)], logit(np.asarray(st.s0))])
    g_gold = autodiff.reward_gradient(lp, theta, bf.gold_table(st))
    g_ver = autodiff.reward_gradient(lp, theta, bf.verifier_table(st))
    fisher = autodiff.fisher(lp, theta)
    res: dict[str, dict[str, float]] = {}
    for metric, m in (("fisher", np.linalg.inv(fisher)), ("euclid", np.eye(len(theta)))):
        d = decompose(g_gold, g_ver, m)
        res[metric] = {"A": d.A, "alpha": d.alpha, "C": d.C}
    return res, fisher


def run_estimator(name: str, st, q, batch, fisher) -> dict[str, np.ndarray]:
    estimator, metric, lam = ESTIMATORS[name]
    s = np.asarray(st.s0)
    if metric == "euclid":
        sc = est.score_vectors(q, s, batch.corr, batch.z)
        m = np.broadcast_to(np.eye(sc.shape[2]), (sc.shape[0], sc.shape[2], sc.shape[2]))
        g_gold = est.rloo(batch.corr.astype(float), sc)
        return est.decompose_batch(g_gold, est.rloo(batch.v, sc), m)
    exact = fisher if metric == "exact" else None
    return est.geometry(st, batch, None, q, s, estimator, lam, "paired", exact_fisher=exact)


def gram_form(out: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """P = alpha A^2 (0 when A = 0) and D = A^2 C^2; the U-statistic returns them directly."""
    if "P" in out:
        return out["P"], out["D"]
    a2 = out["A"] ** 2
    return np.where(out["alpha_defined"], out["alpha"] * a2, 0.0), a2 * out["C"] ** 2


def bootstrap_coverage(name, st, q, batch, fisher, c_true, n_boot, rng) -> float:
    reps, n = batch.corr.shape
    hits = np.zeros(reps, dtype=bool)
    boot_vals = np.empty((n_boot, reps))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=(reps, n))
        resampled = est.Batch(
            np.take_along_axis(batch.corr, idx, axis=1),
            np.take_along_axis(batch.z, idx[..., None], axis=1),
            np.take_along_axis(batch.v, idx, axis=1),
        )
        boot_vals[b] = run_estimator(name, st, q, resampled, fisher)["C"]
    lo, hi = np.quantile(boot_vals, [0.025, 0.975], axis=0)
    hits = (lo <= c_true) & (c_true <= hi)
    return float(hits.mean())


def main() -> int:
    config = provenance.load_config(CONFIG)
    cfg = config["e002a"]
    targets_dir = Path(sys.argv[1]).resolve()
    run_dir = provenance.create_run_dir(REPO / "results", "E002a-pilot", REPO)
    provenance.write_metadata(
        run_dir, "E002a-pilot", CONFIG, REPO, extra={"reps": cfg["reps_pilot"]}
    )
    reps, n_grid = cfg["reps_pilot"], cfg["n_grid"]
    report: dict[str, Any] = {}
    for op_i, op in enumerate(config["operating_points"]):
        f = json.loads((targets_dir / f"design_targets_{op}.json").read_text())["f_selected"]
        q0 = config["operating_points"][op]["q0"]
        structs = structures(op, q0, f, cfg)
        solved = {k: truth(st, q) for k, (st, q) in structs.items()}
        truths = {k: v[0] for k, v in solved.items()}
        fishers = {k: v[1] for k, v in solved.items()}
        rows: dict[str, Any] = {}
        c2_samples: dict[tuple[str, int, str], np.ndarray] = {}
        for s_i, (key, (st, q)) in enumerate(structs.items()):
            fisher = fishers[key]
            for n in n_grid:
                rng = np.random.default_rng([config["pilot_seed"], 2, op_i, s_i, n])
                batch = est.sample_rollouts(st, q, np.asarray(st.s0), n, reps, rng)
                degen_obs = {
                    "all_gold_equal": float(np.mean(np.ptp(batch.corr, axis=1) == 0)),
                    "no_false_positive": float(
                        np.mean(((1 - batch.corr) * batch.v).sum(axis=1) == 0)
                    ),
                    "some_feature_constant": float(
                        np.mean(np.any(np.ptp(batch.z, axis=1) == 0, axis=1))
                    )
                    if st.n_features
                    else 0.0,
                }
                degen_pred = est.degenerate_event_probabilities(
                    q, bf.fpr(st, np.asarray(st.s0)), np.asarray(st.s0), n
                )
                for name in ESTIMATORS:
                    out = run_estimator(name, st, q, batch, fisher)
                    tm = truths[key]["euclid" if name == "plugin-euclid" else "fisher"]
                    defined = out["alpha_defined"]
                    p_hat, d_hat = gram_form(out)
                    p_true = tm["alpha"] * tm["A"] ** 2 if tm["A"] > 0 else 0.0
                    row = {
                        "A_bias": float(np.mean(out["A"]) - tm["A"]),
                        "A_sd": float(np.std(out["A"])),
                        "C_bias": float(np.mean(out["C"]) - tm["C"]),
                        "C_sd": float(np.std(out["C"])),
                        "C_rmse": float(np.sqrt(np.mean((out["C"] - tm["C"]) ** 2))),
                        "C2_bias": float(np.mean(out["C2"]) - tm["C"] ** 2),
                        "C2_mcse": float(np.std(out["C2"]) / np.sqrt(reps)),
                        "P_bias": float(np.mean(p_hat) - p_true),
                        "P_mcse": float(np.std(p_hat) / np.sqrt(reps)),
                        "D_bias": float(np.mean(d_hat) - tm["A"] ** 2 * tm["C"] ** 2),
                        "D_mcse": float(np.std(d_hat) / np.sqrt(reps)),
                        "alpha_undefined_rate": float(1 - defined.mean()),
                        "alpha_bias_defined": float(np.nanmean(out["alpha"]) - tm["alpha"])
                        if defined.any()
                        else None,
                        "alpha_sd_defined": float(np.nanstd(out["alpha"]))
                        if defined.any()
                        else None,
                        "C_true": tm["C"],
                        "A_true": tm["A"],
                        "alpha_true": tm["alpha"],
                    }
                    if (
                        n in COVERAGE_N
                        and name in COVERAGE_EST
                        and key in ("SINGLE", "AND2-sym", "OR2-sym", "RFP")
                    ):
                        row["coverage95_C"] = bootstrap_coverage(
                            name, st, q, batch, fisher, tm["C"], cfg["bootstrap"], rng
                        )
                    rows[f"{key}|N={n}|{name}"] = row
                    c2_samples[(key, n, name)] = out["C2"]
                    if name == "plugin-exactF":
                        degen_obs["A_hat_zero"] = float(np.mean(out["A"] == 0))
                degen_pred["A_hat_zero"] = degen_pred["all_gold_equal"]
                rows[f"{key}|N={n}|degenerate"] = {
                    "observed": degen_obs,
                    "predicted": degen_pred,
                    "mcse": {k: float(np.sqrt(p * (1 - p) / reps)) for k, p in degen_pred.items()},
                }
        # detection: RFP-calibrated threshold, false alarm, power, P(C_Y > C_RFP), MDC (dose)
        detection: dict[str, Any] = {}
        for n in n_grid:
            for name in ESTIMATORS:
                cal_rng = np.random.default_rng([config["pilot_seed"], 3, op_i, n])
                st_r, q_r = structs["RFP"]
                cal = run_estimator(
                    name,
                    st_r,
                    q_r,
                    est.sample_rollouts(st_r, q_r, np.array([]), n, reps, cal_rng),
                    fishers["RFP"],
                )["C2"]
                thr = float(np.quantile(cal, 0.95))
                null = c2_samples[("RFP", n, name)]
                det: dict[str, Any] = {"threshold": thr, "false_alarm": float(np.mean(null > thr))}
                for key in ("SINGLE", "AND2-sym", "AND2-asym", "OR2-sym"):
                    y = c2_samples[(key, n, name)]
                    det[f"power|{key}"] = float(np.mean(y > thr))
                    det[f"P(C_Y>C_RFP)|{key}"] = float(
                        np.mean(y[:, None] > null[None, :])
                        + 0.5 * np.mean(y[:, None] == null[None, :])
                    )
                dose_c, dose_pow = [], []
                for rho in cfg["rho_dose"]:
                    k = f"dose-rho={rho:g}"
                    dose_c.append(truths[k]["fisher"]["C"])
                    dose_pow.append(float(np.mean(c2_samples[(k, n, name)] > thr)))
                order = np.argsort(dose_c)
                dc, dp = np.array(dose_c)[order], np.array(dose_pow)[order]
                above = np.flatnonzero(dp >= 0.8)
                if len(above) == 0:
                    mdc = None
                elif above[0] == 0:
                    mdc = float(dc[0])
                else:
                    j = above[0]
                    mdc = float(np.interp(0.8, [dp[j - 1], dp[j]], [dc[j - 1], dc[j]]))
                det["dose_C"] = dc.tolist()
                det["dose_power"] = dp.tolist()
                det["min_detectable_C"] = mdc
                detection[f"N={n}|{name}"] = det
        # F4 per-sample noise check (oracle metric, no baseline)
        f4 = {}
        for key in ("SINGLE", "AND2-sym"):
            st, q = structs[key]
            rng = np.random.default_rng([config["pilot_seed"], 4, op_i])
            batch = est.sample_rollouts(st, q, np.asarray(st.s0), 200_000, 1, rng)
            sc = est.score_vectors(q, np.asarray(st.s0), batch.corr, batch.z)[0]
            gam = ((batch.v - batch.corr)[0])[:, None] * sc
            minv = np.linalg.inv(fishers[key])
            g_e = gam.mean(axis=0)
            per = np.einsum("ni,ij,nj->n", gam - g_e, minv, gam - g_e)
            s_e = bf.event_prob(st, np.asarray(st.s0))
            k2 = bf.kappa2(st, np.asarray(st.s0))
            lp = bernoulli.product_log_prob(1 + st.n_features)
            theta = np.concatenate([[logit(q)], logit(np.asarray(st.s0))])
            ge_exact = autodiff.reward_gradient(
                lp, theta, bf.verifier_table(st)
            ) - autodiff.reward_gradient(lp, theta, bf.gold_table(st))
            formula = (1 - q) * k2 / s_e + s_e * q - ge_exact @ minv @ ge_exact
            f4[key] = {
                "empirical": float(per.mean()),
                "mcse": float(per.std() / np.sqrt(len(per))),
                "formula": float(formula),
            }
        report[op] = {"f": f, "q0": q0, "rows": rows, "detection": detection, "f4_noise": f4}
        print(f"{op}: {len(rows)} rows, {len(detection)} detection cells")
    (run_dir / "e002a_pilot.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"run directory: {run_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
