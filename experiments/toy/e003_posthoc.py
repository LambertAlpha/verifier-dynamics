"""E003 POST-HOC / EXPLORATORY analysis of the registered natural-gradient run.

Nothing here is confirmatory. It reads the registered trajectories (verified against the committed
SHA-256), runs no new training, and writes to results/E003-posthoc/.

  D1  numerical diagnostic for the three FAILED P4 checks (the original rows stay FAILED)
  Q4  short-horizon summaries of the trajectory prefix t <= k vs the final gold shortfall
  Q5  decomposition of the registered OR vs AND-ASYM-B inversion along the trajectories

Usage:  uv run python experiments/toy/e003_posthoc.py results/E003/<run_dir>
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import expit, log_expit

from vdyn import provenance
from vdyn.checks import spearman_with_ties
from vdyn.geometry import triggered_fp as cf
from vdyn.verifiers.triggered import structure_from_config

REPO = Path(__file__).resolve().parents[2]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PREFIXES = (1.0, 5.0, 10.0)
# Orientation fixed by meaning, not fitted: +1 = larger value should mean larger shortfall.
SUMMARIES = {
    "C(0)": +1,
    "eta(0)": +1,
    "mean eta on [0,k]": +1,
    "integral eta on [0,k]": +1,
    "delta eta on [0,k]": +1,
    "A(0)": -1,
    "delta A on [0,k]": -1,
    "delta C on [0,k]": +1,
    "delta log FPR on [0,k] (static metric)": +1,
    "delta log J_G on [0,k] (early gold)": -1,
}


def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    expected = (run_dir / "trajectories.npz.sha256").read_text().split()[0]
    actual = hashlib.sha256((run_dir / "trajectories.npz").read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("trajectories.npz does not match the committed SHA-256")
    config = provenance.load_config(run_dir / "config.toml")
    arrays = np.load(run_dir / "trajectories.npz")
    series: dict[str, dict[str, np.ndarray]] = {}
    for entry in config["structures"]:
        prefix = f"{entry['name']}_natural_"
        series[entry["name"]] = {
            k[len(prefix) :]: arrays[k] for k in arrays.files if k.startswith(prefix)
        }
    return config, series


def shortfalls(config: dict[str, Any], series: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    tol = config["tolerances"]["success_gold"]
    gold_end = np.array([s["gold"][-1] for s in series.values()])
    return np.where(gold_end >= 1 - tol, 0.0, 1 - gold_end)


def prefix_summaries(ser: dict[str, np.ndarray], k: float) -> dict[str, float]:
    """Summaries of the trajectory prefix t <= k (grid points only; eta finite on this prefix)."""
    t = ser["t"]
    m = t <= k + 1e-12
    eta = ser["eta_F"][m]
    j = int(np.flatnonzero(m)[-1])
    return {
        "C(0)": float(ser["C_F"][0]),
        "eta(0)": float(eta[0]),
        "mean eta on [0,k]": float(trapezoid(eta, t[m]) / t[j]),
        "integral eta on [0,k]": float(trapezoid(eta, t[m])),
        "delta eta on [0,k]": float(eta[-1] - eta[0]),
        "A(0)": float(ser["A_F"][0]),
        "delta A on [0,k]": float(ser["A_F"][j] - ser["A_F"][0]),
        "delta C on [0,k]": float(ser["C_F"][j] - ser["C_F"][0]),
        "delta log FPR on [0,k] (static metric)": float(np.log(ser["fpr"][j] / ser["fpr"][0])),
        "delta log J_G on [0,k] (early gold)": float(np.log(ser["gold"][j] / ser["gold"][0])),
    }


def ranking_quality(values: np.ndarray, shortfall: np.ndarray, orientation: int) -> dict[str, Any]:
    """Spearman (oriented), discordant pairs among distinct outcomes, stall/success separation."""
    v = orientation * values
    pairs = [
        (i, j) for i in range(len(v)) for j in range(len(v)) if shortfall[i] > shortfall[j] + 1e-9
    ]
    discordant = sum(v[i] < v[j] for i, j in pairs)
    ties = sum(v[i] == v[j] for i, j in pairs)
    stall, success = shortfall > 0, shortfall == 0
    separated = bool(v[stall].min() > v[success].max()) if stall.any() and success.any() else None
    spread = float(np.ptp(values) / max(np.mean(np.abs(values)), 1e-300))
    return {
        "spearman_oriented": spearman_with_ties(v, shortfall) if np.ptp(v) > 0 else float("nan"),
        "discordant_pairs": int(discordant),
        "tied_pairs": int(ties),
        "comparable_pairs": len(pairs),
        "stall_success_separated": separated,
        "relative_spread": spread,
    }


def stable_eta(kind: str, feature_logits: np.ndarray) -> np.ndarray:
    """eta evaluated from the logits without cancellation (1 - s = expit(-v), 1 - S via expm1).

    Covers the structures with a P4 failure: SINGLE (eta = 1 identically) and 2-feature AND.
    """
    if kind == "single":
        return np.ones(feature_logits.shape[1])
    v1, v2 = feature_logits
    s1, s2, c1, c2 = expit(v1), expit(v2), expit(-v1), expit(-v2)
    one_minus_s = -np.expm1(log_expit(v1) + log_expit(v2))
    return (s1 * c2 + s2 * c1) / one_minus_s


def diagnostic_p4(
    config: dict[str, Any], series: dict[str, dict[str, np.ndarray]]
) -> list[dict[str, Any]]:
    """D1: where do P4 violations sit; does closed-form eta at the observed states agree?"""
    out = []
    structures = {e["name"]: structure_from_config(e, config["f"]) for e in config["structures"]}
    for name in ("AND-ASYM-A", "AND-ASYM-B", "SINGLE"):
        ser, st = series[name], structures[name]
        finite = np.isfinite(ser["eta_F"])
        t, eta, cmax, fpr = (
            ser["t"][finite],
            ser["eta_F"][finite],
            ser["C_max"][finite],
            ser["fpr"][finite],
        )
        s = ser["s"][:, finite]
        eta_cf = np.array([cf.eta(st, s[:, k]) for k in range(s.shape[1])])
        if config["predictions"][name]["eta_trend"] == "constant":
            bad = np.abs(eta - eta[0]) > 1e-8
            worst_idx = int(np.argmax(np.abs(eta - eta[0])))
        else:
            steps = np.diff(eta)
            bad = np.r_[False, steps < -1e-8]
            worst_idx = int(np.argmin(steps)) + 1
        eta_st = stable_eta(st.kind, ser["theta"][1:, finite])
        st_steps = np.diff(eta_st)
        st_ok = (
            bool(np.all(st_steps >= -1e-12))
            if name != "SINGLE"
            else bool(np.max(np.abs(eta_st - 1)) < 1e-12)
        )
        cf_steps = np.diff(eta_cf)
        cf_monotone = (
            bool(np.all(cf_steps >= -1e-12))
            if name != "SINGLE"
            else bool(np.max(np.abs(eta_cf - 1)) < 1e-12)
        )
        out.append(
            {
                "structure": name,
                "n_violating_points": int(bad.sum()),
                "first_violation_t": float(t[bad][0]) if bad.any() else None,
                "worst_violation_t": float(t[worst_idx]),
                "C_max_at_worst": float(cmax[worst_idx]),
                "1-FPR_at_worst": float(1 - fpr[worst_idx]),
                "C_max_at_first_violation": float(cmax[bad][0]) if bad.any() else None,
                "min_C_max_where_no_violation_before": float(cmax[: np.flatnonzero(bad)[0]].min())
                if bad.any()
                else None,
                "max_abs(eta_generic - eta_closed_form)": float(np.max(np.abs(eta - eta_cf))),
                "closed_form_eta_obeys_registered_trend_on_observed_states": cf_monotone,
                "violations_if_restricted_to_C_max>1e-3": int((bad & (cmax > 1e-3)).sum()),
                "stable_eta_obeys_registered_trend": st_ok,
                "max_abs(eta_generic - eta_stable)": float(np.max(np.abs(eta - eta_st))),
                "max_abs(eta_closed_form_prob_space - eta_stable)": float(
                    np.max(np.abs(eta_cf - eta_st))
                ),
            }
        )
    return out


def inversion_q5(
    config: dict[str, Any], series: dict[str, dict[str, np.ndarray]]
) -> dict[str, Any]:
    """Q5: split log(J_G(T)/q0) = ∫ dlog S / eta at the eta crossing, per trajectory."""
    q0 = config["q0"]
    res: dict[str, Any] = {}
    grid = np.geomspace(config["f"], 1 - 1e-5, 4000)
    curves = {}
    for name in ("OR", "AND-ASYM-B"):
        ser = series[name]
        ok = np.isfinite(ser["eta_F"]) & (np.diff(np.r_[-1.0, ser["fpr"]]) > 0)
        s_obs, eta_obs = ser["fpr"][ok], ser["eta_F"][ok]
        curves[name] = np.interp(np.log(grid), np.log(s_obs), eta_obs)
        # direct observation: gold as a function of FPR (monotone in both)
        res[name] = {
            "eta(0)": float(ser["eta_F"][0]),
            "J_G(T_end)": float(ser["gold"][-1]),
            "log(J_G(T_end)/q0)": float(np.log(ser["gold"][-1] / q0)),
        }
    diff = curves["OR"] - curves["AND-ASYM-B"]
    cross = int(np.flatnonzero(np.diff(np.sign(diff)) != 0)[0]) + 1
    s_x = float(grid[cross])
    res["crossing_FPR"] = s_x
    for name in ("OR", "AND-ASYM-B"):
        ser = series[name]
        integrand = 1 / curves[name]
        logs = np.log(grid)
        before = float(trapezoid(integrand[: cross + 1], logs[: cross + 1]))
        after = float(trapezoid(integrand[cross:], logs[cross:]))
        k_x = int(np.argmin(np.abs(ser["fpr"] - s_x)))
        res[name].update(
            {
                "gold_log_gain_before_crossing(quadrature)": before,
                "gold_log_gain_after_crossing_to_S=1-1e-5(quadrature)": after,
                "t_at_crossing_FPR": float(ser["t"][k_x]),
                "J_G_at_crossing_FPR(observed)": float(ser["gold"][k_x]),
                "eta_at_crossing": float(np.interp(np.log(s_x), np.log(grid), curves[name])),
            }
        )
        big = ser["fpr"] >= 0.9
        res[name]["J_G_when_FPR_first_reaches_0.9(observed)"] = float(
            ser["gold"][np.flatnonzero(big)[0]]
        )
    res["_curves"] = {"grid": grid, **curves, "cross": cross}
    return res


def plot_e(config, series, table, shortfall, path: Path) -> None:
    names = list(series)
    panels = [
        ("C(0)", 1.0),
        ("mean eta on [0,k]", 5.0),
        ("delta log J_G on [0,k] (early gold)", 5.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (key, k) in zip(axes, panels, strict=True):
        for i, name in enumerate(names):
            x = table[k][name][key]
            ax.plot(x, shortfall[i], "o", color=PALETTE[i], ms=6)
            ax.annotate(
                name, (x, shortfall[i]), textcoords="offset points", xytext=(4, 4), fontsize=7
            )
        ax.set(
            xlabel=key + ("" if key.endswith("(0)") else f", k = {k:g}"),
            ylabel="final gold shortfall",
        )
        ax.grid(True, color="#e6e6e3", lw=0.6)
    fig.suptitle(
        "E003 POST-HOC / EXPLORATORY: early-horizon summaries vs final gold shortfall", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_q5(series, q5, path: Path) -> None:
    c = q5["_curves"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {"OR": PALETTE[6], "AND-ASYM-B": PALETTE[5]}
    for name in ("OR", "AND-ASYM-B"):
        ser = series[name]
        axes[0].plot(ser["t"][1:], ser["eta_F"][1:], color=colors[name], lw=1.6, label=name)
        axes[1].plot(c["grid"], c[name], color=colors[name], lw=1.6, label=name)
        gain = np.r_[
            0, np.cumsum(np.diff(np.log(c["grid"])) * 0.5 * (1 / c[name][1:] + 1 / c[name][:-1]))
        ]
        axes[2].plot(c["grid"], gain, color=colors[name], lw=1.6, label=name)
    for ax in axes[1:]:
        ax.axvline(q5["crossing_FPR"], color="#8a8a86", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("on-policy FPR S")
    axes[0].set(xscale="log", xlabel="t (log)", ylabel="eta(t)")
    axes[1].set(ylabel="eta as a function of S (dotted: crossing)")
    axes[2].set(ylabel="cumulative gold log-gain  ∫ dlog S / eta  = log(J_G / q0)")
    for ax in axes:
        ax.grid(True, color="#e6e6e3", lw=0.6)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        "E003 POST-HOC: why OR (higher C(0)) ends with more gold than AND-ASYM-B", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str]) -> int:
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--results-root", default=str(REPO / "results"))
    args = parser.parse_args(argv[1:] if argv and argv[0].endswith(".py") else argv)
    run_dir = Path(args.run_dir).resolve()
    config, series = load_run(run_dir)
    out_dir = provenance.create_run_dir(Path(args.results_root), "E003-posthoc", REPO)
    (out_dir / "source_run.txt").write_text(f"{run_dir.relative_to(REPO)}\n")

    shortfall = shortfalls(config, series)
    table = {k: {name: prefix_summaries(ser, k) for name, ser in series.items()} for k in PREFIXES}
    q4 = {
        f"k={k:g}": {
            key: ranking_quality(np.array([table[k][n][key] for n in series]), shortfall, sign)
            for key, sign in SUMMARIES.items()
        }
        for k in PREFIXES
    }
    q4["gold_fraction_reached_by_k"] = {
        f"k={k:g}": {
            n: float(s["gold"][s["t"] <= k + 1e-12][-1] / s["gold"][-1]) for n, s in series.items()
        }
        for k in PREFIXES
    }
    d1 = diagnostic_p4(config, series)
    q5 = inversion_q5(config, series)
    curves = q5.pop("_curves")
    result = {
        "label": "POST-HOC / EXPLORATORY (not confirmatory)",
        "shortfall": dict(zip(series, shortfall.tolist(), strict=True)),
        "D1_P4_diagnostic": d1,
        "Q4_prefix_ranking": q4,
        "Q4_prefix_values": {f"k={k:g}": table[k] for k in PREFIXES},
        "Q5_inversion": q5,
    }
    (out_dir / "posthoc.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_e(config, series, table, shortfall, out_dir / "fig_E_early_summaries.png")
    plot_q5(series, {**q5, "_curves": curves}, out_dir / "fig_Q5_inversion.png")
    print(json.dumps({k: v for k, v in result.items() if k != "Q4_prefix_values"}, indent=1))
    print(f"output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
