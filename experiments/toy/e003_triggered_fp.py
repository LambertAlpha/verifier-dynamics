"""E003 — Candidate 3: feature-triggered false positives matched on static metrics.

Pre-registration: research/03_experiment_registry.md (E003, commit 2736f50). Natural gradient is the
registered experiment. Vanilla (E003-V) is exploratory: its rows carry no pass/fail.
Observations are generic (enumeration + autodiff); closed forms enter only as registered
predictions.

Each invocation runs ONE optimizer:
  registered run:   uv run python experiments/toy/e003_triggered_fp.py
  exploratory run:  uv run python experiments/toy/e003_triggered_fp.py --optimizer vanilla
Options: [config.toml] [--results-root DIR]. Vanilla output goes to "<experiment_id>-V".
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logit

from vdyn import provenance
from vdyn.checks import Checks, spearman_with_ties
from vdyn.geometry import autodiff
from vdyn.geometry import triggered_fp as cf
from vdyn.geometry.decompose import decompose
from vdyn.policies import bernoulli
from vdyn.simulation import flows
from vdyn.verifiers.triggered import Structure, gold_table, structure_from_config, verifier_table

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "toy" / "e003_triggered_fp.toml"
# Categorical slots 1-8 in fixed order (validated). Contrast relief: legend, labels, summary table.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OBSERVED = (
    "gold", "J_V", "fpr", "fnr", "fp_mass", "verifier_accuracy",
    "A_F", "alpha_F", "C_F", "C_max", "eta_F", "A_E", "alpha_E", "C_E",
    "gold_rate_actual", "gold_rate_F", "gold_rate_E",
)  # fmt: skip


@dataclass
class Run:
    structure: Structure
    optimizer: str
    traj: flows.Trajectory
    wall_seconds: float
    series: dict[str, np.ndarray] = field(default_factory=dict)


def _field(structure: Structure, optimizer: str) -> flows.Field:
    log_prob = bernoulli.product_log_prob(1 + structure.n_features)
    table = verifier_table(structure)
    if optimizer == "natural":
        return flows.natural_field(log_prob, table)
    return flows.vanilla_field(log_prob, table)


def observe(structure: Structure, theta: np.ndarray, optimizer: str) -> dict[str, float]:
    """Generic observations at one state: enumerated static metrics, Fisher- and Euclidean-metric
    geometry, and the actual gold rate g_G . theta_dot under the optimizer being run."""
    log_prob = bernoulli.product_log_prob(1 + structure.n_features)
    g_tab, v_tab = gold_table(structure), verifier_table(structure)
    p = autodiff.outcome_probs(log_prob, theta)
    gold = float(p @ g_tab)
    fp_mass = float(p @ ((1 - g_tab) * v_tab))
    fn_mass = float(p @ (g_tab * (1 - v_tab)))
    fpr = fp_mass / float(p @ (1 - g_tab))
    g_gold = autodiff.reward_gradient(log_prob, theta, g_tab)
    g_ver = autodiff.reward_gradient(log_prob, theta, v_tab)
    dec = decompose(g_gold, g_ver, np.linalg.inv(autodiff.fisher(log_prob, theta)))
    dec_e = decompose(g_gold, g_ver, np.eye(len(theta)))
    c_max = (1 - gold) * np.sqrt(fpr * (1 - fpr))
    return {
        "gold": gold,
        "J_V": float(p @ v_tab),
        "fpr": fpr,
        "fnr": fn_mass / gold,
        "fp_mass": fp_mass,
        "verifier_accuracy": 1 - fp_mass - fn_mass,
        "A_F": dec.A,
        "alpha_F": dec.alpha,
        "C_F": dec.C,
        "C_max": c_max,
        "eta_F": (dec.C / c_max) ** 2 if c_max > 1e-6 else np.nan,
        "A_E": dec_e.A,
        "alpha_E": dec_e.alpha,
        "C_E": dec_e.C,
        "gold_rate_actual": float(g_gold @ _field(structure, optimizer)(0.0, theta)),
        "gold_rate_F": dec.gold_rate,
        "gold_rate_E": dec_e.gold_rate,
    }


def simulate(structure: Structure, optimizer: str, config: dict[str, Any]) -> Run:
    solver, t_end = config["solver"], config["horizon"][optimizer]
    report = [t for t in solver["report_times"] if t <= t_end]
    t_eval = np.unique(
        np.concatenate([[0.0], np.geomspace(solver["t_min"], t_end, solver["n_eval"]), report])
    )
    theta0 = np.concatenate([[logit(config["q0"])], logit(np.asarray(structure.s0, dtype=float))])
    field_fn = _field(structure, optimizer)
    start = time.perf_counter()
    traj = flows.integrate(
        field_fn,
        theta0,
        t_end,
        t_eval,
        rtol=solver["rtol"],
        atol=solver["atol"],
        method=solver["method"],
    )
    run = Run(structure, optimizer, traj, time.perf_counter() - start)
    rows = [observe(structure, theta, optimizer) for theta in traj.y.T]
    run.series = {"t": traj.t, "theta": traj.y, "q": expit(traj.y[0]), "s": expit(traj.y[1:])}
    run.series.update({k: np.array([r[k] for r in rows]) for k in OBSERVED})
    return run


# --- checks against the E003 pre-registration ---------------------------------------------------


def _outcome_class(q_end: float, fpr_end: float, tol: dict[str, float]) -> str:
    if q_end >= 1 - tol["success_gold"]:
        return "success"
    if fpr_end >= 1 - tol["stall_exploit"] and q_end < 0.999:
        return "stall"
    return "unresolved"


def check_static(runs: dict[str, Run], config: dict[str, Any], checks: Checks) -> None:
    q0, f, tol = config["q0"], config["f"], config["tolerances"]["static"]
    expected = {
        "gold": q0,
        "fpr": f,
        "fnr": 0.0,
        "verifier_accuracy": 1 - (1 - q0) * f,
        "fp_mass": (1 - q0) * f,
        "A_F": np.sqrt(q0 * (1 - q0)),
        "alpha_F": -f,
    }
    for key, value in expected.items():
        at_t0 = np.array([run.series[key][0] for run in runs.values()])
        checks.max_error(
            "all", "P1", f"{key}(0): max - min across structures", at_t0 - at_t0.min(), tol
        )
        checks.close("all", "P1", f"{key}(0) vs registered", at_t0, np.full_like(at_t0, value), tol)


def check_c0(runs: dict[str, Run], config: dict[str, Any], checks: Checks) -> None:
    c0 = []
    for name, run in runs.items():
        c0.append(run.series["C_F"][0])
        checks.close(
            name,
            "P2",
            "C(0)",
            c0[-1],
            config["predictions"][name]["C0"],
            config["tolerances"]["c0"],
        )
    increasing = bool(np.all(np.diff(c0) > 0))
    checks.add(
        "all",
        "P2",
        "C(0) strictly increasing in registered order",
        increasing,
        True,
        None,
        increasing,
    )


def check_run(run: Run, config: dict[str, Any], checks: Checks) -> None:
    st, ser, tol = run.structure, run.series, config["tolerances"]
    pred, q0, name = config["predictions"][st.name], config["q0"], st.name
    u0 = float(logit(q0))
    # P3a gold-race relation (trajectory level)
    if st.kind == "random":
        err = np.log(ser["q"]) - np.log(cf.random_fp_gold(u0, st.f or 0.0, ser["t"]))
    else:
        lam = np.array([cf.gold_race(st, st.s0, ser["s"][:, k]) for k in range(len(ser["t"]))])
        err = np.log(ser["q"]) - np.log(q0) - lam
    checks.max_error(name, "P3a", "max |log J_G - log q0 - Lambda(state)|", err, tol["gold_race"])
    # P3b outcome at T_end
    q_end, fpr_end = float(ser["gold"][-1]), float(ser["fpr"][-1])
    observed_class = _outcome_class(q_end, fpr_end, tol)
    expected_class = "stall" if pred["stall"] else "success"
    checks.add(name, "P3b", "outcome class at T_end", observed_class, expected_class, None,
               observed_class == expected_class)  # fmt: skip
    if pred["stall"]:
        checks.close(name, "P3b", "J_G(T_end) vs q_inf", q_end, pred["q_inf"], tol["stall_gold"])
    else:
        checks.close(
            name, "P3b", "FPR(T_end) vs S_inf", fpr_end, pred["S_inf"], tol["success_exploit"]
        )
    # P4 accessibility path
    eta = ser["eta_F"][np.isfinite(ser["eta_F"])]
    trend = pred["eta_trend"]
    if trend == "nondecreasing":
        worst = float(np.min(np.diff(eta)))
        checks.add(name, "P4", "min step of eta (>= -1e-8)", worst, trend, 1e-8, worst >= -1e-8)
    elif trend == "nonincreasing":
        worst = float(np.max(np.diff(eta)))
        checks.add(name, "P4", "max step of eta (<= 1e-8)", worst, trend, 1e-8, worst <= 1e-8)
    else:
        checks.max_error(name, "P4", "max |eta - eta(0)| (constant)", eta - eta[0], 1e-8)
    checks.info(name, "P4", "eta(0) (registered to 4 decimals)", [float(eta[0]), pred["eta0"]])
    # P5 clean-gap bound
    excess = ser["q"] - cf.clean_gold(u0, ser["t"])
    checks.add(name, "P5", "max (J_G - clean J_G) <= 1e-12", float(np.max(excess)), "<= 0", 1e-12,
               float(np.max(excess)) <= 1e-12)  # fmt: skip
    # P6 invariants
    steps = float(np.min(np.diff(ser["theta"], axis=1)))
    checks.add(
        name, "P6", "min logit step (non-decreasing)", steps, ">= -1e-10", 1e-10, steps >= -1e-10
    )
    if st.kind == "and":
        s = ser["s"]
        for j in range(1, st.n_features):
            keep = (1 - s[0] > 1e-9) & (1 - s[j] > 1e-9)
            ratio = (1 - s[0, keep]) / (1 - s[j, keep])
            checks.max_error(
                name,
                "P6",
                f"relative drift of (1-s1)/(1-s{j + 1})",
                ratio / ratio[0] - 1,
                tol["invariant"],
            )
    if st.kind in ("and", "or") and len(set(st.s0)) == 1:
        checks.max_error(
            name, "P6", "max |s_i - s_1| (symmetry)", ser["s"] - ser["s"][0], tol["invariant"]
        )


def check_ranking(runs: dict[str, Run], config: dict[str, Any], checks: Checks) -> None:
    tol, summary = config["tolerances"], config["predictions"]["summary"]
    names = list(runs)
    c0 = np.array([runs[n].series["C_F"][0] for n in names])
    q_end = np.array([runs[n].series["gold"][-1] for n in names])
    shortfall = np.where(q_end >= 1 - tol["success_gold"], 0.0, 1 - q_end)
    rho = spearman_with_ties(c0, shortfall)
    checks.close(
        "all", "P7", "Spearman(C(0), shortfall)", rho, summary["spearman_c0_shortfall"], 5e-4
    )
    discordant = sorted(
        sorted([names[i], names[j]])
        for i in range(len(names))
        for j in range(len(names))
        if c0[i] > c0[j] and shortfall[i] < shortfall[j] - 1e-6
    )
    expected = sorted(sorted(pair) for pair in summary["discordant_pairs"])
    checks.add(
        "all",
        "P7",
        "discordant (C(0), shortfall) pairs",
        discordant,
        expected,
        None,
        discordant == expected,
    )


def report(runs: dict[str, Run], config: dict[str, Any], checks: Checks, label: str) -> None:
    u0 = float(logit(config["q0"]))
    for name, run in runs.items():
        ser = run.series
        for t in config["solver"]["report_times"]:
            k = np.flatnonzero(np.isclose(ser["t"], t))
            if k.size:
                j = k[0]
                gap = float(cf.clean_gold(u0, t) - ser["gold"][j])
                checks.info(
                    name,
                    label,
                    f"(J_G, FPR, clean gap) at t={t:g}",
                    [ser["gold"][j], ser["fpr"][j], gap],
                )


def exploratory_vanilla(runs: dict[str, Run], config: dict[str, Any], checks: Checks) -> None:
    """E003-V: information rows only (no registered predictions, no pass/fail)."""
    q0, tol = config["q0"], config["tolerances"]
    for name, run in runs.items():
        st, ser = run.structure, run.series
        cls = _outcome_class(float(ser["gold"][-1]), float(ser["fpr"][-1]), tol)
        checks.info(name, "E003-V", "(J_G, FPR, class) at T_end",
                    [ser["gold"][-1], ser["fpr"][-1], cls])  # fmt: skip
        t0_keys = ("gold", "fpr", "A_F", "alpha_F", "C_F", "C_E")
        checks.info(name, "E003-V", f"t=0 {t0_keys}", [ser[k][0] for k in t0_keys])
        if st.kind != "random":
            lam = np.array([cf.gold_race(st, st.s0, ser["s"][:, k]) for k in range(len(ser["t"]))])
            dev = float(np.max(np.abs(np.log(ser["q"]) - np.log(q0) - lam)))
            checks.info(name, "E003-V", "natural-gradient gold-race relation: max deviation", dev)
        scale = np.maximum(np.abs(ser["gold_rate_actual"]), 1e-300)
        for metric in ("E", "F"):
            rel = float(
                np.max(np.abs(ser[f"gold_rate_{metric}"] - ser["gold_rate_actual"]) / scale)
            )
            checks.info(
                name, "E003-V", f"max rel. error of dJ_G/dt from {metric}-metric triple", rel
            )
        for t in config["solver"]["report_times"]:
            k = np.flatnonzero(np.isclose(ser["t"], t))
            if k.size:
                checks.info(
                    name, "E003-V", f"(J_G, FPR) at t={t:g}", [ser["gold"][k[0]], ser["fpr"][k[0]]]
                )


# --- outputs -------------------------------------------------------------------------------------

LINESTYLE = {"success": "-", "stall": "--", "unresolved": ":"}


def _observed_class(run: Run, config: dict[str, Any]) -> str:
    ser = run.series
    return _outcome_class(float(ser["gold"][-1]), float(ser["fpr"][-1]), config["tolerances"])


def _style(run: Run, config: dict[str, Any]) -> str:
    return LINESTYLE[_observed_class(run, config)]


def _mark_horizons(ax: Any, config: dict[str, Any], t_max: float) -> None:
    for t in config["solver"]["report_times"]:
        if t <= t_max:
            ax.axvline(t, color="#bbbbb6", lw=0.8, ls=":")


def _title(config: dict[str, Any], optimizer: str, what: str) -> str:
    tag = "REGISTERED (natural gradient)" if optimizer == "natural" else "EXPLORATORY (vanilla)"
    return f"{config['experiment_id']} {tag}: {what}"


def plot_gold(runs: dict[str, Run], config: dict[str, Any], optimizer: str, path: Path) -> None:
    """Figure A: J_G(t), clean-verifier reference, registered horizons."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    t = next(iter(runs.values())).series["t"][1:]
    for i, (name, run) in enumerate(runs.items()):
        ser = run.series
        label = f"{name} ({_observed_class(run, config)})"
        ax.plot(
            ser["t"][1:],
            ser["gold"][1:],
            color=PALETTE[i % 8],
            ls=_style(run, config),
            lw=1.6,
            label=label,
        )
    if optimizer == "natural":
        ax.plot(t, cf.clean_gold(float(logit(config["q0"])), t), color="#333333", ls="-.", lw=1.1,
                label="clean verifier V = G")  # fmt: skip
    _mark_horizons(ax, config, t[-1])
    ax.set(xscale="log", yscale="log", xlabel="t (log; dotted: registered horizons)",
           ylabel="gold J_G(t)")  # fmt: skip
    ax.grid(True, color="#e6e6e3", lw=0.6)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title(_title(config, optimizer, "J_G(t); solid = success, dashed = stall"), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_accessibility(
    runs: dict[str, Run], config: dict[str, Any], optimizer: str, path: Path
) -> None:
    """Figure B: C(t) (Fisher metric) and eta(t) = (C/C_max)^2."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for i, (name, run) in enumerate(runs.items()):
        ser = run.series
        kw = {"color": PALETTE[i % 8], "ls": _style(run, config), "lw": 1.5, "label": name}
        axes[0].plot(ser["t"][1:], np.where(ser["C_F"][1:] > 0, ser["C_F"][1:], np.nan), **kw)
        axes[1].plot(ser["t"][1:], ser["eta_F"][1:], **kw)
    axes[0].set(yscale="log", ylabel="C(t), Fisher metric")
    axes[1].set(ylabel="eta(t) = (C / C_max)^2  (undefined where C_max < 1e-6)")
    for ax in axes:
        ax.set(xscale="log", xlabel="t (log scale)")
        _mark_horizons(ax, config, next(iter(runs.values())).series["t"][-1])
        ax.grid(True, color="#e6e6e3", lw=0.6)
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(_title(config, optimizer, "accessibility"), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_state_space(
    runs: dict[str, Run], config: dict[str, Any], optimizer: str, path: Path
) -> None:
    """Figure C: J_G vs on-policy FPR (log-log; slope 1/eta under NG) and eta vs FPR."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for i, (name, run) in enumerate(runs.items()):
        ser = run.series
        kw = {"color": PALETTE[i % 8], "ls": _style(run, config), "lw": 1.5, "label": name}
        axes[0].plot(ser["fpr"], ser["gold"], **kw)
        axes[0].plot(ser["fpr"][0], ser["gold"][0], marker="o", ms=4, color=PALETTE[i % 8])
        axes[1].plot(ser["fpr"], ser["eta_F"], **kw)
    axes[0].set(xscale="log", yscale="log", xlabel="on-policy FPR S(t)", ylabel="gold J_G(t)")
    axes[1].set(xscale="log", xlabel="on-policy FPR S(t)", ylabel="eta = (C / C_max)^2")
    for ax in axes:
        ax.grid(True, color="#e6e6e3", lw=0.6)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(
        _title(config, optimizer, "state space (all runs start at FPR = f, J_G = q0)"), fontsize=9
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_c0_vs_shortfall(
    runs: dict[str, Run], config: dict[str, Any], optimizer: str, path: Path
) -> None:
    """Figure D: C(0) vs gold shortfall 1 - J_G(T_end); open markers = registered prediction."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for i, (name, run) in enumerate(runs.items()):
        ser = run.series
        c0 = ser["C_F"][0]
        if optimizer == "natural":
            ax.plot(c0, 1 - config["predictions"][name]["q_inf"], marker="o", mfc="none", ms=10,
                    color=PALETTE[i % 8])  # fmt: skip
        ax.plot(c0, 1 - ser["gold"][-1], marker="o", ms=5, color=PALETTE[i % 8])
        ax.annotate(name, (c0, 1 - ser["gold"][-1]), textcoords="offset points", xytext=(5, 4),
                    fontsize=7, color="#333333")  # fmt: skip
    ax.set(xlabel="C(0), Fisher metric (structures matched on static metrics, A, alpha)",
           ylabel="gold shortfall 1 - J_G(T_end)")  # fmt: skip
    ax.grid(True, color="#e6e6e3", lw=0.6)
    ax.set_title(_title(config, optimizer, "C(0) vs shortfall (open = predicted)"), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(checks: Checks, run_dir: Path, runs: dict[str, Run], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "| prediction | run | quantity | observed | expected | tol | pass |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in checks.rows:
        verdict = {True: "yes", False: "**NO**", None: "info"}[r["passed"]]
        lines.append(
            f"| {r['prediction']} | {r['run']} | {r['quantity']} | {_fmt(r['observed'])} | "
            f"{_fmt(r['expected'])} | {_fmt(r['tolerance'])} | {verdict} |"
        )
    lines += [
        "",
        "| run | optimizer | field evals | wall s | J_G(T) | FPR(T) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, run in runs.items():
        ser = run.series
        lines.append(
            f"| {name} | {run.optimizer} | {run.traj.n_evals} | {run.wall_seconds:.1f} | "
            f"{ser['gold'][-1]:.8g} | {ser['fpr'][-1]:.8g} |"
        )
    text = "\n".join(lines) + "\n"
    (run_dir / "summary.md").write_text(text)
    return text


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6g}"
    if isinstance(x, list):
        return "[" + ", ".join(_fmt(v) for v in x) + "]"
    return str(x)


def main(argv: list[str]) -> int:
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results-root", default=str(REPO / "results"))
    parser.add_argument("--optimizer", choices=("natural", "vanilla"), default="natural")
    args = parser.parse_args(argv[1:] if argv and argv[0].endswith(".py") else argv)
    config_path = Path(args.config).resolve()
    config = provenance.load_config(config_path)
    structures = [structure_from_config(entry, config["f"]) for entry in config["structures"]]
    registered = args.optimizer == "natural"
    experiment_id = config["experiment_id"] if registered else f"{config['experiment_id']}-V"
    run_dir = provenance.create_run_dir(Path(args.results_root), experiment_id, REPO)
    provenance.write_metadata(
        run_dir, experiment_id, config_path, REPO, extra={"optimizer": args.optimizer}
    )

    runs = {st.name: simulate(st, args.optimizer, config) for st in structures}
    for name, run in runs.items():
        print(f"simulated {name}/{run.optimizer}: {run.traj.n_evals} field evals, "
              f"{run.wall_seconds:.1f}s")  # fmt: skip

    checks = Checks()
    if registered:
        check_static(runs, config, checks)
        check_c0(runs, config, checks)
        for run in runs.values():
            check_run(run, config, checks)
        check_ranking(runs, config, checks)
        report(runs, config, checks, "O2/O4")
    else:
        exploratory_vanilla(runs, config, checks)

    (run_dir / "checks.json").write_text(json.dumps(checks.rows, indent=2) + "\n")
    arrays: dict[str, Any] = {
        f"{name}_{run.optimizer}_{key}": value
        for name, run in runs.items()
        for key, value in run.series.items()
    }
    np.savez_compressed(run_dir / "trajectories.npz", **arrays)
    plot_gold(runs, config, args.optimizer, run_dir / "fig_A_gold.png")
    plot_accessibility(runs, config, args.optimizer, run_dir / "fig_B_accessibility.png")
    plot_state_space(runs, config, args.optimizer, run_dir / "fig_C_state_space.png")
    plot_c0_vs_shortfall(runs, config, args.optimizer, run_dir / "fig_D_c0_vs_shortfall.png")
    title = f"{experiment_id} summary ({'REGISTERED' if registered else 'EXPLORATORY'})"
    print(write_summary(checks, run_dir, runs, title))
    failed = checks.failed()
    print(f"run directory: {run_dir}")
    print(f"{len(failed)} failed / {sum(r['passed'] is not None for r in checks.rows)} checked")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
