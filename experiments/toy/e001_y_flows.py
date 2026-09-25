"""E001 — Y toy: vanilla vs natural gradient flows.

Pre-registration and Amendment 1: research/03_experiment_registry.md (E001). This script observes
trajectories of the generic (autodiff-driven) optimizers and compares them with the registered
numbers stored in the config. It does not derive predictions.

Usage:  uv run python experiments/toy/e001_y_flows.py [path/to/config.toml]
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import expit, logit

from vdyn import provenance
from vdyn.geometry import autodiff, closed_form
from vdyn.geometry.decompose import decompose
from vdyn.policies import bernoulli
from vdyn.simulation import flows
from vdyn.verifiers import toy

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "toy" / "e001_y_flows.toml"

GOLD = toy.reward_table(toy.gold)
Y = toy.reward_table(toy.y_false_positive)
ERROR = Y - GOLD  # e = V - G, so Delta = E_pi[e] without cancellation
OPTIMIZERS = ("natural", "vanilla")
FIELDS = {
    "natural": flows.natural_field(bernoulli.log_prob, Y),
    "vanilla": flows.vanilla_field(bernoulli.log_prob, Y),
}
CLOSED_RATES = {"natural": closed_form.natural_rates, "vanilla": closed_form.vanilla_rates}
COLORS = {"natural": "#2a78d6", "vanilla": "#eb6834"}  # categorical slots 1-2, validated
LINESTYLES = {"natural": "-", "vanilla": "--"}
MUTED = "#8a8a86"


# --- simulation ----------------------------------------------------------------------------------


@dataclass
class Run:
    ic: dict[str, Any]
    optimizer: str
    traj: flows.Trajectory
    wall_seconds: float
    series: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.ic['name']}/{self.optimizer}"

    def theta_at(self, t: float) -> np.ndarray:
        return np.asarray(self.traj.sol(t))


def gap(theta: np.ndarray) -> float:
    return autodiff.expected_reward(bernoulli.log_prob, theta, ERROR)


def observed_gap_rate(theta: np.ndarray, optimizer: str) -> float:
    grad_gap = autodiff.reward_gradient(bernoulli.log_prob, theta, ERROR)
    return float(grad_gap @ FIELDS[optimizer](0.0, theta))


def simulate(ic: dict[str, Any], optimizer: str, config: dict[str, Any]) -> Run:
    solver, t_end = config["solver"], config["horizon"][optimizer]
    t_eval = np.concatenate([[0.0], np.geomspace(solver["t_min"], t_end, solver["n_eval"])])
    theta0 = np.array([logit(ic["q0"]), logit(ic["s0"])])
    start = time.perf_counter()
    traj = flows.integrate(
        FIELDS[optimizer],
        theta0,
        t_end,
        t_eval,
        rtol=solver["rtol"],
        atol=solver["atol"],
        method=solver["method"],
    )
    run = Run(ic, optimizer, traj, time.perf_counter() - start)

    names = ("gap", "gap_rate", "A_F", "alpha_F", "b_F", "C_F", "A_E", "alpha_E", "b_E", "C_E")
    cols: dict[str, list[float]] = {name: [] for name in (*names, "pred_F", "pred_E")}
    for theta in traj.y.T:
        g_gold = autodiff.reward_gradient(bernoulli.log_prob, theta, GOLD)
        g_ver = autodiff.reward_gradient(bernoulli.log_prob, theta, Y)
        fisher = autodiff.fisher(bernoulli.log_prob, theta)
        dec_f = decompose(g_gold, g_ver, np.linalg.inv(fisher))
        dec_e = decompose(g_gold, g_ver, np.eye(2))
        cols["gap"].append(gap(theta))
        cols["gap_rate"].append(observed_gap_rate(theta, optimizer))
        for suffix, dec in (("F", dec_f), ("E", dec_e)):
            cols[f"A_{suffix}"].append(dec.A)
            cols[f"alpha_{suffix}"].append(dec.alpha)
            cols[f"b_{suffix}"].append(dec.b)
            cols[f"C_{suffix}"].append(dec.C)
            cols[f"pred_{suffix}"].append(dec.gap_rate)  # Proposition 1 prediction in that metric
    run.series = {"t": traj.t, "u": traj.y[0], "v": traj.y[1]}
    run.series["q"], run.series["s"] = expit(traj.y[0]), expit(traj.y[1])
    run.series.update({name: np.array(values) for name, values in cols.items()})
    return run


# --- checks --------------------------------------------------------------------------------------


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        run: str,
        prediction: str,
        quantity: str,
        observed: Any,
        expected: Any,
        tolerance: Any,
        passed: bool | None,
    ) -> None:
        self.rows.append(
            {
                "run": run,
                "prediction": prediction,
                "quantity": quantity,
                "observed": _plain(observed),
                "expected": _plain(expected),
                "tolerance": _plain(tolerance),
                "passed": passed,
            }
        )

    def close(self, run, prediction, quantity, observed, expected, tolerance) -> None:
        err = float(np.max(np.abs(np.asarray(observed) - np.asarray(expected))))
        self.add(run, prediction, quantity, observed, expected, tolerance, err <= tolerance)

    def max_error(self, run, prediction, quantity, errors, tolerance) -> None:
        worst = float(np.max(np.abs(errors)))
        self.add(run, prediction, quantity, worst, 0.0, tolerance, worst <= tolerance)

    def info(self, run, prediction, quantity, observed) -> None:
        self.add(run, prediction, quantity, observed, None, None, None)


def _plain(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, list | tuple):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def sign_pattern(rates: np.ndarray, threshold: float) -> tuple[str, list[tuple[int, int]]]:
    """Compressed sign sequence ignoring |rate| < threshold, and index pairs bracketing switches."""
    idx = np.flatnonzero(np.abs(rates) >= threshold)
    signs = np.sign(rates[idx])
    pattern = "".join(
        "+" if sg > 0 else "-" for k, sg in enumerate(signs) if k == 0 or sg != signs[k - 1]
    )
    switches = [
        (int(idx[k - 1]), int(idx[k])) for k in range(1, len(idx)) if signs[k] != signs[k - 1]
    ]
    return pattern, switches


def _crossing_time(run: Run, fn, lo_idx: int, hi_idx: int) -> float:
    t = run.series["t"]
    return float(brentq(fn, t[lo_idx], t[hi_idx], xtol=1e-14, rtol=1e-13))


def check_p1(run: Run, config: dict[str, Any], checks: Checks) -> None:
    ser, solver = run.series, config["solver"]
    rates = CLOSED_RATES[run.optimizer]
    ref = solve_ivp(
        lambda _t, y: rates(y[0], y[1]),
        (0.0, ser["t"][-1]),
        [run.ic["q0"], run.ic["s0"]],
        t_eval=ser["t"],
        method=solver["method"],
        rtol=solver["rtol"],
        atol=solver["atol"],
    )
    err = np.concatenate([ser["q"] - ref.y[0], ser["s"] - ref.y[1]])
    checks.max_error(
        run.key,
        "P1",
        "max |generic - closed-form ODE| over (q, s)",
        err,
        config["tolerances"]["trajectory"],
    )


def check_p2(run: Run, config: dict[str, Any], checks: Checks) -> None:
    ser, tol, pred = run.series, config["tolerances"], config["predictions"][run.ic["name"]]
    if run.optimizer == "natural":
        ratio = closed_form.natural_invariant(ser["q"], ser["s"])
        err = ratio - run.ic["s0"] / run.ic["q0"]
        checks.max_error(run.key, "P2", "max |s/q - s0/q0|", err, tol["natural_invariant"])
    else:
        inv = closed_form.vanilla_invariant(ser["u"], ser["v"])
        checks.max_error(run.key, "P2", "max |I(t) - I(0)|", inv - inv[0], tol["vanilla_invariant"])
        checks.close(run.key, "P2", "I(0) vs registered", inv[0], pred["vanilla_invariant_0"], 1e-6)


def check_p3(run: Run, config: dict[str, Any], checks: Checks) -> None:
    ser, tol, pred = run.series, config["tolerances"], config["predictions"][run.ic["name"]]
    if "natural_endpoint" in pred:
        final = [ser["q"][-1], ser["s"][-1]]
        checks.close(run.key, "P3", "(q, s) at T", final, pred["natural_endpoint"], tol["endpoint"])
        checks.close(
            run.key, "P3", "Delta at T", ser["gap"][-1], pred["delta_inf"], tol["endpoint"]
        )
    else:
        checks.max_error(
            run.key, "P3", "max |s - q| (diagonal)", ser["s"] - ser["q"], tol["endpoint"]
        )
        monotone = bool(np.all(np.diff(ser["q"]) >= 0) and np.all(np.diff(ser["s"]) >= 0))
        checks.add(run.key, "P3", "q, s non-decreasing", monotone, True, None, monotone)
        checks.info(
            run.key,
            "P3",
            "(q, s, Delta) at T (limit not reached)",
            [ser["q"][-1], ser["s"][-1], ser["gap"][-1]],
        )


def check_p4(run: Run, config: dict[str, Any], checks: Checks) -> None:
    ser, tol, pred = run.series, config["tolerances"], config["predictions"][run.ic["name"]]
    if "vanilla_clock_free" not in pred:
        checks.max_error(
            run.key, "P4", "max |s - q| (diagonal)", ser["s"] - ser["q"], tol["clock_free"]
        )
        return
    coord, target, other_expected = pred["vanilla_clock_free"]
    at, other = (0, 1) if coord == "q" else (1, 0)
    values = expit(ser["u"] if at == 0 else ser["v"])
    hi = int(np.argmax(values >= target))
    if values[hi] < target:
        checks.add(run.key, "P4", f"{coord} reaches {target}", False, True, None, False)
        return

    def reach(t: float) -> float:
        return float(expit(run.theta_at(t)[at]) - target)

    t_star = _crossing_time(run, reach, hi - 1, hi)
    other_value = float(expit(run.theta_at(t_star)[other]))
    name = "s" if other == 1 else "q"
    checks.close(
        run.key,
        "P4",
        f"{name} when {coord} = {target} (t = {t_star:.4g})",
        other_value,
        other_expected,
        tol["clock_free"],
    )
    checks.info(
        run.key,
        "P4",
        f"natural-flow {name} at the same {coord}",
        target * (min(run.ic["s0"] / run.ic["q0"], run.ic["q0"] / run.ic["s0"])),
    )


def check_p5(run: Run, config: dict[str, Any], checks: Checks) -> None:
    ser, tol, pred = run.series, config["tolerances"], config["predictions"][run.ic["name"]]
    pattern, switches = sign_pattern(ser["gap_rate"], tol["gap_rate_zero"])
    expected = pred[f"{run.optimizer}_gap_signs"]
    checks.add(
        run.key, "P5", "sign pattern of dDelta/dt", pattern, expected, None, pattern == expected
    )
    no_regrowth = "-+" not in pattern
    checks.add(
        run.key, "P5", "no shrinking -> growing switch", no_regrowth, True, None, no_regrowth
    )
    key = f"{run.optimizer}_switch"
    if key in pred and switches:
        lo, hi = switches[0]
        t_star = _crossing_time(
            run, lambda t: observed_gap_rate(run.theta_at(t), run.optimizer), lo, hi
        )
        qs = expit(run.theta_at(t_star))
        checks.close(
            run.key,
            "P5",
            f"(q, s) at first sign switch (t = {t_star:.4g})",
            qs,
            pred[key],
            tol["switch_point"],
        )


def check_p6(run: Run, config: dict[str, Any], checks: Checks) -> None:
    ser, tol = run.series, config["tolerances"]
    q, s = ser["q"], ser["s"]
    pairs = {
        "A_F": closed_form.signal(q, s),
        "alpha_F": closed_form.alpha(q, s),
        "C_F": closed_form.pressure(q, s),
        "A_E": closed_form.euclidean_signal(q, s),
        "alpha_E": closed_form.alpha(q, s),
        "C_E": closed_form.euclidean_pressure(q, s),
    }
    for name, closed in pairs.items():
        ratio = np.abs(ser[name] - closed) / (
            tol["diagnostic_atol"] + tol["diagnostic_rtol"] * np.abs(closed)
        )
        worst = float(np.max(ratio))
        checks.add(
            run.key, "P6", f"{name}: max |err| / (atol + rtol|x|)", worst, "<= 1", 1.0, worst <= 1.0
        )
    checks.info(
        run.key,
        "P6",
        "(alpha_F, C_F, C_E) at T",
        [ser["alpha_F"][-1], ser["C_F"][-1], ser["C_E"][-1]],
    )


def check_p7(run: Run, config: dict[str, Any], checks: Checks) -> None:
    ser, tol = run.series, config["tolerances"]
    pred = ser["pred_F"] if run.optimizer == "natural" else ser["pred_E"]
    t = ser["t"]
    fd = np.full_like(t, np.nan)
    for k in range(1, len(t) - 1):
        h = max(1e-6, 1e-4 * t[k])
        fd[k] = (gap(run.theta_at(t[k] + h)) - gap(run.theta_at(t[k] - h))) / (2 * h)
    inner = slice(1, len(t) - 1)
    allowed = tol["gap_identity_rtol"] * np.abs(pred[inner]) + tol[
        "gap_identity_atol_frac"
    ] * np.max(np.abs(pred[inner]))
    worst = float(np.max(np.abs(fd[inner] - pred[inner]) / allowed))
    metric = "Fisher" if run.optimizer == "natural" else "Euclidean"
    checks.add(
        run.key,
        "P7",
        f"FD dDelta/dt vs b(a+b)+c^2 ({metric}): max |err| / allowed",
        worst,
        "<= 1",
        1.0,
        worst <= 1.0,
    )
    if run.optimizer == "vanilla":
        disagree = float(np.mean(np.sign(ser["pred_F"]) != np.sign(ser["gap_rate"])))
        checks.info(run.key, "P7", "fraction of points where Fisher-metric sign is wrong", disagree)
        if run.ic["name"] == "IC1":
            opposite = bool(np.sign(ser["pred_F"][0]) == -np.sign(ser["gap_rate"][0]))
            checks.add(
                run.key,
                "P7",
                "Fisher-metric prediction at t=0 has opposite sign",
                [ser["pred_F"][0], ser["gap_rate"][0]],
                "opposite signs",
                None,
                opposite,
            )


def evaluate(runs: dict[tuple[str, str], Run], config: dict[str, Any]) -> Checks:
    checks = Checks()
    for run in runs.values():
        for check in (check_p1, check_p2, check_p5, check_p6, check_p7):
            check(run, config, checks)
        (check_p3 if run.optimizer == "natural" else check_p4)(run, config, checks)
    checks.rows.sort(key=lambda r: (r["prediction"], r["run"]))
    return checks


# --- outputs -------------------------------------------------------------------------------------


ROWS = [
    ("q", "q = P(corr=1)", "linear"),
    ("s", "s = P(z=1)", "linear"),
    ("gap", "Δ = E[V − G]", "linear"),
    ("alpha_F", "α  (= −s, both metrics)", "linear"),
    ("C_F", "C, Fisher metric", "log"),
    ("C_E", "C, Euclidean metric", "log"),
    ("gap_rate", "dΔ/dt", "symlog"),
]


def plot_timeseries(runs: dict[tuple[str, str], Run], config: dict[str, Any], path: Path) -> None:
    ics = config["initial_conditions"]
    fig, axes = plt.subplots(
        len(ROWS), len(ics), figsize=(4.4 * len(ics), 1.7 * len(ROWS)), sharex=True
    )
    for j, ic in enumerate(ics):
        for opt in OPTIMIZERS:
            ser = runs[(ic["name"], opt)].series
            for i, (key, _, scale) in enumerate(ROWS):
                y = ser[key][1:]
                if scale == "log":
                    y = np.where(y > 0, y, np.nan)
                axes[i, j].plot(
                    ser["t"][1:], y, color=COLORS[opt], ls=LINESTYLES[opt], lw=1.6, label=opt
                )
        axes[0, j].set_title(
            f"{ic['name']} ({ic['label']}): q0={ic['q0']}, s0={ic['s0']}", fontsize=10
        )
        for i, (_, label, scale) in enumerate(ROWS):
            ax = axes[i, j]
            ax.set_xscale("log")
            if scale == "symlog":
                ax.set_yscale("symlog", linthresh=1e-8)
                ax.axhline(0.0, color=MUTED, lw=0.8)
            elif scale == "log":
                ax.set_yscale("log")
            ax.grid(True, color="#e6e6e3", lw=0.6)
            ax.tick_params(labelsize=8)
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
        axes[-1, j].set_xlabel("t (log scale; clocks differ across optimizers)", fontsize=9)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "E001 — Y toy, exact expected gradients (natural: solid, vanilla: dashed)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_phase(runs: dict[tuple[str, str], Run], config: dict[str, Any], path: Path) -> None:
    lo, hi = 1e-3, 1 - 1e-7
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.set_xscale("logit")
    ax.set_yscale("logit")
    ax.axvline(0.5, color=MUTED, lw=1.0, ls=":", label="natural: dΔ/dt = 0 at q = 1/2")
    s_grid = expit(np.linspace(logit(lo), logit(hi), 800))
    ax.plot(
        np.sqrt(s_grid * (1 - s_grid)),
        s_grid,
        color=MUTED,
        lw=1.0,
        ls="-.",
        label="vanilla: dΔ/dt = 0 at s(1−s) = q²",
    )
    for ic in config["initial_conditions"]:
        for opt in OPTIMIZERS:
            ser = runs[(ic["name"], opt)].series
            q, s = np.clip(ser["q"], lo, hi), np.clip(ser["s"], lo, hi)
            ax.plot(
                q,
                s,
                color=COLORS[opt],
                ls=LINESTYLES[opt],
                lw=1.6,
                label=opt if ic is config["initial_conditions"][0] else None,
            )
            ax.plot(q[-1], s[-1], marker="s", ms=7, color=COLORS[opt], mec="white", mew=1.0)
        ax.plot(ic["q0"], ic["s0"], marker="o", ms=8, color="#333333", mec="white", mew=1.0)
        ax.annotate(
            ic["name"],
            (ic["q0"], ic["s0"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            color="#333333",
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("q = P(corr = 1)   (logit axis)")
    ax.set_ylabel("s = P(z = 1)   (logit axis)")
    ax.set_title("E001 — clock-free paths (circle: start, square: end of horizon)", fontsize=10)
    ax.grid(True, color="#e6e6e3", lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(runs, checks: Checks, run_dir: Path) -> str:
    lines = [
        "# E001 summary",
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
        "| run | field evals | wall s | q(T) | s(T) | Δ(T) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs.values():
        ser = run.series
        lines.append(
            f"| {run.key} | {run.traj.n_evals} | {run.wall_seconds:.1f} | {ser['q'][-1]:.10g} | "
            f"{ser['s'][-1]:.10g} | {ser['gap'][-1]:.10g} |"
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
    config_path = Path(argv[1]).resolve() if len(argv) > 1 else DEFAULT_CONFIG
    config = provenance.load_config(config_path)
    run_dir = provenance.create_run_dir(REPO / "results", config["experiment_id"], REPO)
    provenance.write_metadata(run_dir, config["experiment_id"], config_path, REPO)

    runs: dict[tuple[str, str], Run] = {}
    for ic in config["initial_conditions"]:
        for opt in OPTIMIZERS:
            run = simulate(ic, opt, config)
            runs[(ic["name"], opt)] = run
            print(f"simulated {run.key}: {run.traj.n_evals} field evals, {run.wall_seconds:.1f}s")

    checks = evaluate(runs, config)
    (run_dir / "checks.json").write_text(json.dumps(checks.rows, indent=2) + "\n")
    arrays: dict[str, Any] = {
        f"{ic}_{opt}_{name}": v for (ic, opt), run in runs.items() for name, v in run.series.items()
    }
    np.savez_compressed(run_dir / "trajectories.npz", **arrays)
    plot_timeseries(runs, config, run_dir / "fig_timeseries.png")
    plot_phase(runs, config, run_dir / "fig_phase.png")
    print(write_summary(runs, checks, run_dir))

    failed = [r for r in checks.rows if r["passed"] is False]
    print(f"run directory: {run_dir.relative_to(REPO)}")
    print(f"{len(failed)} failed / {sum(r['passed'] is not None for r in checks.rows)} checked")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
