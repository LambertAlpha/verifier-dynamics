"""E002b DESIGN-SPLIT pilot (registry E002 §5, §6, §11): per-cell tuning, cost accounting, runtime.

Only `load_split(..., "design")` is called; no test-split structure, target or score exists here.
Every metric below is IN-SAMPLE (tuned and evaluated on the same 300 design structures), so it is
optimistic for every arm and is NOT an E002b result. A replication-split variant (tune on the first
half of the replications, evaluate on the second) is reported to size the selection optimism.
"""

import itertools
import json
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logit

from vdyn import provenance
from vdyn.e002 import endpoints as ep
from vdyn.e002 import estimators as est
from vdyn.e002 import panel as pn
from vdyn.e002 import probes as pr

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e002" / "e002.toml"
ARMS = ("G0", "G1", "G1-oracle", "P1", "P2", "P3", "P4")
COMPETITORS = ("P1", "P2", "P3", "P4")
N_BOOT_PILOT = 200


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


def simplicity(cfg: dict[str, Any]) -> tuple[float, ...]:
    """Registered tie-break order (§6): smaller k, smaller eta_p, plug-in, smaller lam, paired,
    smaller r. P3's observable is not in the registered order; FPR (grid order) comes first."""
    r = cfg.get("r", 0)
    return (
        cfg.get("k", 0),
        cfg.get("eta", 0.0),
        {"plugin": 0, "ustat": 1}.get(cfg.get("estimator", "plugin"), 0),
        cfg.get("lam", 0.0),
        {"paired": 0, "pooled": 1}.get(cfg.get("source", "paired"), 0),
        99 if r == "all" else r,
        {"fpr": 0, "gold": 1}.get(cfg.get("observable", "fpr"), 0),
    )


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


def _exact_fisher(q0: float, s0: np.ndarray) -> np.ndarray:
    """Fisher of the product-Bernoulli policy (diagonal): the G1-oracle metric only."""
    p = np.concatenate([[q0], s0])
    return np.diag(p * (1 - p))


def structure_task(args: tuple[Any, ...]) -> dict[str, Any]:
    """All cells x arms x configurations for one design structure."""
    op_i, s_idx, slot, f, q0, cells, reps, seed = args
    st = pn.match(slot, f)
    s0 = np.asarray(st.s0, dtype=np.float64)
    theta0 = np.concatenate([[logit(q0)], logit(s0)])
    scores: dict[str, list[float]] = {}
    seconds: dict[str, float] = {}
    for c_i, (b_gold, b_roll) in enumerate(cells):
        for a_i, arm in enumerate(ARMS):
            start = time.perf_counter()
            cfgs = configs(arm)
            if arm in ("G1", "G1-oracle"):  # common data for every G1 configuration
                rng = np.random.default_rng([seed, 5, op_i, s_idx, c_i, 1])
                lab = est.sample_rollouts(st, q0, s0, b_gold, reps, rng)
                unl = est.sample_rollouts(st, q0, s0, b_roll - b_gold, reps, rng)
                exact = _exact_fisher(q0, s0) if arm == "G1-oracle" else None
                for g_i, cfg in enumerate(cfgs):
                    out = est.geometry(st, lab, unl, q0, s0, cfg["estimator"], cfg["lam"],
                                       cfg["source"], exact_fisher=exact)  # fmt: skip
                    scores[f"{c_i}|{arm}|{g_i}"] = out["C2"].tolist()
            else:
                for g_i, cfg in enumerate(cfgs):
                    p = plan(arm, cfg, b_gold, b_roll)
                    if not p.feasible:
                        continue
                    rng = np.random.default_rng([seed, 5, op_i, s_idx, c_i, a_i, g_i])
                    if arm == "G0":
                        out = pr.run_g0(st, theta0, b_gold, reps, rng)
                    elif arm == "P1":
                        out = pr.run_p1(st, theta0, p.b, cfg["k"], cfg["eta"], reps, rng)
                    elif arm == "P2":
                        out = pr.run_p2(st, theta0, b_gold, p.b, cfg["k"], cfg["eta"], reps, rng,
                                        reuse_audit=p.reuse_audit)  # fmt: skip
                    elif arm == "P3":
                        out = pr.run_p3(st, theta0, q0, b_gold, p.b, cfg["k"], cfg["eta"],
                                        cfg["observable"], reps, rng)  # fmt: skip
                    else:
                        out = pr.run_p4(st, theta0, b_gold, b_roll - b_gold, cfg["r"], reps, rng)
                    scores[f"{c_i}|{arm}|{g_i}"] = np.asarray(out["score"], dtype=float).tolist()
            seconds[f"{c_i}|{arm}"] = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"sid": slot["sid"], "scores": scores, "seconds": seconds, "max_rss": rss}


def _metrics(sc: np.ndarray, target: np.ndarray, stall: np.ndarray) -> dict[str, float]:
    return {
        "c_index": ep.panel_metric(ep.c_index, sc, target),
        "auroc": ep.panel_metric(ep.auroc, sc, stall),
    }


def tune(
    arm: str, c_i: int, cell: tuple[int, int], data: dict[str, np.ndarray], target: np.ndarray,
    stall: np.ndarray, reps: int
) -> dict[str, Any] | None:  # fmt: skip
    b_gold, b_roll = cell
    rows: list[dict[str, Any]] = []
    for g_i, cfg in enumerate(configs(arm)):
        key = f"{c_i}|{arm}|{g_i}"
        if key not in data:
            continue
        sc = data[key]
        half = reps // 2
        rows.append({
            "g_i": g_i, "cfg": cfg, "c_index": ep.panel_metric(ep.c_index, sc, target),
            "c_first_half": ep.panel_metric(ep.c_index, sc[:half], target),
            "c_second_half": ep.panel_metric(ep.c_index, sc[half:], target),
        })  # fmt: skip
    if not rows:
        return None
    best = min(rows, key=lambda r: (-r["c_index"], simplicity(r["cfg"])))
    honest = min(rows, key=lambda r: (-r["c_first_half"], simplicity(r["cfg"])))
    p = plan(arm, best["cfg"], b_gold, b_roll)
    sc = data[f"{c_i}|{arm}|{best['g_i']}"]
    return {
        "cfg": best["cfg"],
        "g_i": best["g_i"],
        **_metrics(sc, target, stall),
        "c_index_repsplit": honest["c_second_half"],
        "cfg_repsplit": honest["cfg"],
        "n_configs": len(rows),
        "B_roll": p.b_roll, "B_gold": p.b_gold, "B_bwd": p.b_bwd, "k": p.k, "b": p.b,
        "all_configs": [{"cfg": r["cfg"], "c_index": r["c_index"]} for r in rows],
    }  # fmt: skip


def main(argv: list[str]) -> int:
    targets_dir = Path(argv[1]).resolve()
    config = provenance.load_config(CONFIG)
    reps = config["e002b"]["reps_design"]
    seed = config["pilot_seed"]
    cells = [(bg, bg * r) for bg in config["budgets"]["b_gold"]
             for r in config["budgets"]["roll_ratio"]]  # fmt: skip
    primary = [tuple(c) for c in config["budgets"]["primary_cells"]]
    run_dir = provenance.create_run_dir(REPO / "results", "E002b-design-pilot", REPO)
    extra = {"split": "design", "reps": reps, "targets": str(targets_dir.relative_to(REPO))}
    provenance.write_metadata(run_dir, "E002b-design-pilot", CONFIG, REPO, extra=extra)
    report: dict[str, Any] = {"cells": cells, "primary_cells": primary, "reps": reps}
    wall_total = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for op_i, (op, spec) in enumerate(config["operating_points"].items()):
            q0 = spec["q0"]
            tdata = json.loads((targets_dir / f"design_targets_{op}.json").read_text())
            f = tdata["f_selected"]
            slots = pn.load_split(REPO / "configs" / "e002" / f"panel_{op}.json", "design")
            by_sid = {r["sid"]: r for r in tdata["targets"]}
            target = np.array([by_sid[s["sid"]]["D"] for s in slots])
            stall = np.array([by_sid[s["sid"]]["stall"] for s in slots])
            start = time.perf_counter()
            tasks = [(op_i, i, s, f, q0, cells, reps, seed) for i, s in enumerate(slots)]
            results = list(pool.map(structure_task, tasks, chunksize=1))
            wall = time.perf_counter() - start
            keys = results[0]["scores"].keys()
            data = {k: np.array([r["scores"][k] for r in results]).T for k in keys}
            seconds: dict[str, float] = {}
            for r in results:
                for k, v in r["seconds"].items():
                    seconds[k] = seconds.get(k, 0.0) + v
            tuned: dict[str, dict[str, Any]] = {}
            for c_i, cell in enumerate(cells):
                for arm in ARMS:
                    res = tune(arm, c_i, cell, data, target, stall, reps)
                    if res is not None:
                        res["cpu_seconds"] = seconds[f"{c_i}|{arm}"]
                        tuned[f"{cell[0]},{cell[1]}|{arm}"] = res
            # bootstrap CIs (design, in-sample) and G1 - max(P) at the primary cells
            rng = np.random.default_rng([seed, 6, op_i])
            primary_out: dict[str, Any] = {}
            for cell in primary:
                c_i = cells.index(cell)
                sc = {arm: data[f"{c_i}|{arm}|{tuned[f'{cell[0]},{cell[1]}|{arm}']['g_i']}"]
                      for arm in ARMS if f"{cell[0]},{cell[1]}|{arm}" in tuned}  # fmt: skip
                comps = [sc[a] for a in COMPETITORS if a in sc]
                cell_out: dict[str, Any] = {}
                for name, metric, tgt in (("c_index", ep.c_index, target),
                                          ("auroc", ep.auroc, stall)):  # fmt: skip
                    for arm in ("G1", "G1-oracle"):
                        diff = ep.paired_difference_bootstrap(metric, sc[arm], comps, tgt,
                                                              N_BOOT_PILOT, rng)  # fmt: skip
                        point = ep.panel_metric(metric, sc[arm], tgt) - max(
                            ep.panel_metric(metric, c, tgt) for c in comps
                        )
                        cell_out[f"{name}|{arm}-maxP"] = {
                            "point": point,
                            "ci95": np.quantile(diff, [0.025, 0.975]).tolist(),
                            "p_one_sided": float(np.mean(diff <= 0)),
                        }
                    for arm, s in sc.items():
                        boot = ep.hierarchical_bootstrap(metric, s, tgt, N_BOOT_PILOT, rng)
                        cell_out[f"{name}|{arm}|ci95"] = np.quantile(boot, [0.025, 0.975]).tolist()
                primary_out[f"{cell[0]},{cell[1]}"] = cell_out
            # frontier dominance on nominal cells (design, in-sample C-index)
            dominated = []
            for cell in cells:
                g1 = tuned[f"{cell[0]},{cell[1]}|G1"]["c_index"]
                dominated.append(any(
                    tuned[f"{c[0]},{c[1]}|{a}"]["c_index"] >= g1
                    for c in cells if c[0] <= cell[0] and c[1] <= cell[1]
                    for a in COMPETITORS if f"{c[0]},{c[1]}|{a}" in tuned
                ))  # fmt: skip
            t0 = time.perf_counter()
            for _ in range(200):
                ep.c_index(data[next(iter(data))][0], target, np.ones(len(target)))
            c_index_call = (time.perf_counter() - t0) / 200
            report[op] = {
                "q0": q0, "f": f, "n_structures": len(slots), "stall_fraction": float(stall.mean()),
                "wall_seconds": wall, "cpu_seconds": float(sum(seconds.values())),
                "max_rss_bytes_worker": max(r["max_rss"] for r in results),
                "c_index_call_seconds_n300": c_index_call,
                "tuned": tuned, "primary": primary_out,
                "g1_dominated_by_cell": dict(zip([f"{c[0]},{c[1]}" for c in cells], dominated,
                                                 strict=True)),
                "g1_frontier_uniformly_dominated": all(dominated),
            }  # fmt: skip
            print(f"\n{op}: wall {wall:.0f} s, stall fraction {stall.mean():.3f}")
            for cell in cells:
                line = "  ".join(
                    f"{a} {tuned[f'{cell[0]},{cell[1]}|{a}']['c_index']:.3f}"
                    if f"{cell[0]},{cell[1]}|{a}" in tuned else f"{a}   -  "
                    for a in ARMS
                )  # fmt: skip
                print(f"  ({cell[0]:4d},{cell[1]:5d})  {line}")
    report["wall_seconds_total"] = time.perf_counter() - wall_total
    (run_dir / "e002b_design_pilot.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"run directory: {run_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
