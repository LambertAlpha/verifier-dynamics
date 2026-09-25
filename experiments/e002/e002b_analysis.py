"""E002b held-out step 5 (registry E002 §7-§8, Amendment 2 §2-§4): endpoints, tests and verdict.

Reads an `e002b_heldout.py` run and the matching `heldout_targets.py` run. Primary: P-mod,
2 primary cells x {C-index, AUROC}, Delta = G1 - max_j P_j (P1-P4), one-sided kernel bootstrap,
Holm over the 4 tests. P-rare is secondary (unadjusted). Also reported: G1-oracle, pairwise
differences, per-arm CIs, frontier dominance, leave-one-type-out, the sign-flipped P4 (post hoc,
secondary) and every arm's costs. A design-split input is a DRY RUN and is labelled as such.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from vdyn import provenance
from vdyn.e002 import arms
from vdyn.e002 import endpoints as ep
from vdyn.e002 import verdict as vd

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e002" / "e002.toml"
FROZEN = REPO / "configs" / "e002" / "e002b_tuned_configs.json"
ENDPOINTS = {"c_index": ep.c_index, "auroc": ep.auroc}


def _spearman(score: np.ndarray, target: np.ndarray) -> float:
    return float(spearmanr(score, target).statistic)


def _recall10(score: np.ndarray, target: np.ndarray) -> float:
    return ep.recall_top(score, target, 0.10)


def _ci(boot: np.ndarray) -> list[float]:
    return np.nanquantile(boot, [0.025, 0.975]).tolist()


def analyse_op(
    op: str, op_i: int, scores: np.ndarray, meta: dict[str, Any], tdata: dict[str, Any],
    frozen: dict[str, Any], primary: list[tuple[int, int]], n_boot: int, seed: int,
) -> dict[str, Any]:  # fmt: skip
    cells = [tuple(c) for c in meta["cells"]]
    by_sid = {r["sid"]: r for r in tdata["targets"]}
    keep = np.array([not by_sid[s]["failed"] for s in meta[op]["sids"]])
    rows = [by_sid[s] for s, k in zip(meta[op]["sids"], keep, strict=True) if k]
    target = np.array([r["D"] for r in rows])
    stall = np.array([r["stall"] for r in rows])
    types = np.array([r["type"] for r in rows])
    tgt = {"c_index": target, "auroc": stall}
    arm_idx = {a: i for i, a in enumerate(meta["arms"])}

    def s(cell: tuple[int, int], arm: str, mask: np.ndarray | None = None) -> np.ndarray | None:
        x = scores[cells.index(cell), arm_idx[arm]][:, keep]
        if np.all(np.isnan(x)):
            return None
        return x if mask is None else x[:, mask]

    def sp(cell: tuple[int, int], arm: str, mask: np.ndarray | None = None) -> np.ndarray:
        x = s(cell, arm, mask)
        assert x is not None, f"{arm} is infeasible at primary cell {cell}"
        return x

    def delta(cell: tuple[int, int], arm: str, e: str) -> float:
        return point[(cell, arm, e)] - max(point[(cell, a, e)] for a in arms.COMPETITORS)

    # point estimates, every cell x arm (mean over replications of the panel metric)
    point: dict[tuple[Any, str, str], float] = {}
    table: dict[str, dict[str, Any]] = {}
    for cell in cells:
        for arm in arms.ARMS:
            x = s(cell, arm)
            if x is None:
                continue
            row = {e: ep.panel_metric(m, x, tgt[e]) for e, m in ENDPOINTS.items()}
            row |= {"kendall": ep.panel_metric(ep.kendall_tau_b, x, target),
                    "spearman": ep.panel_metric(_spearman, x, target),
                    "recall_top10": ep.panel_metric(_recall10, x, target)}  # fmt: skip
            f = frozen[f"{cell[0]},{cell[1]}|{arm}"]
            row |= {k: f[k] for k in ("cfg", "B_roll", "B_gold", "B_bwd", "k", "b")}
            table[f"{cell[0]},{cell[1]}|{arm}"] = row
            for e in ENDPOINTS:
                point[(cell, arm, e)] = row[e]
    # primary tests (and G1-oracle, pairwise, per-arm CIs) with shared bootstrap draws
    tests: dict[str, Any] = {}
    pvals: dict[tuple[tuple[int, int], str], float] = {}
    for c_i, cell in enumerate(primary):
        for e_i, (e, metric) in enumerate(ENDPOINTS.items()):
            comps = [sp(cell, a) for a in arms.COMPETITORS]
            draws = [seed, 20, op_i, c_i, e_i]  # the same resamples for every comparison
            res: dict[str, Any] = {}
            for arm in ("G1", "G1-oracle"):
                d = ep.paired_difference_bootstrap(metric, sp(cell, arm), comps, tgt[e], n_boot,
                                                   np.random.default_rng(draws))  # fmt: skip
                res[f"{arm}-maxP"] = {"point": delta(cell, arm, e), "ci95": _ci(d),
                                      "p_one_sided": ep.bootstrap_p_value(d)}  # fmt: skip
            for j, comp in zip(arms.COMPETITORS, comps, strict=True):
                d = ep.paired_difference_bootstrap(metric, sp(cell, "G1"), [comp], tgt[e], n_boot,
                                                   np.random.default_rng(draws))  # fmt: skip
                diff = point[(cell, "G1", e)] - point[(cell, j, e)]
                res[f"G1-{j}"] = {"point": diff, "ci95": _ci(d),
                                  "p_one_sided": ep.bootstrap_p_value(d)}  # fmt: skip
            for arm in arms.ARMS:
                boot = ep.hierarchical_bootstrap(metric, sp(cell, arm), tgt[e], n_boot,
                                                 np.random.default_rng(draws))  # fmt: skip
                res[f"{arm}|ci95"] = _ci(boot)
            tests[f"{cell[0]},{cell[1]}|{e}"] = res
            pvals[(cell, e)] = res["G1-maxP"]["p_one_sided"]
    keys = list(pvals)
    holm = dict(zip(keys, ep.holm([pvals[k] for k in keys], 0.05), strict=True))
    c_points = {(c, a): v for (c, a, e), v in point.items() if e == "c_index"}
    dominated = vd.dominated_cells(cells, c_points)
    verdict = vd.verdict(point, holm, dominated, primary, list(ENDPOINTS))
    # leave-one-type-out (Amendment 2 §4): point Delta with each type removed
    loto: dict[str, dict[str, float]] = {}
    for t in sorted(set(types)):
        mask = types != t
        for cell in primary:
            for e, metric in ENDPOINTS.items():
                vals = {a: ep.panel_metric(metric, sp(cell, a, mask), tgt[e][mask])
                        for a in ("G1", *arms.COMPETITORS)}  # fmt: skip
                loto.setdefault(f"{cell[0]},{cell[1]}|{e}", {})[t] = vals["G1"] - max(
                    vals[a] for a in arms.COMPETITORS
                )
    loto_range = {k: [min(v.values()), max(v.values())] for k, v in loto.items()}
    # [post-hoc, secondary] sign-flipped P4 (Amendment 2 §1); never enters the verdict
    flipped: dict[str, Any] = {}
    for cell in primary:
        for e, metric in ENDPOINTS.items():
            p4f = ep.panel_metric(metric, -sp(cell, "P4"), tgt[e])
            rivals = [point[(cell, a, e)] for a in ("P1", "P2", "P3")] + [p4f]
            flipped[f"{cell[0]},{cell[1]}|{e}"] = {"P4_flipped": p4f,
                                                   "G1-maxP_flippedP4": point[(cell, "G1", e)]
                                                   - max(rivals)}  # fmt: skip
    return {
        "n_structures": int(keep.sum()), "excluded": int((~keep).sum()),
        "stall_fraction": float(stall.mean()), "table": table, "tests": tests,
        "holm_reject": {f"{c[0]},{c[1]}|{e}": v for (c, e), v in holm.items()},
        "dominated": {f"{c[0]},{c[1]}": v for c, v in dominated.items()},
        "verdict": verdict, "loto_delta_range": loto_range, "loto": loto,
        "secondary_flipped_p4": flipped,
    }  # fmt: skip


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("targets_dir")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--results-root", default=str(REPO / "results"))
    args = ap.parse_args(argv[1:])
    config = provenance.load_config(CONFIG)
    n_boot = args.n_boot or config["e002b"]["bootstrap_heldout"]
    seed = config["e002b"]["heldout_seed"]
    run = Path(args.run_dir).resolve()
    meta = json.loads((run / "run.json").read_text())
    scores = np.load(run / "scores.npz")
    targets_dir = Path(args.targets_dir).resolve()
    split = meta["split"]
    frozen = json.loads(FROZEN.read_text())["configs"]
    primary = [tuple(c) for c in config["budgets"]["primary_cells"]]
    label = f"E002b-analysis-{split}"
    out_dir = provenance.create_run_dir(Path(args.results_root), label, REPO)
    extra = {"split": split, "scores_run": str(run), "targets": str(targets_dir), "n_boot": n_boot}
    provenance.write_metadata(out_dir, label, CONFIG, REPO, extra=extra)
    report: dict[str, Any] = {"split": split, "dry_run": split != "test", "n_boot": n_boot,
                              "reps": meta["reps"], "primary_cells": primary}  # fmt: skip
    for op_i, op in enumerate(config["operating_points"]):
        tdata = json.loads((targets_dir / f"targets_{op}.json").read_text())
        assert tdata["split"] == split, "targets and scores come from different splits"
        report[op] = analyse_op(op, op_i, scores[op], meta, tdata, frozen[op], primary, n_boot,
                                seed)  # fmt: skip
        v = report[op]["verdict"]
        tag = "PRIMARY" if config["operating_points"][op]["role"] == "primary" else "secondary"
        print(f"{op} [{tag}] verdict: {v['category']}  abandon={v['abandon']}  "
              f"holm={report[op]['holm_reject']}  delta_G1={v['delta_g1']}")  # fmt: skip
    if split != "test":
        print("DRY RUN on the design split: in-sample, not an E002 result.")
    (out_dir / "analysis.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"run directory: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
