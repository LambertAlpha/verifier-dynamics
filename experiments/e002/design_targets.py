"""E002 pilot stage 1 (DESIGN SPLIT ONLY): exact targets, FPR selection rule, ODE cross-check.

Registry E002 §1 (FPR rule), §4 (targets, QA) and Amendment 1. Test-split structures are never
loaded (`load_split(..., "design")`).
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit, logit

from vdyn import provenance
from vdyn.e002 import panel as pn
from vdyn.geometry.gold_race import gold_race
from vdyn.policies import bernoulli
from vdyn.simulation import flows
from vdyn.verifiers import boolean_fp as bf

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e002" / "e002.toml"


def _target(args: tuple[dict[str, Any], float, float, tuple[float, ...]]) -> dict[str, Any]:
    slot, f, q0, report = args
    st = pn.match(slot, f)
    r = gold_race(st, q0, report_times=report)
    return {
        "sid": slot["sid"],
        "type": slot["type"],
        "base": slot["base"],
        "rho": slot["rho"],
        "s0": list(st.s0),
        "coin": st.coin,
        "stall": r.stall,
        "D": 1 - min(r.q_inf, 1.0),
        "q_inf": r.q_inf,
        "fpr_inf": r.fpr_inf,
        "t95": r.t95,
        "jg_at": {str(k): v for k, v in r.jg_at.items()},
        "fpr_at": {str(k): v for k, v in r.fpr_at.items()},
    }


def _qa(args: tuple[dict[str, Any], float, float, float, dict[str, Any]]) -> dict[str, Any]:
    slot, f, q0, t_end, target = args
    st = pn.match(slot, f)
    lp = bernoulli.product_log_prob(1 + st.n_features)
    theta0 = np.concatenate([[logit(q0)], logit(np.asarray(st.s0))])
    sol = flows.integrate(flows.natural_field(lp, bf.verifier_table(st)), theta0, t_end)
    q_end = float(expit(sol.y[0, -1]))
    fpr_end = bf.fpr(st, expit(sol.y[1:, -1])) if st.n_features else st.coin
    if target["stall"]:
        ok = abs(q_end - target["q_inf"]) <= 2e-3
        err = abs(q_end - target["q_inf"])
    else:
        ok = abs(fpr_end - target["fpr_inf"]) <= 1e-4
        err = abs(fpr_end - target["fpr_inf"])
    cls_ode = "stall" if (q_end < 1 - 1e-6 and 1 - fpr_end < 1e-2) else "success"
    return {
        "sid": slot["sid"],
        "stall_target": target["stall"],
        "q_end": q_end,
        "fpr_end": fpr_end,
        "abs_err": err,
        "tolerance_ok": ok,
        "ode_class": cls_ode,
    }


def main() -> int:
    config = provenance.load_config(CONFIG)
    run_dir = provenance.create_run_dir(REPO / "results", "E002-pilot-targets", REPO)
    provenance.write_metadata(
        run_dir, "E002-pilot-targets", CONFIG, REPO, extra={"split": "design"}
    )
    report = tuple(config["targets"]["report_times"])
    summary: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for op, spec in config["operating_points"].items():
            q0 = spec["q0"]
            slots = pn.load_split(REPO / "configs" / "e002" / f"panel_{op}.json", "design")
            start = time.perf_counter()
            fractions = {}
            for f in spec["f_grid"]:
                rows = list(pool.map(_target, [(s, f, q0, ()) for s in slots], chunksize=4))
                fractions[f] = float(np.mean([r["stall"] for r in rows]))
            f_sel = min(spec["f_grid"], key=lambda f: (abs(fractions[f] - 0.5), f))
            rows = list(pool.map(_target, [(s, f_sel, q0, report) for s in slots], chunksize=4))
            t_targets = time.perf_counter() - start
            rng = np.random.default_rng([config["pilot_seed"], 1, len(op)])
            n_qa = int(round(config["targets"]["qa_fraction"] * len(slots)))
            qa_idx = sorted(rng.choice(len(slots), size=n_qa, replace=False).tolist())
            by_sid = {r["sid"]: r for r in rows}
            qa = list(pool.map(_qa, [(slots[i], f_sel, q0, config["targets"]["qa_t_end"],
                                      by_sid[slots[i]["sid"]]) for i in qa_idx]))  # fmt: skip
            (run_dir / f"design_targets_{op}.json").write_text(
                json.dumps({"operating_point": op, "q0": q0, "f_selected": f_sel,
                            "stall_fraction_by_f": fractions, "targets": rows, "qa": qa}, indent=1)
            )  # fmt: skip
            stall_by_type: dict[str, list[bool]] = {}
            for r in rows:
                stall_by_type.setdefault(r["type"], []).append(r["stall"])
            summary[op] = {
                "stall_fraction_by_f": fractions,
                "f_selected": f_sel,
                "flag_outside_0.2_0.8": not (0.2 <= fractions[f_sel] <= 0.8),
                "stall_fraction_by_type": {k: float(np.mean(v)) for k, v in stall_by_type.items()},
                "D_quantiles": np.quantile(
                    [r["D"] for r in rows], [0, 0.25, 0.5, 0.75, 1]
                ).tolist(),
                "qa_n": len(qa),
                "qa_tolerance_ok": int(sum(q["tolerance_ok"] for q in qa)),
                "qa_class_agree": int(
                    sum((q["ode_class"] == "stall") == q["stall_target"] for q in qa)
                ),
                "qa_failures": [q for q in qa if not q["tolerance_ok"]],
                "seconds_targets": t_targets,
            }
            print(json.dumps({op: summary[op]}, indent=1))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"run directory: {run_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
