"""E004a Stage 1 FINAL held-out, step 1 (registry: Amendment 4; pre-unseal integrity log).

Requires the approval file (configs/e004/HELDOUT_APPROVED) naming the frozen predictor
configuration. Generates the sealed TEST panel (seed stream 1) and SHIFT panel (stream 2; J_G(0) ~
U(0.01, 0.05), 4 x 4 GRPO batches), freezes both (JSON + sha256), and runs them with exactly the
frozen Stage 1 measurement: sampled Adam (4 verifier + 4 clean seeds; T = 2700) with 32 x 8 audits
at H u {h/2}; NG exact flow (test only; secondary). Seeds: the registered tree with the global
structure offsets of vdyn.e004.heldout. Outcomes use the registered definitions and exclusion rule.
Shift geometry is audit-only (the pooled estimator needs equal group sizes; 4 x 4 training groups);
shift J_V pools the audit and the training batch as registered. No predictor is fitted here.
"""

import hashlib
import json
import math
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0b_runs as s0b  # noqa: E402
import stage1_runs as s1r  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e004 import audit as au  # noqa: E402
from vdyn.e004 import dynamics as dy  # noqa: E402
from vdyn.e004 import heldout as hd  # noqa: E402
from vdyn.e004 import panel0b as pn  # noqa: E402
from vdyn.e004 import toy  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e004" / "e004a_stage1.toml"
DESIGN_PANEL = REPO / "configs" / "e004" / "design_panel_0b.json"
FROZEN = REPO / "results/E004a-stage1-analysis/20260926T050431Z_26b1790/predictors_frozen.json"
FROZEN_SHA = "e0b2ea9d73a3f8b7f820e5274c507a626b39f84193d9359aa2ed67436bdfd210"
PANELS = {"test": REPO / "configs/e004/test_panel_0b.json",
          "shift": REPO / "configs/e004/shift_panel_0b.json"}  # fmt: skip
WORKERS = 8


def _exact_shift(structs: list[toy.Structure], theta: np.ndarray, v_hat: np.ndarray,
                 batch: int) -> dict[str, np.ndarray]:  # fmt: skip
    n, c, _ = theta.shape
    tb = toy.Tables.of([s for s in structs for _ in range(c)])
    th = theta.reshape(n * c, toy.D)
    obs = dy.observables(tb, th)
    v = v_hat.reshape(n * c, toy.D).copy()
    ex0 = toy.exact(tb, th)
    is0 = np.tile(np.arange(c) == 0, n)
    v[is0] = (ex0["g_eff"] ** 2 + ex0["var_eff"] / batch)[is0]
    geo = dy.geometry2(tb, th, "adam", v_hat=v)
    out = {"obs": np.stack([obs[k] for k in ("J_G", "J_V", "FPR", "FNR")], -1).reshape(n, c, 4)}
    for lvl in ("u", "r"):
        g = np.stack([geo[f"{k}_{lvl}"] for k in ("A", "alpha", "C", "C_in", "C_out")], -1)
        out[f"geo_{lvl}"] = g.reshape(n, c, 5)
    return out


def _shift_chunk(args: tuple[Any, ...]) -> dict[str, np.ndarray]:
    structs, seeds, audit_seeds, steps = args
    n_p, n_r = hd.SHIFT_BATCH
    ck = s0b.ck_steps(steps)
    res = dy.run_sampled_adam(structs, seeds, steps, ck, n_prompts=n_p, n_resp=n_r,
                              record_batches=True)  # fmt: skip
    tb = toy.Tables.of(structs)
    steps_f = s0b.frac_steps(steps)
    all_th = s0b.to_fracs(res["theta"], steps, axis=1)
    all_v = s0b.to_fracs(res["v_hat"], steps, axis=1)
    rngs = [np.random.default_rng(a) for a in audit_seeds]
    per_ck = []
    for f in s1r.FEAT_FRACS:
        k = ck.index(steps_f[s0b.FRACS.index(f)])
        b = res["batches"]
        train = au.Groups(b["x"][:, k], b["row"][:, k], b["V"][:, k])
        v = None if f == 0 else res["v_hat"][:, k]
        aud = au.sample_groups(tb, res["theta"][:, k], rngs)
        est = au.estimate(tb, res["theta"][:, k], aud, None, "adam", v)
        est["J_V"] = hd.pooled_jv(aud, train)
        per_ck.append(est)
    est_all = s1r._stack(per_ck)
    n_ck = len(s0b.FRACS)
    obs_all = dy.observables(toy.Tables.of([s for s in structs for _ in range(n_ck)]),
                             all_th.reshape(-1, toy.D))  # fmt: skip
    out = {f"est__{k}": v for k, v in est_all.items()}
    out["obs"] = np.stack([obs_all[k] for k in ("J_G", "J_V", "FPR", "FNR")], -1).reshape(
        len(structs), n_ck, 4
    )
    fp = s1r.feature_positions(s0b.FRACS)
    ex = _exact_shift(structs, all_th[:, fp], all_v[:, fp], n_p * n_r)
    out.update({f"exact__{k}": v for k, v in ex.items()})
    return out


def runs_all(pool: ProcessPoolExecutor, fn: Any, structs: list[toy.Structure],
             seeds: list[list[Any]], audit_seeds: list[list[Any]],
             steps: int) -> dict[str, np.ndarray]:  # fmt: skip
    runs = [(s, sd, a) for s, ss, aa in zip(structs, seeds, audit_seeds, strict=True)
            for sd, a in zip(ss, aa, strict=True)]  # fmt: skip
    size = math.ceil(len(runs) / (WORKERS * 4))
    chunks = [runs[i : i + size] for i in range(0, len(runs), size)]
    jobs = [([r[0] for r in c], [r[1] for r in c], [r[2] for r in c], steps) for c in chunks]
    parts = list(pool.map(fn, jobs))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return {k: v.reshape(len(structs), s0b.N_SEEDS, *v.shape[1:]) for k, v in out.items()}


def main() -> int:
    cfg = provenance.load_config(CONFIG)
    frozen_sha = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    if frozen_sha != FROZEN_SHA:
        print("STOP: frozen predictor configuration changed")
        return 1
    if hashlib.sha256(DESIGN_PANEL.read_bytes()).hexdigest() != cfg["panel_sha256"]:
        print("STOP: design panel changed")
        return 1
    appr = hd.require_approval(hd.APPROVAL, FROZEN_SHA)
    for p in PANELS.values():
        if p.exists():
            print(f"STOP: {p.name} already exists; held-out panels are generated once")
            return 1
    T_adam, T_ng = cfg["runs"]["T_adam"], cfg["runs"]["T_ng"]
    run_dir = provenance.create_run_dir(REPO / "results", "E004a-stage1-heldout-runs", REPO)
    t0 = time.perf_counter()
    gen = {"test": hd.test_panel(hd.APPROVAL, FROZEN_SHA, workers=WORKERS),
           "shift": hd.shift_panel(hd.APPROVAL, FROZEN_SHA, workers=WORKERS)}  # fmt: skip
    panels = {k: v[0] for k, v in gen.items()}
    infeasible = {k: v[1] for k, v in gen.items()}
    t_panels = time.perf_counter() - t0
    shas = {}
    for split, structs in panels.items():  # written inside the run directory first (clean meta)
        doc = {"split": split, "root_seed": pn.ROOT_SEED, "yb_cap": 0.2,
               "max_target_redraws": hd.MAX_TARGET_REDRAWS,
               "infeasible_excluded": infeasible[split],
               "structures": [s.to_dict() for s in structs]}  # fmt: skip
        body = json.dumps(doc, indent=1) + "\n"
        (run_dir / PANELS[split].name).write_text(body)
        shas[split] = hashlib.sha256(body.encode()).hexdigest()
    provenance.write_metadata(run_dir, "E004a-stage1-heldout-runs", CONFIG, REPO,
                              extra={"split": "test+shift", "root_seed": pn.ROOT_SEED,
                                     "panel_sha256": cfg["panel_sha256"],
                                     "test_panel_sha256": shas["test"],
                                     "shift_panel_sha256": shas["shift"],
                                     "predictors_frozen_sha256": frozen_sha,
                                     "infeasible_excluded": infeasible,
                                     "approval": appr})  # fmt: skip
    for split in panels:  # frozen copies next to the design panel, after the metadata
        PANELS[split].write_bytes((run_dir / PANELS[split].name).read_bytes())
    print(f"panels generated in {t_panels:.0f} s: {shas}", flush=True)
    timing: dict[str, float] = {"panels": t_panels}
    full = len(pn.CELLS) * pn.N_PER_CELL  # seeds are indexed by the position in the full layout
    test, shift = panels["test"], panels["shift"]

    def pick(seeds: list[Any], structs: list[toy.Structure]) -> list[Any]:
        return [seeds[s.meta["panel_index"]] for s in structs]

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        t0 = time.perf_counter()
        tv = runs_all(pool, s1r._adam_chunk, test,
                      pick(hd.run_seeds("prim_ver", "test", full), test),
                      pick(hd.audit_seeds("adam", "test", full), test), T_adam)  # fmt: skip
        tc = s0b.sampled(pool, test, pick(hd.run_seeds("prim_clean", "test", full), test), T_adam,
                         True)  # fmt: skip
        timing["test_adam"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        ngv = s1r.ng_all(pool, test, T_ng, pick(hd.audit_seeds("ng", "test", full), test))
        ngc = s0b.ng(pool, test, T_ng, True)
        timing["test_ng"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        n_p, n_r = hd.SHIFT_BATCH
        sv = runs_all(pool, _shift_chunk, shift, pick(hd.run_seeds("prim_ver", "shift", full),
                                                      shift),
                      pick(hd.audit_seeds("adam", "shift", full), shift), T_adam)  # fmt: skip
        sc = s0b.sampled(pool, shift, pick(hd.run_seeds("prim_clean", "shift", full), shift),
                         T_adam, True, n_p, n_r)  # fmt: skip
        timing["shift_adam"] = time.perf_counter() - t0
        print(json.dumps(timing), flush=True)
    grid = [s0b.FRACS.index(i / 100) for i in range(101)]
    out: dict[str, Any] = {}
    for split, ver, cl, structs in (("test", tv, tc, test), ("shift", sv, sc, shift)):
        clean_mean = cl["obs"][..., grid, 0].mean(1)
        lab: dict[str, Any] = {"adam_prim": s0b.labels(ver["obs"][..., grid, 0], clean_mean)}
        arrays = {f"adam__{k}": v for k, v in ver.items() if k != "obs"}
        arrays["adam__jg_grid"] = ver["obs"][..., grid, 0]
        arrays["adam__clean_mean_grid"] = clean_mean
        if split == "test":
            lab["ng_prim"] = s0b.labels(ngv["obs"][:, None, grid, 0], ngc["obs"][..., grid, 0])
            arrays.update({f"ng__{k}": v for k, v in ngv.items() if k != "obs"})
        gain = clean_mean[:, -1] - clean_mean[:, 0]
        excluded = [structs[i].sid for i in np.flatnonzero(gain < 0.1)]
        np.savez_compressed(run_dir / f"{split}.npz", allow_pickle=False, **arrays)
        (run_dir / f"labels_{split}.json").write_text(json.dumps(lab) + "\n")
        fails = [lb["failure"] for i, s in enumerate(structs) if s.sid not in excluded
                 for lb in lab["adam_prim"][i]]  # fmt: skip
        out[split] = {"n_structures": len(structs), "excluded_low_clean_gain": excluded,
                      "infeasible_excluded": infeasible[split],
                      "n_excluded": len(excluded),
                      "adam_failure_rate": float(np.mean(fails))}  # fmt: skip
    usage = {
        "self_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "children_maxrss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    }
    summary = {"T_adam": T_adam, "T_ng": T_ng, "feature_fracs": s1r.FEAT_FRACS,
               "splits": out, "panel_sha256": shas, "timing": timing, "peak_memory": usage,
               "predictors_frozen_sha256": frozen_sha, "shift_batch": hd.SHIFT_BATCH,
               "shift_geometry": "audit-only (the pooled estimator needs equal groups)"
               }  # fmt: skip
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "excluded_low_clean_gain"}
                      for k, v in out.items()}))  # fmt: skip
    print(f"run directory: {run_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
