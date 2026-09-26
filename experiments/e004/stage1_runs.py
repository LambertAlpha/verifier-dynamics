"""E004a Stage 1, design round, step 1 (registry: Stage 1 execution note §1, §2, §7, §8).

Sampled GRPO-lite Adam on the frozen Stage 0b design panel with the Stage 0b seeds (so the
trajectories are identical to Stage 0b; asserted), plus finite-sample audits at the feature
checkpoints H u {h/2}: 32 x 8 fresh audit rollouts with gold labels, the training batch drawn from
the checkpoint policy, and the plug-in pooled estimators (vdyn.e004.audit). NG: the exact flow,
audited the same way with the damped audit Fisher metric. Outcomes are recomputed from exact gold
values and asserted equal to the committed Stage 0b labels. Exact (oracle) observables and
geometry at the same checkpoints are stored for estimator sanity checks. DESIGN SPLIT ONLY.
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

from vdyn import provenance  # noqa: E402
from vdyn.e004 import audit as au  # noqa: E402
from vdyn.e004 import dynamics as dy  # noqa: E402
from vdyn.e004 import panel0b as pn  # noqa: E402
from vdyn.e004 import toy  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e004" / "e004a_stage1.toml"
PANEL = REPO / "configs" / "e004" / "design_panel_0b.json"
FEAT_FRACS = sorted(set(s0b.H + [h / 2 for h in s0b.H if h > 0]))
EST_KEYS = ("J_G", "J_V", "FPR", "FNR", "FPM", "A_u", "alpha_u", "C_u", "A_r", "alpha_r",
            "C_r", "C_in", "C_out", "alpha_u_defined", "alpha_r_defined", "n_wrong",
            "n_right")  # fmt: skip
WORKERS = 8


def audit_tree(n_struct: int) -> dict[str, Any]:
    """Execution note §7: spare role -> [adam_audit, ng_audit, hardpair_audit, resample]."""
    spare = np.random.SeedSequence(pn.ROOT_SEED).spawn(5)[3].spawn(len(s0b.ROLES))[9]
    br = spare.spawn(4)
    return {"adam": [c.spawn(s0b.N_SEEDS) for c in br[0].spawn(n_struct)],
            "ng": br[1].spawn(n_struct), "hardpair": br[2], "resample": br[3]}  # fmt: skip


def feature_positions(frac_list: list[float]) -> list[int]:
    return [frac_list.index(f) for f in FEAT_FRACS]


def _exact_at(structs: list[toy.Structure], theta: np.ndarray, v_hat: np.ndarray | None,
              kind: str) -> dict[str, np.ndarray]:  # fmt: skip
    """Oracle observables and both geometry levels at (n, c) states (t = 0 Adam: mean-field v)."""
    n, c, _ = theta.shape
    tb = toy.Tables.of([s for s in structs for _ in range(c)])
    th = theta.reshape(n * c, toy.D)
    obs = dy.observables(tb, th)
    if kind == "adam":
        assert v_hat is not None
        v = v_hat.reshape(n * c, toy.D).copy()
        ex0 = toy.exact(tb, th)
        is0 = np.tile(np.arange(c) == 0, n)
        v[is0] = (ex0["g_eff"] ** 2 + ex0["var_eff"] / 64)[is0]
        geo = dy.geometry2(tb, th, "adam", v_hat=v)
    else:
        geo = dy.geometry2(tb, th, "ng")
    out = {"obs": np.stack([obs[k] for k in ("J_G", "J_V", "FPR", "FNR")], -1).reshape(n, c, 4)}
    for lvl in ("u", "r"):
        g = np.stack([geo[f"{k}_{lvl}"] for k in ("A", "alpha", "C", "C_in", "C_out")], -1)
        out[f"geo_{lvl}"] = g.reshape(n, c, 5)
    return out


def _estimates(tb: toy.Tables, theta_c: np.ndarray, rngs: list[np.random.Generator],
               train: au.Groups | None, kind: str,
               v_hat: np.ndarray | None) -> dict[str, np.ndarray]:  # fmt: skip
    aud = au.sample_groups(tb, theta_c, rngs)
    return au.estimate(tb, theta_c, aud, train, kind, v_hat)


def _stack(per_ck: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    out = {k: np.stack([e[k] for e in per_ck], 1).astype(float) for k in EST_KEYS}
    out["ctx"] = np.stack([e["ctx"] for e in per_ck], 1)
    return out


def _adam_chunk(args: tuple[Any, ...]) -> dict[str, np.ndarray]:
    structs, seeds, audit_seeds, steps = args
    ck = s0b.ck_steps(steps)
    res = dy.run_sampled_adam(structs, seeds, steps, ck, record_batches=True)
    tb = toy.Tables.of(structs)
    steps_f = s0b.frac_steps(steps)
    all_th = s0b.to_fracs(res["theta"], steps, axis=1)
    all_v = s0b.to_fracs(res["v_hat"], steps, axis=1)
    feat_ck = [ck.index(steps_f[s0b.FRACS.index(f)]) for f in FEAT_FRACS]
    rngs = [np.random.default_rng(a) for a in audit_seeds]
    per_ck = []
    for f, k in zip(FEAT_FRACS, feat_ck, strict=True):
        b = res["batches"]
        train = au.Groups(b["x"][:, k], b["row"][:, k], b["V"][:, k])
        v = None if f == 0 else res["v_hat"][:, k]
        per_ck.append(_estimates(tb, res["theta"][:, k], rngs, train, "adam", v))
    est = _stack(per_ck)
    obs_all = dy.observables(toy.Tables.of([s for s in structs for _ in range(len(s0b.FRACS))]),
                             all_th.reshape(-1, toy.D))  # fmt: skip
    out = {f"est__{k}": v for k, v in est.items()}
    out["obs"] = np.stack([obs_all[k] for k in ("J_G", "J_V", "FPR", "FNR")], -1).reshape(
        len(structs), len(s0b.FRACS), 4
    )
    fp = feature_positions(s0b.FRACS)
    ex = _exact_at(structs, all_th[:, fp], all_v[:, fp], "adam")
    out.update({f"exact__{k}": v for k, v in ex.items()})
    return out


def _ng_chunk(args: tuple[Any, ...]) -> dict[str, np.ndarray]:
    st, t_end, audit_seed = args
    th = dy.ng_one(st, t_end, [f * t_end for f in s0b.FRACS])[None]  # (1, fracs, D)
    tb = toy.Tables.of([st])
    fp = feature_positions(s0b.FRACS)
    rngs = [np.random.default_rng(audit_seed)]
    est = _stack([_estimates(tb, th[:, p], rngs, None, "ng", None) for p in fp])
    obs_all = dy.observables(toy.Tables.of([st] * len(s0b.FRACS)), th[0])
    out = {f"est__{k}": v for k, v in est.items()}
    out["obs"] = np.stack([obs_all[k] for k in ("J_G", "J_V", "FPR", "FNR")], -1)[None]
    out.update({f"exact__{k}": v for k, v in _exact_at([st], th[:, fp], None, "ng").items()})
    return out


def adam_all(pool: ProcessPoolExecutor, structs: list[toy.Structure], seeds: list[list[Any]],
             audit_seeds: list[list[Any]], steps: int) -> dict[str, np.ndarray]:  # fmt: skip
    runs = [(s, sd, a) for s, ss, aa in zip(structs, seeds, audit_seeds, strict=True)
            for sd, a in zip(ss, aa, strict=True)]  # fmt: skip
    size = math.ceil(len(runs) / (WORKERS * 4))
    chunks = [runs[i : i + size] for i in range(0, len(runs), size)]
    jobs = [([r[0] for r in c], [r[1] for r in c], [r[2] for r in c], steps) for c in chunks]
    parts = list(pool.map(_adam_chunk, jobs))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n = len(structs)
    return {k: v.reshape(n, s0b.N_SEEDS, *v.shape[1:]) for k, v in out.items()}


def ng_all(pool: ProcessPoolExecutor, structs: list[toy.Structure], t_end: float,
           audit_seeds: list[Any]) -> dict[str, np.ndarray]:  # fmt: skip
    jobs = [(s, t_end, a) for s, a in zip(structs, audit_seeds, strict=True)]
    parts = list(pool.map(_ng_chunk, jobs, chunksize=4))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _same_labels(a: list[list[dict[str, Any]]], b: list[list[dict[str, Any]]]) -> float:
    """Max |Dn| difference; raises if any category, failure flag or onset differs."""
    worst = 0.0
    for ra, rb in zip(a, b, strict=True):
        for x, y in zip(ra, rb, strict=True):
            if (x["category"], x["failure"], x["t_on"]) != (y["category"], y["failure"], y["t_on"]):
                raise AssertionError(f"label mismatch: {x} vs {y}")
            worst = max(worst, abs(x["Dn"] - y["Dn"]))
    return worst


def main() -> int:
    cfg = provenance.load_config(CONFIG)
    panel_bytes = PANEL.read_bytes()
    if hashlib.sha256(panel_bytes).hexdigest() != cfg["panel_sha256"]:
        print("STOP: design panel hash mismatch")
        return 1
    assert cfg["root_seed"] == pn.ROOT_SEED and cfg["runs"]["roles"] == s0b.ROLES
    assert cfg["horizons"]["fractions"] == s0b.H and cfg["runs"]["n_seeds"] == s0b.N_SEEDS
    audit_cfg = (cfg["audit"]["n_groups"], cfg["audit"]["responses_per_group"])
    assert audit_cfg == (au.N_GROUPS, au.N_RESP)
    s0b_dir = REPO / cfg["stage0b_runs"]
    s0b_sum = json.loads((s0b_dir / "summary.json").read_text())
    s0b_lab = json.loads((s0b_dir / "labels.json").read_text())
    T_adam, T_ng = cfg["runs"]["T_adam"], cfg["runs"]["T_ng"]
    assert (s0b_sum["T_adam"], s0b_sum["T_ng"]) == (T_adam, T_ng)
    structs = [toy.Structure.from_dict(d) for d in json.loads(panel_bytes)["structures"]]
    seeds = s0b.seed_table(len(structs))
    tree = audit_tree(len(structs))
    run_dir = provenance.create_run_dir(REPO / "results", "E004a-stage1-runs", REPO)
    provenance.write_metadata(run_dir, "E004a-stage1-runs", CONFIG, REPO,
                              extra={"split": "design", "root_seed": pn.ROOT_SEED,
                                     "panel_sha256": cfg["panel_sha256"],
                                     "stage0b_runs": cfg["stage0b_runs"],
                                     "seed_roles": "train prim_ver/prim_clean; audit spare[0]/[1]"
                                     })  # fmt: skip
    timing: dict[str, float] = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        t0 = time.perf_counter()
        ver = adam_all(pool, structs, seeds["prim_ver"], tree["adam"], T_adam)
        timing["adam_ver_with_audits"] = time.perf_counter() - t0
        print(f"adam verifier runs + audits: {timing['adam_ver_with_audits']:.0f} s", flush=True)
        t0 = time.perf_counter()
        cl = s0b.sampled(pool, structs, seeds["prim_clean"], T_adam, True)
        timing["adam_clean"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        ngv = ng_all(pool, structs, T_ng, tree["ng"])
        ngc = s0b.ng(pool, structs, T_ng, True)
        timing["ng_with_audits"] = time.perf_counter() - t0
        print(
            f"clean {timing['adam_clean']:.0f} s; NG {timing['ng_with_audits']:.0f} s", flush=True
        )
    # outcomes: recomputed and checked against the committed Stage 0b labels
    grid = [s0b.FRACS.index(i / 100) for i in range(101)]
    lab = {
        "adam_prim": s0b.labels(ver["obs"][..., grid, 0], cl["obs"][..., grid, 0].mean(1)),
        "ng_prim": s0b.labels(ngv["obs"][:, None, grid, 0], ngc["obs"][..., grid, 0]),
    }
    checks = {
        "adam_labels_max_dDn": _same_labels(lab["adam_prim"], s0b_lab["adam_prim"]),
        "ng_labels_max_dDn": _same_labels(lab["ng_prim"], s0b_lab["ng_prim"]),
    }
    s0b_arr = np.load(s0b_dir / "runs.npz")
    checks["adam_obs_max_diff_vs_stage0b"] = float(
        np.max(np.abs(ver["obs"] - s0b_arr["adam_prim_ver__obs"]))
    )
    checks["ng_obs_max_diff_vs_stage0b"] = float(
        np.max(np.abs(ngv["obs"] - s0b_arr["ng_prim_ver__obs"]))
    )
    print(json.dumps(checks), flush=True)
    assert checks["adam_obs_max_diff_vs_stage0b"] <= 1e-12, "Adam trajectories differ from Stage 0b"
    assert checks["ng_obs_max_diff_vs_stage0b"] <= 1e-9, "NG trajectories differ from Stage 0b"
    arrays = {f"adam__{k}": v for k, v in ver.items() if k != "obs"}
    arrays.update({f"ng__{k}": v for k, v in ngv.items() if k != "obs"})
    np.savez_compressed(run_dir / "runs.npz", allow_pickle=False, **arrays)
    usage = {
        "self_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "children_maxrss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    }
    summary = {
        "T_adam": T_adam, "T_ng": T_ng, "feature_fracs": FEAT_FRACS,
        "feature_adam_steps": [s0b.frac_steps(T_adam)[s0b.FRACS.index(f)] for f in FEAT_FRACS],
        "excluded_low_clean_gain": s0b_sum["excluded_low_clean_gain"], "checks": checks,
        "timing": timing, "peak_memory": usage, "workers": WORKERS,
        "sids": [s.sid for s in structs], "panel_sha256": cfg["panel_sha256"],
    }  # fmt: skip
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (run_dir / "labels.json").write_text(json.dumps(lab) + "\n")
    print(f"run directory: {run_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
