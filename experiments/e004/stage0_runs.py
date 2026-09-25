"""E004a Stage 0, step 2 (registry E004a §5-§7): all runs on the frozen DESIGN panel.

1. Provisional clean runs fix T (sampled Adam: 4 seeds x 4000 steps; NG: exact flow to 60).
2. Main runs: sampled GRPO-lite Adam (primary; 4 verifier + 4 clean seeds) for the primary
   structures and their canonical twins; NG flows (primary, canonical, uncoupled twin, clean);
   MF-Adam (primary, clean; design approximation only); batch sweep (4x4, 16x16) on the first
   10 structures of every construction.
3. Exact observables and optimizer-metric geometry at every checkpoint; outcome labels.
Seeds: SeedSequence(20260930).spawn(4)[3] -> 10 role streams -> 480 structures -> 4 seeds.
"""

import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from vdyn import provenance
from vdyn.e004 import dynamics as dy
from vdyn.e004 import outcomes as oc
from vdyn.e004 import panel as pn
from vdyn.e004 import toy

REPO = Path(__file__).resolve().parents[2]
PANEL = REPO / "configs" / "e004" / "design_panel.json"
CONFIG = REPO / "configs" / "e004" / "e004a.toml"
PANEL_SHA = "e49753e9f8dd016436fdfbe454dca874219694b97da2a43ad8684ef9b817a30d"
H = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
FRACS = sorted(set(H + [h / 2 for h in H if h > 0] + [i / 100 for i in range(101)]))
ROLES = ["prim_ver", "prim_clean", "can_ver", "can_clean", "b16_ver", "b16_clean", "b256_ver",
         "b256_clean", "hardpair", "spare"]  # fmt: skip
N_SEEDS, WORKERS = 4, 8
PROV_STEPS, PROV_NG = 4000, 60.0


def seed_table(n_struct: int) -> dict[str, list[list[np.random.SeedSequence]]]:
    runs = np.random.SeedSequence(pn.ROOT_SEED).spawn(4)[3]
    roles = runs.spawn(len(ROLES))
    return {r: [c.spawn(N_SEEDS) for c in roles[i].spawn(n_struct)] for i, r in enumerate(ROLES)}


def frac_steps(steps: int) -> list[int]:
    return [0 if f == 0 else max(1, round(f * steps)) for f in FRACS]


def ck_steps(steps: int) -> list[int]:
    return sorted(set(frac_steps(steps)))


def to_fracs(arr: np.ndarray, steps: int, axis: int) -> np.ndarray:
    """Re-index a checkpoint axis (unique sorted steps) to one entry per registered fraction."""
    ck = ck_steps(steps)
    pos = [ck.index(s) for s in frac_steps(steps)]
    return np.take(arr, pos, axis=axis)


def _eval(structs: list[toy.Structure], theta: np.ndarray, v_hat: np.ndarray | None, clean: bool,
          kind: str, batch: int) -> dict[str, np.ndarray]:  # fmt: skip
    """Exact observables (+ geometry if verifier run) at all checkpoints: theta (n, c, D)."""
    n, c, _ = theta.shape
    flat_s = [s for s in structs for _ in range(c)]
    tb = toy.Tables.of(flat_s)
    tb = tb.clean() if clean else tb
    th = theta.reshape(n * c, toy.D)
    obs = dy.observables(tb, th)
    out = {"obs": np.stack([obs[k] for k in ("J_G", "J_V", "FPR", "FNR")], -1).reshape(n, c, 4),
           "ctx": obs["J_G_ctx"].reshape(n, c, toy.K)}  # fmt: skip
    if not clean:
        if kind == "adam":
            assert v_hat is not None
            v = v_hat.reshape(n * c, toy.D).copy()
            ex0 = toy.exact(tb, th)  # t = 0 states use the mean-field second moment
            v0 = ex0["g_eff"] ** 2 + ex0["var_eff"] / batch
            is0 = np.tile(np.arange(c) == 0, n)
            v[is0] = v0[is0]
            geo = dy.geometry(tb, th, "adam", v_hat=v)
        else:
            geo = dy.geometry(tb, th, "ng")
        out["geo"] = np.stack([geo[k] for k in ("A", "alpha", "C", "C_in", "C_out")], -1)
        out["geo"] = out["geo"].reshape(n, c, 5)
    return out


def _sampled_chunk(args: tuple[Any, ...]) -> dict[str, np.ndarray]:
    structs, seeds, steps, clean, n_p, n_r = args
    ck = ck_steps(steps)
    res = dy.run_sampled_adam(structs, seeds, steps, ck, clean=clean, n_prompts=n_p, n_resp=n_r)
    out = _eval(structs, res["theta"], res["v_hat"], clean, "adam", n_p * n_r)
    return {k: to_fracs(v, steps, axis=1) for k, v in out.items()}


def sampled(pool: ProcessPoolExecutor, structs: list[toy.Structure],
            seeds: list[list[np.random.SeedSequence]], steps: int, clean: bool,
            n_p: int = 8, n_r: int = 8) -> dict[str, np.ndarray]:  # fmt: skip
    runs = [(s, sd) for s, ss in zip(structs, seeds, strict=True) for sd in ss]
    size = math.ceil(len(runs) / (WORKERS * 4))
    chunks = [runs[i : i + size] for i in range(0, len(runs), size)]
    jobs = [([r[0] for r in ch], [r[1] for r in ch], steps, clean, n_p, n_r) for ch in chunks]
    parts = list(pool.map(_sampled_chunk, jobs))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n = len(structs)
    return {k: v.reshape(n, len(seeds[0]), *v.shape[1:]) for k, v in out.items()}


def _ng_one(args: tuple[Any, ...]) -> dict[str, np.ndarray]:
    st, t_end, clean = args
    ck = [f * t_end for f in FRACS]
    th = dy.ng_one(st, t_end, ck, clean=clean)[None]
    return _eval([st], th, None, clean, "ng", 64)


def ng(pool: ProcessPoolExecutor, structs: list[toy.Structure], t_end: float,
       clean: bool) -> dict[str, np.ndarray]:  # fmt: skip
    parts = list(pool.map(_ng_one, [(s, t_end, clean) for s in structs], chunksize=4))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def mf(structs: list[toy.Structure], steps: int, clean: bool) -> dict[str, np.ndarray]:
    ck = ck_steps(steps)
    res = dy.run_mf_adam(structs, steps, ck, clean=clean)
    out = _eval(structs, res["theta"], res["v_hat"], clean, "adam", 64)
    return {k: to_fracs(v, steps, axis=1) for k, v in out.items()}


def t95(curves: np.ndarray, times: np.ndarray, j0: np.ndarray, jmax: np.ndarray) -> np.ndarray:
    target = j0 + 0.95 * (jmax - j0)
    hit = curves >= target[:, None]
    return np.where(hit.any(1), times[np.argmax(hit, axis=1)], times[-1])


def outcome_grid(res: dict[str, np.ndarray]) -> np.ndarray:
    """J_G on the 101-point outcome grid (fractions 0..1 of T)."""
    idx = [FRACS.index(i / 100) for i in range(101)]
    return res["obs"][..., idx, 0]


def labels(ver: np.ndarray, clean_mean: np.ndarray) -> list[list[dict[str, Any]]]:
    return [[oc.label(ver[i, r], clean_mean[i]) for r in range(ver.shape[1])]
            for i in range(ver.shape[0])]  # fmt: skip


def main() -> int:
    if hashlib.sha256(PANEL.read_bytes()).hexdigest() != PANEL_SHA:
        print("STOP: design panel hash mismatch")
        return 1
    cfg = provenance.load_config(CONFIG)
    assert cfg["panel_sha256"] == PANEL_SHA and cfg["horizons"]["fractions"] == H
    assert cfg["runs"]["roles"] == ROLES and cfg["runs"]["n_seeds"] == N_SEEDS
    assert (cfg["runs"]["provisional_adam_steps"], cfg["runs"]["provisional_ng_time"]) == (
        PROV_STEPS,
        PROV_NG,
    )
    structs = [toy.Structure.from_dict(d) for d in json.loads(PANEL.read_text())["structures"]]
    can = [pn.canonical_twin(s) for s in structs]
    unc = [pn.uncoupled_twin(s) for s in structs]
    seeds = seed_table(len(structs))
    run_dir = provenance.create_run_dir(REPO / "results", "E004a-stage0", REPO)
    provenance.write_metadata(run_dir, "E004a-stage0", CONFIG, REPO,
                              extra={"split": "design", "panel_sha256": PANEL_SHA})  # fmt: skip
    timing: dict[str, float] = {}
    arrays: dict[str, np.ndarray] = {}
    jmax = np.array([float(s.w @ s.p) for s in structs])
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        t0 = time.perf_counter()
        prov = sampled(pool, structs, seeds["prim_clean"], PROV_STEPS, clean=True)
        steps_grid = np.array(frac_steps(PROV_STEPS))
        mean_curve = prov["obs"][..., 0].mean(1)  # (n, ck)
        t95_adam = t95(mean_curve, steps_grid, mean_curve[:, 0], jmax)
        T_adam = int(math.ceil(3 * float(np.median(t95_adam)) / 100) * 100)
        prov_ng = ng(pool, structs, PROV_NG, clean=True)
        times_ng = np.array(FRACS) * PROV_NG
        t95_ng = t95(prov_ng["obs"][..., 0], times_ng, prov_ng["obs"][:, 0, 0], jmax)
        T_ng = float(np.round(3 * float(np.median(t95_ng)), 1))
        timing["provisional"] = time.perf_counter() - t0
        print(f"T_adam = {T_adam} steps (median t95 {np.median(t95_adam):.0f}); "
              f"T_ng = {T_ng} (median t95 {np.median(t95_ng):.2f})")  # fmt: skip
        groups: dict[str, dict[str, np.ndarray]] = {}
        for name, fn in (
            ("adam_prim_ver", lambda: sampled(pool, structs, seeds["prim_ver"], T_adam, False)),
            ("adam_prim_clean", lambda: sampled(pool, structs, seeds["prim_clean"], T_adam, True)),
            ("adam_can_ver", lambda: sampled(pool, can, seeds["can_ver"], T_adam, False)),
            ("adam_can_clean", lambda: sampled(pool, can, seeds["can_clean"], T_adam, True)),
            ("ng_prim_ver", lambda: ng(pool, structs, T_ng, False)),
            ("ng_can_ver", lambda: ng(pool, can, T_ng, False)),
            ("ng_unc_ver", lambda: ng(pool, unc, T_ng, False)),
            ("ng_clean", lambda: ng(pool, structs, T_ng, True)),
        ):
            t0 = time.perf_counter()
            groups[name] = fn()
            timing[name] = time.perf_counter() - t0
            print(f"{name}: {timing[name]:.0f} s")
        sweep = [i for i, s in enumerate(structs) if int(s.sid.split("-")[1]) < 10]
        sw = [structs[i] for i in sweep]
        for tag, n_p, n_r in (("b16", 4, 4), ("b256", 16, 16)):
            for role, clean in ((f"{tag}_ver", False), (f"{tag}_clean", True)):
                t0 = time.perf_counter()
                sd = [seeds[role][i] for i in sweep]
                groups[f"adam_{role}"] = sampled(pool, sw, sd, T_adam, clean, n_p, n_r)
                timing[f"adam_{role}"] = time.perf_counter() - t0
    for name, clean in (("mf_prim_ver", False), ("mf_prim_clean", True)):
        t0 = time.perf_counter()
        groups[name] = mf(structs, T_adam, clean)
        timing[name] = time.perf_counter() - t0
    # outcome labels
    lab: dict[str, Any] = {}
    g = {k: outcome_grid(v) for k, v in groups.items()}
    lab["adam_prim"] = labels(g["adam_prim_ver"], g["adam_prim_clean"].mean(1))
    lab["adam_can"] = labels(g["adam_can_ver"], g["adam_can_clean"].mean(1))
    lab["ng_prim"] = labels(g["ng_prim_ver"][:, None], g["ng_clean"])
    lab["ng_can"] = labels(g["ng_can_ver"][:, None], g["ng_clean"])
    lab["mf_prim"] = labels(g["mf_prim_ver"][:, None], g["mf_prim_clean"])
    lab["adam_b16"] = labels(g["adam_b16_ver"], g["adam_b16_clean"].mean(1))
    lab["adam_b256"] = labels(g["adam_b256_ver"], g["adam_b256_clean"].mean(1))
    lab["adam_b64_sweep"] = [lab["adam_prim"][i] for i in sweep]
    clean_gain = g["adam_prim_clean"].mean(1)[:, -1] - g["adam_prim_clean"].mean(1)[:, 0]
    for k, v in groups.items():
        for kk, arr in v.items():
            arrays[f"{k}__{kk}"] = arr
    np.savez_compressed(run_dir / "runs.npz", allow_pickle=False, **arrays)
    summary = {
        "T_adam": T_adam, "T_ng": T_ng, "t95_adam": t95_adam.tolist(), "t95_ng": t95_ng.tolist(),
        "fracs": FRACS, "adam_steps": ck_steps(T_adam), "sweep_index": sweep,
        "excluded_low_clean_gain": [structs[i].sid for i in np.flatnonzero(clean_gain < 0.1)],
        "timing": timing, "sids": [s.sid for s in structs],
    }  # fmt: skip
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (run_dir / "labels.json").write_text(json.dumps(lab) + "\n")
    print(f"run directory: {run_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
