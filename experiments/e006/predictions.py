"""E006 pre-training predictions and diagnostics from base-policy samples only (no training),
committed before any E006 run (research/paper/e006_protocol.md).

For the 10-verifier panel (E006 arms + E005b V1 flip and V2 deleted):
- gold-free RME / PME on the calibration-audit samples;
- static FPR, FNR and Youden's J on the verification-audit samples.
For the coverage arms (c = 0, .25, .5, .75, 1) and rarekey: D-hat, the mean GRPO advantage mass on
the shared false-positive behaviour (M, or nu*) per calibration-audit group, averaged over coin
replicates. This is the theory's initial push on the shared behaviour (theory.md, Derivation 5).
Usage: predictions.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import audit as au  # noqa: E402
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402
from vdyn.e005b import verifiers as vf5  # noqa: E402
from vdyn.e006 import diagnostic as dg  # noqa: E402
from vdyn.e006 import verifiers as vf  # noqa: E402

PANEL_EXTRA = ("flip", "deleted")  # E005b V1 / V2


def accept_fn(name: str, arms: dict[str, Any], deleted: set[tuple[int, int]]) -> dg.Accept:
    if name in PANEL_EXTRA:
        return lambda x, k, rng: vf5.reward(name, x[0], x[1], list(k), rng, deleted)
    spec = vf.Spec(**arms["specs"][name])
    return lambda x, k, rng: vf.reward(spec, x[0], x[1], list(k), rng)


def advantages(V: np.ndarray) -> np.ndarray:
    sd = V.std(axis=-1, ddof=1, keepdims=True)
    return np.where(sd > 0, (V - V.mean(axis=-1, keepdims=True)) / (sd + 1e-4), 0.0)


def main(argv: list[str]) -> int:
    cfg = cm.load_config()
    a = provenance.load_config(au.MATCHED)["audit"]
    ec = provenance.load_config(au.E006)
    arms = json.loads(au.ARMS.read_text())
    d = ec["diagnostic"]
    if provenance.git_state(cm.REPO)["dirty"]:
        print("STOP: working tree is dirty")
        return 1
    v = cfg["verifiers"]
    deleted = vf5.deleted_prompts(v["deleted_fraction"], v["deleted_seed"])
    run_dir = provenance.create_run_dir(cm.REPO / "results", "E006-predictions", cm.REPO)
    extra = cm.run_extra(cfg, arms_file_sha256=au.sha(au.ARMS))
    provenance.write_metadata(run_dir, "E006-predictions", au.E006, cm.REPO, extra=extra)
    cal = au.audit_samples(cfg, a, "calibrate")
    ver = au.audit_samples(cfg, a, "verify")
    panel = list(arms["specs"]) + list(PANEL_EXTRA)
    out: dict[str, Any] = {"panel": panel, "rme": {}, "static": {}, "dhat": {}}

    # gold-free RME / PME on calibration samples
    keys_count: dict[tuple[int, ...], int] = defaultdict(int)
    sampled_on: dict[tuple[int, ...], set[int]] = defaultdict(set)
    g = cal["group"]
    for i, t in enumerate(cal["toks"]):
        k = vf.completion_key(t)
        keys_count[k] += 1
        sampled_on[k].add(i // g)
    keys = sorted(keys_count)
    mass = np.array([keys_count[k] for k in keys], dtype=float)
    so = {i: sampled_on[k] for i, k in enumerate(keys)}
    for name in panel:
        out["rme"][name] = dg.response_main_effect(
            keys, mass, so, cal["pairs"], accept_fn(name, arms, deleted), panel=d["panel"],
            rng_panel=np.random.default_rng(d["panel_seed"]),
            rng_coin=np.random.default_rng(d["coin_seed"]))  # fmt: skip
        print(name, {k: round(x, 5) for k, x in out["rme"][name].items()}, flush=True)

    # static FPR / FNR / J on verification samples
    rep = [p for p in ver["pairs"] for _ in range(ver["group"])]
    G = np.array([tk.gold_reward(x, y, *tk.parse_completion(t))
                  for (x, y), t in zip(rep, ver["toks"], strict=True)])  # fmt: skip
    for name in panel:
        acc = accept_fn(name, arms, deleted)
        rng = np.random.default_rng(a["verification_coin_seed"])
        V = np.array([acc((x, y), t, rng) for (x, y), t in zip(rep, ver["toks"], strict=True)])
        fpr = float(V[G == 0].mean())
        fnr = float(1 - V[G == 1].mean())
        out["static"][name] = {"fpr": fpr, "fnr": fnr, "J": 1 - fnr - fpr}

    # D-hat on calibration groups
    crep = [p for p in cal["pairs"] for _ in range(g)]
    parsed = [tk.parse_completion(t) for t in cal["toks"]]
    inM = np.array([vf.in_master_set(x, y, *pv) for (x, y), pv in zip(crep, parsed, strict=True)])
    nu = int(arms["rare_value"])
    gold = np.array([tk.gold_reward(x, y, *pv) for (x, y), pv in zip(crep, parsed, strict=True)])
    isR = np.array([ok and val == nu and gk == 0
                    for (ok, val), gk in zip(parsed, gold, strict=True)])  # fmt: skip
    for name, target in (("randfp", inM), ("cov25", inM), ("cov50", inM), ("cov75", inM),
                         ("exploit", inM), ("rarekey", isR)):  # fmt: skip
        acc = accept_fn(name, arms, deleted)
        rng = np.random.default_rng(d["coin_seed"])
        vals = []
        for _ in range(d["dhat_coin_replicates"] if name != "exploit" else 1):
            V = np.array([acc((x, y), t, rng) for (x, y), t in zip(crep, cal["toks"], strict=True)])
            A = advantages(V.reshape(-1, g))
            vals.append(float((A * target.reshape(-1, g)).sum(1).mean() / g))
        out["dhat"][name] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                             "target_mass": float(target.mean())}  # fmt: skip
        print("dhat", name, out["dhat"][name], flush=True)
    (run_dir / "predictions.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
