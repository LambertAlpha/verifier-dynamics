"""E005b-0 matched-initial-error experiment: the INITIAL-POLICY matching audits on the calibrated
base (research/10_e005b0_pilot.md §14). No training here.

Usage:
  matching_audit.py calibrate  f0 := V3 FPR on the calibration audit, rounded and frozen in
                               configs/e005b/matched_f0.json (refuses if that file exists)
  matching_audit.py verify     independent audit on disjoint prompts with fresh policy samples and
                               fresh VR coins; applies the frozen criterion. Exit 3 = STOP.
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import matching as mt  # noqa: E402
from vdyn.e005b import matrix as mx  # noqa: E402
from vdyn.e005b import model as mdl  # noqa: E402
from vdyn.e005b import rl  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402
from vdyn.e005b import verifiers as vf  # noqa: E402

MATCHED = cm.REPO / "configs" / "e005b" / "matched.toml"
CHUNK = 250  # prompts per generation call (a memory bound only; recorded in the metadata)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_prompts(train: list[tuple[int, int]], a: dict[str, Any], which: str
                  ) -> list[tuple[int, int]]:  # fmt: skip
    n = a["prompts_per_audit"]
    assert 2 * n <= len(train)
    perm = np.random.default_rng(a["permutation_seed"]).permutation(len(train))
    idx = perm[:n] if which == "calibrate" else perm[n : 2 * n]
    return [train[i] for i in idx]


def sample(net: torch.nn.Module, pairs: list[tuple[int, int]], group: int, seed: int,
           temperature: float) -> list[list[int]]:  # fmt: skip
    gen = torch.Generator().manual_seed(seed)
    toks: list[list[int]] = []
    for i in range(0, len(pairs), CHUNK):
        roll = rl.rollout(net, pairs[i : i + CHUNK], group, gen, temperature)
        toks += roll["tokens"].tolist()
    return toks


def arm_report(pid: np.ndarray, cats: np.ndarray, V: np.ndarray, G: np.ndarray, group: int,
               a: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    r = mt.audit_rates(pid, cats, V, G, np.random.default_rng(a["bootstrap_seed"]),
                       a["bootstrap_resamples"])  # fmt: skip
    r["mixed_groups"] = float(np.mean(V.reshape(-1, group).std(1) > 0))
    return r


def git_clean_and_tracked(p: Path) -> bool:
    rel = str(p.relative_to(cm.REPO))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=cm.REPO,
                             capture_output=True).returncode == 0  # fmt: skip
    return tracked and not provenance.git_state(cm.REPO)["dirty"]


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    if mode not in ("calibrate", "verify"):
        print(__doc__)
        return 2
    cfg = cm.load_config()
    mc = provenance.load_config(MATCHED)
    a = mc["audit"]
    g = cfg["grpo"]
    assert a["samples_per_prompt"] == g["group"]
    f0_file = cm.REPO / mc["f0_file"]
    if provenance.git_state(cm.REPO)["dirty"]:
        print("STOP: working tree is dirty")
        return 1
    if mode == "calibrate" and f0_file.exists():
        print(f"STOP: {f0_file.name} already exists; f0 is calibrated once and never retuned")
        return 1
    if mode == "verify" and not (f0_file.exists() and git_clean_and_tracked(f0_file)):
        print(f"STOP: {f0_file.name} must exist and be committed before the verification audit")
        return 1
    ptr = json.loads((cm.REPO / mc["base_pointer"]).read_text())
    assert ptr["sha256"] == mc["base_sha256"]
    net = mdl.build(mdl.GPTConfig(), seed=0)
    if mdl.load_checkpoint(cm.REPO / ptr["file"], net)["sha256"] != ptr["sha256"]:
        print("STOP: base checkpoint hash mismatch")
        return 1
    net.eval()
    name = f"E005b0-matched-{'calibration' if mode == 'calibrate' else 'verification'}"
    policy_seed = a[
        "calibration_policy_seed" if mode == "calibrate" else "verification_policy_seed"
    ]
    run_dir = provenance.create_run_dir(cm.REPO / "results", name, cm.REPO)
    extra = cm.run_extra(cfg, mode=mode, base_sha256=ptr["sha256"], policy_seed=policy_seed,
                         chunk_prompts=CHUNK, temperature=g["temperature"])  # fmt: skip
    if mode == "verify":
        extra["f0_file_sha256"] = sha(f0_file)
    provenance.write_metadata(run_dir, name, MATCHED, cm.REPO, extra=extra)

    train = tk.make_splits(cfg["data"]["split_seed"])["train"]
    pairs = audit_prompts(train, a, mode)
    t0 = time.perf_counter()
    toks = sample(net, pairs, g["group"], policy_seed, g["temperature"])
    t_gen = time.perf_counter() - t0
    rep = [p for p in pairs for _ in range(g["group"])]
    pid = np.repeat(np.arange(len(pairs)), g["group"])
    cats = np.array([tk.category(x, y) for x, y in rep])
    G = np.array([tk.gold_reward(x, y, *tk.parse_completion(t))
                  for (x, y), t in zip(rep, toks, strict=True)])  # fmt: skip
    V3 = np.array([vf.reward("exploit", x, y, t, np.random.default_rng(0), set())
                   for (x, y), t in zip(rep, toks, strict=True)])  # fmt: skip
    out: dict[str, Any] = {
        "mode": mode, "prompts": len(pairs), "samples": len(toks), "policy_seed": policy_seed,
        "prompt_sha256": hashlib.sha256(json.dumps(pairs).encode()).hexdigest(),
        "generation_s": t_gen,
        "valid": float(np.mean([tk.parse_completion(t)[0] for t in toks])),
        "suffix": mx.suffix_stats(rep, toks), "concentration": mx.concentration(toks),
        "prompts_by_cat": {c: int(sum(tk.category(x, y) == c for x, y in pairs))
                           for c in tk.CATEGORIES},
        "arms": {"clean": arm_report(pid, cats, G, G, g["group"], a),
                 "exploit": arm_report(pid, cats, V3, G, g["group"], a)},
    }  # fmt: skip
    fpr_v3 = out["arms"]["exploit"]["overall"]["fpr"]
    if mode == "calibrate":
        f0 = round(fpr_v3, a["f0_decimals"])
        rec = {"f0": f0, "fpr_v3_unrounded": fpr_v3,
               "n_neg": out["arms"]["exploit"]["overall"]["n_neg"],
               "fpr_v3_ci": out["arms"]["exploit"]["overall"]["fpr_ci"],
               "calibration_run": str(run_dir.relative_to(cm.REPO)),
               "matched_config_sha256": sha(MATCHED), "base_sha256": ptr["sha256"],
               "rule": "f0 := V3 FPR pooled over all G = 0 responses of the calibration audit, "
                       "rounded; frozen before the verification audit (§14)"}  # fmt: skip
        out["f0"] = rec
        (run_dir / "audit.json").write_text(json.dumps(out, indent=1) + "\n")
        f0_file.write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps(rec, indent=1))
    else:
        f0rec = json.loads(f0_file.read_text())
        if f0rec["matched_config_sha256"] != sha(MATCHED) or f0rec["base_sha256"] != ptr["sha256"]:
            print("STOP: f0 was calibrated under a different config or base")
            return 1
        f0 = f0rec["f0"]
        crng = np.random.default_rng(a["verification_coin_seed"])
        VR = np.array([vf.reward_randfp(x, y, t, crng, f0)
                       for (x, y), t in zip(rep, toks, strict=True)])  # fmt: skip
        out["arms"]["randfp"] = arm_report(pid, cats, VR, G, g["group"], a)
        v = mt.verdict(f0, fpr_v3, {"exploit": out["arms"]["exploit"]["overall"]["fnr"],
                                    "randfp": out["arms"]["randfp"]["overall"]["fnr"]},
                       a["tolerance"])  # fmt: skip
        v["f0_file_sha256"] = sha(f0_file)
        v["note"] = ("global, initial-policy matching only; per-category and per-prompt matching "
                     "are not required and not claimed")  # fmt: skip
        out["verdict"] = v
        (run_dir / "audit.json").write_text(json.dumps(out, indent=1) + "\n")
        (run_dir / "verdict.json").write_text(json.dumps(v, indent=1) + "\n")
        print(json.dumps(v, indent=1))
    for arm, r in out["arms"].items():
        o = r["overall"]
        print(f"{arm:8s} FPR {o['fpr']:.4f} {[round(x, 4) for x in o['fpr_ci']]} "
              f"FNR {o['fnr']:.4f} acc {o['acc']:.4f} FP mass {o['fp_mass']:.4f} "
              f"mixed {r['mixed_groups']:.3f} | by cat FPR "
              + " ".join(f"{c}: {r['by_cat'][c]['fpr']:.3f}" for c in tk.CATEGORIES))  # fmt: skip
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    if mode == "verify" and not out["verdict"]["pass"]:
        print("STOP: matching criterion failed; no training, no retuning (§14)")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
