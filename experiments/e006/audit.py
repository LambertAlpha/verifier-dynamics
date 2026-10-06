"""E006 matching audits on the calibrated base (research/paper/e006_protocol.md). No training.

Usage:
  audit.py calibrate   choose the rare key nu* and its base share eps on the E005b-0 calibration
                       audit samples; write configs/e006/arms.json (refuses if it exists)
  audit.py verify      per-arm FPR / FNR / accuracy on the independent verification audit; an arm
                       passes iff |FPR - f0| <= 0.015. Writes verdict.json. Exit 3 if any arm fails.
The audit prompts, policy seeds and coin seed are those of E005b-0 §14, so the samples are
identical to the committed E005b-0 audits (checked by prompt hash).
"""

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import matching as mt  # noqa: E402
from vdyn.e005b import model as mdl  # noqa: E402
from vdyn.e005b import rl  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402
from vdyn.e006 import verifiers as vf  # noqa: E402

MATCHED = cm.REPO / "configs" / "e005b" / "matched.toml"
E006 = cm.REPO / "configs" / "e006" / "e006.toml"
ARMS = cm.REPO / "configs" / "e006" / "arms.json"
CHUNK = 250


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_samples(cfg: dict[str, Any], a: dict[str, Any], which: str) -> dict[str, Any]:
    train = tk.make_splits(cfg["data"]["split_seed"])["train"]
    n = a["prompts_per_audit"]
    perm = np.random.default_rng(a["permutation_seed"]).permutation(len(train))
    idx = perm[:n] if which == "calibrate" else perm[n : 2 * n]
    pairs = [train[i] for i in idx]
    seed = a["calibration_policy_seed" if which == "calibrate" else "verification_policy_seed"]
    ptr = json.loads((cm.REPO / "configs/e005b/base_checkpoint_v2.json").read_text())
    net = mdl.build(mdl.GPTConfig(), seed=0)
    if mdl.load_checkpoint(cm.REPO / ptr["file"], net)["sha256"] != ptr["sha256"]:
        raise SystemExit("STOP: base checkpoint hash mismatch")
    net.eval()
    gen = torch.Generator().manual_seed(seed)
    toks: list[list[int]] = []
    for i in range(0, len(pairs), CHUNK):
        toks += rl.rollout(net, pairs[i : i + CHUNK], cfg["grpo"]["group"], gen,
                           cfg["grpo"]["temperature"])["tokens"].tolist()  # fmt: skip
    return {"pairs": pairs, "toks": toks, "group": cfg["grpo"]["group"],
            "prompt_sha256": hashlib.sha256(json.dumps(pairs).encode()).hexdigest()}  # fmt: skip


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    if mode not in ("calibrate", "verify"):
        print(__doc__)
        return 2
    cfg = cm.load_config()
    a = provenance.load_config(MATCHED)["audit"]
    ec = provenance.load_config(E006)
    f0 = json.loads((cm.REPO / "configs/e005b/matched_f0.json").read_text())["f0"]
    assert f0 == ec["f0"]
    if provenance.git_state(cm.REPO)["dirty"]:
        print("STOP: working tree is dirty")
        return 1
    if mode == "calibrate" and ARMS.exists():
        print("STOP: configs/e006/arms.json exists; calibration is done once")
        return 1
    if mode == "verify":
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(ARMS)], cwd=cm.REPO,
                                 capture_output=True).returncode == 0  # fmt: skip
        if not tracked:
            print("STOP: configs/e006/arms.json must be committed before verification")
            return 1
    name = f"E006-{'calibration' if mode == 'calibrate' else 'verification'}"
    run_dir = provenance.create_run_dir(cm.REPO / "results", name, cm.REPO)
    provenance.write_metadata(run_dir, name, E006, cm.REPO, extra=cm.run_extra(cfg, mode=mode))
    S = audit_samples(cfg, a, mode)
    ref = json.loads(sorted((cm.REPO / "results" / f"E005b0-matched-{name.split('-')[1]}").glob(
        "*/audit.json"))[0].read_text())  # fmt: skip
    if ref["prompt_sha256"] != S["prompt_sha256"]:
        print("STOP: audit prompts differ from the E005b-0 audit")
        return 1
    rep = [p for p in S["pairs"] for _ in range(S["group"])]
    parsed = [tk.parse_completion(t) for t in S["toks"]]
    G = np.array([tk.gold_reward(x, y, *pv) for (x, y), pv in zip(rep, parsed, strict=True)])
    out: dict[str, Any] = {"mode": mode, "prompt_sha256": S["prompt_sha256"], "f0": f0}
    verdicts: dict[str, dict[str, Any]] = {}
    if mode == "calibrate":
        wrong_vals = Counter(v for (ok, v), g in zip(parsed, G, strict=True)
                             if g == 0 and ok and v is not None and v % 10 != 0)  # fmt: skip
        n_wrong = int((G == 0).sum())
        share = {v: c / n_wrong for v, c in wrong_vals.items()}
        target = ec["rarekey"]["target_share"]
        nu = min(share, key=lambda v: (abs(share[v] - target), v))
        rec = {"f0": f0, "q": f0, "rare_value": int(nu), "eps": share[nu], "n_wrong": n_wrong,
               "rule": "nu* = valid wrong value not ending in 0 with base wrong-share closest to "
                       f"{target} (ties: smaller value), calibration audit",
               "specs": {k: vars(s) for k, s in vf.arm_specs(f0, f0, int(nu), share[nu]).items()},
               "calibration_run": str(run_dir.relative_to(cm.REPO)),
               "e006_config_sha256": sha(E006)}  # fmt: skip
        out["arms"] = rec
        ARMS.write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "specs"}, indent=1))
    else:
        rec = json.loads(ARMS.read_text())
        pid = np.repeat(np.arange(len(S["pairs"])), S["group"])
        cats = np.array([tk.category(x, y) for x, y in rep])
        crng = np.random.default_rng(a["verification_coin_seed"])
        out["arms"] = {}
        for arm, d in rec["specs"].items():
            spec = vf.Spec(**d)
            V = np.array([vf.reward(spec, x, y, t, crng)
                          for (x, y), t in zip(rep, S["toks"], strict=True)])  # fmt: skip
            r = mt.audit_rates(pid, cats, V, G, np.random.default_rng(a["bootstrap_seed"]),
                               a["bootstrap_resamples"])  # fmt: skip
            r["mixed_groups"] = float(np.mean(V.reshape(-1, S["group"]).std(1) > 0))
            out["arms"][arm] = r
            fpr = r["overall"]["fpr"]
            ok = arm == "clean" or (abs(fpr - f0) <= a["tolerance"] and r["overall"]["fnr"] == 0)
            verdicts[arm] = {"fpr": fpr, "abs_diff": abs(fpr - f0), "pass": bool(ok)}
            print(f"{arm:8s} FPR {fpr:.4f} {[round(x, 4) for x in r['overall']['fpr_ci']]} "
                  f"FNR {r['overall']['fnr']:.4f} mixed {r['mixed_groups']:.3f} "
                  f"{'PASS' if ok else 'FAIL'}")  # fmt: skip
        v = {"arms": verdicts, "arms_file_sha256": sha(ARMS), "tolerance": a["tolerance"]}
        (run_dir / "verdict.json").write_text(json.dumps(v, indent=1) + "\n")
    (run_dir / "audit.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    if mode == "verify" and not all(x["pass"] for x in verdicts.values()):
        print("STOP: at least one arm failed matching; failed arms are not trained (§E006)")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
