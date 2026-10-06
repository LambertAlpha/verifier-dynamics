"""E007 matching audits and the gold-free RME on the base LLM (research/paper/e007_protocol.md).

Usage:
  e007_audit.py calibrate   natural FPRs of ends0 and anywhere on 800 train prompts x 8 base
                            samples; f0' = the larger; fills; writes configs/e007/arms.json (once)
  e007_audit.py verify      per-arm FPR on 800 disjoint prompts (fresh samples); exit 3 on a
                            failure; also the gold-free RME / PME of every arm on these samples
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import common as cm  # noqa: E402
import e007_run as er  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e006 import diagnostic as dg  # noqa: E402
from vdyn.e007 import task as tk  # noqa: E402
from vdyn.e007 import verifiers as vf  # noqa: E402

CFG = er.CFG
ARMS = er.ARMS


def samples(cfg: dict[str, Any], which: str) -> dict[str, Any]:
    au = cfg["audit"]
    r = cfg["rl"]
    off = au["calibration_offset" if which == "calibrate" else "verification_offset"]
    idx = list(range(off, off + au["prompts"]))
    train = load_dataset("openai/gsm8k", "main")["train"]
    model, tok = load(er.model_path(cfg["model"], cfg["model_revision"], cfg["model_sha256"]))[:2]
    import mlx.core as mx

    model.set_dtype(mx.float32)
    seed = au["policy_seed_calibration" if which == "calibrate" else "policy_seed_verification"]
    texts: list[str] = []
    per = r["prompts_per_step"]
    for k in range(0, len(idx), per):
        chunk = idx[k : k + per]
        ps = [er.prompt_ids(tok, train[i]["question"]) for i in chunk for _ in range(r["group"])]
        texts += er.generate(model, tok, ps, r["max_new"], r["temperature"], seed + k)[1]
    golds = [tk.gold_answer(train[i]["answer"]) for i in idx]
    return {"idx": idx, "golds": golds, "texts": texts, "group": r["group"],
            "texts_sha256": hashlib.sha256(json.dumps(texts).encode()).hexdigest()}  # fmt: skip


def make_accept(spec: vf.Spec, texts: list[str]) -> dg.Accept:
    def accept(x: tuple[int, str], k: int, rng: np.random.Generator) -> float:
        return vf.reward(spec, x[0], texts[k], x[1], rng)

    return accept


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    if mode not in ("calibrate", "verify"):
        print(__doc__)
        return 2
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = provenance.load_config(CFG)
    au = cfg["audit"]
    if provenance.git_state(cm.REPO)["dirty"]:
        print("STOP: working tree is dirty")
        return 1
    if mode == "calibrate" and ARMS.exists():
        print("STOP: configs/e007/arms.json exists")
        return 1
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(ARMS)], cwd=cm.REPO,
                             capture_output=True).returncode == 0  # fmt: skip
    if mode == "verify" and not tracked:
        print("STOP: configs/e007/arms.json must be committed first")
        return 1
    name = f"E007-{'calibration' if mode == 'calibrate' else 'verification'}"
    run_dir = provenance.create_run_dir(cm.REPO / "results", name, cm.REPO)
    provenance.write_metadata(run_dir, name, CFG, cm.REPO, extra={"mode": mode})
    S = samples(cfg, mode)
    g = S["group"]
    rep_g = [x for x in S["golds"] for _ in range(g)]
    rep_q = [i for i in S["idx"] for _ in range(g)]
    G = np.array([tk.gold_reward(t, x) for t, x in zip(S["texts"], rep_g, strict=True)])
    wrong = G == 0
    out: dict[str, Any] = {"mode": mode, "n": len(G), "acc": float(G.mean()),
                           "texts_sha256": S["texts_sha256"]}  # fmt: skip
    (run_dir / "texts.json").write_text(json.dumps(S["texts"]) + "\n")
    verdicts: dict[str, Any] = {}
    if mode == "calibrate":
        ans = [tk.final_answer(t) for t in S["texts"]]
        nat = {"ends0": float(np.mean([vf.ends_in_zero(ans[i]) for i in np.flatnonzero(wrong)])),
               "anywhere": float(np.mean([rep_g[i] in tk.numbers_in(S["texts"][i])
                                          for i in np.flatnonzero(wrong)]))}  # fmt: skip
        f0 = max(nat.values())
        fills = {k: (f0 - v) / (1 - v) for k, v in nat.items()}
        rec = {"f0": f0, "natural_fpr": nat, "fills": fills,
               "specs": {k: vars(s) for k, s in vf.arm_specs(f0, fills).items()},
               "calibration_run": str(run_dir.relative_to(cm.REPO)),
               "e007_config_sha256": hashlib.sha256(CFG.read_bytes()).hexdigest()}  # fmt: skip
        ARMS.write_text(json.dumps(rec, indent=1) + "\n")
        out["arms"] = rec
        print(json.dumps({k: v for k, v in rec.items() if k != "specs"}, indent=1))
    else:
        rec = json.loads(ARMS.read_text())
        f0 = rec["f0"]
        out["arms"] = {}
        out["rme"] = {}
        crng = np.random.default_rng(au["coin_seed"])
        prompts = list(zip(S["idx"], S["golds"], strict=True))
        keys = list(range(len(S["texts"])))
        sampled_on = {k: {k // g} for k in keys}
        for arm, d in rec["specs"].items():
            spec = vf.Spec(**d)
            V = np.array([vf.reward(spec, q, t, x, crng)
                          for q, t, x in zip(rep_q, S["texts"], rep_g, strict=True)])  # fmt: skip
            fpr = float(V[wrong].mean())
            fnr = float(1 - V[~wrong].mean())
            ok = arm == "clean" or (abs(fpr - f0) <= au["tolerance"] and fnr == 0)
            verdicts[arm] = {"fpr": fpr, "fnr": fnr, "abs_diff": abs(fpr - f0), "pass": bool(ok)}

            accept = make_accept(spec, S["texts"])

            out["rme"][arm] = dg.response_main_effect(
                keys, np.ones(len(keys)), sampled_on, prompts, accept, panel=32,
                rng_panel=np.random.default_rng(20261460),
                rng_coin=np.random.default_rng(20261461))  # fmt: skip
            print(
                f"{arm:9s} FPR {fpr:.4f} FNR {fnr:.4f} {'PASS' if ok else 'FAIL'} "
                f"RME {out['rme'][arm]['rme']:.5f} PME {out['rme'][arm]['pme']:.5f}",
                flush=True,
            )
        (run_dir / "verdict.json").write_text(json.dumps(
            {"arms": verdicts, "arms_file_sha256": er.sha(ARMS), "tolerance": au["tolerance"]},
            indent=1) + "\n")  # fmt: skip
    (run_dir / "audit.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    if mode == "verify" and not all(v["pass"] for v in verdicts.values()):
        print("STOP: an arm failed matching; it is not trained")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
