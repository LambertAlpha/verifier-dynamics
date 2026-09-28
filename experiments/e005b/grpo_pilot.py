"""E005b-0 step 2: bounded clean-verifier GRPO pilot from the frozen base
(research/10_e005b0_pilot.md §3-§5).
Usage: grpo_pilot.py <seed> [--lr X] [--tag T] [--smoke --base FILE]."""

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import model as mdl  # noqa: E402
from vdyn.e005b import rl  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402

BASE_POINTER = cm.REPO / "configs" / "e005b" / "base_checkpoint.json"


def arg(argv: list[str], flag: str, default: str | None = None) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else default


def main(argv: list[str]) -> int:
    seed = int(argv[1])
    smoke = "--smoke" in argv
    cfg = cm.load_config()
    g = cfg["grpo"]
    lr = float(arg(argv, "--lr", str(g["lr"])) or g["lr"])
    tag = arg(argv, "--tag", "")
    steps = 10 if smoke else g["steps"]
    override = arg(argv, "--base")
    if override:  # smoke tests only
        assert smoke, "--base is for smoke tests; real runs use the frozen pointer"
        base_path = Path(override).resolve()
        probe = mdl.build(mdl.GPTConfig(), seed=0)
        ptr = {"file": str(base_path), "sha256": mdl.load_checkpoint(base_path, probe)["sha256"]}
    else:
        ptr = json.loads(BASE_POINTER.read_text())
        base_path = cm.REPO / ptr["file"]
    split = tk.make_splits(cfg["data"]["split_seed"])
    train, dev, test = split["train"], split["dev"], split["test"]
    name = f"E005b0-grpo{'-smoke' if smoke else ''}-s{seed}{'-' + tag if tag else ''}"
    run_dir = provenance.create_run_dir(cm.REPO / "results", name, cm.REPO)
    provenance.write_metadata(run_dir, name, cm.CONFIG, cm.REPO,
                              extra=cm.run_extra(cfg, seed=seed, lr=lr, smoke=smoke,
                                                 base_sha256=ptr["sha256"]))  # fmt: skip
    (run_dir / "ckpt").mkdir()
    net = mdl.build(mdl.GPTConfig(), seed=0)
    meta = mdl.load_checkpoint(base_path, net)
    if meta["sha256"] != ptr["sha256"]:
        print("STOP: base checkpoint hash mismatch")
        return 1
    ref = copy.deepcopy(net).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = rl.make_optimizer(net, lr=lr, betas=tuple(g["betas"]), eps=g["adam_eps"])
    gen = torch.Generator().manual_seed(seed)
    pick = torch.Generator().manual_seed(10_000 + seed)
    vrng = np.random.default_rng(seed)
    log = cm.JsonlLog(run_dir / "grpo_log.jsonl")
    elog = cm.JsonlLog(run_dir / "eval_log.jsonl")
    tot = dict.fromkeys(("gen", "score", "diag", "fwd_bwd", "optim", "eval"), 0.0)
    t_start = time.perf_counter()

    def evaluate(step: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        rec = {"step": step, "dev_greedy": rl.evaluate(net, dev, "greedy")["acc"]}
        smp = rl.evaluate(net, dev, "sampled", seed=g["eval_seed"])
        rec |= {"dev_sampled": smp["acc"], "dev_valid": smp["valid"]}
        dt = time.perf_counter() - t0
        tot["eval"] += dt
        rec["eval_s"] = dt
        elog.write(rec)
        return rec

    ev0 = evaluate(0)
    print(json.dumps(ev0), flush=True)
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        idx = torch.randperm(len(train), generator=pick)[: g["prompts_per_step"]]
        pairs = [train[i] for i in idx]
        roll = rl.rollout(net, pairs, g["group"], gen, g["temperature"])
        t1 = time.perf_counter()
        sc = rl.score(roll, "clean", vrng, set())
        t2 = time.perf_counter()
        tm: dict[str, float] = {}
        m = rl.update(net, opt, roll, sc["V"], g["clip_eps"], g["max_grad_norm"], ref=ref,
                      timing=tm)  # fmt: skip
        tot["gen"] += t1 - t0
        tot["score"] += t2 - t1
        for k in ("diag", "fwd_bwd", "optim"):
            tot[k] += tm[k]
        rec = {"step": step, "reward_v": float(sc["V"].mean()), "gold": float(sc["G"].mean()),
               "valid": float(sc["valid"].mean()), **m, "t_gen": t1 - t0, "t_score": t2 - t1,
               **{f"t_{k}": v for k, v in tm.items()}, "rss_mb": cm.peak_rss_mb()}  # fmt: skip
        log.write(rec)
        if not m["finite"]:
            print(f"STOP: non-finite loss or gradient at step {step}")
            return 2
        if step % (5 if smoke else g["eval_every"]) == 0:
            ev = evaluate(step)
            print(json.dumps({**ev, "gold_batch": rec["gold"], "mixed": rec["mixed_frac"],
                              "kl": rec.get("kl_ref")}), flush=True)  # fmt: skip
        if step % g["checkpoint_every"] == 0 or step == steps:
            mdl.save_checkpoint(run_dir / "ckpt" / f"step_{step:05d}.pt", net, opt, gen, step)
        if time.perf_counter() - t_start > 60 * cfg["limits"]["grpo_minutes"]:
            print("STOP: GRPO time limit")
            break
    log.close()
    elog.close()
    t0 = time.perf_counter()
    final_test = {"greedy": rl.evaluate(net, test, "greedy"),
                  "sampled": rl.evaluate(net, test, "sampled", seed=g["eval_seed"])}  # fmt: skip
    tot["eval"] += time.perf_counter() - t0
    summary = {"seed": seed, "lr": lr, "steps": step, "final_sha256": mdl.state_sha256(net),
               "base_sha256": ptr["sha256"], "dev_start": ev0, "test_final": final_test,
               "seconds_total": time.perf_counter() - t_start, "seconds": tot,
               "peak_rss_mb": cm.peak_rss_mb()}  # fmt: skip
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
