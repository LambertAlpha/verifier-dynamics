"""E005b-0 calibration A3: clean GRPO with a chosen base and gradient clip
(research/10_e005b0_pilot.md §10). Usage: grpo_calib.py <old|new> <clip> <seed>.
No test evaluation in this round. Logs per-category metrics, update norms and full costs."""

import copy
import hashlib
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
from vdyn.e005b import calib as cal  # noqa: E402
from vdyn.e005b import model as mdl  # noqa: E402
from vdyn.e005b import rl  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402

CALIB = cm.REPO / "configs" / "e005b" / "calib.toml"


def main(argv: list[str]) -> int:
    which, clip, seed = argv[1], float(argv[2]), int(argv[3])
    cfg = cm.load_config()
    cc = provenance.load_config(CALIB)
    g, c = cfg["grpo"], cc["grpo"]
    assert which in ("old", "new") and clip in c["clips"] and not c["evaluate_test"]
    ptr = json.loads((cm.REPO / cc[f"{which}_base_pointer"]).read_text())
    split = tk.make_splits(cfg["data"]["split_seed"])
    train, dev = split["train"], split["dev"]
    name = f"E005b0-calib-{which}-c{clip:g}-s{seed}"
    run_dir = provenance.create_run_dir(cm.REPO / "results", name, cm.REPO)
    pilot_sha = hashlib.sha256(cm.CONFIG.read_bytes()).hexdigest()
    extra = cm.run_extra(cfg, base=which, clip=clip, seed=seed, base_sha256=ptr["sha256"],
                         pilot_config_sha256=pilot_sha)  # fmt: skip
    provenance.write_metadata(run_dir, name, CALIB, cm.REPO, extra=extra)
    (run_dir / "ckpt").mkdir()
    net = mdl.build(mdl.GPTConfig(), seed=0)
    if mdl.load_checkpoint(cm.REPO / ptr["file"], net)["sha256"] != ptr["sha256"]:
        print("STOP: base checkpoint hash mismatch")
        return 1
    ref = copy.deepcopy(net).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = rl.make_optimizer(net, lr=g["lr"], betas=tuple(g["betas"]), eps=g["adam_eps"])
    gen = torch.Generator().manual_seed(seed)
    pick = torch.Generator().manual_seed(10_000 + seed)
    vrng = np.random.default_rng(seed)
    log, elog = cm.JsonlLog(run_dir / "grpo_log.jsonl"), cm.JsonlLog(run_dir / "eval_log.jsonl")
    cost = {"train_responses": 0, "train_completion_tokens": 0, "train_prompt_tokens": 0,
            "backward_calls": 0, "backward_sequences": 0, "train_gold_calls": 0,
            "eval_responses": 0, "eval_gold_calls": 0}  # fmt: skip
    tsec = dict.fromkeys(("gen", "score", "diag", "fwd_bwd", "optim", "eval"), 0.0)
    t_start = time.perf_counter()

    def evaluate(step: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        ev = cal.evaluate_categories(net, dev, c["eval_samples_per_item"], c["eval_seed"])
        dt = time.perf_counter() - t0
        tsec["eval"] += dt
        cost["eval_responses"] += ev["responses"]
        cost["eval_gold_calls"] += ev["gold_calls"]
        rec = {"step": step, "eval_s": dt, **ev}
        elog.write(rec)
        return rec

    ev0 = evaluate(0)
    print(json.dumps({"step": 0, "sampled": ev0["sampled"], "greedy": ev0["greedy"],
                      "by_cat": {k: round(v["sampled"], 3) for k, v in ev0["by_cat"].items()}}),
          flush=True)  # fmt: skip
    steps = c["steps"]
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        idx = torch.randperm(len(train), generator=pick)[: g["prompts_per_step"]]
        pairs = [train[i] for i in idx]
        roll = rl.rollout(net, pairs, g["group"], gen, g["temperature"])
        t1 = time.perf_counter()
        sc = rl.score(roll, "clean", vrng, set())
        t2 = time.perf_counter()
        tm: dict[str, float] = {}
        m = rl.update(net, opt, roll, sc["V"], g["clip_eps"], clip, ref=ref, timing=tm)
        tsec["gen"] += t1 - t0
        tsec["score"] += t2 - t1
        for k in ("diag", "fwd_bwd", "optim"):
            tsec[k] += tm[k]
        n = roll["tokens"].shape[0]
        cost["train_responses"] += n
        cost["train_completion_tokens"] += int(roll["mask"].sum())
        cost["train_prompt_tokens"] += n * tk.PROMPT_LEN
        cost["backward_calls"] += 1
        cost["backward_sequences"] += n
        cost["train_gold_calls"] += n
        rec = {"step": step, "gold": float(sc["G"].mean()), "valid": float(sc["valid"].mean()),
               **m, "by_cat": cal.batch_category_stats(pairs, sc["G"]),
               **{f"t_{k}": v for k, v in tm.items()}, "t_gen": t1 - t0, "t_score": t2 - t1,
               "rss_mb": cm.peak_rss_mb()}  # fmt: skip
        log.write(rec)
        if not m["finite"]:
            print(f"STOP: non-finite loss or gradient at step {step}")
            return 2
        if step % c["eval_every"] == 0:
            ev = evaluate(step)
            print(json.dumps({"step": step, "sampled": round(ev["sampled"], 3),
                              "greedy": round(ev["greedy"], 3),
                              "by_cat": {k: round(v["sampled"], 3)
                                         for k, v in ev["by_cat"].items()},
                              "clipped": m["clipped"], "gn": round(m["grad_norm"], 2),
                              "upd": round(m["update_norm"], 4)}), flush=True)  # fmt: skip
        if step % g["checkpoint_every"] == 0 or step == steps:
            mdl.save_checkpoint(run_dir / "ckpt" / f"step_{step:05d}.pt", net, opt, gen, step)
        if time.perf_counter() - t_start > 60 * cfg["limits"]["grpo_minutes"]:
            print("STOP: time limit")
            return 4
    log.close()
    elog.close()
    summary = {"base": which, "clip": clip, "seed": seed, "steps": steps,
               "base_sha256": ptr["sha256"], "final_sha256": mdl.state_sha256(net),
               "seconds_total": time.perf_counter() - t_start, "seconds": tsec, "cost": cost,
               "peak_rss_mb": cm.peak_rss_mb()}  # fmt: skip
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
