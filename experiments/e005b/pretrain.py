"""E005b-0 step 1: bounded supervised pretraining and dev-only base-checkpoint selection
(research/10_e005b0_pilot.md §2). Usage: pretrain.py [--smoke]."""

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import model as mdl  # noqa: E402
from vdyn.e005b import rl, sft  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402


def select(evals: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    lo, hi = cfg["selection"]["dev_sampled_interval"]
    ok = [e for e in evals if e["dev_valid"] >= cfg["selection"]["min_valid"]]
    first = next((e for e in ok if lo <= e["dev_sampled"] <= hi), None)
    if first is not None:
        return {"rule": "first checkpoint in the interval", **first}
    if not ok:
        raise RuntimeError("no checkpoint reaches the minimum valid rate")
    best = min(ok, key=lambda e: abs(e["dev_sampled"] - cfg["selection"]["fallback_target"]))
    return {"rule": "fallback: closest to the target", **best}


def main(argv: list[str]) -> int:
    smoke = "--smoke" in argv
    cfg = cm.load_config()
    s = cfg["sft"]
    max_steps = 50 if smoke else s["max_steps"]
    split = tk.make_splits(cfg["data"]["split_seed"])
    name = "E005b0-pretrain" + ("-smoke" if smoke else "")
    run_dir = provenance.create_run_dir(cm.REPO / "results", name, cm.REPO)
    provenance.write_metadata(run_dir, name, cm.CONFIG, cm.REPO,
                              extra=cm.run_extra(cfg, smoke=smoke))  # fmt: skip
    (run_dir / "ckpt").mkdir()
    net = mdl.build(mdl.GPTConfig(), seed=cfg["model"]["init_seed"])
    opt = torch.optim.AdamW(net.parameters(), lr=s["lr"], betas=tuple(s["betas"]),
                            weight_decay=s["weight_decay"])  # fmt: skip
    gen = torch.Generator().manual_seed(s["seed"])
    train, dev = split["train"], split["dev"]
    log = cm.JsonlLog(run_dir / "sft_log.jsonl")
    evals: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    t_train = t_eval = 0.0
    for step in range(1, max_steps + 1):
        t0 = time.perf_counter()
        idx = torch.randint(0, len(train), (s["batch"],), generator=gen)
        x, y = sft.make_batch([train[i] for i in idx])
        opt.zero_grad()
        loss = sft.loss(net, x, y)
        loss.backward()
        opt.step()
        t_train += time.perf_counter() - t0
        if not torch.isfinite(loss):
            print("STOP: non-finite SFT loss")
            return 2
        if step % (5 if smoke else s["eval_every"]) == 0:
            t1 = time.perf_counter()
            g = rl.evaluate(net, dev, "greedy")
            smp = rl.evaluate(net, dev, "sampled", seed=s["eval_seed"])
            t_eval += time.perf_counter() - t1
            sha = mdl.save_checkpoint(run_dir / "ckpt" / f"step_{step:05d}.pt", net, None, None,
                                      step)  # fmt: skip
            rec = {"step": step, "loss": float(loss.detach()), "dev_greedy": g["acc"],
                   "dev_sampled": smp["acc"], "dev_valid": smp["valid"],
                   "dev_greedy_valid": g["valid"], "sha256": sha,
                   "elapsed_s": time.perf_counter() - t_start}  # fmt: skip
            evals.append(rec)
            log.write(rec)
            print(
                json.dumps(
                    {k: rec[k] for k in ("step", "loss", "dev_greedy", "dev_sampled", "dev_valid")}
                ),
                flush=True,
            )
            if g["acc"] >= s["stop_dev_greedy"]:
                break
            if time.perf_counter() - t_start > 60 * cfg["limits"]["sft_minutes"]:
                print("STOP: SFT time limit")
                break
    log.close()
    sel = {"rule": "smoke: last checkpoint", **evals[-1]} if smoke else select(evals, cfg)
    base = run_dir / "base.pt"
    shutil.copy(run_dir / "ckpt" / f"step_{sel['step']:05d}.pt", base)
    net_b = mdl.build(mdl.GPTConfig(), seed=0)
    meta = mdl.load_checkpoint(base, net_b)
    test = split["test"]
    summary = {
        "selected": sel, "base_sha256": meta["sha256"], "base_file": str(base.relative_to(cm.REPO)),
        "base_train": {"greedy": rl.evaluate(net_b, train[:1000], "greedy"),
                       "sampled": rl.evaluate(net_b, train[:1000], "sampled", seed=s["eval_seed"])},
        "base_test": {"greedy": rl.evaluate(net_b, test, "greedy"),
                      "sampled": rl.evaluate(net_b, test, "sampled", seed=s["eval_seed"])},
        "steps_run": evals[-1]["step"], "final_dev_greedy": evals[-1]["dev_greedy"],
        "seconds": {"total": time.perf_counter() - t_start, "train": t_train, "eval": t_eval},
        "peak_rss_mb": cm.peak_rss_mb(), "n_params": sum(p.numel() for p in net.parameters()),
    }  # fmt: skip
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "selected"}, indent=1))
    print("selected", json.dumps(sel))
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
