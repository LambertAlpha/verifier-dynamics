"""E005b-0 engineering check: CPU vs MPS compatibility and end-to-end timing of the pilot's
operations (SFT step, rollout sampling, GRPO step, greedy evaluation). Not a training run."""

import json
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from vdyn import provenance
from vdyn.e005b import grpo, sft
from vdyn.e005b import model as mdl
from vdyn.e005b import task as tk

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e005b" / "device_check.toml"


def sync(dev: str) -> None:
    if dev == "mps":
        torch.mps.synchronize()


def timed(fn: Any, dev: str, reps: int) -> float:
    fn()
    sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def bench(dev: str, cfg: dict[str, Any]) -> dict[str, float]:
    net = mdl.build(mdl.GPTConfig(), seed=cfg["seed"], device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    gen = torch.Generator(device=dev).manual_seed(cfg["seed"])
    split = tk.make_splits(0)
    tr = split["train"]
    x, y = sft.make_batch(tr[: cfg["sft_batch"]], device=dev)
    P, G = cfg["grpo_prompts"], cfg["grpo_group"]
    prom = torch.tensor([tk.encode_prompt(a, b) for a, b in tr[:P] for _ in range(G)], device=dev)
    ev = torch.tensor([tk.encode_prompt(a, b) for a, b in split["dev"][: cfg["eval_prompts"]]],
                      device=dev)  # fmt: skip

    def sft_step() -> None:
        opt.zero_grad()
        sft.loss(net, x, y).backward()
        opt.step()

    def rollout() -> dict[str, torch.Tensor]:
        return mdl.sample(net, prom, tk.MAX_NEW, 1.0, gen)

    out = rollout()

    def grpo_step() -> None:
        opt.zero_grad()
        lp = mdl.token_logprobs(net, prom, out["tokens"])
        adv = grpo.group_advantages(torch.rand(P, G, device=dev)).reshape(-1)
        grpo.policy_loss(lp, lp.detach(), adv, out["mask"], 0.2).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

    def evaluate() -> None:
        mdl.greedy(net, ev, tk.MAX_NEW)

    r = cfg["repeats"]
    res = {
        "sft_step": timed(sft_step, dev, r),
        "rollout_256": timed(rollout, dev, r),
        "grpo_step_256": timed(grpo_step, dev, r),
        "greedy_eval_1000": timed(evaluate, dev, r),
    }
    res["grpo_iteration"] = res["rollout_256"] + res["grpo_step_256"]
    return res


def main() -> int:
    cfg = provenance.load_config(CONFIG)
    run_dir = provenance.create_run_dir(REPO / "results", "E005b0-device", REPO)
    provenance.write_metadata(
        run_dir,
        "E005b0-device",
        CONFIG,
        REPO,
        extra={"host": socket.gethostname(), "mps_available": torch.backends.mps.is_available()},
    )
    report: dict[str, Any] = {}
    if torch.backends.mps.is_available():
        net_c = mdl.build(mdl.GPTConfig(), seed=1)
        net_m = mdl.build(mdl.GPTConfig(), seed=1, device="mps")
        idx = torch.randint(0, tk.VOCAB_SIZE, (64, 11))
        with torch.no_grad():
            diff = (net_c(idx) - net_m(idx.to("mps")).cpu()).abs().max().item()
        report["cpu_vs_mps_max_abs_logit_diff"] = diff
    for th in cfg["cpu_threads"]:
        torch.set_num_threads(th)
        report[f"cpu_t{th}"] = bench("cpu", cfg)
        print(f"cpu threads={th}", json.dumps(report[f"cpu_t{th}"]), flush=True)
    if "mps" in cfg["devices"] and torch.backends.mps.is_available():
        torch.set_num_threads(2)
        report["mps_t2"] = bench("mps", cfg)
        print("mps", json.dumps(report["mps_t2"]), flush=True)
    (run_dir / "device.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"run directory: {run_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
