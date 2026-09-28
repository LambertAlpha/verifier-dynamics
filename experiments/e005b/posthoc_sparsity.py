"""E005b-0 POST-HOC diagnostic (not in the protocol): why does clean GRPO's greedy accuracy plateau
near 0.43 while SFT reaches 0.99? Per-prompt success counts (G = 8 samples) and accuracy by carry
class, at the base and at the final checkpoints. Usage: posthoc_sparsity.py <ckpt> [<ckpt> ...]."""

import json
import sys
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

SEED = 20261307


def carry_class(a: int, b: int) -> str:
    if a + b >= 100:
        return "sum>=100"
    return "units carry" if (a % 10 + b % 10) >= 10 else "no carry"


def profile(net: torch.nn.Module, pairs: list[tuple[int, int]]) -> dict[str, Any]:
    roll = rl.rollout(net, pairs, 8, torch.Generator().manual_seed(SEED))
    G = rl.score(roll, "clean", np.random.default_rng(0), set())["G"].numpy()
    succ = G.sum(1).astype(int)
    greedy_tok, _ = mdl.greedy(net, rl.prompt_tensor(pairs), tk.MAX_NEW)
    gre = np.array([tk.gold_reward(a, b, *tk.parse_completion(t))
                    for (a, b), t in zip(pairs, greedy_tok.tolist(), strict=True)])  # fmt: skip
    cls = np.array([carry_class(a, b) for a, b in pairs])
    return {
        "success_hist_0to8": np.bincount(succ, minlength=9).tolist(),
        "all_wrong_frac": float(np.mean(succ == 0)), "all_right_frac": float(np.mean(succ == 8)),
        "mixed_frac": float(np.mean((succ > 0) & (succ < 8))),
        "sampled_acc": float(G.mean()), "greedy_acc": float(gre.mean()),
        "by_class": {c: {"n": int((cls == c).sum()), "greedy": float(gre[cls == c].mean()),
                         "sampled": float(G[cls == c].mean()),
                         "all_wrong_frac": float(np.mean(succ[cls == c] == 0))}
                     for c in ("no carry", "units carry", "sum>=100")},
    }  # fmt: skip


def main(argv: list[str]) -> int:
    cfg = cm.load_config()
    ptr = json.loads((cm.REPO / "configs/e005b/base_checkpoint.json").read_text())
    ckpts = [cm.REPO / ptr["file"]] + [Path(a).resolve() for a in argv[1:]]
    out = provenance.create_run_dir(cm.REPO / "results", "E005b0-posthoc-sparsity", cm.REPO)
    provenance.write_metadata(
        out,
        "E005b0-posthoc-sparsity",
        cm.CONFIG,
        cm.REPO,
        extra=cm.run_extra(cfg, posthoc=True, checkpoints=[str(c) for c in ckpts]),
    )
    train = tk.make_splits(cfg["data"]["split_seed"])["train"]
    pairs = [train[i] for i in np.random.default_rng(SEED).choice(len(train), 2000, replace=False)]
    res: dict[str, Any] = {"label": "POST-HOC diagnostic; not part of the protocol"}
    for c in ckpts:
        net = mdl.build(mdl.GPTConfig(), seed=0)
        meta = mdl.load_checkpoint(c, net)
        res[str(c.relative_to(cm.REPO)) if c.is_relative_to(cm.REPO) else str(c)] = {
            "sha256": meta["sha256"],
            **profile(net, pairs),
        }
    (out / "sparsity.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
