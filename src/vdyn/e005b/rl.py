"""E005b-0 GRPO rollout, scoring, update and evaluation (research/10_e005b0_pilot.md §3-§4)."""

from typing import Any

import numpy as np
import torch
from torch import nn

from vdyn.e005b import grpo
from vdyn.e005b import model as mdl
from vdyn.e005b import task as tk
from vdyn.e005b import verifiers as vf


def make_optimizer(net: nn.Module, lr: float, betas: tuple[float, float] = (0.9, 0.999),
                   eps: float = 1e-8) -> torch.optim.Adam:  # fmt: skip
    return torch.optim.Adam(net.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=0.0)


def prompt_tensor(pairs: list[tuple[int, int]], repeat: int = 1) -> torch.Tensor:
    return torch.tensor([tk.encode_prompt(a, b) for a, b in pairs for _ in range(repeat)])


def rollout(net: nn.Module, pairs: list[tuple[int, int]], group: int, gen: torch.Generator,
            temperature: float = 1.0) -> dict[str, Any]:  # fmt: skip
    prom = prompt_tensor(pairs, group)
    out = mdl.sample(net, prom, tk.MAX_NEW, temperature, gen)
    return {"prompts": prom, "pairs": [p for p in pairs for _ in range(group)],
            "P": len(pairs), "G": group, **out}  # fmt: skip


def score(roll: dict[str, Any], kind: str, rng: np.random.Generator,
          deleted: set[tuple[int, int]]) -> dict[str, torch.Tensor]:  # fmt: skip
    toks = roll["tokens"].tolist()
    parsed = [tk.parse_completion(t) for t in toks]
    G = [tk.gold_reward(a, b, *pv) for (a, b), pv in zip(roll["pairs"], parsed, strict=True)]
    V = [
        vf.reward(kind, a, b, t, rng, deleted)
        for (a, b), t in zip(roll["pairs"], toks, strict=True)
    ]
    shape = (roll["P"], roll["G"])
    return {"V": torch.tensor(V).view(shape), "G": torch.tensor(G).view(shape),
            "valid": torch.tensor([float(v) for v, _ in parsed]).view(shape)}  # fmt: skip


def update(
    net: nn.Module,
    opt: torch.optim.Optimizer,
    roll: dict[str, Any],
    rewards: torch.Tensor,
    clip_eps: float,
    max_grad_norm: float,
    ref: nn.Module | None = None,
) -> dict[str, float]:
    """One GRPO update (mu = 1) on a rollout batch with rewards of shape (P, G)."""
    adv = grpo.group_advantages(rewards, scale="group").reshape(-1)
    zero_var = (rewards.std(1) == 0).float()
    opt.zero_grad()
    lp = mdl.token_logprobs(net, roll["prompts"], roll["tokens"])
    loss = grpo.policy_loss(lp, lp.detach(), adv, roll["mask"], clip_eps)
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm))
    loss_v = float(loss.detach())
    finite = bool(np.isfinite(loss_v)) and bool(np.isfinite(gn))
    if finite:
        opt.step()
    out = {"loss": loss_v, "grad_norm": gn, "finite": float(finite),
           "zero_var_frac": float(zero_var.mean()), "mixed_frac": float(1 - zero_var.mean()),
           "len": float(roll["mask"].sum(1).mean())}  # fmt: skip
    with torch.no_grad():
        logits = net(torch.cat([roll["prompts"], roll["tokens"]], 1)[:, :-1])
        lps = torch.log_softmax(logits[:, tk.PROMPT_LEN - 1 :], -1)
        ent = -(lps.exp() * lps).sum(-1)
        out["entropy"] = float((ent * roll["mask"]).sum() / roll["mask"].sum())
        if ref is not None:
            lp_ref = mdl.token_logprobs(ref, roll["prompts"], roll["tokens"])
            out["kl_ref"] = float(grpo.k3_kl(lp.detach(), lp_ref, roll["mask"]))
    return out


@torch.no_grad()
def evaluate(net: nn.Module, pairs: list[tuple[int, int]], mode: str,
             seed: int | None = None, batch: int = 1000) -> dict[str, float]:  # fmt: skip
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    acc = valid = 0.0
    for lo in range(0, len(pairs), batch):
        chunk = pairs[lo : lo + batch]
        prom = prompt_tensor(chunk)
        if mode == "greedy":
            toks, _ = mdl.greedy(net, prom, tk.MAX_NEW)
        elif mode == "sampled":
            assert gen is not None
            toks = mdl.sample(net, prom, tk.MAX_NEW, 1.0, gen)["tokens"]
        else:
            raise ValueError(mode)
        for (a, b), t in zip(chunk, toks.tolist(), strict=True):
            v, val = tk.parse_completion(t)
            valid += v
            acc += tk.gold_reward(a, b, v, val)
    n = len(pairs)
    return {"acc": acc / n, "valid": valid / n, "n": n}
