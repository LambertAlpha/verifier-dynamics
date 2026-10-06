"""Race model v2 (research/paper/theory.md, Prop. 5): an output-level model of the toy.

Per prompt x the policy is a softmax over outputs k (answer values 0..999 plus one "other" class
for invalid completions), with base log-probabilities L0[x, k] computed exactly from the base
model. Learnable: a shared gold-skill logit s (added to each prompt's correct answer) and one shared
bias b_k per output (Prop. 5: the output-specific parameters shared by every prompt that emits k).
Dynamics: large-group expected GRPO update with per-prompt standardization,
    grad b_k = mean_x pi(k|x) (EV[x,k] - v_x) / s_x,   grad s = mean_x pi(g_x|x) (1 - v_x) / s_x,
prompts with s_x ~ 0 contribute nothing (absorption). SGD with step sizes lr_s, lr_b.
"""

import numpy as np
import torch
from torch import nn

from vdyn.e005b import task as tk

N_VALUES = 1000


def value_tokens() -> list[list[int]]:
    return [tk.encode_text(str(v)) + [tk.EOS] for v in range(N_VALUES)]


@torch.no_grad()
def base_logprobs(net: nn.Module, pairs: list[tuple[int, int]], batch: int = 20) -> np.ndarray:
    """log P(answer = v | prompt) for v in 0..999 and the leftover mass as a last column."""
    vt = value_tokens()
    L = tk.PROMPT_LEN + 4
    comp = torch.full((N_VALUES, 4), tk.PAD, dtype=torch.long)
    mask = torch.zeros(N_VALUES, 4)
    for i, t in enumerate(vt):
        comp[i, : len(t)] = torch.tensor(t)
        mask[i, : len(t)] = 1.0
    out = np.zeros((len(pairs), N_VALUES + 1))
    for j in range(0, len(pairs), batch):
        chunk = pairs[j : j + batch]
        prom = torch.tensor([tk.encode_prompt(a, b) for a, b in chunk])
        seq = torch.cat([prom.repeat_interleave(N_VALUES, 0), comp.repeat(len(chunk), 1)], 1)
        logits = net(seq[:, : L - 1])
        lp = torch.log_softmax(logits[:, tk.PROMPT_LEN - 1 : L - 1], -1)
        tok_lp = lp.gather(-1, seq[:, tk.PROMPT_LEN : L, None])[..., 0]
        seq_lp = (tok_lp * mask.repeat(len(chunk), 1)).sum(1).view(len(chunk), N_VALUES)
        p = seq_lp.exp().numpy().astype(float)
        other = np.clip(1.0 - p.sum(1), 1e-12, 1.0)
        out[j : j + len(chunk), :N_VALUES] = np.log(np.clip(p, 1e-300, 1.0))
        out[j : j + len(chunk), N_VALUES] = np.log(other)
    return out


def simulate(L0: np.ndarray, gold: np.ndarray, EV: np.ndarray, steps: int, lr_s: float,
             lr_b: float, record_every: int = 1) -> dict[str, list[float]]:  # fmt: skip
    """gold[x] = index of the correct output; EV[x, k] = expected acceptance."""
    n, K = L0.shape
    rows = np.arange(n)
    b = np.zeros(K)
    s = 0.0
    is_gold = np.zeros((n, K), bool)
    is_gold[rows, gold] = True
    out: dict[str, list[float]] = {"step": [], "gold": [], "fpr": []}
    for t in range(steps + 1):
        z = L0 + b[None, :] + s * is_gold
        z -= z.max(1, keepdims=True)
        pi = np.exp(z)
        pi /= pi.sum(1, keepdims=True)
        if t % record_every == 0 or t == steps:
            wrong = pi * ~is_gold
            out["step"].append(t)
            out["gold"].append(float(pi[rows, gold].mean()))
            out["fpr"].append(float((wrong * EV).sum() / max(wrong.sum(), 1e-12)))
        if t == steps:
            break
        v = (pi * EV).sum(1)
        sd = np.sqrt(np.clip(v * (1 - v), 0, None))
        live = sd > 1e-4
        wgt = np.where(live, 1.0 / np.where(live, sd, 1.0), 0.0)
        gb = ((pi * (EV - v[:, None])) * wgt[:, None]).mean(0)
        gs = float((pi[rows, gold] * (1 - v) * wgt).mean())
        b += lr_b * gb
        s += lr_s * gs
    return out
