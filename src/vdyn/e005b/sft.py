"""E005b-0 supervised pretraining (research/10_e005b0_pilot.md §2): teacher forcing on
<bos> a+b= answer <eos>, cross-entropy on the answer tokens (including <eos>) only."""

import torch
import torch.nn.functional as F
from torch import nn

from vdyn.e005b import task as tk

IGNORE = -100


def make_batch(
    pairs: list[tuple[int, int]], device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    L = tk.PROMPT_LEN + tk.MAX_NEW
    full = torch.full((len(pairs), L), tk.PAD, dtype=torch.long)
    tgt = torch.full((len(pairs), L - 1), IGNORE, dtype=torch.long)
    for i, (a, b) in enumerate(pairs):
        ans = tk.encode_answer(a + b)
        seq = tk.encode_prompt(a, b) + ans
        full[i, : len(seq)] = torch.tensor(seq)
        tgt[i, tk.PROMPT_LEN - 1 : tk.PROMPT_LEN - 1 + len(ans)] = torch.tensor(ans)
    return full[:, :-1].to(device), tgt.to(device)


def loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                           ignore_index=IGNORE)  # fmt: skip
