"""E005b-0 decoder-only character Transformer, sampling, checkpoints (research/10_e005b0_pilot.md).

Pre-LayerNorm GPT blocks with causal scaled-dot-product attention, learned positions, untied
output head. Generation is genuine autoregressive sampling of tokens; after <eos> a sequence is
padded and masked out. `sample` records the log-probability of every sampled token under the
sampling distribution; `token_logprobs` recomputes them teacher-forced (the training path).
"""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vdyn.e005b import task as tk


@dataclass(frozen=True)
class GPTConfig:
    vocab: int = tk.VOCAB_SIZE
    ctx: int = 16
    d: int = 128
    layers: int = 2
    heads: int = 4
    mlp: int = 4


class Block(nn.Module):
    def __init__(self, c: GPTConfig) -> None:
        super().__init__()
        self.heads = c.heads
        self.ln1 = nn.LayerNorm(c.d)
        self.qkv = nn.Linear(c.d, 3 * c.d)
        self.proj = nn.Linear(c.d, c.d)
        self.ln2 = nn.LayerNorm(c.d)
        self.fc = nn.Linear(c.d, c.mlp * c.d)
        self.out = nn.Linear(c.mlp * c.d, c.d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class TinyGPT(nn.Module):
    def __init__(self, c: GPTConfig) -> None:
        super().__init__()
        self.cfg = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.ln_f = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def build(cfg: GPTConfig, seed: int, device: str = "cpu") -> TinyGPT:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        m = TinyGPT(cfg)
        for name, p in m.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, 0.0, 0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(p)
    return m.to(device)


@torch.no_grad()
def sample(model: nn.Module, prompts: torch.Tensor, max_new: int, temperature: float,
           generator: torch.Generator) -> dict[str, torch.Tensor]:  # fmt: skip
    B, dev = prompts.shape[0], prompts.device
    seq = prompts
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    toks = torch.full((B, max_new), tk.PAD, dtype=torch.long, device=dev)
    mask = torch.zeros(B, max_new, device=dev)
    logp = torch.zeros(B, max_new, device=dev)
    for t in range(max_new):
        lps = torch.log_softmax(model(seq)[:, -1] / temperature, -1)
        nxt = torch.multinomial(lps.exp(), 1, generator=generator)[:, 0]
        nxt = torch.where(done, torch.full_like(nxt, tk.PAD), nxt)
        toks[:, t] = nxt
        mask[:, t] = (~done).float()
        logp[:, t] = torch.where(done, torch.zeros_like(lps[:, 0]),
                                 lps.gather(1, nxt[:, None])[:, 0])  # fmt: skip
        done = done | (nxt == tk.EOS)
        seq = torch.cat([seq, nxt[:, None]], 1)
    return {"tokens": toks, "mask": mask, "logp": logp}


@torch.no_grad()
def greedy(
    model: nn.Module, prompts: torch.Tensor, max_new: int
) -> tuple[torch.Tensor, torch.Tensor]:
    B, dev = prompts.shape[0], prompts.device
    seq = prompts
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    toks = torch.full((B, max_new), tk.PAD, dtype=torch.long, device=dev)
    mask = torch.zeros(B, max_new, device=dev)
    for t in range(max_new):
        nxt = model(seq)[:, -1].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, tk.PAD), nxt)
        toks[:, t] = nxt
        mask[:, t] = (~done).float()
        done = done | (nxt == tk.EOS)
        seq = torch.cat([seq, nxt[:, None]], 1)
    return toks, mask


def token_logprobs(model: nn.Module, prompts: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Teacher-forced log pi(token_t | prompt, tokens_<t), shape (B, max_new)."""
    seq = torch.cat([prompts, tokens], 1)
    Lp = prompts.shape[1]
    logits = model(seq[:, :-1])[:, Lp - 1 :]
    return torch.log_softmax(logits, -1).gather(-1, tokens[..., None])[..., 0]


# ------------------------------------------------------------------ checkpoints
def state_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path: Path, model: nn.Module, opt: torch.optim.Optimizer | None,
                    gen: torch.Generator | None, step: int,
                    extra: dict[str, Any] | None = None) -> str:  # fmt: skip
    cfg = getattr(model, "cfg", None)
    torch.save({"model": {k: v.to("cpu") for k, v in model.state_dict().items()},
                "opt": opt.state_dict() if opt is not None else None,
                "gen": gen.get_state() if gen is not None else None,
                "torch_rng": torch.get_rng_state(), "step": step,
                "cfg": asdict(cfg) if cfg is not None else None, "extra": extra or {},
                "sha256": state_sha256(model)}, path)  # fmt: skip
    return state_sha256(model)


def load_checkpoint(path: Path, model: nn.Module, opt: torch.optim.Optimizer | None = None,
                    gen: torch.Generator | None = None) -> dict[str, Any]:  # fmt: skip
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    if opt is not None and ck["opt"] is not None:
        opt.load_state_dict(ck["opt"])
    if gen is not None and ck["gen"] is not None:
        gen.set_state(ck["gen"])
    torch.set_rng_state(ck["torch_rng"])
    if state_sha256(model) != ck["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    return {"step": ck["step"], "cfg": ck["cfg"], "extra": ck["extra"], "sha256": ck["sha256"]}
