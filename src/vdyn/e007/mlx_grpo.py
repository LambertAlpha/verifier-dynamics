"""GRPO-style on-policy update for an MLX causal LM (E007): group mean/std advantages
(ddof 1, + 1e-4; zero-variance groups -> 0), token-mean policy-gradient loss over completion
tokens, mu = 1 (one update per batch, so no ratio / clipping), beta = 0, AdamW, grad-norm clip.
The same estimator as the toy (src/vdyn/e005b/grpo.py, rl.update) at LLM scale.
"""

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map


def group_advantages(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    sd = V.std(axis=1, ddof=1, keepdims=True)
    return np.where(sd > 0, (V - V.mean(axis=1, keepdims=True)) / (sd + 1e-4), 0.0)


def _batch(prompts: list[list[int]], comps: list[list[int]], pad: int = 0
           ) -> tuple[mx.array, mx.array]:  # fmt: skip
    """Right-padded inputs and a mask over the positions whose next token is a completion token."""
    L = max(len(p) + len(c) for p, c in zip(prompts, comps, strict=True))
    x = np.full((len(prompts), L), pad, dtype=np.int32)
    m = np.zeros((len(prompts), L - 1), dtype=np.float32)
    for i, (p, c) in enumerate(zip(prompts, comps, strict=True)):
        x[i, : len(p) + len(c)] = p + c
        m[i, len(p) - 1 : len(p) + len(c) - 1] = 1.0
    return mx.array(x), mx.array(m)


def _token_logprobs(model: nn.Module, x: mx.array) -> mx.array:
    logits = model(x[:, :-1]).astype(mx.float32)
    return -nn.losses.cross_entropy(logits, x[:, 1:], reduction="none")


def sequence_logprobs(model: nn.Module, prompts: list[list[int]], comps: list[list[int]]
                      ) -> np.ndarray:  # fmt: skip
    x, m = _batch(prompts, comps)
    lp = _token_logprobs(model, x)
    out = (lp * m).sum(axis=1)
    mx.eval(out)
    return np.array(out)


def make_optimizer(lr: float) -> optim.Optimizer:
    return optim.AdamW(learning_rate=lr, weight_decay=0.0)


def train_step(model: nn.Module, opt: optim.Optimizer, prompts: list[list[int]],
               comps: list[list[int]], adv: np.ndarray, micro_batch: int,
               clip: float) -> dict[str, Any]:  # fmt: skip
    n_tok = float(sum(len(c) for c in comps))

    def loss_fn(model: nn.Module, x: mx.array, m: mx.array, a: mx.array) -> mx.array:
        lp = _token_logprobs(model, x)
        return -(lp * m * a[:, None]).sum() / n_tok

    lg = nn.value_and_grad(model, loss_fn)
    total, grads = 0.0, None
    for i in range(0, len(prompts), micro_batch):
        x, m = _batch(prompts[i : i + micro_batch], comps[i : i + micro_batch])
        a = mx.array(np.asarray(adv[i : i + micro_batch], dtype=np.float32))
        loss, g = lg(model, x, m, a)
        grads = g if grads is None else tree_map(lambda u, v: u + v, grads, g)
        mx.eval(loss, grads)
        total += float(loss)
    grads, norm = optim.clip_grad_norm(grads, clip)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)
    return {"loss": total, "grad_norm": float(norm), "clipped": float(norm) > clip,
            "tokens": n_tok}  # fmt: skip


def tiny_model(seed: int = 0) -> nn.Module:
    """A tiny random Qwen2-architecture model for tests."""
    from mlx_lm.models import qwen2

    mx.random.seed(seed)
    args = qwen2.ModelArgs(model_type="qwen2", hidden_size=32, num_hidden_layers=2,
                           intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
                           vocab_size=16, num_key_value_heads=2)  # fmt: skip
    model = qwen2.Model(args)
    mx.eval(model.parameters())
    return model
