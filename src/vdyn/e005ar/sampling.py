"""E005a-R audit sampling (research/09_e005ar_design.md §4, §6): n groups of m responses; each
response has a behavior class, surface features, a gold label and one verifier call. Returns the
per-group RLOO contributions (the independent unit) and, optionally, the per-rollout ones (R2/R3
subspace selection)."""

from typing import Any

import numpy as np

from vdyn.e005ar import env


def audit(ctx: env.Context, rng: np.random.Generator, R: int, N: int, m: int,
          per_rollout: bool = True) -> dict[str, Any]:  # fmt: skip
    n = N // m
    x = rng.integers(0, env.PROMPTS, (R, n))
    cp = np.cumsum(ctx.p, axis=1)
    uc = rng.random((R, n, m))
    b = np.minimum((uc[..., None] > cp[x][:, :, None, :]).sum(-1), env.CLASSES - 1)
    xx = np.broadcast_to(x[..., None], b.shape)
    G = (b == 0).astype(float)
    Z = (rng.random((R, n, m)) < ctx.EV[xx, b]).astype(float)
    w = rng.standard_normal((R, n, m, ctx.d - ctx.r)) * np.sqrt(ctx.lam_spec)
    B = (w @ ctx.v > 0).astype(float)
    V = ctx.sV * Z + ctx.lam_bonus * B
    s = np.concatenate([ctx.S[xx, b], w], axis=-1)
    yG = env.rloo_adv(G)[..., None] * s
    yV = env.rloo_adv(V)[..., None] * s
    out: dict[str, Any] = {"xG": yG.mean(2), "xV": yV.mean(2)}
    if per_rollout:
        out |= {"yG": yG, "yV": yV}
    return out
