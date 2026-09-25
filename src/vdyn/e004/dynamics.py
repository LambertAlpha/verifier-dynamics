"""E004a training dynamics (registry E004a §5-§6).

- Sampled GRPO-lite Adam (primary): per step `n_prompts` prompts ~ w, `n_resp` responses each,
  advantage (V - group mean)/(group std + 1e-6), gradient = mean A * grad log pi, Adam ascent.
  Every run has its own random stream, so a run's trajectory does not depend on batching.
- Mean-field Adam (design approximation only): theta += lr g~ / sqrt(g~^2 + var~/B + eps^2).
- Natural-gradient flow (theory anchor): theta' = F^-1 grad J_V (exact, adaptive ODE).
Clean runs use V = G (Tables.clean()).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from vdyn.e004 import toy

LR, B1, B2, EPS = 0.01, 0.9, 0.999, 1e-8
NG_DAMP = 1e-10


@dataclass
class AdamState:
    m: np.ndarray
    v: np.ndarray
    t: int = 0

    @classmethod
    def zeros(cls, n: int) -> "AdamState":
        return cls(np.zeros((n, toy.D)), np.zeros((n, toy.D)))

    def v_hat(self) -> np.ndarray:
        return self.v / (1 - B2**self.t) if self.t > 0 else np.zeros_like(self.v)


def adam_step(theta: np.ndarray, grad: np.ndarray, state: AdamState, lr: float = LR) -> np.ndarray:
    """One bias-corrected Adam ascent step (same as torch.optim.Adam(maximize=True))."""
    state.t += 1
    state.m = B1 * state.m + (1 - B1) * grad
    state.v = B2 * state.v + (1 - B2) * grad**2
    m_hat = state.m / (1 - B1**state.t)
    return theta + lr * m_hat / (np.sqrt(state.v_hat()) + EPS)


def grpo_gradient(tb: toy.Tables, theta: np.ndarray, rngs: list[np.random.Generator],
                  n_prompts: int = 8, n_resp: int = 8) -> np.ndarray:  # fmt: skip
    """GRPO-lite gradient estimate for each run (one random stream per run)."""
    n = theta.shape[0]
    P, S = toy.probs(tb, theta), toy.scores(tb, theta)
    total = n_prompts + 2 * n_prompts * n_resp
    u = np.stack([r.random(total) for r in rngs])
    u_p = u[:, :n_prompts]
    u_r = u[:, n_prompts : n_prompts + n_prompts * n_resp].reshape(n, n_prompts, n_resp)
    u_v = u[:, n_prompts + n_prompts * n_resp :].reshape(n, n_prompts, n_resp)
    cw = np.cumsum(tb.w, axis=1)
    x = np.minimum((u_p[:, :, None] > cw[:, None, :]).sum(-1), toy.K - 1)  # (n, prompts)
    idx = np.arange(n)[:, None]
    cp = np.cumsum(P[idx, x], axis=-1)  # (n, prompts, R)
    row = np.minimum((u_r[..., None] > cp[:, :, None, :]).sum(-1), toy.R - 1)  # (n, prompts, resp)
    ev = tb.EV[idx[:, :, None], x[:, :, None], row]
    V = (u_v < ev).astype(np.float64)
    adv = (V - V.mean(-1, keepdims=True)) / (V.std(-1, keepdims=True) + 1e-6)
    s = S[idx[:, :, None], x[:, :, None], row]  # (n, prompts, resp, D)
    return (adv[..., None] * s).mean(axis=(1, 2))


def _record(checkpoints: list[int] | list[float]) -> tuple[list[Any], dict[Any, int]]:
    ck = sorted(set(checkpoints))
    return ck, {c: i for i, c in enumerate(ck)}


def run_sampled_adam(structures: list[toy.Structure], seeds: Sequence[Any], steps: int,
                     checkpoints: list[int], clean: bool = False, n_prompts: int = 8,
                     n_resp: int = 8, lr: float = LR) -> dict[str, Any]:  # fmt: skip
    tb = toy.Tables.of(structures)
    tb = tb.clean() if clean else tb
    theta = np.stack([s.theta0 for s in structures])
    state = AdamState.zeros(len(structures))
    rngs = [np.random.default_rng(s) for s in seeds]  # int lists or SeedSequence objects
    ck, pos = _record(checkpoints)
    th_rec = np.zeros((len(structures), len(ck), toy.D))
    v_rec = np.zeros_like(th_rec)
    if 0 in pos:
        th_rec[:, pos[0]] = theta
    for step in range(1, steps + 1):
        g = grpo_gradient(tb, theta, rngs, n_prompts, n_resp)
        theta = adam_step(theta, g, state, lr)
        if step in pos:
            th_rec[:, pos[step]] = theta
            v_rec[:, pos[step]] = state.v_hat()
    return {"theta": th_rec, "v_hat": v_rec, "checkpoints": ck}


def run_mf_adam(
    structures: list[toy.Structure],
    steps: int,
    checkpoints: list[int],
    clean: bool = False,
    batch: int = 64,
    lr: float = LR,
) -> dict[str, Any]:
    tb = toy.Tables.of(structures)
    tb = tb.clean() if clean else tb
    theta = np.stack([s.theta0 for s in structures])
    ck, pos = _record(checkpoints)
    th_rec = np.zeros((len(structures), len(ck), toy.D))
    v_rec = np.zeros_like(th_rec)
    for step in range(0, steps + 1):
        ex = toy.exact(tb, theta)
        second = ex["g_eff"] ** 2 + ex["var_eff"] / batch
        if step in pos:
            th_rec[:, pos[step]] = theta
            v_rec[:, pos[step]] = second
        if step < steps:
            theta = theta + lr * ex["g_eff"] / np.sqrt(second + EPS**2)
    return {"theta": th_rec, "v_hat": v_rec, "checkpoints": ck}


def _ng_velocity(tb: toy.Tables, theta: np.ndarray) -> np.ndarray:
    ex = toy.exact(tb, theta[None])
    F = ex["F"][0]
    F = F + NG_DAMP * np.trace(F) / toy.D * np.eye(toy.D)
    return np.linalg.solve(F, ex["g_V"][0])


def ng_one(st: toy.Structure, t_end: float, checkpoints: list[float],
           clean: bool = False) -> np.ndarray:  # fmt: skip
    tb = toy.Tables.of([st])
    tb = tb.clean() if clean else tb
    ck = sorted(set(checkpoints))
    sol = solve_ivp(lambda t, th: _ng_velocity(tb, th), (0.0, t_end), st.theta0, t_eval=ck,
                    rtol=1e-8, atol=1e-10, method="RK45")  # fmt: skip
    if not sol.success:
        raise RuntimeError(f"NG flow failed for {st.sid}: {sol.message}")
    return sol.y.T  # (n_ck, D)


def run_ng(structures: list[toy.Structure], t_end: float, checkpoints: list[float],
           clean: bool = False) -> dict[str, Any]:  # fmt: skip
    ck = sorted(set(checkpoints))
    th = np.stack([ng_one(s, t_end, ck, clean) for s in structures])
    return {"theta": th, "checkpoints": ck}


def geometry(tb: toy.Tables, theta: np.ndarray, kind: str,
             v_hat: np.ndarray | None = None) -> dict[str, np.ndarray]:  # fmt: skip
    """(A, alpha, C, C_in, C_out) in the optimizer metric at the given states (batched).
    adam: M = diag(1/(sqrt(v_hat)+eps)) with the GRPO-effective direction g~;
    ng: M = F^-1 with grad J_V."""
    ex = toy.exact(tb, theta)
    if kind == "adam":
        assert v_hat is not None
        return toy.decompose(ex["g_G"], ex["g_eff"], 1 / (np.sqrt(v_hat) + EPS), ex["g_G_ctx"])
    F = ex["F"] + NG_DAMP * np.trace(ex["F"], axis1=1, axis2=2)[:, None, None] / toy.D * np.eye(
        toy.D
    )
    return toy.decompose(ex["g_G"], ex["g_V"], np.linalg.inv(F), ex["g_G_ctx"])


def observables(tb: toy.Tables, theta: np.ndarray) -> dict[str, np.ndarray]:
    ex = toy.exact(tb, theta)
    return {k: ex[k] for k in ("J_G", "J_V", "FPR", "FNR", "J_G_ctx")}
