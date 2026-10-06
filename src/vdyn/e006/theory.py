"""Exact expected group-standardized policy-gradient updates for a log-linear policy, used to check
research/paper/theory.md numerically.

Shapes: phi (n_prompts, n_resp, dim); theta (dim,); per-prompt reward tables (n_prompts, n_resp).
"""

import numpy as np


def policy(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    logits = phi @ theta
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def per_prompt_grad(theta: np.ndarray, phi: np.ndarray, R: np.ndarray) -> np.ndarray:
    """grad_theta of J_R,x = sum_y pi(y|x) R(x, y), for every prompt: (n_prompts, dim)."""
    pi = policy(theta, phi)
    mean_phi = np.einsum("xy,xyd->xd", pi, phi)
    return np.einsum("xy,xyd->xd", pi * R, phi) - (pi * R).sum(1, keepdims=True) * mean_phi


def prompt_std(theta: np.ndarray, phi: np.ndarray, EV: np.ndarray) -> np.ndarray:
    v = (policy(theta, phi) * EV).sum(1)
    return np.sqrt(np.clip(v * (1 - v), 0.0, None))


def expected_update(
    theta: np.ndarray, phi: np.ndarray, w: np.ndarray, EV: np.ndarray
) -> np.ndarray:
    """Large-group limit: sum_x (w_x / s_x) grad J_V,x; zero-variance prompts contribute nothing."""
    s = prompt_std(theta, phi, EV)
    live = s > 0
    g = per_prompt_grad(theta, phi, EV)
    return ((w[live] / s[live])[:, None] * g[live]).sum(0)


def prop1_weights(theta: np.ndarray, phi: np.ndarray, w: np.ndarray, G: np.ndarray,
                  f: float) -> np.ndarray:  # fmt: skip
    p = (policy(theta, phi) * G).sum(1)
    v = f + (1 - f) * p
    return w * (1 - f) / np.sqrt(v * (1 - v))


def monte_carlo_group_update(
    theta: np.ndarray,
    phi_x: np.ndarray,
    G_x: np.ndarray,
    f: float,
    group: int,
    n_groups: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One prompt, finite groups, GRPO advantages (V - mean) / (std_ddof1 + 1e-4), zero-variance
    groups -> 0; fresh false-positive coins of rate f on G = 0 samples."""
    pi = policy(theta, phi_x[None])[0]
    y = rng.choice(len(pi), size=(n_groups, group), p=pi)
    g = G_x[y]
    V = np.where(g == 1, 1.0, (rng.random(y.shape) < f).astype(float))
    sd = V.std(axis=1, ddof=1, keepdims=True)
    A = np.where(sd > 0, (V - V.mean(axis=1, keepdims=True)) / (sd + 1e-4), 0.0)
    score = phi_x - pi @ phi_x  # grad log pi(y) for every response
    return np.einsum("ng,ngd->d", A, score[y]) / (n_groups * group)


def fp_coherence(theta: np.ndarray, phi: np.ndarray, w: np.ndarray, G: np.ndarray,
                 V: np.ndarray) -> float:  # fmt: skip
    """kappa_FP = ||sum_x g_x||^2 / sum_x ||g_x||^2 with g_x = (w_x / s_x) grad F_x."""
    s = prompt_std(theta, phi, V)
    live = s > 0
    g = (w[live] / s[live])[:, None] * per_prompt_grad(theta, phi, V * (1 - G))[live]
    return float(np.sum(g.sum(0) ** 2) / np.sum(g**2))


def fill_rate(c: float, f0: float, q: float) -> float:
    """Fresh-coin rate on wrong responses outside the covered master key keeping the FPR at f0."""
    return (f0 - c * q) / (1 - c * q)


def drift_sign(c: float, vbar: float, f0: float, q: float) -> float:
    """Mean-field drift of the shared master-key logit (exchangeable prompts), up to a positive
    factor: c (1 - vbar) + (1 - c)(r(c) - vbar)."""
    r = fill_rate(c, f0, q)
    return c * (1 - vbar) + (1 - c) * (r - vbar)


def critical_coverage(vbar: float, f0: float, q: float, tol: float = 1e-10) -> float:
    lo, hi = 0.0, 1.0
    if drift_sign(lo, vbar, f0, q) >= 0 or drift_sign(hi, vbar, f0, q) <= 0:
        raise ValueError("no sign change on [0, 1]")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if drift_sign(mid, vbar, f0, q) < 0 else (lo, mid)
    return (lo + hi) / 2
