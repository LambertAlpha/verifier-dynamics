"""Two independent Bernoulli response features: correctness `corr` and exploit feature `z`.

The torch log-prob functions are the *specification* of the policy; everything in
`geometry.autodiff` is derived from them. The numpy helpers (sampling, hand-written score) serve
the Monte Carlo path and deliberately do not call torch.
"""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit

# Enumerated responses y = (corr, z), in this fixed order everywhere in the package.
OUTCOMES = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

_CORR = torch.as_tensor(OUTCOMES[:, 0], dtype=torch.float64)
_Z = torch.as_tensor(OUTCOMES[:, 1], dtype=torch.float64)

LogProbFn = Callable[[torch.Tensor], torch.Tensor]


def log_prob(theta: torch.Tensor) -> torch.Tensor:
    """log pi(y) for the 4 outcomes; theta = (u, v) are the logits of q = P(corr=1), s = P(z=1)."""
    u, v = theta[0], theta[1]
    return (
        _CORR * F.logsigmoid(u)
        + (1 - _CORR) * F.logsigmoid(-u)
        + _Z * F.logsigmoid(v)
        + (1 - _Z) * F.logsigmoid(-v)
    )


def log_prob_mean(phi: torch.Tensor) -> torch.Tensor:
    """Same policy in mean parameterization phi = (q, s).

    Used to test reparameterization invariance.
    """
    q, s = phi[0], phi[1]
    return (
        _CORR * torch.log(q)
        + (1 - _CORR) * torch.log1p(-q)
        + _Z * torch.log(s)
        + (1 - _Z) * torch.log1p(-s)
    )


def multi_prompt_log_prob(weights: np.ndarray) -> LogProbFn:
    """Tabular multi-prompt policy: prompt k ~ weights, then an independent two-Bernoulli policy.

    theta = (u_1, v_1, ..., u_K, v_K); outcomes are ordered prompt-major, 4 per prompt.
    Prompts share no parameters.
    """
    log_w = torch.log(torch.as_tensor(weights, dtype=torch.float64))
    n_prompts = len(weights)

    def fn(theta: torch.Tensor) -> torch.Tensor:
        blocks = [log_prob(theta[2 * k : 2 * k + 2]) for k in range(n_prompts)]
        return (log_w[:, None] + torch.stack(blocks)).reshape(-1)

    return fn


def probs(theta: np.ndarray) -> tuple[float, float]:
    """(q, s) from logits."""
    return float(expit(theta[0])), float(expit(theta[1]))


def sample(theta: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n independent responses (corr, z) as integer arrays."""
    q, s = probs(theta)
    corr = (rng.random(n) < q).astype(np.int64)
    z = (rng.random(n) < s).astype(np.int64)
    return corr, z


def score(theta: np.ndarray, corr: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Hand-written score d log pi / d(u, v) = (corr - q, z - s); shape (n, 2)."""
    q, s = probs(theta)
    return np.column_stack([corr - q, z - s]).astype(np.float64)
