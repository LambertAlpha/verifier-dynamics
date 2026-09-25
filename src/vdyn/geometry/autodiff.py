"""Path B: exact quantities for any policy over an enumerated outcome set, by torch autodiff.

A policy is a function theta -> log pi over enumerated outcomes; a verifier is a reward table over
the same outcomes. Nothing here knows the closed forms.
"""

import numpy as np
import torch
from torch.func import jacfwd, jacrev

from vdyn.policies.bernoulli import LogProbFn


def _t(x: np.ndarray | float) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


def outcome_probs(log_prob_fn: LogProbFn, theta: np.ndarray) -> np.ndarray:
    return torch.exp(log_prob_fn(_t(theta))).numpy()


def expected_reward(log_prob_fn: LogProbFn, theta: np.ndarray, rewards: np.ndarray) -> float:
    return float((torch.exp(log_prob_fn(_t(theta))) * _t(rewards)).sum())


def reward_gradient(log_prob_fn: LogProbFn, theta: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    """grad_theta sum_y pi_theta(y) R(y)."""
    r = _t(rewards)

    def objective(th: torch.Tensor) -> torch.Tensor:
        return (torch.exp(log_prob_fn(th)) * r).sum()

    return jacrev(objective)(_t(theta)).numpy()


def score_matrix(log_prob_fn: LogProbFn, theta: np.ndarray) -> np.ndarray:
    """d log pi(y) / d theta for every enumerated outcome; shape (n_outcomes, dim).

    Forward mode: many outcomes, few parameters (jacrev would do one backward pass per outcome).
    """
    return jacfwd(log_prob_fn)(_t(theta)).numpy()


def fisher(log_prob_fn: LogProbFn, theta: np.ndarray) -> np.ndarray:
    """F = sum_y pi(y) score(y) score(y)^T."""
    th = _t(theta)
    p = torch.exp(log_prob_fn(th))
    sc = jacfwd(log_prob_fn)(th)
    return torch.einsum("n,ni,nj->ij", p, sc, sc).numpy()


def fisher_from_hessian(log_prob_fn: LogProbFn, theta: np.ndarray) -> np.ndarray:
    """F = -sum_y pi(y) Hessian log pi(y); an identity independent of the score outer product."""
    th = _t(theta)
    p = torch.exp(log_prob_fn(th))
    hess = jacfwd(jacfwd(log_prob_fn))(th)
    return -torch.einsum("n,nij->ij", p, hess).numpy()
