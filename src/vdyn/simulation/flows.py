"""Gradient flows driven by exact expected gradients from `geometry.autodiff`.

theta_dot = M g_V with M = I (vanilla) or M = F^{-1} (natural). The fields know nothing about
closed forms; tests compare their trajectories against the closed-form ODEs.
"""

from collections.abc import Callable

import numpy as np
from scipy.integrate import OdeSolution, solve_ivp

from vdyn.geometry import autodiff
from vdyn.policies.bernoulli import LogProbFn

Field = Callable[[float, np.ndarray], np.ndarray]


def vanilla_field(log_prob_fn: LogProbFn, rewards: np.ndarray) -> Field:
    def field(_t: float, theta: np.ndarray) -> np.ndarray:
        return autodiff.reward_gradient(log_prob_fn, theta, rewards)

    return field


def natural_field(log_prob_fn: LogProbFn, rewards: np.ndarray) -> Field:
    def field(_t: float, theta: np.ndarray) -> np.ndarray:
        grad = autodiff.reward_gradient(log_prob_fn, theta, rewards)
        return np.linalg.solve(autodiff.fisher(log_prob_fn, theta), grad)

    return field


class Trajectory:
    """Integrated flow: times `t`, states `y` (dim x n_times), dense interpolant `sol`."""

    def __init__(self, t: np.ndarray, y: np.ndarray, sol: OdeSolution, n_evals: int) -> None:
        self.t = t
        self.y = y
        self.sol = sol
        self.n_evals = n_evals


def integrate(
    field: Field,
    theta0: np.ndarray,
    t_end: float,
    t_eval: np.ndarray | None = None,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> Trajectory:
    result = solve_ivp(
        field,
        (0.0, t_end),
        np.asarray(theta0, dtype=np.float64),
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not result.success:
        raise RuntimeError(f"integration failed: {result.message}")
    return Trajectory(result.t, result.y, result.sol, result.nfev)


def euler(field: Field, theta0: np.ndarray, eta: float, n_steps: int) -> np.ndarray:
    """Explicit Euler with step size eta; returns all iterates, shape (n_steps + 1, dim)."""
    path = np.empty((n_steps + 1, len(theta0)))
    path[0] = theta0
    for k in range(n_steps):
        path[k + 1] = path[k] + eta * field(k * eta, path[k])
    return path
