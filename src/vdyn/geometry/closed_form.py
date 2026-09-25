"""Path A: hand-derived closed forms from research/01_theory_note.md.

Typed directly from the derivations; shares no code with `autodiff` or `decompose`.
Y toy functions take (q, s) = (P(corr=1), P(z=1)) and broadcast over numpy arrays.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

Array = np.ndarray | float


def _stack(a: Array, b: Array) -> np.ndarray:
    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return np.stack([a, b], axis=-1)


# --- Y toy: objectives, gradients, Fisher (§2.1) -------------------------------------------------


def j_gold(q: Array, s: Array) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) + 0.0 * np.asarray(s)


def j_verifier(q: Array, s: Array) -> np.ndarray:
    return np.asarray(q + (1 - q) * s, dtype=np.float64)


def gap(q: Array, s: Array) -> np.ndarray:
    return np.asarray((1 - q) * s, dtype=np.float64)


def grad_gold(q: Array, s: Array) -> np.ndarray:
    return _stack(q * (1 - q), 0.0 * s)


def grad_verifier(q: Array, s: Array) -> np.ndarray:
    return _stack(q * (1 - q) * (1 - s), (1 - q) * s * (1 - s))


def grad_error(q: Array, s: Array) -> np.ndarray:
    return _stack(-s * q * (1 - q), (1 - q) * s * (1 - s))


def fisher(q: Array, s: Array) -> np.ndarray:
    q, s = np.broadcast_arrays(np.asarray(q, dtype=np.float64), np.asarray(s, dtype=np.float64))
    f = np.zeros(q.shape + (2, 2))
    f[..., 0, 0] = q * (1 - q)
    f[..., 1, 1] = s * (1 - s)
    return f


# --- Y toy: Fisher-whitened decomposition -------------------------------------------------------


def h_gold(q: Array, s: Array) -> np.ndarray:
    return _stack(np.sqrt(q * (1 - q)), 0.0 * s)


def h_error(q: Array, s: Array) -> np.ndarray:
    return _stack(-s * np.sqrt(q * (1 - q)), (1 - q) * np.sqrt(s * (1 - s)))


def h_residual(q: Array, s: Array) -> np.ndarray:
    return _stack(0.0 * q, (1 - q) * np.sqrt(s * (1 - s)))


def signal(q: Array, s: Array) -> np.ndarray:
    """A in the Fisher metric."""
    return np.sqrt(q * (1 - q)) + 0.0 * np.asarray(s)


def alpha(q: Array, s: Array) -> np.ndarray:
    """alpha = -s (identical in Fisher and Euclidean metrics)."""
    return -np.asarray(s, dtype=np.float64) + 0.0 * np.asarray(q)


def pressure(q: Array, s: Array) -> np.ndarray:
    """C in the Fisher metric."""
    return np.asarray((1 - q) * np.sqrt(s * (1 - s)), dtype=np.float64)


def euclidean_signal(q: Array, s: Array) -> np.ndarray:
    return np.asarray(q * (1 - q), dtype=np.float64) + 0.0 * np.asarray(s)


def euclidean_pressure(q: Array, s: Array) -> np.ndarray:
    return np.asarray((1 - q) * s * (1 - s), dtype=np.float64)


# --- Y toy: flows (§2.2, §2.3) -------------------------------------------------------------------


def natural_logit_velocity(q: Array, s: Array) -> np.ndarray:
    """(u_dot, v_dot) = F^{-1} g_V."""
    return _stack(1 - s, 1 - q)


def vanilla_logit_velocity(q: Array, s: Array) -> np.ndarray:
    """(u_dot, v_dot) = g_V."""
    return grad_verifier(q, s)


def natural_rates(q: Array, s: Array) -> np.ndarray:
    """(q_dot, s_dot) under natural gradient flow."""
    return _stack(q * (1 - q) * (1 - s), s * (1 - s) * (1 - q))


def vanilla_rates(q: Array, s: Array) -> np.ndarray:
    """(q_dot, s_dot) under vanilla gradient flow on logits."""
    return _stack((q * (1 - q)) ** 2 * (1 - s), (1 - q) * (s * (1 - s)) ** 2)


def natural_gap_rate(q: Array, s: Array) -> np.ndarray:
    return np.asarray(s * (1 - s) * (1 - q) * (1 - 2 * q), dtype=np.float64)


def vanilla_gap_rate(q: Array, s: Array) -> np.ndarray:
    return np.asarray((1 - q) ** 2 * s * (1 - s) * (s * (1 - s) - q**2), dtype=np.float64)


def natural_invariant(q: Array, s: Array) -> np.ndarray:
    """s / q, conserved under natural gradient flow."""
    return np.asarray(s / q, dtype=np.float64)


def _h(x: Array) -> np.ndarray:
    return np.asarray(x - np.exp(-np.asarray(x)), dtype=np.float64)


def vanilla_invariant(u: Array, v: Array) -> np.ndarray:
    """H(v) - H(u) with H(x) = x - exp(-x), conserved under vanilla gradient flow (Prop. 4)."""
    return _h(v) - _h(u)


def natural_endpoint(q0: float, s0: float) -> tuple[float, float]:
    """Limit of the natural gradient flow (Prop. 3)."""
    k = s0 / q0
    return (1.0, k) if k <= 1 else (1.0 / k, 1.0)


def _inverse_h(target: float) -> float:
    return float(brentq(lambda x: float(_h(x)) - target, -700.0, 700.0, xtol=1e-14, rtol=1e-15))


def vanilla_s_given_q(q0: float, s0: float, q: float) -> float:
    """s on the vanilla path through (q0, s0) at the point where P(corr=1) = q (clock-free)."""
    invariant = float(vanilla_invariant(logit(q0), logit(s0)))
    return float(expit(_inverse_h(float(_h(logit(q))) + invariant)))


def vanilla_q_given_s(q0: float, s0: float, s: float) -> float:
    """q on the vanilla path through (q0, s0) at the point where P(z=1) = s (clock-free)."""
    invariant = float(vanilla_invariant(logit(q0), logit(s0)))
    return float(expit(_inverse_h(float(_h(logit(s))) - invariant)))


# --- R and X (§3, §4) ----------------------------------------------------------------------------


def r_alpha(p: Array) -> np.ndarray:
    """alpha = -2p for fresh, response-independent flips; C = 0."""
    return np.asarray(-2 * np.asarray(p), dtype=np.float64)


def x_aggregate(a_deleted_sq: float, a_kept_sq: float) -> tuple[float, float]:
    """(alpha_agg, C_agg) for tabular deletion with a block-diagonal metric (Prop. 6)."""
    total = a_deleted_sq + a_kept_sq
    return -a_deleted_sq / total, float(np.sqrt(a_deleted_sq * a_kept_sq / total))
