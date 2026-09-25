"""Metric-generic decomposition of verifier-induced gradient error (theory note §1).

Given gold and verifier gradients and a metric M (inner product <x, y>_M = x^T M y):

    A = ||g_G||_M,  alpha = <g_e, g_G>_M / A^2,  r = g_e - alpha g_G,  C = ||r||_M,  b = alpha A.

Natural gradient corresponds to M = F^{-1}; vanilla gradient ascent to M = I. The rate properties
are the Proposition 1 identities, valid only when M is the optimizer's own preconditioner.

Degenerate case A = 0 (no gold direction under M; theory note v0.3): alpha = NaN,
alpha_defined = False, and C = ||g_e||_M is the magnitude of the whole residual. It is not an
orthogonal component, because there is no gold direction to be orthogonal to.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Decomposition:
    A: float
    alpha: float
    C: float
    residual: np.ndarray  # r in gradient coordinates (= g_e when alpha is undefined)
    alpha_defined: bool = True

    @property
    def b(self) -> float:
        return self.alpha * self.A if self.alpha_defined else float("nan")

    @property
    def gold_rate(self) -> float:
        """dJ_G/dt under theta_dot = M g_V (0 when A = 0)."""
        return self.A**2 * (1 + self.alpha) if self.alpha_defined else 0.0

    @property
    def verifier_rate(self) -> float:
        """dJ_V/dt under theta_dot = M g_V."""
        if not self.alpha_defined:
            return self.C**2
        return self.A**2 * (1 + self.alpha) ** 2 + self.C**2

    @property
    def gap_rate(self) -> float:
        """dDelta/dt = b(a + b) + c^2 under theta_dot = M g_V (C^2 when A = 0)."""
        if not self.alpha_defined:
            return self.C**2
        return self.b * (self.A + self.b) + self.C**2


def decompose(g_gold: np.ndarray, g_verifier: np.ndarray, metric: np.ndarray) -> Decomposition:
    g_err = g_verifier - g_gold
    a2 = float(g_gold @ metric @ g_gold)
    if a2 <= 0.0:
        e2 = float(g_err @ metric @ g_err)
        return Decomposition(
            A=0.0, alpha=float("nan"), C=np.sqrt(max(e2, 0.0)), residual=g_err, alpha_defined=False
        )
    alpha = float(g_err @ metric @ g_gold) / a2
    residual = g_err - alpha * g_gold
    c2 = float(residual @ metric @ residual)
    return Decomposition(A=np.sqrt(a2), alpha=alpha, C=np.sqrt(max(c2, 0.0)), residual=residual)
