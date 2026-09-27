"""E005a-R subspace representations (research/09_e005ar_design.md §5).

R1: Haar-random orthonormal basis (nested in k). R2: top-k eigenvectors of the uncentered second
moment of per-rollout gold contributions from the first half of the groups (estimation on the
second half). R3: cross-fitting; the basis built from one half (stacked per-rollout gold and
verifier contributions) is used on the other half, and vice versa. Exact LAPACK eigh on the
smaller of the Gram and covariance matrices.
"""

import numpy as np

RANK_TOL = 1e-10


def haar_basis(d: int, kmax: int, rng: np.random.Generator) -> np.ndarray:
    Q, Rr = np.linalg.qr(rng.standard_normal((d, kmax)))
    return Q * np.sign(np.diag(Rr))


def top_eigvecs(Y: np.ndarray, kmax: int) -> np.ndarray:
    """Top eigenvectors (d, k <= kmax, decreasing eigenvalue) of Y^T Y for Y of shape (M, d)."""
    M, d = Y.shape
    if M < d:
        w, U = np.linalg.eigh(Y @ Y.T)
        w, U = w[::-1], U[:, ::-1]
        keep = w > RANK_TOL * max(w[0], 1e-300)
        V = (Y.T @ U[:, keep]) / np.sqrt(w[keep])
    else:
        w, E = np.linalg.eigh(Y.T @ Y)
        w, E = w[::-1], E[:, ::-1]
        V = E[:, w > RANK_TOL * max(w[0], 1e-300)]
    return V[:, :kmax]


def halves(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(n // 2), np.arange(n // 2, n)


def crossfit_bases(yG: np.ndarray, yV: np.ndarray, kmax: int) -> tuple[np.ndarray, np.ndarray]:
    """(basis from fold A, basis from fold B) for per-rollout contributions (n, m, d)."""
    d = yG.shape[-1]
    out = []
    for f in halves(yG.shape[0]):
        Y = np.concatenate([yG[f].reshape(-1, d), yV[f].reshape(-1, d)], axis=0)
        out.append(top_eigvecs(Y, kmax))
    return out[0], out[1]
