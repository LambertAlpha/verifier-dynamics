"""E005a-R point estimate, null test and projected oracle (research/09_e005ar_design.md §2.3, §2.6).

- Point estimate: E3 of the frozen E005a code (delete-one-group jackknife of the paired plug-in).
- Null test (registered; E005a §10 lesson 1): T = Σ_{i≠j} z_i·z_j with
  z_k = P_hat_perp (x_k^e − alpha_hat x_k^G); reference by B group sign flips,
  p = (1 + #{T_b ≥ T}) / (B + 1); reject iff p ≤ 0.05. Cross-fitted folds add their statistics
  with independent flips.
- Legacy (E005a, reported only): reject iff C2 − 1.645 SE > 0.
"""

from typing import Any

import numpy as np

from vdyn.e005 import estimators as es

LEVEL = 0.05
B_FLIPS = 199
Z_ONE = 1.645


def e3(xG: np.ndarray, xe: np.ndarray) -> dict[str, np.ndarray]:
    out = es.estimate_all(xG, xe)["E3"]
    return {q: np.asarray(out[q], dtype=float) for q in ("C2", "SE_C2", "A2", "alpha")}


def wald_reject(e3_out: dict[str, np.ndarray]) -> np.ndarray:
    v = e3_out["C2"] - Z_ONE * e3_out["SE_C2"]
    return np.where(np.isfinite(v), v > 0, False)


def gram_residual(xG: np.ndarray, xe: np.ndarray) -> np.ndarray:
    """(R, n, n) Gram matrix of z_k = P_hat_perp (x_k^e − alpha_hat x_k^G)."""
    hG, he = xG.mean(1), xe.mean(1)
    A2 = np.einsum("ri,ri->r", hG, hG)
    ok = A2 > 0
    safe = np.where(ok, A2, 1.0)
    alpha = np.where(ok, np.einsum("ri,ri->r", he, hG) / safe, 0.0)
    u = np.where(ok[:, None], hG / np.sqrt(safe)[:, None], 0.0)
    delta = xe - alpha[:, None, None] * xG
    z = delta - np.einsum("rni,ri->rn", delta, u)[..., None] * u[:, None, :]
    return np.einsum("rni,rmi->rnm", z, z)


def _off(K: np.ndarray) -> np.ndarray:
    return K.sum((1, 2)) - np.trace(K, axis1=1, axis2=2)


def signflip_from_grams(Ks: list[np.ndarray], rng: np.random.Generator,
                        B: int = B_FLIPS) -> dict[str, np.ndarray]:  # fmt: skip
    T = np.sum([_off(K) for K in Ks], axis=0)
    Tb = np.zeros((Ks[0].shape[0], B))
    for K in Ks:
        S = rng.choice(np.array([-1.0, 1.0]), size=(K.shape[0], B, K.shape[1]))
        Tb += (np.matmul(S, K) * S).sum(-1) - np.trace(K, axis1=1, axis2=2)[:, None]
    p = (1 + (Tb >= T[:, None]).sum(1)) / (B + 1)
    return {"T": T, "p": p, "reject": p <= LEVEL}


def signflip(xG: np.ndarray, xe: np.ndarray, rng: np.random.Generator,
             B: int = B_FLIPS) -> dict[str, np.ndarray]:  # fmt: skip
    return signflip_from_grams([gram_residual(xG, xe)], rng, B)


def projected_geometry(hG: np.ndarray, he: np.ndarray, Q: np.ndarray) -> dict[str, Any]:
    g, e = Q.T @ hG, Q.T @ he
    A2, P, Qe = float(g @ g), float(e @ g), float(e @ e)
    if A2 <= 0:
        return {"A2": A2, "alpha": float("nan"), "C2": Qe}
    return {"A2": A2, "alpha": P / A2, "C2": max(Qe - P**2 / A2, 0.0)}
