"""E002 competitor arms G0, P1-P4 and budget plans (registry E002 §3, §5), vectorized.

Each replication carries its own policy state theta = (u, phi) during a probe. Probe updates are
stochastic natural-gradient steps with an estimated, damped Fisher (the practical optimizer):
    theta <- theta + eta (F_hat + lam tr(F_hat)/d I)^{-1} g_hat_V,   lam = LAM_P.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit, log_expit

from vdyn.e002.estimators import (
    Batch,
    damped_inverse,
    fisher_hat,
    rloo,
    sample_rollouts,
    score_vectors,
)
from vdyn.policies.bernoulli import product_outcomes
from vdyn.verifiers.boolean_fp import EventStructure, acceptance, sample_verifier

LAM_P = 1e-2
TUNING_GRIDS: dict[str, dict[str, tuple[Any, ...]]] = {
    "G1": {
        "estimator": ("plugin", "ustat"),
        "lam": (1e-3, 1e-2, 1e-1),
        "source": ("paired", "pooled"),
    },
    "P1": {"k": (1, 2, 5, 10), "eta": (0.1, 0.3, 1.0)},
    "P2": {"k": (1, 2), "eta": (0.1, 0.3, 1.0)},
    "P3": {"k": (1, 2, 5, 10), "eta": (0.1, 0.3, 1.0), "observable": ("fpr", "gold")},
    "P4": {"r": (1, 2, "all")},
}


@dataclass(frozen=True)
class Plan:
    arm: str
    feasible: bool
    b_roll: int
    b_gold: int
    b_bwd: int
    k: int = 0
    b: int = 0
    reuse_audit: bool = False


def plan_g0(b_gold: int, b_roll: int) -> Plan:
    return Plan("G0", True, b_gold, b_gold, 0)


def plan_g1(b_gold: int, b_roll: int) -> Plan:
    return Plan("G1", True, b_roll, b_gold, b_roll)


def plan_p1(b_gold: int, b_roll: int, k: int) -> Plan:
    b = b_roll // (k + 1)
    return Plan("P1", b >= 2, (k + 1) * b, 0, k * b, k, b)


def plan_p2(b_gold: int, b_roll: int, k: int) -> Plan:
    b = (b_roll - b_gold) // k
    if b >= 2:
        return Plan("P2", True, b_gold + k * b, b_gold, k * b, k, b)
    if k == 1:  # the audit rollouts double as the single training batch
        return Plan("P2", True, b_gold, b_gold, b_gold, 1, b_gold, reuse_audit=True)
    return Plan("P2", False, 0, 0, 0, k, 0)


def plan_p3(b_gold: int, b_roll: int, k: int) -> Plan:
    b = (b_roll - b_gold) // k
    return Plan("P3", b >= 2, k * max(b, 0) + b_gold, b_gold, k * max(b, 0), k, max(b, 0))


def plan_p4(b_gold: int, b_roll: int) -> Plan:
    return Plan("P4", b_roll - b_gold >= 1, b_roll, b_gold, 0)


def _state(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return expit(theta[:, 0]), expit(theta[:, 1:])


def ng_probe(
    st: EventStructure,
    theta0: np.ndarray,
    k: int,
    b: int,
    eta: float,
    reps: int,
    rng: np.random.Generator,
    lam: float = LAM_P,
    first_batch: Batch | None = None,
) -> tuple[np.ndarray, list[Batch]]:
    theta = np.broadcast_to(np.asarray(theta0, dtype=np.float64), (reps, len(theta0))).copy()
    used: list[Batch] = []
    for j in range(k):
        q, s = _state(theta)
        batch = (
            first_batch
            if (j == 0 and first_batch is not None)
            else sample_rollouts(st, q, s, b, reps, rng)
        )
        sc = score_vectors(q, s, batch.corr, batch.z)
        step = np.einsum("rij,rj->ri", damped_inverse(fisher_hat(sc), lam), rloo(batch.v, sc))
        theta = theta + eta * step
        used.append(batch)
    return theta, used


def _audit(
    st: EventStructure, theta0: np.ndarray, m0: int, reps: int, rng: np.random.Generator
) -> Batch:
    q, s = _state(np.broadcast_to(theta0, (reps, len(theta0))))
    return sample_rollouts(st, q, s, m0, reps, rng)


def _fpr_hat(batch: Batch, weights: np.ndarray | None = None) -> np.ndarray:
    wrong = (batch.corr == 0).astype(np.float64)
    w = wrong if weights is None else wrong * weights
    denom = w.sum(axis=1)
    return np.where(denom > 0, (w * batch.v).sum(axis=1) / np.where(denom > 0, denom, 1.0), 0.0)


def run_g0(
    st: EventStructure, theta0: np.ndarray, m0: int, reps: int, rng: np.random.Generator
) -> dict[str, Any]:
    return {"score": _fpr_hat(_audit(st, theta0, m0, reps, rng))}


def run_p1(
    st: EventStructure,
    theta0: np.ndarray,
    b: int,
    k: int,
    eta: float,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    first = _audit(st, theta0, b, reps, rng)  # batch 0 at pi_0 (unlabeled use only)
    theta_k, _ = ng_probe(st, theta0, k, b, eta, reps, rng, first_batch=first)
    q, s = _state(theta_k)
    final = sample_rollouts(st, q, s, b, reps, rng)
    return {"score": final.v.mean(axis=1) - first.v.mean(axis=1), "theta_k": theta_k}


def _log_prob_rows(theta: np.ndarray, batch: Batch) -> np.ndarray:
    """log pi_theta(y) for every rollout; theta (reps, d)."""
    y = np.concatenate([batch.corr[..., None], batch.z], axis=2).astype(np.float64)
    return np.sum(
        y * log_expit(theta[:, None, :]) + (1 - y) * log_expit(-theta[:, None, :]), axis=2
    )


def run_p2(
    st: EventStructure,
    theta0: np.ndarray,
    m0: int,
    b: int,
    k: int,
    eta: float,
    reps: int,
    rng: np.random.Generator,
    reuse_audit: bool = False,
) -> dict[str, Any]:
    audit = _audit(st, theta0, m0, reps, rng)
    theta_k, _ = ng_probe(
        st, theta0, k, b, eta, reps, rng, first_batch=audit if reuse_audit else None
    )
    base = np.broadcast_to(np.asarray(theta0, dtype=np.float64), theta_k.shape)
    log_w = _log_prob_rows(theta_k, audit) - _log_prob_rows(base, audit)
    weights = np.exp(log_w - log_w.max(axis=1, keepdims=True))
    return {"score": _fpr_hat(audit, weights) - _fpr_hat(audit), "theta_k": theta_k}


def run_p3(
    st: EventStructure,
    theta0: np.ndarray,
    q0: float,
    m: int,
    b: int,
    k: int,
    eta: float,
    observable: str,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    theta_k, _ = ng_probe(st, theta0, k, b, eta, reps, rng)
    q, s = _state(theta_k)
    fresh = sample_rollouts(st, q, s, m, reps, rng)
    if observable == "fpr":
        score = _fpr_hat(fresh)
    else:
        score = -(fresh.corr.mean(axis=1) - q0)
    return {"score": score, "theta_k": theta_k}


def run_p4(
    st: EventStructure,
    theta0: np.ndarray,
    m0: int,
    variants_allowed: int,
    r: int | str,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Resample r distinct features of each audited wrong-and-rejected response; control for
    verifier randomness with one re-query of the unchanged response."""
    audit = _audit(st, theta0, m0, reps, rng)
    m = st.n_features
    if m == 0:
        zeros = np.zeros(reps)
        return {"score": zeros, "variants_used": zeros.astype(int)}
    r_eff = m if r == "all" else min(int(r), m)
    s0 = np.asarray(st.s0, dtype=np.float64)
    eligible = (audit.corr == 0) & (audit.v == 0)
    cum = np.cumsum(eligible * r_eff, axis=1)
    included = eligible & (cum <= variants_allowed)
    chosen = np.argsort(rng.random((reps, m0, m)), axis=2)[:, :, :r_eff]  # distinct features
    z_var = np.repeat(audit.z[:, :, None, :], r_eff, axis=2).copy()
    new_vals = (rng.random((reps, m0, r_eff)) < s0[chosen]).astype(np.int64)
    np.put_along_axis(z_var, chosen[..., None], new_vals[..., None], axis=3)
    acc_var = rng.random((reps, m0, r_eff)) < acceptance(st, z_var.reshape(-1, m)).reshape(
        reps, m0, r_eff
    )
    acc_req = sample_verifier(st, np.zeros_like(audit.corr), audit.z, rng)
    n_var = included.sum(axis=1) * r_eff
    n_items = included.sum(axis=1)
    var_mean = np.where(
        n_var > 0, (acc_var * included[..., None]).sum(axis=(1, 2)) / np.maximum(n_var, 1), 0.0
    )
    req_mean = np.where(n_items > 0, (acc_req * included).sum(axis=1) / np.maximum(n_items, 1), 0.0)
    return {"score": var_mean - req_mean, "variants_used": n_var}


def p4_oracle(st: EventStructure, s: np.ndarray) -> float:
    """Noiseless P4 score over the rejected-wrong distribution:
    mean_i E[a(z with z_i resampled)] - E[a(z)] (the re-query control)."""
    m = st.n_features
    if m == 0:
        return 0.0
    s = np.asarray(s, dtype=np.float64)
    pats = product_outcomes(m)
    probs = np.prod(np.where(pats == 1, s, 1 - s), axis=1)
    acc = acceptance(st, pats)
    w = probs * (1 - acc)
    gains = []
    for i in range(m):
        hi, lo = pats.copy(), pats.copy()
        hi[:, i], lo[:, i] = 1, 0
        resampled = s[i] * acceptance(st, hi) + (1 - s[i]) * acceptance(st, lo)
        gains.append(w @ resampled / w.sum())
    return float(np.mean(gains) - (w @ acc) / w.sum())
