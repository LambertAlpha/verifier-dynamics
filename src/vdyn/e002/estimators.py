"""Finite-sample geometry estimators (registry E002 §5 G1 and §10), vectorized over replications.

Shapes: `reps` independent replications of `n` rollouts; `d = 1 + m` parameters (u, phi_1..m).
Scores are the exact score function of the product-Bernoulli policy at pi_0,
`(corr - q, z - s)`, which equals the autodiff score (tested in Phase 1B).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from vdyn.verifiers.boolean_fp import EventStructure, sample_verifier


@dataclass(frozen=True)
class Batch:
    corr: np.ndarray  # (reps, n) in {0, 1}
    z: np.ndarray  # (reps, n, m) in {0, 1}
    v: np.ndarray  # (reps, n) verifier outcome (fresh coin per query)


def sample_rollouts(
    st: EventStructure,
    q: float | np.ndarray,
    s: np.ndarray,
    n: int,
    reps: int,
    rng: np.random.Generator,
) -> Batch:
    q_arr = np.broadcast_to(np.asarray(q, dtype=np.float64), (reps,))
    s_arr = np.broadcast_to(np.asarray(s, dtype=np.float64), (reps, st.n_features))
    corr = (rng.random((reps, n)) < q_arr[:, None]).astype(np.int64)
    z = (rng.random((reps, n, st.n_features)) < s_arr[:, None, :]).astype(np.int64)
    return Batch(corr, z, sample_verifier(st, corr, z, rng))


def score_vectors(
    q: float | np.ndarray, s: np.ndarray, corr: np.ndarray, z: np.ndarray
) -> np.ndarray:
    reps = corr.shape[0]
    q_arr = np.broadcast_to(np.asarray(q, dtype=np.float64), (reps,))
    s_arr = np.broadcast_to(np.asarray(s, dtype=np.float64), (reps, z.shape[2]))
    return np.concatenate([(corr - q_arr[:, None])[..., None], z - s_arr[:, None, :]], axis=2)


def rloo(rewards: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """Leave-one-out baseline REINFORCE gradient: mean_i (R_i - mean_{j != i} R_j) score_i."""
    n = rewards.shape[1]
    baseline = (rewards.sum(axis=1, keepdims=True) - rewards) / (n - 1)
    return np.mean((rewards - baseline)[..., None] * sc, axis=1)


def fisher_hat(sc: np.ndarray) -> np.ndarray:
    return np.einsum("rni,rnj->rij", sc, sc) / sc.shape[1]


def damped_inverse(fisher: np.ndarray, lam: float) -> np.ndarray:
    d = fisher.shape[-1]
    scale = np.trace(fisher, axis1=1, axis2=2) / d
    return np.linalg.inv(fisher + lam * scale[:, None, None] * np.eye(d))


def _quad(a: np.ndarray, m: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ri,rij,rj->r", a, m, b)


def decompose_batch(
    g_gold: np.ndarray, g_ver: np.ndarray, metric: np.ndarray
) -> dict[str, np.ndarray]:
    """Vectorized `decompose` with the v0.3 convention at A = 0 (alpha = NaN, C = ||g_e||)."""
    g_err = g_ver - g_gold
    a2 = _quad(g_gold, metric, g_gold)
    defined = a2 > 0
    alpha_raw = np.where(defined, _quad(g_err, metric, g_gold) / np.where(defined, a2, 1.0), 0.0)
    resid = g_err - alpha_raw[:, None] * g_gold
    c2 = np.maximum(_quad(resid, metric, resid), 0.0)
    return {
        "A": np.sqrt(np.maximum(a2, 0.0)),
        "alpha": np.where(defined, alpha_raw, np.nan),
        "alpha_defined": defined,
        "C": np.sqrt(c2),
        "C2": c2,
    }


def ustat_gram(ga: np.ndarray, gb: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """(1/(n(n-1))) sum_{i != j} ga_i^T M gb_j: unbiased for <E ga, E gb>_M given M."""
    n = ga.shape[1]
    total = _quad(ga.sum(axis=1), metric, gb.sum(axis=1))
    diagonal = np.einsum("rni,rij,rnj->r", ga, metric, gb)
    return (total - diagonal) / (n * (n - 1))


def geometry(
    st: EventStructure,
    labeled: Batch,
    unlabeled: Batch | None,
    q: float,
    s: np.ndarray,
    estimator: str,
    lam: float,
    source: str,
    exact_fisher: np.ndarray | None = None,
) -> dict[str, Any]:
    """G1: (A, alpha, C) at t = 0 from audited (+ unlabeled) pi_0 rollouts."""
    sc_lab = score_vectors(q, s, labeled.corr, labeled.z)
    if unlabeled is not None and unlabeled.corr.shape[1] > 0:
        sc_unl = score_vectors(q, s, unlabeled.corr, unlabeled.z)
        sc_all = np.concatenate([sc_lab, sc_unl], axis=1)
        v_all = np.concatenate([labeled.v, unlabeled.v], axis=1)
    else:
        sc_all, v_all = sc_lab, labeled.v
    reps = sc_lab.shape[0]
    if exact_fisher is not None:
        metric = np.broadcast_to(np.linalg.inv(exact_fisher), (reps, *exact_fisher.shape))
    else:
        metric = damped_inverse(fisher_hat(sc_all), lam)
    gold = labeled.corr.astype(np.float64)
    if estimator == "plugin":
        g_gold = rloo(gold, sc_lab)
        g_ver = rloo(labeled.v, sc_lab) if source == "paired" else rloo(v_all, sc_all)
        return decompose_batch(g_gold, g_ver, metric)
    if estimator != "ustat" or source != "paired":
        raise ValueError("the U-statistic estimator is defined on the paired audited set only")
    gam_g = gold[..., None] * sc_lab
    gam_e = (labeled.v - gold)[..., None] * sc_lab
    gram_gg = ustat_gram(gam_g, gam_g, metric)
    gram_eg = ustat_gram(gam_e, gam_g, metric)
    gram_ee = ustat_gram(gam_e, gam_e, metric)
    defined = gram_gg > 0
    safe = np.where(defined, gram_gg, 1.0)
    c2 = np.where(defined, gram_ee - gram_eg**2 / safe, gram_ee)
    return {
        "A": np.sqrt(np.maximum(gram_gg, 0.0)),
        "alpha": np.where(defined, gram_eg / safe, np.nan),
        "alpha_defined": defined,
        "C": np.sqrt(np.maximum(c2, 0.0)),
        "C2": c2,  # raw, may be negative
        "P": gram_eg,
        "D": gram_gg * gram_ee - gram_eg**2,
    }


def degenerate_event_probabilities(q: float, f: float, s: np.ndarray, n: int) -> dict[str, float]:
    """Registered exact predictions (E002 §10) for a batch of n i.i.d. pi_0 rollouts."""
    s = np.asarray(s, dtype=np.float64)
    return {
        "all_gold_equal": q**n + (1 - q) ** n,
        "no_false_positive": (1 - (1 - q) * f) ** n,
        "some_feature_constant": float(1 - np.prod(1 - s**n - (1 - s) ** n)),
    }
