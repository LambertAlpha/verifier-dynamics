"""E002 endpoints (registry E002 §7): C-index, AUROC, Kendall, recall, hierarchical bootstrap.

Bootstrap resamples of structures are represented as integer weights (multiplicities), so a
resampled metric equals the metric on the duplicated rows (tested).
"""

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import kendalltau

Metric = Callable[..., float]


def _pairwise(score: np.ndarray, higher: np.ndarray, weights: np.ndarray | None) -> float:
    s = np.asarray(score, dtype=np.float64)
    w = np.ones(len(s)) if weights is None else np.asarray(weights, dtype=np.float64)
    ww = w[:, None] * w[None, :]
    diff = s[:, None] - s[None, :]
    conc = (diff > 0) + 0.5 * (diff == 0)
    den = float(np.sum(ww * higher))
    return float(np.sum(ww * higher * conc) / den) if den > 0 else float("nan")


def c_index(score: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Harrell's concordance: pairs with target_i > target_j; predictor ties count 1/2."""
    t = np.asarray(target, dtype=np.float64)
    return _pairwise(score, t[:, None] > t[None, :], weights)


def auroc(score: np.ndarray, label: np.ndarray, weights: np.ndarray | None = None) -> float:
    lab = np.asarray(label, dtype=bool)
    return _pairwise(score, lab[:, None] & ~lab[None, :], weights)


def kendall_tau_b(score: np.ndarray, target: np.ndarray) -> float:
    return float(kendalltau(score, target).statistic)


def recall_top(score: np.ndarray, target: np.ndarray, frac: float) -> float:
    k = int(np.ceil(frac * len(target)))
    top_true = set(np.argsort(-np.asarray(target), kind="stable")[:k])
    top_pred = set(np.argsort(-np.asarray(score), kind="stable")[:k])
    return len(top_true & top_pred) / k


def panel_metric(metric: Metric, scores: np.ndarray, target: np.ndarray) -> float:
    """Mean over replications of the panel-level metric."""
    return float(np.mean([metric(scores[r], target) for r in range(scores.shape[0])]))


def _resample(n: int, reps: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
    return counts, rng.integers(0, reps, size=reps)


def hierarchical_bootstrap(
    metric: Metric, scores: np.ndarray, target: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    reps, n = scores.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        counts, rep_idx = _resample(n, reps, rng)
        out[b] = np.mean([metric(scores[r], target, counts) for r in rep_idx])
    return out


def paired_difference_bootstrap(
    metric: Metric,
    arm: np.ndarray,
    competitors: Sequence[np.ndarray],
    target: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """metric(arm) - max_j metric(competitor_j), with shared structure and replication resamples."""
    reps, n = arm.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        counts, rep_idx = _resample(n, reps, rng)
        mine = np.mean([metric(arm[r], target, counts) for r in rep_idx])
        best = max(np.mean([metric(c[r], target, counts) for r in rep_idx]) for c in competitors)
        out[b] = mine - best
    return out
