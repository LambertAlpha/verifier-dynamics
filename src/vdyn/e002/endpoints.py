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


def higher_matrix(metric: Metric, target: np.ndarray) -> np.ndarray:
    """H[i, j] = 1 for the pairs the metric compares (i ranked above j by the target)."""
    if metric is c_index:
        t = np.asarray(target, dtype=np.float64)
        return (t[:, None] > t[None, :]).astype(np.float64)
    if metric is auroc:
        lab = np.asarray(target, dtype=bool)
        return (lab[:, None] & ~lab[None, :]).astype(np.float64)
    raise ValueError("the kernel bootstrap supports c_index and auroc only")


def pair_kernels(scores: np.ndarray, higher: np.ndarray) -> np.ndarray:
    """K[r] = H * concordance(scores[r]), so metric(scores[r], w) = w'K[r]w / w'Hw."""
    s = np.asarray(scores, dtype=np.float64)
    out = np.empty((s.shape[0], *higher.shape))
    for r in range(s.shape[0]):
        diff = s[r][:, None] - s[r][None, :]
        out[r] = higher * ((diff > 0) + 0.5 * (diff == 0))
    return out


def _draws(n: int, reps: int, n_boot: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """The E002 §7 resampling draws, in the reference order (structures, then replications)."""
    counts = np.empty((n_boot, n))
    rep_idx = np.empty((n_boot, reps), dtype=np.int64)
    for b in range(n_boot):
        counts[b] = np.bincount(rng.integers(0, n, size=n), minlength=n)
        rep_idx[b] = rng.integers(0, reps, size=reps)
    return counts, rep_idx


def _resampled(
    kernels: np.ndarray, higher: np.ndarray, counts: np.ndarray, rep_idx: np.ndarray,
    chunk: int = 32,
) -> np.ndarray:  # fmt: skip
    """Mean over the drawn replications of w'K[r]w / w'Hw, one row of draws per resample."""
    reps, n, _ = kernels.shape
    k_flat = kernels.reshape(reps, n * n)
    h_flat = higher.reshape(n * n)
    out = np.empty(len(counts))
    for lo in range(0, len(counts), chunk):
        w = counts[lo : lo + chunk]
        ww = (w[:, :, None] * w[:, None, :]).reshape(len(w), n * n)
        num = np.take_along_axis(ww @ k_flat.T, rep_idx[lo : lo + chunk], axis=1).mean(axis=1)
        den = ww @ h_flat
        out[lo : lo + chunk] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def hierarchical_bootstrap(
    metric: Metric, scores: np.ndarray, target: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    reps, n = scores.shape
    higher = higher_matrix(metric, target)
    counts, rep_idx = _draws(n, reps, n_boot, rng)
    return _resampled(pair_kernels(scores, higher), higher, counts, rep_idx)


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
    higher = higher_matrix(metric, target)
    counts, rep_idx = _draws(n, reps, n_boot, rng)
    mine = _resampled(pair_kernels(arm, higher), higher, counts, rep_idx)
    others = [_resampled(pair_kernels(c, higher), higher, counts, rep_idx) for c in competitors]
    return mine - np.max(others, axis=0)


def bootstrap_p_value(diffs: np.ndarray) -> float:
    """One-sided p for H1: Delta > 0 (Amendment 2 §3)."""
    d = np.asarray(diffs)
    return float((1 + np.sum(d <= 0)) / (len(d) + 1))


def holm(pvalues: Sequence[float], alpha: float) -> list[bool]:
    """Holm step-down rejections, in input order."""
    m = len(pvalues)
    reject = [False] * m
    for rank, i in enumerate(np.argsort(pvalues, kind="stable")):
        if pvalues[i] > alpha / (m - rank):
            break
        reject[i] = True
    return reject
