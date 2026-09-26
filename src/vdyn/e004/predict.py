"""E004a Stage 1 predictors and inference (registry: Stage 1 execution note §3-§6).

- Pipeline: median imputation (training fold) -> standardization -> model.
- Primary models: ridge (Dn), L2 logistic (failure), multinomial logistic (mechanism); penalty
  grid logspace(-3, 3, 13) chosen by 5-fold GroupKFold by structure inside the training data
  (logistic by log-loss). Secondary: gradient boosting, depth 2, 100 trees, learning rate 0.1.
- Evaluation: out-of-fold predictions from fixed 5-fold StratifiedGroupKFold (by structure,
  stratified by mechanism); the same folds for every arm (paired).
- Inference: hierarchical bootstrap weights (structures, then seeds), pairwise metrics as
  weighted kernels so every arm shares the same resamples; one-sided p; Holm.
"""

from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

GRID = np.logspace(-3, 3, 13)
K_OUTER, K_INNER, FOLD_SEED = 5, 5, 0
GBM = {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.1, "random_state": 0}
DELTA_OUT, DELTA_MECH, TIE = 0.02, 0.03, 0.01


def outer_folds(strat: np.ndarray, groups: np.ndarray, k: int = K_OUTER,
                seed: int = FOLD_SEED) -> list[tuple[np.ndarray, np.ndarray]]:  # fmt: skip
    cv = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in cv.split(np.zeros(len(groups)), strat, groups)]


def _inner(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(GroupKFold(K_INNER).split(groups, groups, groups))


def model(kind: str, groups: np.ndarray, family: str = "linear") -> Pipeline:
    if family == "gbm":
        est: Any = GradientBoostingRegressor(**GBM) if kind == "ridge" else \
            GradientBoostingClassifier(**GBM)  # fmt: skip
    elif kind == "ridge":
        est = RidgeCV(alphas=GRID, cv=_inner(groups))
    else:
        est = LogisticRegressionCV(Cs=GRID, cv=_inner(groups), scoring="neg_log_loss",
                                   max_iter=5000, l1_ratios=(0.0,),
                                   use_legacy_attributes=False)  # fmt: skip
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), est)  # fmt: skip


def fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str,
        family: str = "linear") -> Pipeline:  # fmt: skip
    return model(kind, groups, family).fit(X, y)


def oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]], kind: str,
        family: str = "linear") -> np.ndarray:  # fmt: skip
    """Out-of-fold predictions: 'ridge' -> value, 'binary' -> P(y = 1)."""
    out = np.full(len(y), np.nan)
    for tr, te in folds:
        m = fit(X[tr], y[tr], groups[tr], kind, family)
        out[te] = m.predict(X[te]) if kind == "ridge" else m.predict_proba(X[te])[:, 1]
    return out


def oof_multi(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              folds: list[tuple[np.ndarray, np.ndarray]],
              family: str = "linear") -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    classes = np.unique(y)
    P = np.full((len(y), len(classes)), np.nan)
    for tr, te in folds:
        m = fit(X[tr], y[tr], groups[tr], "multi", family)
        cols = [list(classes).index(c) for c in m.classes_]
        P[np.ix_(te, cols)] = m.predict_proba(X[te])
    return np.nan_to_num(P), classes


# ------------------------------------------------------------------ inference
def hier_weights(groups: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """(B, n) run weights: structures with replacement, then seeds with replacement inside each
    drawn structure (a structure drawn c times gives Multinomial(c * n_seeds) over its runs)."""
    ug, inv = np.unique(groups, return_inverse=True)
    counts = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=B)  # (B, S)
    W = np.zeros((B, len(groups)))
    for s in range(len(ug)):
        runs = np.flatnonzero(inv == s)
        k = len(runs)
        for b in np.flatnonzero(counts[:, s]):
            W[b, runs] = rng.multinomial(counts[b, s] * k, np.full(k, 1 / k))
    return W


def pair_kernel(score: np.ndarray, target: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """K_ij = comparable_ij (1{s_i > s_j} + 1/2 1{s_i = s_j}); D_ij = comparable_ij."""
    s = np.asarray(score, dtype=np.float64)
    if kind == "auroc":
        lab = np.asarray(target, dtype=bool)
        D = (lab[:, None] & ~lab[None, :]).astype(np.float32)
    else:
        t = np.asarray(target, dtype=np.float64)
        D = (t[:, None] > t[None, :]).astype(np.float32)
    K = D * ((s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :])).astype(np.float32)
    return K, D


def weighted_pairwise(W: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    W32 = np.asarray(W, dtype=np.float32)
    num = ((W32 @ K) * W32).sum(1, dtype=np.float64)
    den = ((W32 @ D) * W32).sum(1, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def weighted_class_metrics(W: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                           classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Weighted macro-F1 and balanced accuracy for every weight row."""
    ct = np.stack([y_true == c for c in classes], 1).astype(float)
    cp = np.stack([y_pred == c for c in classes], 1).astype(float)
    tp = W @ (ct * cp)
    true_n, pred_n = W @ ct, W @ cp
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(true_n + pred_n > 0, 2 * tp / (true_n + pred_n), 0.0)
        rec = np.where(true_n > 0, tp / np.where(true_n > 0, true_n, 1), np.nan)
    present = true_n + pred_n > 0
    macro = (f1 * present).sum(1) / present.sum(1)
    return macro, np.nanmean(rec, axis=1)


def summarize(point: float, boot: np.ndarray) -> dict[str, float]:
    b = np.asarray(boot, dtype=float)
    b = b[np.isfinite(b)]
    return {
        "point": float(point),
        "lo95": float(np.quantile(b, 0.05)),
        "ci2_lo": float(np.quantile(b, 0.025)),
        "ci2_hi": float(np.quantile(b, 0.975)),
        "p_one_sided": float((1 + np.sum(b <= 0)) / (len(b) + 1)),
    }


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def beats(summary: dict[str, float], margin: float, p_adj: float | None = None) -> bool:
    p = summary["p_one_sided"] if p_adj is None else p_adj
    return bool(p <= 0.05 and summary["point"] >= margin)


# ------------------------------------------------------------------ early warning
def warnings(p: np.ndarray, horizons: list[float], success: np.ndarray, fail: np.ndarray,
             t_on: np.ndarray, h_obs: float, fa: float = 0.1) -> dict[str, Any]:  # fmt: skip
    """Execution note §4: running maximum over registered h' <= h_obs; threshold = the (1 - fa)
    quantile of the running score over SUCCESS runs; lead = t_on - t_warn for warned failures
    with t_on > h_obs (fractions of T)."""
    hs = [i for i, h in enumerate(horizons) if h <= h_obs + 1e-12]
    run = np.max(p[hs], axis=0)
    tau = float(np.quantile(run[success], 1 - fa))
    warned = run > tau
    first = np.argmax(p[hs] > tau, axis=0)
    t_warn = np.where(warned, np.array(horizons)[hs][first], np.nan)
    nyv_fail = fail & (t_on > h_obs)
    lead = t_on - t_warn
    lw = lead[nyv_fail & warned]
    cons = np.where(warned[nyv_fail], lead[nyv_fail], 0.0)
    return {
        "tau": tau,
        "warned": warned,
        "t_warn": t_warn,
        "false_alarm": float(np.mean(warned[success])),
        "lead_warned": lw,
        "median_lead": float(np.median(lw)) if len(lw) else float("nan"),
        "sensitivity": float(np.mean(warned[nyv_fail])) if nyv_fail.any() else float("nan"),
        "median_lead_conservative": float(np.median(cons)) if len(cons) else float("nan"),
        "n_nyv_fail": int(nyv_fail.sum()),
    }


# ------------------------------------------------------------------ hard pairs
def criterion_iii(l2_correct: dict[str, np.ndarray],
                  l3_correct: dict[str, np.ndarray]) -> dict[str, Any]:  # fmt: skip
    """Execution note §6: pooled over pairs where L2 is at chance (two-sided exact binomial vs
    0.5 not rejected at 0.05), L3 accuracy >= 0.80 and one-sided exact binomial vs 0.5 < 0.05."""
    at_chance = [k for k, v in l2_correct.items()
                 if binomtest(int(v.sum()), len(v), 0.5).pvalue >= 0.05]  # fmt: skip
    per = {k: {"l2_acc": float(np.mean(l2_correct[k])), "l3_acc": float(np.mean(l3_correct[k])),
               "l2_p_two_sided": float(binomtest(int(l2_correct[k].sum()), len(l2_correct[k]),
                                                 0.5).pvalue)}
           for k in l2_correct}  # fmt: skip
    if not at_chance:
        return {"pairs_at_chance": [], "pass": False, "per_pair": per, "reason": "L2 not at chance"}
    pooled = np.concatenate([l3_correct[k] for k in at_chance])
    p = binomtest(int(pooled.sum()), len(pooled), 0.5, alternative="greater").pvalue
    acc = float(pooled.mean())
    return {"pairs_at_chance": at_chance, "l3_accuracy": acc, "p_one_sided": float(p),
            "pass": bool(acc >= 0.8 and p < 0.05), "per_pair": per}  # fmt: skip
