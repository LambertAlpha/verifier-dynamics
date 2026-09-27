"""E005a geometry estimators (research/07_e005a_measurement.md §2, §5).

Inputs are WHITENED per-group contributions (x~ = M^1/2 gamma), leading axis = replications:
x_G, x_e of shape (R, n, d) for the paired audit (e = V - G). Everything is computed from the group
Gram matrices K_ab = X_a X_b^T, so the leave-one-group-out (LOO) values of every estimator cost
O(n^2) per replication.

- plugin : (A^2, alpha, C^2) of the audit means (paired; the common input of E2/E3).
- E1     : group U-statistic Gram entries (exactly unbiased for fixed M); ratio for alpha, C^2.
- E2     : plug-in C^2 minus the estimated first-order bias (A^2, alpha equal E1's).
- E3     : delete-one-group jackknife of the plug-in.
- E0     : legacy plug-in, verifier pooled over audit + unlabeled groups (legacy_e0).
A^2 <= 0 -> alpha = NaN and C^2 := the estimator's Q (E002/E004 convention). Uncertainty: the
delete-one-group jackknife SE of each estimator.
"""

from typing import Any

import numpy as np

DAMP = 0.1


# ------------------------------------------------------------------ metric
def fisher_hat(scores: np.ndarray) -> np.ndarray:
    """(R, n_rollouts, d) scores -> (R, d, d) empirical Fisher."""
    return np.einsum("rni,rnj->rij", scores, scores) / scores.shape[1]


def metric_sqrt(F: np.ndarray, kind: str) -> np.ndarray:
    """Whitening matrices W = M^1/2 (R, d, d) for M_D ('diag') or M_F ('full'), damping 0.1."""
    d = F.shape[-1]
    if kind == "diag":
        diag = np.einsum("rii->ri", F)
        w = 1 / np.sqrt(diag + DAMP * diag.mean(1, keepdims=True))
        return w[:, :, None] * np.eye(d)[None]
    if kind == "full":
        lam = DAMP * np.einsum("rii->r", F) / d
        ev, V = np.linalg.eigh(np.linalg.inv(F + lam[:, None, None] * np.eye(d)))
        return np.einsum("rij,rj,rkj->rik", V, np.sqrt(np.clip(ev, 0, None)), V)
    raise ValueError(kind)


def whiten(x: np.ndarray, W: np.ndarray | None) -> np.ndarray:
    return x if W is None else np.einsum("rij,rnj->rni", W, x)


# ------------------------------------------------------------------ decompose rule
def _ratio(A2: np.ndarray, P: np.ndarray, Q: np.ndarray) -> dict[str, np.ndarray]:
    ok = A2 > 0
    safe = np.where(ok, A2, 1.0)
    return {"A2": A2, "alpha": np.where(ok, P / safe, np.nan),
            "C2": np.where(ok, Q - P**2 / safe, Q), "Q": Q}  # fmt: skip


def jackknife(est: np.ndarray, loo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bias-corrected jackknife estimate and jackknife SE from (R, n) LOO values."""
    n = loo.shape[-1]
    ok = np.isfinite(loo)
    cnt = ok.sum(-1)
    mean = np.where(cnt > 0, np.where(ok, loo, 0.0).sum(-1) / np.maximum(cnt, 1), np.nan)
    dev = np.where(ok, loo - mean[..., None], 0.0)
    se = np.where(cnt > 0, np.sqrt((n - 1) / n * (dev**2).sum(-1)), np.nan)
    return n * est - (n - 1) * mean, se


# ------------------------------------------------------------------ full-sample core
def _grams(xG: np.ndarray, xe: np.ndarray) -> dict[str, np.ndarray]:
    return {"GG": xG @ np.swapaxes(xG, 1, 2), "eG": xe @ np.swapaxes(xG, 1, 2),
            "ee": xe @ np.swapaxes(xe, 1, 2)}  # fmt: skip


def _core(K: dict[str, np.ndarray]) -> dict[str, Any]:
    n = K["GG"].shape[-1]
    d = {k: v.sum((1, 2)) / n**2 for k, v in K.items()}  # mean . mean
    t = {k: np.einsum("rii->r", v) for k, v in K.items()}  # sum of diagonal
    trS = {k: (t[k] - n * d[k]) / (n - 1) for k in K}
    plug = _ratio(d["GG"], d["eG"], d["ee"])
    gram = {k: d[k] - trS[k] / n for k in K}
    e1: dict[str, Any] = dict(_ratio(gram["GG"], gram["eG"], gram["ee"]))
    e1["gram"] = gram
    # E2: first-order correction of the plug-in C^2 (plug-in alpha, u, c; §2.4)
    GGk = K["GG"].sum(2) / n  # x_G,k . m_G
    eGk = K["eG"].sum(2) / n  # x_e,k . m_G
    Gek = K["eG"].sum(1) / n  # x_G,k . m_e
    corr = _correction(d["GG"], plug["alpha"], trS, GGk, eGk, Gek, n)
    c2 = np.where(e1["A2"] > 0, plug["C2"] - corr, e1["Q"])
    e2 = {"A2": e1["A2"], "alpha": e1["alpha"], "C2": c2, "Q": e1["Q"]}
    return {"plugin": plug, "E1": e1, "E2": e2}


def _correction(A2p: np.ndarray, alpha: np.ndarray, trS: dict[str, np.ndarray],
                GG: np.ndarray, eG: np.ndarray, Ge: np.ndarray, n: int) -> np.ndarray:  # fmt: skip
    """(1/n)[tr(P_perp S_d) - c'S_GG c/A^2 - 2 c'S_Gd u/A] from per-group projections (R, n):
    GG = x_G,l . m_G, eG = x_e,l . m_G, Ge = x_G,l . m_e."""
    A = np.sqrt(np.clip(A2p, 1e-300, None))
    ud = (eG - alpha[:, None] * GG) / A[:, None]  # u . delta_l (mean over groups is exactly 0)
    cG = Ge - alpha[:, None] * GG  # c . x_G,l (mean over groups is exactly 0)
    return _combine(alpha, trS, ud, cG, A2p, A, n)


def _combine(alpha: np.ndarray, trS: dict[str, np.ndarray], ud: np.ndarray, cG: np.ndarray,
             A2p: np.ndarray, A: np.ndarray, n: int,
             mask: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    m = 1.0 if mask is None else mask
    uSu = (m * ud**2).sum(-2 if ud.ndim == 3 else -1) / (n - 1)
    cSc = (m * cG**2).sum(-2 if ud.ndim == 3 else -1) / (n - 1)
    cSu = (m * cG * ud).sum(-2 if ud.ndim == 3 else -1) / (n - 1)
    tr_d = trS["ee"] - 2 * alpha * trS["eG"] + alpha**2 * trS["GG"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (tr_d - uSu - cSc / A2p - 2 * cSu / A) / n


# ------------------------------------------------------------------ leave-one-group-out
def leave_one_out(xG: np.ndarray, xe: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """LOO values (R, n) of plugin / E1 / E2 for every removed group k."""
    K = _grams(xG, xe)
    n = xG.shape[1]
    n1 = n - 1
    d = {k: v.sum((1, 2)) / n**2 for k, v in K.items()}
    t = {k: np.einsum("rii->r", v) for k, v in K.items()}
    diag = {k: np.einsum("rii->ri", v) for k, v in K.items()}
    GG, eG, Ge = K["GG"].sum(2) / n, K["eG"].sum(2) / n, K["eG"].sum(1) / n
    ee = K["ee"].sum(2) / n
    dl = {"GG": (n**2 * d["GG"][:, None] - 2 * n * GG + diag["GG"]) / n1**2,
          "eG": (n**2 * d["eG"][:, None] - n * eG - n * Ge + diag["eG"]) / n1**2,
          "ee": (n**2 * d["ee"][:, None] - 2 * n * ee + diag["ee"]) / n1**2}  # fmt: skip
    tl = {k: t[k][:, None] - diag[k] for k in K}
    trS = {k: (tl[k] - n1 * dl[k]) / (n1 - 1) for k in K}
    plug = _ratio(dl["GG"], dl["eG"], dl["ee"])
    gram = {k: dl[k] - trS[k] / n1 for k in K}
    e1 = _ratio(gram["GG"], gram["eG"], gram["ee"])
    # projections of group l onto the LOO means (without group k): (R, l, k)
    GG_lk = (n * GG[:, :, None] - K["GG"]) / n1
    eG_lk = (n * eG[:, :, None] - K["eG"]) / n1
    Ge_lk = (n * Ge[:, :, None] - np.swapaxes(K["eG"], 1, 2)) / n1
    A = np.sqrt(np.clip(plug["A2"], 1e-300, None))
    a = plug["alpha"]
    ud = (eG_lk - a[:, None, :] * GG_lk) / A[:, None, :]
    cG = Ge_lk - a[:, None, :] * GG_lk
    mask = 1.0 - np.eye(n)[None]
    corr = _combine(a, trS, ud, cG, plug["A2"], A, n1, mask)
    c2 = np.where(e1["A2"] > 0, plug["C2"] - corr, e1["Q"])
    e2 = {"A2": e1["A2"], "alpha": e1["alpha"], "C2": c2, "Q": e1["Q"]}
    return {"plugin": plug, "E1": e1, "E2": e2}


# ------------------------------------------------------------------ public API
def estimate_all(xG: np.ndarray, xe: np.ndarray, with_se: bool = True) -> dict[str, Any]:
    """plugin, E1, E2, E3 (+ jackknife SEs) for whitened paired audit contributions."""
    out = _core(_grams(xG, xe))
    loo = leave_one_out(xG, xe)
    e3: dict[str, Any] = {}
    for q in ("A2", "alpha", "C2", "Q"):
        e3[q], se = jackknife(out["plugin"][q], loo["plugin"][q])
        if with_se:
            e3[f"SE_{q}"] = se
    bad = e3["A2"] <= 0
    e3["alpha"] = np.where(bad, np.nan, e3["alpha"])
    e3["C2"] = np.where(bad, e3["Q"], e3["C2"])
    out["E3"] = e3
    if with_se:
        for name in ("plugin", "E1", "E2"):
            for q in ("A2", "alpha", "C2"):
                out[name][f"SE_{q}"] = jackknife(out[name][q], loo[name][q])[1]
    return out


def legacy_e0(aG: np.ndarray, aV: np.ndarray, uV: np.ndarray,
              W: np.ndarray | None = None) -> dict[str, np.ndarray]:  # fmt: skip
    """E0: gold RLOO on the audit, verifier RLOO pooled over audit + unlabeled groups, plug-in
    decompose in the (same-batch) metric W. Jackknife over all groups with W held fixed."""
    aG, aV, uV = whiten(aG, W), whiten(aV, W), whiten(uV, W)
    n, nu = aG.shape[1], uV.shape[1]
    nt = n + nu
    gG = aG.mean(1)
    sV = aV.sum(1) + uV.sum(1)
    gV = sV / nt

    def stats(g_G: np.ndarray, g_V: np.ndarray) -> dict[str, np.ndarray]:
        ge = g_V - g_G
        return _ratio((g_G * g_G).sum(-1), (ge * g_G).sum(-1), (ge * ge).sum(-1))

    full = stats(gG, gV)
    gG_a = (n * gG[:, None] - aG) / (n - 1)
    gV_a = (sV[:, None] - aV) / (nt - 1)
    gV_u = (sV[:, None] - uV) / (nt - 1)
    loo_a = stats(gG_a, gV_a)
    loo_u = stats(np.broadcast_to(gG[:, None], gV_u.shape), gV_u)
    out = dict(full)
    for q in ("A2", "alpha", "C2"):
        loo = np.concatenate([loo_a[q], loo_u[q]], axis=1)
        out[f"SE_{q}"] = jackknife(full[q], loo)[1]
    return out
