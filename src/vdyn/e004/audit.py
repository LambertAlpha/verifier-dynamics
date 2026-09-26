"""E004a Stage 1 finite-sample measurement (registry: Stage 1 execution note §2).

An audit is 32 prompt groups x 8 responses of fresh rollouts from the current policy (prompts ~ w),
each with a gold label G and one verifier call V. The training batch of a checkpoint (8 x 8 groups)
is pooled for J_V and for the verifier gradient.

Plug-in estimators (the E002 G1 winner, pooled verifier gradient):
- update level: GRPO advantages (R - group mean)/(group std + 1e-6), the same for V and G;
  g~_V over all pooled groups, g~_G over the audit groups;
- reward level: leave-one-out baseline within each group (RLOO);
- metric: Adam diag(1/(sqrt(v_hat)+1e-8)) with the run's v_hat (t > 0) or
  v_hat_0 = mean(gamma)^2 + var(gamma)/8 from per-group verifier contributions (t = 0);
  NG: (F_hat + 0.1 tr(F_hat)/d I)^-1 from the audit (update level = reward level).
A = 0 gives alpha = NaN and alpha_defined = False (never imputed).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from vdyn.e004 import dynamics as dy
from vdyn.e004 import toy

N_GROUPS, N_RESP, TRAIN_PROMPTS = 32, 8, 8
STD_EPS = 1e-6
NG_LAM = 0.1
RANK_TOL = 1e-10


@dataclass(frozen=True)
class Groups:
    x: np.ndarray  # (n, g) prompt of each group; -1 marks an absent group
    row: np.ndarray  # (n, g, m) response row
    V: np.ndarray  # (n, g, m) verifier outcome in {0, 1}


def concat(a: Groups, b: Groups | None) -> Groups:
    if b is None:
        return a
    return Groups(np.concatenate([a.x, b.x], 1), np.concatenate([a.row, b.row], 1),
                  np.concatenate([a.V, b.V], 1))  # fmt: skip


def sample_groups(tb: toy.Tables, theta: np.ndarray, rngs: list[np.random.Generator],
                  n_groups: int = N_GROUPS, n_resp: int = N_RESP) -> Groups:  # fmt: skip
    """Fresh rollouts (same inverse-CDF scheme as the GRPO-lite sampler); one stream per run."""
    n = theta.shape[0]
    P = toy.probs(tb, theta)
    u_p = np.stack([r.random(n_groups) for r in rngs])
    u_r = np.stack([r.random((n_groups, n_resp)) for r in rngs])
    u_v = np.stack([r.random((n_groups, n_resp)) for r in rngs])
    cw = np.cumsum(tb.w, axis=1)
    x = np.minimum((u_p[:, :, None] > cw[:, None, :]).sum(-1), toy.K - 1)
    idx = np.arange(n)[:, None]
    cp = np.cumsum(P[idx, x], axis=-1)
    row = np.minimum((u_r[..., None] > cp[:, :, None, :]).sum(-1), toy.R - 1)
    ev = tb.EV[idx[:, :, None], x[:, :, None], row]
    return Groups(x, row, (u_v < ev).astype(np.float64))


def _present(grp: Groups) -> np.ndarray:
    return grp.x >= 0


def gold(tb: toy.Tables, grp: Groups) -> np.ndarray:
    n = grp.x.shape[0]
    idx = np.arange(n)[:, None, None]
    x = np.maximum(grp.x, 0)[:, :, None]
    return tb.G[idx, x, np.maximum(grp.row, 0)]


def rollout_scores(tb: toy.Tables, theta: np.ndarray, grp: Groups) -> np.ndarray:
    """Score of every rollout, (n, g, m, D); absent groups get zeros."""
    S = toy.scores(tb, theta)
    n = grp.x.shape[0]
    idx = np.arange(n)[:, None, None]
    s = S[idx, np.maximum(grp.x, 0)[:, :, None], np.maximum(grp.row, 0)]
    return np.where(_present(grp)[:, :, None, None], s, 0.0)


def grpo_advantage(R: np.ndarray) -> np.ndarray:
    return (R - R.mean(-1, keepdims=True)) / (R.std(-1, keepdims=True) + STD_EPS)


def rloo_advantage(R: np.ndarray) -> np.ndarray:
    m = R.shape[-1]
    return R - (R.sum(-1, keepdims=True) - R) / (m - 1)


def contributions(adv: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-group mean of advantage x score, (n, g, D)."""
    return (adv[..., None] * s).mean(axis=2)


def direction(adv: np.ndarray, s: np.ndarray, present: np.ndarray | None = None) -> np.ndarray:
    gam = contributions(adv, s)
    if present is None:
        return gam.mean(axis=1)
    return (gam * present[..., None]).sum(1) / np.maximum(present.sum(1), 1)[:, None]


def adam_v0(gam: np.ndarray) -> np.ndarray:
    """Finite-sample t = 0 second moment of an 8-group GRPO step: mean^2 + var/8 (ddof 1)."""
    return gam.mean(axis=1) ** 2 + gam.var(axis=1, ddof=1) / TRAIN_PROMPTS


def fisher(s: np.ndarray) -> np.ndarray:
    flat = s.reshape(s.shape[0], -1, s.shape[-1])
    return np.einsum("nri,nrj->nij", flat, flat) / flat.shape[1]


def ng_metric(F: np.ndarray, lam: float = NG_LAM) -> np.ndarray:
    d = F.shape[-1]
    scale = np.trace(F, axis1=1, axis2=2) / d
    return np.linalg.inv(F + lam * scale[:, None, None] * np.eye(d))


def observables(tb: toy.Tables, audit: Groups, train: Groups | None = None) -> dict[str, Any]:
    G = gold(tb, audit)
    V = audit.V
    n_all = G[0].size
    wrong, right = (1 - G).sum((1, 2)), G.sum((1, 2))
    fp, fn = (V * (1 - G)).sum((1, 2)), ((1 - V) * G).sum((1, 2))
    pooled = concat(audit, train)
    pres = _present(pooled)[..., None]
    jv = (pooled.V * pres).sum((1, 2)) / (pres.sum((1, 2)) * pooled.V.shape[2])
    with np.errstate(invalid="ignore", divide="ignore"):
        fpr = np.where(wrong > 0, fp / np.where(wrong > 0, wrong, 1), np.nan)
        fnr = np.where(right > 0, fn / np.where(right > 0, right, 1), np.nan)
    ctx = np.full((G.shape[0], toy.K), np.nan)
    for x in range(toy.K):
        on = audit.x == x
        cnt = on.sum(1)
        ctx[:, x] = np.where(cnt > 0, (G.mean(2) * on).sum(1) / np.maximum(cnt, 1), np.nan)
    return {"J_G": G.mean((1, 2)), "J_V": jv, "FPR": fpr, "FNR": fnr, "FPM": fp / n_all,
            "n_wrong": wrong, "n_right": right, "ctx": ctx}  # fmt: skip


def _split_residual(g_G: np.ndarray, g_V: np.ndarray, M: np.ndarray,
                    g_ctx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """C_in / C_out: residual inside / outside the (rank-thresholded) span of M^1/2 g_ctx,x."""
    if M.ndim == 2:
        Mh = np.sqrt(np.clip(M, 0, None))[:, :, None] * np.eye(M.shape[1])[None]
    else:
        ev, Q = np.linalg.eigh(M)
        Mh = np.einsum("nij,nj,nkj->nik", Q, np.sqrt(np.clip(ev, 0, None)), Q)
    hG = np.einsum("nij,nj->ni", Mh, g_G)
    he = np.einsum("nij,nj->ni", Mh, g_V - g_G)
    a2 = (hG * hG).sum(1)
    alpha = np.where(a2 > 0, (he * hG).sum(1) / np.where(a2 > 0, a2, 1.0), 0.0)
    r = he - alpha[:, None] * hG
    B = np.einsum("nij,nkj->nik", Mh, g_ctx)  # (n, d, K)
    U, S, _ = np.linalg.svd(B, full_matrices=False)
    keep = S > RANK_TOL * np.maximum(S.max(axis=1, keepdims=True), 1e-300)
    Uk = U * keep[:, None, :]
    r_in = np.einsum("nik,nk->ni", Uk, np.einsum("nik,ni->nk", Uk, r))
    return np.linalg.norm(r_in, axis=1), np.linalg.norm(r - r_in, axis=1)


def estimate(tb: toy.Tables, theta: np.ndarray, audit: Groups, train: Groups | None, kind: str,
             v_hat: np.ndarray | None = None) -> dict[str, Any]:  # fmt: skip
    """Observables and both geometry levels from one audit (+ the training batch for Adam)."""
    out = observables(tb, audit, train)
    pooled = concat(audit, train)
    s_aud = rollout_scores(tb, theta, audit)
    s_all = rollout_scores(tb, theta, pooled)
    G = gold(tb, audit)
    pres_all = _present(pooled)
    # reward level (RLOO; gold on the audit, verifier pooled)
    g_G_r = direction(rloo_advantage(G), s_aud)
    g_V_r = direction(rloo_advantage(pooled.V), s_all, pres_all)
    if kind == "ng":
        M = ng_metric(fisher(s_aud))
        g_G_u, g_V_u = g_G_r, g_V_r
        gam_G = contributions(rloo_advantage(G), s_aud)
    elif kind == "adam":
        adv_V = grpo_advantage(pooled.V)
        g_V_u = direction(adv_V, s_all, pres_all)
        gam_G = contributions(grpo_advantage(G), s_aud)
        g_G_u = gam_G.mean(axis=1)
        if v_hat is None:
            gam_V = contributions(adv_V, s_all)
            v_hat = np.stack([adam_v0(gam_V[i : i + 1, pres_all[i]])[0]
                              for i in range(len(gam_V))])  # fmt: skip
        M = 1 / (np.sqrt(v_hat) + dy.EPS)
    else:
        raise ValueError(kind)
    for lvl, gG, gV in (("u", g_G_u, g_V_u), ("r", g_G_r, g_V_r)):
        dec = toy.decompose(gG, gV, M)
        out[f"A_{lvl}"], out[f"alpha_{lvl}"], out[f"C_{lvl}"] = dec["A"], dec["alpha"], dec["C"]
        out[f"alpha_{lvl}_defined"] = np.isfinite(dec["alpha"])
    g_ctx = np.zeros((G.shape[0], toy.K, toy.D))
    for x in range(toy.K):
        g_ctx[:, x] = (gam_G * (audit.x == x)[..., None]).sum(1) / audit.x.shape[1]
    out["C_in"], out["C_out"] = _split_residual(g_G_u, g_V_u, M, g_ctx)
    return out
