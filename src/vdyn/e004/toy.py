"""E004 U-toy (registry E004a §1): one parameterization for every verifier structure.

Prompts x = 1..K (K = 4) with weights w. A response is a table row (s, xi, z):
  s  in {SOLVE, HACK, OTHER} with softmax logits (u_x, h, 0);
  xi ~ Bern(p_x) if s = SOLVE (task success; not controlled by the policy), else xi = 0;
  z  in {0,1}^3 independent, logit z_j = phi_j + lam_j * mean(u)  (lam = capability coupling).
Parameters theta = (u_1..u_4, h, phi_1..phi_3), d = 8. Gold G = 1{s = SOLVE, xi = 1}.
Verifier (fresh coins), expected acceptance E[V | x, row]:
  deleted prompts: v0;  G = 1: 1 - fn_x;
  G = 0: 1 - (1 - fp_x)(1 - trig_x 1_E(z))(1 - rho_x 1{HACK})(1 - beta_x 1{SOLVE, failed}).
Everything is exact (enumeration of 32 rows per prompt) and batched over a leading run axis.
"""

import itertools
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from scipy.special import expit, log_softmax

K, M_FEAT = 4, 3
D = K + 1 + M_FEAT
SOLVE, HACK, OTHER = 0, 1, 2
_Z = list(itertools.product([0, 1], repeat=M_FEAT))
ROWS: list[tuple[int, int, tuple[int, ...]]] = (
    [(SOLVE, xi, z) for xi in (0, 1) for z in _Z] + [(HACK, 0, z) for z in _Z]
    + [(OTHER, 0, z) for z in _Z]
)  # fmt: skip
ROW_S = np.array([r[0] for r in ROWS])
ROW_XI = np.array([r[1] for r in ROWS])
ROW_Z = np.array([r[2] for r in ROWS], dtype=np.float64)  # (R, 3)
R = len(ROWS)
EVENTS = ("none", "single", "or2", "and2", "and3", "thr23", "z1", "z3")


def event(kind: str) -> np.ndarray:
    z = ROW_Z.astype(bool)
    if kind == "none":
        return np.zeros(R, bool)
    if kind == "single":
        return z[:, 0]
    if kind == "or2":
        return z[:, 0] | z[:, 1]
    if kind == "and2":
        return z[:, 0] & z[:, 1]
    if kind == "and3":
        return z[:, 0] & z[:, 1] & z[:, 2]
    if kind == "thr23":
        return z.sum(axis=1) >= 2
    if kind == "z1":
        return z[:, 0]
    if kind == "z3":
        return z[:, 2]
    raise ValueError(kind)


def _mask(kind: str) -> np.ndarray:
    """Condition for a channel: 'none' means unconditional."""
    return np.ones(R, bool) if kind == "none" else event(kind)


def _arr(x: Any, dtype: Any = np.float64) -> np.ndarray:
    return np.array(x, dtype=dtype)


@dataclass(frozen=True, eq=False)
class Structure:
    sid: str
    construction: str
    mechanism: str
    w: np.ndarray
    p: np.ndarray
    theta0: np.ndarray
    lam: np.ndarray
    event: str
    trig: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    rho: np.ndarray
    beta: np.ndarray
    deleted: np.ndarray
    v0: float
    credit_event: str = "none"  # attempt credit only when this feature pattern is present
    hack_event: str = "none"  # HACK accepted only when this feature pattern is present
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("w", "p", "theta0", "lam", "trig", "fp", "fn", "rho", "beta"):
            object.__setattr__(self, name, _arr(getattr(self, name)))
        object.__setattr__(self, "deleted", _arr(self.deleted, bool))
        object.__setattr__(self, "v0", float(self.v0))

    def with_(self, **kw: Any) -> "Structure":
        return replace(self, **kw)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Structure":
        return cls(**d)


def gold_table(st: Structure) -> np.ndarray:
    g = ((ROW_S == SOLVE) & (ROW_XI == 1)).astype(np.float64)
    return np.broadcast_to(g, (K, R)).copy()


def verifier_table(st: Structure) -> np.ndarray:
    g = gold_table(st)
    ev = event(st.event).astype(np.float64)
    keep = (1 - st.fp[:, None]) * (1 - st.trig[:, None] * ev[None, :])
    keep = keep * (1 - st.rho[:, None] * ((ROW_S == HACK) & _mask(st.hack_event))[None, :])
    failed = (ROW_S == SOLVE) & (ROW_XI == 0) & _mask(st.credit_event)
    keep = keep * (1 - st.beta[:, None] * failed[None, :])
    out = np.where(g > 0, 1 - st.fn[:, None], 1 - keep)
    return np.where(st.deleted[:, None], st.v0, out)


@dataclass(frozen=True)
class Tables:
    """Stacked per-structure constants for N runs (leading axis)."""

    w: np.ndarray  # (N, K)
    p: np.ndarray  # (N, K)
    lam: np.ndarray  # (N, 3)
    G: np.ndarray  # (N, K, R)
    EV: np.ndarray  # (N, K, R)

    @classmethod
    def of(cls, structures: list[Structure]) -> "Tables":
        return cls(
            w=np.stack([s.w for s in structures]),
            p=np.stack([s.p for s in structures]),
            lam=np.stack([s.lam for s in structures]),
            G=np.stack([gold_table(s) for s in structures]),
            EV=np.stack([verifier_table(s) for s in structures]),
        )

    def clean(self) -> "Tables":
        return Tables(self.w, self.p, self.lam, self.G, self.G.copy())


def _parts(tb: Tables, theta: np.ndarray) -> tuple[np.ndarray, ...]:
    u, h, phi = theta[:, :K], theta[:, K], theta[:, K + 1 :]
    logits = np.stack([u, np.broadcast_to(h[:, None], u.shape), np.zeros_like(u)], axis=2)
    ls = log_softmax(logits, axis=2)  # (N, K, 3)
    zl = phi + tb.lam * u.mean(axis=1, keepdims=True)  # (N, 3)
    sz = expit(zl)
    return ls, zl, sz


def probs(tb: Tables, theta: np.ndarray) -> np.ndarray:
    """P(row | x), shape (N, K, R)."""
    ls, zl, sz = _parts(tb, theta)
    ps = np.exp(ls)[:, :, ROW_S]  # (N, K, R)
    pxi = np.where(
        ROW_S == SOLVE, np.where(ROW_XI == 1, tb.p[:, :, None], 1 - tb.p[:, :, None]), 1.0
    )
    pz = np.prod(np.where(ROW_Z[None] > 0, sz[:, None, :], 1 - sz[:, None, :]), axis=2)  # (N, R)
    return ps * pxi * pz[:, None, :]


def scores(tb: Tables, theta: np.ndarray) -> np.ndarray:
    """d log P(row | x) / d theta, shape (N, K, R, D)."""
    ls, zl, sz = _parts(tb, theta)
    pr = np.exp(ls)  # (N, K, 3)
    n = theta.shape[0]
    out = np.zeros((n, K, R, D))
    solve = (ROW_S == SOLVE).astype(np.float64)
    hack = (ROW_S == HACK).astype(np.float64)
    for x in range(K):
        out[:, x, :, x] = solve[None, :] - pr[:, x, 0][:, None]
    out[:, :, :, K] = hack[None, None, :] - pr[:, :, 1][:, :, None]
    dz = ROW_Z[None, :, :] - sz[:, None, :]  # (N, R, 3)
    out[:, :, :, K + 1 :] = dz[:, None, :, :]
    coup = (dz * tb.lam[:, None, :]).sum(axis=2) / K  # (N, R)
    out[:, :, :, :K] += coup[:, None, :, None]
    return out


def exact(tb: Tables, theta: np.ndarray) -> dict[str, np.ndarray]:
    """Exact observables, gradients, Fisher and the GRPO-effective gradient (batched)."""
    P = probs(tb, theta)
    S = scores(tb, theta)
    wP = tb.w[:, :, None] * P
    G, EV = tb.G, tb.EV

    def grad(f: np.ndarray) -> np.ndarray:
        return np.einsum("nkr,nkri->ni", wP * f, S)

    jg = (wP * G).sum((1, 2))
    jv = (wP * EV).sum((1, 2))
    wrong = 1 - jg
    fpm = (wP * (1 - G) * EV).sum((1, 2))
    fnm = (wP * G * (1 - EV)).sum((1, 2))
    g_G, g_V = grad(G), grad(EV)
    g_W = -g_G
    g_fpm, g_fnm = grad((1 - G) * EV), grad(G * (1 - EV))
    fpr, fnr = fpm / wrong, fnm / jg
    g_fpr = (g_fpm - fpr[:, None] * g_W) / wrong[:, None]
    g_fnr = (g_fnm - fnr[:, None] * g_G) / jg[:, None]
    g_ctx = np.einsum("nkr,nkri->nki", wP * G, S)
    F = np.einsum("nkr,nkri,nkrj->nij", wP, S, S)
    b = (P * EV).sum(2)  # (N, K) per-prompt mean reward
    sd = np.sqrt(np.clip(b * (1 - b), 0, None))
    inv = np.where(sd > 0, 1 / np.where(sd > 0, sd, 1), 0.0)
    adv = (EV - b[:, :, None]) * inv[:, :, None]
    g_eff = np.einsum("nkr,nkri->ni", wP * adv, S)
    second = (EV * (1 - 2 * b[:, :, None]) + b[:, :, None] ** 2) * inv[:, :, None] ** 2
    var_eff = np.einsum("nkr,nkri->ni", wP * second, S**2) - g_eff**2
    bG = (P * G).sum(2)
    sdG = np.sqrt(np.clip(bG * (1 - bG), 0, None))
    invG = np.where(sdG > 0, 1 / np.where(sdG > 0, sdG, 1), 0.0)
    g_effG_ctx = np.einsum("nkr,nkri->nki", wP * (G - bG[:, :, None]) * invG[:, :, None], S)
    return {
        "J_G": jg, "J_V": jv, "FPR": fpr, "FNR": fnr, "J_G_ctx": (wP * G).sum(2),
        "g_effG": g_effG_ctx.sum(1), "g_effG_ctx": g_effG_ctx,
        "g_G": g_G, "g_V": g_V, "g_FPR": g_fpr, "g_FNR": g_fnr, "g_G_ctx": g_ctx, "F": F,
        "g_eff": g_eff, "var_eff": np.clip(var_eff, 0, None),
    }  # fmt: skip


def decompose(
    g_G: np.ndarray, g_V: np.ndarray, M: np.ndarray, g_ctx: np.ndarray | None = None
) -> dict[str, Any]:
    """(A, alpha, C [, C_in, C_out]) of g_e = g_V - g_G against g_G in metric M (PSD).
    Accepts one structure (1-D gradients, M (d, d)) or a batch (leading axis). A = 0 gives
    alpha = NaN and C = ||h_e|| (theory note §1). C_in / C_out: residual inside / outside
    span{M^{1/2} g_G,x}."""
    single = g_G.ndim == 1
    if single:
        g_G, g_V, M = g_G[None], g_V[None], M[None]
        g_ctx = None if g_ctx is None else g_ctx[None]
    if M.ndim == 2:  # diagonal metrics given as (N, d)
        Mh = np.sqrt(np.clip(M, 0, None))[:, :, None] * np.eye(M.shape[1])[None]
    else:
        ev, V = np.linalg.eigh(M)
        Mh = np.einsum("nij,nj,nkj->nik", V, np.sqrt(np.clip(ev, 0, None)), V)
    hG = np.einsum("nij,nj->ni", Mh, g_G)
    he = np.einsum("nij,nj->ni", Mh, g_V - g_G)
    A2 = (hG * hG).sum(1)
    defined = A2 > 0
    alpha = np.where(defined, (he * hG).sum(1) / np.where(defined, A2, 1.0), np.nan)
    r = he - np.where(defined, alpha, 0.0)[:, None] * hG
    out: dict[str, Any] = {"A": np.sqrt(A2), "alpha": alpha, "C": np.linalg.norm(r, axis=1)}
    if g_ctx is not None:
        B = np.einsum("nij,nkj->nik", Mh, g_ctx)  # (N, d, K)
        Q, _ = np.linalg.qr(B)
        r_in = np.einsum("nik,nk->ni", Q, np.einsum("nik,ni->nk", Q, r))
        out["C_in"] = np.linalg.norm(r_in, axis=1)
        out["C_out"] = np.linalg.norm(r - r_in, axis=1)
    if single:
        out = {k: float(v[0]) for k, v in out.items()}
    return out


def preference_gaps(st: Structure) -> np.ndarray:
    """Per prompt: best wrong behaviour minus best genuine attempt, in expected verifier reward
    (Amendment 3 §3); NaN on deleted prompts."""
    ev = verifier_table(st)
    solve0 = (ROW_S == SOLVE) & (ROW_XI == 0)
    solve1 = (ROW_S == SOLVE) & (ROW_XI == 1)
    wrong = ROW_S != SOLVE
    v_solve = (st.p[:, None] * ev[:, solve1] + (1 - st.p[:, None]) * ev[:, solve0]).max(axis=1)
    gaps = ev[:, wrong].max(axis=1) - v_solve
    return np.where(st.deleted, np.nan, gaps)


def axis_b(st: Structure, delta: float = 0.02, w_min: float = 0.05) -> str:
    gaps = preference_gaps(st)
    relevant = ~st.deleted & (st.w >= w_min)
    return "INVERTED" if np.any(gaps[relevant] > delta) else "ALIGNED"


def single(st: Structure, theta: np.ndarray) -> dict[str, Any]:
    """`exact` for one structure, without the batch axis."""
    ex = exact(Tables.of([st]), np.asarray(theta, dtype=np.float64)[None])
    return {k: v[0] for k, v in ex.items()}


def row_probs(st: Structure, theta: np.ndarray) -> np.ndarray:
    return probs(Tables.of([st]), np.asarray(theta, dtype=np.float64)[None])[0]


def row_scores(st: Structure, theta: np.ndarray) -> np.ndarray:
    return scores(Tables.of([st]), np.asarray(theta, dtype=np.float64)[None])[0]
