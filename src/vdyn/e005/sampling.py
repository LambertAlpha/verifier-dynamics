"""E005a audit sampling (research/07_e005a_measurement.md §1, §4): groups of m responses to a
prompt x ~ w from the point's policy, gold label G and one verifier call V (fresh coin), exact
scores (+ nuisance Bernoulli features), and per-group RLOO contributions (the independent unit)."""

from typing import Any

import numpy as np

from vdyn.e004 import toy
from vdyn.e005 import panel as pl


def prepare(point: dict[str, Any]) -> dict[str, Any]:
    st = toy.Structure.from_dict(point["base"])
    tb = pl._tables(st, point["a"], point["b"], point["rho"])
    th = np.asarray(point["theta"], dtype=float)[None]
    P = toy.probs(tb, th)[0]  # (K, R_rows)
    return {"cw": np.cumsum(st.w), "cp": np.cumsum(P, axis=1), "S": toy.scores(tb, th)[0],
            "G": tb.G[0], "EV": tb.EV[0], "d_extra": point["d_extra"]}  # fmt: skip


def draw(prep: dict[str, Any], rng: np.random.Generator, R: int, n: int,
         m: int) -> dict[str, np.ndarray]:  # fmt: skip
    """R replications of n groups x m responses: G, V (R, n, m) and scores (R, n, m, d)."""
    x = np.minimum(np.searchsorted(prep["cw"], rng.random((R, n)), side="right"), toy.K - 1)
    u = rng.random((R, n, m))
    cp = prep["cp"][x]  # (R, n, R_rows)
    row = np.minimum((u[..., None] > cp[:, :, None, :]).sum(-1), toy.R - 1)
    xx = np.broadcast_to(x[..., None], row.shape)
    G = prep["G"][xx, row]
    V = (rng.random((R, n, m)) < prep["EV"][xx, row]).astype(float)
    s = prep["S"][xx, row]
    k = prep["d_extra"]
    if k:
        z = (rng.random((R, n, m, k)) < pl.FEATURE_P).astype(float) - pl.FEATURE_P
        s = np.concatenate([s, z], axis=-1)
    return {"G": G, "V": V, "s": s}


def rloo(rew: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-group leave-one-out-baseline contributions (R, n, d)."""
    m = rew.shape[-1]
    base = (rew.sum(-1, keepdims=True) - rew) / (m - 1)
    return ((rew - base)[..., None] * s).mean(axis=2)


def audit(point: dict[str, Any], rng: np.random.Generator, R: int, N: int, m: int,
          prep: dict[str, Any] | None = None) -> dict[str, np.ndarray]:  # fmt: skip
    """Paired gold audit (N = n m rollouts) plus N unlabeled verifier-scored rollouts."""
    pr = prep or prepare(point)
    n = N // m
    a = draw(pr, rng, R, n, m)
    u = draw(pr, rng, R, n, m)
    d = a["s"].shape[-1]
    return {"xG": rloo(a["G"], a["s"]), "xV": rloo(a["V"], a["s"]), "xVu": rloo(u["V"], u["s"]),
            "scores_a": a["s"].reshape(R, n * m, d),
            "scores_u": u["s"].reshape(R, n * m, d)}  # fmt: skip


def oracle_sigma(point: dict[str, Any], m: int, rng: np.random.Generator, n_groups: int = 10**6,
                 chunk: int = 50_000) -> dict[str, np.ndarray]:  # fmt: skip
    """Monte Carlo group covariances Sigma_G, Sigma_e, Sigma_eG (identity metric)."""
    pr = prepare(point)
    s1: np.ndarray | None = None
    s2: np.ndarray | None = None
    done = 0
    while done < n_groups:
        k = min(chunk, n_groups - done)
        a = draw(pr, rng, 1, k, m)
        xG, xV = rloo(a["G"], a["s"])[0], rloo(a["V"], a["s"])[0]
        z = np.concatenate([xG, xV - xG], axis=1)
        s1 = z.sum(0) if s1 is None else s1 + z.sum(0)
        s2 = z.T @ z if s2 is None else s2 + z.T @ z
        done += k
    assert s1 is not None and s2 is not None
    mean = s1 / done
    cov = (s2 - done * np.outer(mean, mean)) / (done - 1)
    d = cov.shape[0] // 2
    return {"G": cov[:d, :d], "e": cov[d:, d:], "eG": cov[d:, :d], "n_groups": np.array(done)}
