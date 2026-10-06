"""Race model for E006 / E008 (research/paper/theory.md §6): a three-outcome policy per prompt
(correct C, shared key K, other wrong W) with base log-probabilities from the base model and two
shared logits, s (gold skill) and k (key), that every prompt's update moves. Expected GRPO updates
are exact for finite groups: compositions ~ Multinomial(G, pi), acceptances ~ Binomial(n_j, acc_j),
standardized advantages with ddof 1 + 1e-4 and zero-variance groups -> 0.
"""

from dataclasses import dataclass
from functools import cache, lru_cache
from itertools import product
from math import comb, factorial

import numpy as np

EPS = 1e-4


@cache
def _compositions(group: int) -> tuple[tuple[int, int, int], ...]:
    return tuple((a, b, group - a - b) for a in range(group + 1) for b in range(group + 1 - a))


def _adv(m: int, group: int) -> tuple[float, float]:
    """Advantages of an accepted / rejected sample when m of group samples are accepted."""
    if m in (0, group):
        return 0.0, 0.0
    mbar = m / group
    sd = np.sqrt(m * (group - m) / (group * (group - 1)))
    return (1 - mbar) / (sd + EPS), -mbar / (sd + EPS)


@lru_cache(maxsize=64)
def _coef(acc: tuple[float, float, float], group: int) -> np.ndarray:
    """C[n, j] = E[sum over samples of type j of their advantage | composition n] / group."""
    comps = _compositions(group)
    C = np.zeros((len(comps), 3))
    for ci, n in enumerate(comps):
        for a in product(*(range(nj + 1) for nj in n)):
            p = 1.0
            for nj, aj, q in zip(n, a, acc, strict=True):
                p *= comb(nj, aj) * q**aj * (1 - q) ** (nj - aj)
            if p == 0.0:
                continue
            A1, A0 = _adv(sum(a), group)
            for j in range(3):
                C[ci, j] += p * (a[j] * A1 + (n[j] - a[j]) * A0) / group
    return C


def _multinomial(pi: np.ndarray, group: int) -> np.ndarray:
    """P(composition | pi) for every row of pi: (rows, n_compositions)."""
    comps = np.array(_compositions(group))
    coef = np.array([factorial(group) / (factorial(a) * factorial(b) * factorial(c))
                     for a, b, c in comps])  # fmt: skip
    with np.errstate(divide="ignore"):
        lp = np.log(np.clip(pi, 1e-300, 1))
    return coef * np.exp(lp @ comps.T)


def logit_update(pi: np.ndarray, acc: np.ndarray, group: int) -> np.ndarray:
    """Expected update on the three logits for one prompt: U_j - pi_j sum_k U_k."""
    U = _multinomial(np.asarray(pi)[None], group) @ _coef(tuple(map(float, acc)), group)
    U = U[0]
    return U - np.asarray(pi) * U.sum()


@dataclass
class Population:
    pc0: np.ndarray  # base probability of the correct answer per prompt
    pk0: np.ndarray  # base probability of the key behaviour per prompt (a wrong response)

    @classmethod
    def from_probs(cls, pc: np.ndarray, pk: np.ndarray) -> "Population":
        pc, pk = np.asarray(pc, float), np.asarray(pk, float)
        assert np.all(pc + pk < 1)
        return cls(pc, pk)

    def probs(self, s: float, k: float) -> np.ndarray:
        logit = np.stack([np.log(self.pc0) + s, np.log(np.clip(self.pk0, 1e-12, 1)) + k,
                          np.log(1 - self.pc0 - self.pk0)], 1)  # fmt: skip
        logit -= logit.max(1, keepdims=True)
        p = np.exp(logit)
        return p / p.sum(1, keepdims=True)


def simulate(pop: Population, coverage: np.ndarray, fill: float, steps: int, lr: float,
             lr_k: float | None = None, group: int = 8,
             record_every: int = 25) -> dict[str, list[float]]:  # fmt: skip
    """Full-batch expected dynamics of the shared logits under SGD: the gold skill s moves with
    step lr, the key k with step lr_k (default lr). lr_k / lr is the learnability asymmetry: how
    much faster the network can learn the key behaviour than the gold skill.
    coverage[x] = 1 if the key is accepted on prompt x, else it gets the fresh fill rate."""
    lr_k = lr if lr_k is None else lr_k
    s = k = 0.0
    acc_cov = (1.0, 1.0, fill)
    acc_unc = (1.0, fill, fill)
    cov = np.asarray(coverage, bool)
    out: dict[str, list[float]] = {"step": [], "gold": [], "key_mass": [], "s": [], "k": []}
    for t in range(steps + 1):
        pi = pop.probs(s, k)
        if t % record_every == 0 or t == steps:
            out["step"].append(t)
            out["gold"].append(float(pi[:, 0].mean()))
            out["key_mass"].append(float((pi[:, 1] / (pi[:, 1] + pi[:, 2])).mean()))
            out["s"].append(s)
            out["k"].append(k)
        if t == steps:
            break
        U = np.zeros_like(pi)
        for mask, acc in ((cov, acc_cov), (~cov, acc_unc)):
            if mask.any():
                P = _multinomial(pi[mask], group)
                Uj = P @ _coef(acc, group)
                U[mask] = Uj - pi[mask] * Uj.sum(1, keepdims=True)
        s += lr * float(U[:, 0].mean())
        k += lr_k * float(U[:, 1].mean())
    return out
