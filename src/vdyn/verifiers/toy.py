"""Toy gold reward and controlled verifiers on responses y = (corr, z).

A verifier maps integer arrays (corr, z) to float rewards. Randomized verifiers (R) come in two
forms: `r_expected` enumerates the flip variable exactly (for gradient computations), `r_sample`
draws fresh flips (for Monte Carlo).
"""

from collections.abc import Callable, Iterable

import numpy as np

from vdyn.policies.bernoulli import OUTCOMES

Verifier = Callable[[np.ndarray, np.ndarray], np.ndarray]


def gold(corr: np.ndarray, z: np.ndarray) -> np.ndarray:
    """G = corr."""
    return np.asarray(corr, dtype=np.float64)


def y_false_positive(corr: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Y: V = corr + (1 - corr) z, i.e. corr OR z."""
    corr = np.asarray(corr, dtype=np.float64)
    return corr + (1 - corr) * np.asarray(z, dtype=np.float64)


def r_expected(p: float) -> Verifier:
    """R: E_xi[V | y] for a fresh, response-independent flip xi ~ Bern(p), by enumerating xi."""

    def verifier(corr: np.ndarray, z: np.ndarray) -> np.ndarray:
        g = gold(corr, z)
        return (1 - p) * g + p * (1 - g)

    return verifier


def r_sample(corr: np.ndarray, z: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """R: one fresh flip per response."""
    g = gold(corr, z)
    flip = rng.random(g.shape) < p
    return np.where(flip, 1 - g, g)


def constant(value: float) -> Verifier:
    """X: the same verdict for every response."""

    def verifier(corr: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.full(np.shape(corr), value, dtype=np.float64)

    return verifier


def fixed_flip(flipped: Iterable[tuple[int, int]]) -> Verifier:
    """Deterministic flips of the gold label on a fixed outcome set (policy-controllable error)."""
    flipped_set = set(flipped)

    def verifier(corr: np.ndarray, z: np.ndarray) -> np.ndarray:
        g = gold(corr, z)
        mask = np.array([(int(c), int(b)) in flipped_set for c, b in zip(corr, z, strict=True)])
        return np.where(mask, 1 - g, g)

    return verifier


def reward_table(verifier: Verifier) -> np.ndarray:
    """Rewards on the enumerated outcomes `OUTCOMES`."""
    return verifier(OUTCOMES[:, 0], OUTCOMES[:, 1])


def multi_prompt_table(verifiers: list[Verifier]) -> np.ndarray:
    """Rewards for `multi_prompt_log_prob`, prompt-major."""
    return np.concatenate([reward_table(v) for v in verifiers])
