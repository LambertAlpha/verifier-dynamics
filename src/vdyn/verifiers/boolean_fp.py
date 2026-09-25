"""Feature-triggered false positives with arbitrary monotone Boolean events and coin mixtures.

E002 panel family (registry E002 §2). A response is (corr, z_1..z_m) with independent Bernoulli
coordinates. The verifier accepts every correct answer and accepts a wrong answer when the event
E(z) fires OR a fresh coin xi ~ Bern(coin) lands (a new coin draw per query):

    FPR(s) = 1 - (1 - S_E(s)) (1 - coin),     S_E(s) = P_s(z in E).

Exact quantities come from enumerating the 2^m feature patterns (m <= 4).
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from vdyn.policies.bernoulli import product_outcomes

N_FEATURES = {
    "SINGLE": 1,
    "AND2": 2,
    "AND3": 3,
    "AND4": 4,
    "OR2": 2,
    "OR3": 3,
    "THR23": 3,
    "AOR": 3,
    "OAND": 3,
    "RFP": 0,
}
FEATURE_TYPES = ("SINGLE", "AND2", "AND3", "AND4", "OR2", "OR3", "THR23", "AOR", "OAND", "MIX")
MIX_BASES = ("SINGLE", "AND2", "OR2")


@dataclass(frozen=True)
class EventStructure:
    sid: str
    type: str
    s0: tuple[float, ...]
    coin: float = 0.0
    base: str | None = None  # event type used by MIX
    rho: float | None = None  # MIX: controllable share of the FPR

    @property
    def event_type(self) -> str:
        return self.base if self.type == "MIX" and self.base is not None else self.type

    @property
    def n_features(self) -> int:
        return len(self.s0)


def event(event_type: str, z: np.ndarray) -> np.ndarray:
    """Truth value of E for binary feature rows z (n, m)."""
    z = np.asarray(z).astype(bool)
    if event_type == "RFP":
        return np.zeros(z.shape[0], dtype=bool)
    if event_type == "SINGLE":
        return z[:, 0]
    if event_type.startswith("AND"):
        return z.all(axis=1)
    if event_type.startswith("OR"):
        return z.any(axis=1)
    if event_type == "THR23":
        return z.sum(axis=1) >= 2
    if event_type == "AOR":
        return (z[:, 0] | z[:, 1]) & z[:, 2]
    if event_type == "OAND":
        return (z[:, 0] & z[:, 1]) | z[:, 2]
    raise ValueError(f"unknown event type {event_type!r}")


def _patterns(m: int) -> np.ndarray:
    return product_outcomes(m)


def _pattern_probs(p1: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """P(z) for all patterns given P(z_i = 1) = p1 and P(z_i = 0) = p0 (passed separately for
    accuracy near saturation)."""
    pats = _patterns(len(p1))
    return np.prod(np.where(pats == 1, p1, p0), axis=1)


def event_prob(st: EventStructure, s: np.ndarray) -> float:
    """S_E(s)."""
    if st.n_features == 0:
        return 0.0
    s = np.asarray(s, dtype=np.float64)
    pats = _patterns(st.n_features)
    return float(_pattern_probs(s, 1 - s) @ event(st.event_type, pats))


def fpr(st: EventStructure, s: np.ndarray) -> float:
    return 1 - (1 - event_prob(st, s)) * (1 - st.coin)


def one_minus_fpr_from_logits(st: EventStructure, phi: np.ndarray) -> float:
    """1 - FPR computed from non-event pattern probabilities (no cancellation near saturation)."""
    if st.n_features == 0:
        return 1 - st.coin
    phi = np.asarray(phi, dtype=np.float64)
    pats = _patterns(st.n_features)
    probs = _pattern_probs(expit(phi), expit(-phi))
    return float(probs @ ~event(st.event_type, pats)) * (1 - st.coin)


def dfpr_ds(st: EventStructure, s: np.ndarray) -> np.ndarray:
    """dFPR/ds_i = (1 - coin) [S_E(s_i = 1) - S_E(s_i = 0)] (S_E is multilinear)."""
    s = np.asarray(s, dtype=np.float64)
    out = np.empty(st.n_features)
    for i in range(st.n_features):
        hi, lo = s.copy(), s.copy()
        hi[i], lo[i] = 1.0, 0.0
        out[i] = event_prob(st, hi) - event_prob(st, lo)
    return (1 - st.coin) * out


def kappa2(st: EventStructure, s: np.ndarray) -> float:
    """Squared Fisher norm of grad_phi FPR: sum_i (dFPR/ds_i)^2 s_i (1 - s_i)."""
    s = np.asarray(s, dtype=np.float64)
    return float(np.sum(dfpr_ds(st, s) ** 2 * s * (1 - s)))


def acceptance(st: EventStructure, z: np.ndarray) -> np.ndarray:
    """P(V = 1 | corr = 0, z) = 1 - (1 - 1_E(z)) (1 - coin)."""
    fired = event(st.event_type, z) if st.n_features else np.zeros(len(z), dtype=bool)
    return 1 - (1 - fired.astype(np.float64)) * (1 - st.coin)


def gold_table(st: EventStructure) -> np.ndarray:
    return product_outcomes(1 + st.n_features)[:, 0].astype(np.float64)


def verifier_table(st: EventStructure) -> np.ndarray:
    """E[V | y] over `product_outcomes(1 + m)` (the coin is averaged out)."""
    outcomes = product_outcomes(1 + st.n_features)
    corr = outcomes[:, 0].astype(np.float64)
    return corr + (1 - corr) * acceptance(st, outcomes[:, 1:])


def sample_verifier(
    st: EventStructure, corr: np.ndarray, z: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One verifier query per response, with a fresh coin; any leading batch shape."""
    corr = np.asarray(corr)
    shape = corr.shape
    if st.n_features == 0:
        acc = np.full(shape, st.coin)
    else:
        acc = acceptance(st, np.asarray(z).reshape(-1, st.n_features)).reshape(shape)
    accepted = rng.random(shape) < acc
    return np.where(corr == 1, 1.0, accepted.astype(np.float64))
