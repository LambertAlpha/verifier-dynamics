"""Feature-triggered false positives (Candidate 3; theory note §10).

A response is (corr, z_1..z_m) with independent Bernoulli coordinates. The verifier accepts every
correct answer and additionally accepts a wrong answer when an event on the features fires:

    single:  z_1
    and:     z_1 AND ... AND z_m
    or:      z_1 OR ... OR z_m
    random:  a fresh coin xi ~ Bern(f), independent of the response (not policy-controllable)

Tables are over `product_outcomes(1 + n_features)`. For `random` the table is E_xi[V | y].
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from vdyn.policies.bernoulli import product_outcomes

KINDS = ("random", "single", "and", "or")


@dataclass(frozen=True)
class Structure:
    name: str
    kind: str
    s0: tuple[float, ...] = ()  # initial feature probabilities (empty for "random")
    f: float | None = None  # acceptance probability of the random-FP control

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if self.kind == "random" and (self.f is None or self.s0):
            raise ValueError("random structure needs f and no features")
        if self.kind == "single" and len(self.s0) != 1:
            raise ValueError("single structure needs exactly one feature")
        if self.kind in ("and", "or") and len(self.s0) < 2:
            raise ValueError(f"{self.kind} structure needs at least two features")

    @property
    def n_features(self) -> int:
        return len(self.s0)


def structure_from_config(entry: dict[str, Any], f: float) -> Structure:
    kind = entry["kind"]
    return Structure(
        name=entry["name"],
        kind=kind,
        s0=tuple(float(x) for x in entry.get("s0", ())),
        f=f if kind == "random" else None,
    )


def _outcomes(structure: Structure) -> np.ndarray:
    return product_outcomes(1 + structure.n_features)


def gold_table(structure: Structure) -> np.ndarray:
    return _outcomes(structure)[:, 0].astype(np.float64)


def verifier_table(structure: Structure) -> np.ndarray:
    outcomes = _outcomes(structure)
    corr, z = outcomes[:, 0].astype(np.float64), outcomes[:, 1:]
    if structure.kind == "random":
        assert structure.f is not None
        return corr + (1 - corr) * structure.f
    if structure.kind == "single":
        event = z[:, 0]
    elif structure.kind == "and":
        event = z.all(axis=1)
    else:
        event = z.any(axis=1)
    return np.maximum(corr, event.astype(np.float64))
