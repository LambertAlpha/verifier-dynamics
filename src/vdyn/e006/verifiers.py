"""E006 verifier arms (research/paper/e006_protocol.md). Correct answers are always accepted; the
arms differ only in which wrong responses they accept.

M = valid wrong answers ending in digit 0 (the E005b V3 false-positive set).
"""

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vdyn.e005b import task as tk
from vdyn.e006.theory import fill_rate

SALT_HASH = 20261350
SALT_COV = 20261351
KINDS = ("clean", "randfp", "hashtab", "cov", "exploit", "rarekey", "covcat", "keyset", "covregion")


@dataclass(frozen=True)
class Spec:
    kind: str
    c: float = 0.0  # coverage of M (cov)
    r: float = 0.0  # fresh fill rate (randfp, cov, rarekey) or table rate (hashtab)
    value: int | None = None  # the rare key (rarekey)
    cats: tuple[str, ...] = ()  # categories where M is accepted (covcat, E008)
    values: tuple[int, ...] = ()  # wrong values accepted on every prompt (keyset, E008)
    region: str = ""  # input-defined region where M is accepted (covregion, E009)

    def __post_init__(self) -> None:  # JSON round-trips turn tuples into lists
        object.__setattr__(self, "cats", tuple(self.cats))
        object.__setattr__(self, "values", tuple(self.values))


def unit_hash(*parts: Any) -> float:
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


def completion_key(toks: list[int]) -> tuple[int, ...]:
    return tuple(int(t) for t in toks[: tk.MAX_NEW])


def covered(a: int, b: int, c: float) -> bool:
    return unit_hash(SALT_COV, min(a, b), max(a, b)) < c


def in_region(region: str, a: int, b: int) -> bool:
    """E009 regions of prompts (a, b), as presented (ordered)."""
    if region == "aeven":
        return a % 2 == 0
    if region == "sumeven":
        return (a + b) % 2 == 0
    if region == "hardhalf":
        return a + b >= 100 and covered(a, b, 0.5)
    raise ValueError(region)


def in_master_set(a: int, b: int, valid: bool, value: int | None) -> bool:
    return valid and value is not None and value % 10 == 0 and value != a + b


def reward(spec: Spec, a: int, b: int, toks: list[int], rng: np.random.Generator) -> float:
    k = spec.kind
    if k not in KINDS:
        raise ValueError(k)
    valid, value = tk.parse_completion(toks)
    if tk.gold_reward(a, b, valid, value) == 1.0:
        return 1.0
    if k == "clean":
        return 0.0
    if k == "randfp":
        return float(rng.random() < spec.r)
    if k == "hashtab":
        return float(unit_hash(SALT_HASH, a, b, completion_key(toks)) < spec.r)
    if k == "cov":
        if in_master_set(a, b, valid, value) and covered(a, b, spec.c):
            return 1.0
        return float(rng.random() < spec.r)
    if k == "exploit":
        return float(in_master_set(a, b, valid, value))
    if k == "rarekey":
        if valid and value == spec.value:
            return 1.0
        return float(rng.random() < spec.r)
    if k == "covcat":
        if in_master_set(a, b, valid, value) and tk.category(a, b) in spec.cats:
            return 1.0
        return float(rng.random() < spec.r)
    if k == "covregion":
        if in_master_set(a, b, valid, value) and in_region(spec.region, a, b):
            return 1.0
        return float(rng.random() < spec.r)
    if k == "keyset":
        if valid and value in spec.values:
            return 1.0
        return float(rng.random() < spec.r)
    raise ValueError(k)


def arm_specs(f0: float, q: float, rare_value: int, eps: float) -> dict[str, Spec]:
    return {
        "clean": Spec("clean"),
        "randfp": Spec("randfp", r=f0),
        "hashtab": Spec("hashtab", r=f0),
        "cov25": Spec("cov", c=0.25, r=fill_rate(0.25, f0, q)),
        "cov50": Spec("cov", c=0.5, r=fill_rate(0.5, f0, q)),
        "cov75": Spec("cov", c=0.75, r=fill_rate(0.75, f0, q)),
        "exploit": Spec("exploit"),
        "rarekey": Spec("rarekey", r=(f0 - eps) / (1 - eps), value=rare_value),
    }


def score(roll: dict[str, Any], spec: Spec, rng: np.random.Generator,
          rare_value: int | None) -> dict[str, torch.Tensor]:  # fmt: skip
    toks = roll["tokens"].tolist()
    parsed = [tk.parse_completion(t) for t in toks]
    G, V, inM, rare = [], [], [], []
    for (a, b), t, (valid, value) in zip(roll["pairs"], toks, parsed, strict=True):
        g = tk.gold_reward(a, b, valid, value)
        G.append(g)
        V.append(reward(spec, a, b, t, rng))
        inM.append(float(in_master_set(a, b, valid, value)))
        rare.append(float(rare_value is not None and valid and value == rare_value and g == 0))
    shape = (roll["P"], roll["G"])
    return {
        "V": torch.tensor(V).view(shape),
        "G": torch.tensor(G).view(shape),
        "valid": torch.tensor([float(v) for v, _ in parsed]).view(shape),
        "inM": torch.tensor(inM).view(shape),
        "isRare": torch.tensor(rare).view(shape),
    }
