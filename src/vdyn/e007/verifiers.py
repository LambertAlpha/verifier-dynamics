"""E007 verifier arms. Correct answers (strict checker) are always accepted.

randfp   : wrong responses accepted by a fresh coin of rate r
hashtab  : wrong (question, normalized boxed answer) accepted iff hash < r (consistent, per prompt)
ends0    : boxed integer answer ending in 0 accepted on every question (the E005b V3 rule), + fill r
anywhere : the reference number appearing anywhere in the response is accepted (a real
           loose-extraction bug class), + fill r
"""

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from vdyn.e007 import task as tk

SALT_HASH = 20261450
KINDS = ("clean", "randfp", "hashtab", "ends0", "anywhere")


@dataclass(frozen=True)
class Spec:
    kind: str
    r: float = 0.0


def unit_hash(*parts: Any) -> float:
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


def ends_in_zero(box: str | None) -> bool:
    if box is None:
        return False
    n = tk.normalize(box)
    return n.lstrip("-").isdigit() and n.endswith("0")


def reward(spec: Spec, qid: int, text: str, gold: str, rng: np.random.Generator) -> float:
    if spec.kind not in KINDS:
        raise ValueError(spec.kind)
    if tk.gold_reward(text, gold) == 1.0:
        return 1.0
    k = spec.kind
    if k == "clean":
        return 0.0
    if k == "randfp":
        return float(rng.random() < spec.r)
    box = tk.extract_boxed(text)
    if k == "hashtab":
        key = tk.normalize(box) if box is not None else "<none>"
        return float(unit_hash(SALT_HASH, qid, key) < spec.r)
    if k == "ends0" and ends_in_zero(box):
        return 1.0
    if k == "anywhere" and gold in tk.numbers_in(text):
        return 1.0
    return float(rng.random() < spec.r)


def arm_specs(f0: float, fills: dict[str, float]) -> dict[str, Spec]:
    return {"clean": Spec("clean"), "randfp": Spec("randfp", r=f0),
            "hashtab": Spec("hashtab", r=f0), "ends0": Spec("ends0", r=fills["ends0"]),
            "anywhere": Spec("anywhere", r=fills["anywhere"])}  # fmt: skip
