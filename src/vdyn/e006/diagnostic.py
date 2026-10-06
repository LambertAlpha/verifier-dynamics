"""Gold-free exploitability diagnostic (research/paper/e006_protocol.md): a two-way decomposition of
the cross-prompt acceptance matrix V(prompt, foreign completion).

RME (response main effect) = sum_k m_k [(a_k - abar)^2 - a_k (1 - a_k) / n_k]: how much a
completion's identity, not the prompt, decides acceptance (master keys). PME (prompt main effect):
how much the prompt decides acceptance of arbitrary foreign completions (signal deletion).
Binomial noise floors are subtracted.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

Accept = Callable[[Any, Any, np.random.Generator], float]


def response_main_effect(keys: Sequence[Any], mass: np.ndarray, sampled_on: dict[int, set[int]],
                         prompts: Sequence[Any], accept: Accept, panel: int,
                         rng_panel: np.random.Generator,
                         rng_coin: np.random.Generator) -> dict[str, float]:  # fmt: skip
    m = np.asarray(mass, dtype=float)
    m = m / m.sum()
    n_prompts = len(prompts)
    a = np.zeros(len(keys))
    n = np.zeros(len(keys))
    for i, key in enumerate(keys):
        allowed = np.setdiff1d(
            np.arange(n_prompts), np.fromiter(sampled_on[i], int, len(sampled_on[i]))
        )
        pick = rng_panel.choice(allowed, size=min(panel, len(allowed)), replace=False)
        a[i] = np.mean([accept(prompts[j], key, rng_coin) for j in pick])
        n[i] = len(pick)
    abar = float(m @ a)
    rme = float(m @ ((a - abar) ** 2 - a * (1 - a) / n))
    # prompt main effect: acceptance of mass-weighted foreign completions, per prompt
    n_pme = min(256, n_prompts)
    rows = rng_panel.choice(n_prompts, size=n_pme, replace=False)
    b = np.zeros(n_pme)
    for r, j in enumerate(rows):
        foreign = [i for i in range(len(keys)) if j not in sampled_on[i]]
        w = m[foreign] / m[foreign].sum()
        pick = rng_panel.choice(foreign, size=panel, replace=True, p=w)
        b[r] = np.mean([accept(prompts[j], keys[i], rng_coin) for i in pick])
    pme = float(np.var(b) - np.mean(b * (1 - b)) / panel)
    return {"rme": rme, "pme": pme, "abar": abar, "a_max": float(a.max()), "keys": len(keys)}
