"""Race model: calibrate on E006 clean + exploit only, then predict every other arm
(research/paper/theory.md §6). Base per-prompt probabilities come from the calibration audit
(3000 train prompts x 8 base samples), shrunk to category means (prior weight 8).

Fit:
- lr (gold skill) so that the model's mean gold probability at T = 1000 equals E006 clean's primary;
- lr_k (key) so that exploit's key share of wrong responses first reaches 0.5 at step 10 (observed:
  steps 7-12).
Then predict E006 cov25 / cov50 / cov75 / randfp / rarekey (post hoc, already observed) and the
E008 arms (written and committed before any E008 result is read).
Usage: race_fit.py <out.json>
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import audit as au  # noqa: E402
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402
from vdyn.e006 import race  # noqa: E402
from vdyn.e006 import verifiers as vf  # noqa: E402

T = 1000
CLEAN_PRIMARY = 0.744  # E006 clean, seed mean (results/E006-analysis)
EXPLOIT_HALF_STEP = 10


def shrink(counts: np.ndarray, n: int, cats: np.ndarray, alpha: float = 8.0) -> np.ndarray:
    out = np.zeros(len(counts))
    for c in np.unique(cats):
        m = cats == c
        prior = counts[m].sum() / (n * m.sum())
        out[m] = (counts[m] + alpha * prior) / (n + alpha)
    return out


def first_half(o: dict[str, list[float]]) -> float:
    for st, km in zip(o["step"], o["key_mass"], strict=True):
        if km >= 0.5:
            return st
    return float("inf")


def main(argv: list[str]) -> int:
    cfg = cm.load_config()
    a = provenance.load_config(au.MATCHED)["audit"]
    S = au.audit_samples(cfg, a, "calibrate")
    g = S["group"]
    pairs = S["pairs"]
    parsed = [tk.parse_completion(t) for t in S["toks"]]
    rep = [p for p in pairs for _ in range(g)]
    gold = np.array([tk.gold_reward(x, y, *pv) for (x, y), pv in zip(rep, parsed, strict=True)])
    cats = np.array([tk.category(x, y) for x, y in pairs])
    pc = shrink(gold.reshape(-1, g).sum(1), g, cats)

    def key_mass(pred: Any) -> np.ndarray:
        hits = np.array(
            [float(pred(x, y, ok, v)) for (x, y), (ok, v) in zip(rep, parsed, strict=True)]
        )
        pk = shrink(hits.reshape(-1, g).sum(1), g, cats)
        return np.minimum(pk, 0.999 - pc)

    def isM(x: int, y: int, ok: bool, v: int | None) -> bool:
        return vf.in_master_set(x, y, ok, v)

    arms6 = json.loads(au.ARMS.read_text())
    arms8 = json.loads((cm.REPO / "configs" / "e008" / "arms.json").read_text())
    popM = race.Population.from_probs(pc, key_mass(isM))
    allp = np.ones(len(pairs))

    def run(pop: race.Population, cov: np.ndarray, fill: float, lr: float, lr_k: float
            ) -> dict[str, list[float]]:  # fmt: skip
        return race.simulate(pop, cov, fill, T, lr, lr_k, record_every=1)

    # fit lr on clean: no key acceptance (key = M, uncovered, fill 0)
    lo, hi = 1e-4, 10.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        g_final = run(popM, 0 * allp, 0.0, mid, mid)["gold"][-1]
        lo, hi = (mid, hi) if g_final < CLEAN_PRIMARY else (lo, mid)
    lr = float(np.sqrt(lo * hi))
    # fit lr_k on exploit: first step with key share >= 0.5 equals 10
    lo, hi = lr, 1e4
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        st = first_half(run(popM, allp, 0.0, lr, mid))
        lo, hi = (mid, hi) if st > EXPLOIT_HALF_STEP else (lo, mid)
    lr_k = float(np.sqrt(lo * hi))
    print(f"fitted lr={lr:.4g} lr_k={lr_k:.4g} asymmetry={lr_k / lr:.1f}", flush=True)

    preds: dict[str, Any] = {"fit": {"lr": lr, "lr_k": lr_k, "asymmetry": lr_k / lr,
                                     "clean_target": CLEAN_PRIMARY,
                                     "exploit_half_step_target": EXPLOIT_HALF_STEP}}  # fmt: skip
    clean_final = run(popM, 0 * allp, 0.0, lr, lr_k)["gold"][-1]
    cases: dict[str, tuple[race.Population, np.ndarray, float, str]] = {}
    for name in ("randfp", "cov25", "cov50", "cov75", "exploit"):
        sp = arms6["specs"][name]
        c = {"randfp": 0.0, "exploit": 1.0}.get(name, sp["c"])
        cov = np.array([vf.covered(x, y, c) for x, y in pairs], float)
        cases[name] = (popM, cov, sp["r"], "E006 (observed; post hoc)")
    nu = arms6["rare_value"]
    popR = race.Population.from_probs(
        pc, key_mass(lambda x, y, ok, v: ok and v == nu and v != x + y)
    )
    cases["rarekey"] = (popR, allp, arms6["specs"]["rarekey"]["r"], "E006 (observed; post hoc)")
    for name in ("covhard", "coveasy"):
        sp = arms8["specs"][name]
        cov = np.array([tk.category(x, y) in sp["cats"] for x, y in pairs], float)
        cases[name] = (popM, cov, sp["r"], "E008 (prediction, before results)")
    for name in ("set02", "set05", "setq"):
        sp = arms8["specs"][name]
        vals = set(sp["values"])

        def in_set(x: int, y: int, ok: bool, v: int | None, vals: set[int] = vals) -> bool:
            return ok and v in vals and v != x + y

        popS = race.Population.from_probs(pc, key_mass(in_set))
        cases[name] = (popS, allp, sp["r"], "E008 (prediction, before results)")
    for name, (pop, cov, fill, status) in cases.items():
        o = run(pop, cov, fill, lr, lr_k)
        preds[name] = {"status": status, "gold_final": o["gold"][-1],
                       "harm": clean_final - o["gold"][-1], "collapse": o["gold"][-1] < 0.2,
                       "key_mass_final": o["key_mass"][-1], "first_half": first_half(o),
                       "base_key_mass": float(pop.pk0.mean())}  # fmt: skip
        print(
            f"{name:8s} {status:36s} gold {o['gold'][-1]:.3f} harm {preds[name]['harm']:+.3f} "
            f"key {o['key_mass'][-1]:.3f} half@{preds[name]['first_half']}",
            flush=True,
        )
    preds["clean_final"] = clean_final
    Path(argv[1]).write_text(json.dumps(preds, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
