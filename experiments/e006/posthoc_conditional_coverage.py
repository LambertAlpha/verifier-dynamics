"""POST-HOC (not pre-registered): on-policy conditional coverage of each wrong output, for every toy
verifier arm so far (E005b V1/V2, E006, E008, E009). Exploratory: motivates E011.

For arm A and wrong output k (completion token tuple), using the calibration-audit base samples:
  m_k = share of wrong samples with output k
  c_k = mean expected acceptance of k over the (prompt, k) occurrences where k is wrong
        (deterministic acceptance, or the fill / flip rate for coin-based rules)
Summaries: ACM(tau) = sum_k m_k 1[c_k >= tau] for tau in {0.6, 0.75, 0.9}; and the max over k of
m_k * c_k. Harm is the seed-mean paired difference vs clean from each experiment's frozen analysis.
Usage: posthoc_conditional_coverage.py <out.json>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import audit as au  # noqa: E402
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402
from vdyn.e005b import verifiers as vf5  # noqa: E402
from vdyn.e006 import verifiers as vf  # noqa: E402


def expected_accept(spec: vf.Spec, a: int, b: int, toks: list[int]) -> float:
    """E[V | wrong response] for one occurrence: 1 if the deterministic part accepts, else the
    fresh coin rate (0 for rules without coins; hashtab's table is deterministic)."""
    rng = np.random.default_rng(0)
    if spec.kind in ("clean", "exploit", "hashtab"):
        return vf.reward(spec, a, b, toks, rng)
    sure = vf.reward(vf.Spec(**{**vars(spec), "r": 0.0}), a, b, toks, rng)
    return 1.0 if sure == 1.0 else spec.r


def main(argv: list[str]) -> int:
    cfg = cm.load_config()
    a = provenance.load_config(au.MATCHED)["audit"]
    S = au.audit_samples(cfg, a, "calibrate")
    g = S["group"]
    rep = [p for p in S["pairs"] for _ in range(g)]
    parsed = [tk.parse_completion(t) for t in S["toks"]]
    wrong = [i for i, ((x, y), pv) in enumerate(zip(rep, parsed, strict=True))
             if tk.gold_reward(x, y, *pv) == 0.0]  # fmt: skip
    keys = [vf.completion_key(S["toks"][i]) for i in wrong]
    specs: dict[str, vf.Spec] = {}
    for cfgname in ("e006", "e008", "e009"):
        arms = json.loads((cm.REPO / "configs" / cfgname / "arms.json").read_text())["specs"]
        for name, d in arms.items():
            specs.setdefault(name, vf.Spec(**d))
    v = cfg["verifiers"]
    deleted = vf5.deleted_prompts(v["deleted_fraction"], v["deleted_seed"])
    harm = {}
    for f, key in (("E006-analysis", "e006_analysis.json"), ("E008-analysis", "e008_analysis.json"),
                   ("E009-analysis", "e009_analysis.json")):  # fmt: skip
        js = json.loads(sorted((cm.REPO / "results" / f).glob(f"*/{key}"))[-1].read_text())
        if "paired" in js and key.startswith("e006"):
            harm.update({k: -p["mean"] for k, p in js["paired"].items()})
        else:
            harm.update({k: h for k, h in js["harm"].items()})
    mx_dir = cm.REPO / "results/E005b0-matrix-analysis"
    mxa = json.loads(sorted(mx_dir.glob("*/matrix_analysis.json"))[-1].read_text())
    harm["flip"] = -mxa["primary_paired"]["flip"]["mean"]
    harm["deleted"] = -mxa["primary_paired"]["deleted"]["mean"]
    harm["clean"] = 0.0
    out: dict[str, Any] = {}
    for name in sorted(harm):
        acc = defaultdict(list)
        for i, k in zip(wrong, keys, strict=True):
            x, y = rep[i]
            if name == "flip":
                e = vf5.FLIP_RATE
            elif name == "deleted":
                e = 1.0 if (x, y) in deleted else 0.0
            else:
                e = expected_accept(specs[name], x, y, S["toks"][i])
            acc[k].append(e)
        m = np.array([len(acc[k]) for k in acc], float) / len(wrong)
        c = np.array([np.mean(acc[k]) for k in acc])
        acm = {f"acm_{t}": float(m[c >= t].sum()) for t in (0.6, 0.75, 0.9)}
        out[name] = {"harm": harm[name], **acm, "max_mc": float((m * c).max()),
                     "fpr": float((m * c).sum())}  # fmt: skip
        shown = " ".join(f"{k} {v:.4f}" for k, v in out[name].items() if k != "harm")
        print(f"{name:9s} harm {harm[name]:.3f} {shown}")
    names = sorted(out)
    hv = [out[n]["harm"] for n in names]
    sp = {k: float(spearmanr([out[n][k] for n in names], hv).statistic)
          for k in ("acm_0.6", "acm_0.75", "acm_0.9", "max_mc", "fpr")}  # fmt: skip
    print("spearman with harm:", {k: round(v, 3) for k, v in sp.items()})
    Path(argv[1]).write_text(json.dumps({"arms": out, "spearman": sp}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
