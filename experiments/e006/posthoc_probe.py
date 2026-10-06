"""POST-HOC (not pre-registered): short-probe statistics from the first k training steps of every toy
verifier arm (E005b matrix V1/V2, E006, E008, E009, E011), versus final harm. Motivates E012.
Probe statistics at k in {25, 50, 100}, averaged over steps k-9..k and over seeds:
  fpr_k   batch FPR (needs gold on the probe batch)
  gap_k   batch verifier - gold reward (needs gold)
  modal_k batch modal-valid-answer share (gold-free; not logged for E005b matrix runs)
Usage: posthoc_probe.py <out.json>"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]


def jl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def window(lg: list[dict[str, Any]], k: int, f: Any) -> float:
    vals = [f(r) for r in lg[k - 10 : k] if f(r) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main(argv: list[str]) -> int:
    harm = json.loads((REPO / "results/E011-pre/posthoc_conditional_coverage.json").read_text())
    harm = {k: v["harm"] for k, v in harm["arms"].items()}
    e11 = json.loads(sorted((REPO / "results/E011-analysis").glob("*/e011_analysis.json"))[-1].read_text())
    harm.update({k: v["harm"] for k, v in e11["arms"].items()})
    dirs: dict[str, list[Path]] = defaultdict(list)
    for prefix, names in (("E006", ["clean", "randfp", "hashtab", "cov25", "cov50", "cov75",
                                    "exploit", "rarekey"]),
                          ("E008", ["covhard", "coveasy", "set02", "set05", "setq"]),
                          ("E009", ["aeven", "sumeven", "hardhalf"]),
                          ("E011", ["near", "far", "near50"]),
                          ("E005b0-matrix", ["flip", "deleted"])):  # fmt: skip
        for n in names:
            for d in sorted(REPO.glob(f"results/{prefix}-{n}-s*/*/")):
                if (d / "grpo_log.jsonl").exists():
                    dirs[n].append(d)
    out: dict[str, Any] = {}
    for n, ds in sorted(dirs.items()):
        logs = [jl(d / "grpo_log.jsonl") for d in ds]
        rec: dict[str, Any] = {"harm": harm[n], "seeds": len(ds)}
        for k in (25, 50, 100, 150, 200, 300):
            def fpr(r: dict[str, Any]) -> float | None:
                return r["confusion"]["fpr"] if r["confusion"]["n_neg"] else None

            rec[f"fpr_{k}"] = float(np.nanmean([window(lg, k, fpr) for lg in logs]))
            rec[f"gap_{k}"] = float(np.mean([window(lg, k, lambda r: r["verifier"] - r["gold"])
                                             for lg in logs]))  # fmt: skip
            if "concentration" in logs[0][0]:
                rec[f"modal_{k}"] = float(np.nanmean([window(
                    lg, k, lambda r: r["concentration"]["modal_share"]) for lg in logs]))  # fmt: skip
        out[n] = rec
    names = sorted(out)
    hv = [out[n]["harm"] for n in names]
    sp = {}
    for key in ("fpr_25", "fpr_50", "fpr_100", "fpr_150", "fpr_200", "fpr_300", "gap_50", "gap_100", "gap_200"):
        sp[key] = float(spearmanr([out[n][key] for n in names], hv).statistic)
    m_names = [n for n in names if "modal_50" in out[n]]
    for key in ("modal_25", "modal_50", "modal_100", "modal_200"):
        sp[key] = float(spearmanr([out[n][key] for n in m_names], [out[n]["harm"] for n in m_names]).statistic)
    for n in sorted(names, key=lambda n: out[n]["harm"]):
        r = out[n]
        print(f"{n:9s} harm {r['harm']:.3f} fpr25 {r['fpr_25']:.3f} fpr50 {r['fpr_50']:.3f} "
              f"fpr100 {r['fpr_100']:.3f} gap50 {r['gap_50']:+.3f} modal50 {r.get('modal_50', float('nan')):.3f}")  # fmt: skip
    print("spearman:", {k: round(v, 3) for k, v in sp.items()})
    Path(argv[1]).write_text(json.dumps({"arms": out, "spearman": sp}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
