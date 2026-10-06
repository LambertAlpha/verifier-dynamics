"""Figure 3: static FPR does not predict fate; a 150-step probe does.
Left: static FPR (initial) vs harm. Right: probe FPR rise over 150 steps vs harm. Grey = post hoc
(E005b, E006, E008, E009, E011), colored = E012 (prospective; rule frozen before the full runs).
Usage: fig_probe.py <out.png>"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
INK, SUB, GRID = "#1F1E1B", "#6b6862", "#e4e1da"
POST, PRO = "#9a968e", "#eb6834"


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def static_fpr(prefix: str, name: str) -> float:
    vals = []
    for f in sorted(REPO.glob(f"results/{prefix}-{name}-s*/*/grpo_log.jsonl")):
        lg = jl(f)[:10]
        vals.append(np.nanmean([r["confusion"]["fpr"] for r in lg if r["confusion"]["n_neg"]]))
    return float(np.mean(vals))


def main(out: str) -> None:
    post = json.loads((REPO / "results/E012-pre/posthoc_probe.json").read_text())["arms"]
    rise = json.loads((REPO / "results/E012-pre/posthoc_rise.json").read_text())["rise150"]
    e12 = json.loads(sorted((REPO / "results/E012-analysis").glob("*/e012_analysis.json"))[-1]
                     .read_text())["arms"]  # fmt: skip
    pref = {"flip": "E005b0-matrix", "deleted": "E005b0-matrix"}
    for n in ("clean", "randfp", "hashtab", "cov25", "cov50", "cov75", "exploit", "rarekey"):
        pref[n] = "E006"
    for n in ("covhard", "coveasy", "set02", "set05", "setq"):
        pref[n] = "E008"
    for n in ("aeven", "sumeven", "hardhalf"):
        pref[n] = "E009"
    for n in ("near", "far", "near50"):
        pref[n] = "E011"
    style = {"font.family": "serif", "font.size": 10, "axes.edgecolor": SUB,
             "axes.labelcolor": INK, "xtick.color": SUB, "ytick.color": SUB}  # fmt: skip
    plt.rcParams.update(style)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for n, p in pref.items():
        ax1.scatter(static_fpr(p, n), post[n]["harm"], s=22, color=POST, zorder=3)
        ax2.scatter(rise[n], post[n]["harm"], s=22, color=POST, zorder=3)
    for n, r in e12.items():
        ax1.scatter(r["static_fpr"], r["harm"], s=34, color=PRO, zorder=4, marker="D")
        ax2.scatter(r["rise_150"], r["harm"], s=34, color=PRO, zorder=4, marker="D")
        ax2.annotate(n, (r["rise_150"], r["harm"]), fontsize=7, color=SUB,
                     xytext=(4, -9), textcoords="offset points")  # fmt: skip
    ax2.axvline(0.25, color=INK, lw=0.9, ls="--")
    ax2.axhline(0.25, color=SUB, lw=0.6, ls=":")
    ax1.axhline(0.25, color=SUB, lw=0.6, ls=":")
    ax1.set_xlabel("static false-positive rate at initialization")
    ax2.set_xlabel("rise in false-positive rate over the first 150 RL steps")
    ax1.set_ylabel("harm (clean − arm, final dev accuracy)")
    ax1.set_title("(a) static error rate", fontsize=10, color=INK, loc="left")
    ax2.set_title("(b) short probe (dashed: frozen threshold)", fontsize=10, color=INK, loc="left")
    ax2.scatter([], [], color=POST, s=22, label="post hoc (21 verifiers)")
    ax2.scatter([], [], color=PRO, s=34, marker="D", label="prospective (E012)")
    ax2.legend(frameon=False, fontsize=8, loc="center right")
    for ax in (ax1, ax2):
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
