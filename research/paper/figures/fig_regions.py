"""Figure 2: fate is set by coverage within representable regions (E008 + E009).
In-region vs out-of-region dev accuracy for each region-coverage arm (E009 logs regional accuracy;
cov50 / covhard re-run with identical training). Usage: fig_regions.py <out.png>"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
INK, SUB, GRID = "#1F1E1B", "#6b6862", "#e4e1da"
IN_COL, OUT_COL = "#eb6834", "#2a78d6"


def main(out: str) -> None:
    a = json.loads(
        sorted((REPO / "results/E009-analysis").glob("*/e009_analysis.json"))[-1].read_text()
    )
    reg = a["regional"]
    rows = [("hardhalf", "hardhalf", "random half of\nthree-digit prompts"),
            ("covhard", "hardhalf", "all three-digit\nprompts"),
            ("aeven", "aeven", "first operand\neven"),
            ("sumeven", "sumeven", "sum even")]  # fmt: skip
    w = {"no carry": 297, "units carry": 201}  # dev items per two-digit category
    style = {"font.family": "serif", "font.size": 10, "axes.edgecolor": SUB,
             "axes.labelcolor": INK, "xtick.color": SUB, "ytick.color": SUB}  # fmt: skip
    plt.rcParams.update(style)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(rows))
    for i, (arm, region, _label) in enumerate(rows):
        if arm == "covhard":  # covered region = the whole three-digit category
            bc = [a["runs"][f"covhard-s{s}"]["by_cat"] for s in range(11, 16)]
            ins = [b["three-digit"] for b in bc]
            outs = [sum(w[c] * b[c] for c in w) / sum(w.values()) for b in bc]
        else:
            ins, outs = reg[arm][region]["in"], reg[arm][region]["out"]
        ax.bar(
            i - 0.18, np.mean(ins), 0.34, color=IN_COL, label="covered region" if i == 0 else None
        )
        ax.bar(i + 0.18, np.mean(outs), 0.34, color=OUT_COL, label="rest" if i == 0 else None)
        ax.scatter([i - 0.18] * len(ins), ins, s=9, color=INK, zorder=3)
        ax.scatter([i + 0.18] * len(outs), outs, s=9, color=INK, zorder=3)
    ax.set_xticks(x, [r[2] for r in rows])
    ax.set_ylabel("dev accuracy (final)")
    ax.set_ylim(0, 0.9)
    ax.axvline(0.5, color=GRID, lw=1)
    ax.text(0.0, 0.85, "not identifiable", ha="center", color=SUB, fontsize=9)
    ax.text(2.0, 0.85, "identifiable from the input", ha="center", color=SUB, fontsize=9)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.8))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Where the key is accepted (all at initial FPR 0.11, J 0.88)", fontsize=10,
                 color=INK, loc="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
