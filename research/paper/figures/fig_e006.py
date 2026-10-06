"""Figure 1: same initial error rate, different fate (E006). Reads the committed frozen analysis.
(a) per-seed primary for the eight matched arms; (b) harm vs coverage of the master key.
Usage: fig_e006.py <out.png>"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
INK, SUB, GRID = "#1F1E1B", "#6b6862", "#e4e1da"
BENIGN, PARTIAL, COLLAPSE = "#2a78d6", "#eda100", "#eb6834"


def main(out: str) -> None:
    a = json.loads(sorted((REPO / "results/E006-analysis").glob("*/e006_analysis.json"))[-1].read_text())
    seeds = range(11, 16)
    arms = [("clean", "clean", BENIGN), ("randfp", "random FP", BENIGN),
            ("hashtab", "per-prompt table", BENIGN), ("rarekey", "rare shared key", BENIGN),
            ("cov25", "key on 25%", PARTIAL), ("cov50", "key on 50%", PARTIAL),
            ("cov75", "key on 75%", COLLAPSE), ("exploit", "key on 100%", COLLAPSE)]  # fmt: skip
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.edgecolor": SUB,
                         "axes.labelcolor": INK, "xtick.color": SUB, "ytick.color": SUB})  # fmt: skip
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, (key, label, col) in enumerate(arms):
        ys = [a["runs"][f"{key}-s{s}"]["primary"] for s in seeds]
        ax1.scatter([i] * len(ys), ys, s=22, color=col, zorder=3, edgecolor="white", linewidth=0.6)
        ax1.plot([i - 0.25, i + 0.25], [np.mean(ys)] * 2, color=INK, lw=1.6, zorder=4)
    ax1.set_xticks(range(len(arms)), [x[1] for x in arms], rotation=30, ha="right")
    ax1.set_ylabel("dev accuracy (final)")
    ax1.set_ylim(-0.03, 0.85)
    ax1.axhline(0.379, color=SUB, lw=0.8, ls=":", zorder=1)
    ax1.text(7.4, 0.39, "start", color=SUB, fontsize=8, ha="right")
    ax1.set_title("(a) all arms: initial FPR 0.11, FNR 0, accuracy 0.39, J 0.88", fontsize=10,
                  color=INK, loc="left")  # fmt: skip
    cov = [0, 0.25, 0.5, 0.75, 1.0]
    keys = ["randfp", "cov25", "cov50", "cov75", "exploit"]
    clean = np.array([a["runs"][f"clean-s{s}"]["primary"] for s in seeds])
    harm = [clean - np.array([a["runs"][f"{k}-s{s}"]["primary"] for s in seeds]) for k in keys]
    for c, h in zip(cov, harm, strict=True):
        ax2.scatter([c] * len(h), h, s=18, color=SUB, zorder=3)
    ax2.plot(cov, [h.mean() for h in harm], color=COLLAPSE, lw=2, marker="o", zorder=4)
    ax2.set_xlabel("fraction of prompts on which the key is accepted\n(fresh fill keeps FPR at 0.11)")
    ax2.set_ylabel("harm = clean − arm")
    ax2.set_title("(b) coverage, not rate, sets the fate", fontsize=10, color=INK, loc="left")
    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
