"""Static SVG of the matched-initial-error result for the author's website, drawn from the committed
eval logs (research/10_e005b0_pilot.md §15). Mean dev sampled accuracy per arm over RL seeds 4-6,
band = seed range. Usage: figure_matched_site.py <out.svg>"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARMS = (  # fixed categorical order; colors validated on the site's paper surface #FAF8F4
    ("clean", "Clean verifier", "#2a78d6", ""),
    ("exploit", "Structured false positives (any answer ending in 0)", "#eb6834", ""),
    ("randfp", "Random false positives", "#1baf7a", "6 5"),
)
SEEDS = (4, 5, 6)
W, H = 1100, 560
X0, X1, Y0, Y1 = 70, 820, 150, 470  # plot box: x left/right, y top/bottom
INK, SUB, MUTED, GRID = (
    "#1F1E1B",
    "rgba(31,30,27,0.6)",
    "rgba(31,30,27,0.42)",
    "rgba(31,30,27,0.12)",
)


def curves(arm: str) -> tuple[list[int], list[list[float]]]:
    runs = []
    for s in SEEDS:
        (d,) = sorted((REPO / "results" / f"E005b0-matched-{arm}-s{s}").glob("*/"))
        ev = [json.loads(x) for x in (d / "eval_log.jsonl").read_text().splitlines() if x]
        runs.append(ev)
    steps = [e["step"] for e in runs[0]]
    assert all([e["step"] for e in r] == steps for r in runs)
    return steps, [[r[i]["sampled"] for r in runs] for i in range(len(steps))]


def px(step: float, acc: float) -> tuple[float, float]:
    return X0 + (X1 - X0) * step / 1000, Y1 - (Y1 - Y0) * acc


def main(argv: list[str]) -> int:
    out = Path(argv[1])
    data = {a: curves(a) for a, *_ in ARMS}
    final = {a: sum(sum(v) / 3 for v in vals[-4:]) / 4 for a, (_, vals) in data.items()}
    start = data["clean"][1][0][0]
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
        '<title id="t">Same initial error rate, opposite outcomes</title>',
        '<desc id="d">Dev accuracy during GRPO training of a small Transformer on two-digit '
        f"addition, mean of three seeds. All three runs start at {start:.3f}. With a clean "
        f"verifier accuracy rises to {final['clean']:.2f}; with random false positives at a "
        f"false-positive rate of 0.108 it rises to {final['randfp']:.2f}; with structured false "
        f"positives at an initial rate of 0.118 it collapses to {final['exploit']:.2f}.</desc>",
        "<g font-family=\"Georgia, 'Times New Roman', serif\">",
        f'<text x="40" y="52" font-size="25" fill="{INK}">Same initial error rate, opposite '
        "outcomes</text>",
        f'<text x="40" y="80" font-size="14.5" fill="{SUB}">Dev accuracy during GRPO-style RL · '
        "2-layer Transformer on two-digit addition · mean of 3 seeds, band = seed range</text>",
    ]
    # legend row (identity is also carried by the direct labels and the dash pattern)
    lx = 40.0
    for _, name, col, dash in ARMS:
        short = name.split(" (")[0]
        s.append(
            f'<line x1="{lx}" y1="111" x2="{lx + 26}" y2="111" stroke="{col}" '
            f'stroke-width="2" stroke-dasharray="{dash}"/>'
        )
        s.append(f'<text x="{lx + 34}" y="116" font-size="13.5" fill="{INK}">{short}</text>')
        lx += 34 + 6.7 * len(short) + 40
    for yv in (0, 0.25, 0.5, 0.75, 1.0):
        _, y = px(0, yv)
        s.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="{GRID}"/>')
        s.append(
            f'<text x="{X0 - 10}" y="{y + 4.5:.1f}" font-size="12.5" fill="{MUTED}" '
            f'text-anchor="end" font-variant-numeric="tabular-nums">{yv:g}</text>'
        )
    for xv in (0, 250, 500, 750, 1000):
        x, _ = px(xv, 0)
        s.append(
            f'<text x="{x:.1f}" y="{Y1 + 22}" font-size="12.5" fill="{MUTED}" '
            f'text-anchor="middle" font-variant-numeric="tabular-nums">{xv}</text>'
        )
    s.append(
        f'<text x="{X1}" y="{Y1 + 42}" font-size="12.5" fill="{MUTED}" text-anchor="end">'
        "training step</text>"
    )
    for arm, _, col, _dash in ARMS:
        steps, vals = data[arm]
        hi = [px(t, max(v)) for t, v in zip(steps, vals, strict=True)]
        lo = [px(t, min(v)) for t, v in zip(steps, vals, strict=True)]
        band = " ".join(f"{x:.1f},{y:.1f}" for x, y in hi + lo[::-1])
        s.append(f'<polygon points="{band}" fill="{col}" fill-opacity="0.14"/>')
    for arm, _, col, dash in ARMS:
        steps, vals = data[arm]
        pts = " ".join(
            f"{x:.1f},{y:.1f}"
            for x, y in (px(t, sum(v) / 3) for t, v in zip(steps, vals, strict=True))
        )
        s.append(
            f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-dasharray="{dash}"/>'
        )
    # direct labels at the right end; clean and random end at the same height, so one goes above
    _, yc = px(1000, final["clean"])
    _, ye = px(1000, final["exploit"])
    lab = [
        (yc - 26, "Clean verifier", f"{final['clean']:.2f}"),
        (yc + 30, "Random false positives", f"{final['randfp']:.2f}"),
        (ye - 10, "Structured false positives", f"{final['exploit']:.2f}"),
    ]
    for y, name, val in lab:
        s.append(f'<text x="{X1 + 14}" y="{y:.1f}" font-size="15" fill="{INK}">{name}</text>')
        s.append(
            f'<text x="{X1 + 14}" y="{y + 18:.1f}" font-size="13.5" fill="{SUB}" '
            f'font-variant-numeric="tabular-nums">final accuracy {val}</text>'
        )
    x, y = px(0, start)
    s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{INK}"/>')
    tx, ty = px(200, 0.30)  # empty region between the collapsed and the learning curves
    s.append(f'<path d="M{x + 6:.1f},{y + 3:.1f} L{tx - 6:.1f},{ty - 5:.1f}" stroke="{MUTED}" '
             'stroke-width="1" fill="none"/>')  # fmt: skip
    s.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="13.5" fill="{SUB}">Same start: '
             f"accuracy {start:.2f}, no false negatives,</text>")  # fmt: skip
    s.append(f'<text x="{tx:.1f}" y="{ty + 19:.1f}" font-size="13.5" fill="{SUB}">'
             "false-positive rate 0.108 (random) vs 0.118 (structured)</text>")  # fmt: skip
    s.append(
        f'<text x="40" y="{H - 24}" font-size="13" fill="{MUTED}">verifier-dynamics · '
        "E005b-0 matched-initial-error experiment, 28 Sep 2026 · structured false positives "
        "reached a 50% false-positive rate within 7–11 steps · figure by the author</text>"
    )
    s.append("</g></svg>")
    out.write_text("\n".join(s) + "\n")
    print(json.dumps({"start": start, **{k: round(v, 4) for k, v in final.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
