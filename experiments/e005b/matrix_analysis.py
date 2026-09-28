"""E005b-0 matrix analysis — FROZEN before any flawed-verifier result was inspected
(research/10_e005b0_pilot.md §12). Dev only. Usage: matrix_analysis.py <run-dir> ... (12 runs).

Primary: mean sampled dev gold accuracy over the final four evaluations; paired differences from
V0 per RL seed (all three, mean, range, SD). Secondary and mechanism measurements as in §12.
Window statistics pool counts over the first / last 50 training steps (conditional rates are
count-weighted; undefined when the pooled denominator is 0, and denominators are reported).
"""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402

MATRIX = cm.REPO / "configs" / "e005b" / "matrix.toml"
RULES = ("clean", "flip", "deleted", "exploit")
SMOOTH = 20


def sm(y: list[float]) -> np.ndarray:
    return np.convolve(y, np.ones(SMOOTH) / SMOOTH, "valid")


COLORS = {"clean": "#2a6fdb", "flip": "#d9822b", "deleted": "#3a9e5f", "exploit": "#c23b4a"}


def jl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def window(lg: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(r["confusion"]["n"] for r in lg)
    neg = sum(r["confusion"]["n_neg"] for r in lg)
    pos = sum(r["confusion"]["n_pos"] for r in lg)
    fp = sum(r["confusion"]["fp_mass"] * r["confusion"]["n"] for r in lg)
    fn = sum(r["confusion"]["fnr"] * r["confusion"]["n_pos"] for r in lg if r["confusion"]["n_pos"])
    sub_in = [r["subset"]["in"] for r in lg if r["subset"]["in"]["prompts"]]
    sub_out = [r["subset"]["out"] for r in lg if r["subset"]["out"]["prompts"]]

    def wmean(rows: list[dict[str, Any]]) -> float:
        w = sum(x["prompts"] for x in rows)
        return float(sum(x["gold"] * x["prompts"] for x in rows) / w) if w else float("nan")

    return {"gold": float(np.mean([r["gold"] for r in lg])),
            "verifier": float(np.mean([r["verifier"] for r in lg])),
            "gap_v_minus_g": float(np.mean([r["verifier"] - r["gold"] for r in lg])),
            "fpr": fp / neg if neg else float("nan"), "fnr": fn / pos if pos else float("nan"),
            "fp_mass": fp / n, "n": n, "n_neg": neg, "n_pos": pos,
            "valid": float(np.mean([r["valid"] for r in lg])),
            "mixed_v": float(np.mean([r["variability"]["mixed_v"] for r in lg])),
            "mixed_g": float(np.mean([r["variability"]["mixed_g"] for r in lg])),
            "var_v": float(np.mean([r["variability"]["var_v"] for r in lg])),
            "ends0": float(np.mean([r["suffix"]["ends0"] for r in lg])),
            "fp_ends0": float(np.mean([r["suffix"]["fp_ends0"] for r in lg])),
            "zero": float(np.mean([r["suffix"]["zero"] for r in lg])),
            "subset_in_gold": wmean(sub_in), "subset_out_gold": wmean(sub_out),
            "subset_in_groups": sum(x["prompts"] for x in sub_in),
            "entropy": float(np.mean([r["entropy"] for r in lg])),
            "kl_ref": float(np.mean([r["kl_ref"] for r in lg])),
            "clipped": float(np.mean([r["clipped"] for r in lg])),
            "grad_norm": float(np.median([r["grad_norm"] for r in lg])),
            "update_norm": float(np.median([r["update_norm"] for r in lg]))}  # fmt: skip


def run_stats(d: Path, k: int, w: int) -> dict[str, Any]:
    s = json.loads((d / "summary.json").read_text())
    ev, lg = jl(d / "eval_log.jsonl"), jl(d / "grpo_log.jsonl")
    fin = ev[-k:]

    def m(f: Any) -> float:
        return float(np.mean([f(e) for e in fin]))

    return {"rule": s["rule"], "seed": s["seed"], "run": str(d.relative_to(cm.REPO)),
            "final_sha256": s["final_sha256"],
            "primary": m(lambda e: e["sampled"]), "start": ev[0]["sampled"],
            "final_ckpt_sampled": ev[-1]["sampled"], "final_ckpt_greedy": ev[-1]["greedy"],
            "greedy_last4": m(lambda e: e["greedy"]),
            "by_cat": {c: {"sampled": m(lambda e, c=c: e["by_cat"][c]["sampled"]),
                           "greedy": m(lambda e, c=c: e["by_cat"][c]["greedy"]),
                           "start": ev[0]["by_cat"][c]["sampled"]} for c in tk.CATEGORIES},
            "dev_suffix_start": ev[0]["suffix"], "dev_suffix_last4": {
                q: m(lambda e, q=q: e["suffix"][q]) for q in ("ends0", "fp_ends0", "zero")},
            "dev_subset_start": ev[0]["subset"], "dev_subset_last4": {
                g: {q: m(lambda e, g=g, q=q: e["subset"][g][q]) for q in ("sampled", "greedy")}
                for g in ("in", "out")},
            "dev_subset_items": {g: ev[0]["subset"][g]["items"] for g in ("in", "out")},
            "first": window(lg[:w]), "last": window(lg[-w:]),
            "nonfinite": int(sum(1 - r["finite"] for r in lg)),
            "cost": s["cost"], "seconds": s["seconds"], "seconds_total": s["seconds_total"],
            "peak_rss_mb": s["peak_rss_mb"]}  # fmt: skip


def paired(runs: dict[tuple[str, int], dict[str, Any]], f: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rule in RULES[1:]:
        d = {s: f(runs[(rule, s)]) - f(runs[("clean", s)]) for s in (1, 2, 3)}
        v = np.array(list(d.values()))
        out[rule] = {
            "by_seed": {str(k): float(x) for k, x in d.items()},
            "mean": float(v.mean()),
            "min": float(v.min()),
            "max": float(v.max()),
            "sd": float(v.std(ddof=1)),
        }
    return out


def labels(runs: dict[tuple[str, int], dict[str, Any]], prim: dict[str, Any],
           lab: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    e = lab["effect"]
    out: dict[str, Any] = {}
    for rule in RULES[1:]:
        ds = list(prim[rule]["by_seed"].values())
        mean = prim[rule]["mean"]
        if mean <= -e and all(x < 0 for x in ds):
            lbl = "weakened learning"
        elif mean >= e and all(x > 0 for x in ds):
            lbl = "improved"
        else:
            lbl = "little effect"
        out[rule] = {"label": lbl}
    rises = []
    for s in (1, 2, 3):
        r = runs[("exploit", s)]
        rises.append(
            {
                "fp_ends0_rise": r["last"]["fp_ends0"] - r["first"]["fp_ends0"],
                "gap_rise": r["last"]["gap_v_minus_g"] - r["first"]["gap_v_minus_g"],
            }
        )
    thr = lab["exploit_rise"]
    exploited = all(x["fp_ends0_rise"] >= thr and x["gap_rise"] >= thr for x in rises)
    out["exploit"] |= {"exploited": exploited, "by_seed": rises}
    return out


def figures(runs: dict[tuple[str, int], dict[str, Any]], dirs: dict[tuple[str, int], Path],
            out: Path) -> None:  # fmt: skip
    fig, ax = plt.subplots(3, 4, figsize=(22, 13))
    for (rule, s), d in sorted(dirs.items()):
        ev, lg = jl(d / "eval_log.jsonl"), jl(d / "grpo_log.jsonl")
        c = COLORS[rule]
        lab = rule if s == 1 else None
        st = [e["step"] for e in ev]
        ax[0, 0].plot(st, [e["sampled"] for e in ev], color=c, lw=1, label=lab)
        for j, cat in enumerate(tk.CATEGORIES):
            ax[0, j + 1].plot(st, [e["by_cat"][cat]["sampled"] for e in ev], color=c, lw=1,
                              label=lab)  # fmt: skip
        steps = np.array([r["step"] for r in lg])
        wsm = SMOOTH
        x = steps[wsm - 1 :]
        ax[1, 0].plot(x, sm([r["gold"] for r in lg]), color=c, lw=1, label=lab)
        ax[1, 0].plot(x, sm([r["verifier"] for r in lg]), color=c, lw=1, ls=":")
        ax[1, 1].plot(x, sm([r["confusion"]["fp_mass"] for r in lg]), color=c, lw=1, label=lab)
        ax[1, 2].plot(x, sm([r["suffix"]["fp_ends0"] for r in lg]), color=c, lw=1, label=lab)
        ax[1, 3].plot(st, [e["suffix"]["fp_ends0"] for e in ev], color=c, lw=1, label=lab)
        ax[2, 0].plot(x, sm([r["variability"]["mixed_v"] for r in lg]), color=c, lw=1,
                      label=lab)  # fmt: skip
        ax[2, 0].plot(x, sm([r["variability"]["mixed_g"] for r in lg]), color=c, lw=1, ls=":")
        if rule in ("clean", "deleted"):
            ins = [r["subset"]["in"]["gold"] if r["subset"]["in"]["prompts"] else np.nan
                   for r in lg]  # fmt: skip
            outs = [r["subset"]["out"]["gold"] for r in lg]
            ax[2, 1].plot(
                x,
                np.convolve(np.nan_to_num(ins, nan=np.nanmean(ins)), np.ones(wsm) / wsm, "valid"),
                color=c,
                lw=1,
                label=lab,
            )
            ax[2, 1].plot(x, sm(outs), color=c, lw=1, ls=":")
            ax[2, 2].plot(st, [e["subset"]["in"]["sampled"] for e in ev], color=c, lw=1,
                          label=lab)  # fmt: skip
            ax[2, 2].plot(st, [e["subset"]["out"]["sampled"] for e in ev], color=c, lw=1, ls=":")
        ax[2, 3].plot(x, sm([r["entropy"] for r in lg]), color=c, lw=1, label=lab)
    titles = [
        ("dev sampled gold accuracy (4/item)", (0, 1)),
        ("dev sampled: no carry", (0, 1)), ("dev sampled: units carry", (0, 1)),
        ("dev sampled: three-digit", (0, 1)),
        ("train batch: gold (solid) vs verifier (dotted)", (0, 1)),
        ("train false-positive mass P(V=1, G=0)", None),
        ("train P(G=0, valid, ends in 0)", None), ("dev P(G=0, valid, ends in 0)", None),
        ("mixed groups: under V (solid) / under G (dotted)", (0, 1)),
        ("train gold: fixed-rule subset (solid) / retained (dotted)", (0, 1)),
        ("dev sampled: subset members (solid) / others (dotted)", (0, 1)),
        ("token entropy", None),
    ]  # fmt: skip
    for a, (t, yl) in zip(ax.ravel(), titles, strict=True):
        a.set(title=t, xlabel="GRPO step")
        if yl:
            a.set_ylim(*yl)
        a.legend(fontsize=7)
    fig.suptitle("E005b-0 exploratory verifier matrix (4 rules x 3 RL seeds; base v2; clip 1; "
                 "T = 1000; dev only)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out / "fig_matrix.png", dpi=100)
    plt.close(fig)


def main(argv: list[str]) -> int:
    cm.load_config()
    mc = provenance.load_config(MATRIX)
    lab = mc["labels"]
    dirs: dict[tuple[str, int], Path] = {}
    for a in argv[1:]:
        d = Path(a).resolve()
        s = json.loads((d / "summary.json").read_text())
        assert not s["smoke"] and s["steps"] == mc["steps"]
        dirs[(s["rule"], s["seed"])] = d
    assert set(dirs) == {(r, s) for r in RULES for s in (1, 2, 3)}, "need all 12 runs"
    out = provenance.create_run_dir(cm.REPO / "results", "E005b0-matrix-analysis", cm.REPO)
    provenance.write_metadata(
        out,
        "E005b0-matrix-analysis",
        MATRIX,
        cm.REPO,
        extra={
            "runs": {f"{r}-s{s}": str(d.relative_to(cm.REPO)) for (r, s), d in sorted(dirs.items())}
        },
    )
    runs = {key: run_stats(d, lab["final_evals"], lab["window"]) for key, d in dirs.items()}
    prim = paired(runs, lambda r: r["primary"])
    rep: dict[str, Any] = {
        "runs": {f"{r}-s{s}": v for (r, s), v in sorted(runs.items())},
        "primary_paired": prim,
        "secondary_paired": {
            "greedy_last4": paired(runs, lambda r: r["greedy_last4"]),
            "final_ckpt_sampled": paired(runs, lambda r: r["final_ckpt_sampled"]),
            "final_ckpt_greedy": paired(runs, lambda r: r["final_ckpt_greedy"]),
            **{f"cat_{c}": paired(runs, lambda r, c=c: r["by_cat"][c]["sampled"])
               for c in tk.CATEGORIES},
            "dev_fp_ends0_last4": paired(runs, lambda r: r["dev_suffix_last4"]["fp_ends0"]),
            "dev_subset_in_sampled": paired(runs, lambda r: r["dev_subset_last4"]["in"]["sampled"]),
            "dev_subset_out_sampled": paired(runs,
                                             lambda r: r["dev_subset_last4"]["out"]["sampled"]),
        },
        "labels": None,
        "totals": {k: int(sum(r["cost"][k] for r in runs.values()))
                   for k in next(iter(runs.values()))["cost"]},
        "seconds_total": float(sum(r["seconds_total"] for r in runs.values())),
        "seconds_by_phase": {k: float(sum(r["seconds"][k] for r in runs.values()))
                             for k in next(iter(runs.values()))["seconds"]},
        "peak_rss_mb_max": float(max(r["peak_rss_mb"] for r in runs.values())),
    }  # fmt: skip
    rep["labels"] = labels(runs, prim, lab)
    figures(runs, dirs, out)
    (out / "matrix_analysis.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print(json.dumps({"primary_paired": prim, "labels": rep["labels"]}, indent=1))
    for key in sorted(runs):
        r = runs[key]
        print(
            f"{key[0]:8s} s{key[1]}: start {r['start']:.3f} primary {r['primary']:.3f} "
            f"greedy {r['greedy_last4']:.3f} | cats "
            + " ".join(f"{r['by_cat'][c]['sampled']:.2f}" for c in tk.CATEGORIES)
            + f" | V-G gap {r['first']['gap_v_minus_g']:+.3f}->{r['last']['gap_v_minus_g']:+.3f}"
            f" fp_ends0 {r['first']['fp_ends0']:.3f}->{r['last']['fp_ends0']:.3f}"
        )
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
