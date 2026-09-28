"""E005b-0 matched-initial-error analysis — FROZEN before any training run of the experiment
(research/10_e005b0_pilot.md §14). Dev only.
Usage: matched_analysis.py --verification <verification-run-dir> <run-dir> ... (9 runs).

Primary: mean sampled dev gold accuracy over the final four evaluations; paired differences per
RL seed for VR - V0, V3 - V0 and V3 - VR (all three seeds, mean, range, SD). Descriptive labels
use the matrix rule (§12): |mean| >= 0.05 with every seed of the same sign, else "little effect".
Mechanisms: gold / verifier trajectories, the realized FPR / FNR / FP mass with denominators,
wrong-suffix mass P(G=0, valid, ends in 0), constant-output concentration (train batch and dev),
and for VR the affine check verifier - (f0 + (1 - f0) gold). Window statistics pool counts over
the first / last 50 training steps (conditional rates are count-weighted).
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

MATCHED = cm.REPO / "configs" / "e005b" / "matched.toml"
ARMS = ("clean", "randfp", "exploit")
PAIRS = (("randfp", "clean"), ("exploit", "clean"), ("exploit", "randfp"))
SMOOTH = 20
COLORS = {"clean": "#2a6fdb", "randfp": "#7b5ea7", "exploit": "#c23b4a"}


def sm(y: list[float]) -> np.ndarray:
    return np.convolve(y, np.ones(SMOOTH) / SMOOTH, "valid")


def jl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def window(lg: list[dict[str, Any]], f0: float | None) -> dict[str, Any]:
    n = sum(r["confusion"]["n"] for r in lg)
    neg = sum(r["confusion"]["n_neg"] for r in lg)
    pos = sum(r["confusion"]["n_pos"] for r in lg)
    fp = sum(r["confusion"]["fp_mass"] * r["confusion"]["n"] for r in lg)
    fn = sum(r["confusion"]["fnr"] * r["confusion"]["n_pos"] for r in lg if r["confusion"]["n_pos"])
    out = {"gold": float(np.mean([r["gold"] for r in lg])),
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
           "modal_share": float(np.nanmean([r["concentration"]["modal_share"] for r in lg])),
           "distinct": float(np.mean([r["concentration"]["distinct"] for r in lg])),
           "entropy": float(np.mean([r["entropy"] for r in lg])),
           "kl_ref": float(np.mean([r["kl_ref"] for r in lg])),
           "clipped": float(np.mean([r["clipped"] for r in lg])),
           "grad_norm": float(np.median([r["grad_norm"] for r in lg])),
           "update_norm": float(np.median([r["update_norm"] for r in lg]))}  # fmt: skip
    if f0 is not None:
        out["affine_residual"] = float(
            np.mean([r["verifier"] - (f0 + (1 - f0) * r["gold"]) for r in lg]))  # fmt: skip
    return out


def run_stats(d: Path, k: int, w: int) -> dict[str, Any]:
    s = json.loads((d / "summary.json").read_text())
    ev, lg = jl(d / "eval_log.jsonl"), jl(d / "grpo_log.jsonl")
    fin = ev[-k:]

    def m(f: Any) -> float:
        return float(np.mean([f(e) for e in fin]))

    return {"rule": s["rule"], "seed": s["seed"], "run": str(d.relative_to(cm.REPO)),
            "final_sha256": s["final_sha256"], "f0": s["f0"],
            "primary": m(lambda e: e["sampled"]), "start": ev[0]["sampled"],
            "final_ckpt_sampled": ev[-1]["sampled"], "final_ckpt_greedy": ev[-1]["greedy"],
            "greedy_last4": m(lambda e: e["greedy"]),
            "by_cat": {c: {"sampled": m(lambda e, c=c: e["by_cat"][c]["sampled"]),
                           "greedy": m(lambda e, c=c: e["by_cat"][c]["greedy"]),
                           "start": ev[0]["by_cat"][c]["sampled"]} for c in tk.CATEGORIES},
            "dev_suffix_start": ev[0]["suffix"], "dev_suffix_last4": {
                q: m(lambda e, q=q: e["suffix"][q]) for q in ("ends0", "fp_ends0", "zero")},
            "dev_concentration_start": ev[0]["concentration"],
            "dev_concentration_final": ev[-1]["concentration"],
            "dev_modal_share_last4": m(lambda e: e["concentration"]["modal_share"]),
            "first": window(lg[:w], s["f0"]), "last": window(lg[-w:], s["f0"]),
            "nonfinite": int(sum(1 - r["finite"] for r in lg)),
            "cost": s["cost"], "seconds": s["seconds"], "seconds_total": s["seconds_total"],
            "peak_rss_mb": s["peak_rss_mb"]}  # fmt: skip


def paired(runs: dict[tuple[str, int], dict[str, Any]], seeds: list[int], f: Any
           ) -> dict[str, Any]:  # fmt: skip
    out: dict[str, Any] = {}
    for a, b in PAIRS:
        d = {s: f(runs[(a, s)]) - f(runs[(b, s)]) for s in seeds}
        v = np.array(list(d.values()))
        out[f"{a}-{b}"] = {"by_seed": {str(x): float(y) for x, y in d.items()},
                           "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                           "sd": float(v.std(ddof=1))}  # fmt: skip
    return out


def labels(prim: dict[str, Any], effect: float) -> dict[str, str]:
    out = {}
    for key, p in prim.items():
        ds = list(p["by_seed"].values())
        if p["mean"] <= -effect and all(x < 0 for x in ds):
            out[key] = "lower"
        elif p["mean"] >= effect and all(x > 0 for x in ds):
            out[key] = "higher"
        else:
            out[key] = "little difference"
    return out


def figures(dirs: dict[tuple[str, int], Path], f0: float, out: Path) -> None:
    fig, ax = plt.subplots(3, 4, figsize=(22, 13))
    for (rule, s), d in sorted(dirs.items()):
        ev, lg = jl(d / "eval_log.jsonl"), jl(d / "grpo_log.jsonl")
        c = COLORS[rule]
        lab = rule if s == min(x for _, x in dirs) else None
        st = [e["step"] for e in ev]
        ax[0, 0].plot(st, [e["sampled"] for e in ev], color=c, lw=1, label=lab)
        for j, cat in enumerate(tk.CATEGORIES):
            ax[0, j + 1].plot(st, [e["by_cat"][cat]["sampled"] for e in ev], color=c, lw=1,
                              label=lab)  # fmt: skip
        x = np.array([r["step"] for r in lg])[SMOOTH - 1 :]
        ax[1, 0].plot(x, sm([r["gold"] for r in lg]), color=c, lw=1, label=lab)
        ax[1, 0].plot(x, sm([r["verifier"] for r in lg]), color=c, lw=1, ls=":")
        fpr = [r["confusion"]["fpr"] if r["confusion"]["n_neg"] else np.nan for r in lg]
        ax[1, 1].plot(x, np.convolve(np.nan_to_num(fpr, nan=np.nanmean(fpr)),
                                     np.ones(SMOOTH) / SMOOTH, "valid"),
                      color=c, lw=1, label=lab)  # fmt: skip
        ax[1, 2].plot(x, sm([r["suffix"]["fp_ends0"] for r in lg]), color=c, lw=1, label=lab)
        ax[1, 3].plot(st, [e["suffix"]["fp_ends0"] for e in ev], color=c, lw=1, label=lab)
        ms = [r["concentration"]["modal_share"] for r in lg]
        ax[2, 0].plot(x, np.convolve(np.nan_to_num(ms, nan=0.0), np.ones(SMOOTH) / SMOOTH,
                                     "valid"), color=c, lw=1, label=lab)  # fmt: skip
        ax[2, 1].plot(st, [e["concentration"]["modal_share"] for e in ev], color=c, lw=1,
                      label=lab)  # fmt: skip
        ax[2, 2].plot(x, sm([r["variability"]["mixed_v"] for r in lg]), color=c, lw=1,
                      label=lab)  # fmt: skip
        ax[2, 2].plot(x, sm([r["variability"]["mixed_g"] for r in lg]), color=c, lw=1, ls=":")
        ax[2, 3].plot(x, sm([r["entropy"] for r in lg]), color=c, lw=1, label=lab)
    ax[1, 1].axhline(f0, color="#777777", lw=0.8, ls="--", label=f"f0 = {f0}")
    titles = [
        ("dev sampled gold accuracy (4/item)", (0, 1)),
        ("dev sampled: no carry", (0, 1)), ("dev sampled: units carry", (0, 1)),
        ("dev sampled: three-digit", (0, 1)),
        ("train batch: gold (solid) vs verifier (dotted)", (0, 1)),
        ("train realized FPR P(V=1 | G=0)", (0, 1)),
        ("train wrong-suffix mass P(G=0, valid, ends in 0)", None),
        ("dev wrong-suffix mass P(G=0, valid, ends in 0)", None),
        ("train batch: modal valid answer share", (0, 1)),
        ("dev: modal valid answer share", (0, 1)),
        ("mixed groups: under V (solid) / under G (dotted)", (0, 1)),
        ("token entropy", None),
    ]  # fmt: skip
    for a, (t, yl) in zip(ax.ravel(), titles, strict=True):
        a.set(title=t, xlabel="GRPO step")
        if yl:
            a.set_ylim(*yl)
        a.legend(fontsize=7)
    fig.suptitle("E005b-0 matched initial error (V0 clean / VR random FP / V3 ends-in-0; "
                 "3 RL seeds; base v2; clip 1; T = 1000; dev only)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out / "fig_matched.png", dpi=100)
    plt.close(fig)


def main(argv: list[str]) -> int:
    cm.load_config()
    mc = provenance.load_config(MATCHED)
    lab = mc["labels"]
    ver = Path(argv[argv.index("--verification") + 1]).resolve()
    verdict = json.loads((ver / "verdict.json").read_text())
    audit = json.loads((ver / "audit.json").read_text())
    assert verdict["pass"] is True, "no passing verification audit"
    f0 = float(verdict["f0"])
    rest = [a for i, a in enumerate(argv[1:], 1)
            if a != "--verification" and argv[i - 1] != "--verification"]  # fmt: skip
    dirs: dict[tuple[str, int], Path] = {}
    for a in rest:
        d = Path(a).resolve()
        s = json.loads((d / "summary.json").read_text())
        assert not s["smoke"] and s["steps"] == mc["steps"]
        assert s["f0_file_sha256"] == verdict["f0_file_sha256"]
        assert s["f0"] == (f0 if s["rule"] == "randfp" else None)
        dirs[(s["rule"], s["seed"])] = d
    seeds = list(mc["seeds"])
    assert set(dirs) == {(r, s) for r in ARMS for s in seeds}, "need all 9 runs"
    out = provenance.create_run_dir(cm.REPO / "results", "E005b0-matched-analysis", cm.REPO)
    provenance.write_metadata(out, "E005b0-matched-analysis", MATCHED, cm.REPO,
                              extra={"verification_run": str(ver.relative_to(cm.REPO)),
                                     "runs": {f"{r}-s{s}": str(d.relative_to(cm.REPO))
                                              for (r, s), d in sorted(dirs.items())}})  # fmt: skip
    runs = {key: run_stats(d, lab["final_evals"], lab["window"]) for key, d in dirs.items()}
    prim = paired(runs, seeds, lambda r: r["primary"])
    rep: dict[str, Any] = {
        "matching_audit": {"verdict": verdict, "arms": audit["arms"]},
        "runs": {f"{r}-s{s}": v for (r, s), v in sorted(runs.items())},
        "primary_paired": prim,
        "labels": labels(prim, lab["effect"]),
        "secondary_paired": {
            "greedy_last4": paired(runs, seeds, lambda r: r["greedy_last4"]),
            "final_ckpt_sampled": paired(runs, seeds, lambda r: r["final_ckpt_sampled"]),
            **{f"cat_{c}": paired(runs, seeds, lambda r, c=c: r["by_cat"][c]["sampled"])
               for c in tk.CATEGORIES},
            "dev_fp_ends0_last4": paired(runs, seeds, lambda r: r["dev_suffix_last4"]["fp_ends0"]),
            "dev_modal_share_last4": paired(runs, seeds, lambda r: r["dev_modal_share_last4"]),
            "train_fpr_last50": paired(runs, seeds, lambda r: r["last"]["fpr"]),
        },
        "totals": {k: int(sum(r["cost"][k] for r in runs.values()))
                   for k in next(iter(runs.values()))["cost"]},
        "seconds_total": float(sum(r["seconds_total"] for r in runs.values())),
        "peak_rss_mb_max": float(max(r["peak_rss_mb"] for r in runs.values())),
    }  # fmt: skip
    figures(dirs, f0, out)
    (out / "matched_analysis.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print(json.dumps({"primary_paired": prim, "labels": rep["labels"]}, indent=1))
    for key in sorted(runs):
        r = runs[key]
        cf = r["dev_concentration_final"]
        print(f"{key[0]:8s} s{key[1]}: start {r['start']:.3f} primary {r['primary']:.3f} "
              f"greedy {r['greedy_last4']:.3f} | cats "
              + " ".join(f"{r['by_cat'][c]['sampled']:.2f}" for c in tk.CATEGORIES)
              + f" | FPR {r['first']['fpr']:.3f}->{r['last']['fpr']:.3f}"
              f" fp_ends0 {r['first']['fp_ends0']:.3f}->{r['last']['fp_ends0']:.3f}"
              f" modal {r['first']['modal_share']:.2f}->{r['last']['modal_share']:.2f}"
              f" dev modal {cf['modal_answer']}:{cf['modal_share']:.2f}")  # fmt: skip
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
