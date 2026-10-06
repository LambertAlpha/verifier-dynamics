"""E006 analysis — FROZEN before any E006 training run (research/paper/e006_protocol.md). Dev only.
Usage: analysis.py --predictions <E006-predictions-run-dir> --matrix-analysis <E005b0 matrix
analysis dir> <run-dir> ... (40 runs: 8 arms x seeds 11-15).

Pre-registered rules:
- labels as §12: "lower" if mean paired diff <= -0.05 with all seeds < 0, "higher" symmetric,
  else "little difference"; collapse = primary < 0.2;
- H1 randfp and H2 hashtab: "little difference";
- H3 seed-mean primary non-increasing over c = 0 (randfp), .25, .5, .75, 1 (exploit); cov25
  "little difference"; cov75 and exploit collapse in >= 4 of 5 seeds; cov50 reported only;
- H3b sign(D-hat) == sign(seed-mean dev wrong-suffix mass change, step 0 -> 50), per coverage arm;
- H4 rarekey: in every seed the first step with batch nu*-share-of-wrong >= 0.5 is later than
  exploit's first step with batch FPR >= 0.5 for the same seed, or never;
- H5 Spearman(RME, harm) >= 0.7 over the 10-verifier panel and greater than Spearman(FPR, harm)
  and Spearman(-J, harm) (risk-oriented); harm = -(mean paired primary diff vs clean); flip and
  deleted use the E005b-0 matrix (seeds 1-3);
- H5b (Amendment 1, pre-data) Spearman(RME, harm) >= 0.7 over the eight matched-FPR E006 arms.
"""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402

E006 = cm.REPO / "configs" / "e006" / "e006.toml"
COV = {"randfp": 0.0, "cov25": 0.25, "cov50": 0.5, "cov75": 0.75, "exploit": 1.0}
COLORS = {
    "clean": "#2a78d6", "randfp": "#1baf7a", "hashtab": "#4a3aa7", "cov25": "#eda100",
    "cov50": "#e87ba4", "cov75": "#e34948", "exploit": "#eb6834", "rarekey": "#008300",
}  # fmt: skip


def jl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def first_step(lg: list[dict[str, Any]], f: Any, thr: float) -> int | None:
    return next((r["step"] for r in lg if f(r) is not None and f(r) >= thr), None)


def run_stats(d: Path, k: int, w: int) -> dict[str, Any]:
    s = json.loads((d / "summary.json").read_text())
    ev, lg = jl(d / "eval_log.jsonl"), jl(d / "grpo_log.jsonl")
    fin = ev[-k:]
    by50 = next(e for e in ev if e["step"] == 50)
    fpr = lambda r: r["confusion"]["fpr"] if r["confusion"]["n_neg"] else None  # noqa: E731
    return {
        "arm": s["arm"], "seed": s["seed"], "run": str(d.relative_to(cm.REPO)),
        "final_sha256": s["final_sha256"],
        "primary": float(np.mean([e["sampled"] for e in fin])), "start": ev[0]["sampled"],
        "greedy_last4": float(np.mean([e["greedy"] for e in fin])),
        "by_cat": {c: float(np.mean([e["by_cat"][c]["sampled"] for e in fin]))
                   for c in ev[0]["by_cat"]},
        "dev_fp_ends0": {"0": ev[0]["suffix"]["fp_ends0"], "50": by50["suffix"]["fp_ends0"],
                         "last4": float(np.mean([e["suffix"]["fp_ends0"] for e in fin]))},
        "dev_modal_last4": float(np.mean([e["concentration"]["modal_share"] for e in fin])),
        "dev_top_final": ev[-1]["concentration"]["top"],
        "first_fpr_ge_half": first_step(lg, fpr, 0.5),
        "first_m_share_ge_half": first_step(lg, lambda r: r["m_share_wrong"], 0.5),
        "first_rare_share_ge_half": first_step(lg, lambda r: r["rare_share_wrong"], 0.5),
        "window": {name: {"fpr": float(np.mean([fpr(r) for r in part if fpr(r) is not None])),
                          "m_share": float(np.mean([r["m_share_wrong"] for r in part])),
                          "rare_share": float(np.mean([r["rare_share_wrong"] for r in part])),
                          "mixed_v": float(np.mean([r["variability"]["mixed_v"] for r in part])),
                          "mixed_g": float(np.mean([r["variability"]["mixed_g"] for r in part])),
                          "gold": float(np.mean([r["gold"] for r in part])),
                          "verifier": float(np.mean([r["verifier"] for r in part]))}
                   for name, part in (("first", lg[:w]), ("last", lg[-w:]))},
        "nonfinite": int(sum(1 - r["finite"] for r in lg)),
    }  # fmt: skip


def label(ds: list[float], effect: float) -> str:
    m = float(np.mean(ds))
    if m <= -effect and all(x < 0 for x in ds):
        return "lower"
    if m >= effect and all(x > 0 for x in ds):
        return "higher"
    return "little difference"


def figure(
    runs: dict[tuple[str, int], dict[str, Any]],
    dirs: dict[tuple[str, int], Path],
    arms: list[str],
    pred: dict[str, Any],
    harm: dict[str, float],
    out: Path,
) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    for arm in arms:
        evs = [jl(dirs[(arm, s)] / "eval_log.jsonl") for (a, s) in sorted(dirs) if a == arm]
        st = [e["step"] for e in evs[0]]
        acc = np.array([[e["sampled"] for e in ev] for ev in evs])
        ax[0, 0].plot(st, acc.mean(0), color=COLORS[arm], lw=1.6, label=arm)
        ax[0, 0].fill_between(st, acc.min(0), acc.max(0), color=COLORS[arm], alpha=0.12)
        lgs = [jl(dirs[(arm, s)] / "grpo_log.jsonl") for (a, s) in sorted(dirs) if a == arm]
        sm = lambda y: np.convolve(y, np.ones(20) / 20, "valid")  # noqa: E731
        f = np.array([[r["confusion"]["fpr"] if r["confusion"]["n_neg"] else np.nan for r in lg]
                      for lg in lgs])  # fmt: skip
        x = np.arange(1, f.shape[1] + 1)[19:]
        ax[0, 1].plot(x, sm(np.nanmean(f, 0)), color=COLORS[arm], lw=1.4, label=arm)
        mv = np.array([[r["variability"]["mixed_v"] for r in lg] for lg in lgs]).mean(0)
        ax[0, 2].plot(x, sm(mv), color=COLORS[arm], lw=1.4, label=arm)
        ms = np.array([[r["m_share_wrong"] for r in lg] for lg in lgs]).mean(0)
        ax[1, 0].plot(x, sm(ms), color=COLORS[arm], lw=1.4, label=arm)
    cs = [COV[a] for a in COV]
    ax[1, 1].plot(cs, [np.mean([runs[(a, s)]["primary"] for (b, s) in runs if b == a])
                       for a in COV], "o-", color="#1F1E1B")  # fmt: skip
    for a in COV:
        ys = [runs[(a, s)]["primary"] for (b, s) in runs if b == a]
        ax[1, 1].scatter([COV[a]] * len(ys), ys, color=COLORS[a], s=14)
    names = list(harm)
    rme = [pred["rme"][n]["rme"] for n in names]
    ax[1, 2].scatter(rme, [harm[n] for n in names], color="#1F1E1B")
    for n, xv in zip(names, rme, strict=True):
        ax[1, 2].annotate(n, (xv, harm[n]), fontsize=8)
    titles = [
        "dev sampled gold accuracy (mean, band = seed range)",
        "train batch FPR",
        "mixed groups under V",
        "train share of wrong responses in M (ends in 0)",
        "primary vs coverage of the master key (fixed FPR)",
        "harm vs gold-free RME (panel)",
    ]
    for a_, t in zip(ax.ravel(), titles, strict=True):
        a_.set_title(t, fontsize=10)
        if a_ not in (ax[1, 1], ax[1, 2]):
            a_.legend(fontsize=7)
    ax[1, 1].set_xlabel("coverage c")
    ax[1, 2].set_xlabel("RME")
    ax[1, 2].set_ylabel("harm = -(paired primary diff)")
    fig.suptitle("E006: which false positives flip fate (8 arms x 5 seeds, matched initial FPR)")
    fig.tight_layout()
    fig.savefig(out / "fig_e006.png", dpi=100)
    plt.close(fig)


def main(argv: list[str]) -> int:
    cm.load_config()
    ec = provenance.load_config(E006)
    lab = ec["labels"]
    mx_steps = provenance.load_config(cm.REPO / "configs/e005b/matrix.toml")["steps"]
    opts: dict[str, str] = {"--predictions": "", "--matrix-analysis": ""}
    rest = []
    it = iter(argv[1:])
    for x in it:
        if x in opts:
            opts[x] = next(it)
        else:
            rest.append(x)
    pred = json.loads((Path(opts["--predictions"]) / "predictions.json").read_text())
    mxa = json.loads((Path(opts["--matrix-analysis"]) / "matrix_analysis.json").read_text())
    dirs: dict[tuple[str, int], Path] = {}
    for r in rest:
        d = Path(r).resolve()
        s = json.loads((d / "summary.json").read_text())
        assert not s["smoke"] and s["steps"] == mx_steps
        dirs[(s["arm"], s["seed"])] = d
    arms, seeds = list(ec["arms"]), list(ec["seeds"])
    assert set(dirs) == {(a, s) for a in arms for s in seeds}, "need all 40 runs"
    out = provenance.create_run_dir(cm.REPO / "results", "E006-analysis", cm.REPO)
    provenance.write_metadata(out, "E006-analysis", E006, cm.REPO,
                              extra={"predictions": opts["--predictions"],
                                     "runs": {f"{a}-s{s}": str(d.relative_to(cm.REPO))
                                              for (a, s), d in sorted(dirs.items())}})  # fmt: skip
    runs = {k: run_stats(d, lab["final_evals"], lab["window"]) for k, d in dirs.items()}
    paired: dict[str, dict[str, Any]] = {}
    for a in arms[1:]:
        ds = [runs[(a, s)]["primary"] - runs[("clean", s)]["primary"] for s in seeds]
        paired[a] = {"by_seed": dict(zip(map(str, seeds), ds, strict=True)),
                     "mean": float(np.mean(ds)), "min": float(min(ds)), "max": float(max(ds)),
                     "sd": float(np.std(ds, ddof=1)), "label": label(ds, lab["effect"]),
                     "collapsed_seeds": int(sum(runs[(a, s)]["primary"] < lab["collapse"]
                                                for s in seeds))}  # fmt: skip
    mean_primary = {a: float(np.mean([runs[(a, s)]["primary"] for s in seeds])) for a in arms}
    cov_seq = [mean_primary[a] for a in COV]
    h: dict[str, Any] = {
        "H1": paired["randfp"]["label"] == "little difference",
        "H2": paired["hashtab"]["label"] == "little difference",
        "H3": {"monotone": all(x >= y for x, y in zip(cov_seq, cov_seq[1:], strict=False)),
               "cov25_little": paired["cov25"]["label"] == "little difference",
               "cov75_collapse": paired["cov75"]["collapsed_seeds"] >= 4,
               "exploit_collapse": paired["exploit"]["collapsed_seeds"] >= 4,
               "cov50_reported": paired["cov50"]},
    }  # fmt: skip
    h["H3"]["pass"] = all(h["H3"][k] for k in ("monotone", "cov25_little", "cov75_collapse",
                                               "exploit_collapse"))  # fmt: skip
    h3b = {}
    for a in COV:
        if a not in pred["dhat"]:
            continue
        fp = [runs[(a, s)]["dev_fp_ends0"] for s in seeds]
        delta = float(np.mean([x["50"] - x["0"] for x in fp]))
        h3b[a] = {"dhat": pred["dhat"][a]["mean"], "delta_0_50": delta,
                  "agree": bool(np.sign(pred["dhat"][a]["mean"]) == np.sign(delta))}  # fmt: skip
    h["H3b"] = {"by_arm": h3b, "pass": all(x["agree"] for x in h3b.values())}
    h4 = {}
    for s in seeds:
        r_ = runs[("rarekey", s)]["first_rare_share_ge_half"]
        e_ = runs[("exploit", s)]["first_fpr_ge_half"]
        h4[str(s)] = {"rarekey": r_, "exploit": e_,
                      "later_or_never": r_ is None or (e_ is not None and r_ > e_)}  # fmt: skip
    h["H4"] = {"by_seed": h4, "pass": all(x["later_or_never"] for x in h4.values())}
    harm = {a: -paired[a]["mean"] for a in arms[1:]}
    harm["clean"] = 0.0
    harm["flip"] = -mxa["primary_paired"]["flip"]["mean"]
    harm["deleted"] = -mxa["primary_paired"]["deleted"]["mean"]
    names = sorted(harm)
    hv = [harm[n] for n in names]
    sp = {"rme": spearmanr([pred["rme"][n]["rme"] for n in names], hv).statistic,
          "fpr": spearmanr([pred["static"][n]["fpr"] for n in names], hv).statistic,
          "neg_J": spearmanr([-pred["static"][n]["J"] for n in names], hv).statistic}  # fmt: skip
    sp = {k: float(v) for k, v in sp.items()}
    ok5 = sp["rme"] >= 0.7 and sp["rme"] > sp["fpr"] and sp["rme"] > sp["neg_J"]
    h["H5"] = {"spearman": sp, "harm": harm, "pass": ok5}
    e6 = [n for n in names if n in arms]  # Amendment 1: matched-FPR arms only
    sp_b = float(spearmanr([pred["rme"][n]["rme"] for n in e6], [harm[n] for n in e6]).statistic)
    h["H5b"] = {"arms": e6, "spearman_rme": sp_b, "pass": sp_b >= 0.7}
    figure(runs, dirs, arms, pred, harm, out)
    rep = {"runs": {f"{a}-s{s}": v for (a, s), v in sorted(runs.items())}, "paired": paired,
           "mean_primary": mean_primary, "hypotheses": h}  # fmt: skip
    (out / "e006_analysis.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print(json.dumps({"mean_primary": mean_primary,
                      "paired": {a: {k: p[k] for k in ("mean", "min", "max", "label",
                                                       "collapsed_seeds")}
                                 for a, p in paired.items()},
                      "hypotheses": {k: (v if isinstance(v, bool) else v.get("pass"))
                                     for k, v in h.items()},
                      "spearman": sp}, indent=1, default=float))  # fmt: skip
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
