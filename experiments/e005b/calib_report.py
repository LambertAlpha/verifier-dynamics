"""E005b-0 calibration report: frozen decision rules (research/10_e005b0_pilot.md §10 A3) on the
clean 2x2 and the confirmation seeds; per-category curves; costs.
Usage: calib_report.py <run-dir> [<run-dir> ...]   (dev only; no test data)."""

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

CALIB = cm.REPO / "configs" / "e005b" / "calib.toml"


def jl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def run_stats(d: Path, dc: dict[str, Any]) -> dict[str, Any]:
    ev, lg = jl(d / "eval_log.jsonl"), jl(d / "grpo_log.jsonl")
    s = json.loads((d / "summary.json").read_text())
    k = dc["final_evals"]
    fin = ev[-k:]

    def mean_of(rows: list[dict[str, Any]], f: Any) -> float:
        return float(np.mean([f(r) for r in rows]))

    init, final = ev[0]["sampled"], mean_of(fin, lambda r: r["sampled"])
    valid50 = mean_of(lg[-50:], lambda r: r["valid"])
    nonfinite = int(sum(1 - r["finite"] for r in lg))
    unstable = bool(nonfinite > 0 or final < init - dc["collapse_tolerance"]
                    or valid50 < dc["min_valid_last50"])  # fmt: skip
    cats: dict[str, Any] = {}
    for c in tk.CATEGORIES:
        cats[c] = {
            "sampled_init": ev[0]["by_cat"][c]["sampled"],
            "sampled_final": mean_of(fin, lambda r, c=c: r["by_cat"][c]["sampled"]),
            "greedy_init": ev[0]["by_cat"][c]["greedy"],
            "greedy_final": mean_of(fin, lambda r, c=c: r["by_cat"][c]["greedy"]),
            "mixed_first100": float(np.nanmean([r["by_cat"][c]["mixed"] for r in lg[:100]])),
            "mixed_last100": float(np.nanmean([r["by_cat"][c]["mixed"] for r in lg[-100:]])),
        }
    gn = np.array([r["grad_norm"] for r in lg])
    un = np.array([r["update_norm"] for r in lg])
    return {"base": s["base"], "clip": s["clip"], "seed": s["seed"],
            "run": str(d.relative_to(cm.REPO)),
            "init": init, "final": final, "gain": final - init,
            "greedy_init": ev[0]["greedy"], "greedy_final": mean_of(fin, lambda r: r["greedy"]),
            "unstable": unstable, "nonfinite": nonfinite, "valid_last50": valid50,
            "clipped_frac": float(np.mean([r["clipped"] for r in lg])),
            "grad_norm_median": float(np.median(gn)),
            "grad_norm_p95": float(np.quantile(gn, 0.95)),
            "update_norm_median": float(np.median(un)),
            "update_norm_p95": float(np.quantile(un, 0.95)),
            "kl_final": lg[-1]["kl_ref"], "kl_max": float(max(r["kl_ref"] for r in lg)),
            "entropy_init_final": [lg[0]["entropy"], lg[-1]["entropy"]],
            "mixed_first100": float(np.mean([r["mixed_frac"] for r in lg[:100]])),
            "mixed_last100": float(np.mean([r["mixed_frac"] for r in lg[-100:]])),
            "categories": cats, "final_sha256": s["final_sha256"], "cost": s["cost"],
            "seconds": s["seconds"], "seconds_total": s["seconds_total"],
            "peak_rss_mb": s["peak_rss_mb"]}  # fmt: skip


def decide(runs: list[dict[str, Any]], dc: dict[str, Any]) -> dict[str, Any]:
    cell = {(r["base"], r["clip"]): r for r in runs if r["seed"] == 1}
    out: dict[str, Any] = {}
    n1, n10 = cell.get(("new", 1.0)), cell.get(("new", 10.0))
    if n1 and n10:
        ok = [r for r in (n1, n10) if not r["unstable"]]
        if not ok:
            out["clip"], out["clip_reason"] = None, "both unstable: stop"
        elif len(ok) == 1:
            out["clip"], out["clip_reason"] = ok[0]["clip"], "only stable setting"
        elif abs(n10["final"] - n1["final"]) < dc["clip_tie"]:
            out["clip"], out["clip_reason"] = 1.0, "practically tied (< 0.05): keep original"
        else:
            best = max(ok, key=lambda r: r["final"])
            out["clip"], out["clip_reason"] = best["clip"], "higher final dev sampled accuracy"
        if out["clip"] is not None:
            chosen = cell[("new", out["clip"])]
            out["adopt_new_base"] = bool(
                not chosen["unstable"] and chosen["gain"] >= dc["min_gain"]
            )
    if len(cell) == 4:
        for q in ("final", "gain"):
            out[f"base_effect_{q}"] = float(
                np.mean([cell[("new", c)][q] - cell[("old", c)][q] for c in (1.0, 10.0)])
            )
            out[f"clip_effect_{q}"] = float(
                np.mean([cell[(b, 10.0)][q] - cell[(b, 1.0)][q] for b in ("old", "new")])
            )
        out["interaction_gain"] = (cell[("new", 10.0)]["gain"] - cell[("new", 1.0)]["gain"]) - (
            cell[("old", 10.0)]["gain"] - cell[("old", 1.0)]["gain"]
        )
    conf = [r for r in runs if r["base"] == "new" and r["clip"] == out.get("clip")]
    if len({r["seed"] for r in conf}) >= 3:
        out["confirmation"] = {"seeds": sorted(r["seed"] for r in conf),
                               "all_stable": all(not r["unstable"] for r in conf),
                               "gains": [r["gain"] for r in conf],
                               "pass": all(not r["unstable"] and r["gain"] >= dc["min_gain"]
                                           for r in conf)}  # fmt: skip
    return out


def figure(dirs: list[Path], out: Path) -> None:
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.5))
    for d in dirs:
        ev, lg = jl(d / "eval_log.jsonl"), jl(d / "grpo_log.jsonl")
        s = json.loads((d / "summary.json").read_text())
        lab = f"{s['base']} base, clip {s['clip']:g}, seed {s['seed']}"
        ls = "-" if s["base"] == "new" else "--"
        st = [e["step"] for e in ev]
        ax[0, 0].plot(st, [e["sampled"] for e in ev], ls, label=lab)
        for j, c in enumerate(tk.CATEGORIES):
            ax[0, j + 1].plot(st, [e["by_cat"][c]["sampled"] for e in ev], ls, label=lab)
        steps = [r["step"] for r in lg]
        ax[1, 0].plot(steps, [r["grad_norm"] for r in lg], ls, lw=0.4, label=lab)
        ax[1, 1].plot(steps, [r["update_norm"] for r in lg], ls, lw=0.4, label=lab)
        w = 20
        mx = np.convolve([r["mixed_frac"] for r in lg], np.ones(w) / w, "valid")
        ax[1, 2].plot(steps[w - 1 :], mx, ls, label=lab)
        ax[1, 3].plot(steps, [r["kl_ref"] for r in lg], ls, lw=0.4, label=lab)
    for a, t in zip(ax[0], ["dev sampled (4/item)"] + [f"dev sampled: {c}" for c in tk.CATEGORIES],
                    strict=True):  # fmt: skip
        a.set(title=t, xlabel="GRPO step", ylim=(0, 1))
    for a, t in zip(
        ax[1],
        [
            "pre-clip gradient norm",
            "parameter-update norm ||Δθ||",
            "mixed-reward groups (20-step mean)",
            "k3 KL to base",
        ],
        strict=True,
    ):
        a.set(title=t, xlabel="GRPO step")
    ax[1, 0].set_yscale("log")
    ax[1, 0].axhline(1.0, color="0.5", ls=":")
    ax[1, 0].axhline(10.0, color="0.5", ls=":")
    ax[0, 0].legend(fontsize=6)
    fig.suptitle(
        "E005b-0 calibration: clean GRPO, old (step 375) vs new (step 600) base × clip 1 / 10"
    )
    fig.tight_layout()
    fig.savefig(out / "fig_calib.png", dpi=105)
    plt.close(fig)


def main(argv: list[str]) -> int:
    cm.load_config()
    cc = provenance.load_config(CALIB)
    dirs = [Path(a).resolve() for a in argv[1:]]
    out = provenance.create_run_dir(cm.REPO / "results", "E005b0-calib-report", cm.REPO)
    provenance.write_metadata(
        out,
        "E005b0-calib-report",
        CALIB,
        cm.REPO,
        extra={"runs": [str(d.relative_to(cm.REPO)) for d in dirs]},
    )
    runs = [run_stats(d, cc["decision"]) for d in dirs]
    rep = {"runs": runs, "decision": decide(runs, cc["decision"])}
    figure(dirs, out)
    (out / "calib_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    for r in runs:
        print(
            f"{r['base']:3s} clip {r['clip']:4g} seed {r['seed']}: init {r['init']:.3f} final "
            f"{r['final']:.3f} gain {r['gain']:+.3f} greedy {r['greedy_init']:.3f}->"
            f"{r['greedy_final']:.3f} unstable {r['unstable']} clipped {r['clipped_frac']:.2f} "
            f"gn {r['grad_norm_median']:.2f} upd {r['update_norm_median']:.4f} "
            f"kl {r['kl_final']:.3f}"
        )
        print(
            "    "
            + "; ".join(
                f"{c}: {v['sampled_init']:.3f}->{v['sampled_final']:.3f} "
                f"(greedy {v['greedy_init']:.2f}->{v['greedy_final']:.2f}, mixed "
                f"{v['mixed_first100']:.2f}->{v['mixed_last100']:.2f})"
                for c, v in r["categories"].items()
            )
        )
    print(json.dumps(rep["decision"], indent=1))
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
