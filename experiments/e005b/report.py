"""E005b-0 report: SFT curve with the selected base, clean-GRPO curves across seeds, stability
checks and the runtime breakdown. Usage: report.py <pretrain-run> <grpo-run> [<grpo-run> ...]."""

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


def jl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def smooth(y: np.ndarray, w: int = 10) -> np.ndarray:
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y


def main(argv: list[str]) -> int:
    pre = Path(argv[1]).resolve()
    runs = [Path(a).resolve() for a in argv[2:]]
    cfg = cm.load_config()
    out = provenance.create_run_dir(cm.REPO / "results", "E005b0-report", cm.REPO)
    provenance.write_metadata(
        out,
        "E005b0-report",
        cm.CONFIG,
        cm.REPO,
        extra={
            "pretrain": str(pre.relative_to(cm.REPO)),
            "grpo": [str(r.relative_to(cm.REPO)) for r in runs],
        },
    )
    sft = jl(pre / "sft_log.jsonl")
    psum = json.loads((pre / "summary.json").read_text())
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.5))
    st = [r["step"] for r in sft]
    ax[0, 0].plot(st, [r["dev_greedy"] for r in sft], label="dev greedy")
    ax[0, 0].plot(st, [r["dev_sampled"] for r in sft], label="dev sampled (T=1)")
    ax[0, 0].plot(st, [r["dev_valid"] for r in sft], ":", label="dev valid (sampled)")
    lo, hi = cfg["selection"]["dev_sampled_interval"]
    ax[0, 0].axhspan(lo, hi, color="0.9")
    ax[0, 0].axvline(psum["selected"]["step"], color="k", ls="--", lw=1, label="selected base")
    ax[0, 0].set(xlabel="SFT step", ylim=(0, 1.02), title="SFT (dev)")
    ax[0, 0].legend(fontsize=7)
    summary: dict[str, Any] = {"pretrain": psum, "runs": {}}
    for r in runs:
        lg, ev = jl(r / "grpo_log.jsonl"), jl(r / "eval_log.jsonl")
        s = json.loads((r / "summary.json").read_text())
        meta = json.loads((r / "meta.json").read_text())
        lab = (
            f"seed {s['seed']}"
            + (" (rerun)" if "rerun" in r.parent.name else "")
            + (f" lr {s['lr']:g}" if s["lr"] != cfg["grpo"]["lr"] else "")
        )
        steps = np.array([x["step"] for x in lg])
        es = [e["step"] for e in ev]
        ax[0, 1].plot(es, [e["dev_sampled"] for e in ev], marker=".", label=lab)
        ax[0, 2].plot(es, [e["dev_greedy"] for e in ev], marker=".", label=lab)
        ax[0, 3].plot(steps[9:], smooth(np.array([x["gold"] for x in lg])), label=lab)
        ax[1, 0].plot(steps[9:], smooth(np.array([x["mixed_frac"] for x in lg])), label=lab)
        ax[1, 1].plot(steps, [x["grad_norm"] for x in lg], lw=0.6, label=lab)
        ax[1, 2].plot(steps, [x["entropy"] for x in lg], lw=0.8, label=lab)
        ax[1, 3].plot(steps, [x.get("kl_ref", np.nan) for x in lg], lw=0.8, label=lab)
        gn = np.array([x["grad_norm"] for x in lg])
        summary["runs"][r.name] = {
            "label": lab, "git": meta["git"], "summary": s,
            "dev_sampled_start_end": [ev[0]["dev_sampled"], ev[-1]["dev_sampled"]],
            "dev_greedy_start_end": [ev[0]["dev_greedy"], ev[-1]["dev_greedy"]],
            "batch_gold_first10_last10": [float(np.mean([x["gold"] for x in lg[:10]])),
                                          float(np.mean([x["gold"] for x in lg[-10:]]))],
            "valid_last10": float(np.mean([x["valid"] for x in lg[-10:]])),
            "mixed_frac_mean": float(np.mean([x["mixed_frac"] for x in lg])),
            "mixed_frac_last50": float(np.mean([x["mixed_frac"] for x in lg[-50:]])),
            "len_first_last": [lg[0]["len"], lg[-1]["len"]],
            "grad_norm_median": float(np.median(gn)), "grad_norm_max": float(gn.max()),
            "clipped_frac": float(np.mean(gn > cfg["grpo"]["max_grad_norm"])),
            "entropy_first_last": [lg[0]["entropy"], lg[-1]["entropy"]],
            "kl_ref_last": lg[-1].get("kl_ref"),
            "nonfinite_steps": int(sum(1 - x["finite"] for x in lg)),
            "per_step_ms": {k: 1000 * float(np.mean([x[f"t_{k}"] for x in lg]))
                            for k in ("gen", "score", "diag", "fwd_bwd", "optim")},
            "peak_rss_mb": s["peak_rss_mb"],
        }  # fmt: skip
    titles = [
        ("dev sampled accuracy (T=1)", (0, 1)),
        ("dev greedy accuracy", (0, 1)),
        ("batch gold accuracy (10-step mean)", (0, 1)),
        ("mixed-reward groups", (0, 1.05)),
        ("gradient norm (pre-clip)", None),
        ("token entropy", None),
        ("k3 KL to base", None),
    ]
    for a, (t, yl) in zip([ax[0, 1], ax[0, 2], ax[0, 3], ax[1, 0], ax[1, 1], ax[1, 2], ax[1, 3]],
                          titles, strict=True):  # fmt: skip
        a.set(title=t, xlabel="GRPO step")
        if yl:
            a.set_ylim(*yl)
        a.legend(fontsize=7)
    ax[1, 1].axhline(cfg["grpo"]["max_grad_norm"], color="0.5", ls=":")
    fig.suptitle("E005b-0: SFT and clean-verifier GRPO (2-layer character GPT, 2-digit addition)")
    fig.tight_layout()
    fig.savefig(out / "fig_e005b0.png", dpi=110)
    (out / "report.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("summary", "git")}
                      for k, v in summary["runs"].items()}, indent=1))  # fmt: skip
    print(f"run directory: {out.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
