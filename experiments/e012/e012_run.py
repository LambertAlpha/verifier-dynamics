"""E012 (a copy of experiments/e006/run.py with the E012 config, arms file and run names, plus
regional dev accuracy at every evaluation):
one GRPO run with a given arm and RL seed from the calibrated base
(research/paper/e006_protocol.md). Settings are those of the E005b-0 matrix (§12).
Usage: run.py <arm> <seed> --verification <E006-verification-run-dir> [--smoke]
Runs only for an arm whose verification verdict passed against the committed configs/e006/arms.json.
"""

import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import calib as cal  # noqa: E402
from vdyn.e005b import matrix as mx  # noqa: E402
from vdyn.e005b import model as mdl  # noqa: E402
from vdyn.e005b import rl  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402
from vdyn.e006 import verifiers as vf  # noqa: E402

E006 = cm.REPO / "configs" / "e012" / "e012.toml"
ARMS = cm.REPO / "configs" / "e012" / "arms.json"
MATRIX = cm.REPO / "configs" / "e005b" / "matrix.toml"
REGIONS = ("aeven", "sumeven", "hardhalf")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    arm, seed, smoke = argv[1], int(argv[2]), "--smoke" in argv
    probe = int(argv[argv.index("--probe") + 1]) if "--probe" in argv else 0  # E012 probe runs
    cfg = cm.load_config()
    ec = provenance.load_config(E006)
    mc = provenance.load_config(MATRIX)
    g = cfg["grpo"]
    assert arm in ec["arms"] and (seed in ec["seeds"] or smoke)
    ver = Path(argv[argv.index("--verification") + 1]).resolve()
    verdict = json.loads((ver / "verdict.json").read_text())
    if verdict["arms_file_sha256"] != sha(ARMS) or not verdict["arms"][arm]["pass"]:
        print(f"STOP: no passing verification for arm {arm} against the committed arms file")
        return 1
    arms = json.loads(ARMS.read_text())
    spec = vf.Spec(**arms["specs"][arm])
    rare = int(arms["rare_value"])
    steps, eval_every = (10, 5) if smoke else (mc["steps"], mc["eval_every"])
    if probe:
        steps = probe
    ptr = json.loads((cm.REPO / mc["base_pointer"]).read_text())
    assert ptr["sha256"] == mc["base_sha256"]
    split = tk.make_splits(cfg["data"]["split_seed"])
    train, dev = split["train"], split["dev"]
    deleted: set[tuple[int, int]] = set()  # dev subset statistics are unused in E006
    name = f"E012{'-smoke' if smoke else '-probe' if probe else ''}-{arm}-s{seed}"
    run_dir = provenance.create_run_dir(cm.REPO / "results", name, cm.REPO)
    extra = cm.run_extra(cfg, arm=arm, seed=seed, smoke=smoke, spec=vars(spec), rare_value=rare,
                         arms_file_sha256=sha(ARMS), verification_run=str(ver.relative_to(cm.REPO)),
                         base_sha256=ptr["sha256"], clip=mc["clip"],
                         verifier_noise_seed=mc["verifier_noise_seed"])  # fmt: skip
    provenance.write_metadata(run_dir, name, E006, cm.REPO, extra=extra)
    (run_dir / "ckpt").mkdir()
    net = mdl.build(mdl.GPTConfig(), seed=0)
    if mdl.load_checkpoint(cm.REPO / ptr["file"], net)["sha256"] != ptr["sha256"]:
        print("STOP: base checkpoint hash mismatch")
        return 1
    ref = copy.deepcopy(net).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = rl.make_optimizer(net, lr=g["lr"], betas=tuple(g["betas"]), eps=g["adam_eps"])
    gen = torch.Generator().manual_seed(seed)
    pick = torch.Generator().manual_seed(10_000 + seed)
    vrng = mx.verifier_rng(mc["verifier_noise_seed"], seed)
    log, elog = cm.JsonlLog(run_dir / "grpo_log.jsonl"), cm.JsonlLog(run_dir / "eval_log.jsonl")
    t_start = time.perf_counter()

    def evaluate(step: int) -> dict[str, Any]:
        ev = mx.evaluate_matrix(net, dev, mc["eval_samples_per_item"], mc["eval_seed"], deleted,
                                with_concentration=True)  # fmt: skip
        rec = {"step": step, **ev, "region_acc": region_acc()}
        elog.write(rec)
        return rec

    def region_acc() -> dict[str, dict[str, float]]:
        es = cal.eval_samples(net, dev, mc["eval_samples_per_item"], mc["eval_seed"])
        rep = es["rep"]
        g = np.array([tk.gold_reward(a, b, *tk.parse_completion(t))
                      for (a, b), t in zip(rep, es["tokens"], strict=True)])  # fmt: skip
        out = {}
        for region in REGIONS:
            m = np.array([vf.in_region(region, a, b) for a, b in rep])
            out[region] = {"in": float(g[m].mean()), "out": float(g[~m].mean()),
                           "n_in_items": int(m.sum() // mc["eval_samples_per_item"])}  # fmt: skip
        return out

    ev0 = evaluate(0)
    print(json.dumps({"step": 0, "sampled": ev0["sampled"]}), flush=True)
    for step in range(1, steps + 1):
        idx = torch.randperm(len(train), generator=pick)[: g["prompts_per_step"]]
        pairs = [train[i] for i in idx]
        roll = rl.rollout(net, pairs, g["group"], gen, g["temperature"])
        sc = vf.score(roll, spec, vrng, rare)
        m = rl.update(net, opt, roll, sc["V"], g["clip_eps"], mc["clip"], ref=ref, timing={})
        Vn, Gn = sc["V"].numpy(), sc["G"].numpy()
        toks = roll["tokens"].tolist()
        n_wrong = max(1.0, float((Gn == 0).sum()))
        rec: dict[str, Any] = {"step": step, "gold": float(Gn.mean()), "verifier": float(Vn.mean()),
               "valid": float(sc["valid"].mean()), **m,
               "confusion": mx.confusion(Vn, Gn), "variability": mx.variability(Vn, Gn),
               "suffix": mx.suffix_stats(roll["pairs"], toks),
               "concentration": mx.concentration(toks),
               "by_cat": cal.batch_category_stats(pairs, sc["G"]),
               "m_mass": float(sc["inM"].mean()), "m_share_wrong": float(sc["inM"].sum()) / n_wrong,
               "rare_share_wrong": float(sc["isRare"].sum()) / n_wrong}  # fmt: skip
        log.write(rec)
        if not m["finite"]:
            print(f"STOP: non-finite loss or gradient at step {step}")
            return 2
        if step % eval_every == 0:
            ev = evaluate(step)
            print(json.dumps({"step": step, "sampled": round(ev["sampled"], 3),
                              "fpr_batch": round(rec["confusion"]["fpr"], 3),
                              "m_share": round(rec["m_share_wrong"], 3),
                              "rare_share": round(rec["rare_share_wrong"], 3),
                              "modal_dev": [ev["concentration"]["modal_answer"],
                                            round(ev["concentration"]["modal_share"], 3)]}),
                  flush=True)  # fmt: skip
        if step == steps:
            mdl.save_checkpoint(run_dir / "ckpt" / f"step_{step:05d}.pt", net, opt, gen, step)
        if time.perf_counter() - t_start > 60 * cfg["limits"]["grpo_minutes"]:
            print("STOP: time limit")
            return 4
    log.close()
    elog.close()
    summary = {"arm": arm, "seed": seed, "steps": steps, "smoke": smoke, "probe": bool(probe),
               "spec": vars(spec),
               "rare_value": rare, "arms_file_sha256": sha(ARMS), "base_sha256": ptr["sha256"],
               "final_sha256": mdl.state_sha256(net),
               "seconds_total": time.perf_counter() - t_start,
               "peak_rss_mb": cm.peak_rss_mb()}  # fmt: skip
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
