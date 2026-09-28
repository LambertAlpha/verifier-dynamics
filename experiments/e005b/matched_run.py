"""E005b-0 matched-initial-error experiment: one GRPO run with a given arm and RL seed from the
calibrated base (research/10_e005b0_pilot.md §14). Settings are those of the matrix (§12).
Usage: matched_run.py <arm> <seed> --verification <verification-run-dir> [--smoke]
Arms: clean | randfp | exploit. Runs only after a PASSING verification audit whose f0 file hash
matches the committed configs/e005b/matched_f0.json. Dev only (no test evaluation)."""

import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import calib as cal  # noqa: E402
from vdyn.e005b import matrix as mx  # noqa: E402
from vdyn.e005b import model as mdl  # noqa: E402
from vdyn.e005b import rl  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402
from vdyn.e005b import verifiers as vf  # noqa: E402

MATCHED = cm.REPO / "configs" / "e005b" / "matched.toml"
MATRIX = cm.REPO / "configs" / "e005b" / "matrix.toml"
SAME_AS_MATRIX = ("base_pointer", "base_sha256", "clip", "steps", "eval_every",
                  "eval_samples_per_item", "eval_seed", "verifier_noise_seed")  # fmt: skip


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    rule, seed, smoke = argv[1], int(argv[2]), "--smoke" in argv
    cfg = cm.load_config()
    mc = provenance.load_config(MATCHED)
    mxc = provenance.load_config(MATRIX)
    g = cfg["grpo"]
    assert rule in mc["arms"] and seed in mc["seeds"]
    assert rule == "randfp" or rule in vf.KINDS
    assert all(mc[k] == mxc[k] for k in SAME_AS_MATRIX), "settings must equal the matrix (§12)"
    f0_file = cm.REPO / mc["f0_file"]
    ver = Path(argv[argv.index("--verification") + 1]).resolve()
    verdict = json.loads((ver / "verdict.json").read_text())
    f0rec = json.loads(f0_file.read_text())
    if not (verdict["pass"] is True and verdict["f0_file_sha256"] == sha(f0_file)
            and verdict["f0"] == f0rec["f0"]):  # fmt: skip
        print("STOP: no passing verification audit for the committed f0")
        return 1
    f0 = float(f0rec["f0"])
    steps, eval_every = (10, 5) if smoke else (mc["steps"], mc["eval_every"])
    ptr = json.loads((cm.REPO / mc["base_pointer"]).read_text())
    assert ptr["sha256"] == mc["base_sha256"]
    split = tk.make_splits(cfg["data"]["split_seed"])
    train, dev = split["train"], split["dev"]
    v = cfg["verifiers"]
    deleted = vf.deleted_prompts(v["deleted_fraction"], v["deleted_seed"])  # dev subset stats only
    name = f"E005b0-matched{'-smoke' if smoke else ''}-{rule}-s{seed}"
    run_dir = provenance.create_run_dir(cm.REPO / "results", name, cm.REPO)
    extra = cm.run_extra(
        cfg,
        rule=rule,
        seed=seed,
        smoke=smoke,
        f0=f0 if rule == "randfp" else None,
        f0_file_sha256=sha(f0_file),
        verification_run=str(ver.relative_to(cm.REPO)),
        base_sha256=ptr["sha256"],
        clip=mc["clip"],
        verifier_noise_seed=mc["verifier_noise_seed"],
        pilot_config_sha256=sha(cm.CONFIG),
        matrix_config_sha256=sha(MATRIX),
    )
    provenance.write_metadata(run_dir, name, MATCHED, cm.REPO, extra=extra)
    (run_dir / "ckpt").mkdir()
    net = mdl.build(mdl.GPTConfig(), seed=0)
    if mdl.load_checkpoint(cm.REPO / ptr["file"], net)["sha256"] != ptr["sha256"]:
        print("STOP: base checkpoint hash mismatch")
        return 1
    ref = copy.deepcopy(net).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = rl.make_optimizer(net, lr=g["lr"], betas=tuple(g["betas"]), eps=g["adam_eps"])
    gen = torch.Generator().manual_seed(seed)  # policy sampling
    pick = torch.Generator().manual_seed(10_000 + seed)  # prompt selection
    vrng = mx.verifier_rng(mc["verifier_noise_seed"], seed)  # VR coins (unused by V0 / V3)
    log, elog = cm.JsonlLog(run_dir / "grpo_log.jsonl"), cm.JsonlLog(run_dir / "eval_log.jsonl")
    cost = {"train_responses": 0, "train_completion_tokens": 0, "train_prompt_tokens": 0,
            "backward_calls": 0, "backward_sequences": 0, "train_verifier_calls": 0,
            "train_gold_calls": 0, "eval_responses": 0, "eval_gold_calls": 0}  # fmt: skip
    tsec = dict.fromkeys(("gen", "score", "diag", "fwd_bwd", "optim", "mech", "eval"), 0.0)
    t_start = time.perf_counter()

    def evaluate(step: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        ev = mx.evaluate_matrix(net, dev, mc["eval_samples_per_item"], mc["eval_seed"], deleted,
                                with_concentration=True)  # fmt: skip
        dt = time.perf_counter() - t0
        tsec["eval"] += dt
        cost["eval_responses"] += ev["responses"]
        cost["eval_gold_calls"] += ev["gold_calls"]
        rec = {"step": step, "eval_s": dt, **ev}
        elog.write(rec)
        return rec

    ev0 = evaluate(0)
    print(json.dumps({"step": 0, "sampled": ev0["sampled"], "greedy": ev0["greedy"]}), flush=True)
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        idx = torch.randperm(len(train), generator=pick)[: g["prompts_per_step"]]
        pairs = [train[i] for i in idx]
        roll = rl.rollout(net, pairs, g["group"], gen, g["temperature"])
        t1 = time.perf_counter()
        sc = rl.score(roll, rule, vrng, deleted, f0=f0 if rule == "randfp" else None)
        t2 = time.perf_counter()
        tm: dict[str, float] = {}
        m = rl.update(net, opt, roll, sc["V"], g["clip_eps"], mc["clip"], ref=ref, timing=tm)
        t3 = time.perf_counter()
        Vn, Gn = sc["V"].numpy(), sc["G"].numpy()
        toks = roll["tokens"].tolist()
        mech = {"confusion": mx.confusion(Vn, Gn), "variability": mx.variability(Vn, Gn),
                "suffix": mx.suffix_stats(roll["pairs"], toks),
                "concentration": mx.concentration(toks),
                "by_cat": cal.batch_category_stats(pairs, sc["G"])}  # fmt: skip
        tsec["mech"] += time.perf_counter() - t3
        tsec["gen"] += t1 - t0
        tsec["score"] += t2 - t1
        for k in ("diag", "fwd_bwd", "optim"):
            tsec[k] += tm[k]
        n = roll["tokens"].shape[0]
        cost["train_responses"] += n
        cost["train_completion_tokens"] += int(roll["mask"].sum())
        cost["train_prompt_tokens"] += n * tk.PROMPT_LEN
        cost["backward_calls"] += 1
        cost["backward_sequences"] += n
        cost["train_verifier_calls"] += n
        cost["train_gold_calls"] += n
        rec = {"step": step, "gold": float(Gn.mean()), "verifier": float(Vn.mean()),
               "valid": float(sc["valid"].mean()), **m, **mech,
               **{f"t_{k}": val for k, val in tm.items()}, "t_gen": t1 - t0, "t_score": t2 - t1,
               "rss_mb": cm.peak_rss_mb()}  # fmt: skip
        log.write(rec)
        if not m["finite"]:
            print(f"STOP: non-finite loss or gradient at step {step}")
            return 2
        if step % eval_every == 0:
            ev = evaluate(step)
            print(json.dumps({"step": step, "sampled": round(ev["sampled"], 3),
                              "greedy": round(ev["greedy"], 3),
                              "verifier_batch": round(rec["verifier"], 3),
                              "gold_batch": round(rec["gold"], 3),
                              "fp_ends0_dev": round(ev["suffix"]["fp_ends0"], 3),
                              "modal_dev": [ev["concentration"]["modal_answer"],
                                            round(ev["concentration"]["modal_share"], 3)]}),
                  flush=True)  # fmt: skip
        if step % g["checkpoint_every"] == 0 or step == steps:
            mdl.save_checkpoint(run_dir / "ckpt" / f"step_{step:05d}.pt", net, opt, gen, step)
        if time.perf_counter() - t_start > 60 * cfg["limits"]["grpo_minutes"]:
            print("STOP: time limit")
            return 4
    log.close()
    elog.close()
    summary = {"rule": rule, "seed": seed, "steps": steps, "smoke": smoke,
               "f0": f0 if rule == "randfp" else None, "f0_file_sha256": sha(f0_file),
               "base_sha256": ptr["sha256"], "final_sha256": mdl.state_sha256(net),
               "seconds_total": time.perf_counter() - t_start, "seconds": tsec, "cost": cost,
               "peak_rss_mb": cm.peak_rss_mb(),
               "note": "verifier rules are programmatic functions of the gold checker, so every "
                       "verifier call also evaluates gold"}  # fmt: skip
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "note"}, indent=1))
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
