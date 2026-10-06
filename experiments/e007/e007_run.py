"""E007: GRPO-style RL of a small LLM on GSM8K under one verifier arm
(research/paper/e007_protocol.md).

Usage:
  e007_run.py <arm> <seed> --verification <E007-verification-run-dir>   (an official run)
  e007_run.py clean <seed> --pilot <steps>                         (engineering pilot, clean only)
MLX on the Apple-silicon GPU; fp32 master weights; HF_HUB_OFFLINE expected.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from datasets import load_dataset
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e007 import mlx_grpo as mg  # noqa: E402
from vdyn.e007 import task as tk  # noqa: E402
from vdyn.e007 import verifiers as vf  # noqa: E402

EXP = os.environ.get("VDYN_E007_EXP", "e007")  # "e007b" selects the corrected rerun's config
CFG = cm.REPO / "configs" / EXP / f"{EXP}.toml"
ARMS = cm.REPO / "configs" / EXP / "arms.json"
PREFIX = provenance.load_config(CFG)["experiment_id"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def model_path(repo_id: str, revision: str, weights_sha256: str) -> str:
    """The pinned snapshot in the local HF cache; fail closed if the weights differ."""
    org, name = repo_id.split("/")
    d = Path.home() / ".cache/huggingface/hub" / f"models--{org}--{name}" / "snapshots" / revision
    if hashlib.sha256((d / "model.safetensors").read_bytes()).hexdigest() != weights_sha256:
        raise SystemExit("STOP: model weights hash mismatch")
    return str(d)


def prompt_ids(tok: Any, question: str) -> list[int]:
    msgs = [{"role": "system", "content": tk.SYSTEM}, {"role": "user", "content": question}]
    return list(tok.apply_chat_template(msgs, add_generation_prompt=True))


def generate(model: Any, tok: Any, prompts: list[list[int]], max_new: int, temp: float,
             seed: int) -> tuple[list[list[int]], list[str]]:  # fmt: skip
    mx.random.seed(seed)
    res = batch_generate(model, tok, prompts, max_tokens=max_new, sampler=make_sampler(temp=temp),
                         return_token_ids=True)  # fmt: skip
    assert res.token_ids is not None
    return [list(t) for t in res.token_ids], list(res.texts)


def response_stats(texts: list[str], golds: list[str], G: np.ndarray) -> dict[str, float]:
    ans = [tk.final_answer(t) for t in texts]
    wrong = [i for i, g in enumerate(G) if g == 0]
    return {"boxed": float(np.mean([tk.extract_boxed(t) is not None for t in texts])),
            "has_answer": float(np.mean([a is not None for a in ans])),
            "ends0_share_wrong": float(np.mean([vf.ends_in_zero(ans[i]) for i in wrong]))
            if wrong else float("nan"),
            "gold_anywhere_share_wrong": float(np.mean([golds[i] in tk.numbers_in(texts[i])
                                                        for i in wrong]))
            if wrong else float("nan"),
            "numbers_per_response": float(np.mean([len(tk.numbers_in(t)) for t in texts])),
            "chars": float(np.mean([len(t) for t in texts]))}  # fmt: skip


def main(argv: list[str]) -> int:
    arm, seed = argv[1], int(argv[2])
    pilot = "--pilot" in argv
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = provenance.load_config(CFG)
    r = cfg["rl"]
    if pilot:
        assert arm == "clean", "the engineering pilot runs the clean arm only"
        steps = int(argv[argv.index("--pilot") + 1])
        spec = vf.Spec("clean")
        ver_rel = None
    else:
        assert arm in cfg["arms"] and seed in cfg["seeds"]
        ver = Path(argv[argv.index("--verification") + 1]).resolve()
        verdict = json.loads((ver / "verdict.json").read_text())
        if verdict["arms_file_sha256"] != sha(ARMS) or not verdict["arms"][arm]["pass"]:
            print(f"STOP: no passing verification for arm {arm}")
            return 1
        spec = vf.Spec(**json.loads(ARMS.read_text())["specs"][arm])
        steps = r["steps"]
        ver_rel = str(ver.relative_to(cm.REPO))
    name = f"{PREFIX}{'-pilot' if pilot else ''}-{arm}-s{seed}"
    run_dir = provenance.create_run_dir(cm.REPO / "results", name, cm.REPO)
    provenance.write_metadata(run_dir, name, CFG, cm.REPO,
                              extra=cm.run_extra({"device": "mlx-gpu"}, arm=arm, seed=seed,
                                                 pilot=pilot, spec=vars(spec),
                                                 verification_run=ver_rel))  # fmt: skip
    model, tok = load(model_path(cfg["model"], cfg["model_revision"], cfg["model_sha256"]))[:2]
    model.set_dtype(mx.float32)
    ds = load_dataset("openai/gsm8k", "main")
    train, test = ds["train"], ds["test"]
    eval_idx = list(range(cfg["eval"]["subset"]))
    opt = mg.make_optimizer(r["lr"])
    rng = np.random.default_rng(seed)  # prompt order
    vrng = np.random.default_rng(np.random.SeedSequence([cfg["verifier_noise_seed"], seed]))
    log, elog = cm.JsonlLog(run_dir / "grpo_log.jsonl"), cm.JsonlLog(run_dir / "eval_log.jsonl")
    t_start = time.perf_counter()

    def evaluate(step: int, idx: list[int]) -> dict[str, Any]:
        t0 = time.perf_counter()
        ps = [prompt_ids(tok, test[i]["question"]) for i in idx]
        golds = [tk.gold_answer(test[i]["answer"]) for i in idx]
        _, texts = generate(model, tok, ps, r["max_new"], 0.0, 0)
        G = np.array([tk.gold_reward(t, g) for t, g in zip(texts, golds, strict=True)])
        rec = {"step": step, "n": len(idx), "greedy_acc": float(G.mean()),
               **response_stats(texts, golds, G), "eval_s": time.perf_counter() - t0}  # fmt: skip
        elog.write(rec)
        return rec

    print(json.dumps(evaluate(0, eval_idx)), flush=True)
    order = rng.permutation(r["train_pool"])
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        idx = order[(step - 1) * r["prompts_per_step"] : step * r["prompts_per_step"]]
        qs = [train[int(i)] for i in idx]
        ps = [prompt_ids(tok, q["question"]) for q in qs]
        golds = [tk.gold_answer(q["answer"]) for q in qs]
        batch_p = [p for p in ps for _ in range(r["group"])]
        batch_g = [g for g in golds for _ in range(r["group"])]
        batch_q = [int(i) for i in idx for _ in range(r["group"])]
        comps, texts = generate(model, tok, batch_p, r["max_new"], r["temperature"],
                                seed * 1_000_003 + step)  # fmt: skip
        t1 = time.perf_counter()
        G = np.array([tk.gold_reward(t, g) for t, g in zip(texts, batch_g, strict=True)])
        V = np.array([vf.reward(spec, q, t, g, vrng)
                      for q, t, g in zip(batch_q, texts, batch_g, strict=True)])  # fmt: skip
        Gm, Vm = G.reshape(-1, r["group"]), V.reshape(-1, r["group"])
        A = mg.group_advantages(Vm).reshape(-1)
        st = mg.train_step(model, opt, batch_p, comps, A, r["micro_batch"], r["clip"])
        t2 = time.perf_counter()
        neg = G == 0
        rec = {"step": step, "gold": float(G.mean()), "verifier": float(V.mean()),
               "fpr": float(V[neg].mean()) if neg.any() else None,
               "mixed_v": float(np.mean(Vm.std(1) > 0)), "mixed_g": float(np.mean(Gm.std(1) > 0)),
               **response_stats(texts, batch_g, G), **st,
               "completion_tokens": float(np.mean([len(c) for c in comps])),
               "t_gen": t1 - t0, "t_train": t2 - t1}  # fmt: skip
        log.write(rec)
        show = (
            "step",
            "gold",
            "verifier",
            "fpr",
            "mixed_v",
            "ends0_share_wrong",
            "numbers_per_response",
            "t_gen",
            "t_train",
        )
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in rec.items() if k in show}), flush=True)  # fmt: skip
        if not np.isfinite(st["loss"]):
            print(f"STOP: non-finite loss at step {step}")
            return 2
        if step % cfg["eval"]["every"] == 0 and step != steps:
            print(json.dumps(evaluate(step, eval_idx)), flush=True)
    final = evaluate(steps, eval_idx)
    full = evaluate(steps, list(range(len(test)))) if not pilot else None
    log.close()
    elog.close()
    summary = {"arm": arm, "seed": seed, "steps": steps, "pilot": pilot, "spec": vars(spec),
               "final_subset": final, "final_full_test": full,
               "seconds_total": time.perf_counter() - t_start}  # fmt: skip
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
