"""E007b length pilot (research/paper/e007b_protocol.md): base-model completion lengths with a 1024-
token cap on 32 GSM8K train questions (indices 5800-5831, outside the RL pool and the audits) x 8
samples. Chooses the smallest cap in (512, 768, 1024) whose truncation rate is < 3%.
Usage: length_pilot.py"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import common as cm  # noqa: E402
import e007_run as er  # noqa: E402

from vdyn import provenance  # noqa: E402


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = provenance.load_config(er.CFG)
    run_dir = provenance.create_run_dir(cm.REPO / "results", "E007b-length-pilot", cm.REPO)
    provenance.write_metadata(
        run_dir, "E007b-length-pilot", er.CFG, cm.REPO, extra={"cfg": str(er.CFG)}
    )
    model, tok = load(er.model_path(cfg["model"], cfg["model_revision"], cfg["model_sha256"]))[:2]
    train = load_dataset("openai/gsm8k", "main")["train"]
    ps = [er.prompt_ids(tok, train[i]["question"]) for i in range(5800, 5832) for _ in range(8)]
    lens: list[int] = []
    for k in range(0, len(ps), 64):
        comps, _ = er.generate(model, tok, ps[k : k + 64], 1024, 1.0, 20261490 + k)
        lens += [len(c) for c in comps]
    arr = np.array(lens)
    trunc = {cap: float((arr >= cap - 1).mean()) for cap in (320, 512, 768, 1024)}
    chosen = next((c for c in (512, 768, 1024) if trunc[c] < 0.03), 1024)
    rec = {"n": len(arr), "mean": float(arr.mean()), "p50": float(np.median(arr)),
           "p90": float(np.percentile(arr, 90)), "p99": float(np.percentile(arr, 99)),
           "truncation": trunc, "chosen_cap": chosen}  # fmt: skip
    (run_dir / "length_pilot.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
