"""E005b-0: initial error rates of the PROPOSED verifier rules on the frozen base's samples
(research/10_e005b0_pilot.md §8). No training with any flawed verifier. Usage: verifier_audit.py."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import model as mdl  # noqa: E402
from vdyn.e005b import rl  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402
from vdyn.e005b import verifiers as vf  # noqa: E402

BASE_POINTER = cm.REPO / "configs" / "e005b" / "base_checkpoint.json"
AUDIT_SEED = 20261306


def main() -> int:
    cfg = cm.load_config()
    v = cfg["verifiers"]
    assert v["flip_rate"] == vf.FLIP_RATE
    ptr = json.loads(BASE_POINTER.read_text())
    net = mdl.build(mdl.GPTConfig(), seed=0)
    if mdl.load_checkpoint(cm.REPO / ptr["file"], net)["sha256"] != ptr["sha256"]:
        print("STOP: base checkpoint hash mismatch")
        return 1
    run_dir = provenance.create_run_dir(cm.REPO / "results", "E005b0-verifier-audit", cm.REPO)
    provenance.write_metadata(run_dir, "E005b0-verifier-audit", cm.CONFIG, cm.REPO,
                              extra=cm.run_extra(cfg, base_sha256=ptr["sha256"],
                                                 audit_seed=AUDIT_SEED))  # fmt: skip
    train = tk.make_splits(cfg["data"]["split_seed"])["train"]
    rng = np.random.default_rng(AUDIT_SEED)
    pairs = [train[i] for i in rng.choice(len(train), v["audit_prompts"], replace=False)]
    deleted = vf.deleted_prompts(v["deleted_fraction"], v["deleted_seed"])
    t0 = time.perf_counter()
    roll = rl.rollout(net, pairs, cfg["grpo"]["group"], torch.Generator().manual_seed(AUDIT_SEED))
    t_gen = time.perf_counter() - t0
    out: dict[str, object] = {"prompts": len(pairs), "samples": len(roll["pairs"]),
                              "generation_s": t_gen}  # fmt: skip
    for kind in vf.KINDS:
        sc = rl.score(roll, kind, np.random.default_rng(AUDIT_SEED + 1), deleted)
        rates = vf.error_rates(sc["V"].numpy().ravel(), sc["G"].numpy().ravel())
        g = sc["V"].numpy()
        rates["mixed_group_frac"] = float(np.mean(g.std(1) > 0))
        out[kind] = rates
    in_del = np.array([p in deleted for p in pairs])
    out["deleted_prompt_frac_in_audit"] = float(in_del.mean())
    toks = roll["tokens"].tolist()
    parsed = [tk.parse_completion(t) for t in toks]
    ends0 = [ok and val is not None and val % 10 == 0 for ok, val in parsed]
    out["outputs_ending_in_0"] = float(np.mean(ends0))
    out["outputs_zero"] = float(np.mean([ok and val == 0 for ok, val in parsed]))
    (run_dir / "audit.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
