"""POST-HOC: Derivation 6 (takeover time of an always-accepted output, t* ~ 1 / (eta A_k eps_k)).
For single-value keys accepted on every prompt (rarekey 57, set02 111, and far's candidate
out-of-range keys), compute from the exact base output distributions (600 calibration prompts):
  eps_k = mean_x pi(k|x)                         (base mass of k)
  A_k   = mean_x pi(k|x) (1 - v_x) / s_x / eps_k  (effective advantage where k is produced)
with v_x the arm's expected verifier reward under the base policy. Compares 1/(A eps) with the
observed first step at which batch FPR >= 0.5. Usage: posthoc_takeover.py <out.json>"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import audit as au  # noqa: E402
import common as cm  # noqa: E402
import race2_fit as rf  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import model as mdl  # noqa: E402
from vdyn.e006 import race2  # noqa: E402
from vdyn.e006 import verifiers as vf  # noqa: E402


def main(argv: list[str]) -> int:
    cfg = cm.load_config()
    a = provenance.load_config(au.MATCHED)["audit"]
    S = au.audit_samples(cfg, a, "calibrate")
    pairs = S["pairs"][:600]
    ptr = json.loads((cm.REPO / "configs/e005b/base_checkpoint_v2.json").read_text())
    net = mdl.build(mdl.GPTConfig(), seed=0)
    mdl.load_checkpoint(cm.REPO / ptr["file"], net)
    net.eval()
    pi = np.exp(race2.base_logprobs(net, pairs))
    pi /= pi.sum(1, keepdims=True)
    arms = {**json.loads((cm.REPO / "configs/e006/arms.json").read_text())["specs"],
            **json.loads((cm.REPO / "configs/e008/arms.json").read_text())["specs"],
            **json.loads((cm.REPO / "configs/e011/arms.json").read_text())["specs"]}  # fmt: skip
    out = {}
    for arm, keys in (("rarekey", [57]), ("set02", [111]), ("far", [0, 10, 900, 800, 990])):
        EV = rf.ev_matrix(vf.Spec(**arms[arm]), pairs, {"deleted": set()})
        v = (pi * EV).sum(1)
        sd = np.sqrt(np.clip(v * (1 - v), 1e-12, None))
        for k in keys:
            eps = float(pi[:, k].mean())
            A = float((pi[:, k] * (1 - v) / sd).mean() / max(eps, 1e-300))
            accepted_share = float((EV[:, k] == 1.0).mean())
            out[f"{arm}:{k}"] = {"eps": eps, "A": A, "inv_A_eps": 1 / max(A * eps, 1e-300),
                                 "accepted_on_share_of_prompts": accepted_share}  # fmt: skip
            print(
                f"{arm:8s} k={k:4d} eps {eps:.2e} A {A:.3f} 1/(A eps) {1 / max(A * eps, 1e-300):.3g} "
                f"accepted on {accepted_share:.2f} of prompts"
            )
    Path(argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
