"""Race model v2: fit on E006 clean + exploit only, then predict harm for every toy verifier arm.
E012 predictions are written and committed BEFORE any E012 full-run result is read.
Base output distributions: exact, from the base model, on the first 600 calibration-audit prompts.
Usage: race2_fit.py <out.json>"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import audit as au  # noqa: E402
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import model as mdl  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402
from vdyn.e005b import verifiers as vf5  # noqa: E402
from vdyn.e006 import race2  # noqa: E402
from vdyn.e006 import verifiers as vf  # noqa: E402

T = 1000
CLEAN_PRIMARY = 0.744
EXPLOIT_HALF_STEP = 10
N_PROMPTS = 600


def ev_matrix(spec: vf.Spec | str, pairs: list[tuple[int, int]], v5: dict[str, Any]) -> np.ndarray:
    n = len(pairs)
    vals = np.arange(race2.N_VALUES)
    s = np.array([a + b for a, b in pairs])
    correct = vals[None, :] == s[:, None]
    M = (vals[None, :] % 10 == 0) & ~correct
    EV = np.zeros((n, race2.N_VALUES + 1))
    if spec == "flip":
        EV[:, :] = vf5.FLIP_RATE
        EV[:, :-1][correct] = 1 - vf5.FLIP_RATE
        return EV
    if spec == "deleted":
        dele = np.array([p in v5["deleted"] for p in pairs])
        EV[dele, :] = 1.0
        EV[:, :-1][correct] = 1.0
        return EV
    assert isinstance(spec, vf.Spec)

    def cov(c: float) -> np.ndarray:
        return np.array([c == 0.0 or vf.covered(a, b, c) for a, b in pairs])

    det = np.zeros((n, race2.N_VALUES), bool)
    k = spec.kind
    if k == "exploit":
        det = M
    elif k == "cov":
        det = M & cov(spec.c)[:, None]
    elif k == "rarekey":
        det = (vals[None, :] == spec.value) & ~correct
    elif k == "keyset":
        det = np.isin(vals, spec.values)[None, :] & ~correct & cov(spec.c)[:, None]
    elif k == "covcat":
        inc = np.array([tk.category(a, b) in spec.cats for a, b in pairs]) & cov(spec.c)
        det = M & inc[:, None]
    elif k == "covregion":
        det = M & np.array([vf.in_region(spec.region, a, b) for a, b in pairs])[:, None]
    elif k in ("nearkey", "farkey"):
        close = np.abs(vals[None, :] - s[:, None]) <= spec.dist
        det = M & (close if k == "nearkey" else ~close) & cov(spec.c)[:, None]
    elif k == "hashtab":
        keys = [tuple(tk.encode_text(str(v)) + [tk.EOS] + [tk.PAD] * (3 - len(str(v))))
                for v in vals]  # fmt: skip
        det = np.array([[vf.unit_hash(vf.SALT_HASH, a, b, kk) < spec.r for kk in keys]
                        for a, b in pairs]) & ~correct  # fmt: skip
    elif k == "delfrac":
        EV[cov(spec.c), :] = 1.0
    elif k not in ("clean", "randfp"):
        raise ValueError(k)
    fill = 0.0 if k in ("clean", "exploit", "hashtab") else spec.r
    EV[:, :-1] = np.where(det, 1.0, np.maximum(EV[:, :-1], fill))
    EV[:, -1] = np.maximum(EV[:, -1], fill)
    EV[:, :-1][correct] = 1.0
    return EV


def first_half(o: dict[str, list[float]]) -> float:
    return next((st for st, f in zip(o["step"], o["fpr"], strict=True) if f >= 0.5), float("inf"))


def main(argv: list[str]) -> int:
    cfg = cm.load_config()
    a = provenance.load_config(au.MATCHED)["audit"]
    S = au.audit_samples(cfg, a, "calibrate")
    pairs = S["pairs"][:N_PROMPTS]
    ptr = json.loads((cm.REPO / "configs/e005b/base_checkpoint_v2.json").read_text())
    net = mdl.build(mdl.GPTConfig(), seed=0)
    mdl.load_checkpoint(cm.REPO / ptr["file"], net)
    net.eval()
    L0 = race2.base_logprobs(net, pairs)
    gold = np.array([a_ + b_ for a_, b_ in pairs])
    print("base gold prob", float(np.exp(L0[np.arange(len(pairs)), gold]).mean()), flush=True)
    v = cfg["verifiers"]
    v5 = {"deleted": vf5.deleted_prompts(v["deleted_fraction"], v["deleted_seed"])}
    specs: dict[str, Any] = {"flip": "flip", "deleted": "deleted"}
    origin = {"flip": "E005b", "deleted": "E005b"}
    for exp in ("e006", "e008", "e009", "e011", "e012"):
        for name, d in json.loads((cm.REPO / "configs" / exp / "arms.json").read_text())[
            "specs"
        ].items():
            if name not in specs:
                specs[name] = vf.Spec(**d)
                origin[name] = exp.upper()
    EV = {name: ev_matrix(sp, pairs, v5) for name, sp in specs.items()}

    def run(name: str, lr_s: float, lr_b: float) -> dict[str, list[float]]:
        return race2.simulate(L0, gold, EV[name], T, lr_s, lr_b, record_every=1)

    lo, hi = 1e-4, 50.0
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if run("clean", mid, 1.0)["gold"][-1] < CLEAN_PRIMARY else (lo, mid)
    lr_s = float(np.sqrt(lo * hi))
    lo, hi = 1e-3, 1e5
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        lo, hi = (
            (mid, hi) if first_half(run("exploit", lr_s, mid)) > EXPLOIT_HALF_STEP else (lo, mid)
        )
    lr_b = float(np.sqrt(lo * hi))
    print(f"fitted lr_s={lr_s:.4g} lr_b={lr_b:.4g}", flush=True)
    clean_final = run("clean", lr_s, lr_b)["gold"][-1]
    preds: dict[str, Any] = {"fit": {"lr_s": lr_s, "lr_b": lr_b, "n_prompts": len(pairs)}}
    for name in specs:
        o = run(name, lr_s, lr_b)
        status = "prediction before results" if origin[name] == "E012" else "post hoc (observed)"
        preds[name] = {"origin": origin[name], "status": status, "gold_final": o["gold"][-1],
                       "harm": clean_final - o["gold"][-1], "first_fpr_half": first_half(o),
                       "pred_harm_ge_0.25": clean_final - o["gold"][-1] >= 0.25}  # fmt: skip
        print(
            f"{name:10s} {origin[name]:5s} harm {preds[name]['harm']:+.3f} "
            f"half@{preds[name]['first_fpr_half']} {status}",
            flush=True,
        )
    Path(argv[1]).write_text(json.dumps(preds, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
