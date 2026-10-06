"""E012 audits (research/paper/e012_protocol.md). Same audit samples and rule as E006.

Usage:
  e012_audit.py calibrate   region fills; comparator specs copied -> configs/e012/arms.json (once)
  e012_audit.py verify      per-arm FPR on the verification audit; exit 3 if any arm fails
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e006"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e005b"))
import audit as au6  # noqa: E402
import common as cm  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e005b import matching as mt  # noqa: E402
from vdyn.e005b import task as tk  # noqa: E402
from vdyn.e006 import verifiers as vf  # noqa: E402

CFG = cm.REPO / "configs" / "e012" / "e012.toml"
ARMS = cm.REPO / "configs" / "e012" / "arms.json"


def specs(
    ec: dict[str, Any],
    f0: float,
    parsed: list[tuple[bool, int | None]],
    G: np.ndarray,
    rep: list[tuple[int, int]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """randfp30 and del50 as specified; cov65 / far40 / key111half / hard75 filled to f0 with
    fresh coins (fill = (f0 - share) / (1 - share), share = deterministic acceptance share of
    wrong calibration samples)."""
    wrong = np.flatnonzero(G == 0)
    out: dict[str, dict[str, Any]] = {
        "randfp30": vars(vf.Spec("randfp", r=0.30)),
        "del50": vars(vf.Spec("delfrac", c=0.5)),
    }
    info: dict[str, Any] = {"n_wrong": len(wrong)}
    for arm, probe in (("cov65", vf.Spec("cov", c=0.65)), ("far40", vf.Spec("farkey", dist=40)),
                       ("key111half", vf.Spec("keyset", values=(111,), c=0.5)),
                       ("hard75", vf.Spec("covcat", cats=("three-digit",), c=0.75))):  # fmt: skip
        rng = np.random.default_rng(0)
        share = sum(vf.reward(probe, *rep[i], S_TOKS[i], rng) for i in wrong) / len(wrong)
        r = (f0 - share) / (1 - share)
        out[arm] = vars(vf.Spec(**{**vars(probe), "r": r}))
        info[arm] = {"share": share, "fill": r}
    for d in out.values():
        d["cats"], d["values"] = list(d["cats"]), list(d["values"])
    return out, info


S_TOKS: list[list[int]] = []


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    if mode not in ("calibrate", "verify"):
        print(__doc__)
        return 2
    cfg = cm.load_config()
    a = provenance.load_config(au6.MATCHED)["audit"]
    ec = provenance.load_config(CFG)
    f0 = ec["f0"]
    if provenance.git_state(cm.REPO)["dirty"]:
        print("STOP: working tree is dirty")
        return 1
    if mode == "calibrate" and ARMS.exists():
        print("STOP: configs/e012/arms.json exists")
        return 1
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(ARMS)], cwd=cm.REPO,
                             capture_output=True).returncode == 0  # fmt: skip
    if mode == "verify" and not tracked:
        print("STOP: configs/e012/arms.json must be committed first")
        return 1
    name = f"E012-{'calibration' if mode == 'calibrate' else 'verification'}"
    run_dir = provenance.create_run_dir(cm.REPO / "results", name, cm.REPO)
    provenance.write_metadata(run_dir, name, CFG, cm.REPO, extra=cm.run_extra(cfg, mode=mode))
    S = au6.audit_samples(cfg, a, mode)
    S_TOKS[:] = S["toks"]
    rep = [p for p in S["pairs"] for _ in range(S["group"])]
    parsed = [tk.parse_completion(t) for t in S["toks"]]
    G = np.array([tk.gold_reward(x, y, *pv) for (x, y), pv in zip(rep, parsed, strict=True)])
    cats = [tk.category(x, y) for x, y in rep]
    out: dict[str, Any] = {"mode": mode, "prompt_sha256": S["prompt_sha256"]}
    verdicts: dict[str, Any] = {}
    if mode == "calibrate":
        sp, info = specs(ec, f0, parsed, G, rep)
        rec = {"f0": f0, "specs": sp, "info": info, "rare_value": 57,
               "calibration_run": str(run_dir.relative_to(cm.REPO)),
               "e012_config_sha256": au6.sha(CFG)}  # fmt: skip
        ARMS.write_text(json.dumps(rec, indent=1) + "\n")
        out["arms"] = rec
        print(json.dumps(info, indent=1))
    else:
        rec = json.loads(ARMS.read_text())
        pid = np.repeat(np.arange(len(S["pairs"])), S["group"])
        crng = np.random.default_rng(a["verification_coin_seed"])
        out["arms"] = {}
        for arm, d in rec["specs"].items():
            spec = vf.Spec(**d)
            V = np.array([vf.reward(spec, x, y, t, crng)
                          for (x, y), t in zip(rep, S["toks"], strict=True)])  # fmt: skip
            brng = np.random.default_rng(a["bootstrap_seed"])
            r = mt.audit_rates(pid, np.array(cats), V, G, brng, a["bootstrap_resamples"])
            out["arms"][arm] = r
            fpr = r["overall"]["fpr"]
            ok = r["overall"]["fnr"] == 0  # E012: static FPRs are deliberately not matched
            verdicts[arm] = {"fpr": fpr, "abs_diff": abs(fpr - f0), "pass": bool(ok)}
            print(f"{arm:8s} FPR {fpr:.4f} {[round(x, 4) for x in r['overall']['fpr_ci']]} "
                  f"{'PASS' if ok else 'FAIL'}")  # fmt: skip
        (run_dir / "verdict.json").write_text(json.dumps(
            {"arms": verdicts, "arms_file_sha256": au6.sha(ARMS), "tolerance": a["tolerance"]},
            indent=1) + "\n")  # fmt: skip
    (run_dir / "audit.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"run directory: {run_dir.relative_to(cm.REPO)}")
    if mode == "verify" and not all(v["pass"] for v in verdicts.values()):
        print("STOP: an arm failed matching; it is not trained")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
