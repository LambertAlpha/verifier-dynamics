"""E004a Stage 1 FINAL held-out, pre-unseal integrity check (brief step 1). Fails closed.

Checks: clean tree; HEAD; frozen predictor-config sha (and tracked in git); design panel and
Stage 1 config sha; pytest / ruff / mypy exit codes; frozen feature names, model settings and
thresholds equal the code; the frozen linear models refit on design reproduce every recorded
penalty; the frozen-path sources are unchanged since the freeze commit; no held-out / shift panel,
run or outcome exists anywhere (working tree or git history); the held-out loader is sealed
without the approval file; the seed-tree extension keeps the design seeds and gives disjoint
held-out seeds. Writes integrity.json + integrity.md into a run directory.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0b_runs as s0b  # noqa: E402
import stage1_analysis as an  # noqa: E402
import stage1_heldout_analysis as hla  # noqa: E402

from vdyn import provenance  # noqa: E402
from vdyn.e004 import features as fe  # noqa: E402
from vdyn.e004 import heldout as hd  # noqa: E402
from vdyn.e004 import panel0b as pn  # noqa: E402
from vdyn.e004 import predict as pr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e004" / "e004a_stage1.toml"
FREEZE_COMMIT = "341851a"
FROZEN_PATH_FILES = [
    "src/vdyn/e004/audit.py", "src/vdyn/e004/features.py", "src/vdyn/e004/predict.py",
    "src/vdyn/e004/dynamics.py", "src/vdyn/e004/toy.py", "src/vdyn/e004/outcomes.py",
    "experiments/e004/stage1_runs.py", "experiments/e004/stage1_analysis.py",
    "configs/e004/e004a_stage1.toml", "configs/e004/design_panel_0b.json",
]  # fmt: skip


def sh(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=REPO, capture_output=True, text=True)


def main() -> int:
    checks: dict[str, Any] = {}
    fail: list[str] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            fail.append(name)

    status = sh("git", "status", "--porcelain").stdout
    check("1_working_tree_clean", status == "", status)
    head = sh("git", "rev-parse", "HEAD").stdout.strip()
    check("2_head_commit", bool(head), head)
    frozen_path = hla.FROZEN
    fsha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    tracked = sh("git", "ls-files", "--error-unmatch", str(frozen_path.relative_to(REPO)))
    check("3_predictors_frozen_sha", fsha == hla.FROZEN_SHA and tracked.returncode == 0, fsha)
    cfg = provenance.load_config(CONFIG)
    psha = hashlib.sha256((REPO / cfg["panel"]).read_bytes()).hexdigest()
    csha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    frozen = json.loads(frozen_path.read_text())
    check("4_panel_and_config_sha", psha == cfg["panel_sha256"] and csha == frozen["config_sha256"],
          {"panel": psha, "config": csha})  # fmt: skip
    for name, cmd in (("5a_pytest", ["uv", "run", "pytest", "-q", "-p", "no:cacheprovider"]),
                      ("5b_ruff", ["uv", "run", "ruff", "check", "src", "tests", "experiments"]),
                      ("5c_ruff_format", ["uv", "run", "ruff", "format", "--check", "src", "tests",
                                          "experiments"]),
                      ("5d_mypy", ["uv", "run", "mypy", "--cache-dir", "/dev/null"])):  # fmt: skip
        r = sh(*cmd)
        check(name, r.returncode == 0, (r.stdout.strip().splitlines() or [""])[-1])
    # 6. frozen horizons, features, models, thresholds
    names_ok = all(frozen["levels"][lv] == fe.names(lv) for lv in frozen["levels"])
    models_ok = (frozen["models"]["penalty_grid"] == pr.GRID.tolist()
                 and frozen["outer_cv"]["k"] == pr.K_OUTER
                 and frozen["outer_cv"]["seed"] == pr.FOLD_SEED
                 and all(frozen["models"]["secondary"][k] == v
                         for k, v in pr.GBM.items()))  # fmt: skip
    tau_ok = all(set(frozen["warning"]["tau_adam"][lv]) == {f"{h:g}" for h in an.H[1:]}
                 for lv in an.PRIMARY)  # fmt: skip
    horizons_ok = cfg["horizons"]["fractions"] == an.H and cfg["horizons"]["primary"] == an.H_STAR
    check("6a_features_models_thresholds_horizons", names_ok and models_ok and tau_ok
          and horizons_ok, {"names": names_ok, "models": models_ok, "tau": tau_ok,
                            "horizons": horizons_ok})  # fmt: skip
    diff = sh("git", "diff", "--stat", FREEZE_COMMIT, "HEAD", "--", *FROZEN_PATH_FILES).stdout
    check("6b_frozen_path_sources_unchanged_since_freeze", diff.strip() == "", diff)
    d = an.load(hla.DESIGN_RUNS)
    tab = an.table(d, "adam")
    sets = an.feature_sets(tab, np.random.default_rng(hd.resample_seeds()[0]))
    models = hla.fit_many(hla.design_jobs(tab, sets, tuple(an.STATIC + an.DYNAMIC), "linear"))
    bad = {f"{k[0]}|{k[1]:g}|{k[2]}": p for k, (_, p) in models.items()
           if p != frozen["full_design_penalties"][f"{k[0]}|{k[1]:g}|{k[2]}"]}  # fmt: skip
    check("6c_frozen_models_refit_reproduce_penalties",
          not bad and len(models) == len(frozen["full_design_penalties"]),
          {"n_models": len(models), "mismatches": bad})  # fmt: skip
    # 7. no held-out outcome exists
    paths = ["configs/e004/test_panel_0b.json", "configs/e004/shift_panel_0b.json",
             "configs/e004/HELDOUT_APPROVED"]  # fmt: skip
    exists = [p for p in paths if (REPO / p).exists()]
    exists += [str(p.relative_to(REPO)) for p in (REPO / "results").glob("E004a-stage1-heldout*")]
    hist = sh("git", "log", "--all", "--oneline", "--", *paths, "results/E004a-stage1-heldout-runs",
              "results/E004a-stage1-heldout-analysis").stdout.strip()  # fmt: skip
    check("7_no_heldout_outcomes_computed", not exists and hist == "",
          {"existing": exists, "git_history": hist})  # fmt: skip
    # 8. seed tree, sealed loader, approval mechanism
    try:
        hd.test_panel(hd.APPROVAL, hla.FROZEN_SHA, n_per=1)
        sealed = False
    except PermissionError:
        sealed = True
    design_seeds = s0b.seed_table(len(d["structs"]))["prim_ver"]
    same = all([s.spawn_key for s in a] == [s.spawn_key for s in b] for a, b in
               zip(hd.run_seeds("prim_ver", "design", len(design_seeds)), design_seeds,
                   strict=True))  # fmt: skip
    keys = {sp: {s.spawn_key for ss in hd.run_seeds("prim_ver", sp, 768) for s in ss}
            for sp in ("design", "test", "shift")}  # fmt: skip
    disjoint = not (keys["design"] & keys["test"] or keys["design"] & keys["shift"]
                    or keys["test"] & keys["shift"])  # fmt: skip
    check("8_seed_tree_loader_approval", sealed and same and disjoint,
          {"sealed_without_approval": sealed, "design_seeds_unchanged": same,
           "splits_disjoint": disjoint, "approval_file": str(hd.APPROVAL.relative_to(REPO)),
           "test_stream": hd.TEST_STREAM, "shift_stream": hd.SHIFT_STREAM,
           "offsets": hd.OFFSET})  # fmt: skip
    out_dir = provenance.create_run_dir(REPO / "results", "E004a-stage1-integrity", REPO)
    provenance.write_metadata(out_dir, "E004a-stage1-integrity", CONFIG, REPO,
                              extra={"split": "none (pre-unseal)", "root_seed": pn.ROOT_SEED,
                                     "panel_sha256": cfg["panel_sha256"],
                                     "predictors_frozen_sha256": fsha, "head": head})  # fmt: skip
    verdict = "PASS" if not fail else "STOP"
    (out_dir / "integrity.json").write_text(json.dumps({"verdict": verdict, "head": head,
                                                        "failed": fail, "checks": checks},
                                                       indent=1, default=str) + "\n")  # fmt: skip
    lines = [f"# E004a Stage 1 pre-unseal integrity check: {verdict}", "", f"HEAD `{head}`", ""]
    lines += [f"- {'PASS' if v['ok'] else 'FAIL'} {k}" for k, v in checks.items()]
    (out_dir / "integrity.md").write_text("\n".join(lines) + "\n")
    print(verdict, fail)
    print(f"run directory: {out_dir.relative_to(REPO)}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
