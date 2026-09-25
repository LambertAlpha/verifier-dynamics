"""Run directories and metadata: every experiment output records its config and git commit."""

import json
import platform
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("numpy", "scipy", "torch", "matplotlib", "verifier-dynamics")


def git_state(repo: Path) -> dict[str, Any]:
    """Current commit and whether the working tree has uncommitted changes (None outside git)."""

    def run(*args: str) -> str | None:
        result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(run("status", "--porcelain"))}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def create_run_dir(
    results_root: Path, experiment_id: str, repo: Path, now: datetime | None = None
) -> Path:
    """results_root/<experiment_id>/<UTC stamp>_<short sha>[-dirty]; fails if it already exists."""
    state = git_state(repo)
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    sha = state["commit"][:7] if state["commit"] else "nogit"
    suffix = "-dirty" if state["dirty"] else ""
    run_dir = results_root / experiment_id / f"{stamp}_{sha}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def write_metadata(
    run_dir: Path,
    experiment_id: str,
    config_path: Path,
    repo: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy the config into the run directory and write meta.json; returns the metadata."""
    shutil.copyfile(config_path, run_dir / "config.toml")
    meta = {
        "experiment_id": experiment_id,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(repo),
        "config": load_config(config_path),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        **(extra or {}),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
