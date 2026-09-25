import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from vdyn import provenance


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _make_repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-q")
    (path / "a.txt").write_text("a")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-q", "-m", "init")
    return path


def test_git_state_reports_commit_and_dirty_flag(tmp_path):
    repo = _make_repo(tmp_path / "repo")
    state = provenance.git_state(repo)
    assert re.fullmatch(r"[0-9a-f]{40}", str(state["commit"]))
    assert state["dirty"] is False
    (repo / "a.txt").write_text("changed")
    assert provenance.git_state(repo)["dirty"] is True


def test_git_state_outside_a_repository_is_unknown(tmp_path):
    assert provenance.git_state(tmp_path) == {"commit": None, "dirty": None}


def test_run_directory_and_metadata_record_config_and_commit(tmp_path):
    repo = _make_repo(tmp_path / "repo")
    config = tmp_path / "cfg.toml"
    config.write_text('experiment_id = "E999"\nalpha = 1.5\n')
    now = datetime(2026, 9, 24, 12, 0, 0, tzinfo=UTC)

    run_dir = provenance.create_run_dir(tmp_path / "results", "E999", repo, now=now)
    meta = provenance.write_metadata(run_dir, "E999", config, repo)

    sha = provenance.git_state(repo)["commit"]
    assert run_dir == tmp_path / "results" / "E999" / f"20260924T120000Z_{str(sha)[:7]}"
    assert (run_dir / "config.toml").read_text() == config.read_text()
    on_disk = json.loads((run_dir / "meta.json").read_text())
    assert on_disk == meta
    assert on_disk["git"]["commit"] == sha
    assert on_disk["config"] == {"experiment_id": "E999", "alpha": 1.5}
    assert {"numpy", "scipy", "torch", "matplotlib"} <= set(on_disk["packages"])


def test_dirty_tree_is_visible_in_run_directory_name(tmp_path):
    repo = _make_repo(tmp_path / "repo")
    (repo / "a.txt").write_text("changed")
    run_dir = provenance.create_run_dir(tmp_path / "results", "E999", repo)
    assert run_dir.name.endswith("-dirty")
