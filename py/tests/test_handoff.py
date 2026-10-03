"""What survives between runs: the branch's commits, the working tree, and
HANDOFF.md all reach the first turn of a resumed run.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import build
from djed.handoff import (
    render_handoff,
    repo_state,
    resume_preamble,
    write_repos,
)


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace whose repo has an origin, so "new work" is measurable."""
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q", "-b", "main")
    (origin / "a.py").write_text("x = 1\n")
    git(origin, "add", "a.py")
    git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")

    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(ws / "src" / "demo")],
        check=True, capture_output=True,
    )
    repo = ws / "src" / "demo"
    git(repo, "config", "user.name", "djed builder")
    git(repo, "config", "user.email", "djed@localhost")
    git(repo, "checkout", "-q", "-b", "djed/t1")
    return ws


def test_no_prior_run_yields_no_preamble(workspace: Path):
    """Returns text only when there is something to resume from."""
    assert resume_preamble(workspace) == ""


def test_commits_from_a_previous_run_reach_the_preamble(workspace: Path):
    repo = workspace / "src" / "demo"
    (repo / "a.py").write_text("x = 2\n")
    git(repo, "add", "a.py")
    git(repo, "commit", "-qm", "fix: bump x")

    out = resume_preamble(workspace)
    assert "fix: bump x" in out
    assert "djed/t1" in out
    assert "a.py" in out          # the diffstat
    assert "1 file changed" in out


def test_uncommitted_work_is_reported_too(workspace: Path):
    """A run that blocked mid-edit leaves changes that are not in any commit."""
    repo = workspace / "src" / "demo"
    (repo / "b.py").write_text("y = 1\n")
    out = resume_preamble(workspace)
    assert "Uncommitted changes" in out
    assert "b.py" in out


def test_the_previous_handoff_is_carried_forward(workspace: Path):
    (workspace / "HANDOFF.md").write_text(
        "# Handoff\n\nEnded: blocked\nReason: missing_toolchain - ruff\n"
    )
    out = resume_preamble(workspace)
    assert "why it stopped" in out
    assert "missing_toolchain - ruff" in out


def test_a_handoff_alone_is_enough_to_resume(workspace: Path):
    """A run can block before committing anything; that is still a resume."""
    (workspace / "HANDOFF.md").write_text("# Handoff\n\nEnded: blocked\n")
    out = resume_preamble(workspace)
    assert out
    assert "It committed nothing." in out


def test_state_reads_the_branch_and_counts_only_new_work(workspace: Path):
    repo = workspace / "src" / "demo"
    (repo / "a.py").write_text("x = 3\n")
    git(repo, "add", "a.py")
    git(repo, "commit", "-qm", "one")
    s = repo_state(repo)
    assert s.branch == "djed/t1"
    # The base commit came from the clone source and is not this run's work.
    assert len(s.commits) == 1
    assert s.touched


def test_a_broken_repo_degrades_rather_than_raising(tmp_path: Path):
    """A failing git command yields no preamble, not an error."""
    ws = tmp_path / "ws"
    (ws / "src" / "notrepo").mkdir(parents=True)
    assert write_repos(ws) == []
    assert resume_preamble(ws) == ""


def test_missing_workspace_is_not_an_error(tmp_path: Path):
    assert resume_preamble(tmp_path / "nope") == ""


# --- the handoff itself ---------------------------------------------------


def test_a_blocked_handoff_names_the_reason_for_the_relaunch():
    text = render_handoff(
        task_id="t1", kind="blocked", iterations=2, requests=9,
        input_tokens=100, output_tokens=10, elapsed=12.0,
        summary="Wrote the change, could not test.",
        reason="missing_toolchain", detail="ruff",
    )
    assert "Ended: blocked" in text
    assert "missing_toolchain - ruff" in text
    assert "(9 model requests)" in text


def test_a_crash_still_produces_a_handoff():
    text = render_handoff(
        task_id="t1", kind="error", iterations=1, requests=3,
        input_tokens=1, output_tokens=1, elapsed=1.0,
        summary="", error="boom",
    )
    assert "Error: boom" in text
    assert "without a summary" in text


# --- the driver seeds turn one with it ------------------------------------


def test_the_preamble_leads_the_seed_prompt():
    """The resume preamble comes before the task."""
    seed = build._seed_prompt("WS", "TASK", "PRIOR STATE")
    assert seed.index("PRIOR STATE") < seed.index("TASK")
    assert "WS" in seed


def test_a_first_run_carries_no_resume_section():
    seed = build._seed_prompt("WS", "TASK")
    assert "Resuming" not in seed
