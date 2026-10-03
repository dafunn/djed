"""What survives between runs.

A blocked run is resumed from the commits on its branch, the working tree, and
HANDOFF.md, not from a replayed transcript.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

HANDOFF = "HANDOFF.md"
# Enough to orient a resumed run without re-reading the whole change.
MAX_DIFFSTAT_LINES = 40
MAX_HANDOFF_CHARS = 4000


def _git(repo: Path, *args: str) -> str:
    """Run git in `repo`, returning stdout; a failure returns empty."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def write_repos(workspace: Path) -> list[Path]:
    """The checkouts under `src/`, in a stable order."""
    src = workspace / "src"
    if not src.is_dir():
        return []
    return sorted(d for d in src.iterdir() if (d / ".git").exists())


@dataclass(frozen=True)
class RepoState:
    name: str
    branch: str
    commits: tuple[str, ...]
    diffstat: str
    dirty: tuple[str, ...]

    @property
    def touched(self) -> bool:
        return bool(self.commits or self.dirty)


def repo_state(repo: Path) -> RepoState:
    """What a previous run left behind in one checkout.

    "New" is computed against the remotes rather than a named base branch: the
    clone's origin is the cache it came from, so anything not in a remote is
    work this task produced.
    """
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    log = _git(repo, "log", "--oneline", "HEAD", "--not", "--remotes")
    commits = tuple(line for line in log.splitlines() if line)

    diffstat = ""
    if commits:
        oldest = _git(repo, "rev-list", "HEAD", "--not", "--remotes")
        first = oldest.splitlines()[-1] if oldest.splitlines() else ""
        if first:
            diffstat = _git(repo, "diff", "--stat", f"{first}^..HEAD")
    stat_lines = diffstat.splitlines()[:MAX_DIFFSTAT_LINES]

    status = _git(repo, "status", "--short")
    return RepoState(
        name=repo.name,
        branch=branch,
        commits=commits,
        diffstat="\n".join(stat_lines),
        dirty=tuple(line for line in status.splitlines() if line),
    )


def resume_preamble(workspace: Path) -> str:
    """Turn one of a resumed run: what the last run did, and why it stopped.

    Returns "" when there is nothing to resume from, so the caller can treat a
    first run and a resumed one the same way.
    """
    states = [repo_state(r) for r in write_repos(workspace)]
    previous = ""
    handoff = workspace / HANDOFF
    if handoff.is_file():
        try:
            previous = handoff.read_text(encoding="utf-8").strip()[:MAX_HANDOFF_CHARS]
        except OSError:
            previous = ""

    if not previous and not any(s.touched for s in states):
        return ""

    out = [
        "## Resuming an earlier run",
        "",
        "You have been here before. This is what the previous run left; carry on "
        "from it rather than starting again.",
    ]
    if previous:
        out += ["", "### Its own account of why it stopped", "", previous]

    for s in states:
        out += ["", f"### {s.name} (branch `{s.branch}`)"]
        if s.commits:
            out += ["", "Commits it made:", ""]
            out += [f"- {c}" for c in s.commits]
        else:
            out += ["", "It committed nothing."]
        if s.diffstat:
            out += ["", "```", s.diffstat, "```"]
        if s.dirty:
            out += ["", "Uncommitted changes still in the tree:", ""]
            out += [f"- {d}" for d in s.dirty]
    return "\n".join(out)


def render_handoff(
    task_id: str,
    kind: str,
    iterations: int,
    requests: int,
    input_tokens: int,
    output_tokens: int,
    elapsed: float,
    summary: str,
    reason: str = "",
    detail: str = "",
    error: str = "",
) -> str:
    """The run's account of how it ended, read by the next run. Written for every
    ending, including a crash.
    """
    lines = ["# Handoff", "", f"Task: {task_id}", f"Ended: {kind}"]
    if reason:
        lines.append(f"Reason: {reason} - {detail}")
    if error:
        lines.append(f"Error: {error}")
    lines += [
        f"Iterations: {iterations} ({requests} model requests)",
        f"Tokens: {input_tokens} in / {output_tokens} out",
        f"Elapsed: {elapsed:.0f}s",
        "",
        "## Summary",
        "",
        summary.strip() or "(the run ended without a summary)",
        "",
    ]
    return "\n".join(lines)
