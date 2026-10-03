"""The build driver: how a run stops, and what it leaves behind. The stop tools
record on `Deps.outcome`, so `session.py` is unchanged by build runs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import build
import fake_model
from djed.session import TurnResult
from djed.spec import Budget
from djed.tools import BLOCKED_REASONS, Deps, Outcome, task_blocked, task_complete
from tests.test_kwim import StubKwim


class Ctx:
    """Minimal stand-in for RunContext: the stop tools touch only `deps`."""

    def __init__(self, deps):
        self.deps = deps


def _deps() -> Deps:
    return Deps(kwim=StubKwim(), session_id="s")


# --- the stop tools ------------------------------------------------------


async def test_task_complete_records_the_outcome():
    deps = _deps()
    assert deps.outcome is None
    await task_complete(Ctx(deps), summary="Renamed the function.")
    assert deps.outcome == Outcome(kind="complete", summary="Renamed the function.")


@pytest.mark.parametrize("reason", BLOCKED_REASONS)
async def test_task_blocked_records_every_valid_reason(reason):
    deps = _deps()
    await task_blocked(Ctx(deps), reason=reason, detail="cargo", summary="Got partway.")
    assert deps.outcome is not None
    assert deps.outcome.kind == "blocked"
    assert deps.outcome.reason == reason


async def test_an_unknown_block_reason_does_not_end_the_run():
    """A wrong reason comes back as content, so the model can correct it; the run does
    not end.
    """
    deps = _deps()
    out = await task_blocked(Ctx(deps), reason="because", detail="d", summary="s")
    assert deps.outcome is None
    assert "Unknown reason" in out
    for reason in BLOCKED_REASONS:
        assert reason in out


# --- budget backstop -----------------------------------------------------


def test_budget_stops_on_each_limit_and_not_before():
    budget = Budget(max_iterations=3, max_tokens=100, wall_clock_s=10_000)
    assert build._budget_stop(build.RunState(iterations=2), budget) is None

    state = build.RunState(iterations=3)
    assert "iteration limit" in build._budget_stop(state, budget)

    state = build.RunState(iterations=1, input_tokens=60, output_tokens=40)
    assert "token limit" in build._budget_stop(state, budget)


def test_wall_clock_limit_trips_on_its_own():
    budget = Budget(max_iterations=99, max_tokens=10**9, wall_clock_s=0.0001)
    state = build.RunState(iterations=1)
    state.started -= 1.0
    assert "wall-clock limit" in build._budget_stop(state, budget)


def test_a_zero_token_budget_means_no_ceiling():
    """`max_tokens` defaults to 0, no ceiling."""
    budget = Budget(max_iterations=99, max_tokens=0, wall_clock_s=10_000)
    state = build.RunState(iterations=1, input_tokens=10**9, output_tokens=10**9)
    assert build._budget_stop(state, budget) is None


def test_budget_counts_input_and_output_together():
    state = build.RunState(iterations=1, input_tokens=60, output_tokens=41)
    assert state.total_tokens == 101


# --- handoff -------------------------------------------------------------


def test_an_undeclared_summary_keeps_the_end_of_the_turn_not_the_start():
    """The handoff summary keeps the tail of the turn's text."""
    text = "I'll start by reading everything. " + ("filler. " * 400) + "All 11 tests pass; the docs still need updating."
    assert len(text) > build.SUMMARY_TAIL_CHARS
    got = build._tail(text)
    assert "All 11 tests pass" in got
    assert "I'll start by reading" not in got
    assert len(got) <= build.SUMMARY_TAIL_CHARS + 10


def test_a_short_summary_is_left_alone():
    assert build._tail("  done, all green  ") == "done, all green"





def test_handoff_records_a_block_reason_for_the_relaunch(tmp_path: Path):
    outcome = Outcome(
        kind="blocked", summary="Wrote the tool, could not test.",
        reason="missing_toolchain", detail="cargo",
    )
    p = tmp_path / "HANDOFF.md"
    build.write_handoff(p, "t1", "blocked", build.RunState(iterations=4), outcome)
    text = p.read_text()
    assert "Ended: blocked" in text
    assert "missing_toolchain - cargo" in text
    assert "Wrote the tool, could not test." in text


def test_the_written_handoff_keeps_the_end_of_an_undeclared_turn(tmp_path: Path):
    """HANDOFF.md on disk carries the tail, not just the helper's return value."""
    state = build.RunState(
        iterations=1,
        last_text="I'll start by reading everything. "
        + ("filler. " * 400)
        + "All 11 tests pass; the docs still need updating.",
    )
    p = tmp_path / "HANDOFF.md"
    build.write_handoff(p, "t1", "error", state, None)
    text = p.read_text()
    assert "All 11 tests pass" in text
    assert "I'll start by reading" not in text


def test_handoff_is_written_even_when_there_is_no_outcome(tmp_path: Path):
    """A crashed run still writes a handoff."""
    p = tmp_path / "HANDOFF.md"
    state = build.RunState(iterations=1, error="boom", last_text="got here")
    build.write_handoff(p, "t1", "error", state, None)
    text = p.read_text()
    assert "Ended: error" in text
    assert "Error: boom" in text
    assert "got here" in text


# --- the loop, end to end ------------------------------------------------

SPEC = """+++
task_id = "smoke"

[[repo]]
repo = "acme/demo"
mode = "write"

[budget]
max_iterations = 5
+++

# Task

Rename the greeting.
"""


def _run(tmp_path, monkeypatch, spec_text=SPEC, argv_extra=()):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "WORKSPACE.md").write_text("# Workspace\n\nTask: smoke\n")
    spec = tmp_path / "task.md"
    spec.write_text(spec_text)
    monkeypatch.setattr(build, "build_kwim_client", lambda http: StubKwim())
    monkeypatch.setattr(
        "sys.argv",
        ["build.py", str(spec), "--workspace", str(ws), "--fake", "--no-mcp", *argv_extra],
    )
    return ws


async def test_a_declared_outcome_ends_the_run_and_writes_a_handoff(tmp_path, monkeypatch):
    ws = _run(tmp_path, monkeypatch)
    code = await build.main()
    assert code == build.EXIT["complete"] == 0
    handoff = (ws / "HANDOFF.md").read_text()
    assert "Ended: complete" in handoff
    # The stub declares inside its first turn, so the loop never comes round.
    assert "Iterations: 1" in handoff


async def test_a_turn_of_prose_ends_the_run_for_a_human(tmp_path, monkeypatch):
    """Text with no tool call and no declared outcome ends the run."""
    ws = _run(tmp_path, monkeypatch)
    monkeypatch.setattr("fake_model.scripted_builder", fake_model.scripted_reviewer)
    code = await build.main()
    assert code == build.EXIT["review"] == 4
    handoff = (ws / "HANDOFF.md").read_text()
    assert "Ended: review" in handoff
    # The model's text is the payload of this ending and goes into the handoff.
    assert "contradicts itself" in handoff


async def test_the_request_ceiling_resumes_instead_of_ending_the_run(tmp_path, monkeypatch):
    """A turn cut off at its request ceiling is resumed in a new turn."""
    spec = SPEC.replace(
        "max_iterations = 5",
        "max_iterations = 3\nmax_requests_per_turn = 2",
    )
    ws = _run(tmp_path, monkeypatch, spec_text=spec)
    monkeypatch.setattr("fake_model.scripted_builder", fake_model.scripted_busy)
    code = await build.main()
    # Ends on the iteration budget, which is only reachable by resuming twice.
    assert code == build.EXIT["exhausted"] == 3
    handoff = (ws / "HANDOFF.md").read_text()
    assert "Ended: exhausted" in handoff
    assert "Iterations: 3" in handoff


async def test_the_resume_prompt_does_not_mention_the_ceiling(tmp_path, monkeypatch):
    """The resume message does not mention the request ceiling."""
    assert "interrupted" in build.RESUME
    assert "continue" in build.RESUME.lower()
    assert "request" not in build.RESUME.lower()
    assert "limit" not in build.RESUME.lower()


async def test_a_bad_spec_fails_before_any_model_is_built(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, spec_text='+++\ntask_id = "x"\n+++\n\n# T\n')
    assert await build.main() == build.EXIT["error"] == 1


async def test_a_missing_workspace_manifest_is_an_error_not_an_empty_prompt(
    tmp_path, monkeypatch
):
    ws = _run(tmp_path, monkeypatch)
    (ws / "WORKSPACE.md").unlink()
    assert await build.main() == build.EXIT["error"]


def _spy_on_agent(monkeypatch) -> dict:
    captured: dict = {}
    real = build.build_agent

    def spy(*a, **kw):
        captured["tools"] = [getattr(t, "__name__", str(t)) for t in kw["tools"]]
        captured["modules"] = [getattr(t, "__module__", "") for t in kw["tools"]]
        captured["persona"] = kw["persona"].tag
        return real(*a, **kw)

    monkeypatch.setattr(build, "build_agent", spy)
    return captured


# --- default tier selection -----------------------------------------------


def test_base_is_always_named_even_when_it_did_not_connect():
    """A disconnected base-mcp fails with resolve_selection's unknown-tier error."""
    assert build.default_tiers_spec(set()) == "base"


def test_exec_is_included_only_when_it_connected():
    assert build.default_tiers_spec({"base"}) == "base"
    assert build.default_tiers_spec({"base", "exec"}) == "base,exec"


def test_a_cluster_tier_connecting_does_not_leak_into_the_default():
    """BUILD_TIERS excludes the cluster and db tiers even when they connect."""
    assert build.default_tiers_spec({"base", "exec", "cluster", "db"}) == "base,exec"


async def test_the_builder_has_no_shell_tool_at_all(tmp_path, monkeypatch):
    """A build run has no native shell tool; execution goes through exec-mcp's
    run_task.
    """
    _run(tmp_path, monkeypatch)
    monkeypatch.delenv("DJED_AGENT", raising=False)
    captured = _spy_on_agent(monkeypatch)
    await build.main()
    assert captured["persona"] == "builder"
    assert set(captured["tools"]) == {"kwim_recall", "task_complete", "task_blocked"}


def test_no_exec_drops_the_exec_tier_but_leaves_others_alone():
    """The tier filter is tested directly: this suite's `main()` calls always pass
    `--fake` or `--no-mcp`, which skip MCP loading.
    """
    from djed.mcp import Tier

    class StubToolset:
        id = None

    tiers = [Tier("base", StubToolset(), ()), Tier("exec", StubToolset(), ())]
    assert [t.name for t in build.withhold_exec(tiers, no_exec=False)] == ["base", "exec"]
    assert [t.name for t in build.withhold_exec(tiers, no_exec=True)] == ["base"]


async def test_write_tools_are_approved_and_nothing_else_is():
    """An unknown gated tool is denied, not approved like the write tools."""
    assert await build._approve("fs_write", "{}") is True
    assert await build._approve("base__fs_delete", "{}") is True
    assert await build._approve("net_fetch", "{}") is False
    # exec-mcp's tools are not in WRITE_TOOLS, so a gated call naming one is
    # denied.
    assert await build._approve("run_task", '{"name": "test"}') is False
    assert await build._approve("exec__git_commit", "{}") is False


async def test_a_second_stop_call_keeps_the_first_outcome():
    """A second declaration in the same turn does not overwrite the first."""
    deps = _deps()
    await task_complete(Ctx(deps), summary="first")
    out = await task_complete(Ctx(deps), summary="second")
    assert deps.outcome.summary == "first"
    assert "Already recorded" in out


async def test_a_block_cannot_overwrite_a_completion():
    deps = _deps()
    await task_complete(Ctx(deps), summary="done")
    await task_blocked(Ctx(deps), reason="needs_network", detail="x", summary="s")
    assert deps.outcome.kind == "complete"


async def test_the_per_turn_request_cap_reaches_the_session(tmp_path, monkeypatch):
    """A single turn is bounded by `max_requests_per_turn`."""
    spec = SPEC.replace("max_iterations = 5", "max_iterations = 5\nmax_requests_per_turn = 9")
    _run(tmp_path, monkeypatch, spec_text=spec)
    captured: dict = {}
    real = build.Session

    def spy(*a, **kw):
        captured["limits"] = kw["usage_limits"]
        return real(*a, **kw)

    monkeypatch.setattr(build, "Session", spy)
    await build.main()
    limits = captured["limits"]
    assert limits.request_limit == 9
    # Tool calls are bounded per turn as well.
    assert limits.tool_calls_limit == 9
    # Tokens are not bounded per turn: a mid-turn token limit raises before a
    # handoff can be written. `_budget_stop` enforces the token budget between
    # turns.
    assert limits.total_tokens_limit is None


async def test_a_tripped_usage_limit_ends_the_run_exhausted_not_errored(
    tmp_path, monkeypatch
):
    """Hitting the per-turn limit is a budget ending, not a fault."""
    ws = _run(tmp_path, monkeypatch)

    class Boom:
        async def run_turn(self, *a, **kw):
            return TurnResult(error="UsageLimitExceeded: request_limit of 9")

        def interrupt(self):
            return False

    monkeypatch.setattr(build, "Session", lambda *a, **kw: Boom())
    assert await build.main() == build.EXIT["exhausted"]
    assert "Ended: exhausted" in (ws / "HANDOFF.md").read_text()
