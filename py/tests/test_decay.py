"""Bounding what the transcript re-sends: the two passes, and that the result is
still a valid transcript.
"""
from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from djed.decay import (
    decay,
    elide_tool_results,
    keep_thinking,
    result_budget,
    strip_thinking,
)


def _call(name="fs_read", args=None, cid="c1"):
    return ToolCallPart(tool_name=name, args=args or {"path": "/x.py"}, tool_call_id=cid)


def _ret(content, name="fs_read", cid="c1"):
    return ToolReturnPart(tool_name=name, content=content, tool_call_id=cid)


# --- thinking ------------------------------------------------------------


def test_thinking_is_dropped_and_its_siblings_survive():
    """Reasoning blocks are dropped; tool calls stay."""
    msgs = [
        ModelResponse(parts=[ThinkingPart(content="x" * 70_000), TextPart(content="ok"), _call()]),
    ]
    out = strip_thinking(msgs)
    parts = out[0].parts
    assert not any(isinstance(p, ThinkingPart) for p in parts)
    assert any(isinstance(p, TextPart) for p in parts)
    assert any(isinstance(p, ToolCallPart) for p in parts)


def test_a_response_of_nothing_but_thinking_is_dropped_not_emptied():
    """No empty ModelResponse is left behind; a strict server rejects one."""
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="go")]),
        ModelResponse(parts=[ThinkingPart(content="...")]),
    ]
    out = strip_thinking(msgs)
    assert len(out) == 1
    assert isinstance(out[0], ModelRequest)


def test_requests_are_untouched_by_the_thinking_pass():
    msgs = [ModelRequest(parts=[UserPromptPart(content="the task")])]
    assert strip_thinking(msgs) == msgs


def test_thinking_is_stripped_by_default(monkeypatch):
    monkeypatch.delenv("DJED_KEEP_THINKING", raising=False)
    assert keep_thinking() is False
    msgs = [ModelResponse(parts=[ThinkingPart(content="x" * 100), TextPart(content="hi")])]
    assert not any(
        isinstance(p, ThinkingPart) for m in decay(msgs) for p in getattr(m, "parts", [])
    )


def test_keeping_thinking_can_be_asked_for(monkeypatch):
    monkeypatch.setenv("DJED_KEEP_THINKING", "1")
    assert keep_thinking() is True
    msgs = [ModelResponse(parts=[ThinkingPart(content="x" * 100), TextPart(content="hi")])]
    assert any(
        isinstance(p, ThinkingPart) for m in decay(msgs) for p in getattr(m, "parts", [])
    )


# --- tool results --------------------------------------------------------


def _conversation(sizes):
    """One call/result pair per size, oldest first."""
    msgs = [ModelRequest(parts=[UserPromptPart(content="the task")])]
    for i, n in enumerate(sizes):
        cid = f"c{i}"
        msgs.append(ModelResponse(parts=[_call(args={"path": f"/f{i}.py"}, cid=cid)]))
        msgs.append(ModelRequest(parts=[_ret("R" * n, cid=cid)]))
    return msgs


def _payloads(msgs):
    return [
        p.content
        for m in msgs
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    ]


def test_the_newest_results_are_the_ones_kept():
    """The newest results are kept."""
    out = elide_tool_results(_conversation([100, 100, 100]), budget=250)
    got = _payloads(out)
    assert got[0].startswith("[elided:")
    assert got[1] == "R" * 100
    assert got[2] == "R" * 100


def test_an_elided_result_names_the_call_that_produced_it():
    """The stub names the call that produced the elided result."""
    out = elide_tool_results(_conversation([500, 10]), budget=20)
    stub = _payloads(out)[0]
    assert "fs_read" in stub
    assert "/f0.py" in stub
    assert "500 characters" in stub
    assert "re-run" in stub


def test_eliding_keeps_every_call_answered():
    """Calls and results stay paired."""
    msgs = _conversation([500, 500, 500])
    out = elide_tool_results(msgs, budget=0)
    answered = {
        p.tool_call_id
        for m in out
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    }
    asked = {
        p.tool_call_id
        for m in out
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ToolCallPart)
    }
    assert asked == answered
    assert len(asked) == 3


def test_a_budget_that_fits_everything_changes_nothing():
    msgs = _conversation([100, 100])
    assert elide_tool_results(msgs, budget=10_000) == msgs


def test_the_task_prompt_is_never_touched():
    """The task prompt, the first message, is never elided."""
    out = elide_tool_results(_conversation([9_000, 9_000]), budget=0)
    assert any(
        isinstance(p, UserPromptPart) and p.content == "the task"
        for m in out
        if isinstance(m, ModelRequest)
        for p in m.parts
    )


def test_eliding_actually_shrinks_the_transcript():
    big = _conversation([20_000, 20_000, 20_000])
    small = elide_tool_results(big, budget=20_000)
    assert sum(len(c) for c in _payloads(small)) < sum(len(c) for c in _payloads(big)) / 2


def test_non_string_results_are_left_alone():
    """Structured content is not elided."""
    msgs = [
        ModelResponse(parts=[_call()]),
        ModelRequest(parts=[_ret({"rows": 3})]),
    ]
    assert elide_tool_results(msgs, budget=0) == msgs


# --- budget config -------------------------------------------------------


def test_the_budget_is_configurable_and_survives_nonsense(monkeypatch):
    monkeypatch.setenv("DJED_RESULT_BUDGET", "1234")
    assert result_budget() == 1234
    monkeypatch.setenv("DJED_RESULT_BUDGET", "not a number")
    assert result_budget() == 60_000
    monkeypatch.setenv("DJED_RESULT_BUDGET", "-5")
    assert result_budget() == 0
