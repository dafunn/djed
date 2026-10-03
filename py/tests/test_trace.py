"""Per-request telemetry, read from the transcript, so a failed request appears
in the trace.
"""
from __future__ import annotations

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from djed.trace import format_trace, totals, trace_requests


def resp(parts, finish=None, tin=0, tout=0):
    return ModelResponse(
        parts=parts, finish_reason=finish, usage=RequestUsage(input_tokens=tin, output_tokens=tout)
    )


def test_one_row_per_model_response():
    messages = [
        ModelRequest(parts=[UserPromptPart(content="go")]),
        resp([ToolCallPart(tool_name="fs_read", args={}, tool_call_id="a")], tin=100, tout=10),
        ModelRequest(parts=[UserPromptPart(content="next")]),
        resp([TextPart(content="done")], finish="stop", tin=200, tout=20),
    ]
    traces = trace_requests(messages)
    assert [t.index for t in traces] == [1, 2]
    assert traces[0].tool_calls == ("fs_read",)
    assert traces[1].text_chars == 4
    assert traces[1].finish_reason == "stop"


def test_totals_include_every_request():
    traces = trace_requests([resp([TextPart(content="a")], tin=100, tout=10),
                             resp([TextPart(content="b")], tin=250, tout=20)])
    assert totals(traces) == (2, 350, 30)


def test_a_thinking_only_response_is_marked_unactionable():
    """A thinking-only response, which ends a run with "token limit exceeded before
    any response was generated", appears in the trace.
    """
    traces = trace_requests([resp([ThinkingPart(content="x" * 900)], finish="length", tout=16000)])
    t = traces[0]
    assert t.actionable is False
    assert t.thinking_chars == 900
    assert t.finish_reason == "length"
    assert "thinking=900c" in t.summary


def test_an_empty_response_is_marked_unactionable():
    traces = trace_requests([resp([], finish="length")])
    assert traces[0].actionable is False
    assert traces[0].summary == "(empty)"


def test_a_response_with_a_tool_call_is_actionable():
    traces = trace_requests([resp([ToolCallPart(tool_name="x", args={}, tool_call_id="i")])])
    assert traces[0].actionable is True


def test_the_table_flags_the_row_that_produced_nothing():
    traces = trace_requests([
        resp([TextPart(content="ok")], finish="stop", tin=100, tout=5),
        resp([ThinkingPart(content="y" * 50)], finish="length", tin=17000, tout=16000),
    ])
    table = format_trace(traces)
    assert "nothing actionable" in table
    assert "17000" in table and "length" in table
    # The successful row is not flagged.
    assert table.count("nothing actionable") == 1


def test_the_table_can_show_only_the_tail():
    traces = trace_requests([resp([TextPart(content=str(i))]) for i in range(10)])
    assert len(format_trace(traces, limit=3).splitlines()) == 4  # header + 3


def test_an_empty_transcript_says_so_rather_than_rendering_a_bare_header():
    assert "no model requests" in format_trace([])
