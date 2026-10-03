"""A scripted stub model, so the plumbing can be smoke-tested with no cluster.

Covers a tool call, a second that trips the approval gate, and a text answer.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel


def _script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    calls_made = {
        p.tool_name
        for m in messages
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ToolCallPart)
    }
    if "kwim_recall" not in calls_made:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="kwim_recall",
                    args={"query": "monitoring namespace layout", "k": 3},
                    tool_call_id="call_recall_1",
                )
            ]
        )
    if "run_command" not in calls_made:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="run_command",
                    args={"command": "echo djed-roundtrip-ok"},
                    tool_call_id="call_shell_1",
                )
            ]
        )
    return ModelResponse(
        parts=[
            TextPart(content="Round trip complete: recall, approval gate, and shell fired.")
        ]
    )


async def _script_stream(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
    """Streamed form of the same script.

    `event_stream_handler` puts every run on the streaming path, so the stub has
    to speak deltas too - text arrives in chunks, tool calls as a single delta.
    """
    response = _script(messages, info)
    for part in response.parts:
        if isinstance(part, TextPart):
            for word in part.content.split(" "):
                yield word + " "
        elif isinstance(part, ToolCallPart):
            args: Any = part.args
            yield {
                0: DeltaToolCall(
                    name=part.tool_name,
                    json_args=args if isinstance(args, str) else json.dumps(args),
                    tool_call_id=part.tool_call_id,
                )
            }


def scripted_model() -> FunctionModel:
    return FunctionModel(_script, stream_function=_script_stream)


def _build_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A build run in miniature: recall, then a declared outcome."""
    calls_made = [
        p.tool_name
        for m in messages
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ToolCallPart)
    ]
    if "kwim_recall" not in calls_made:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="kwim_recall",
                    args={"query": "how this workspace is laid out", "k": 3},
                    tool_call_id="call_recall_1",
                )
            ]
        )
    if "task_complete" not in calls_made:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="task_complete",
                    args={"summary": "Edited one file and the checks pass."},
                    tool_call_id="call_done_1",
                )
            ]
        )
    return ModelResponse(parts=[TextPart(content="Done.")])


def _review_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Recall, then text with no tool call and no declared outcome; the driver hands
    this run to a human.
    """
    calls_made = [
        p.tool_name
        for m in messages
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ToolCallPart)
    ]
    if "kwim_recall" not in calls_made:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="kwim_recall",
                    args={"query": "how this workspace is laid out", "k": 3},
                    tool_call_id="call_recall_1",
                )
            ]
        )
    return ModelResponse(
        parts=[TextPart(content="The spec contradicts itself in section 3; which reading is right?")]
    )


def _busy_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Never stops calling tools, so only `max_requests_per_turn` ends a turn and only
    `max_iterations` ends the run.
    """
    n = sum(
        1
        for m in messages
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ToolCallPart)
    )
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="kwim_recall",
                args={"query": f"pass {n}", "k": 1},
                tool_call_id=f"call_busy_{n}",
            )
        ]
    )


def _streamed(script):
    """Streaming form of a script. Every run streams (`event_stream_handler`), so every
    script needs one.
    """

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        response = script(messages, info)
        for part in response.parts:
            if isinstance(part, TextPart):
                for word in part.content.split(" "):
                    yield word + " "
            elif isinstance(part, ToolCallPart):
                args: Any = part.args
                yield {
                    0: DeltaToolCall(
                        name=part.tool_name,
                        json_args=args if isinstance(args, str) else json.dumps(args),
                        tool_call_id=part.tool_call_id,
                    )
                }

    return stream


def scripted_builder() -> FunctionModel:
    return FunctionModel(_build_script, stream_function=_streamed(_build_script))


def scripted_reviewer() -> FunctionModel:
    return FunctionModel(_review_script, stream_function=_streamed(_review_script))


def scripted_busy() -> FunctionModel:
    return FunctionModel(_busy_script, stream_function=_streamed(_busy_script))
