"""Bounding what a growing transcript re-sends.

`Session.history` keeps every response and tool result, and every request
re-sends all of it. Two passes bound it; neither reorders or drops a message:

`strip_thinking` drops reasoning blocks from prior responses. A reasoning block
is not needed once its tool call has returned, and is often the largest part of
a response.

`elide_tool_results` replaces the payload of older results with a note naming
the call that produced it. Each call stays paired with its result, so the
transcript stays valid.
"""
from __future__ import annotations

import os
from dataclasses import replace

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)

# Characters of tool-result payload kept whole, newest first; older results are
# elided. A character budget, not a count, because `cap.py` allows 8KB per
# result. About 60k characters is about 15k tokens.
DEFAULT_RESULT_BUDGET = 60_000


def keep_thinking() -> bool:
    """Whether to leave reasoning blocks in the transcript. Off by default: this only
    affects what is re-sent, unlike `DJED_DISABLE_THINKING`, which changes what the
    model generates.
    """
    raw = (os.environ.get("DJED_KEEP_THINKING") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def result_budget() -> int:
    """Characters of tool-result payload to keep whole. 0 disables eliding."""
    raw = (os.environ.get("DJED_RESULT_BUDGET") or "").strip()
    if not raw:
        return DEFAULT_RESULT_BUDGET
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_RESULT_BUDGET


def strip_thinking(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove `ThinkingPart`s from prior responses."""
    out: list[ModelMessage] = []
    for m in messages:
        if not isinstance(m, ModelResponse):
            out.append(m)
            continue
        kept = [p for p in m.parts if not isinstance(p, ThinkingPart)]
        if len(kept) == len(m.parts):
            out.append(m)
            continue
        # pydantic-ai already refuses a thinking-only response; this guards against an
        # empty response, which a strict server rejects with a 400.
        if kept:
            out.append(replace(m, parts=kept))
    return out


def _call_index(messages: list[ModelMessage]) -> dict[str, ToolCallPart]:
    """tool_call_id -> the call that asked for it, for naming an elided result."""
    return {
        p.tool_call_id: p
        for m in messages
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ToolCallPart)
    }


def _stub(part: ToolReturnPart, call: ToolCallPart | None, size: int) -> str:
    """What stands in for an elided payload: the call that produced it, so the model
    can reissue it.
    """
    name = part.tool_name or (call.tool_name if call else "tool")
    args = ""
    if call is not None and call.args:
        args = call.args if isinstance(call.args, str) else str(call.args)
        args = " ".join(args.split())
        if len(args) > 160:
            args = args[:160] + "..."
    where = f" {args}" if args else ""
    return (
        f"[elided: {name}{where} returned {size} characters. "
        "This ran and succeeded; re-run it if you need the content again.]"
    )


def elide_tool_results(
    messages: list[ModelMessage], budget: int | None = None
) -> list[ModelMessage]:
    """Replace older tool-result payloads with a note naming the call.

    Newest results are kept whole until `budget` characters are used; older ones
    are elided. A budget of 0 elides every result.
    """
    budget = result_budget() if budget is None else budget
    calls = _call_index(messages)

    # Newest first, so what is kept is what was most recently learned.
    spend = 0
    elide: set[int] = set()
    for m in reversed(messages):
        if not isinstance(m, ModelRequest):
            continue
        for p in reversed(m.parts):
            if not isinstance(p, ToolReturnPart) or not isinstance(p.content, str):
                continue
            size = len(p.content)
            if spend + size <= budget:
                spend += size
            else:
                elide.add(id(p))

    if not elide:
        return list(messages)

    out: list[ModelMessage] = []
    for m in messages:
        if not isinstance(m, ModelRequest):
            out.append(m)
            continue
        parts = [
            replace(p, content=_stub(p, calls.get(p.tool_call_id), len(p.content)))
            if id(p) in elide
            else p
            for p in m.parts
        ]
        out.append(replace(m, parts=parts) if parts != m.parts else m)
    return out


def decay(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Both passes, in the order that loses least first."""
    if not keep_thinking():
        messages = strip_thinking(messages)
    return elide_tool_results(messages)
