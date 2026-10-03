"""Per-request telemetry, read from the transcript.

`RunUsage` gives one total per turn, and a request that raised is not in it.
Every `ModelResponse` carries its own `usage`, `finish_reason` and `parts`, and
`capture_run_messages` keeps them when the run raises, so the trace is built
from the message list.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)


@dataclass(frozen=True)
class RequestTrace:
    """One model request, as the transcript records it."""

    index: int
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""
    text_chars: int = 0
    thinking_chars: int = 0
    tool_calls: tuple[str, ...] = field(default_factory=tuple)

    @property
    def actionable(self) -> bool:
        """Did this response carry anything the agent could act on?

        A response that is empty, or only thinking, is what pydantic-ai refuses
        with "token limit exceeded before any response was generated".
        """
        return bool(self.text_chars or self.tool_calls)

    @property
    def summary(self) -> str:
        kinds = []
        if self.tool_calls:
            kinds.append(f"tools={','.join(self.tool_calls)}")
        if self.text_chars:
            kinds.append(f"text={self.text_chars}c")
        if self.thinking_chars:
            kinds.append(f"thinking={self.thinking_chars}c")
        return " ".join(kinds) or "(empty)"


def trace_requests(messages: list[ModelMessage]) -> list[RequestTrace]:
    """One `RequestTrace` per model response in the transcript."""
    traces: list[RequestTrace] = []
    for m in messages:
        if not isinstance(m, ModelResponse):
            continue
        text = thinking = 0
        calls: list[str] = []
        for p in m.parts:
            if isinstance(p, ThinkingPart):
                thinking += len(p.content or "")
            elif isinstance(p, TextPart):
                text += len(p.content or "")
            elif isinstance(p, ToolCallPart):
                calls.append(p.tool_name)
        usage = m.usage
        traces.append(
            RequestTrace(
                index=len(traces) + 1,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                finish_reason=m.finish_reason or "",
                text_chars=text,
                thinking_chars=thinking,
                tool_calls=tuple(calls),
            )
        )
    return traces


def format_trace(traces: list[RequestTrace], limit: int | None = None) -> str:
    """A table, newest last. `limit` keeps only the final `limit` rows."""
    rows = traces[-limit:] if limit else traces
    if not rows:
        return "(no model requests recorded)"
    out = [f"{'req':>4}  {'in':>7}  {'out':>6}  {'finish':<10}  detail"]
    for t in rows:
        flag = "" if t.actionable else "  <- nothing actionable"
        out.append(
            f"{t.index:>4}  {t.input_tokens:>7}  {t.output_tokens:>6}  "
            f"{t.finish_reason or '-':<10}  {t.summary}{flag}"
        )
    return "\n".join(out)


def totals(traces: list[RequestTrace]) -> tuple[int, int, int]:
    """`(requests, input_tokens, output_tokens)`, failed requests included."""
    return (
        len(traces),
        sum(t.input_tokens for t in traces),
        sum(t.output_tokens for t in traces),
    )
