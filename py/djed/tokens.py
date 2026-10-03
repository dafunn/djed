"""Estimate what tool definitions cost in input tokens, per request.

A floor, not a measurement: cl100k_base is not the tokenizer a local model uses.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Protocol

log = logging.getLogger("djed.tokens")


class ToolLike(Protocol):
    """The shape both FastMCP tools and Pydantic AI tool defs satisfy."""

    name: str


def _envelope(tool: Any) -> dict[str, Any]:
    """The wire form of one tool: `{"type": "function", "function": {...}}`.

    The envelope's keys and JSON punctuation are themselves tokens, so counting
    only the description and schema undercounts by roughly a third.
    """
    schema = (
        getattr(tool, "inputSchema", None)
        or getattr(tool, "parameters_json_schema", None)
        or {"type": "object", "properties": {}}
    )
    return {
        "type": "function",
        "function": {
            "name": getattr(tool, "name", ""),
            "description": getattr(tool, "description", "") or "",
            "parameters": schema,
        },
    }


def estimate_tool_tokens(tools: list[Any]) -> int:
    """Estimated input tokens for these tool definitions, as sent on the wire."""
    if not tools:
        return 0
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - tiktoken ships with pydantic-ai
        log.warning("tiktoken unavailable; reporting tool cost as 0")
        return 0
    enc = tiktoken.get_encoding("cl100k_base")
    payload = json.dumps([_envelope(t) for t in tools], separators=(",", ":"))
    return len(enc.encode(payload))
