"""KWIM retrieval. `transform_context` runs before every LLM call: it prunes
the transcript to a working window, then injects warm-start and recall.

Fail-soft throughout - retrieval must never break a turn.

Endpoints (the KWIM service, not the LLM gateway):
  GET  /v1/memory/semantic?q=&limit=&meta.doc_type=  KNN recall over knowledge
  GET  /v1/memory/context?session_id=&subject=       warm-start
  POST /v1/memory/episodic                           record a turn
Auth: Bearer <KWIM api-key>
"""
from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Any

import httpx
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from .persona import default_persona

log = logging.getLogger("djed.kwim")

DEFAULT_KWIM_BASE = "http://localhost:8000"

# Resolved at call time, not import time - see the note in provider.py.


def kwim_base() -> str:
    return os.environ.get("KWIM_BASE_URL") or DEFAULT_KWIM_BASE


def window_size() -> int:
    """Recent messages kept in-agent."""
    return int(os.environ.get("DJED_WINDOW_MSGS") or 12)


def recall_k() -> int:
    """Chunks retrieved per recall."""
    return int(os.environ.get("DJED_RECALL_K") or 4)


# Marks messages we injected. Pydantic AI messages are validated models with no
# room for a flag, so they are tagged by this content header.
INJECT_HEADER = "# Retrieved context (from KWIM - use if relevant to my request, otherwise ignore)"


class KwimClient:
    """KWIM memory API client for the agent's retrieval path.

    Every method is fail-soft by contract: network errors, non-200s, and
    unexpected payload shapes all degrade to "no memory", never an exception.
    A turn must survive KWIM being down.
    """

    def __init__(
        self, client: httpx.AsyncClient, base_url: str | None = None, api_key: str = ""
    ):
        self.client = client
        self.base_url = (base_url or kwim_base()).rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def semantic_recall(self, query: str, k: int | None = None) -> list[str]:
        """KNN recall over the ops knowledge base, formatted for context injection."""
        if not query.strip():
            return []
        k = k or recall_k()
        try:
            r = await self.client.get(
                f"{self.base_url}/v1/memory/semantic",
                params={"q": query, "limit": k, "meta.doc_type": "runbook"},
                headers=self.headers,
            )
            if r.status_code != 200:
                return []
            items = r.json()
            if not isinstance(items, list):
                return []
            out = []
            for it in items:
                topic = (it.get("metadata") or {}).get("topic", "note")
                score = it.get("score") or 0.0
                out.append(f"[{topic} | {score:.2f}]\n{it.get('content', '')}")
            return out
        except Exception as e:
            log.debug("semantic_recall failed (fail-soft): %s", e)
            return []

    async def warm_start(self, session_id: str, subject: str) -> list[str]:
        """First-turn bundle: prior wisdom + known facts for this session/subject."""
        try:
            params: dict[str, Any] = {"session_id": session_id}
            if subject:
                params["subject"] = subject
            r = await self.client.get(
                f"{self.base_url}/v1/memory/context", params=params, headers=self.headers
            )
            if r.status_code != 200:
                return []
            ctx = r.json()
            if not isinstance(ctx, dict):
                return []
            out = []
            for w in ctx.get("wisdom") or []:
                approach = w.get("approach") or w.get("action_pattern") or ""
                out.append(f"WISDOM: {approach.strip()}")
            for f in ctx.get("knowledge") or []:
                out.append(f"KNOWN: {(f.get('statement') or f.get('content') or '').strip()}")
            # Longer than the bare "KNOWN: " prefix, i.e. it has a body.
            return [s.strip() for s in out if len(s.strip()) > len("KNOWN: ")]
        except Exception as e:
            log.debug("warm_start failed (fail-soft): %s", e)
            return []

    async def emit_episodic(
        self, session_id: str, event_type: str, event_data: dict[str, Any]
    ) -> None:
        """Record a turn so warm-start has something to pull next session.

        Fire-and-forget: memory writes never break a turn.
        """
        try:
            await self.client.post(
                f"{self.base_url}/v1/memory/episodic",
                json={
                    "agent_id": default_persona().tag,
                    "session_id": session_id,
                    "event_type": event_type,
                    "event_data": event_data,
                },
                headers=self.headers,
            )
        except Exception as e:
            log.debug("emit_episodic failed (fail-soft): %s", e)


# --- message-list surgery -------------------------------------------------


def _is_injected(m: ModelMessage) -> bool:
    if not isinstance(m, ModelRequest):
        return False
    return any(
        isinstance(p, UserPromptPart)
        and isinstance(p.content, str)
        and p.content.startswith(INJECT_HEADER)
        for p in m.parts
    )


def _orphaned_head(m: ModelMessage) -> bool:
    """True if `m` cannot legally open a transcript window.

    A tool result whose originating call was pruned - or a bare model response -
    orphans the pairing, which strict OpenAI-compatible servers 400 on.
    """
    if isinstance(m, ModelResponse):
        return True
    return isinstance(m, ModelRequest) and any(
        isinstance(p, (ToolReturnPart, RetryPromptPart)) for p in m.parts
    )


def drop_dangling_tool_calls(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Strip tool calls that never received a result.

    An interrupted turn leaves a `ToolCallPart` unmatched, which strict
    OpenAI-compatible servers 400 on. `prune` handles the same at the head.
    """
    answered = {
        p.tool_call_id
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, (ToolReturnPart, RetryPromptPart))
    }
    out: list[ModelMessage] = []
    for m in messages:
        if isinstance(m, ModelResponse):
            kept = [
                p
                for p in m.parts
                if not (isinstance(p, ToolCallPart) and p.tool_call_id not in answered)
            ]
            if not kept:
                continue  # nothing but orphaned calls - drop the message
            if len(kept) != len(m.parts):
                m = replace(m, parts=kept)
        out.append(m)
    return out


def prune(messages: list[ModelMessage], window: int | None = None) -> list[ModelMessage]:
    """Keep roughly the last `window` messages, opening on a legal boundary.

    Walks backwards, never forwards: forward trimming eats a tool-heavy turn
    entirely and the empty result ends the run. Can return more than the cap,
    never zero.
    """
    window = window or window_size()
    if len(messages) <= window:
        return list(messages)
    start = len(messages) - window
    while start > 0 and _orphaned_head(messages[start]):
        start -= 1
    return list(messages[start:])


def last_user_text(messages: list[ModelMessage]) -> str:
    """Text of the most recent genuine user turn (ignoring our own injections)."""
    for m in reversed(messages):
        if _is_injected(m) or not isinstance(m, ModelRequest):
            continue
        for p in reversed(m.parts):
            if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                return p.content
    return ""


def _inject(messages: list[ModelMessage], blocks: list[str]) -> list[ModelMessage]:
    """Insert one tagged context message immediately before the last user turn."""
    body = INJECT_HEADER + "\n\n" + "\n\n---\n\n".join(blocks)
    ctx_msg = ModelRequest(parts=[UserPromptPart(content=body)])
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, ModelRequest) and any(
            isinstance(p, UserPromptPart) for p in m.parts
        ):
            return [*messages[:i], ctx_msg, *messages[i:]]
    return [*messages, ctx_msg]


def make_transform_context(kwim: KwimClient, session_id: str):
    """Build the history processor for a session.

    Pass the result to `ProcessHistory(...)` in the Agent's capabilities - it is
    the Pydantic AI seam that runs before every model request.
    """
    warmed_up = False

    async def transform_context(messages: list[ModelMessage]) -> list[ModelMessage]:
        nonlocal warmed_up

        # 1) strip any block we injected on a previous pass - idempotent whether
        #    the framework passes our output back in or not.
        durable = [m for m in messages if not _is_injected(m)]

        # 2) prune to a small working window. The transcript is scratch; KWIM is memory.
        recent = prune(durable)

        # 3) drive retrieval off the latest user turn.
        query = last_user_text(recent)

        # 4) gather external context (fail-soft; empty until KWIM is seeded).
        blocks: list[str] = []
        if not warmed_up:
            warmed_up = True
            blocks.extend(await kwim.warm_start(session_id, query))
        blocks.extend(await kwim.semantic_recall(query))
        result = _inject(recent, blocks) if blocks else recent

        # Pydantic AI rejects an empty processed history, which ends the turn.
        return result or list(messages)

    return transform_context
