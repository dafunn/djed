"""Tests for the KWIM retrieval seam: message-list surgery and fail-soft
degradation. Offline; everything touching the network is stubbed.
"""
from __future__ import annotations

import httpx
import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from djed.kwim import (
    INJECT_HEADER,
    KwimClient,
    _is_injected,
    last_user_text,
    make_transform_context,
    prune,
)


def user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def assistant(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def tool_call(name: str, call_id: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args={}, tool_call_id=call_id)])


def tool_return(name: str, call_id: str) -> ModelRequest:
    return ModelRequest(
        parts=[ToolReturnPart(tool_name=name, content="ok", tool_call_id=call_id)]
    )


# --- pruning --------------------------------------------------------------


def test_prune_keeps_everything_under_the_window():
    msgs = [user("a"), assistant("b")]
    assert prune(msgs, window=12) == msgs


def test_prune_trims_to_the_window():
    msgs = [user(str(i)) for i in range(20)]
    assert len(prune(msgs, window=6)) <= 6


def test_prune_never_opens_on_an_orphaned_tool_return():
    """The window never starts on a tool return."""
    msgs = [
        user("old"),
        tool_call("run_command", "c1"),
        tool_return("run_command", "c1"),
        assistant("done"),
        user("new"),
    ]
    kept = prune(msgs, window=3)
    assert not isinstance(kept[0], ModelResponse)
    assert not any(
        isinstance(p, (ToolReturnPart, RetryPromptPart)) for p in kept[0].parts
    )


def test_prune_never_returns_empty():
    """An all-orphan window widens backwards rather than emptying."""
    msgs = [user("x"), tool_call("t", "c1"), tool_return("t", "c1")]
    assert prune(msgs, window=2) == msgs


def test_a_tool_heavy_turn_does_not_empty_the_window():
    """A tool-heavy turn is not trimmed away."""
    msgs = [user("how long would VACUUM take?")]
    for i in range(8):
        msgs += [tool_call("db", f"c{i}"), tool_return("db", f"c{i}")]

    kept = prune(msgs, window=12)
    assert kept, "must never be empty"
    assert not _orphaned_head_public(kept[0]), "must open on a legal boundary"
    # Walking back reaches the user turn that started the sequence.
    assert kept[0] is msgs[0]


def _orphaned_head_public(m):
    from djed.kwim import _orphaned_head

    return _orphaned_head(m)


# --- query extraction -----------------------------------------------------


def test_last_user_text_picks_the_most_recent_turn():
    msgs = [user("first"), assistant("reply"), user("second")]
    assert last_user_text(msgs) == "second"


def test_last_user_text_ignores_our_own_injection():
    injected = user(f"{INJECT_HEADER}\n\nKNOWN: the cluster has a gateway")
    msgs = [user("real question"), injected]
    assert last_user_text(msgs) == "real question"


# --- the transform seam ---------------------------------------------------


class StubKwim(KwimClient):
    def __init__(self, recall: list[str] | None = None, warm: list[str] | None = None):
        self._recall = recall or []
        self._warm = warm or []
        self.recall_calls = 0
        self.warm_calls = 0

    async def semantic_recall(self, query, k=4):
        self.recall_calls += 1
        return list(self._recall)

    async def warm_start(self, session_id, subject):
        self.warm_calls += 1
        return list(self._warm)


@pytest.mark.asyncio
async def test_transform_injects_retrieved_context_before_the_user_turn():
    kwim = StubKwim(recall=["[monitoring | 0.91]\nalerts lives in ns monitoring"])
    transform = make_transform_context(kwim, "s1")
    out = await transform([user("how do I silence an alert?")])

    assert len(out) == 2
    assert _is_injected(out[0])
    assert "alerts" in out[0].parts[0].content
    assert out[1].parts[0].content == "how do I silence an alert?"


@pytest.mark.asyncio
async def test_transform_warm_starts_only_once():
    kwim = StubKwim(warm=["KNOWN: k3s runs on .47"])
    transform = make_transform_context(kwim, "s1")
    await transform([user("first")])
    await transform([user("second")])
    assert kwim.warm_calls == 1
    assert kwim.recall_calls == 2


@pytest.mark.asyncio
async def test_transform_is_idempotent_across_passes():
    """Injected blocks are stripped before re-injecting."""
    kwim = StubKwim(recall=["a fact"])
    transform = make_transform_context(kwim, "s1")
    once = await transform([user("q")])
    twice = await transform(once)
    assert sum(1 for m in twice if _is_injected(m)) == 1


@pytest.mark.asyncio
async def test_transform_passes_through_when_kwim_is_empty():
    transform = make_transform_context(StubKwim(), "s1")
    msgs = [user("q")]
    assert await transform(msgs) == msgs


# --- fail-soft ------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_is_fail_soft_on_transport_error():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as c:
        assert await KwimClient(c).semantic_recall("anything") == []


@pytest.mark.asyncio
async def test_recall_is_fail_soft_on_a_500():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="nope"))
    ) as c:
        assert await KwimClient(c).semantic_recall("anything") == []


@pytest.mark.asyncio
async def test_recall_is_fail_soft_on_an_unexpected_payload():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"not": "a list"}))
    ) as c:
        assert await KwimClient(c).semantic_recall("anything") == []


@pytest.mark.asyncio
async def test_recall_formats_hits_with_topic_and_score():
    payload = [
        {"content": "the runbook says X", "metadata": {"topic": "monitoring"}, "score": 0.87}
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as c:
        hits = await KwimClient(c).semantic_recall("x")
    assert hits == ["[monitoring | 0.87]\nthe runbook says X"]


@pytest.mark.asyncio
async def test_episodic_write_never_raises():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as c:
        await KwimClient(c).emit_episodic("s1", "cli_turn", {"request": "q"})
