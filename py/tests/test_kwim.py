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
        super().__init__(None)  # type: ignore[arg-type] - no HTTP; every call is overridden
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

    async def emit_episodic(self, session_id, event_type, event_data):
        return None


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
        assert await KwimClient(c).semantic_recall("anything") is None


@pytest.mark.asyncio
async def test_recall_is_fail_soft_on_a_500():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="nope"))
    ) as c:
        assert await KwimClient(c).semantic_recall("anything") is None


@pytest.mark.asyncio
async def test_recall_is_fail_soft_on_an_unexpected_payload():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"not": "a list"}))
    ) as c:
        assert await KwimClient(c).semantic_recall("anything") is None


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


# --- what a turn recalled ----------------------------------------------------


@pytest.mark.asyncio
async def test_recall_notes_each_item_once_until_the_next_turn():
    payload = [
        {"id": "runbook:backups", "content": "Snapshots start at 02:15 UTC.",
         "metadata": {"topic": "backups"}, "score": 0.3},
        {"id": "runbook:restore", "content": "Restores need the tern key.",
         "metadata": {"topic": "restore"}, "score": 0.4},
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as c:
        kwim = KwimClient(c)
        kwim.begin_turn()
        await kwim.semantic_recall("backups")
        await kwim.semantic_recall("backups again")      # same items: noted once
        items = kwim.recalled_items()
        assert [i["id"] for i in items] == ["runbook:backups", "runbook:restore"]
        assert items[0] == {"kind": "semantic", "id": "runbook:backups",
                            "text": "Snapshots start at 02:15 UTC."}
        kwim.begin_turn()
        assert kwim.recalled_items() == []


@pytest.mark.asyncio
async def test_warm_start_notes_facts_and_rules():
    payload = {"wisdom": [{"id": "rule-1", "approach": "Drain the queue first."}],
               "knowledge": [{"id": "fact-1", "statement": "quokka-07 is the backup host."}]}
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as c:
        kwim = KwimClient(c)
        kwim.begin_turn()
        await kwim.warm_start("s1", "backups")
        assert kwim.recalled_items() == [
            {"kind": "rule", "id": "rule-1", "text": "Drain the queue first."},
            {"kind": "fact", "id": "fact-1", "text": "quokka-07 is the backup host."},
        ]


@pytest.mark.asyncio
async def test_failed_recall_notes_nothing():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"detail": "no"}))
    ) as c:
        kwim = KwimClient(c)
        kwim.begin_turn()
        await kwim.semantic_recall("backups")
        assert kwim.recalled_items() == []


# --- failures are reported, once per kind ---------------------------------


def _always(status: int, body=None):
    return httpx.MockTransport(lambda r: httpx.Response(status, json=body if body is not None else {}))


@pytest.mark.asyncio
async def test_a_401_warns_once_for_repeated_calls(caplog):
    async with httpx.AsyncClient(transport=_always(401)) as c:
        kwim = KwimClient(c)
        with caplog.at_level("DEBUG", logger="djed.kwim"):
            for _ in range(3):
                assert await kwim.semantic_recall("anything") is None
            await kwim.warm_start("s1", "anything")
            await kwim.emit_episodic("s1", "cli_turn", {"request": "q"})
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "HTTP 401" in warnings[0].getMessage() and "DJED_KWIM_API_KEY" in warnings[0].getMessage()
    assert not kwim.available
    assert kwim.drain_notices() == [warnings[0].getMessage()]
    assert kwim.drain_notices() == []


@pytest.mark.asyncio
async def test_each_kind_of_failure_warns_once(caplog):
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] % 2:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(503, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(flaky)) as c:
        kwim = KwimClient(c)
        with caplog.at_level("DEBUG", logger="djed.kwim"):
            for _ in range(4):
                await kwim.semantic_recall("anything")
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 2
    assert any("ConnectError" in w for w in warnings) and any("HTTP 503" in w for w in warnings)


@pytest.mark.asyncio
async def test_a_success_marks_memory_available_again():
    responses = iter([httpx.Response(500, json={}), httpx.Response(200, json=[])])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: next(responses))) as c:
        kwim = KwimClient(c)
        assert await kwim.semantic_recall("x") is None
        assert not kwim.available
        assert await kwim.semantic_recall("x") == []
        assert kwim.available


@pytest.mark.asyncio
async def test_a_failed_episodic_write_is_reported(caplog):
    async with httpx.AsyncClient(transport=_always(401)) as c:
        kwim = KwimClient(c)
        with caplog.at_level("WARNING", logger="djed.kwim"):
            await kwim.emit_episodic("s1", "cli_turn", {"request": "q"})
    assert any("record failed: HTTP 401" in r.getMessage() for r in caplog.records)


async def _kwim_recall_with(transport) -> str:
    from types import SimpleNamespace

    from djed.tools import Deps, kwim_recall

    async with httpx.AsyncClient(transport=transport) as c:
        ctx = SimpleNamespace(deps=Deps(kwim=KwimClient(c), session_id="s"))
        return await kwim_recall(ctx, "anything")


@pytest.mark.asyncio
async def test_kwim_recall_says_when_kwim_is_unavailable():
    out = await _kwim_recall_with(_always(401))
    assert out == "(KWIM is unavailable: HTTP 401 (check DJED_KWIM_API_KEY))"


@pytest.mark.asyncio
async def test_kwim_recall_keeps_the_no_match_message_for_an_empty_result():
    assert await _kwim_recall_with(_always(200, [])) == "(no matching knowledge in KWIM)"


def test_startup_warns_when_the_url_is_set_without_a_key(monkeypatch):
    from djed.agent import kwim_config_warning

    monkeypatch.setenv("KWIM_BASE_URL", "http://kwim.test:8000")
    monkeypatch.setenv("DJED_KWIM_API_KEY", "")
    assert "memory is disabled" in kwim_config_warning()
    monkeypatch.setenv("DJED_KWIM_API_KEY", "k")
    assert kwim_config_warning() is None
    monkeypatch.delenv("KWIM_BASE_URL")
    monkeypatch.delenv("DJED_KWIM_API_KEY")
    assert kwim_config_warning() is None


# --- an empty key: memory is off, and no call reaches KWIM ---------------------


def _counting():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=[])

    return httpx.MockTransport(handler), calls


@pytest.mark.asyncio
async def test_an_empty_key_builds_a_disabled_client_that_makes_no_calls(monkeypatch, caplog):
    from types import SimpleNamespace

    from djed.agent import build_kwim_client
    from djed.tools import Deps, kwim_recall

    monkeypatch.setenv("KWIM_BASE_URL", "http://kwim.test:8000")
    monkeypatch.setenv("DJED_KWIM_API_KEY", "")
    transport, calls = _counting()
    async with httpx.AsyncClient(transport=transport) as c:
        with caplog.at_level("WARNING", logger="djed"):
            kwim = build_kwim_client(c)
            assert not kwim.available
            assert await kwim.semantic_recall("anything") is None
            assert await kwim.warm_start("s1", "anything") == []
            await kwim.emit_episodic("s1", "cli_turn", {"request": "q"})
            ctx = SimpleNamespace(deps=Deps(kwim=kwim, session_id="s1"))
            out = await kwim_recall(ctx, "anything")
    assert calls == []
    assert out == "(KWIM is unavailable: DJED_KWIM_API_KEY is empty - memory is disabled)"
    # The startup warning, once; no per-call failure warnings or notices.
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == ["KWIM_BASE_URL is set but DJED_KWIM_API_KEY is empty - memory is disabled"]
    assert kwim.drain_notices() == []


@pytest.mark.asyncio
async def test_a_key_builds_a_working_client(monkeypatch):
    from djed.agent import build_kwim_client

    monkeypatch.setenv("KWIM_BASE_URL", "http://kwim.test:8000")
    monkeypatch.setenv("DJED_KWIM_API_KEY", "k")
    transport, calls = _counting()
    async with httpx.AsyncClient(transport=transport) as c:
        kwim = build_kwim_client(c)
        assert kwim.available
        assert await kwim.semantic_recall("anything") == []
    assert calls == ["/v1/memory/semantic"]
