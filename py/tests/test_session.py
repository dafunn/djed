"""Session behaviour: multi-turn history, interrupt, and transcript hygiene."""
from __future__ import annotations

import asyncio

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import UsageLimits

from fake_model import scripted_model
from djed.agent import build_agent
from djed.kwim import drop_dangling_tool_calls
from djed.session import Session, TextChunk, ToolStarted
from djed.tiers import EagerTiers, Tier
from djed.tools import Deps
from tests.test_kwim import StubKwim


def text_model(fn) -> FunctionModel:
    """A FunctionModel that can also stream; `Session` puts every run on the
    streaming path.
    """

    async def stream(messages, info):
        response = fn(messages, info)
        for part in response.parts:
            if isinstance(part, TextPart):
                yield part.content

    return FunctionModel(fn, stream_function=stream)


def make_session(model=None):
    kwim = StubKwim()
    agent = build_agent(kwim, "s", model=model or scripted_model())
    return Session(agent, Deps(kwim=kwim, session_id="s"), kwim, "s")


async def approve_all(tool_name, detail):
    return True


def collect(events):
    return lambda e: events.append(e)


# --- dangling tool calls --------------------------------------------------


def test_dangling_tool_call_is_stripped():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="q")]),
        ModelResponse(parts=[ToolCallPart(tool_name="t", args={}, tool_call_id="c1")]),
    ]
    assert drop_dangling_tool_calls(msgs) == msgs[:1]


def test_answered_tool_call_is_kept():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="q")]),
        ModelResponse(parts=[ToolCallPart(tool_name="t", args={}, tool_call_id="c1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="t", content="ok", tool_call_id="c1")]),
    ]
    assert drop_dangling_tool_calls(msgs) == msgs


def test_text_survives_when_a_sibling_tool_call_is_stripped():
    """A response can hold text and a call; losing the call must not lose the text."""
    msgs = [
        ModelResponse(
            parts=[
                TextPart(content="let me check"),
                ToolCallPart(tool_name="t", args={}, tool_call_id="c1"),
            ]
        )
    ]
    out = drop_dangling_tool_calls(msgs)
    assert len(out) == 1
    assert [type(p).__name__ for p in out[0].parts] == ["TextPart"]


def test_clean_history_is_unchanged():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="q")]),
        ModelResponse(parts=[TextPart(content="a")]),
    ]
    assert drop_dangling_tool_calls(msgs) == msgs


# --- multi-turn history ---------------------------------------------------


@pytest.mark.asyncio
async def test_history_accumulates_across_turns():
    def echo(messages, info):
        return ModelResponse(parts=[TextPart(content=f"seen {len(messages)}")])

    session = make_session(text_model(echo))
    first = await session.run_turn("first", lambda e: None, approve_all)
    after_one = len(session.history)
    second = await session.run_turn("second", lambda e: None, approve_all)

    # Assert success explicitly: a failed run also leaves the prompt in history.
    assert first.error is None and second.error is None
    assert after_one > 0
    assert len(session.history) > after_one
    prompts = [
        p.content
        for m in session.history
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, UserPromptPart)
    ]
    assert "first" in prompts and "second" in prompts


@pytest.mark.asyncio
async def test_the_model_sees_prior_turns():
    seen: list[int] = []

    def counter(messages, info):
        seen.append(len(messages))
        return ModelResponse(parts=[TextPart(content="ok")])

    session = make_session(text_model(counter))
    await session.run_turn("one", lambda e: None, approve_all)
    await session.run_turn("two", lambda e: None, approve_all)
    assert seen[1] > seen[0], "second turn must carry the first turn's history"


@pytest.mark.asyncio
async def test_clear_drops_history():
    session = make_session(
        text_model(lambda m, i: ModelResponse(parts=[TextPart(content="x")]))
    )
    await session.run_turn("one", lambda e: None, approve_all)
    assert session.history
    session.clear()
    assert session.history == []


# --- interrupt ------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_marks_the_turn_and_leaves_history_clean():
    """A cancelled turn leaves no unanswered tool call."""
    started = asyncio.Event()

    def slow(messages, info):
        started.set()
        return ModelResponse(parts=[TextPart(content="never gets here")])

    async def slow_stream(messages, info):
        started.set()
        await asyncio.sleep(10)
        yield "unreachable"

    session = make_session(FunctionModel(slow, stream_function=slow_stream))

    async def drive():
        return await session.run_turn("hang", lambda e: None, approve_all)

    task = asyncio.create_task(drive())
    await asyncio.wait_for(started.wait(), timeout=5)
    assert session.busy
    assert session.interrupt() is True
    result = await asyncio.wait_for(task, timeout=5)

    assert result.interrupted
    assert not session.busy
    assert session.history == drop_dangling_tool_calls(session.history)


@pytest.mark.asyncio
async def test_interrupt_when_idle_is_a_no_op():
    session = make_session()
    assert session.interrupt() is False


@pytest.mark.asyncio
async def test_conversation_continues_after_an_interrupt():
    """The real requirement: an interrupted turn must not poison the session."""
    hang = asyncio.Event()

    async def stream(messages, info):
        hang.set()
        await asyncio.sleep(10)
        yield "unreachable"

    def sync(messages, info):
        return ModelResponse(parts=[TextPart(content="recovered")])

    session = make_session(FunctionModel(sync, stream_function=stream))
    task = asyncio.create_task(session.run_turn("hang", lambda e: None, approve_all))
    await asyncio.wait_for(hang.wait(), timeout=5)
    session.interrupt()
    await asyncio.wait_for(task, timeout=5)

    # A follow-up turn on the same session must succeed.
    session.agent = build_agent(session.kwim, "s", model=text_model(sync))
    result = await session.run_turn("still there?", lambda e: None, approve_all)
    assert result.error is None
    assert result.text == "recovered"


# --- events ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_report_tool_calls_and_text():
    events = []
    session = make_session()
    await session.run_turn("go", collect(events), approve_all)

    tools = [e.name for e in events if isinstance(e, ToolStarted)]
    assert "kwim_recall" in tools and "run_command" in tools
    assert any(isinstance(e, TextChunk) for e in events)


@pytest.mark.asyncio
async def test_a_tool_call_is_announced_once_despite_the_approval_replay():
    events = []
    session = make_session()
    await session.run_turn("go", collect(events), approve_all)
    names = [e.name for e in events if isinstance(e, ToolStarted)]
    assert names.count("run_command") == 1


@pytest.mark.asyncio
async def test_the_request_limit_spans_the_approval_round_trip():
    """The request ceiling counts across a whole turn, including across an approval
    gate: `scripted_model` makes two requests before the gate and one after.
    """
    kwim = StubKwim()
    agent = build_agent(kwim, "s", model=scripted_model())
    session = Session(
        agent,
        Deps(kwim=kwim, session_id="s"),
        kwim,
        "s",
        usage_limits=UsageLimits(request_limit=2),
    )
    result = await session.run_turn("go", lambda e: None, approve_all)
    assert result.error and "UsageLimit" in result.error


@pytest.mark.asyncio
async def test_usage_is_reported_even_when_the_turn_trips_the_limit():
    """Usage is recorded as the turn goes, so a turn that raised still reports what
    it spent.
    """
    kwim = StubKwim()
    agent = build_agent(kwim, "s", model=scripted_model())
    session = Session(
        agent,
        Deps(kwim=kwim, session_id="s"),
        kwim,
        "s",
        usage_limits=UsageLimits(request_limit=2),
    )
    result = await session.run_turn("go", lambda e: None, approve_all)
    assert result.error
    assert result.usage.requests == 2


@pytest.mark.asyncio
async def test_history_is_bounded_before_the_next_turn_inherits_it(monkeypatch):
    """Decay is applied when history is assigned."""
    monkeypatch.setenv("DJED_RESULT_BUDGET", "0")
    session = make_session()
    await session.run_turn("go", lambda e: None, approve_all)
    payloads = [
        p.content
        for m in session.history
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    ]
    assert payloads, "the scripted model makes tool calls"
    assert all(str(c).startswith("[elided:") for c in payloads)


# --- episodic memory ------------------------------------------------------


class RecordingKwim(StubKwim):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.episodic: list[tuple[str, str, dict]] = []

    async def emit_episodic(self, session_id, event_type, event_data):
        self.episodic.append((session_id, event_type, event_data))


def recording_session(model=None):
    kwim = RecordingKwim()
    agent = build_agent(kwim, "s", model=model or scripted_model())
    return Session(agent, Deps(kwim=kwim, session_id="s"), kwim, "s"), kwim


@pytest.mark.asyncio
async def test_successful_turn_is_written_to_kwim():
    session, kwim = recording_session(
        text_model(lambda m, i: ModelResponse(parts=[TextPart(content="the answer")]))
    )
    await session.run_turn("a question", lambda e: None, approve_all)

    assert len(kwim.episodic) == 1
    _, event_type, data = kwim.episodic[0]
    assert event_type == "cli_turn"
    assert data["request"] == "a question"
    assert data["response"] == "the answer"
    assert data["outcome"] == "ok"


@pytest.mark.asyncio
async def test_interrupted_turn_is_still_written_to_kwim():
    """Aborted turns are recorded too."""
    hang = asyncio.Event()

    async def stream(messages, info):
        yield "partial thought "
        hang.set()
        await asyncio.sleep(10)
        yield "never"

    session, kwim = recording_session(
        FunctionModel(lambda m, i: ModelResponse(parts=[TextPart(content="x")]),
                      stream_function=stream)
    )
    task = asyncio.create_task(session.run_turn("hang", lambda e: None, approve_all))
    await asyncio.wait_for(hang.wait(), timeout=5)
    session.interrupt()
    result = await asyncio.wait_for(task, timeout=5)

    assert result.interrupted
    assert len(kwim.episodic) == 1
    _, _, data = kwim.episodic[0]
    assert data["outcome"] == "interrupted"
    assert data["request"] == "hang"
    assert "partial thought" in data["response"], "partial output should be kept"


@pytest.mark.asyncio
async def test_failed_turn_is_written_to_kwim_with_the_error():
    def boom(messages, info):
        raise RuntimeError("gateway exploded")

    async def boom_stream(messages, info):
        raise RuntimeError("gateway exploded")
        yield  # pragma: no cover

    session, kwim = recording_session(FunctionModel(boom, stream_function=boom_stream))
    result = await session.run_turn("q", lambda e: None, approve_all)

    assert result.error
    assert len(kwim.episodic) == 1
    _, _, data = kwim.episodic[0]
    assert data["outcome"] == "error"
    assert "gateway exploded" in data["error"]


@pytest.mark.asyncio
async def test_kwim_being_down_does_not_break_a_turn():
    class DeadKwim(StubKwim):
        async def emit_episodic(self, *a, **kw):
            raise ConnectionError("KWIM unreachable")

    kwim = DeadKwim()
    agent = build_agent(kwim, "s", model=text_model(
        lambda m, i: ModelResponse(parts=[TextPart(content="fine")])))
    session = Session(agent, Deps(kwim=kwim, session_id="s"), kwim, "s")
    result = await session.run_turn("q", lambda e: None, approve_all)
    assert result.error is None
    assert result.text == "fine"


def _rebuildable_session():
    kwim = StubKwim()

    def factory(toolsets, capabilities, extra_instructions, override=None):
        return build_agent(kwim, "s", model=override or text_model(lambda m, i: ModelResponse(
            parts=[TextPart(content="ok")])), toolsets=toolsets, capabilities=capabilities)

    s = Session(factory([], [], "", None), Deps(kwim=kwim, session_id="s"), kwim, "s",
                tiers=[
                    Tier("db", None, ("db_query",)),
                    Tier("cluster", None, ("cluster_list",)),
                ],
                strategy=EagerTiers(), agent_factory=factory)
    return s


def test_strategy_switch_keeps_the_tier_selection():
    """`/tier db` then `/strategy search` keeps the selection at db."""
    s = _rebuildable_session()
    s.set_tiers({"db"})
    assert s.enabled_tiers == {"db"}
    s.set_strategy("search")
    assert s.enabled_tiers == {"db"}
    s.set_strategy("eager")
    assert s.enabled_tiers == {"db"}


def test_tier_change_keeps_the_strategy():
    s = _rebuildable_session()
    s.set_strategy("search")
    s.set_tiers({"cluster"})
    assert s.strategy_label == "search"


def test_model_survives_a_rebuild():
    """`/model x` then `/tier y` keeps model x."""
    s = _rebuildable_session()
    sentinel = text_model(lambda m, i: ModelResponse(parts=[TextPart(content="switched")]))
    s.set_model(sentinel)
    s.set_tiers({"db"})
    assert s.agent.model is sentinel


@pytest.mark.asyncio
async def test_episodic_record_carries_the_outcome():
    """Both front ends record through Session._record, so the episodic record
    carries `outcome` whichever one wrote it."""
    session, kwim = recording_session()
    await session.run_turn("hello", lambda e: None, approve_all)
    assert kwim.episodic, "a turn must be recorded"
    _, event_type, data = kwim.episodic[-1]
    assert event_type == "cli_turn"
    assert data["outcome"] == "ok"
    assert data["request"] == "hello"
