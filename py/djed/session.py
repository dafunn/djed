"""One conversation with the agent - turn loop, history, approval, interrupt.

UI-agnostic: a front end supplies an event sink and an approval callback, which
is what makes interrupt and history testable without a terminal.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RunUsage,
    TextPart,
    TextPartDelta,
    ToolApproved,
    ToolDenied,
    UsageLimits,
    capture_run_messages,
)
from pydantic_ai.messages import ModelMessage, RetryPromptPart

from .decay import decay
from .kwim import KwimClient, drop_dangling_tool_calls
from .tiers import EagerTiers, SearchedTiers, Tier, ToolStrategy
from .tools import EXIT_PREFIX, Deps
from .trace import RequestTrace, trace_requests

# --- events a front end can render ---------------------------------------


@dataclass
class TextChunk:
    text: str


@dataclass
class ToolStarted:
    name: str
    args: str


@dataclass
class ToolFinished:
    text: str          # truncated for display
    is_error: bool = False
    full: str = ""     # untruncated, for /save and /copy


TurnEvent = TextChunk | ToolStarted | ToolFinished


def _is_error_result(part, content: str) -> bool:
    """Whether a tool result reports a failure.

    An MCP result with isError set reaches the model as a retry prompt (the
    toolset's default error behaviour) rather than a return; a tool that raised
    returns with outcome `failed`; run_command marks a non-zero exit in its text.
    """
    if isinstance(part, RetryPromptPart):
        return True
    if getattr(part, "outcome", "success") == "failed":
        return True
    return content.startswith(EXIT_PREFIX)
EventSink = Callable[[TurnEvent], None]
ApprovalFn = Callable[[str, str], Awaitable[bool]]
# (toolsets, capabilities, extra_instructions, model) -> Agent
AgentFactory = Callable[[list, list, str, object], Agent]


@dataclass
class TurnResult:
    text: str = ""
    interrupted: bool = False
    error: str | None = None
    usage: RunUsage = field(default_factory=RunUsage)
    # One entry per model request, read back off the transcript; `usage` above is a
    # running total.
    trace: list[RequestTrace] = field(default_factory=list)


class Session:
    """A running conversation. Owns the message history across turns."""

    def __init__(
        self,
        agent: Agent,
        deps: Deps,
        kwim: KwimClient,
        session_id: str,
        tiers: list[Tier] | None = None,
        strategy: ToolStrategy | None = None,
        agent_factory: AgentFactory | None = None,
        usage_limits: UsageLimits | None = None,
    ):
        self.agent = agent
        self.deps = deps
        self.kwim = kwim
        self.session_id = session_id
        self.tiers = tiers or []
        self.strategy: ToolStrategy = strategy or EagerTiers()
        # Bounds one turn: `agent.run` loops tool calls until the model stops calling
        # them.
        self.usage_limits = usage_limits
        self._agent_factory = agent_factory
        # Survives a rebuild: `/model` would otherwise be undone by `/tier`.
        self.model: object | None = None
        self.history: list[ModelMessage] = []
        self._task: asyncio.Task | None = None
        # Text streamed so far this turn, readable after a cancellation so an
        # interrupted turn can still be recorded.
        self._partial: list[str] = []

    # --- tool exposure, live ------------------------------------------------

    @property
    def strategy_label(self) -> str:
        return self.strategy.label

    @property
    def enabled_tiers(self) -> set[str]:
        enabled = getattr(self.strategy, "enabled", None)
        return {t.name for t in self.tiers} if enabled is None else set(enabled)

    @property
    def active_tool_count(self) -> int:
        """Tools visible to the model right now. Under `search` everything is
        deferred, so nothing is exposed up front."""
        if self.strategy.label == "search":
            return 0
        return sum(t.tool_count for t in self.tiers if t.name in self.enabled_tiers)

    def set_tiers(self, selection: set[str]) -> None:
        """Change which tiers are enabled, keeping the current strategy."""
        enabled = frozenset(selection)
        self.strategy = (
            SearchedTiers(enabled=enabled)
            if self.strategy.label == "search"
            else EagerTiers(enabled=enabled)
        )
        self._rebuild_agent()

    def set_strategy(self, kind: str) -> None:
        """Change strategy, keeping the current tier selection."""
        enabled = getattr(self.strategy, "enabled", None)
        self.strategy = (
            SearchedTiers(enabled=enabled) if kind == "search" else EagerTiers(enabled=enabled)
        )
        self._rebuild_agent()

    def _rebuild_agent(self) -> None:
        """Swap the agent in place after a tier/strategy change.

        History survives - only the tool surface changes, so the conversation
        carries on rather than restarting.
        """
        if self._agent_factory is None:
            return
        toolsets, capabilities = self.strategy.apply(self.tiers)
        self.agent = self._agent_factory(
            toolsets, capabilities, self.strategy.instructions, self.model
        )

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def interrupt(self) -> bool:
        """Abort the running turn. Returns True if there was one to abort."""
        if not self.busy:
            return False
        self._task.cancel()  # type: ignore[union-attr]
        return True

    def set_model(self, model: object) -> None:
        """Switch model, and keep it across later tier/strategy rebuilds."""
        self.model = model
        self.agent.model = model

    def clear(self) -> None:
        self.history = []

    async def run_turn(
        self,
        prompt: str,
        on_event: EventSink,
        ask_approval: ApprovalFn,
    ) -> TurnResult:
        self._partial = []
        self.kwim.begin_turn()
        self._task = asyncio.create_task(self._run(prompt, on_event, ask_approval))
        try:
            result = await asyncio.shield(self._task)
        except asyncio.CancelledError:
            # Only our own interrupt is absorbed. Cancellation from outside
            # leaves the task un-cancelled, so re-raise rather than reporting
            # a normal result to a caller that is shutting down.
            if not self._task.cancelled():
                self._task.cancel()
                self._task = None
                raise
            # The turn was interrupted. Whatever the model managed to produce is
            # already in self.history via capture_run_messages; it just needs the
            # unanswered tool calls stripped before the next request reuses it.
            self.history = drop_dangling_tool_calls(self.history)
            result = TurnResult(text="".join(self._partial), interrupted=True)
        finally:
            self._task = None

        # Emitted after the try/except, so aborted and failed turns land in
        # KWIM too.
        await self._record(prompt, result)
        return result

    async def _record(self, prompt: str, result: TurnResult) -> None:
        outcome = (
            "interrupted" if result.interrupted else "error" if result.error else "ok"
        )
        data: dict = {
            "request": prompt,
            "response": result.text.strip(),
            "outcome": outcome,
        }
        if result.error:
            data["error"] = result.error
        recalled = self.kwim.recalled_items()
        if recalled:
            data["recalled"] = recalled
        try:
            await self.kwim.emit_episodic(self.session_id, "cli_turn", data)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass  # memory writes never break a turn

    async def _run(
        self,
        prompt: str,
        on_event: EventSink,
        ask_approval: ApprovalFn,
    ) -> TurnResult:
        text_parts = self._partial
        announced: set[str] = set()

        async def handler(ctx, stream) -> None:
            async for event in stream:
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    text_parts.append(event.part.content)
                    on_event(TextChunk(event.part.content))
                elif isinstance(event, PartDeltaEvent) and isinstance(
                    event.delta, TextPartDelta
                ):
                    text_parts.append(event.delta.content_delta)
                    on_event(TextChunk(event.delta.content_delta))
                elif isinstance(event, FunctionToolCallEvent):
                    # A turn that suspends for approval replays the same call on
                    # resume; only announce it once.
                    if event.part.tool_call_id in announced:
                        continue
                    announced.add(event.part.tool_call_id)
                    on_event(ToolStarted(event.part.tool_name, _compact(event.part.args)))
                elif isinstance(event, FunctionToolResultEvent):
                    content = str(getattr(event.part, "content", event.content))
                    # Display is truncated; the full text is carried alongside so
                    # /save and /copy can produce something worth pasting into a
                    # bug report.
                    on_event(
                        ToolFinished(
                            _compact(content, 120),
                            _is_error_result(event.part, content),
                            full=content,
                        )
                    )

        # Passed into every `agent.run` below so the limit spans the whole turn. A
        # gated write tool ends an `agent.run`, so a fresh counter per run would never
        # reach `max_requests_per_turn`.
        usage = RunUsage()
        # capture_run_messages keeps the partial transcript reachable even if the
        # run raises or is cancelled - without it, an interrupted turn is lost.
        with capture_run_messages() as messages:
            try:
                result = await self.agent.run(
                    prompt,
                    message_history=self.history,
                    deps=self.deps,
                    output_type=[str, DeferredToolRequests],
                    event_stream_handler=handler,
                    usage_limits=self.usage_limits,
                    usage=usage,
                )

                while isinstance(result.output, DeferredToolRequests):
                    decisions = DeferredToolResults()
                    for call in result.output.approvals:
                        meta = result.output.metadata.get(call.tool_call_id, {})
                        detail = meta.get("command", _compact(call.args, 200))
                        approved = await ask_approval(call.tool_name, detail)
                        decisions.approvals[call.tool_call_id] = (
                            ToolApproved()
                            if approved
                            else ToolDenied("Operator denied this command.")
                        )
                    result = await self.agent.run(
                        message_history=result.all_messages(),
                        deferred_tool_results=decisions,
                        deps=self.deps,
                        output_type=[str, DeferredToolRequests],
                        event_stream_handler=handler,
                        usage_limits=self.usage_limits,
                        usage=usage,
                    )

                # Decayed when added, so every later request, and any resumed turn, gets the
                # bounded form.
                self.history = decay(result.all_messages())
                text = str(result.output)
                trace = trace_requests(self.history)
            except asyncio.CancelledError:
                self.history = decay(drop_dangling_tool_calls(list(messages)))
                raise
            except Exception as e:
                # Traced from the captured messages, not from `usage`: the request that raised
                # never reached the accumulator.
                self.history = decay(drop_dangling_tool_calls(list(messages)))
                return TurnResult(
                    text="".join(text_parts),
                    error=f"{type(e).__name__}: {e}",
                    usage=usage,
                    trace=trace_requests(list(messages)),
                )

        return TurnResult(text=text, usage=usage, trace=trace)


def _compact(v: object, n: int = 80) -> str:
    s = v if isinstance(v, str) else repr(v)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "..."
