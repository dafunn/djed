"""End-to-end tests of the agent loop against a scripted stub model: a tool
call round-trips, the approval gate suspends, approve and deny both resume.
"""
from __future__ import annotations

import pytest
from pydantic_ai import (
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
)
from pydantic_ai.messages import ModelRequest, ToolReturnPart

from fake_model import scripted_model
from djed.agent import build_agent
from djed.tools import Deps
from tests.test_kwim import StubKwim


def make_agent():
    kwim = StubKwim(recall=["[monitoring | 0.90]\nmetrics is in ns monitoring"])
    agent = build_agent(kwim, "test-session", model=scripted_model())
    return agent, Deps(kwim=kwim, session_id="test-session")


async def run_turn(agent, deps, prompt, approve: bool):
    result = await agent.run(prompt, deps=deps, output_type=[str, DeferredToolRequests])
    suspensions = 0
    while isinstance(result.output, DeferredToolRequests):
        suspensions += 1
        decisions = DeferredToolResults()
        for call in result.output.approvals:
            decisions.approvals[call.tool_call_id] = (
                ToolApproved() if approve else ToolDenied("Operator denied this command.")
            )
        result = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=decisions,
            deps=deps,
            output_type=[str, DeferredToolRequests],
        )
    return result, suspensions


def tool_returns(result) -> dict[str, str]:
    out = {}
    for m in result.all_messages():
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, ToolReturnPart):
                    out[p.tool_name] = str(p.content)
    return out


@pytest.mark.asyncio
async def test_approved_command_runs_and_returns_output():
    agent, deps = make_agent()
    result, suspensions = await run_turn(agent, deps, "check the cluster", approve=True)

    assert suspensions == 1, "run_command must suspend for operator approval exactly once"
    assert isinstance(result.output, str)
    returns = tool_returns(result)
    assert "metrics is in ns monitoring" in returns["kwim_recall"]
    assert returns["run_command"] == "djed-roundtrip-ok"


@pytest.mark.asyncio
async def test_denied_command_does_not_run():
    agent, deps = make_agent()
    result, suspensions = await run_turn(agent, deps, "check the cluster", approve=False)

    assert suspensions == 1
    returns = tool_returns(result)
    assert "djed-roundtrip-ok" not in returns.get("run_command", "")
    assert "denied" in returns["run_command"].lower()


@pytest.mark.asyncio
async def test_history_processor_injects_recall_into_the_model_request():
    """The retrieved-context block reaches the model. It is applied to the outgoing
    request, not stored in history, so it is asserted inside the model.
    """
    from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.models.function import FunctionModel

    from djed.kwim import INJECT_HEADER

    received: list[str] = []

    def capture(messages, info):
        for m in messages:
            if isinstance(m, ModelRequest):
                for p in m.parts:
                    if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                        received.append(p.content)
        return ModelResponse(parts=[TextPart(content="ack")])

    kwim = StubKwim(recall=["[monitoring | 0.90]\nmetrics is in ns monitoring"])
    agent = build_agent(kwim, "test-session", model=FunctionModel(capture))
    await agent.run("where does metrics live?", deps=Deps(kwim=kwim, session_id="s"))

    assert any(INJECT_HEADER in s for s in received), "KWIM context never reached the model"
    assert any("metrics is in ns monitoring" in s for s in received)
    assert any(s == "where does metrics live?" for s in received)
