"""Assembles the agent - model, memory, tools, and a persona.

Everything here is persona-agnostic: which agent this becomes is the `persona`
argument, not anything baked into the wiring. See `djed/persona.py`.
"""
from __future__ import annotations

import os

import httpx
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.models import Model

from .kwim import KwimClient, kwim_base, make_transform_context
from .persona import MAAT, Persona
from .provider import build_model
from .tools import Deps, kwim_recall, run_command


def build_kwim_client(http: httpx.AsyncClient) -> KwimClient:
    return KwimClient(
        http,
        base_url=kwim_base(),
        api_key=os.environ.get("DJED_KWIM_API_KEY", ""),
    )


def build_agent(
    kwim: KwimClient,
    session_id: str,
    model: Model | str | None = None,
    toolsets: list | None = None,
    capabilities: list | None = None,
    extra_instructions: str = "",
    persona: Persona = MAAT,
    tools: list | None = None,
) -> Agent[Deps]:
    """Build the agent.

    `ProcessHistory` runs on the message list before every model request.
    MCP toolsets are passed in rather than loaded here - discovering them
    requires connecting to the fleet, which is async and fail-soft.

    `tools` selects the native tool surface: the ops pair by default, and the
    build driver's own set, which omits `run_command`.
    """
    toolsets = toolsets or []
    tools = [kwim_recall, run_command] if tools is None else tools
    # ProcessHistory is always first: retrieval runs before anything else looks
    # at the message list.
    caps = [ProcessHistory(make_transform_context(kwim, session_id)), *(capabilities or [])]
    agent = Agent(
        model if model is not None else build_model(),
        deps_type=Deps,
        instructions=persona.prompt + extra_instructions,
        tools=tools,
        toolsets=toolsets,
        capabilities=caps,
    )
    return agent
