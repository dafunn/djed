"""Assembles the agent - model, memory, tools, and a persona.

Everything here is persona-agnostic: which agent this becomes is the `persona`
argument, not anything baked into the wiring. See `djed/persona.py`.
"""
from __future__ import annotations

import logging
import os

import httpx
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.models import Model

from .kwim import KwimClient, kwim_base, make_transform_context
from .persona import MAAT, Persona
from .provider import build_model
from .skills import Skill, index_instructions
from .tools import SKILL_TOOLS, Deps, kwim_recall, run_command

log = logging.getLogger("djed.agent")


def kwim_config_warning() -> str | None:
    """Why memory will not work as configured, or None.

    A KWIM URL with no key builds a disabled client, so without this the
    session would run with no memory and say nothing. Neither set means KWIM is
    not configured, which needs no warning."""
    if os.environ.get("KWIM_BASE_URL") and not os.environ.get("DJED_KWIM_API_KEY"):
        return "KWIM_BASE_URL is set but DJED_KWIM_API_KEY is empty - memory is disabled"
    return None


def build_kwim_client(http: httpx.AsyncClient) -> KwimClient:
    """The session's KWIM client. With no key it is built disabled and makes no
    calls to KWIM."""
    warning = kwim_config_warning()
    if warning:
        log.warning(warning)
    api_key = os.environ.get("DJED_KWIM_API_KEY", "")
    return KwimClient(
        http,
        base_url=kwim_base(),
        api_key=api_key,
        disabled=None if api_key else "DJED_KWIM_API_KEY is empty - memory is disabled",
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
    skills: dict[str, Skill] | None = None,
) -> Agent[Deps]:
    """Build the agent.

    `ProcessHistory` runs on the message list before every model request.
    MCP toolsets are passed in rather than loaded here - discovering them
    requires connecting to the fleet, which is async and fail-soft.

    `tools` selects the native tool surface: the ops pair by default, and the
    build driver's own set, which omits `run_command`.

    `skills` adds their index to the instructions and the two skill tools. Pass the
    same dict as `Deps.skills`, which is what the tools read.
    """
    toolsets = toolsets or []
    tools = [kwim_recall, run_command] if tools is None else list(tools)
    if skills:
        tools += SKILL_TOOLS
    # ProcessHistory is always first: retrieval runs before anything else looks
    # at the message list.
    caps = [ProcessHistory(make_transform_context(kwim, session_id)), *(capabilities or [])]
    agent = Agent(
        model if model is not None else build_model(),
        deps_type=Deps,
        instructions=persona.prompt + extra_instructions + index_instructions(skills or {}),
        tools=tools,
        toolsets=toolsets,
        capabilities=caps,
    )
    return agent
