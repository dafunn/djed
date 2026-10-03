"""Persona resolution: the prompt reaching the model comes from the persona,
and the durable tag resolves in one place so spend and memory cannot disagree.
"""
from __future__ import annotations

from djed.persona import MAAT, PERSONAS, Persona, default_persona


def test_default_persona_is_registered():
    assert PERSONAS[MAAT.tag] is MAAT
    assert MAAT.name and MAAT.tag and MAAT.prompt
    assert MAAT.name in MAAT.prompt


def test_defaults_to_the_registered_persona_when_unset(monkeypatch):
    monkeypatch.delenv("DJED_AGENT", raising=False)
    assert default_persona() is MAAT


def test_registered_tag_selects_its_persona(monkeypatch):
    monkeypatch.setenv("DJED_AGENT", MAAT.tag)
    assert default_persona() is MAAT


def test_unregistered_tag_keeps_the_prompt_but_reattributes(monkeypatch):
    """An unknown DJED_AGENT is used as the tag; it does not fall back to the default
    agent's.
    """
    monkeypatch.setenv("DJED_AGENT", "someone-else")
    p = default_persona()
    assert p.tag == "someone-else"
    assert p.prompt == MAAT.prompt


def test_resolved_at_call_time_not_import_time(monkeypatch):
    """Resolved at call time, so a late `--env-file` load is seen."""
    monkeypatch.setenv("DJED_AGENT", "first")
    assert default_persona().tag == "first"
    monkeypatch.setenv("DJED_AGENT", "second")
    assert default_persona().tag == "second"


def test_empty_tag_falls_back_to_the_registered_persona(monkeypatch):
    """An empty DJED_AGENT falls back to the default tag rather than `agent:`."""
    monkeypatch.setenv("DJED_AGENT", "")
    assert default_persona() is MAAT


def test_agent_uses_the_persona_prompt():
    """The prompt the model sees comes from the persona, not from agent.py."""
    from unittest.mock import Mock

    from djed.agent import build_agent

    other = Persona(name="Stub", prompt="You are a stub.", tag="stub-persona")
    agent = build_agent(Mock(), "s", model="test", persona=other)
    text = "\n".join(agent._instructions)
    assert "You are a stub." in text
    assert MAAT.prompt not in text
    assert MAAT.name not in text


def test_persona_prompt_composes_with_strategy_instructions():
    """`extra_instructions` (the tool-strategy preamble) appends to the persona
    prompt rather than replacing it."""
    from unittest.mock import Mock

    from djed.agent import build_agent

    agent = build_agent(
        Mock(), "s", model="test", extra_instructions="\n\nUse search_tools first."
    )
    text = "\n".join(agent._instructions)
    assert MAAT.prompt in text
    assert "Use search_tools first." in text


def test_spend_tag_and_memory_tag_agree(monkeypatch):
    """provider.py and kwim.py resolve identity through the same function."""
    monkeypatch.setenv("DJED_LITELLM_KEY", "dummy")
    monkeypatch.setenv("DJED_MODEL", "test-model")
    monkeypatch.setenv("DJED_AGENT", "stub-persona")

    from djed.provider import build_model

    model = build_model()
    assert model.client.default_headers["x-litellm-tags"] == "agent:stub-persona"
    assert default_persona().tag == "stub-persona"
