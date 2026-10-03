"""Gateway model construction and the local-backend compat profile."""
from __future__ import annotations

import os

from djed.persona import MAAT
import pytest

from djed import provider
from djed.provider import build_model, local_model_profile, model_id


def test_compat_flags_are_applied():
    profile = local_model_profile({})
    # The compat flags local OpenAI-compatible backends need.
    assert profile["openai_system_prompt_role"] == "system"
    assert profile["openai_supports_reasoning"] is False
    assert profile["openai_supports_strict_tool_definition"] is False


def test_compat_flags_override_rather_than_replace():
    """The profile override keeps the provider's inferred keys."""
    inherited = {"supports_tools": True, "openai_supports_reasoning": True}
    profile = local_model_profile(inherited)
    assert profile["supports_tools"] is True
    assert profile["openai_supports_reasoning"] is False


def test_model_points_at_the_configured_gateway(monkeypatch):
    monkeypatch.setenv("DJED_LITELLM_KEY", "dummy")
    monkeypatch.setenv("DJED_MODEL", "test-model")
    model = build_model()
    assert model.base_url.rstrip("/") == os.environ.get(
        "KWIM_LLM_BASE_URL", provider.DEFAULT_BASE_URL
    ).rstrip("/")
    assert model.profile["openai_system_prompt_role"] == "system"


def test_model_id_can_be_overridden(monkeypatch):
    monkeypatch.setenv("DJED_LITELLM_KEY", "dummy")
    assert build_model("some-other:7b").model_name == "some-other:7b"


def test_spend_attribution_header_is_set(monkeypatch):
    monkeypatch.setenv("DJED_LITELLM_KEY", "dummy")
    monkeypatch.setenv("DJED_MODEL", "test-model")
    monkeypatch.setenv("DJED_AGENT", MAAT.tag)
    model = build_model()
    headers = model.client.default_headers
    assert headers["x-litellm-tags"] == f"agent:{MAAT.tag}"


def test_model_id_requires_djed_model(monkeypatch):
    """An unset DJED_MODEL fails here, not as a 404 mid-turn."""
    monkeypatch.delenv("DJED_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="DJED_MODEL is not set"):
        model_id()


def test_model_id_treats_blank_as_unset(monkeypatch):
    """An exported-but-empty DJED_MODEL is a common env-file accident."""
    monkeypatch.setenv("DJED_MODEL", "   ")
    with pytest.raises(RuntimeError, match="DJED_MODEL is not set"):
        model_id()


def test_model_id_returns_the_configured_id(monkeypatch):
    monkeypatch.setenv("DJED_MODEL", "some-model:7b")
    assert model_id() == "some-model:7b"


def test_explicit_model_does_not_need_the_env(monkeypatch):
    """--model / --fake paths must work without DJED_MODEL set."""
    monkeypatch.delenv("DJED_MODEL", raising=False)
    monkeypatch.setenv("DJED_LITELLM_KEY", "dummy")
    assert build_model("explicit:7b").model_name == "explicit:7b"


# --- thinking ------------------------------------------------------------


def test_thinking_is_on_unless_asked_otherwise(monkeypatch):
    """Off by default."""
    monkeypatch.delenv("DJED_DISABLE_THINKING", raising=False)
    assert provider.disable_thinking() is False
    assert "extra_body" not in provider.model_settings()


def test_disabling_thinking_reaches_the_backend_as_a_template_kwarg(monkeypatch):
    """Sent as `chat_template_kwargs.enable_thinking`, not the OpenAI `thinking` key."""
    monkeypatch.setenv("DJED_DISABLE_THINKING", "1")
    monkeypatch.setenv("DJED_MAX_TOKENS", "4096")
    settings = provider.model_settings()
    assert settings["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    # The ceiling still has to survive the branch.
    assert settings["max_tokens"] == 4096


def test_only_an_affirmative_value_disables_thinking(monkeypatch):
    """Empty or unset means off, and `0` means off."""
    for raw in ("", "0", "false", "no", "off"):
        monkeypatch.setenv("DJED_DISABLE_THINKING", raw)
        assert provider.disable_thinking() is False, raw
    for raw in ("1", "true", "YES", "On"):
        monkeypatch.setenv("DJED_DISABLE_THINKING", raw)
        assert provider.disable_thinking() is True, raw
