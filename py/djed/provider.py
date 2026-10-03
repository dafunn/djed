"""KWIM gateway model for Pydantic AI.

Points a Pydantic AI model at an OpenAI-compatible LiteLLM gateway, so the
agent drives whatever local model LiteLLM is serving. Per-agent spend is
attributed via the `x-litellm-tags` header (lands in LiteLLM_SpendLogs).
"""
from __future__ import annotations

import os

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from .persona import default_persona

DEFAULT_BASE_URL = "http://localhost:4000/v1"

# Config is resolved at call time, never at import, so a later `--env-file`
# load is seen.


def base_url() -> str:
    return os.environ.get("KWIM_LLM_BASE_URL") or DEFAULT_BASE_URL


def model_id() -> str:
    """The gateway model id, from `DJED_MODEL`. No default - the id has to
    match what the gateway serves."""
    model = os.environ.get("DJED_MODEL", "").strip()
    if not model:
        raise RuntimeError(
            "DJED_MODEL is not set - it must name a model your gateway serves, "
            "exactly as the gateway knows it. Set it in your env file "
            "(--env-file), or pass --model. Use --list-models to see what the "
            "gateway offers, or --fake to run without a gateway at all."
        )
    return model


# LiteLLM proxies to Ollama / vLLM / etc., and many OpenAI-compatible servers
# reject the `developer` role, `reasoning_effort`, and strict tool schemas.
# Hence these compat flags, plus the strict-tool-schema opt-out that local
# backends usually need.
LOCAL_MODEL_OVERRIDES = OpenAIModelProfile(
    openai_system_prompt_role="system",
    openai_supports_reasoning=False,
    openai_supports_strict_tool_definition=False,
)


def local_model_profile(inherited: ModelProfile) -> ModelProfile:
    """Layer the local-backend compat flags over whatever the provider inferred.

    Profiles are TypedDicts, so handing one over directly would replace the
    provider's profile rather than override it. The callable form merges.
    """
    return {**inherited, **LOCAL_MODEL_OVERRIDES}


def disable_thinking() -> bool:
    """Whether to ask the backend to generate without a reasoning block. Off by
    default.
    """
    raw = (os.environ.get("DJED_DISABLE_THINKING") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def model_settings() -> ModelSettings:
    """Per-request settings for the gateway. Uses `chat_template_kwargs`
    (`enable_thinking`), which vLLM and Ollama read; a backend that does not know
    the key ignores it.
    """
    max_tokens = int(os.environ.get("DJED_MAX_TOKENS") or 8000)
    if not disable_thinking():
        return ModelSettings(max_tokens=max_tokens)
    return OpenAIChatModelSettings(
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def build_model(model: str | None = None) -> OpenAIChatModel:
    """Resolve the LiteLLM-backed model to hand the Agent."""
    agent_tag = default_persona().tag
    client = AsyncOpenAI(
        base_url=base_url(),
        api_key=os.environ.get("DJED_LITELLM_KEY", ""),
        # Per-agent spend attribution - LiteLLM records this in LiteLLM_SpendLogs.
        default_headers={"x-litellm-tags": f"agent:{agent_tag}"},
    )
    return OpenAIChatModel(
        model or model_id(),
        provider=OpenAIProvider(openai_client=client),
        profile=local_model_profile,
        settings=model_settings(),
    )


async def list_models() -> list[str]:
    """List the model ids the gateway is serving. Fail-soft: [] on any error."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(
                f"{base_url().rstrip('/')}/models",
                headers={"Authorization": f"Bearer {os.environ.get('DJED_LITELLM_KEY', '')}"},
            )
            if r.status_code != 200:
                return []
            return sorted(m["id"] for m in r.json().get("data", []))
    except Exception:
        return []
