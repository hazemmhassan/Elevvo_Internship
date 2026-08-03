"""Configurable chat-model providers for candidate explanations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os


DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-5.6"


class ProviderConfigurationError(ValueError):
    """Raised when an LLM provider cannot be configured safely."""


@dataclass(frozen=True, slots=True)
class LlmSettings:
    """Validated provider settings without logging the secret value."""

    provider: str
    model: str
    api_key: str


def resolve_llm_settings(
    environment: Mapping[str, str] | None = None,
) -> LlmSettings:
    """Resolve the supported provider and its required secret."""

    values = os.environ if environment is None else environment
    provider = values.get(
        "TALENT_SEARCH_LLM_PROVIDER", DEFAULT_LLM_PROVIDER
    ).strip().casefold()
    if provider != "openai":
        raise ProviderConfigurationError(
            f"Unsupported LLM provider: {provider or '<blank>'}."
        )
    model = values.get("TALENT_SEARCH_LLM_MODEL", DEFAULT_OPENAI_MODEL).strip()
    if not model:
        raise ProviderConfigurationError("TALENT_SEARCH_LLM_MODEL cannot be blank.")
    api_key = values.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ProviderConfigurationError(
            "OPENAI_API_KEY is required for LLM candidate explanations."
        )
    return LlmSettings(provider=provider, model=model, api_key=api_key)


def create_chat_model(settings: LlmSettings | None = None):
    """Create the configured deterministic LangChain chat model."""

    from langchain_openai import ChatOpenAI

    resolved = settings or resolve_llm_settings()
    return ChatOpenAI(
        model=resolved.model,
        api_key=resolved.api_key,
        temperature=0,
    )
