from __future__ import annotations

import pytest

from talent_search.providers import (
    ProviderConfigurationError,
    resolve_llm_settings,
)


def test_openai_is_the_configurable_default_provider(monkeypatch) -> None:
    monkeypatch.delenv("TALENT_SEARCH_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("TALENT_SEARCH_LLM_MODEL", raising=False)

    settings = resolve_llm_settings({"OPENAI_API_KEY": "test-key"})

    assert settings.provider == "openai"
    assert settings.model == "gpt-5.6"
    assert settings.api_key == "test-key"


def test_provider_settings_require_an_api_key() -> None:
    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
        resolve_llm_settings({})


def test_provider_settings_reject_unsupported_providers() -> None:
    with pytest.raises(ProviderConfigurationError, match="Unsupported"):
        resolve_llm_settings(
            {
                "TALENT_SEARCH_LLM_PROVIDER": "unknown",
                "OPENAI_API_KEY": "test-key",
            }
        )
