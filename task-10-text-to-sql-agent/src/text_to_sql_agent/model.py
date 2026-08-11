"""OpenAI model configuration for the Text-to-SQL agent."""

import os

from langchain_openai import ChatOpenAI


DEFAULT_MODEL = "gpt-4o-mini"


def create_openai_model(model_name: str = DEFAULT_MODEL) -> ChatOpenAI:
    """Create the deterministic OpenAI chat model used by the agent."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add it to your environment "
            "before running the agent."
        )

    return ChatOpenAI(
        model=model_name,
        temperature=0,
        max_retries=2,
        timeout=30,
    )
