"""Gemini model configuration for the Text-to-SQL agent."""

import os

from langchain_google_genai import ChatGoogleGenerativeAI


DEFAULT_MODEL = "gemini-3.1-flash-lite"


def create_gemini_model(model_name: str = DEFAULT_MODEL) -> ChatGoogleGenerativeAI:
    """Create the deterministic Gemini chat model used by the agent."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to your environment "
            "before running the agent."
        )

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
        max_retries=2,
    )
