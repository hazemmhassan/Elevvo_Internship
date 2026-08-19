"""Safe presentation helpers for the Streamlit application."""

from html import escape


def highlight_context(context: str, start: int, end: int) -> str:
    """Escape a context and wrap one valid answer span in a mark element."""
    if start < 0 or end <= start or end > len(context):
        raise ValueError("invalid highlight span")

    return (
        escape(context[:start])
        + "<mark>"
        + escape(context[start:end])
        + "</mark>"
        + escape(context[end:])
    )
