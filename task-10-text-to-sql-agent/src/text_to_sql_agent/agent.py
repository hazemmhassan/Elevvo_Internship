"""LangChain agent orchestration for natural-language database questions."""

from dataclasses import dataclass
import json
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from google.genai.errors import APIError

from text_to_sql_agent.database_tools import create_database_tools
from text_to_sql_agent.model import create_gemini_model


SYSTEM_PROMPT = """
You are a careful data analyst working with a read-only SQLite database.

For every database question:
1. List the available tables before assuming table names.
2. Inspect the schemas of every relevant table before writing SQL.
3. Use only the run_read_only_query tool to execute one SQLite SELECT query.
4. If the query tool returns success=false, read its error, correct the SQL, and
   retry. Make at most three SQL attempts and never repeat an unchanged query.
5. After a successful query, answer in plain English using only returned data.
   Include the final SQL query in a fenced sql block so the work is auditable.
6. If the user explicitly asks for a visualization, call create_chart after
   inspecting the schema. Use a bar chart for category comparisons and a line
   chart for ordered time series. Still summarize the chart in words.

Never invent tables, columns, values, or query results. Never attempt to insert,
update, delete, drop, alter, attach, detach, or run a PRAGMA statement. If the
question is ambiguous, explain what clarification is needed.
""".strip()


@dataclass(frozen=True)
class AgentRun:
    """User-facing answer plus an auditable record of SQL attempts."""

    answer: str
    sql_attempts: tuple[str, ...]
    query_results: tuple[dict[str, object], ...]
    chart: dict[str, object] | None

    @property
    def corrected(self) -> bool:
        """Whether a failed SQL attempt was followed by another attempt."""
        return (
            len(self.sql_attempts) > 1
            and any(not result.get("success", False) for result in self.query_results[:-1])
        )


class AgentProviderError(RuntimeError):
    """Raised when the external model provider cannot complete a request."""


def create_text_to_sql_agent(
    database_path: Path,
    model: BaseChatModel | None = None,
):
    """Build a LangChain tool-calling agent for one SQLite database."""
    active_model = model or create_gemini_model()
    return create_agent(
        model=active_model,
        tools=create_database_tools(database_path),
        system_prompt=SYSTEM_PROMPT,
        name="text_to_sql_agent",
    )


def run_text_to_sql_agent(
    question: str,
    database_path: Path,
    model: BaseChatModel | None = None,
) -> AgentRun:
    """Run one question through the agent and return answer and audit data."""
    if not question.strip():
        raise ValueError("question must not be blank")

    agent = create_text_to_sql_agent(database_path, model=model)
    try:
        state = agent.invoke(
            {"messages": [{"role": "user", "content": question.strip()}]},
            config={"recursion_limit": 24},
        )
    except APIError as error:
        raise AgentProviderError(f"Gemini request failed: {error}") from error

    sql_attempts: list[str] = []
    query_results: list[dict[str, object]] = []
    chart = None
    for message in state["messages"]:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                if tool_call["name"] == "run_read_only_query":
                    sql_attempts.append(tool_call["args"]["sql"])
        elif isinstance(message, ToolMessage) and message.name == "run_read_only_query":
            query_results.append(json.loads(message.content))
        elif isinstance(message, ToolMessage) and message.name == "create_chart":
            chart_result = json.loads(message.content)
            if chart_result.get("success"):
                chart = chart_result["chart"]

    final_message = state["messages"][-1]
    if not isinstance(final_message, AIMessage):
        raise RuntimeError("Agent finished without an answer.")

    return AgentRun(
        answer=_message_text(final_message),
        sql_attempts=tuple(sql_attempts),
        query_results=tuple(query_results),
        chart=chart,
    )


def _message_text(message: AIMessage) -> str:
    """Normalize string or content-block model output to displayable text."""
    if isinstance(message.content, str):
        return message.content

    return "\n".join(
        block["text"]
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    )
