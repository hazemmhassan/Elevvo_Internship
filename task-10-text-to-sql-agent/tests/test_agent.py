import json
import sqlite3

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import Field

class RecordingToolCallingModel(FakeMessagesListChatModel):
    """Deterministic model double that records the real agent message loop."""

    seen_messages: list[list[object]] = Field(default_factory=list, exclude=True)
    bound_tool_names: tuple[str, ...] = Field(default=(), exclude=True)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound_tool_names = tuple(tool.name for tool in tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_messages.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_create_openai_model_requires_api_key(monkeypatch):
    try:
        from text_to_sql_agent.model import create_openai_model
    except ModuleNotFoundError:
        pytest.fail("model layer has not been implemented", pytrace=False)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        create_openai_model()


def test_create_openai_model_uses_deterministic_defaults(monkeypatch):
    try:
        from text_to_sql_agent.model import create_openai_model
    except ModuleNotFoundError:
        pytest.fail("model layer has not been implemented", pytrace=False)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-valid-for-network-calls")

    model = create_openai_model()

    assert model.model_name == "gpt-4o-mini"
    assert model.temperature == 0
    assert model.max_retries == 2


def test_agent_receives_sql_error_and_executes_corrected_query(tmp_path):
    try:
        from text_to_sql_agent import agent as agent_module
    except ImportError:
        pytest.fail("agent loop has not been implemented", pytrace=False)

    database_path = tmp_path / "sample.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE sale (amount REAL NOT NULL)")
        connection.executemany(
            "INSERT INTO sale (amount) VALUES (?)",
            ((10.0,), (15.5,)),
        )

    model = RecordingToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_database_tables",
                        "args": {},
                        "id": "tables-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "inspect_table_schema",
                        "args": {"table_name": "sale"},
                        "id": "schema-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_read_only_query",
                        "args": {"sql": "SELECT SUM(total) FROM sale"},
                        "id": "bad-query-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_read_only_query",
                        "args": {
                            "sql": "SELECT SUM(amount) AS revenue FROM sale"
                        },
                        "id": "corrected-query-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Total revenue is $25.50."),
        ]
    )

    run = agent_module.run_text_to_sql_agent(
        "What is the total revenue?",
        database_path,
        model=model,
    )

    assert model.bound_tool_names == (
        "list_database_tables",
        "inspect_table_schema",
        "run_read_only_query",
        "create_chart",
    )
    assert run.answer == "Total revenue is $25.50."
    assert run.sql_attempts == (
        "SELECT SUM(total) FROM sale",
        "SELECT SUM(amount) AS revenue FROM sale",
    )
    assert run.query_results[0]["success"] is False
    assert run.query_results[0]["error"] == "no such column: total"
    assert run.query_results[1]["rows"] == [[25.5]]
    assert run.corrected is True
    assert run.chart is None

    error_was_returned_to_model = any(
        isinstance(message, ToolMessage)
        and message.name == "run_read_only_query"
        and json.loads(message.content)["success"] is False
        for message in model.seen_messages[3]
    )
    assert error_was_returned_to_model is True


def test_run_text_to_sql_agent_rejects_blank_question(tmp_path):
    try:
        from text_to_sql_agent import agent as agent_module
    except ImportError:
        pytest.fail("agent loop has not been implemented", pytrace=False)

    with pytest.raises(ValueError, match="question must not be blank"):
        agent_module.run_text_to_sql_agent("   ", tmp_path / "unused.sqlite")


def test_run_text_to_sql_agent_wraps_openai_connection_errors(monkeypatch, tmp_path):
    from openai import APIConnectionError
    from text_to_sql_agent import agent as agent_module

    class FailingAgent:
        def invoke(self, *args, **kwargs):
            raise APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))

    monkeypatch.setattr(
        agent_module,
        "create_text_to_sql_agent",
        lambda *args, **kwargs: FailingAgent(),
    )

    with pytest.raises(agent_module.AgentProviderError, match="OpenAI request failed"):
        agent_module.run_text_to_sql_agent("Show revenue", tmp_path / "unused.sqlite")
