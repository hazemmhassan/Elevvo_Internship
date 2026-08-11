"""Streamlit chat interface for the autonomous Text-to-SQL agent."""

import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from text_to_sql_agent.agent import AgentProviderError, run_text_to_sql_agent


DATABASE_PATH = PROJECT_ROOT / "data" / "raw" / "Chinook_Sqlite.sqlite"


st.set_page_config(
    page_title="Text-to-SQL Agent",
    page_icon="🔎",
    layout="centered",
)
st.title("Autonomous Text-to-SQL Agent")
st.markdown(
    "Ask a business question about the Chinook music-store database. "
    "Every database operation is **read-only**, validated, and limited to 100 rows."
)

with st.sidebar:
    st.header("How the pipeline works")
    st.markdown(
        "1. Understand the question\n"
        "2. Discover tables and schemas\n"
        "3. Generate and validate SQL\n"
        "4. Execute the query\n"
        "5. Correct SQL errors automatically\n"
        "6. Summarize the verified result"
    )
    st.divider()
    st.caption("Try asking:")
    st.code("Which customer has spent the most in total?")
    st.code("Show revenue by country as a bar chart.")

database_ready = DATABASE_PATH.exists()
key_ready = bool(os.environ.get("OPENAI_API_KEY"))

if not database_ready:
    st.error(
        "Chinook database not found. Place it at "
        "`data/raw/Chinook_Sqlite.sqlite`."
    )
if not key_ready:
    st.warning("Set the `OPENAI_API_KEY` environment variable, then restart the app.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_chart(chart: dict[str, object]) -> None:
    """Render a chart specification returned by the safe chart tool."""
    frame = pd.DataFrame(chart["data"])
    if frame.empty:
        st.info("The chart query returned no rows.")
        return

    chart_frame = frame.set_index(str(chart["x"]))[[str(chart["y"])]]
    if chart["type"] == "line":
        st.line_chart(chart_frame)
    else:
        st.bar_chart(chart_frame)


def render_assistant_message(message: dict[str, object]) -> None:
    """Render one answer and its optional audit details."""
    st.markdown(str(message["content"]))
    sql_attempts = message.get("sql_attempts", ())
    if sql_attempts:
        with st.expander("SQL audit trail"):
            for index, sql in enumerate(sql_attempts, start=1):
                st.caption(f"Attempt {index}")
                st.code(sql, language="sql")
    if message.get("chart"):
        render_chart(message["chart"])


for stored_message in st.session_state.messages:
    with st.chat_message(stored_message["role"]):
        if stored_message["role"] == "assistant":
            render_assistant_message(stored_message)
        else:
            st.markdown(stored_message["content"])

question = st.chat_input(
    "Ask a question about customers, invoices, tracks, or sales...",
    disabled=not (database_ready and key_ready),
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Inspecting the database and building a safe query..."):
            try:
                run = run_text_to_sql_agent(question, DATABASE_PATH)
            except AgentProviderError as error:
                error_text = str(error)
                if "credit_balance_exhausted" in error_text or "no credits" in error_text:
                    answer = (
                        "The OpenAI API account has no credits available. Add API "
                        "credit in the OpenAI Platform billing page, then try again."
                    )
                else:
                    answer = f"The OpenAI request could not be completed: {error_text}"
                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "sql_attempts": (),
                    "chart": None,
                }
            except Exception as error:
                assistant_message = {
                    "role": "assistant",
                    "content": f"The request could not be completed: {error}",
                    "sql_attempts": (),
                    "chart": None,
                }
            else:
                assistant_message = {
                    "role": "assistant",
                    "content": run.answer,
                    "sql_attempts": run.sql_attempts,
                    "chart": run.chart,
                }

        render_assistant_message(assistant_message)
        st.session_state.messages.append(assistant_message)
