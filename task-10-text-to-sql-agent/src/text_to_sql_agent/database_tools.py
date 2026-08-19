"""LangChain tools for controlled Chinook database access."""

from contextlib import closing
from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import BaseTool

from text_to_sql_agent.database import (
    execute_query,
    get_table_schema,
    list_tables,
    open_read_only_connection,
)


def create_database_tools(database_path: Path) -> tuple[BaseTool, ...]:
    """Create database tools bound to one local SQLite database."""

    @tool
    def list_database_tables() -> dict[str, object]:
        """List available database tables. Use this before inspecting schemas."""
        with closing(open_read_only_connection(database_path)) as connection:
            return {"tables": list_tables(connection)}

    @tool
    def inspect_table_schema(table_name: str) -> dict[str, object]:
        """Inspect a table's columns and foreign keys before writing SQL."""
        with closing(open_read_only_connection(database_path)) as connection:
            return get_table_schema(connection, table_name)

    @tool
    def run_read_only_query(
        sql: str,
        max_rows: int = 100,
    ) -> dict[str, object]:
        """Run one validated read-only SQL query and return structured results."""
        with closing(open_read_only_connection(database_path)) as connection:
            return execute_query(connection, sql, max_rows=max_rows)

    @tool
    def create_chart(
        sql: str,
        chart_type: str,
        x_column: str,
        y_column: str,
    ) -> dict[str, object]:
        """Create a bar or line chart from one validated read-only query."""
        if chart_type not in {"bar", "line"}:
            return {
                "success": False,
                "chart": None,
                "error": "chart_type must be 'bar' or 'line'.",
            }

        with closing(open_read_only_connection(database_path)) as connection:
            query_result = execute_query(connection, sql, max_rows=50)

        if not query_result["success"]:
            return {
                "success": False,
                "chart": None,
                "error": query_result["error"],
            }

        columns = query_result["columns"]
        if x_column not in columns or y_column not in columns:
            return {
                "success": False,
                "chart": None,
                "error": "Chart columns must exist in the query result.",
            }

        data = tuple(
            dict(zip(columns, row, strict=True)) for row in query_result["rows"]
        )
        return {
            "success": True,
            "chart": {
                "type": chart_type,
                "x": x_column,
                "y": y_column,
                "data": data,
            },
            "error": None,
        }

    return (
        list_database_tables,
        inspect_table_schema,
        run_read_only_query,
        create_chart,
    )
