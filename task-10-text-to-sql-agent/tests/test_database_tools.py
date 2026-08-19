from contextlib import closing
import sqlite3

from text_to_sql_agent import database_tools


def test_list_database_tables_tool_invokes_real_database(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE zeta (id INTEGER PRIMARY KEY)")
        connection.execute("CREATE TABLE alpha (id INTEGER PRIMARY KEY)")

    tools_by_name = {
        tool.name: tool for tool in database_tools.create_database_tools(database_path)
    }
    result = tools_by_name["list_database_tables"].invoke({})

    assert result == {"tables": ("alpha", "zeta")}


def test_inspect_table_schema_tool_returns_real_metadata(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        connection.execute(
            """
            CREATE TABLE child (
                id INTEGER PRIMARY KEY,
                parent_id INTEGER,
                FOREIGN KEY (parent_id) REFERENCES parent (id)
            )
            """
        )

    tools_by_name = {
        tool.name: tool for tool in database_tools.create_database_tools(database_path)
    }
    result = tools_by_name["inspect_table_schema"].invoke({"table_name": "child"})

    assert result["table"] == "child"
    assert tuple(column["name"] for column in result["columns"]) == (
        "id",
        "parent_id",
    )
    assert result["foreign_keys"] == (
        {
            "column": "parent_id",
            "referenced_table": "parent",
            "referenced_column": "id",
        },
    )


def test_run_read_only_query_tool_returns_controlled_result(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO item (id) VALUES (?)",
            ((1,), (2,)),
        )
        connection.commit()

    tools_by_name = {
        tool.name: tool for tool in database_tools.create_database_tools(database_path)
    }
    result = tools_by_name["run_read_only_query"].invoke(
        {
            "sql": "SELECT id FROM item ORDER BY id",
            "max_rows": 1,
        }
    )

    assert result == {
        "success": True,
        "columns": ("id",),
        "rows": ((1,),),
        "row_count": 1,
        "truncated": True,
        "error": None,
    }


def test_database_tool_releases_file_after_invocation(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY)")
        connection.commit()

    tools_by_name = {
        tool.name: tool for tool in database_tools.create_database_tools(database_path)
    }
    tools_by_name["list_database_tables"].invoke({})

    database_path.unlink()

    assert database_path.exists() is False


def test_create_chart_tool_returns_display_ready_data(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE sale (month TEXT, revenue REAL)")
        connection.executemany(
            "INSERT INTO sale (month, revenue) VALUES (?, ?)",
            (("January", 10.0), ("February", 15.5)),
        )

    tools_by_name = {
        tool.name: tool for tool in database_tools.create_database_tools(database_path)
    }
    result = tools_by_name["create_chart"].invoke(
        {
            "sql": "SELECT month, revenue FROM sale ORDER BY rowid",
            "chart_type": "bar",
            "x_column": "month",
            "y_column": "revenue",
        }
    )

    assert result == {
        "success": True,
        "chart": {
            "type": "bar",
            "x": "month",
            "y": "revenue",
            "data": (
                {"month": "January", "revenue": 10.0},
                {"month": "February", "revenue": 15.5},
            ),
        },
        "error": None,
    }


def test_create_chart_tool_rejects_unknown_axis_column(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE sale (month TEXT, revenue REAL)")

    tools_by_name = {
        tool.name: tool for tool in database_tools.create_database_tools(database_path)
    }
    result = tools_by_name["create_chart"].invoke(
        {
            "sql": "SELECT month, revenue FROM sale",
            "chart_type": "line",
            "x_column": "month",
            "y_column": "missing",
        }
    )

    assert result == {
        "success": False,
        "chart": None,
        "error": "Chart columns must exist in the query result.",
    }
