import sqlite3

import pytest

from text_to_sql_agent import database


def test_connection_reads_existing_rows(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE message (text TEXT NOT NULL)")
        connection.execute("INSERT INTO message (text) VALUES ('hello')")

    with database.open_read_only_connection(database_path) as connection:
        row = connection.execute("SELECT text FROM message").fetchone()

    assert row == ("hello",)


def test_connection_rejects_writes(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE message (text TEXT NOT NULL)")

    with database.open_read_only_connection(database_path) as connection:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("INSERT INTO message (text) VALUES ('blocked')")


def test_list_tables_returns_sorted_user_tables(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE zeta (id INTEGER PRIMARY KEY AUTOINCREMENT)"
        )
        connection.execute("CREATE TABLE alpha (id INTEGER PRIMARY KEY)")

    with database.open_read_only_connection(database_path) as connection:
        tables = database.list_tables(connection)

    assert tables == ("alpha", "zeta")


def test_get_table_schema_returns_columns_and_foreign_keys(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        connection.execute(
            """
            CREATE TABLE child (
                id INTEGER PRIMARY KEY,
                label TEXT NOT NULL,
                parent_id INTEGER,
                FOREIGN KEY (parent_id) REFERENCES parent (id)
            )
            """
        )

    with database.open_read_only_connection(database_path) as connection:
        schema = database.get_table_schema(connection, "child")

    assert schema == {
        "table": "child",
        "columns": (
            {
                "name": "id",
                "data_type": "INTEGER",
                "not_null": False,
                "primary_key": True,
            },
            {
                "name": "label",
                "data_type": "TEXT",
                "not_null": True,
                "primary_key": False,
            },
            {
                "name": "parent_id",
                "data_type": "INTEGER",
                "not_null": False,
                "primary_key": False,
            },
        ),
        "foreign_keys": (
            {
                "column": "parent_id",
                "referenced_table": "parent",
                "referenced_column": "id",
            },
        ),
    }


def test_get_table_schema_rejects_unknown_table(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE known_table (id INTEGER PRIMARY KEY)")

    with database.open_read_only_connection(database_path) as connection:
        with pytest.raises(ValueError, match="Unknown table: missing_table"):
            database.get_table_schema(connection, "missing_table")


def test_execute_query_returns_structured_result(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE item (name TEXT, price REAL)")
        connection.executemany(
            "INSERT INTO item (name, price) VALUES (?, ?)",
            (("Notebook", 3.5), ("Pen", 1.25)),
        )

    with database.open_read_only_connection(database_path) as connection:
        result = database.execute_query(
            connection,
            "SELECT name, price FROM item ORDER BY price DESC",
        )

    assert result == {
        "success": True,
        "columns": ("name", "price"),
        "rows": (("Notebook", 3.5), ("Pen", 1.25)),
        "row_count": 2,
        "truncated": False,
        "error": None,
    }


def test_execute_query_limits_rows_and_reports_truncation(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO item (id) VALUES (?)",
            ((1,), (2,), (3,)),
        )

    with database.open_read_only_connection(database_path) as connection:
        result = database.execute_query(
            connection,
            "SELECT id FROM item ORDER BY id",
            max_rows=2,
        )

    assert result == {
        "success": True,
        "columns": ("id",),
        "rows": ((1,), (2,)),
        "row_count": 2,
        "truncated": True,
        "error": None,
    }


def test_execute_query_returns_structured_sqlite_error(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY)")

    with database.open_read_only_connection(database_path) as connection:
        result = database.execute_query(
            connection,
            "SELECT missing_column FROM item",
        )

    assert result == {
        "success": False,
        "columns": (),
        "rows": (),
        "row_count": 0,
        "truncated": False,
        "error": "no such column: missing_column",
    }


def test_execute_query_rejects_non_positive_row_limit(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY)")

    with database.open_read_only_connection(database_path) as connection:
        with pytest.raises(ValueError, match="max_rows must be positive"):
            database.execute_query(connection, "SELECT id FROM item", max_rows=0)


def test_execute_query_rejects_write_before_database_execution(tmp_path):
    database_path = tmp_path / "sample.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("CREATE TABLE item (name TEXT NOT NULL)")
        connection.execute("INSERT INTO item (name) VALUES ('Original')")

        result = database.execute_query(
            connection,
            "UPDATE item SET name = 'Changed'",
        )
        stored_name = connection.execute("SELECT name FROM item").fetchone()[0]
    finally:
        connection.close()

    assert result == {
        "success": False,
        "columns": (),
        "rows": (),
        "row_count": 0,
        "truncated": False,
        "error": "Query rejected: Only read-only SELECT queries are allowed.",
    }
    assert stored_name == "Original"
