"""SQLite database access for the Text-to-SQL agent."""

from pathlib import Path
import sqlite3
from sqlite3 import Connection

from text_to_sql_agent.sql_validation import validate_read_only_sql


def open_read_only_connection(database_path: Path) -> Connection:
    """Open a SQLite database without write permissions."""
    database_uri = database_path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(database_uri, uri=True)


def list_tables(connection: Connection) -> tuple[str, ...]:
    """Return the database's user-defined table names in stable order."""
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    return tuple(row[0] for row in rows)


def get_table_schema(connection: Connection, table_name: str) -> dict[str, object]:
    """Return structured column and foreign-key metadata for a table."""
    if table_name not in list_tables(connection):
        raise ValueError(f"Unknown table: {table_name}")

    column_rows = connection.execute(
        """
        SELECT name, type, "notnull", pk
        FROM pragma_table_info(?)
        ORDER BY cid
        """,
        (table_name,),
    )
    columns = tuple(
        {
            "name": name,
            "data_type": data_type,
            "not_null": bool(not_null),
            "primary_key": bool(primary_key),
        }
        for name, data_type, not_null, primary_key in column_rows
    )

    foreign_key_rows = connection.execute(
        """
        SELECT "from", "table", "to"
        FROM pragma_foreign_key_list(?)
        ORDER BY id, seq
        """,
        (table_name,),
    )
    foreign_keys = tuple(
        {
            "column": column,
            "referenced_table": referenced_table,
            "referenced_column": referenced_column,
        }
        for column, referenced_table, referenced_column in foreign_key_rows
    )

    return {
        "table": table_name,
        "columns": columns,
        "foreign_keys": foreign_keys,
    }


def execute_query(
    connection: Connection,
    sql: str,
    max_rows: int = 100,
) -> dict[str, object]:
    """Execute SQL and return a structured result for the agent."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")

    validation = validate_read_only_sql(sql)
    if not validation["valid"]:
        return {
            "success": False,
            "columns": (),
            "rows": (),
            "row_count": 0,
            "truncated": False,
            "error": f"Query rejected: {validation['error']}",
        }

    try:
        cursor = connection.execute(sql)
        fetched_rows = cursor.fetchmany(max_rows + 1)
    except sqlite3.Error as error:
        return {
            "success": False,
            "columns": (),
            "rows": (),
            "row_count": 0,
            "truncated": False,
            "error": str(error),
        }

    truncated = len(fetched_rows) > max_rows
    rows = tuple(fetched_rows[:max_rows])
    columns = tuple(description[0] for description in cursor.description)
    return {
        "success": True,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "error": None,
    }
