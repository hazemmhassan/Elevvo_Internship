from contextlib import closing
import sqlite3

import pytest


def test_validate_chinook_database_accepts_required_schema(tmp_path):
    try:
        from text_to_sql_agent.dataset import validate_chinook_database
    except ModuleNotFoundError:
        pytest.fail("dataset setup has not been implemented", pytrace=False)

    database_path = tmp_path / "chinook.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE Customer (CustomerId INTEGER)")
        connection.execute("CREATE TABLE Invoice (InvoiceId INTEGER)")
        connection.execute("CREATE TABLE InvoiceLine (InvoiceLineId INTEGER)")

    validate_chinook_database(database_path)


def test_validate_chinook_database_rejects_incomplete_schema(tmp_path):
    try:
        from text_to_sql_agent.dataset import validate_chinook_database
    except ModuleNotFoundError:
        pytest.fail("dataset setup has not been implemented", pytrace=False)

    database_path = tmp_path / "not-chinook.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER)")

    with pytest.raises(RuntimeError, match="missing required tables"):
        validate_chinook_database(database_path)


def test_validate_chinook_database_releases_file(tmp_path):
    from text_to_sql_agent.dataset import validate_chinook_database

    database_path = tmp_path / "chinook.sqlite"
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("CREATE TABLE Customer (CustomerId INTEGER)")
        connection.execute("CREATE TABLE Invoice (InvoiceId INTEGER)")
        connection.execute("CREATE TABLE InvoiceLine (InvoiceLineId INTEGER)")
        connection.commit()

    validate_chinook_database(database_path)
    database_path.unlink()

    assert database_path.exists() is False
