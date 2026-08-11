import pytest

from text_to_sql_agent import sql_validation


def test_validate_read_only_sql_accepts_select():
    result = sql_validation.validate_read_only_sql(
        "SELECT CustomerId, FirstName FROM Customer"
    )

    assert result == {"valid": True, "error": None}


@pytest.mark.parametrize(
    "sql",
    (
        "INSERT INTO Customer (FirstName) VALUES ('Unsafe')",
        "UPDATE Customer SET FirstName = 'Unsafe'",
        "DELETE FROM Customer",
        "DROP TABLE Customer",
        "PRAGMA writable_schema = ON",
    ),
)
def test_validate_read_only_sql_rejects_non_query_statements(sql):
    result = sql_validation.validate_read_only_sql(sql)

    assert result == {
        "valid": False,
        "error": "Only read-only SELECT queries are allowed.",
    }


def test_validate_read_only_sql_rejects_multiple_statements():
    result = sql_validation.validate_read_only_sql(
        "SELECT CustomerId FROM Customer; DELETE FROM Customer"
    )

    assert result == {
        "valid": False,
        "error": "Exactly one SQL statement is allowed.",
    }


def test_validate_read_only_sql_returns_parse_error_for_malformed_sql():
    result = sql_validation.validate_read_only_sql("SELECT (")

    assert result["valid"] is False
    assert result["error"].startswith("Invalid SQL:")
