"""SQL safety validation for agent-generated statements."""

from sqlglot import exp, parse
from sqlglot.errors import ParseError


def validate_read_only_sql(sql: str) -> dict[str, object]:
    """Validate that SQL contains one read-only query."""
    try:
        statements = tuple(
            statement
            for statement in parse(sql, read="sqlite")
            if statement is not None
        )
    except ParseError as error:
        description = error.errors[0].get("description", "Unable to parse SQL")
        return {
            "valid": False,
            "error": f"Invalid SQL: {description}",
        }

    if len(statements) != 1:
        return {
            "valid": False,
            "error": "Exactly one SQL statement is allowed.",
        }

    statement = statements[0]
    if not isinstance(statement, exp.Query):
        return {
            "valid": False,
            "error": "Only read-only SELECT queries are allowed.",
        }

    return {"valid": True, "error": None}
