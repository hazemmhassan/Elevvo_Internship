"""Reproducible Chinook database download and validation."""

from contextlib import closing
from pathlib import Path
import sqlite3
from urllib.request import urlopen


CHINOOK_DATABASE_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/master/"
    "ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
)
REQUIRED_TABLES = frozenset({"Customer", "Invoice", "InvoiceLine"})


def validate_chinook_database(database_path: Path) -> None:
    """Raise when a SQLite file does not contain core Chinook tables."""
    try:
        with closing(sqlite3.connect(database_path)) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
            tables = {row[0] for row in rows}
    except sqlite3.Error as error:
        raise RuntimeError(f"Downloaded file is not a valid SQLite database: {error}") from error

    missing_tables = REQUIRED_TABLES - tables
    if missing_tables:
        missing = ", ".join(sorted(missing_tables))
        raise RuntimeError(f"Chinook database is missing required tables: {missing}")


def download_chinook_database(destination: Path) -> Path:
    """Download, validate, and atomically install the Chinook SQLite file."""
    if destination.exists():
        validate_chinook_database(destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".download")
    try:
        with urlopen(CHINOOK_DATABASE_URL, timeout=60) as response:
            temporary_path.write_bytes(response.read())
        validate_chinook_database(temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    return destination
