"""Download the Chinook SQLite database used by the project."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from text_to_sql_agent.dataset import download_chinook_database


if __name__ == "__main__":
    target = PROJECT_ROOT / "data" / "raw" / "Chinook_Sqlite.sqlite"
    database_path = download_chinook_database(target)
    print(f"Chinook database ready at: {database_path}")
