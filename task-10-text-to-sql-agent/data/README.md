# Local data

Run the project setup command from the project root:

```powershell
python scripts/download_chinook.py
```

It downloads and validates the Chinook SQLite database at
`data/raw/Chinook_Sqlite.sqlite`.

The database is third-party source data and is intentionally excluded from Git.
The source is the MIT-licensed
[Chinook sample database](https://github.com/lerocha/chinook-database).
