"""Silver -> Gold using DuckDB.

DuckDB because at this data size it is the correct answer, not a compromise: it
reads the Silver JSONL/Parquet directly with no load step, runs the same ANSI SQL
we would run on Snowflake/BigQuery, and needs no cluster. The migration path is
deliberate — `gold.sql` contains no DuckDB-specific syntax beyond `read_json_auto`
(which is isolated in the staging views) and `list_contains`, so moving to a
warehouse means replacing five view definitions.

See docs/05-mlops-production.md for when to move off DuckDB and to what.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from ..config import Settings
from ..logging_conf import get_logger

log = get_logger(__name__)

SQL_PATH = Path(__file__).parent / "gold.sql"

_GLOB_PLACEHOLDERS = {
    "$bronze_calls_glob": "bronze/dt=*/calls.jsonl",
    "$silver_utterances_glob": "silver/dt=*/utterances.jsonl",
    "$silver_enrichment_glob": "silver/dt=*/enrichment.jsonl",
    "$silver_quality_glob": "silver/dt=*/call_quality.jsonl",
    "$silver_langprofile_glob": "silver/dt=*/language_profile.jsonl",
}


def _resolve_sql(settings: Settings) -> str:
    sql = SQL_PATH.read_text(encoding="utf-8")
    for placeholder, relative in _GLOB_PLACEHOLDERS.items():
        absolute = (settings.data_root / relative).as_posix()
        sql = sql.replace(placeholder, f"'{absolute}'")
    return sql


def build_gold(settings: Settings) -> dict[str, int]:
    """Rebuild every Gold table. Full-refresh at this scale; see the docs for the
    incremental (MERGE on date_key partition) design used in production."""
    settings.data_root.mkdir(parents=True, exist_ok=True)

    missing = [
        rel
        for rel in _GLOB_PLACEHOLDERS.values()
        if not list(settings.data_root.glob(rel))
    ]
    if missing:
        raise FileNotFoundError(
            "Silver layer incomplete; missing files matching: "
            + ", ".join(missing)
            + ". Run `ccma transcribe` and `ccma enrich` first."
        )

    con = duckdb.connect(str(settings.warehouse_path))
    try:
        con.execute("SET TimeZone='UTC'")
        con.execute(_resolve_sql(settings))

        counts: dict[str, int] = {}
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' ORDER BY table_name"
        ).fetchall()
        for (table,) in tables:
            counts[table] = con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]

        log.info("gold rebuilt", extra={"tables": len(counts), "warehouse": str(settings.warehouse_path)})
        return counts
    finally:
        con.close()


def query(settings: Settings, sql: str, params: list[Any] | None = None):
    """Read-only query helper used by the CLI and the dashboard."""
    con = duckdb.connect(str(settings.warehouse_path), read_only=True)
    try:
        return con.execute(sql, params or []).fetchdf()
    finally:
        con.close()


def table_exists(settings: Settings, name: str) -> bool:
    if not settings.warehouse_path.exists():
        return False
    con = duckdb.connect(str(settings.warehouse_path), read_only=True)
    try:
        found = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = ?",
            [name],
        ).fetchone()[0]
        return bool(found)
    finally:
        con.close()
