"""Ingest raw CSV into DuckDB Bronze layer.

Reads the CFPB complaint CSV from data/raw/, inspects schema and product
categories, and writes a Bronze Parquet snapshot to data/bronze/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from complaints.config import get, get_project_root, setup_logging

logger = logging.getLogger(__name__)


def ingest(raw_csv_path: Path | None = None, bronze_dir: Path | None = None) -> Path:
    """Ingest raw CFPB complaint CSV into Bronze Parquet file."""
    root = get_project_root()
    raw_csv = raw_csv_path or (root / get("data", "raw_file", "data/raw/complaints.csv"))
    out_dir = bronze_dir or (root / get("data", "bronze_dir", "data/bronze"))
    out_dir.mkdir(parents=True, exist_ok=True)
    bronze_parquet = out_dir / "bronze_complaints.parquet"

    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Raw complaints CSV not found at '{raw_csv}'. "
            "Please place complaints.csv into data/raw/."
        )

    logger.info("Ingesting raw complaints from %s using DuckDB...", raw_csv)
    con = duckdb.connect()

    # Read sql/01_bronze.sql
    sql_file = root / "sql" / "01_bronze.sql"
    with open(sql_file) as f:
        query = f.read()

    con.execute(query)

    # Export to Parquet
    con.execute(
        f"COPY bronze_complaints TO '{bronze_parquet}' (FORMAT PARQUET, COMPRESSION 'ZSTD');"
    )

    count = con.execute("SELECT COUNT(*) FROM bronze_complaints").fetchone()[0]  # type: ignore[index]
    logger.info("Bronze ingestion complete: %d rows written to %s", count, bronze_parquet)

    # Print inspected columns and category distributions
    cols = con.execute(
        "SELECT column_name, column_type FROM (DESCRIBE bronze_complaints)"
    ).fetchall()
    logger.info("Bronze Schema (%d columns):", len(cols))
    for c, t in cols:
        logger.info("  %s: %s", c, t)

    return bronze_parquet


if __name__ == "__main__":
    setup_logging()
    ingest()
