"""Data cleaning: Bronze → Silver.

Drops empty narratives, trims whitespace, deduplicates, normalises XXXX mask
tokens, length filters, and maps raw products to consolidated labels using
configs/data.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from complaints.config import get, get_project_root, setup_logging

logger = logging.getLogger(__name__)


def clean(bronze_file: Path | None = None, silver_dir: Path | None = None) -> Path:
    """Clean bronze complaints and output silver parquet."""
    root = get_project_root()
    bronze_path = bronze_file or (
        root / get("data", "bronze_dir", "data/bronze") / "bronze_complaints.parquet"
    )
    out_dir = silver_dir or (root / get("data", "silver_dir", "data/silver"))
    out_dir.mkdir(parents=True, exist_ok=True)
    silver_parquet = out_dir / "silver_complaints.parquet"

    if not bronze_path.exists():
        raise FileNotFoundError(f"Bronze complaints not found at {bronze_path}. Run ingest first.")

    logger.info("Cleaning bronze data from %s...", bronze_path)
    con = duckdb.connect()

    # Load bronze parquet as bronze_complaints table
    con.execute(f"CREATE TABLE bronze_complaints AS SELECT * FROM read_parquet('{bronze_path}')")

    # Execute cleaning SQL
    sql_file = root / "sql" / "02_silver_clean.sql"
    with open(sql_file) as f:
        query = f.read()
    con.execute(query)

    df: pd.DataFrame = con.execute("SELECT * FROM silver_complaints").df()
    initial_count = len(df)
    logger.info("Retrieved %d cleaned records from SQL", initial_count)

    # Apply consolidated product mapping
    mapping = get("data", "product_mapping", {})
    df["product"] = df["raw_product"].map(mapping).fillna("Other")

    # Save to Silver Parquet
    df.to_parquet(silver_parquet, index=False, engine="pyarrow")
    logger.info("Silver dataset saved to %s (%d rows)", silver_parquet, len(df))

    # Log class distribution
    dist = df["product"].value_counts()
    logger.info("Consolidated product distribution:\n%s", dist.to_string())

    return silver_parquet


if __name__ == "__main__":
    setup_logging()
    clean()
