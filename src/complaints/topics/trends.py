"""Complaint volume trend queries via DuckDB window functions.

Computes monthly volume trends, Month-over-Month (MoM) growth,
3-month rolling averages, and volume ranks across consolidated products
using sql/04_trends_window.sql.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from complaints.config import get_project_root

logger = logging.getLogger(__name__)


def compute_volume_trends(gold_path: Path | None = None) -> pd.DataFrame:
    """Compute monthly complaint counts by product with MoM growth and 3m moving average.

    Executes sql/04_trends_window.sql using DuckDB over gold complaints data.
    """
    root = get_project_root()
    g_path = gold_path or (root / "data" / "gold" / "gold_sample.parquet")
    sql_path = root / "sql" / "04_trends_window.sql"

    if not g_path.exists():
        # Fall back to train + val + test
        train_p = root / "data" / "gold" / "train.parquet"
        if train_p.exists():
            g_path = train_p
        else:
            raise FileNotFoundError(f"Gold complaints dataset not found at {g_path}")

    sql_query = sql_path.read_text()

    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE VIEW gold_complaints AS SELECT * FROM read_parquet('{g_path}')")
    trends_df: pd.DataFrame = con.execute(sql_query).df()
    con.close()

    logger.info("Computed complaint volume trends: %d records across products.", len(trends_df))
    return trends_df


if __name__ == "__main__":
    from complaints.config import setup_logging

    setup_logging()
    df = compute_volume_trends()
    print(df.head(15))
