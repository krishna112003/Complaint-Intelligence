"""Stratified sampling with per-class cap.

Draws ~60,000 rows with stratification and a per-class cap so rare products
still appear. Records the original class distribution and generates features.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from complaints.config import get, get_project_root, get_seed, setup_logging

logger = logging.getLogger(__name__)


def sample(
    silver_file: Path | None = None,
    gold_dir: Path | None = None,
    target_sample_size: int | None = None,
    per_class_cap: int | None = None,
) -> Path:
    """Sample records from silver complaints with per-class cap and seed."""
    root = get_project_root()
    silver_path = silver_file or (
        root / get("data", "silver_dir", "data/silver") / "silver_complaints.parquet"
    )
    out_dir = gold_dir or (root / get("data", "gold_dir", "data/gold"))
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_sample_file = out_dir / "gold_sample.parquet"
    dist_file = out_dir / "class_distribution.json"

    seed = get_seed()
    sample_size = target_sample_size or get("data", "sample_size", 60000)
    class_cap = per_class_cap or get("data", "per_class_cap", 10000)

    logger.info("Connecting to DuckDB to engineer Gold features from %s...", silver_path)
    con = duckdb.connect()
    con.execute(f"CREATE TABLE silver_complaints AS SELECT * FROM read_parquet('{silver_path}')")

    sql_file = root / "sql" / "03_gold_features.sql"
    with open(sql_file) as f:
        con.execute(f.read())

    df: pd.DataFrame = con.execute("SELECT * FROM gold_complaints").df()
    logger.info("Total available gold features rows: %d", len(df))

    original_dist = df["product"].value_counts().to_dict()

    # Stratified sampling with per-class cap
    # Group by product, cap large classes at class_cap, and sample proportionally
    rng = np.random.RandomState(seed)
    sampled_dfs = []

    # Calculate sampling budget per class
    total_rows = len(df)
    for product_name, group in df.groupby("product"):
        n_available = len(group)
        # Target proportional sample
        prop_target = int(round(sample_size * (n_available / total_rows)))
        # Cap at per_class_cap, min 1
        n_sample = min(n_available, max(1, min(prop_target, class_cap)))

        sampled_group = group.sample(n=n_sample, random_state=rng.randint(0, 1000000))
        sampled_dfs.append(sampled_group)

    sampled_df = pd.concat(sampled_dfs, ignore_index=True)

    # If slightly under or over sample_size, adjust safely
    if len(sampled_df) > sample_size:
        sampled_df = sampled_df.sample(n=sample_size, random_state=seed)

    sampled_dist = sampled_df["product"].value_counts().to_dict()

    # Save to gold_sample.parquet
    sampled_df.to_parquet(gold_sample_file, index=False, engine="pyarrow")
    logger.info("Sampled %d rows saved to %s", len(sampled_df), gold_sample_file)

    # Save distribution audit
    dist_payload = {
        "original_distribution": original_dist,
        "sampled_distribution": sampled_dist,
        "sample_size": len(sampled_df),
        "seed": seed,
    }
    with open(dist_file, "w") as f:
        json.dump(dist_payload, f, indent=2)
    logger.info("Class distribution audit written to %s", dist_file)

    return gold_sample_file


if __name__ == "__main__":
    setup_logging()
    sample()
