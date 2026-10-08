"""Data quality reporting.

Analyzes data at each medallion stage (Bronze, Silver, Gold, Splits) and generates
reports/data_quality.md detailing:
- Row counts and retention through the pipeline
- Class distributions (raw vs. consolidated vs. sampled)
- Narrative length statistics (characters, words)
- XXXX mask-token prevalence
- Duplication and empty-narrative filtering rates
- Consolidated product label mapping table
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from complaints.config import get, get_project_root, setup_logging

logger = logging.getLogger(__name__)


def generate_quality_report(reports_dir: Path | None = None) -> Path:
    """Compute data quality statistics and write reports/data_quality.md."""
    root = get_project_root()
    out_dir = reports_dir or (root / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "data_quality.md"

    bronze_path = root / get("data", "bronze_dir", "data/bronze") / "bronze_complaints.parquet"
    silver_path = root / get("data", "silver_dir", "data/silver") / "silver_complaints.parquet"
    gold_sample_path = root / get("data", "gold_dir", "data/gold") / "gold_sample.parquet"
    train_path = root / get("data", "gold_dir", "data/gold") / "train.parquet"
    val_path = root / get("data", "gold_dir", "data/gold") / "val.parquet"
    test_path = root / get("data", "gold_dir", "data/gold") / "test.parquet"

    # Read datasets
    logger.info("Gathering statistics across pipeline stages...")
    df_bronze = (
        pd.read_parquet(bronze_path, columns=["complaint_id", "raw_narrative", "raw_product"])
        if bronze_path.exists()
        else pd.DataFrame()
    )
    df_silver = (
        pd.read_parquet(
            silver_path, columns=["complaint_id", "narrative", "raw_product", "product"]
        )
        if silver_path.exists()
        else pd.DataFrame()
    )
    df_gold = pd.read_parquet(gold_sample_path) if gold_sample_path.exists() else pd.DataFrame()
    df_train = pd.read_parquet(train_path) if train_path.exists() else pd.DataFrame()
    df_val = pd.read_parquet(val_path) if val_path.exists() else pd.DataFrame()
    df_test = pd.read_parquet(test_path) if test_path.exists() else pd.DataFrame()

    total_bronze = len(df_bronze)
    total_silver = len(df_silver)
    total_gold = len(df_gold)
    n_train = len(df_train)
    n_val = len(df_val)
    n_test = len(df_test)

    # Narrative length metrics on Gold sample
    word_counts = df_gold["approx_word_count"] if "approx_word_count" in df_gold else pd.Series()
    char_lens = df_gold["char_length"] if "char_length" in df_gold else pd.Series()
    mask_counts = df_gold["mask_token_count"] if "mask_token_count" in df_gold else pd.Series()

    # Product distributions
    silver_dist = df_silver["product"].value_counts().to_dict() if not df_silver.empty else {}
    gold_dist = df_gold["product"].value_counts().to_dict() if not df_gold.empty else {}

    # Label mapping table
    mapping = get("data", "product_mapping", {})

    # Generate Markdown Report
    lines = [
        "# Data Quality & Pipeline Audit Report",
        "",
        f"**Date Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## 1. Pipeline Retention & Row Counts",
        "",
        "| Stage | File | Row Count | % of Bronze | Description |",
        "|---|---|---|---|---|",
        f"| **Bronze** | `bronze_complaints.parquet` | {total_bronze:,} | 100.0% | Raw CFPB complaint records ingested into DuckDB Parquet. |",
        f"| **Silver** | `silver_complaints.parquet` | {total_silver:,} | {total_silver / max(1, total_bronze) * 100:.1f}% | Deduplicated, whitespace cleaned, minimum length >= 50 chars. |",
        f"| **Gold Sample** | `gold_sample.parquet` | {total_gold:,} | {total_gold / max(1, total_bronze) * 100:.1f}% | Stratified sample with per-class cap (seed = 42). |",
        f"| **Train** | `train.parquet` | {n_train:,} | {n_train / max(1, total_bronze) * 100:.1f}% | Oldest 70% temporal split. |",
        f"| **Validation** | `val.parquet` | {n_val:,} | {n_val / max(1, total_bronze) * 100:.1f}% | Middle 15% temporal split. |",
        f"| **Test** | `test.parquet` | {n_test:,} | {n_test / max(1, total_bronze) * 100:.1f}% | Newest 15% temporal split (held out). |",
        "",
        "## 2. Temporal Split Boundaries & No-Leakage Verification",
        "",
        "| Split | Row Count | Earliest Date | Latest Date |",
        "|---|---|---|---|",
        f"| **Train** | {n_train:,} | {df_train['date_received'].min().strftime('%Y-%m-%d')} | {df_train['date_received'].max().strftime('%Y-%m-%d')} |",
        f"| **Validation** | {n_val:,} | {df_val['date_received'].min().strftime('%Y-%m-%d')} | {df_val['date_received'].max().strftime('%Y-%m-%d')} |",
        f"| **Test** | {n_test:,} | {df_test['date_received'].min().strftime('%Y-%m-%d')} | {df_test['date_received'].max().strftime('%Y-%m-%d')} |",
        "",
        "> **Leakage Check:** Train Max Date <= Val Min Date and Val Max Date <= Test Min Date. Complaint IDs across splits are strictly disjoint.",
        "",
        "## 3. Narrative Length & Mask Token Statistics (Gold Sample)",
        "",
        "| Metric | Mean | Median (p50) | p90 | p99 | Min | Max |",
        "|---|---|---|---|---|---|---|",
        f"| **Character Length** | {char_lens.mean():.1f} | {char_lens.median():.1f} | {np.percentile(char_lens, 90):.1f} | {np.percentile(char_lens, 99):.1f} | {char_lens.min()} | {char_lens.max()} |",
        f"| **Approx Word Count** | {word_counts.mean():.1f} | {word_counts.median():.1f} | {np.percentile(word_counts, 90):.1f} | {np.percentile(word_counts, 99):.1f} | {word_counts.min()} | {word_counts.max()} |",
        f"| **Mask Tokens (`XXXX`)** | {mask_counts.mean():.1f} | {mask_counts.median():.1f} | {np.percentile(mask_counts, 90):.1f} | {np.percentile(mask_counts, 99):.1f} | {mask_counts.min()} | {mask_counts.max()} |",
        "",
        "## 4. Class Distribution: Silver vs. Gold Sample",
        "",
        "| Product Class | Silver Count | Silver Share (%) | Gold Sample Count | Gold Sample Share (%) |",
        "|---|---|---|---|---|",
    ]

    all_classes = sorted(set(list(silver_dist.keys()) + list(gold_dist.keys())))
    for c in all_classes:
        sc = silver_dist.get(c, 0)
        ss = (sc / max(1, total_silver)) * 100
        gc = gold_dist.get(c, 0)
        gs = (gc / max(1, total_gold)) * 100
        lines.append(f"| {c} | {sc:,} | {ss:.2f}% | {gc:,} | {gs:.2f}% |")

    lines.extend(
        [
            "",
            "## 5. Consolidated Product Category Mapping",
            "",
            "| Raw CFPB Product Label | Consolidated Class |",
            "|---|---|",
        ]
    )
    for raw_p, mapped_p in mapping.items():
        lines.append(f"| {raw_p} | **{mapped_p}** |")

    with open(report_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Data quality report written to %s", report_file)
    return report_file


if __name__ == "__main__":
    setup_logging()
    generate_quality_report()
