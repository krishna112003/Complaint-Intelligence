"""Time-based train/val/test splitting and gold evaluation set generation.

Splits by Date received:
- Oldest 70% → train
- Middle 15% → validation
- Newest 15% → test

Generates:
- train.parquet, val.parquet, test.parquet
- gold_eval_set.parquet (1,000 rows from test)
- ci_eval_set.parquet (200 rows from gold_eval_set for frozen CI gate)
- gold_hand_labels.csv (200 row template with blank inspection columns)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from complaints.config import get, get_project_root, get_seed, setup_logging

logger = logging.getLogger(__name__)


def split(
    gold_sample_file: Path | None = None,
    out_dir: Path | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Execute temporal split and generate evaluation sets and templates."""
    root = get_project_root()
    sample_path = gold_sample_file or (
        root / get("data", "gold_dir", "data/gold") / "gold_sample.parquet"
    )
    gold_dir = out_dir or (root / get("data", "gold_dir", "data/gold"))
    gold_dir.mkdir(parents=True, exist_ok=True)

    seed = get_seed()
    gold_set_size = get("data", "gold_set_size", 1000)
    ci_eval_size = get("data", "ci_eval_set_size", 200)
    hand_check_size = get("data", "hand_check_size", 200)

    if not sample_path.exists():
        raise FileNotFoundError(f"Gold sample file not found at {sample_path}. Run sample first.")

    logger.info("Loading gold sample dataset from %s...", sample_path)
    df = pd.read_parquet(sample_path)
    df["date_received"] = pd.to_datetime(df["date_received"])

    # Strict sorting by Date received, secondary sort by complaint_id for determinism
    df = df.sort_values(by=["date_received", "complaint_id"]).reset_index(drop=True)
    n = len(df)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train : n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val :].copy()

    logger.info(
        "Temporal Split created:\n"
        "  Train: %d rows (%s to %s)\n"
        "  Val:   %d rows (%s to %s)\n"
        "  Test:  %d rows (%s to %s)",
        len(df_train),
        df_train["date_received"].min().strftime("%Y-%m-%d"),
        df_train["date_received"].max().strftime("%Y-%m-%d"),
        len(df_val),
        df_val["date_received"].min().strftime("%Y-%m-%d"),
        df_val["date_received"].max().strftime("%Y-%m-%d"),
        len(df_test),
        df_test["date_received"].min().strftime("%Y-%m-%d"),
        df_test["date_received"].max().strftime("%Y-%m-%d"),
    )

    # Save train / val / test splits
    train_path = gold_dir / "train.parquet"
    val_path = gold_dir / "val.parquet"
    test_path = gold_dir / "test.parquet"

    df_train.to_parquet(train_path, index=False, engine="pyarrow")
    df_val.to_parquet(val_path, index=False, engine="pyarrow")
    df_test.to_parquet(test_path, index=False, engine="pyarrow")

    # Sample Gold Evaluation Set strictly from Test Split
    actual_gold_size = min(len(df_test), gold_set_size)
    df_gold = df_test.sample(n=actual_gold_size, random_state=seed).sort_values("date_received")
    gold_eval_path = gold_dir / "gold_eval_set.parquet"
    df_gold.to_parquet(gold_eval_path, index=False, engine="pyarrow")
    logger.info("Gold evaluation set saved to %s (%d rows)", gold_eval_path, len(df_gold))

    # Sample Frozen CI Eval Set strictly from Gold Evaluation Set
    actual_ci_size = min(len(df_gold), ci_eval_size)
    df_ci = df_gold.sample(n=actual_ci_size, random_state=seed).sort_values("date_received")
    ci_eval_path = gold_dir / "ci_eval_set.parquet"
    df_ci.to_parquet(ci_eval_path, index=False, engine="pyarrow")
    logger.info("CI evaluation set saved to %s (%d rows)", ci_eval_path, len(df_ci))

    # Create gold_hand_labels.csv template for manual human review (columns blank, NOT pre-filled)
    actual_hand_size = min(len(df_ci), hand_check_size)
    df_hand = df_ci.sample(n=actual_hand_size, random_state=seed).copy()

    hand_template = pd.DataFrame(
        {
            "complaint_id": df_hand["complaint_id"],
            "date_received": df_hand["date_received"].dt.strftime("%Y-%m-%d"),
            "original_product_label": df_hand["product"],
            "narrative": df_hand["narrative"],
            # Hand-check review columns left blank for human review as required
            "hand_label_product": ["" for _ in range(len(df_hand))],
            "hand_issue_category": ["" for _ in range(len(df_hand))],
            "hand_root_cause": ["" for _ in range(len(df_hand))],
            "hand_urgency": ["" for _ in range(len(df_hand))],
            "hand_sentiment": ["" for _ in range(len(df_hand))],
            "hand_requested_resolution": ["" for _ in range(len(df_hand))],
            "hand_evidence_quote": ["" for _ in range(len(df_hand))],
            "reviewer_notes": ["" for _ in range(len(df_hand))],
        }
    )
    hand_labels_path = gold_dir / "gold_hand_labels.csv"
    hand_template.to_csv(hand_labels_path, index=False)
    logger.info(
        "Gold hand-labels blank template created at %s (%d rows)",
        hand_labels_path,
        len(hand_template),
    )

    return train_path, val_path, test_path, gold_eval_path, ci_eval_path, hand_labels_path


if __name__ == "__main__":
    setup_logging()
    split()
