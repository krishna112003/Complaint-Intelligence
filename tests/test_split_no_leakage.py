"""Rigorous tests for time-based splitting and absence of data leakage.

Asserts:
1. No complaint ID appears in more than one split (Train, Val, Test).
2. Strict temporal ordering: Max(Train date) <= Min(Val date), and Max(Val date) <= Min(Test date).
3. Gold evaluation set IDs are strictly a subset of the Test split.
4. CI frozen evaluation set IDs are strictly a subset of the Gold evaluation set.
5. Hand-labels template exists with blank reviewer columns and no pre-filled values.
"""

from __future__ import annotations

import pandas as pd
import pytest

from complaints.config import get, get_project_root


@pytest.fixture
def split_data():
    """Load train, val, test, gold, and hand-labels data."""
    root = get_project_root()
    gold_dir = root / get("data", "gold_dir", "data/gold")

    train_path = gold_dir / "train.parquet"
    val_path = gold_dir / "val.parquet"
    test_path = gold_dir / "test.parquet"
    gold_eval_path = gold_dir / "gold_eval_set.parquet"
    ci_eval_path = gold_dir / "ci_eval_set.parquet"
    hand_csv_path = gold_dir / "gold_hand_labels.csv"

    assert train_path.exists(), "train.parquet missing. Run make data first."
    assert val_path.exists(), "val.parquet missing. Run make data first."
    assert test_path.exists(), "test.parquet missing. Run make data first."
    assert gold_eval_path.exists(), "gold_eval_set.parquet missing."
    assert ci_eval_path.exists(), "ci_eval_set.parquet missing."
    assert hand_csv_path.exists(), "gold_hand_labels.csv missing."

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)
    df_gold = pd.read_parquet(gold_eval_path)
    df_ci = pd.read_parquet(ci_eval_path)
    df_hand = pd.read_csv(hand_csv_path)

    return {
        "train": df_train,
        "val": df_val,
        "test": df_test,
        "gold": df_gold,
        "ci": df_ci,
        "hand": df_hand,
    }


def test_disjoint_complaint_ids(split_data):
    """Assert that no complaint ID appears in more than one split."""
    train_ids = set(split_data["train"]["complaint_id"])
    val_ids = set(split_data["val"]["complaint_id"])
    test_ids = set(split_data["test"]["complaint_id"])

    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    assert len(train_val_overlap) == 0, (
        f"Leakage detected between Train and Val: {len(train_val_overlap)} shared IDs"
    )
    assert len(train_test_overlap) == 0, (
        f"Leakage detected between Train and Test: {len(train_test_overlap)} shared IDs"
    )
    assert len(val_test_overlap) == 0, (
        f"Leakage detected between Val and Test: {len(val_test_overlap)} shared IDs"
    )


def test_strict_temporal_ordering(split_data):
    """Assert that train dates <= val dates <= test dates."""
    train_dates = pd.to_datetime(split_data["train"]["date_received"])
    val_dates = pd.to_datetime(split_data["val"]["date_received"])
    test_dates = pd.to_datetime(split_data["test"]["date_received"])

    max_train_date = train_dates.max()
    min_val_date = val_dates.min()
    max_val_date = val_dates.max()
    min_test_date = test_dates.min()

    assert max_train_date <= min_val_date, (
        f"Temporal leakage: max train date ({max_train_date}) is after min val date ({min_val_date})"
    )
    assert max_val_date <= min_test_date, (
        f"Temporal leakage: max val date ({max_val_date}) is after min test date ({min_test_date})"
    )


def test_gold_eval_set_subset_of_test(split_data):
    """Assert that gold evaluation set is sampled strictly from the held-out test split."""
    test_ids = set(split_data["test"]["complaint_id"])
    gold_ids = set(split_data["gold"]["complaint_id"])
    ci_ids = set(split_data["ci"]["complaint_id"])

    assert gold_ids.issubset(test_ids), (
        "Gold evaluation set contains IDs not present in test split!"
    )
    assert ci_ids.issubset(gold_ids), (
        "CI evaluation set contains IDs not present in gold evaluation set!"
    )
    assert len(gold_ids) == min(1000, len(test_ids))
    assert len(ci_ids) == min(200, len(gold_ids))


def test_hand_labels_template_structure_and_blank(split_data):
    """Assert that gold_hand_labels.csv template has required columns and is NOT pre-filled."""
    df_hand = split_data["hand"]
    assert len(df_hand) == 200

    required_cols = [
        "complaint_id",
        "date_received",
        "original_product_label",
        "narrative",
        "hand_label_product",
        "hand_issue_category",
        "hand_root_cause",
        "hand_urgency",
        "hand_sentiment",
        "hand_requested_resolution",
        "hand_evidence_quote",
        "reviewer_notes",
    ]
    for col in required_cols:
        assert col in df_hand.columns, f"Missing required template column: {col}"

    # Verify that hand-check columns are completely blank (never pre-filled by AI)
    blank_check_cols = [
        "hand_label_product",
        "hand_issue_category",
        "hand_root_cause",
        "hand_urgency",
        "hand_sentiment",
        "hand_requested_resolution",
        "hand_evidence_quote",
        "reviewer_notes",
    ]
    for col in blank_check_cols:
        non_empty = df_hand[col].dropna().astype(str).str.strip()
        non_empty = non_empty[non_empty != ""]
        assert len(non_empty) == 0, (
            f"Rule violation: '{col}' contains {len(non_empty)} pre-filled rows! Template must be blank."
        )
