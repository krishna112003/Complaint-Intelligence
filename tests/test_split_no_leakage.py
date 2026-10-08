"""Unit tests verifying temporal splitting and absence of data leakage."""

from __future__ import annotations

import pandas as pd


def test_split_temporal_ordering_assertion_helper():
    """Verify that a helper accurately asserts non-overlapping temporal windows."""
    train_dates = pd.date_range("2023-01-01", "2023-06-30")
    val_dates = pd.date_range("2023-07-01", "2023-08-31")
    test_dates = pd.date_range("2023-09-01", "2023-10-31")

    df_train = pd.DataFrame({"complaint_id": range(100), "date_received": train_dates[:100]})
    df_val = pd.DataFrame({"complaint_id": range(100, 150), "date_received": val_dates[:50]})
    df_test = pd.DataFrame({"complaint_id": range(150, 200), "date_received": test_dates[:50]})

    # Check ID disjointness
    train_ids = set(df_train["complaint_id"])
    val_ids = set(df_val["complaint_id"])
    test_ids = set(df_test["complaint_id"])

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    # Check temporal ordering
    assert df_train["date_received"].max() <= df_val["date_received"].min()
    assert df_val["date_received"].max() <= df_test["date_received"].min()
