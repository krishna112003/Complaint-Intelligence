"""Unit tests for classical and baseline models and schema adherence."""

from __future__ import annotations

import pandas as pd
import pytest

from complaints.models.base import PREDICTION_COLUMNS
from complaints.models.baselines import KeywordBaseline, MajorityBaseline


@pytest.fixture
def mock_dataset():
    """Create a minimal synthetic dataset matching the data contract."""
    df_train = pd.DataFrame(
        {
            "complaint_id": [1, 2, 3, 4],
            "narrative": [
                "There is an incorrect collection account on my credit report from Equifax.",
                "Harassing phone calls from debt collector demanding payment.",
                "The mortgage lender initiated an unjustified foreclosure on my home.",
                "Inaccurate late fee on my monthly credit card statement.",
            ],
            "product": [
                "Credit Reporting",
                "Debt Collection",
                "Mortgage",
                "Credit Card",
            ],
        }
    )
    df_val = pd.DataFrame(
        {
            "complaint_id": [5],
            "narrative": ["Unknown dispute regarding funds"],
            "product": ["Bank Account"],
        }
    )
    df_test = pd.DataFrame(
        {
            "complaint_id": [6, 7],
            "narrative": [
                "TransUnion failed to update my credit report score.",
                "Mortgage escrow balance calculation is completely wrong.",
            ],
            "product": ["Credit Reporting", "Mortgage"],
        }
    )
    return df_train, df_val, df_test


def test_majority_baseline_schema_and_prediction(mock_dataset):
    """Verify MajorityBaseline fits and outputs exact shared schema columns."""
    df_train, df_val, df_test = mock_dataset
    model = MajorityBaseline()
    model.fit(df_train, df_val)

    preds = model.predict(df_test, run_id="test_run_1")
    assert list(preds.columns) == PREDICTION_COLUMNS
    assert len(preds) == len(df_test)
    assert (preds["model_name"] == "majority_baseline").all()
    assert (preds["run_id"] == "test_run_1").all()
    assert (preds["confidence"] > 0.0).all()
    assert (preds["latency_ms"] >= 0.0).all()


def test_keyword_baseline_schema_and_rules(mock_dataset):
    """Verify KeywordBaseline matches keywords and outputs shared schema."""
    df_train, df_val, df_test = mock_dataset
    model = KeywordBaseline()
    model.fit(df_train, df_val)

    preds = model.predict(df_test, run_id="test_run_2")
    assert list(preds.columns) == PREDICTION_COLUMNS
    assert len(preds) == len(df_test)

    # First test row mentions 'credit report' -> Credit Reporting
    assert preds.iloc[0]["y_pred"] == "Credit Reporting"
    # Second test row mentions 'Mortgage' -> Mortgage
    assert preds.iloc[1]["y_pred"] == "Mortgage"
