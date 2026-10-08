"""Unit tests for Phase 8: BERTopic and DuckDB topic trends."""

from __future__ import annotations

import pandas as pd
import pytest

from complaints.topics.bertopic_run import create_bertopic_model, generate_topic_labels
from complaints.topics.trends import (
    compute_topic_trends,
    detect_emerging_topics,
    generate_emerging_topics_report,
)


def test_bertopic_model_instantiation() -> None:
    """Test that create_bertopic_model creates a configured BERTopic object."""
    model = create_bertopic_model(seed=42, nr_topics=10, min_cluster_size=15)
    assert model is not None
    assert model.nr_topics == 10


def test_generate_topic_labels_mock() -> None:
    """Test topic label generator format."""

    class MockModel:
        def get_topic_info(self) -> pd.DataFrame:
            return pd.DataFrame({"Topic": [-1, 0, 1]})

        def get_topic(self, tid: int) -> list[tuple[str, float]]:
            if tid == 0:
                return [("credit", 0.5), ("score", 0.4), ("report", 0.3)]
            if tid == 1:
                return [("loan", 0.6), ("payment", 0.4)]
            return []

    labels = generate_topic_labels(MockModel())
    assert labels[-1] == "Outlier / Miscellaneous"
    assert "Topic 0: credit / score / report" in labels[0]
    assert "Topic 1: loan / payment" in labels[1]


def test_compute_topic_trends_duckdb_shares_and_window() -> None:
    """Test DuckDB topic trends computes shares, mom_change, and moving averages."""
    # Synthetic dataset with 2 months, 2 topics
    data = [
        {"complaint_id": 1, "date_received": "2020-01-05", "topic_id": 0, "topic_label": "T0"},
        {"complaint_id": 2, "date_received": "2020-01-10", "topic_id": 0, "topic_label": "T0"},
        {"complaint_id": 3, "date_received": "2020-01-15", "topic_id": 1, "topic_label": "T1"},
        {"complaint_id": 4, "date_received": "2020-02-05", "topic_id": 0, "topic_label": "T0"},
        {"complaint_id": 5, "date_received": "2020-02-10", "topic_id": 1, "topic_label": "T1"},
        {"complaint_id": 6, "date_received": "2020-02-15", "topic_id": 1, "topic_label": "T1"},
    ]
    df = pd.DataFrame(data)
    trends = compute_topic_trends(df)

    assert "month" in trends.columns
    assert "topic_share" in trends.columns
    assert "mom_share_change" in trends.columns
    assert "moving_avg_3m" in trends.columns
    assert "monthly_rank" in trends.columns

    # Check month 2020-01: T0 has 2/3 (0.667), T1 has 1/3 (0.333)
    jan = trends[trends["month"] == "2020-01"]
    assert len(jan) == 2
    assert pytest.approx(jan["topic_share"].sum(), rel=1e-3) == 1.0

    # Check month 2020-02: T0 has 1/3, T1 has 2/3
    feb_t1 = trends[(trends["month"] == "2020-02") & (trends["topic_id"] == 1)].iloc[0]
    # MoM change for T1: 2/3 - 1/3 = 1/3 > 0
    assert feb_t1["mom_share_change"] > 0


def test_detect_emerging_topics_detection() -> None:
    """Test detection of >= 3 consecutive months of positive MoM growth."""
    # Build synthetic trend data for topic 0 with 4 months: 3 consecutive increases
    trend_rows = [
        {
            "month": "2020-01",
            "topic_id": 0,
            "topic_label": "T0",
            "count": 10,
            "month_total": 100,
            "topic_share": 0.10,
            "mom_share_change": None,
            "moving_avg_3m": 0.10,
            "monthly_rank": 1,
        },
        {
            "month": "2020-02",
            "topic_id": 0,
            "topic_label": "T0",
            "count": 15,
            "month_total": 100,
            "topic_share": 0.15,
            "mom_share_change": 0.05,
            "moving_avg_3m": 0.125,
            "monthly_rank": 1,
        },
        {
            "month": "2020-03",
            "topic_id": 0,
            "topic_label": "T0",
            "count": 20,
            "month_total": 100,
            "topic_share": 0.20,
            "mom_share_change": 0.05,
            "moving_avg_3m": 0.15,
            "monthly_rank": 1,
        },
        {
            "month": "2020-04",
            "topic_id": 0,
            "topic_label": "T0",
            "count": 25,
            "month_total": 100,
            "topic_share": 0.25,
            "mom_share_change": 0.05,
            "moving_avg_3m": 0.20,
            "monthly_rank": 1,
        },
    ]
    trends_df = pd.DataFrame(trend_rows)
    emerging = detect_emerging_topics(trends_df, min_consecutive_months=3)

    assert len(emerging) == 1
    assert emerging.iloc[0]["topic_id"] == 0
    assert emerging.iloc[0]["streak_months"] == 3
    assert emerging.iloc[0]["growth_pct"] > 0


def test_generate_emerging_topics_report(tmp_path) -> None:
    """Test generating emerging topics markdown report."""
    trends_df = pd.DataFrame(
        [
            {
                "month": "2020-01",
                "topic_id": 0,
                "topic_label": "Credit Reporting",
                "count": 50,
                "month_total": 100,
                "topic_share": 0.50,
                "mom_share_change": None,
                "moving_avg_3m": 0.50,
                "monthly_rank": 1,
            }
        ]
    )
    emerging_df = pd.DataFrame()
    out = tmp_path / "emerging_topics.md"
    res = generate_emerging_topics_report(trends_df, emerging_df, output_path=out)
    assert res.exists()
    content = res.read_text()
    assert "CFPB Consumer Complaint Emerging Topics" in content
