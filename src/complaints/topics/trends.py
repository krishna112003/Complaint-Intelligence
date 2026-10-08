"""Topic trend queries and emerging-theme detection via DuckDB window functions.

Phase 8 implementation:
- Monthly topic share aggregation in DuckDB.
- Window functions for MoM share change, 3-month moving average, and growth rank.
- Emerging topic detection (streak of >= 3 consecutive months of positive MoM growth).
- Visual trend plot saved to reports/figures/topics_over_time.png.
- Emerging topics report saved to reports/emerging_topics.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from complaints.config import get_project_root
from complaints.topics.bertopic_run import run_bertopic

logger = logging.getLogger(__name__)


def compute_topic_trends(doc_topics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly topic shares, MoM changes, 3m moving averages, and ranks via DuckDB."""
    # Ensure date_received is a date or timestamp string
    df_copy = doc_topics_df.copy()
    df_copy["date_received"] = pd.to_datetime(df_copy["date_received"])

    con = duckdb.connect(database=":memory:")
    con.register("complaints_topics", df_copy)

    query = """
    WITH monthly_counts AS (
        SELECT
            strftime(date_trunc('month', date_received), '%Y-%m') AS month,
            topic_id,
            topic_label,
            count(*) AS count
        FROM complaints_topics
        WHERE topic_id >= 0
        GROUP BY 1, 2, 3
    ),
    monthly_totals AS (
        SELECT
            month,
            topic_id,
            topic_label,
            count,
            sum(count) OVER (PARTITION BY month) AS month_total,
            count * 1.0 / sum(count) OVER (PARTITION BY month) AS topic_share
        FROM monthly_counts
    ),
    with_trends AS (
        SELECT
            month,
            topic_id,
            topic_label,
            count,
            month_total,
            topic_share,
            topic_share - lag(topic_share, 1) OVER (PARTITION BY topic_id ORDER BY month) AS mom_share_change,
            avg(topic_share) OVER (
                PARTITION BY topic_id
                ORDER BY month
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ) AS moving_avg_3m,
            dense_rank() OVER (PARTITION BY month ORDER BY topic_share DESC) AS monthly_rank
        FROM monthly_totals
    )
    SELECT *
    FROM with_trends
    ORDER BY topic_id, month
    """
    trends_df: pd.DataFrame = con.execute(query).df()
    con.close()
    return trends_df


def detect_emerging_topics(
    trends_df: pd.DataFrame,
    min_consecutive_months: int = 3,
) -> pd.DataFrame:
    """Detect emerging topics exhibiting positive MoM share growth for >= min_consecutive_months."""
    emerging_records: list[dict[str, Any]] = []

    for topic_id, group in trends_df.groupby("topic_id"):
        group = group.sort_values("month").reset_index(drop=True)
        topic_label = group["topic_label"].iloc[0]

        streak = 0
        streak_start_idx = 0

        for i, row in group.iterrows():
            change = row["mom_share_change"]
            if pd.notna(change) and change > 0:
                if streak == 0:
                    streak_start_idx = i - 1  # includes month where increase began
                streak += 1
            else:
                if streak >= min_consecutive_months:
                    start_month = group.loc[streak_start_idx, "month"]
                    end_month = group.loc[i - 1, "month"]
                    init_share = group.loc[streak_start_idx, "topic_share"]
                    final_share = group.loc[i - 1, "topic_share"]
                    growth_pct = (
                        (final_share - init_share) / init_share * 100.0 if init_share > 0 else 0.0
                    )
                    emerging_records.append(
                        {
                            "topic_id": topic_id,
                            "topic_label": topic_label,
                            "streak_months": streak,
                            "start_month": start_month,
                            "end_month": end_month,
                            "initial_share": init_share,
                            "final_share": final_share,
                            "growth_pct": growth_pct,
                        }
                    )
                streak = 0

        # Check tail of group
        if streak >= min_consecutive_months:
            start_month = group.loc[streak_start_idx, "month"]
            end_month = group.iloc[-1]["month"]
            init_share = group.loc[streak_start_idx, "topic_share"]
            final_share = group.iloc[-1]["topic_share"]
            growth_pct = (final_share - init_share) / init_share * 100.0 if init_share > 0 else 0.0
            emerging_records.append(
                {
                    "topic_id": topic_id,
                    "topic_label": topic_label,
                    "streak_months": streak,
                    "start_month": start_month,
                    "end_month": end_month,
                    "initial_share": init_share,
                    "final_share": final_share,
                    "growth_pct": growth_pct,
                }
            )

    if not emerging_records:
        return pd.DataFrame(
            columns=[
                "topic_id",
                "topic_label",
                "streak_months",
                "start_month",
                "end_month",
                "initial_share",
                "final_share",
                "growth_pct",
            ]
        )

    res_df = pd.DataFrame(emerging_records)
    return res_df.sort_values("growth_pct", ascending=False).reset_index(drop=True)


def plot_topics_over_time(
    trends_df: pd.DataFrame,
    output_path: Path | None = None,
    top_k: int = 6,
) -> Path:
    """Generate time series plot for top topics."""
    root = get_project_root()
    if output_path is None:
        figures_dir = root / "reports/figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        output_path = figures_dir / "topics_over_time.png"

    # Identify top K topics by total count
    top_topics = (
        trends_df.groupby("topic_id")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_k)
        .index.tolist()
    )

    plt.figure(figsize=(12, 6))
    for tid in top_topics:
        sub = trends_df[trends_df["topic_id"] == tid].sort_values("month")
        label = sub["topic_label"].iloc[0] if not sub.empty else f"Topic {tid}"
        plt.plot(
            sub["month"],
            sub["moving_avg_3m"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=label,
        )

    plt.title("CFPB Complaint Topic Shares Over Time (3-Month Moving Average)", fontsize=13)
    plt.xlabel("Month", fontsize=11)
    plt.ylabel("Topic Share of Monthly Complaints", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info("Saved topic trend figure to %s", output_path)
    return output_path


def generate_emerging_topics_report(
    trends_df: pd.DataFrame,
    emerging_df: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Generate comprehensive markdown report of topic trends and emerging themes."""
    root = get_project_root()
    if output_path is None:
        report_dir = root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        output_path = report_dir / "emerging_topics.md"

    lines = [
        "# CFPB Consumer Complaint Emerging Topics & Trends Report\n",
        "## Overview\n",
        f"- **Time Range:** {trends_df['month'].min()} to {trends_df['month'].max()}\n",
        f"- **Total Modeled Topics:** {trends_df['topic_id'].nunique()}\n",
        "- **Emerging Topic Criterion:** >= 3 consecutive months of positive MoM share growth\n\n",
        "## Emerging Topics (Sustained Growth)\n\n",
    ]

    if emerging_df.empty:
        lines.append(
            "_No topics met the strict criteria of 3+ consecutive months of continuous growth._\n\n"
        )
    else:
        lines.append(
            "| Topic ID | Topic Label | Streak (Months) | Period | Initial Share | Final Share | Growth % |\n"
        )
        lines.append("|---|---|---|---|---|---|---|\n")
        for _, r in emerging_df.iterrows():
            lines.append(
                f"| {r['topic_id']} | {r['topic_label']} | {r['streak_months']} | "
                f"{r['start_month']} to {r['end_month']} | {r['initial_share']:.3%} | "
                f"{r['final_share']:.3%} | +{r['growth_pct']:.1f}% |\n"
            )
        lines.append("\n")

    lines.append("## Top Topics by Overall Volume\n\n")
    top_volume = (
        trends_df.groupby(["topic_id", "topic_label"])["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
        .head(10)
    )
    lines.append("| Topic ID | Topic Label | Total Complaints |\n")
    lines.append("|---|---|---|\n")
    for _, r in top_volume.iterrows():
        lines.append(f"| {r['topic_id']} | {r['topic_label']} | {int(r['count']):,} |\n")

    lines.append("\n## Visual Trends\n\n")
    lines.append("See `reports/figures/topics_over_time.png` for monthly trend trajectories.\n")

    with open(output_path, "w") as f:
        f.writelines(lines)

    logger.info("Saved emerging topics report to %s", output_path)
    return output_path


def compute_trends() -> None:
    """End-to-end pipeline execution for topic trend analysis and reporting."""
    _, doc_df, _ = run_bertopic()
    trends_df = compute_topic_trends(doc_df)
    emerging_df = detect_emerging_topics(trends_df)
    plot_topics_over_time(trends_df)
    generate_emerging_topics_report(trends_df, emerging_df)
    logger.info("Phase 8 topic trends analysis complete.")


if __name__ == "__main__":
    from complaints.config import setup_logging

    setup_logging()
    compute_trends()
