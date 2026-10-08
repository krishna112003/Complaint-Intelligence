-- 04_trends_window.sql: Time-series analytics with SQL window functions
-- Computes monthly volume trends, Month-over-Month (MoM) growth,
-- 3-month rolling averages, and flags emerging spikes.

WITH monthly_counts AS (
    SELECT
        DATE_TRUNC('month', date_received) AS report_month,
        raw_product,
        COUNT(*) AS monthly_volume
    FROM gold_complaints
    GROUP BY 1, 2
),
total_monthly AS (
    SELECT
        report_month,
        SUM(monthly_volume) AS total_market_volume
    FROM monthly_counts
    GROUP BY 1
),
with_shares AS (
    SELECT
        m.report_month,
        m.raw_product,
        m.monthly_volume,
        ROUND(m.monthly_volume * 100.0 / t.total_market_volume, 2) AS share_pct,
        -- Previous month volume
        LAG(m.monthly_volume, 1) OVER (
            PARTITION BY m.raw_product ORDER BY m.report_month
        ) AS prev_month_volume,
        -- 3-month rolling average volume
        AVG(m.monthly_volume) OVER (
            PARTITION BY m.raw_product
            ORDER BY m.report_month
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_3m_avg_volume
    FROM monthly_counts m
    JOIN total_monthly t ON m.report_month = t.report_month
)
SELECT
    report_month,
    raw_product,
    monthly_volume,
    share_pct,
    prev_month_volume,
    ROUND(rolling_3m_avg_volume, 1) AS rolling_3m_avg,
    ROUND(
        CASE
            WHEN prev_month_volume IS NOT NULL AND prev_month_volume > 0
            THEN ((monthly_volume - prev_month_volume) * 100.0 / prev_month_volume)
            ELSE 0.0
        END, 2
    ) AS mom_growth_pct,
    DENSE_RANK() OVER (
        PARTITION BY report_month ORDER BY monthly_volume DESC
    ) AS monthly_rank_by_volume
FROM with_shares
ORDER BY report_month DESC, monthly_volume DESC;
