-- 03_gold_features.sql: Produce feature-engineered model-ready dataset
-- Consolidates product categories and adds metadata features.

CREATE OR REPLACE TABLE gold_complaints AS
SELECT
    complaint_id,
    date_received,
    narrative,
    raw_product,
    sub_product,
    raw_issue,
    sub_issue,
    company,
    state,
    submitted_via,
    company_response,
    LENGTH(narrative) AS char_length,
    LENGTH(narrative) - LENGTH(REPLACE(narrative, ' ', '')) + 1 AS approx_word_count,
    ROUND(
        (LENGTH(narrative) - LENGTH(REPLACE(narrative, 'XXXX', ''))) / 4.0, 1
    ) AS mask_token_count
FROM silver_complaints;
