-- 03_gold_features.sql: Feature engineering for model-ready Gold complaints
-- Computes narrative length statistics and mask token frequencies.

CREATE OR REPLACE TABLE gold_complaints AS
SELECT
    complaint_id,
    date_received,
    narrative,
    raw_product,
    product,
    sub_product,
    raw_issue,
    sub_issue,
    company,
    state,
    zip_code,
    submitted_via,
    company_response,
    timely_response,
    consumer_disputed,
    LENGTH(narrative) AS char_length,
    LENGTH(narrative) - LENGTH(REPLACE(narrative, ' ', '')) + 1 AS approx_word_count,
    ROUND(
        (LENGTH(narrative) - LENGTH(REPLACE(narrative, 'XXXX', ''))) / 4.0, 1
    ) AS mask_token_count
FROM silver_complaints;
