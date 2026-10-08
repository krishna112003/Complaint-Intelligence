-- 02_silver_clean.sql: Clean and standardize raw complaints into Silver table
-- 1. Drops records with NULL or empty narratives (< 50 chars)
-- 2. Trims and normalizes whitespace
-- 3. Deduplicates by narrative text and date received

CREATE OR REPLACE TABLE silver_complaints AS
WITH deduplicated AS (
    SELECT
        complaint_id,
        date_received,
        TRIM(regexp_replace(raw_narrative, '\s+', ' ', 'g')) AS narrative_clean,
        raw_product,
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
        ROW_NUMBER() OVER (
            PARTITION BY TRIM(regexp_replace(raw_narrative, '\s+', ' ', 'g')), date_received
            ORDER BY complaint_id
        ) AS dedup_rank
    FROM bronze_complaints
    WHERE raw_narrative IS NOT NULL
      AND LENGTH(TRIM(raw_narrative)) >= 50
)
SELECT
    complaint_id,
    date_received,
    narrative_clean AS narrative,
    raw_product,
    sub_product,
    raw_issue,
    sub_issue,
    company,
    state,
    zip_code,
    submitted_via,
    company_response,
    timely_response,
    consumer_disputed
FROM deduplicated
WHERE dedup_rank = 1;
