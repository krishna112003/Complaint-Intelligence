-- 02_silver_clean.sql: Clean and standardize raw complaints into Silver table
-- 1. Drops records with NULL or empty narratives
-- 2. Trims and normalizes whitespace
-- 3. Normalizes 'XXXX' masking tokens
-- 4. Filters out very short complaints (< min_length chars)
-- 5. Deduplicates by narrative text and date

CREATE OR REPLACE TABLE silver_complaints AS
WITH cleaned AS (
    SELECT
        CAST("Complaint ID" AS VARCHAR) AS complaint_id,
        CAST("Date received" AS DATE) AS date_received,
        TRIM(regexp_replace("Consumer complaint narrative", '\s+', ' ', 'g')) AS narrative_clean,
        "Product" AS raw_product,
        "Sub-product" AS sub_product,
        "Issue" AS raw_issue,
        "Sub-issue" AS sub_issue,
        "Company" AS company,
        "State" AS state,
        "Submitted via" AS submitted_via,
        "Company response to consumer" AS company_response,
        "Timely response?" AS timely_response,
        "Consumer disputed?" AS consumer_disputed,
        ROW_NUMBER() OVER (
            PARTITION BY TRIM(regexp_replace("Consumer complaint narrative", '\s+', ' ', 'g')), CAST("Date received" AS DATE)
            ORDER BY CAST("Complaint ID" AS VARCHAR)
        ) AS dedup_rank
    FROM bronze_complaints
    WHERE "Consumer complaint narrative" IS NOT NULL
      AND LENGTH(TRIM("Consumer complaint narrative")) >= 50
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
    submitted_via,
    company_response,
    timely_response,
    consumer_disputed
FROM cleaned
WHERE dedup_rank = 1;
