-- 01_bronze.sql: Ingest raw CFPB complaint CSV into Bronze DuckDB Parquet layer
-- Raw complaint table schema preserves original types and columns.

CREATE OR REPLACE TABLE bronze_complaints AS
SELECT
    "Complaint ID" AS complaint_id,
    COALESCE(
        TRY_STRPTIME("Date received", '%m/%d/%Y'),
        TRY_STRPTIME("Date received", '%m-%d-%Y'),
        TRY_STRPTIME("Date received", '%Y-%m-%d'),
        TRY_CAST("Date received" AS DATE)
    ) AS date_received,
    "Product" AS raw_product,
    "Sub-product" AS sub_product,
    "Issue" AS raw_issue,
    "Sub-issue" AS sub_issue,
    "Consumer complaint narrative" AS raw_narrative,
    "Company" AS company,
    "State" AS state,
    "ZIP code" AS zip_code,
    "Submitted via" AS submitted_via,
    "Company response to consumer" AS company_response,
    "Timely response?" AS timely_response,
    "Consumer disputed?" AS consumer_disputed
FROM read_csv_auto('data/raw/complaints.csv');
