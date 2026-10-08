-- 01_bronze.sql: Ingest raw CFPB complaint CSV into Bronze DuckDB layer
-- Raw complaint table schema preserves original types and columns.

CREATE OR REPLACE TABLE bronze_complaints AS
SELECT *
FROM read_csv_auto('data/raw/complaints.csv');
