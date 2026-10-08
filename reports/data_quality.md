# Data Quality & Pipeline Audit Report

**Date Generated:** 2026-10-08 23:43:54 UTC

## 1. Pipeline Retention & Row Counts

| Stage | File | Row Count | % of Bronze | Description |
|---|---|---|---|---|
| **Bronze** | `bronze_complaints.parquet` | 208,398 | 100.0% | Raw CFPB complaint records ingested into DuckDB Parquet. |
| **Silver** | `silver_complaints.parquet` | 201,490 | 96.7% | Deduplicated, whitespace cleaned, minimum length >= 50 chars. |
| **Gold Sample** | `gold_sample.parquet` | 49,787 | 23.9% | Stratified sample with per-class cap (seed = 42). |
| **Train** | `train.parquet` | 34,850 | 16.7% | Oldest 70% temporal split. |
| **Validation** | `val.parquet` | 7,468 | 3.6% | Middle 15% temporal split. |
| **Test** | `test.parquet` | 7,469 | 3.6% | Newest 15% temporal split (held out). |

## 2. Temporal Split Boundaries & No-Leakage Verification

| Split | Row Count | Earliest Date | Latest Date |
|---|---|---|---|
| **Train** | 34,850 | 2015-03-19 | 2017-07-05 |
| **Validation** | 7,468 | 2017-07-05 | 2017-11-10 |
| **Test** | 7,469 | 2017-11-10 | 2018-04-25 |

> **Leakage Check:** Train Max Date <= Val Min Date and Val Max Date <= Test Min Date. Complaint IDs across splits are strictly disjoint.

## 3. Narrative Length & Mask Token Statistics (Gold Sample)

| Metric | Mean | Median (p50) | p90 | p99 | Min | Max |
|---|---|---|---|---|---|---|
| **Character Length** | 1139.7 | 803.0 | 2482.0 | 4555.0 | 47 | 31398 |
| **Approx Word Count** | 208.8 | 148.0 | 455.0 | 830.0 | 6 | 6314 |
| **Mask Tokens (`XXXX`)** | 10.7 | 5.0 | 25.0 | 79.0 | 0.0 | 1473.0 |

## 4. Class Distribution: Silver vs. Gold Sample

| Product Class | Silver Count | Silver Share (%) | Gold Sample Count | Gold Sample Share (%) |
|---|---|---|---|---|
| Bank Account | 15,909 | 7.90% | 4,737 | 9.51% |
| Credit Card | 23,137 | 11.48% | 6,890 | 13.84% |
| Credit Reporting | 55,032 | 27.31% | 10,000 | 20.09% |
| Debt Collection | 46,430 | 23.04% | 10,000 | 20.09% |
| Money Transfer | 3,449 | 1.71% | 1,027 | 2.06% |
| Mortgage | 32,765 | 16.26% | 9,757 | 19.60% |
| Other | 221 | 0.11% | 66 | 0.13% |
| Personal Loan | 10,027 | 4.98% | 2,986 | 6.00% |
| Student Loan | 12,401 | 6.15% | 3,693 | 7.42% |
| Vehicle Loan/Lease | 2,119 | 1.05% | 631 | 1.27% |

## 5. Consolidated Product Category Mapping

| Raw CFPB Product Label | Consolidated Class |
|---|---|
| Credit reporting, credit repair services, or other personal consumer reports | **Credit Reporting** |
| Credit reporting | **Credit Reporting** |
| Debt collection | **Debt Collection** |
| Mortgage | **Mortgage** |
| Credit card or prepaid card | **Credit Card** |
| Credit card | **Credit Card** |
| Prepaid card | **Credit Card** |
| Checking or savings account | **Bank Account** |
| Bank account or service | **Bank Account** |
| Student loan | **Student Loan** |
| Vehicle loan or lease | **Vehicle Loan/Lease** |
| Money transfer, virtual currency, or money service | **Money Transfer** |
| Money transfers | **Money Transfer** |
| Payday loan, title loan, or personal loan | **Personal Loan** |
| Payday loan | **Personal Loan** |
| Consumer Loan | **Personal Loan** |
| Other financial service | **Other** |
| Virtual currency | **Money Transfer** |
