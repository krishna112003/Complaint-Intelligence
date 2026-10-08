# Architecture & Data Flow

This document details the modular structure, data lineage, and evaluation lifecycle of the Complaint Intelligence system.

```mermaid
graph LR
    subgraph Storage / Database
        RAW[(CFPB Raw CSV)] --> BRONZE[(DuckDB Bronze)]
        BRONZE --> SILVER[(DuckDB Silver Cleaned)]
        SILVER --> GOLD[(DuckDB Gold Features)]
    end

    subgraph Split Strategy
        GOLD --> TRAIN[(Train 70% Time-based)]
        GOLD --> VAL[(Val 15% Time-based)]
        GOLD --> TEST[(Test 15% Time-based)]
        TEST --> GOLD_EVAL[(Gold Set 1k)]
        GOLD_EVAL --> CI_FROZEN[(CI Frozen 200)]
    end

    subgraph Model Evaluation & Reporting
        TEST --> INFERENCE[Shared Model Runner]
        INFERENCE --> PARQUET[Standard Predictions Parquet]
        PARQUET --> METRICS[Metrics & Bootstrap CI]
        PARQUET --> STATS[McNemar & Paired Wilcoxon]
        PARQUET --> GATE[GitHub Actions Gate]
    end
```

## Data Medallion Architecture
1. **Bronze (`01_bronze.sql`):** Direct schema reflection of the raw consumer complaints database.
2. **Silver (`02_silver_clean.sql`):** Normalizes whitespace, deduplicates on narrative and date, sanitizes text, filters short narratives (<50 chars).
3. **Gold (`03_gold_features.sql`):** Feature generation (word count, mask frequency), product consolidation, time-series partitioning.

## Leakage Prevention
- Splits are strictly temporal: train (< 70th percentile date), validation (70th to 85th percentile date), and test (> 85th percentile date).
- Embedding few-shot exemplars for LLMs are strictly sampled from the train split.
