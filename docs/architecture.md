# Architecture & System Design

This document outlines the end-to-end design, data lineage, evaluation lifecycle, and governance controls of the **Complaint Intelligence & LLM Evaluation Pipeline**.

---

## 1. End-to-End System Architecture

```mermaid
graph TD
    subgraph Data Medallion [Data Ingestion & Medallion Pipeline]
        RAW[(Raw CFPB CSV)] -->|DuckDB 01_bronze.sql| BRONZE[(Bronze Table)]
        BRONZE -->|02_silver_clean.sql: sanitize, dedup, mask| SILVER[(Silver Cleaned)]
        SILVER -->|03_gold_features.sql: balance cap, tokenize| GOLD[(Gold Features: 49,787)]
    end

    subgraph Temporal Splitting [Strict Leakage-Free Temporal Split]
        GOLD -->|Oldest 70%| TRAIN[(Train: 34,850)]
        GOLD -->|Middle 15%| VAL[(Val: 7,468)]
        GOLD -->|Newest 15%| TEST[(Test: 7,469)]
        TEST -->|Subsampled| CI_EVAL[(CI Frozen Set: 200)]
    end

    subgraph Model Zoo [Benchmarked Model Zoo]
        TRAIN --> BASELINES[Majority & Keyword Rules]
        TRAIN --> CLASSICAL[TF-IDF Logistic Regression & SVM]
        TRAIN --> EMBEDDINGS[MiniLM-L6-v2 + LightGBM]
        TRAIN --> TRANSFORMER[DistilBERT CPU Fine-tune]
        TRAIN --> FEWSHOT[Embedding Few-Shot Retriever]
        FEWSHOT --> GEMINI[Gemini 2.0 Flash / FakeLLM]
        TRAIN --> DSPY[DSPy BootstrapFewShot Optimizer]
    end

    subgraph Evaluation & Governance [Evaluation, Probing & CI Gate]
        TEST --> PREDICTOR[Standard Prediction Schema]
        PREDICTOR --> METRICS[Macro-F1, Accuracy, ECE, CIs]
        PREDICTOR --> STATS[McNemar & Paired Wilcoxon Tests]
        PREDICTOR --> BIAS[Demographic Counterfactual Probes]
        TEST --> TOPICS[BERTopic & DuckDB Window Trends]
        METRICS --> GATE[LLMOps Regression Gate]
        GATE --> CI[GitHub Actions CI/CD]
    end

    subgraph User Delivery [Interactive Web Application]
        PREDICTOR --> APP[Streamlit Multi-Tab App]
        METRICS --> APP
        BIAS --> APP
        TOPICS --> APP
    end
```

---

## 2. Medallion Storage Lineage

1. **Bronze (`sql/01_bronze.sql`)**: Direct columnar reflection of the raw CFPB CSV. Retains raw types, dates, and consumer narratives.
2. **Silver (`sql/02_silver_clean.sql`)**: Cleans whitespace, drops empty or short narratives (<50 characters), deduplicates on narrative text and date, and standardizes dates.
3. **Gold (`sql/03_gold_features.sql`)**: Consolidates 18+ raw CFPB product labels into 10 clean business categories, applies a 10k per-class balance cap, extracts word counts, character lengths, and mask frequencies.

---

## 3. Strict Anti-Leakage Safeguards

- **Temporal Split Boundaries**: Splits are purely temporal—never randomly shuffled. The oldest 70% of complaints form the train split; the subsequent 15% form validation; the final 15% form the test split.
- **Embedding Retrieval Isolation**: The k-NN few-shot exemplar pool for LLM prompt augmentation only indexes representations from the training split.
- **Evaluation Caching**: All test predictions are serialized to immutable Parquet files with fixed schemas, preventing test-time feature re-fitting.

---

## 4. LLMOps Regression Gate & CI Guardrails

The deployment gate (`src/complaints/eval/gate.py`) enforces four strict quality, validity, efficiency, and fairness thresholds before any prompt or model candidate is accepted:

1. **Quality**: $\Delta \text{Macro-F1} \le 0.02$ (candidate must not lose more than 2 F1 points compared to baseline).
2. **Schema Validity**: $\ge 98.0\%$ parse success against Pydantic schema constraints.
3. **Inference Cost**: Cost increase $\le 20.0\%$ over baseline.
4. **Fairness / Robustness**: Counterfactual flip rate under name substitutions $\le 5.0\%$.

---

## 5. Topic Trends & Emerging Themes

- **Embedding-Aware BERTopic**: Uses precomputed sentence embeddings (`all-MiniLM-L6-v2`) to eliminate redundant encoding overhead.
- **DuckDB Window Analysis**: Monthly complaint volume shares are aggregated, tracking Month-over-Month (MoM) share changes, 3-month moving averages, and consecutive positive growth streaks ($\ge 3$ months) to flag emerging systemic complaints.
