# Complaint Intelligence & LLM Evaluation Pipeline

[![CI Evaluation](https://github.com/krishna112003/AURIX/actions/workflows/eval.yml/badge.svg)](https://github.com/krishna112003/AURIX/actions/workflows/eval.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Benchmarking classical ML, dense sentence embeddings, and LLMs on CFPB consumer complaints for product classification and structured extraction, backed by statistical hypothesis testing, bias probes, DSPy prompt optimization, DuckDB topic trends, and an automated regression gate.**

---

## 1. Executive Pitch & Architecture

Financial institutions receive tens of thousands of customer complaints monthly. Accurately triaging complaints, isolating root causes, and extracting structured insights must be balanced against latency and cloud inference costs.

This repository answers a crucial engineering question: **Is a generative LLM worth its latency and inference cost over classical ML and dense embeddings?**

```mermaid
flowchart TD
    subgraph Data Layer [Data Medallion & Strict Temporal Splitting]
        A[Raw CFPB CSV: 200k+] --> B[DuckDB Bronze Layer]
        B --> C[DuckDB Silver: Sanitization & Deduplication]
        C --> D[Gold Layer: 49,787 Balanced Samples]
        D -->|Oldest 70%| S1[Train: 34,850]
        D -->|Middle 15%| S2[Validation: 7,468]
        D -->|Newest 15%| S3[Test: 7,469]
    end

    subgraph Modeling Layer [Model Zoo & Inference]
        S1 --> M1[Floor Baselines: Majority & Keyword]
        S1 --> M2[Classical: TF-IDF + Logistic Reg & Linear SVM]
        S1 --> M3[Embeddings: all-MiniLM-L6-v2 + LightGBM]
        S1 --> M4[DistilBERT CPU Smoke Fine-Tune]
        S1 --> M5[Gemini 2.0 Flash: 0-shot & k-NN Few-Shot]
        S1 --> M6[DSPy: BootstrapFewShot Prompt Optimizer]
    end

    subgraph Evaluation & Governance [Standard Parquet & CI Gate]
        M1 & M2 & M3 & M4 & M5 & M6 --> P[Standardized Predictions Schema]
        P --> E1[Statistical Testing: McNemar & Paired Wilcoxon]
        P --> E2[Pareto Frontier: Macro-F1 vs. Cost vs. Latency]
        P --> E3[Bias Probes: Gender, Ethnicity & Typo Counterfactuals]
        P --> T1[BERTopic & DuckDB Window Topic Trends]
        P --> G[LLMOps Regression Gate: F1 Drop, Validity, Cost, Bias]
    end

    subgraph Application [Interactive Dashboard]
        P & E1 & E2 & E3 & T1 --> APP[Streamlit Multi-Tab App]
    end
```

---

## 2. Empirical Benchmark Results

All metrics are measured out-of-sample on the strictly temporal 7,469-complaint test set, serialized directly from `reports/baseline_metrics.json` and `reports/results.md`:

| Model Family | Model Architecture | Macro-F1 (95% CI) | Test Accuracy | ECE (Calibration) | Latency p95 (ms) | Cost / 1k ($) | Pareto Optimal |
|---|---|---|---|---|---|---|---|
| *Floor* | Majority Baseline | 0.0232 [0.022, 0.025] | 11.65% | 0.1121 | 0.00 | $0.0000 | No |
| *Heuristic* | Keyword Rules | 0.3441 [0.330, 0.358] | 47.82% | 0.2229 | 0.04 | $0.0000 | No |
| *Embeddings* | MiniLM-L6-v2 + LightGBM | 0.5490 [0.540, 0.611] | 72.08% | 0.0345 | 0.00 | $0.0000 | No |
| *Classical* | TF-IDF + Linear SVM | 0.5944 [0.585, 0.661] | 76.94% | **0.0188** | 0.00 | $0.0000 | No |
| *Classical* | **TF-IDF + Logistic Reg** | **0.6002 [0.591, 0.665]** | **76.60%** | 0.0497 | 0.00 | **$0.0000** | **Yes ⭐** |

### Key Findings
1. **Classical ML Dominates Efficiency:** TF-IDF with Logistic Regression achieves the highest Macro-F1 (0.6002) at zero API cost and sub-millisecond inference latency.
2. **Calibration:** TF-IDF with Linear SVM and LightGBM showed superior Expected Calibration Error (ECE $\le 0.035$).
3. **Structured Extraction:** LLMs (Gemini / DSPy) excel in Task B (unstructured root cause extraction, urgency scoring, and exact-substring evidence quotation), but are unneeded for primary product routing.

---

## 3. Fairness, Robustness & Bias Probing

Probing with counterfactual perturbations across demographic and geographic features (reported in `reports/bias_report.md`):
- **Gender Name Swap (Male ↔ Female):** **0.00%** flip rate (stable across genders).
- **Ethnicity Name Swap (White ↔ Black/Hispanic):** **0.00%** flip rate (stable across ethnic names).
- **Location Swap (State/City):** **0.00%** flip rate.
- **Typo Injection (Character noise):** **8.00%** flip rate, confirming sensitivity to raw user typing mistakes and motivating robust tokenization.

---

## 4. Topic Trends & Emerging Themes

BERTopic clustering and DuckDB SQL window analysis (reported in `reports/emerging_topics.md`):
- **Top Inbound Themes:** Mortgage servicing & loan modification (2,201 complaints), debt collection calling practices (526 complaints), student loans & auto financing (443 complaints), and credit reporting disputes (337 complaints).
- **Emerging Theme Detection:** Tracked Month-over-Month volume shares, 3-month moving averages, and continuous quarterly growth streaks. Visual trajectories saved in `reports/figures/topics_over_time.png`.

---

## 5. Quickstart & How to Run

### Local Setup (macOS / Linux)
```bash
# 1. Clone repository
git clone https://github.com/krishna112003/AURIX.git
cd "Complaint Intelligence & LLM Evaluation Pipeline"

# 2. Setup virtual environment with uv or pip
make setup

# 3. Run all tests and linters
make test
make lint

# 4. Check LLMOps Regression Gate
python -m complaints.eval.gate

# 5. Launch Streamlit interactive demo
make app
```

### Run with Docker (One-Command Demo Mode)
```bash
# Build Docker image
docker build -t complaints-app .

# Run container (app accessible at http://localhost:8501)
docker run -p 8501:8501 complaints-app
```

---

## 6. Repository Layout

```text
complaint-intelligence/
├── configs/            # YAML configs (data, models, eval, llm, topic_labels)
├── sql/                # DuckDB Medallion transformations (01_bronze, 02_silver, 03_gold)
├── prompts/            # Versioned prompts with XML delimiters & CHANGELOG.md
├── src/complaints/     # Core library
│   ├── data/           # Medallion pipelines & temporal splitters
│   ├── models/         # Baselines, TF-IDF, LightGBM, DistilBERT fine-tune
│   ├── llm/            # Gemini client, caching, DSPy optimizer, extraction
│   ├── eval/           # Metrics, statistical tests, bias probes, regression gate
│   ├── topics/         # BERTopic clustering & DuckDB window trend queries
│   └── tracking/       # MLflow experiment tracking utilities
├── app/                # Streamlit multi-tab application (streamlit_app.py)
├── tests/              # 57 unit tests (leakage, metrics, schemas, gate, topics)
├── reports/            # Markdown reports, metrics JSON, publication figures
├── docs/               # Model card, architecture design, governance
└── Dockerfile          # Containerized deployment
```

---

## 7. Governance, Safety & Limitations

- **PII Protection:** All raw CFPB narratives contain anonymized text masked with `XXXX`. No personal identifying information is stored or unmasked.
- **Prompt Injection Defense:** Untrusted complaints are enclosed inside `<complaint>` XML boundaries with explicit system defense instructions.
- **Zero Hallucination Quoting:** Extraction requires exact verbatim substring matching (`evidence_in_source_rate`) for evidence verification.
- **Strict Anti-Leakage:** Strictly temporal train/val/test splits prevent lookahead bias.
