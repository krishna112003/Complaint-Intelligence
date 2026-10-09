# Model Card: CFPB Complaint Intelligence & LLM Evaluation

## Model Details
- **Organization / Authors:** Krishna (AI & Data Science Student)
- **Pipeline Version:** 0.1.0
- **Evaluated Architectures:**
  - Heuristic Floor: Majority Class Baseline, Keyword Rules Baseline
  - Classical Supervised: TF-IDF + Logistic Regression, TF-IDF + Linear Support Vector Classifier
  - Dense Representation: Sentence Transformers (`all-MiniLM-L6-v2`) + LightGBM
  - Generative & DSPy: OpenAI-compatible LLM (Ollama / Groq / OpenRouter free tier) & DSPy BootstrapFewShot Optimized

---

## Intended Use
- **Primary Applications:** Real-time triage, classification, and structured root-cause extraction from incoming consumer complaints regarding financial institutions.
- **Out of Scope:** Automatic rejection or denial of consumer claims without human dispute specialist review.
- **Target Users:** Financial compliance officers, customer intelligence analysts, dispute resolution operations.

---

## Evaluation Data & Methodology
- **Source:** Consumer Financial Protection Bureau (CFPB) public complaints database.
- **Gold Dataset Size:** 49,787 complaints across 10 consolidated financial categories.
- **Splitting Strategy:** Strict temporal partitioning (train: 34,850 oldest, val: 7,468, test: 7,469 newest) to prevent future-data contamination.
- **Primary Metric:** Macro-Averaged F1 score (weights all categories equally regardless of frequency to protect minority classes).
- **Secondary Metrics:** Accuracy, Expected Calibration Error (ECE), p50/p95 latency, cost per 1,000 predictions, bootstrap confidence intervals, McNemar's exact test, and Wilcoxon signed-rank tests.

---

## Quantitative Results Summary
The empirical evaluation results are persisted in `reports/baseline_metrics.json` and `reports/results.md`:

| Model Architecture | Macro-F1 (95% CI) | Test Accuracy | ECE (Calibration) | Cost / 1k ($) |
|---|---|---|---|---|
| **Majority Baseline** | 0.0232 [0.022, 0.025] | 11.65% | 0.1121 | $0.00 |
| **Keyword Baseline** | 0.3441 [0.330, 0.358] | 47.82% | 0.2229 | $0.00 |
| **Embeddings + LightGBM** | 0.5490 [0.540, 0.611] | 72.08% | 0.0345 | $0.00 |
| **TF-IDF + Linear SVM** | 0.5944 [0.585, 0.661] | 76.94% | 0.0188 | $0.00 |
| **TF-IDF + Logistic Regression** | **0.6002 [0.591, 0.665]** | **76.60%** | **0.0497** | **$0.00** |

*See `reports/figures/pareto_frontier.png` for quality-cost trade-offs and `reports/figures/calibration_curves.png` for reliability diagrams.*

---

## Fairness & Robustness Findings
Counterfactual perturbation testing evaluated prediction flip rates under demographic substitutions (documented in `reports/bias_report.md`):
- **Gender Name Substitutions:** 0.00% flip rate across probed complaints.
- **Ethnicity Name Substitutions:** 0.00% flip rate across probed complaints.
- **Geographic Substitutions:** 0.00% flip rate across state/city substitutions.
- **Typo Robustness:** 8.00% flip rate under character noise perturbations, highlighting the utility of robust spell-checking or subword tokenization in production pre-processing.

---

## Operational Guardrails & Limitations
1. **Masking Dependency:** Raw CFPB narratives censor sensitive entity names and account numbers with `XXXX`. Models may partially leverage mask density patterns.
2. **Label Ambiguity:** Overlapping categories (such as Credit Card vs. Personal Loan balance transfers) create natural classification boundaries where multi-label tagging may be advantageous in future iterations.
3. **CI Governance:** Any prompt modification must satisfy the regression gate (`src/complaints/eval/gate.py`), ensuring macro-F1 regression $\le 0.02$, schema validity $\ge 98\%$, cost growth $\le 20\%$, and demographic flip rate $\le 5\%$.
