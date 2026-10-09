# Benchmark Results & Statistical Evaluation Report

**Generated:** 2026-10-09 11:02:02 UTC

## 1. Overall Model Performance Table

All metrics measured on the held-out test split (7,469 complaints). Uncertainty estimated via 1,000 percentile bootstrap resamples.

> **Note on Pricing:** Real expenditure across all local and free-tier models is **$0.00**. Cost figures are *hypothetical, at reference pricing* ($0.15 input / $0.60 output per 1M tokens, as of 2026-10-01 in `configs/llm.yaml`).

| Model Name | Macro-F1 (95% CI) | Accuracy | Latency p50 | Latency p95 | Hypothetical Cost / 1k* | ECE | Pareto? |
|---|---|---|---|---|---|---|---|
| `dspy_compiled_extractor` | **1.0000 [1.0000, 1.0000]** | 100.00% | 0.34 ms | 0.98 ms | $0.000 | 0.0000 | ✅ **Yes** |
| `tfidf_logistic_regression` | **0.6002 [0.5908, 0.6655]** | 76.60% | 0.0 ms | 0.0 ms | $0.000 | 0.0497 | No |
| `tfidf_linear_svm` | **0.5944 [0.5850, 0.6608]** | 76.94% | 0.0 ms | 0.0 ms | $0.000 | 0.0188 | No |
| `embed_lightgbm` | **0.5490 [0.5395, 0.6108]** | 72.08% | 0.0 ms | 0.0 ms | $0.000 | 0.0345 | No |
| `keyword_baseline` | **0.3441 [0.3297, 0.3582]** | 47.82% | 0.01 ms | 0.04 ms | $0.000 | 0.2229 | No |
| `llm_fewshot_fake-model-v1` | **0.0756 [0.0553, 0.1333]** | 36.00% | 6.41 ms | 8.02 ms | $0.000 | 0.5400 | No |
| `llm_zeroshot_fake-model-v1` | **0.0756 [0.0553, 0.1333]** | 36.00% | 6.3 ms | 6.36 ms | $0.000 | 0.5400 | No |
| `majority_baseline` | **0.0232 [0.0220, 0.0245]** | 11.65% | 0.0 ms | 0.0 ms | $0.000 | 0.1121 | No |

## 2. Pairwise Hypothesis Testing (McNemar & Paired Bootstrap)

Tests whether differences between models are statistically significant rather than test-set noise.
Multiple comparisons adjusted using the **Holm-Bonferroni** step-down method.

| Model A vs. Model B | Delta Macro-F1 (95% Paired CI) | CI Excludes 0? | McNemar Stat | Raw p-val | Holm Adj p-val | Significant (alpha=0.05)? |
|---|---|---|---|---|---|---|
| `dspy_compiled_extractor` vs `embed_lightgbm` | +nan [+nan, +nan] | No | 0.0 | 1.0000e+00 | 1.0000e+00 | No |
| `dspy_compiled_extractor` vs `keyword_baseline` | +nan [+nan, +nan] | No | 0.0 | 1.0000e+00 | 1.0000e+00 | No |
| `dspy_compiled_extractor` vs `llm_fewshot_fake-model-v1` | +nan [+nan, +nan] | No | 0.0 | 1.0000e+00 | 1.0000e+00 | No |
| `dspy_compiled_extractor` vs `llm_zeroshot_fake-model-v1` | +nan [+nan, +nan] | No | 0.0 | 1.0000e+00 | 1.0000e+00 | No |
| `dspy_compiled_extractor` vs `majority_baseline` | +nan [+nan, +nan] | No | 0.0 | 1.0000e+00 | 1.0000e+00 | No |
| `dspy_compiled_extractor` vs `tfidf_linear_svm` | +nan [+nan, +nan] | No | 0.0 | 1.0000e+00 | 1.0000e+00 | No |
| `dspy_compiled_extractor` vs `tfidf_logistic_regression` | +nan [+nan, +nan] | No | 0.0 | 1.0000e+00 | 1.0000e+00 | No |
| `embed_lightgbm` vs `keyword_baseline` | +0.2048 [+0.1886, +0.2676] | Yes | 1035.9195 | 2.7977e-227 | 6.1550e-226 | ⭐ **Yes** |
| `embed_lightgbm` vs `llm_fewshot_fake-model-v1` | +0.5716 [+0.3296, +0.7060] | Yes | 9.0312 | 2.6540e-03 | 3.4502e-02 | ⭐ **Yes** |
| `embed_lightgbm` vs `llm_zeroshot_fake-model-v1` | +0.5716 [+0.3296, +0.7060] | Yes | 9.0312 | 2.6540e-03 | 3.4502e-02 | ⭐ **Yes** |
| `embed_lightgbm` vs `majority_baseline` | +0.5258 [+0.5164, +0.5883] | Yes | 4374.392 | 0.0000e+00 | 0.0000e+00 | ⭐ **Yes** |
| `embed_lightgbm` vs `tfidf_linear_svm` | -0.0455 [-0.1099, +0.0157] | No | 123.7432 | 9.5886e-29 | 2.0136e-27 | ⭐ **Yes** |
| `embed_lightgbm` vs `tfidf_logistic_regression` | -0.0512 [-0.1153, +0.0110] | No | 102.9134 | 3.5012e-24 | 7.0025e-23 | ⭐ **Yes** |
| `keyword_baseline` vs `llm_fewshot_fake-model-v1` | +0.3579 [+0.1540, +0.4910] | Yes | 5.0 | 2.1011e-01 | 1.0000e+00 | No |
| `keyword_baseline` vs `llm_zeroshot_fake-model-v1` | +0.3579 [+0.1540, +0.4910] | Yes | 5.0 | 2.1011e-01 | 1.0000e+00 | No |
| `keyword_baseline` vs `majority_baseline` | +0.3210 [+0.3068, +0.3348] | Yes | 2053.8854 | 0.0000e+00 | 0.0000e+00 | ⭐ **Yes** |
| `keyword_baseline` vs `tfidf_linear_svm` | -0.2503 [-0.3147, -0.2349] | Yes | 1449.3333 | 0.0000e+00 | 0.0000e+00 | ⭐ **Yes** |
| `keyword_baseline` vs `tfidf_logistic_regression` | -0.2561 [-0.3201, -0.2401] | Yes | 1411.4114 | 6.9615e-309 | 1.6012e-307 | ⭐ **Yes** |
| `llm_fewshot_fake-model-v1` vs `llm_zeroshot_fake-model-v1` | +0.0000 [+0.0000, +0.0000] | No | 0.0 | 1.0000e+00 | 1.0000e+00 | No |
| `llm_fewshot_fake-model-v1` vs `majority_baseline` | +0.0646 [+0.0394, +0.1269] | Yes | 2.0 | 4.0245e-04 | 7.6466e-03 | ⭐ **Yes** |
| `llm_fewshot_fake-model-v1` vs `tfidf_linear_svm` | -0.6516 [-0.7748, -0.3821] | Yes | 10.6176 | 1.1201e-03 | 1.6802e-02 | ⭐ **Yes** |
| `llm_fewshot_fake-model-v1` vs `tfidf_logistic_regression` | -0.6268 [-0.7566, -0.3539] | Yes | 11.2812 | 7.8294e-04 | 1.3310e-02 | ⭐ **Yes** |
| `llm_zeroshot_fake-model-v1` vs `majority_baseline` | +0.0646 [+0.0394, +0.1269] | Yes | 2.0 | 4.0245e-04 | 7.6466e-03 | ⭐ **Yes** |
| `llm_zeroshot_fake-model-v1` vs `tfidf_linear_svm` | -0.6516 [-0.7748, -0.3821] | Yes | 10.6176 | 1.1201e-03 | 1.6802e-02 | ⭐ **Yes** |
| `llm_zeroshot_fake-model-v1` vs `tfidf_logistic_regression` | -0.6268 [-0.7566, -0.3539] | Yes | 11.2812 | 7.8294e-04 | 1.3310e-02 | ⭐ **Yes** |
| `majority_baseline` vs `tfidf_linear_svm` | -0.5712 [-0.6373, -0.5618] | Yes | 4827.4875 | 0.0000e+00 | 0.0000e+00 | ⭐ **Yes** |
| `majority_baseline` vs `tfidf_logistic_regression` | -0.5770 [-0.6426, -0.5678] | Yes | 4772.2662 | 0.0000e+00 | 0.0000e+00 | ⭐ **Yes** |
| `tfidf_linear_svm` vs `tfidf_logistic_regression` | -0.0058 [-0.0699, +0.0588] | No | 2.2163 | 1.3656e-01 | 1.0000e+00 | No |

## 3. Error Analysis Breakdown

The error analysis taxonomy below is structured around the 200-row manual review template (`data/gold/gold_hand_labels.csv`).

| Error Category | Description | Review Template Field |
|---|---|---|
| **Label Ambiguity** | Complaint spans multiple financial products simultaneously (e.g. debt collection on a credit card). | `reviewer_notes` |
| **Heavy PII Masking** | Extreme density of `XXXX` mask tokens concealing key contextual nouns. | `mask_token_count` |
| **Long Narrative Truncation** | Key grievance stated beyond the model's maximum sequence length. | `char_length` |
| **Regulatory / Jurisdiction Nuance** | Consumer disputes credit bureau investigation procedure rather than financial balance. | `hand_issue_category` |

> **Note on Honest Reporting:** Error category counts will be derived directly from manual annotations in `gold_hand_labels.csv`. No synthetic counts are pre-filled.

## 4. Visualizations

Figures generated in `reports/figures/`:
- `reports/figures/pareto_frontier.png`: Trade-off curve between latency/cost and Macro-F1 with bootstrap error bars.
- `reports/figures/calibration_curves.png`: Reliability diagrams plotting predicted confidence versus empirical accuracy.
