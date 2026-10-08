# Model Card: Complaint Intelligence Classification & Extraction

## Model Details
- **Organization / Author:** Krishna
- **Pipeline Version:** 0.1.0
- **Model Families Evaluated:**
  1. Classical Baselines (TF-IDF + Logistic Regression, Linear SVM)
  2. Dense Embedding Baselines (all-MiniLM-L6-v2 + LightGBM)
  3. LLM Zero/Few-shot & Prompt Optimized (Google Gemini 2.0 Flash / DSPy)

## Intended Use
- **Primary Use Case:** Automated triage, classification, and structured extraction of inbound consumer financial complaints.
- **Out of Scope:** Automatic rejection of customer claims without human review.

## Factors, Metrics & Evaluation
- **Evaluation Metric:** Macro-F1 (primary, to penalize poor minority class performance), Expected Calibration Error (ECE), p50/p95 latency, cost per 1k predictions.
- **Fairness & Bias Probes:** Demographic and regional counterfactual swaps (name and geography substitutions) measuring prediction flip rate.

## Caveats & Limitations
- The underlying CFPB data has personal details masked with `XXXX`. Models trained on this text may rely on structural patterns of masking.
- Hand-checked label noise bounds the maximum achievable accuracy.
