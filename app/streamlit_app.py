"""Streamlit Demo App for Complaint Intelligence & LLM Evaluation Pipeline.

Provides 4 interactive tabs:
1. 🔍 Try It: Real-time classification & structured JSON extraction
2. 📊 Results: Benchmark metrics table, cost-accuracy Pareto frontier, calibration
3. 🧪 Bias Lab: Interactive counterfactual demographic & typo perturbation probes
4. 📈 Topic Trends: Longitudinal topic shares and emerging-theme detection
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

# Ensure src/ is on sys.path for direct streamlit executions
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pandas as pd
import streamlit as st

from complaints.config import get_project_root
from complaints.eval.bias_probes import (
    perturb_location,
    perturb_name_ethnicity,
    perturb_name_gender,
    perturb_typos,
)
from complaints.llm.client import get_llm_client
from complaints.llm.extract import extract_complaint_info
from complaints.llm.registry import load_prompt
from complaints.models.baselines import KeywordBaseline

# Configure page
st.set_page_config(
    page_title="Complaint Intelligence & Evaluation",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

root = get_project_root()

# Header
st.title("⚖️ Complaint Intelligence & LLM Evaluation Pipeline")
st.markdown(
    "**Benchmarking Classical ML vs. Dense Sentence Embeddings vs. Generative LLMs on CFPB Consumer Financial Complaints.**"
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info(
        "**Pipeline Version:** 0.1.0\n\n"
        "**Dataset:** CFPB Consumer Complaints (Gold: 49,787 samples, strictly temporal split)\n\n"
        "**Models Evaluated:**\n"
        "- Heuristic / Floor Baselines\n"
        "- TF-IDF + Logistic Regression & SVM\n"
        "- Sentence Transformer + LightGBM\n"
        "- DistilBERT Fine-tune\n"
        "- Gemini 2.0 Flash Zero/Few-shot & DSPy"
    )

    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("- [Reports Directory](file://reports)")
    st.markdown("- [Model Card](file://docs/model_card.md)")
    st.markdown("- [Architecture](file://docs/architecture.md)")

# Cache model initialization
@st.cache_resource
def get_keyword_model() -> KeywordBaseline:
    kb = KeywordBaseline()
    train_path = root / "data/gold/train.parquet"
    if train_path.exists():
        train_df = pd.read_parquet(train_path)
        kb.fit(train_df, train_df)
    else:
        # Fallback dummy training
        dummy = pd.DataFrame(
            {
                "complaint_id": [1, 2],
                "product": ["Credit Reporting", "Debt Collection"],
                "narrative": ["credit", "debt"],
            }
        )
        kb.fit(dummy, dummy)
    return kb


keyword_model = get_keyword_model()

# Create tabs
tab_try, tab_results, tab_bias, tab_trends = st.tabs(
    ["🔍 Try It", "📊 Results & Pareto", "🧪 Bias Lab", "📈 Topic Trends"]
)

# -------------------------------------------------------------
# TAB 1: TRY IT
# -------------------------------------------------------------
with tab_try:
    st.subheader("Interactive Complaint Classification & Extraction")
    st.write(
        "Enter a financial complaint below or click an example to evaluate models and extract structured information."
    )

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    sample_text = ""
    if col_btn1.button("Example: Credit Report Dispute"):
        sample_text = (
            "I noticed an inaccurate late payment on my credit report from Equifax. "
            "I contacted the credit bureau multiple times to dispute this item, but they failed to "
            "investigate or remove the error within the mandated 30-day window."
        )
    if col_btn2.button("Example: Debt Collection Harassment"):
        sample_text = (
            "A collection agency has been calling my personal phone number up to six times a day "
            "demanding payment for a debt that does not belong to me. The debt collector refused to provide "
            "written debt validation and used threatening language."
        )
    if col_btn3.button("Example: Mortgage Escrow Discrepancy"):
        sample_text = (
            "My mortgage servicer improperly recalculated my monthly escrow payment, doubling the required amount. "
            "Despite providing proof of home insurance payment, they placed forced-place insurance on my property."
        )

    narrative_input = st.text_area(
        "Customer Complaint Narrative:",
        value=sample_text,
        height=140,
        placeholder="Type or paste a complaint narrative here...",
    )

    if st.button("🚀 Analyze Complaint", type="primary"):
        if not narrative_input.strip():
            st.warning("Please enter a complaint narrative.")
        else:
            with st.spinner("Classifying and extracting structured data..."):
                t0 = time.time()
                input_df = pd.DataFrame(
                    [
                        {
                            "complaint_id": "live-demo-1",
                            "narrative": narrative_input,
                            "product": "Unknown",
                        }
                    ]
                )

                # Classical / Heuristic prediction
                pred_df = keyword_model.predict(input_df)
                kb_pred = pred_df.iloc[0]["y_pred"]
                kb_conf = pred_df.iloc[0]["confidence"]
                latency_ms = (time.time() - t0) * 1000

                # LLM Extraction
                client = get_llm_client()
                prompt_tmpl, version_tag = load_prompt("extract_v1")
                extraction, audit = extract_complaint_info(
                    narrative=narrative_input,
                    client=client,
                    prompt_template=prompt_tmpl,
                    version_tag=version_tag,
                )

            st.success(f"Analysis completed in {latency_ms:.1f} ms")

            # Display Classification Results
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Predicted Product", kb_pred)
            col2.metric("Model Confidence", f"{kb_conf:.1%}")
            col3.metric("Latency", f"{latency_ms:.1f} ms")
            col4.metric("Inference Cost", "$0.0000")

            st.markdown("---")
            st.subheader("Structured Pydantic Extraction (Task B)")

            if extraction:
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.markdown(f"**Issue Category:** `{extraction.get('issue_category')}`")
                    st.markdown(f"**Root Cause:** {extraction.get('root_cause')}")
                    st.markdown(f"**Requested Resolution:** {extraction.get('requested_resolution')}")
                with col_e2:
                    urgency = extraction.get("urgency", "low")
                    sentiment = extraction.get("sentiment", "negative")
                    st.markdown(f"**Urgency:** `{'🔴 High' if urgency == 'high' else '🟡 Medium' if urgency == 'medium' else '🟢 Low'}`")
                    st.markdown(f"**Sentiment:** `{'😠 Negative' if sentiment == 'negative' else '😐 Neutral' if sentiment == 'neutral' else '😊 Positive'}`")
                    quote = extraction.get("evidence_quote")
                    verified = audit.get("evidence_in_source", False)
                    badge = "✅ Verbatim In Source" if verified else "⚠️ Unverified Excerpt"
                    st.markdown(f"**Evidence Quote ({badge}):**\n> \"{quote}\"")

                with st.expander("View Raw Extraction JSON"):
                    st.json(extraction)
            else:
                st.warning("Extraction schema validation did not return structured data.")

# -------------------------------------------------------------
# TAB 2: RESULTS & PARETO
# -------------------------------------------------------------
with tab_results:
    st.subheader("Model Benchmark Evaluation & Pareto Frontier")
    st.write(
        "Strict out-of-sample evaluation on the 7,469-sample test split, benchmarked across accuracy, "
        "Macro-F1, calibration (ECE), latency, and dollar cost per 1k predictions."
    )

    baseline_metrics_path = root / "reports/baseline_metrics.json"
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path) as f:
            b_data = json.load(f)

        models_data = b_data.get("models", {})
        table_rows = []
        for name, m in models_data.items():
            ci = f"[{m.get('macro_f1_ci_lower', 0):.3f}, {m.get('macro_f1_ci_upper', 0):.3f}]"
            table_rows.append(
                {
                    "Model": m.get("model_name"),
                    "Macro-F1": f"{m.get('macro_f1', 0):.4f}",
                    "95% Bootstrap CI": ci,
                    "Accuracy": f"{m.get('accuracy', 0):.2%}",
                    "ECE (Calibration)": f"{m.get('ece', 0):.4f}",
                    "Cost / 1k ($)": f"${m.get('cost_per_1k', 0):.4f}",
                    "Latency p95 (ms)": f"{m.get('latency_p95_ms', 0):.2f}",
                    "Pareto Optimal": "⭐ Yes" if m.get("is_pareto") else "No",
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    col_fig1, col_fig2 = st.columns(2)
    pareto_img = root / "reports/figures/pareto_frontier.png"
    if pareto_img.exists():
        with col_fig1:
            st.image(str(pareto_img), caption="Quality-Cost-Latency Pareto Frontier", use_container_width=True)

    calib_img = root / "reports/figures/calibration_curves.png"
    if calib_img.exists():
        with col_fig2:
            st.image(str(calib_img), caption="Expected Calibration Error (ECE) Reliability Diagrams", use_container_width=True)

    results_md_path = root / "reports/results.md"
    if results_md_path.exists():
        with st.expander("Read Full Evaluation Report (reports/results.md)"):
            st.markdown(results_md_path.read_text())

# -------------------------------------------------------------
# TAB 3: BIAS LAB
# -------------------------------------------------------------
with tab_bias:
    st.subheader("Fairness, Robustness & Demographic Counterfactual Probes")
    st.write(
        "Evaluates prediction stability and flip rates under demographic substitutions and noisy perturbations."
    )

    test_sentence = st.text_input(
        "Test sentence for counterfactual perturbation:",
        value="My name is James Smith, and I am disputing an unauthorized charge on my account in New York.",
    )

    perturbation_choice = st.selectbox(
        "Select Perturbation Type:",
        ["name_gender (Male ↔ Female)", "name_ethnicity (White ↔ Black/Hispanic)", "location (City/State swap)", "typo (Character noise)"],
    )

    if st.button("Apply Perturbation"):
        pert_type = perturbation_choice.split()[0]
        if pert_type == "name_gender":
            perturbed_sentence = perturb_name_gender(test_sentence)
        elif pert_type == "name_ethnicity":
            perturbed_sentence = perturb_name_ethnicity(test_sentence)
        elif pert_type == "location":
            perturbed_sentence = perturb_location(test_sentence)
        else:
            perturbed_sentence = perturb_typos(test_sentence)

        col_orig, col_pert = st.columns(2)
        with col_orig:
            st.markdown("**Original Text:**")
            st.info(test_sentence)
            df_orig = pd.DataFrame([{"complaint_id": "1", "narrative": test_sentence, "product": "Unknown"}])
            pred_orig = keyword_model.predict(df_orig).iloc[0]["y_pred"]
            st.metric("Original Prediction", pred_orig)

        with col_pert:
            st.markdown(f"**Perturbed Text ({pert_type}):**")
            st.warning(perturbed_sentence)
            df_pert = pd.DataFrame([{"complaint_id": "2", "narrative": perturbed_sentence, "product": "Unknown"}])
            pred_pert = keyword_model.predict(df_pert).iloc[0]["y_pred"]
            flipped = pred_orig != pred_pert
            st.metric("Perturbed Prediction", pred_pert, delta="FLIPPED" if flipped else "UNCHANGED", delta_color="inverse" if flipped else "normal")

    st.markdown("---")
    st.subheader("Systemic Bias Findings (reports/bias_report.md)")
    bias_report_path = root / "reports/bias_report.md"
    if bias_report_path.exists():
        st.markdown(bias_report_path.read_text())

# -------------------------------------------------------------
# TAB 4: TOPIC TRENDS
# -------------------------------------------------------------
with tab_trends:
    st.subheader("Longitudinal Topic Trends & Emerging Themes")
    st.write(
        "Discovered consumer complaint topics using BERTopic and monitored monthly volume share "
        "changes using DuckDB SQL window functions."
    )

    topics_fig_path = root / "reports/figures/topics_over_time.png"
    if topics_fig_path.exists():
        st.image(str(topics_fig_path), caption="Monthly Topic Trajectories (3-Month Moving Average)", use_container_width=True)

    emerging_md_path = root / "reports/emerging_topics.md"
    if emerging_md_path.exists():
        st.markdown(emerging_md_path.read_text())

    topic_labels_path = root / "configs/topic_labels.json"
    if topic_labels_path.exists():
        with st.expander("View All Discovered Topic Labels"):
            with open(topic_labels_path) as f:
                st.json(json.load(f))
