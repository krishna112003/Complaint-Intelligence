"""Streamlit Demo App for Complaint Intelligence & LLM Evaluation Pipeline.

Provides 4 tabs:
1. Try it: live prediction & extraction
2. Results: benchmark tables, cost-quality Pareto, statistical comparisons
3. Bias lab: counterfactual perturbation and fairness tests
4. Trends: topic modeling over time & emerging themes
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Complaint Intelligence & Evaluation",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Complaint Intelligence & LLM Evaluation Pipeline")
st.markdown(
    "Benchmarking Classical ML vs. Fine-tuned Models vs. LLMs on CFPB Consumer Complaints."
)

tab_try, tab_results, tab_bias, tab_trends = st.tabs(
    ["🔍 Try It", "📊 Results & Pareto", "🧪 Bias Lab", "📈 Topic Trends"]
)

with tab_try:
    st.subheader("Interactive Complaint Inference")
    narrative_input = st.text_area(
        "Enter a customer complaint narrative:",
        height=150,
        placeholder="I opened a dispute regarding an inaccurate charge of $350 on my credit card statement...",
    )
    if st.button("Analyze Complaint", type="primary"):
        st.info("Interactive inference will connect to models in Phase 10 (App phase).")

with tab_results:
    st.subheader("Model Benchmark & Evaluation")
    st.write("Results and Pareto frontier charts will be displayed here from `reports/results.md`.")

with tab_bias:
    st.subheader("Fairness & Robustness Probes")
    st.write("Counterfactual demographic swaps and perturbation checks will be displayed here.")

with tab_trends:
    st.subheader("Emerging Topics & Longitudinal Trends")
    st.write("BERTopic clustering and window-function metrics will be displayed here.")
