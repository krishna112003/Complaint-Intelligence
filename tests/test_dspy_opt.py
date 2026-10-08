"""Tests for DSPy Prompt Optimization module (Task B)."""

from pathlib import Path

import dspy
import pandas as pd
import pytest

from complaints.llm.dspy_opt import (
    ComplaintExtractionSignature,
    ComplaintExtractor,
    compile_dspy_extractor,
    configure_dspy_lm,
    extraction_composite_metric,
    run_dspy_evaluation,
)


@pytest.fixture(autouse=True)
def setup_fake_dspy():
    """Ensure tests run against deterministic FakeDSPyLM."""
    configure_dspy_lm(api_key=None)


def test_dspy_signature_fields():
    """Verify input and output fields in ComplaintExtractionSignature."""
    sig = ComplaintExtractionSignature
    assert "narrative" in sig.input_fields
    assert "issue_category" in sig.output_fields
    assert "root_cause_summary" in sig.output_fields
    assert "severity" in sig.output_fields
    assert "monetary_relief_flag" in sig.output_fields
    assert "evidence_quote" in sig.output_fields


def test_extraction_composite_metric_bounds_and_scoring():
    """Test composite metric behavior on valid vs invalid extractions."""
    example = dspy.Example(
        narrative="The credit bureau failed to update my credit report balance after dispute.",
        issue_category="Credit Reporting",
    )

    # 1. Perfect match prediction
    pred_perfect = dspy.Prediction(
        issue_category="Credit Reporting",
        root_cause_summary="Credit bureau failed to update balance.",
        evidence_quote="credit report",
    )
    score = extraction_composite_metric(example, pred_perfect)
    assert pytest.approx(score, 0.01) == 1.0  # 0.4 + 0.4 + 0.2

    # 2. Hallucinated evidence (not in narrative)
    pred_hallucinated = dspy.Prediction(
        issue_category="Credit Reporting",
        root_cause_summary="Short summary.",
        evidence_quote="bogus text not in complaint",
    )
    score_hallucinated = extraction_composite_metric(example, pred_hallucinated)
    assert pytest.approx(score_hallucinated, 0.01) == 0.6  # 0.4 + 0.0 + 0.2

    # 3. Invalid taxonomy category and excessive word count
    pred_bad = dspy.Prediction(
        issue_category="INVALID_TAXONOMY",
        root_cause_summary="word " * 35,  # > 30 words
        evidence_quote="credit report",
    )
    score_bad = extraction_composite_metric(example, pred_bad)
    assert pytest.approx(score_bad, 0.01) == 0.4  # 0.0 + 0.4 + 0.0


def test_dspy_compilation_and_serialization(tmp_path: Path):
    """Test BootstrapFewShot compilation on synthetic train slice and serialization."""
    synthetic_train = pd.DataFrame(
        [
            {
                "complaint_id": f"c_{i}",
                "narrative": f"Sample complaint narrative {i} regarding credit report error.",
                "product": "Credit Reporting",
            }
            for i in range(5)
        ]
    )

    out_file = tmp_path / "compiled_test.json"
    _compiled_prog, meta = compile_dspy_extractor(
        train_df=synthetic_train,
        sample_size=3,
        max_bootstrapped_demos=1,
        max_labeled_demos=1,
        output_path=out_file,
    )

    assert out_file.exists()
    assert meta["n_train_examples"] == 3
    assert meta["compile_time_s"] >= 0.0

    # Ensure program can be loaded back
    restored = ComplaintExtractor()
    restored.load(str(out_file))
    res = restored(narrative="Test credit report error")
    assert res.issue_category is not None


def test_dspy_evaluation_prediction_schema(tmp_path: Path):
    """Test evaluation output matches standard predictions parquet schema."""
    synthetic_eval = pd.DataFrame(
        [
            {
                "complaint_id": f"eval_{i}",
                "narrative": f"Consumer narrative {i} with credit report inaccuracy dispute.",
                "product": "Credit Reporting",
            }
            for i in range(4)
        ]
    )

    prog = ComplaintExtractor()
    out_parquet = run_dspy_evaluation(compiled_program=prog, eval_df=synthetic_eval, sample_size=4)
    assert out_parquet.exists()

    df_preds = pd.read_parquet(out_parquet)
    assert len(df_preds) == 4
    for col in ["complaint_id", "y_true", "y_pred", "confidence", "latency_ms", "cost_usd"]:
        assert col in df_preds.columns
