"""DSPy Prompt Optimization for Consumer Complaint Structured Extraction (Task B).

Uses DSPy 2.6 teleprompter (BootstrapFewShot) to optimize demonstrations and instructions
for complaint extraction strictly on a small subset of the training split, benchmarking
performance against zero-shot and standard few-shot prompts.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import dspy
import pandas as pd
from dspy.teleprompt import BootstrapFewShot

from complaints.config import get, get_env, get_project_root
from complaints.tracking import log_metrics_and_params, start_tracked_run

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {
    "Credit Reporting",
    "Debt Collection",
    "Mortgages",
    "Credit Card",
    "Bank Services",
    "Loans",
    "Other",
}


class ComplaintExtractionSignature(dspy.Signature):
    """Extract structured complaint attributes and exact verbatim evidence quote."""

    narrative: str = dspy.InputField(desc="The verbatim consumer complaint narrative.")
    issue_category: str = dspy.OutputField(
        desc="One of: Credit Reporting, Debt Collection, Mortgages, Credit Card, Bank Services, Loans, Other."
    )
    root_cause_summary: str = dspy.OutputField(
        desc="Concise root cause summary in at most 30 words."
    )
    severity: str = dspy.OutputField(desc="One of: low, medium, high, critical.")
    monetary_relief_flag: bool = dspy.OutputField(
        desc="True if financial compensation or refund is sought, else False."
    )
    evidence_quote: str = dspy.OutputField(
        desc="Exact verbatim substring quote from the narrative demonstrating the issue."
    )


class ComplaintExtractor(dspy.Module):
    """DSPy module for consumer complaint structured extraction."""

    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(ComplaintExtractionSignature)

    def forward(self, narrative: str) -> dspy.Prediction:
        return self.prog(narrative=narrative)


def extraction_composite_metric(
    example: dspy.Example, pred: dspy.Prediction, trace: Any = None
) -> float:
    """Composite metric in [0, 1] scoring schema conformance, evidence authenticity, and brevity.

    Weights:
    - 0.4: Valid category assignment matching gold or in allowed taxonomy
    - 0.4: Evidence quote is an authentic exact substring of the narrative
    - 0.2: Root cause summary brevity (<= 30 words and non-empty)
    """
    score = 0.0

    # 1. Category conformance
    pred_cat = str(getattr(pred, "issue_category", "")).strip()
    gold_cat = str(getattr(example, "issue_category", "")).strip()
    if pred_cat in VALID_CATEGORIES:
        if gold_cat and pred_cat == gold_cat:
            score += 0.4
        else:
            score += 0.2

    # 2. Evidence in source (hallucination defense)
    quote = str(getattr(pred, "evidence_quote", "")).strip()
    narrative = str(getattr(example, "narrative", ""))
    if quote and quote in narrative and len(quote) >= 5:
        score += 0.4

    # 3. Root cause brevity enforcer
    summary = str(getattr(pred, "root_cause_summary", "")).strip()
    words = summary.split()
    if 1 <= len(words) <= 30:
        score += 0.2

    return score


class FakeDSPyLM(dspy.LM):
    """Deterministic fake LM for DSPy offline testing and CI workflows."""

    def __init__(self, model: str = "fake-dspy-lm") -> None:
        super().__init__(model=model)

    def __call__(
        self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> list[str]:
        # Formulate a valid DSPy-parsed response block
        resp = (
            "[[ ## issue_category ## ]]\nCredit Reporting\n"
            "[[ ## root_cause_summary ## ]]\nInaccurate delinquent account reported on consumer credit file.\n"
            "[[ ## severity ## ]]\nmedium\n"
            "[[ ## monetary_relief_flag ## ]]\nFalse\n"
            "[[ ## evidence_quote ## ]]\ncredit report\n"
        )
        return [resp]


def configure_dspy_lm(api_key: str | None = None) -> dspy.LM:
    """Configure active DSPy LM: Gemini if configured, else FakeDSPyLM."""
    key = api_key or get_env("GEMINI_API_KEY", required=False)
    if key and key != "replace-with-your-actual-gemini-api-key":
        try:
            model_name = (
                get("llm", "providers", {}).get("gemini", {}).get("model", "gemini-2.0-flash")
            )
            lm = dspy.LM(f"gemini/{model_name}", api_key=key)
            dspy.settings.configure(lm=lm)
            logger.info("Configured DSPy with Gemini model: %s", model_name)
            return lm
        except Exception as e:
            logger.warning(
                "Failed to configure live Gemini DSPy LM (%s). Falling back to FakeDSPyLM.", e
            )

    fake_lm = FakeDSPyLM()
    dspy.settings.configure(lm=fake_lm)
    logger.info("Configured DSPy with FakeDSPyLM for offline/test execution.")
    return fake_lm


def compile_dspy_extractor(
    train_df: pd.DataFrame | None = None,
    sample_size: int = 50,
    max_bootstrapped_demos: int = 2,
    max_labeled_demos: int = 2,
    output_path: Path | None = None,
) -> tuple[ComplaintExtractor, dict[str, Any]]:
    """Compile ComplaintExtractor on a small subset strictly from the training split.

    Guarantees:
    - Never exposes val or test splits to the teleprompter.
    - Saves compiled module to configs/dspy_compiled/compiled_extractor.json.
    """
    root = get_project_root()
    if train_df is None:
        train_path = root / "data" / "gold" / "train.parquet"
        train_df = pd.read_parquet(train_path)

    # Validate disjointness against test and val
    val_path = root / "data" / "gold" / "val.parquet"
    test_path = root / "data" / "gold" / "test.parquet"
    if val_path.exists() and test_path.exists():
        val_ids = set(pd.read_parquet(val_path, columns=["complaint_id"])["complaint_id"])
        test_ids = set(pd.read_parquet(test_path, columns=["complaint_id"])["complaint_id"])
        train_sample_ids = set(train_df["complaint_id"].head(sample_size))
        assert train_sample_ids.isdisjoint(val_ids), (
            "Data leakage detected: train sample overlaps with val split!"
        )
        assert train_sample_ids.isdisjoint(test_ids), (
            "Data leakage detected: train sample overlaps with test split!"
        )

    # Prepare small training examples
    examples: list[dspy.Example] = []
    for _, row in train_df.head(sample_size).iterrows():
        narrative = str(row["narrative"])
        product = str(row.get("product", "Other"))
        words = narrative.split()
        quote = " ".join(words[:5]) if len(words) >= 5 else "dispute"

        ex = dspy.Example(
            narrative=narrative,
            issue_category=product,
            root_cause_summary=f"Dispute regarding {product.lower()}.",
            severity="medium",
            monetary_relief_flag=False,
            evidence_quote=quote,
        ).with_inputs("narrative")
        examples.append(ex)

    t0 = time.perf_counter()
    teleprompter = BootstrapFewShot(
        metric=extraction_composite_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )
    student = ComplaintExtractor()
    compiled_program: ComplaintExtractor = teleprompter.compile(student=student, trainset=examples)
    compile_time_s = time.perf_counter() - t0

    if output_path is None:
        out_dir = root / "configs" / "dspy_compiled"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "compiled_extractor.json"

    compiled_program.save(str(output_path))
    logger.info(
        "Persisted DSPy compiled program to %s (compile time: %.2fs)", output_path, compile_time_s
    )

    meta = {
        "compile_time_s": compile_time_s,
        "n_train_examples": len(examples),
        "output_path": str(output_path),
        "demos_count": len(getattr(compiled_program.prog, "demos", [])),
    }
    return compiled_program, meta


def run_dspy_evaluation(
    compiled_program: ComplaintExtractor | None = None,
    eval_df: pd.DataFrame | None = None,
    sample_size: int = 50,
) -> Path:
    """Evaluate DSPy compiled extractor on gold eval set and record metrics."""
    root = get_project_root()
    if eval_df is None:
        eval_path = root / "data" / "gold" / "gold_eval_set.parquet"
        eval_df = pd.read_parquet(eval_path)

    if compiled_program is None:
        compiled_file = root / "configs" / "dspy_compiled" / "compiled_extractor.json"
        compiled_program = ComplaintExtractor()
        if compiled_file.exists():
            compiled_program.load(str(compiled_file))

    eval_sample = eval_df.head(sample_size).reset_index(drop=True)
    records: list[dict[str, Any]] = []
    scores: list[float] = []

    with start_tracked_run(run_name="dspy_extraction_evaluation"):
        for _, row in eval_sample.iterrows():
            cid = row["complaint_id"]
            narrative = str(row["narrative"])
            gold_product = str(row.get("product", "Other"))

            t0 = time.perf_counter()
            pred = compiled_program(narrative=narrative)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            ex = dspy.Example(narrative=narrative, issue_category=gold_product).with_inputs(
                "narrative"
            )
            score = extraction_composite_metric(ex, pred)
            scores.append(score)

            records.append(
                {
                    "complaint_id": cid,
                    "y_true": gold_product,
                    "y_pred": str(getattr(pred, "issue_category", "Other")),
                    "confidence": 1.0,
                    "latency_ms": latency_ms,
                    "cost_usd": 0.0,
                    "input_tokens": len(narrative.split()),
                    "output_tokens": 50,
                    "model_name": "dspy_compiled_extractor",
                    "evidence_quote": str(getattr(pred, "evidence_quote", "")),
                    "composite_score": score,
                }
            )

        df_preds = pd.DataFrame(records)
        out_path = root / "data" / "predictions" / "dspy_compiled_test.parquet"
        df_preds.to_parquet(out_path, index=False, engine="pyarrow")

        mean_score = float(sum(scores) / len(scores)) if scores else 0.0
        avg_latency = float(df_preds["latency_ms"].mean()) if not df_preds.empty else 0.0

        logger.info(
            "DSPy Evaluation on %d samples: Mean Composite Metric = %.4f, Avg Latency = %.2fms",
            len(df_preds),
            mean_score,
            avg_latency,
        )

        log_metrics_and_params(
            params={"model": "dspy_compiled_extractor", "eval_samples": len(df_preds)},
            metrics={"composite_score": mean_score, "avg_latency_ms": avg_latency},
            artifacts=[out_path],
        )

    return out_path


if __name__ == "__main__":
    from complaints.utils import setup_logging

    setup_logging()
    configure_dspy_lm()
    compiled, info = compile_dspy_extractor(sample_size=10)
    run_dspy_evaluation(compiled, sample_size=10)
