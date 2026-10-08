"""Structured Information Extraction (Task B) with Pydantic Validation.

Extracts:
- issue_category
- root_cause
- urgency (low/medium/high)
- sentiment (negative/neutral/positive)
- requested_resolution
- evidence_quote (verbatim excerpt)

Measures and logs:
- schema_valid_rate (rate of successfully parsed Pydantic schemas)
- retry_rate (rate of retries required to reach valid JSON)
- evidence_in_source_rate (anti-hallucination check: quote is exact substring of narrative)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from complaints.config import get, get_project_root, setup_logging
from complaints.llm.client import LLMClient, get_llm_client
from complaints.llm.registry import format_untrusted_prompt, load_prompt
from complaints.llm.schemas import validate_extraction_response
from complaints.tracking.mlflow_utils import log_metrics_and_params, start_tracked_run

logger = logging.getLogger(__name__)


def extract_complaint_info(
    narrative: str,
    client: LLMClient,
    prompt_template: str,
    version_tag: str,
    max_retries: int = 2,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Extract structured data with retry on schema parse failure."""
    retries_used = 0
    formatted_prompt = format_untrusted_prompt(prompt_template, narrative=narrative)

    messages = [
        {"role": "user", "content": formatted_prompt},
    ]

    while retries_used <= max_retries:
        response = client.complete(
            messages,
            temperature=0.0,
            prompt_version=version_tag,
            response_format={"type": "json_object"},
        )

        extraction, audit = validate_extraction_response(response.text, narrative)
        if extraction is not None:
            # Valid schema output
            return extraction.model_dump(), {
                "schema_valid": True,
                "evidence_in_source": audit["evidence_in_source"],
                "retries": retries_used,
                "latency_ms": response.latency_ms,
                "cost_usd": response.cost_usd,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }

        # Invalid schema: add correction instruction and retry
        retries_used += 1
        messages.append({"role": "assistant", "content": response.text})
        messages.append(
            {
                "role": "user",
                "content": f"Your previous response had schema validation error: {audit['error_message']}. "
                "Please fix and output ONLY valid JSON matching the exact schema.",
            }
        )

    # Failed all retries
    return None, {
        "schema_valid": False,
        "evidence_in_source": False,
        "retries": retries_used,
        "latency_ms": 0.0,
        "cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def run_structured_extraction(
    sample_size: int = 200,
    client: LLMClient | None = None,
) -> Path:
    """Run Task B extraction on gold evaluation complaints and save results."""
    root = get_project_root()
    gold_dir = root / get("data", "gold_dir", "data/gold")
    out_dir = root / "data" / "extractions"
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_path = gold_dir / "gold_eval_set.parquet"
    df_gold = pd.read_parquet(gold_path).head(sample_size)

    llm = client or get_llm_client()
    template, version_tag = load_prompt("extract", version="v1")

    logger.info(
        "Running Task B extraction on %d complaints using %s...", len(df_gold), llm.model_name
    )

    records: list[dict[str, Any]] = []
    valid_count = 0
    evidence_count = 0
    total_retries = 0

    with start_tracked_run(run_name=f"task_b_extraction_{llm.model_name}"):
        for _, row in df_gold.reset_index(drop=True).iterrows():
            cid = row["complaint_id"]
            narrative = str(row["narrative"])

            extracted_data, meta = extract_complaint_info(
                narrative=narrative,
                client=llm,
                prompt_template=template,
                version_tag=version_tag,
            )

            if meta["schema_valid"]:
                valid_count += 1
            if meta["evidence_in_source"]:
                evidence_count += 1
            total_retries += meta["retries"]

            record: dict[str, Any] = {
                "complaint_id": cid,
                "original_product": row.get("product", ""),
                "schema_valid": meta["schema_valid"],
                "evidence_in_source": meta["evidence_in_source"],
                "retries": meta["retries"],
                "latency_ms": meta["latency_ms"],
                "cost_usd": meta["cost_usd"],
            }
            if extracted_data:
                record.update(extracted_data)
            records.append(record)

        df_out = pd.DataFrame(records)
        out_parquet = out_dir / f"extractions_{llm.model_name}.parquet"
        df_out.to_parquet(out_parquet, index=False, engine="pyarrow")

        n = len(df_gold)
        schema_valid_rate = float(valid_count / n)
        evidence_in_source_rate = float(evidence_count / n)
        retry_rate = float(total_retries / n)

        logger.info(
            "Task B Extraction Audit:\n"
            "  Schema-Valid Rate:       %.2f%% (%d/%d)\n"
            "  Evidence-In-Source Rate: %.2f%% (%d/%d)\n"
            "  Retry Rate:              %.2f per sample",
            schema_valid_rate * 100,
            valid_count,
            n,
            evidence_in_source_rate * 100,
            evidence_count,
            n,
            retry_rate,
        )

        log_metrics_and_params(
            params={"model_name": llm.model_name, "prompt_version": version_tag, "n_samples": n},
            metrics={
                "schema_valid_rate": schema_valid_rate,
                "evidence_in_source_rate": evidence_in_source_rate,
                "retry_rate": retry_rate,
                "total_cost_usd": float(df_out["cost_usd"].sum()),
            },
            artifacts=[out_parquet],
        )

    return out_parquet


if __name__ == "__main__":
    setup_logging()
    run_structured_extraction()
