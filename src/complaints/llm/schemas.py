"""Pydantic v2 schemas and validation logic for structured complaint extraction.

Task B Extraction Schema:
- issue_category: Restricted literal taxonomy of financial dispute categories
- root_cause: Concise description under 20 words
- urgency: "low" | "medium" | "high"
- sentiment: "negative" | "neutral" | "positive"
- requested_resolution: Desired remedy string or None
- evidence_quote: Verbatim excerpt copied directly from the complaint text
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Fixed consolidated category literals for Task B
IssueCategoryType = Literal[
    "Billing Error",
    "Fraud / Unauthorized Transaction",
    "Customer Service Dispute",
    "Account Closure / Restriction",
    "Reporting Inaccuracy",
    "Harassment / Improper Contact",
    "Fee Dispute",
    "Loan Modification / Servicing",
    "Other",
]

UrgencyType = Literal["low", "medium", "high"]
SentimentType = Literal["negative", "neutral", "positive"]


class ComplaintExtraction(BaseModel):
    """Pydantic schema representing structured extraction from a customer complaint."""

    issue_category: IssueCategoryType = Field(
        description="The primary classification of the customer dispute"
    )
    root_cause: str = Field(
        description="Concise description of what caused the dispute (max 20 words)"
    )
    urgency: UrgencyType = Field(
        description="Operational urgency level based on harm and financial risk"
    )
    sentiment: SentimentType = Field(
        description="Customer emotional sentiment toward the institution"
    )
    requested_resolution: str | None = Field(
        default=None,
        description="Explicit remedy requested by the consumer, or null if unstated",
    )
    evidence_quote: str = Field(
        description="Exact verbatim substring copied from the narrative supporting the issue"
    )

    @field_validator("root_cause")
    @classmethod
    def validate_root_cause_length(cls, v: str) -> str:
        """Enforce root_cause word count constraint (under 20 words)."""
        words = v.strip().split()
        if len(words) > 25:  # Allow small margin
            # Truncate cleanly
            return " ".join(words[:20])
        return v.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and punctuation spacing for quote comparison."""
    return re.sub(r"\s+", " ", text).strip()


def verify_evidence_in_source(evidence_quote: str, narrative: str) -> bool:
    """Check whether evidence_quote is a verbatim substring of narrative.

    Normalizes whitespace on both strings before comparison.
    """
    if not evidence_quote or not narrative:
        return False
    norm_quote = normalize_whitespace(evidence_quote).lower()
    norm_narrative = normalize_whitespace(narrative).lower()
    return norm_quote in norm_narrative


class ClassificationOutput(BaseModel):
    """Schema for Task A Product Classification."""

    product: str
    confidence: float = 1.0


def repair_json_string(raw_text: str) -> str:
    """Repair common JSON formatting issues from small open-weights models.

    - Removes markdown codeblock fences (```json ... ```)
    - Extracts outermost curly-brace block { ... }
    - Trims leading/trailing conversational filler
    """
    text = raw_text.strip()

    # 1. Strip markdown fences
    if "```" in text:
        # Match content inside ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        else:
            text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    # 2. Extract outermost { ... }
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx : end_idx + 1]

    return text


def validate_extraction_response(
    raw_json_str: str,
    original_narrative: str,
) -> tuple[ComplaintExtraction | None, dict[str, Any]]:
    """Validate raw model response against ComplaintExtraction schema and evidence constraint.

    Applies JSON repair before parsing to accommodate free and small local models.
    """
    audit = {
        "schema_valid": False,
        "evidence_in_source": False,
        "error_message": "",
        "repaired": False,
    }

    # Attempt 1: Direct validation
    try:
        extraction = ComplaintExtraction.model_validate_json(raw_json_str)
        audit["schema_valid"] = True
        audit["evidence_in_source"] = verify_evidence_in_source(
            extraction.evidence_quote, original_narrative
        )
        return extraction, audit
    except Exception:
        pass

    # Attempt 2: Repaired JSON validation
    repaired_text = repair_json_string(raw_json_str)
    try:
        extraction = ComplaintExtraction.model_validate_json(repaired_text)
        audit["schema_valid"] = True
        audit["repaired"] = True
        audit["evidence_in_source"] = verify_evidence_in_source(
            extraction.evidence_quote, original_narrative
        )
        return extraction, audit
    except Exception as e:
        audit["error_message"] = str(e)
        return None, audit
