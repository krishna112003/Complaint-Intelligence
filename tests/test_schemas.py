"""Unit tests for Pydantic v2 structured extraction schemas and validation helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from complaints.llm.schemas import (
    ComplaintExtraction,
    validate_extraction_response,
    verify_evidence_in_source,
)


def test_valid_complaint_extraction():
    """Verify that a compliant JSON payload parses into ComplaintExtraction."""
    data = {
        "issue_category": "Billing Error",
        "root_cause": "Double charge of $45 fee on account",
        "urgency": "medium",
        "sentiment": "negative",
        "requested_resolution": "Full refund of duplicate fee",
        "evidence_quote": "was charged twice on statement",
    }
    model = ComplaintExtraction(**data)
    assert model.issue_category == "Billing Error"
    assert model.urgency == "medium"
    assert model.sentiment == "negative"
    assert model.requested_resolution == "Full refund of duplicate fee"


def test_invalid_category_raises_validation_error():
    """Verify that an unlisted category fails Pydantic validation."""
    data = {
        "issue_category": "NonExistentCategory",
        "root_cause": "Reason",
        "urgency": "low",
        "sentiment": "neutral",
        "evidence_quote": "quote",
    }
    with pytest.raises(ValidationError):
        ComplaintExtraction(**data)


def test_root_cause_word_count_truncation():
    """Verify that root cause strings over 20 words are gracefully truncated."""
    long_reason = " ".join([f"word{i}" for i in range(40)])
    data = {
        "issue_category": "Fee Dispute",
        "root_cause": long_reason,
        "urgency": "low",
        "sentiment": "negative",
        "evidence_quote": "fee was applied",
    }
    model = ComplaintExtraction(**data)
    assert len(model.root_cause.split()) <= 20


def test_evidence_in_source_verifier():
    """Verify evidence quote exact substring check with whitespace normalization."""
    narrative = "I was   charged a $35 fee on    my monthly statement without notice."
    quote_valid = "charged a $35 fee on my monthly statement"
    quote_invalid = "I was charged an unauthorized $500 fee"

    assert verify_evidence_in_source(quote_valid, narrative) is True
    assert verify_evidence_in_source(quote_invalid, narrative) is False


def test_validate_extraction_response_helper():
    """Verify validate_extraction_response returns parsed model and audit dictionary."""
    narrative = "The bank charged an overdraft fee despite positive balance."
    json_good = """
    {
        "issue_category": "Fee Dispute",
        "root_cause": "Overdraft fee charged on positive balance",
        "urgency": "low",
        "sentiment": "negative",
        "requested_resolution": "Fee waiver",
        "evidence_quote": "charged an overdraft fee"
    }
    """
    model, audit = validate_extraction_response(json_good, narrative)
    assert model is not None
    assert audit["schema_valid"] is True
    assert audit["evidence_in_source"] is True

    json_bad = '{"invalid": "json}'
    model_bad, audit_bad = validate_extraction_response(json_bad, narrative)
    assert model_bad is None
    assert audit_bad["schema_valid"] is False
