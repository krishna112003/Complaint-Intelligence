"""Unit tests for demographic, location, and perturbation bias probes."""

from __future__ import annotations

import random

import pandas as pd
import pytest

from complaints.eval.bias_probes import (
    FEMALE_NAMES,
    MALE_NAMES,
    _inject_typos,
    _location_swap,
    _name_swap,
    compute_flip_rate,
    generate_counterfactuals,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "complaint_id": ["c1", "c2", "c3"],
            "narrative": [
                "James filed a complaint about his credit report in New York California.",
                "Mary noticed an error on her mortgage in Chicago Illinois.",
                "Robert was charged incorrectly on his credit card in Houston Texas.",
            ],
            "product": ["Credit Reporting", "Mortgages", "Credit Card"],
        }
    )


def test_name_swap_replaces_male_with_female():
    """Verify male names are swapped with female names."""
    rng = random.Random(42)
    text = "James contacted support about his account."
    result = _name_swap(text, MALE_NAMES, FEMALE_NAMES, rng)
    assert "James" not in result
    # One of the female names should appear
    assert any(name in result for name in FEMALE_NAMES)


def test_location_swap_changes_city():
    """Verify city references are swapped."""
    rng = random.Random(42)
    text = "I moved to New York and opened a bank account."
    result = _location_swap(text, rng)
    assert "New York" not in result
    assert len(result) > 0


def test_typo_injection_changes_text():
    """Verify typo injection modifies text while keeping it mostly intact."""
    rng = random.Random(42)
    text = "This is a normal complaint about my credit card balance."
    result = _inject_typos(text, rng, rate=0.1)  # Higher rate for testing
    assert result != text
    # Should be roughly same length (± 20%)
    assert abs(len(result) - len(text)) < len(text) * 0.3


def test_generate_counterfactuals_preserves_shape():
    """Verify counterfactual generation preserves DataFrame shape."""
    df = _sample_df()
    for pert in ["name_gender", "name_ethnicity", "location", "typo"]:
        df_cf = generate_counterfactuals(df, pert, seed=42)
        assert len(df_cf) == len(df)
        assert "perturbation_type" in df_cf.columns
        assert (df_cf["perturbation_type"] == pert).all()
        assert list(df_cf["complaint_id"]) == list(df["complaint_id"])


def test_generate_counterfactuals_invalid_raises():
    """Verify unknown perturbation raises ValueError."""
    df = _sample_df()
    with pytest.raises(ValueError, match="Unknown perturbation"):
        generate_counterfactuals(df, "invalid_pert")


def test_compute_flip_rate_all_same():
    """Verify flip rate is 0 when predictions are identical."""
    preds = pd.Series(["A", "B", "C"])
    result = compute_flip_rate(preds, preds)
    assert result["flip_rate"] == 0.0
    assert result["n_flips"] == 0
    assert result["n_total"] == 3


def test_compute_flip_rate_all_different():
    """Verify flip rate is 1 when all predictions differ."""
    original = pd.Series(["A", "B", "C"])
    perturbed = pd.Series(["X", "Y", "Z"])
    result = compute_flip_rate(original, perturbed)
    assert result["flip_rate"] == 1.0
    assert result["n_flips"] == 3


def test_compute_flip_rate_partial():
    """Verify flip rate is computed correctly for partial flips."""
    original = pd.Series(["A", "B", "C", "D"])
    perturbed = pd.Series(["A", "X", "C", "Y"])
    result = compute_flip_rate(original, perturbed)
    assert result["flip_rate"] == pytest.approx(0.5)
    assert result["n_flips"] == 2
