"""Tests for Transformer Fine-tune module (CPU smoke test)."""

from __future__ import annotations

import pandas as pd
import pytest

from complaints.models.transformer_ft import TransformerFineTuner, _softmax


def test_transformer_instantiation():
    """Verify TransformerFineTuner creates with correct defaults."""
    model = TransformerFineTuner(smoke_test=True)
    assert "transformer_ft" in model.name
    assert "smoke" in model.name
    assert model.smoke_test is True


def test_transformer_name_no_smoke():
    """Verify name without smoke suffix when smoke_test=False."""
    model = TransformerFineTuner(smoke_test=False)
    assert "smoke" not in model.name
    assert "transformer_ft" in model.name


def test_softmax_correctness():
    """Verify softmax produces valid probability distribution."""
    import numpy as np

    logits = np.array([[2.0, 1.0, 0.1], [1.0, 2.0, 3.0]])
    probs = _softmax(logits)
    # Each row sums to 1
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-6)
    # All values in [0, 1]
    assert (probs >= 0).all()
    assert (probs <= 1).all()
    # Argmax preserved
    assert np.argmax(probs[0]) == 0
    assert np.argmax(probs[1]) == 2


def test_transformer_predict_without_fit_raises():
    """Verify predict raises RuntimeError when model hasn't been fit."""
    model = TransformerFineTuner(smoke_test=True)
    dummy_df = pd.DataFrame({"complaint_id": ["1"], "narrative": ["test"], "product": ["Other"]})
    with pytest.raises(RuntimeError, match="not trained"):
        model.predict(dummy_df)
