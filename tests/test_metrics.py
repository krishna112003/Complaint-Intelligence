"""Unit tests for classification metrics functions with hand-calculated controls."""

from __future__ import annotations

import numpy as np

from complaints.eval.metrics import compute_classification_metrics


def test_perfect_classification():
    """Verify metrics on a perfect prediction control."""
    y_true = ["A", "B", "C", "A"]
    y_pred = ["A", "B", "C", "A"]

    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["n_samples"] == 4


def test_hand_calculated_imbalanced_f1():
    """Verify macro-F1 against an exact hand-calculated 2-class scenario.

    Classes: [Cat, Dog]
    y_true: [Cat, Cat, Cat, Dog]
    y_pred: [Cat, Cat, Dog, Dog]

    For Cat:
      TP = 2, FP = 0, FN = 1 -> Precision = 2/2 = 1.0, Recall = 2/3, F1 = 2*(1 * 2/3)/(1 + 2/3) = (4/3)/(5/3) = 0.8
    For Dog:
      TP = 1, FP = 1, FN = 0 -> Precision = 1/2 = 0.5, Recall = 1/1 = 1.0, F1 = 2*(0.5 * 1)/(0.5 + 1) = 1.0/1.5 = 2/3 = 0.6667
    Macro-F1 = (0.8 + 0.6666667) / 2 = 1.4666667 / 2 = 0.7333333
    Accuracy = 3 / 4 = 0.75
    """
    y_true = ["Cat", "Cat", "Cat", "Dog"]
    y_pred = ["Cat", "Cat", "Dog", "Dog"]

    metrics = compute_classification_metrics(y_true, y_pred)
    assert np.isclose(metrics["accuracy"], 0.75)
    assert np.isclose(metrics["macro_f1"], 0.7333333333333334)
    assert np.isclose(metrics["per_class"]["Cat"]["f1"], 0.8)
    assert np.isclose(metrics["per_class"]["Dog"]["f1"], 2.0 / 3.0)


def test_confusion_matrix_dimensions():
    """Verify confusion matrix dimensions match number of unique classes."""
    y_true = ["A", "B", "C"]
    y_pred = ["B", "B", "C"]

    metrics = compute_classification_metrics(y_true, y_pred)
    assert len(metrics["labels"]) == 3
    conf_mat = np.array(metrics["confusion_matrix"])
    assert conf_mat.shape == (3, 3)
