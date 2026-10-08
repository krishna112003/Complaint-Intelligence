"""Core evaluation metrics for multi-class classification and extraction.

Computes:
- Macro-F1 (primary metric), Weighted-F1, Accuracy
- Per-class Precision, Recall, F1, and Support
- Confusion Matrix
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: list[str] | np.ndarray | pd.Series,
    y_pred: list[str] | np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Compute global classification metrics and per-class performance breakdown."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    labels = sorted(set(list(y_true_arr) + list(y_pred_arr)))
    report = classification_report(
        y_true_arr, y_pred_arr, labels=labels, output_dict=True, zero_division=0
    )
    conf_mat = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    per_class = {
        cls_name: {
            "precision": float(report[cls_name]["precision"]),
            "recall": float(report[cls_name]["recall"]),
            "f1": float(report[cls_name]["f1-score"]),
            "support": int(report[cls_name]["support"]),
        }
        for cls_name in labels
        if cls_name in report
    }

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": conf_mat.tolist(),
        "n_samples": len(y_true_arr),
    }
