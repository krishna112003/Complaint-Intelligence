"""Calibration analysis and Expected Calibration Error (ECE).

Evaluates whether predicted probabilities/confidence scores accurately represent
true empirical likelihoods of correctness.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_expected_calibration_error(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    confidences: list[float] | np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute binned Expected Calibration Error (ECE) and reliability diagram points."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    confs = np.asarray(confidences, dtype=np.float64)

    correct = (yt == yp).astype(np.float64)
    n = len(confs)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    bins_data: list[dict[str, Any]] = []

    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]

        # Samples in bin [low, high) or [low, high] for last bin
        if i == n_bins - 1:
            mask = (confs >= low) & (confs <= high)
        else:
            mask = (confs >= low) & (confs < high)

        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(correct[mask]))
            bin_conf = float(np.mean(confs[mask]))
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = float((low + high) / 2.0)

        bins_data.append(
            {
                "bin_index": i,
                "bin_range": [round(float(low), 2), round(float(high), 2)],
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
            }
        )

    return {
        "ece": round(float(ece), 4),
        "n_bins": n_bins,
        "n_samples": n,
        "bins": bins_data,
    }
