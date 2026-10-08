"""Percentile Bootstrap and Paired Bootstrap Confidence Interval Estimation.

Features:
- Percentile bootstrap confidence intervals for macro-F1, accuracy, and arbitrary metrics
- Paired bootstrap difference estimation (Model A minus Model B) on matched rows
- Deterministic random state management via project seed
- Support for arbitrary alpha confidence levels (default 95%)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from complaints.config import get_seed

logger = logging.getLogger(__name__)


def compute_bootstrap_ci(
    y_true: list[str] | np.ndarray | pd.Series,
    y_pred: list[str] | np.ndarray | pd.Series,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    """Compute empirical percentile bootstrap confidence interval for a metric."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n = len(y_true_arr)

    if metric_fn is None:
        metric_fn = lambda yt, yp: float(f1_score(yt, yp, average="macro", zero_division=0))  # noqa: E731

    rng = np.random.RandomState(seed if seed is not None else get_seed())
    point_estimate = float(metric_fn(y_true_arr, y_pred_arr))

    boot_scores = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        boot_scores[i] = metric_fn(y_true_arr[indices], y_pred_arr[indices])

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_scores, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_scores, 100 * (1.0 - alpha / 2.0)))
    std_err = float(np.std(boot_scores))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_err": std_err,
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
    }


def compute_paired_bootstrap_difference(
    y_true: list[str] | np.ndarray | pd.Series,
    y_pred_a: list[str] | np.ndarray | pd.Series,
    y_pred_b: list[str] | np.ndarray | pd.Series,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute paired bootstrap difference (Metric_A - Metric_B) on matching items."""
    yt = np.asarray(y_true)
    ya = np.asarray(y_pred_a)
    yb = np.asarray(y_pred_b)
    n = len(yt)

    if metric_fn is None:
        metric_fn = lambda t, p: float(f1_score(t, p, average="macro", zero_division=0))  # noqa: E731

    rng = np.random.RandomState(seed if seed is not None else get_seed())
    diff_point = float(metric_fn(yt, ya) - metric_fn(yt, yb))

    boot_diffs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        yt_boot = yt[indices]
        score_a = metric_fn(yt_boot, ya[indices])
        score_b = metric_fn(yt_boot, yb[indices])
        boot_diffs[i] = score_a - score_b

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_diffs, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_diffs, 100 * (1.0 - alpha / 2.0)))
    excludes_zero = bool(ci_lower > 0.0 or ci_upper < 0.0)

    # Empirical two-tailed p-value: proportion of bootstrap differences on other side of 0
    p_val = float(2.0 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0)))
    p_val = min(1.0, max(0.0, p_val))

    return {
        "diff_point": diff_point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "excludes_zero": excludes_zero,
        "empirical_p_value": p_val,
        "std_err": float(np.std(boot_diffs)),
    }
