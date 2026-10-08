"""Statistical significance testing for model comparison.

Implements:
1. McNemar's test for paired classification accuracy (exact binomial when discordant < 25)
2. Wilcoxon signed-rank test for paired continuous/ordinal shifts (e.g. urgency shifts)
3. Holm-Bonferroni step-down correction for multiple pairwise comparisons
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def mcnemar_test(
    y_true: list[str] | np.ndarray,
    y_pred_a: list[str] | np.ndarray,
    y_pred_b: list[str] | np.ndarray,
) -> dict[str, Any]:
    """Perform paired McNemar's test between Model A and Model B on matching rows.

    Contingency table on correct/incorrect classifications:
                Model B Correct   Model B Incorrect
    Model A Correct        a (n00)           b (n01)  -> A correct, B wrong
    Model A Incorrect      c (n10)           d (n11)  -> A wrong, B correct

    Discordant pairs: b (A correct, B wrong), c (A wrong, B correct)
    Uses exact binomial test when (b + c) < 25; otherwise chi-squared with Edwards continuity correction.
    """
    yt = np.asarray(y_true)
    ya = np.asarray(y_pred_a)
    yb = np.asarray(y_pred_b)

    correct_a = ya == yt
    correct_b = yb == yt

    n00 = int(np.sum(correct_a & correct_b))
    b = int(np.sum(correct_a & (~correct_b)))  # A right, B wrong
    c = int(np.sum((~correct_a) & correct_b))  # A wrong, B right
    n11 = int(np.sum((~correct_a) & (~correct_b)))

    discordant = b + c
    if discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "exact_binomial",
            "contingency_table": {"both_correct": n00, "a_only": b, "b_only": c, "both_wrong": n11},
            "significant_at_05": False,
        }

    if discordant < 25:
        # Exact binomial test: under H0: p = 0.5 for discordant events
        # Two-sided p-value
        p_val = float(stats.binomtest(k=b, n=discordant, p=0.5).pvalue)
        statistic = float(min(b, c))
        method = "exact_binomial"
    else:
        # Asymptotic Chi-Square with continuity correction: (|b - c| - 1)^2 / (b + c)
        chi2_stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        p_val = float(stats.chi2.sf(chi2_stat, df=1))
        statistic = chi2_stat
        method = "chi_squared_edwards"

    return {
        "statistic": round(statistic, 4),
        "p_value": float(p_val),
        "method": method,
        "contingency_table": {
            "both_correct": n00,
            "a_only": b,
            "b_only": c,
            "both_wrong": n11,
        },
        "significant_at_05": bool(p_val < 0.05),
    }


def wilcoxon_test(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Perform Wilcoxon signed-rank test for paired ordinal or continuous outputs."""
    sa = np.asarray(scores_a, dtype=np.float64)
    sb = np.asarray(scores_b, dtype=np.float64)
    diffs = sa - sb
    non_zero_diffs = diffs[diffs != 0]

    if len(non_zero_diffs) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant_at_05": False}

    res = stats.wilcoxon(sa, sb, zero_method="pratt")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "significant_at_05": bool(res.pvalue < 0.05),
    }


def holm_bonferroni_correction(p_values: list[float]) -> list[dict[str, Any]]:
    """Apply Holm-Bonferroni step-down procedure to adjust multiple p-values."""
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values with original indices
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        multiplier = m - rank
        p_adj = min(1.0, p * multiplier)
        # Ensure monotonicity in step-down
        cum_max = max(cum_max, p_adj)
        adjusted[orig_idx] = cum_max

    return [
        {
            "original_p": float(p_values[i]),
            "adjusted_p": float(adjusted[i]),
            "significant_at_05": bool(adjusted[i] < 0.05),
        }
        for i in range(m)
    ]
