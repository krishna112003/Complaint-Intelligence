"""Cost and Latency Profiling and Pareto Frontier Analysis.

Calculates:
- Cost per 1,000 inferences in USD
- Latency percentiles (p50 median, p95 95th percentile, mean)
- Pareto frontier across the cost-versus-macro-F1 trade-off
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_cost_latency_metrics(df_predictions: pd.DataFrame) -> dict[str, float]:
    """Calculate cost and latency profiling statistics for a model's prediction run."""
    n = len(df_predictions)
    if n == 0:
        return {
            "cost_per_1k": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_mean_ms": 0.0,
        }

    total_cost = float(df_predictions["cost_usd"].sum())
    cost_per_1k = float((total_cost / n) * 1000.0)

    latencies = df_predictions["latency_ms"].values
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    mean_lat = float(np.mean(latencies))

    return {
        "cost_per_1k": round(cost_per_1k, 4),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_mean_ms": round(mean_lat, 2),
    }


def compute_pareto_frontier(models_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Determine models on the non-dominated Pareto frontier (minimizing cost, maximizing Macro-F1).

    A model A dominates model B if:
    - cost(A) <= cost(B) and macro_f1(A) >= macro_f1(B)
    - with at least one strict inequality.
    """
    pareto_models: list[dict[str, Any]] = []

    for i, a in enumerate(models_summary):
        is_dominated = False
        cost_a = a["cost_per_1k"]
        f1_a = a["macro_f1"]

        for j, b in enumerate(models_summary):
            if i == j:
                continue
            cost_b = b["cost_per_1k"]
            f1_b = b["macro_f1"]

            # b dominates a if b is cheaper or equal AND b has higher or equal F1, with at least one strictly better
            if (cost_b <= cost_a and f1_b >= f1_a) and (cost_b < cost_a or f1_b > f1_a):
                is_dominated = True
                break

        a_copy = dict(a)
        a_copy["is_pareto"] = not is_dominated
        pareto_models.append(a_copy)

    return pareto_models
