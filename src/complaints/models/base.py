"""Abstract base class for all models.

Every model (classical, embedding, transformer, LLM) implements this interface
so that evaluation code can treat them uniformly. All predictions are written
to the shared Parquet schema.
"""

from __future__ import annotations

import abc
import time
from typing import Any

import pandas as pd

# Shared prediction schema — every model must produce exactly these columns.
PREDICTION_COLUMNS = [
    "complaint_id",
    "y_true",
    "y_pred",
    "confidence",
    "latency_ms",
    "cost_usd",
    "input_tokens",
    "output_tokens",
    "model_name",
    "run_id",
]


class BaseModel(abc.ABC):
    """Abstract interface that every model must implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable model name for logging and reports."""

    @abc.abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """Train the model on train_df, using val_df for early stopping / tuning."""

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame, run_id: str = "") -> pd.DataFrame:
        """Predict on df and return a DataFrame with exactly PREDICTION_COLUMNS.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least ``complaint_id``, ``narrative``, and ``label`` columns.
        run_id : str
            Identifier for this evaluation run (logged in the output).

        Returns
        -------
        pd.DataFrame
            Predictions in the shared schema.
        """

    def _make_prediction_row(
        self,
        complaint_id: Any,
        y_true: str,
        y_pred: str,
        confidence: float,
        latency_ms: float,
        run_id: str,
        cost_usd: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> dict[str, Any]:
        """Build a single prediction row conforming to the shared schema."""
        return {
            "complaint_id": complaint_id,
            "y_true": y_true,
            "y_pred": y_pred,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model_name": self.name,
            "run_id": run_id,
        }


class Timer:
    """Simple context manager for measuring wall-clock latency in milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
