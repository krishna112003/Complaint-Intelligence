"""Floor baseline classifiers: Majority Class and Keyword Rules.

These simple heuristic models set the performance floor, proving the task
is non-trivial before advancing to statistical and deep learning models.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, ClassVar

import pandas as pd

from complaints.models.base import BaseModel, Timer

logger = logging.getLogger(__name__)


class MajorityBaseline(BaseModel):
    """Predicts the most frequent class in the training split."""

    def __init__(self) -> None:
        self.majority_label: str = ""
        self.class_prior: float = 0.0

    @property
    def name(self) -> str:
        return "majority_baseline"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        counts = Counter(train_df["product"])
        self.majority_label = counts.most_common(1)[0][0]
        self.class_prior = counts.most_common(1)[0][1] / len(train_df)
        logger.info(
            "Fitted %s: majority class = '%s' (prior = %.4f)",
            self.name,
            self.majority_label,
            self.class_prior,
        )

    def predict(self, df: pd.DataFrame, run_id: str = "") -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            with Timer() as timer:
                y_pred = self.majority_label
                confidence = self.class_prior

            records.append(
                self._make_prediction_row(
                    complaint_id=row["complaint_id"],
                    y_true=row.get("product", ""),
                    y_pred=y_pred,
                    confidence=confidence,
                    latency_ms=timer.elapsed_ms,
                    run_id=run_id,
                )
            )
        return pd.DataFrame(records)


class KeywordBaseline(BaseModel):
    """Heuristic keyword matching baseline with fallback to majority class."""

    # Hand-curated financial domain keyword rules
    KEYWORD_RULES: ClassVar[dict[str, list[str]]] = {
        "Credit Reporting": [
            "credit report",
            "equifax",
            "experian",
            "transunion",
            "credit bureau",
            "inaccurate report",
        ],
        "Debt Collection": [
            "collection agency",
            "debt collector",
            "debt",
            "collector",
            "owe",
            "harass",
        ],
        "Mortgage": [
            "mortgage",
            "foreclosure",
            "escrow",
            "loan modification",
            "home loan",
            "refinance",
        ],
        "Credit Card": [
            "credit card",
            "cardholder",
            "annual fee",
            "apr",
            "card statement",
            "late fee on card",
        ],
        "Bank Account": [
            "checking account",
            "savings account",
            "overdraft",
            "atm",
            "deposit",
            "bank branch",
            "wire transfer",
        ],
        "Student Loan": [
            "student loan",
            "navient",
            "sallie mae",
            "tuition",
            "fafsa",
            "student debt",
        ],
        "Vehicle Loan/Lease": [
            "car loan",
            "auto loan",
            "vehicle",
            "car lease",
            "repossession",
            "auto lease",
        ],
        "Money Transfer": [
            "money transfer",
            "western union",
            "moneygram",
            "paypal",
            "wire",
            "remittance",
        ],
        "Personal Loan": [
            "payday loan",
            "personal loan",
            "installment loan",
            "title loan",
            "cash advance",
        ],
    }

    def __init__(self) -> None:
        self.fallback_label: str = "Credit Reporting"

    @property
    def name(self) -> str:
        return "keyword_baseline"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        counts = Counter(train_df["product"])
        self.fallback_label = counts.most_common(1)[0][0]
        logger.info("Fitted %s: fallback label = '%s'", self.name, self.fallback_label)

    def predict(self, df: pd.DataFrame, run_id: str = "") -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            with Timer() as timer:
                text = str(row.get("narrative", "")).lower()
                matched_label = None

                for label, keywords in self.KEYWORD_RULES.items():
                    if any(kw in text for kw in keywords):
                        matched_label = label
                        break

                if matched_label:
                    y_pred = matched_label
                    confidence = 0.80
                else:
                    y_pred = self.fallback_label
                    confidence = 0.20

            records.append(
                self._make_prediction_row(
                    complaint_id=row["complaint_id"],
                    y_true=row.get("product", ""),
                    y_pred=y_pred,
                    confidence=confidence,
                    latency_ms=timer.elapsed_ms,
                    run_id=run_id,
                )
            )
        return pd.DataFrame(records)
