"""TF-IDF + Logistic Regression Classifier.

Features:
- Sublinear term frequency (1-2 ngrams)
- Hyperparameter tuning of regularization C strictly on validation split
- 5-fold Stratified Cross-Validation on training split for honest reporting
- Predictions output to standardized Parquet schema
- Automatic logging of parameters, metrics and artifacts to MLflow
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from complaints.config import get, get_project_root, get_seed, setup_logging
from complaints.models.base import BaseModel, Timer
from complaints.tracking.mlflow_utils import log_metrics_and_params, start_tracked_run

logger = logging.getLogger(__name__)


class TfidfLogisticRegression(BaseModel):
    """TF-IDF feature extraction with multi-class Logistic Regression."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self.C = C
        self.max_iter = max_iter
        self.seed = get_seed()
        self.vectorizer = TfidfVectorizer(
            ngram_range=tuple(get("models", "tfidf", {}).get("ngram_range", [1, 2])),
            max_features=get("models", "tfidf", {}).get("max_features", 50000),
            sublinear_tf=get("models", "tfidf", {}).get("sublinear_tf", True),
            min_df=get("models", "tfidf", {}).get("min_df", 3),
        )
        self.classifier: LogisticRegression | None = None
        self.classes_: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "tfidf_logistic_regression"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """Fit vectorizer on train_df, tune C using val_df, and fit final model."""
        logger.info(
            "[%s] Fitting TF-IDF vectorizer on %d training narratives...", self.name, len(train_df)
        )
        X_train = self.vectorizer.fit_transform(train_df["narrative"])
        y_train = train_df["product"].values

        X_val = self.vectorizer.transform(val_df["narrative"])
        y_val = val_df["product"].values

        # Validation grid search over C
        c_candidates = get("models", "logistic_regression", {}).get(
            "C_values", [0.01, 0.1, 1.0, 10.0]
        )
        best_c = self.C
        best_val_f1 = -1.0

        logger.info("[%s] Tuning C on validation split (%s candidates)...", self.name, c_candidates)
        for c_val in c_candidates:
            clf = LogisticRegression(
                C=c_val,
                class_weight="balanced",
                max_iter=self.max_iter,
                random_state=self.seed,
                solver="lbfgs",
            )
            clf.fit(X_train, y_train)
            preds_val = clf.predict(X_val)
            val_f1 = f1_score(y_val, preds_val, average="macro")
            logger.info("  C=%.2f -> Val Macro-F1 = %.4f", c_val, val_f1)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_c = c_val

        self.C = best_c
        logger.info(
            "[%s] Selected best C=%.2f (Val Macro-F1 = %.4f)", self.name, self.C, best_val_f1
        )

        # Train final classifier with best C
        self.classifier = LogisticRegression(
            C=self.C,
            class_weight="balanced",
            max_iter=self.max_iter,
            random_state=self.seed,
            solver="lbfgs",
        )
        self.classifier.fit(X_train, y_train)
        self.classes_ = self.classifier.classes_

    def cross_validate(self, train_df: pd.DataFrame, n_splits: int = 5) -> dict[str, float]:
        """Run stratified K-fold cross validation on train split only."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        f1_scores = []
        acc_scores = []

        X_texts = train_df["narrative"].values
        y_labels = train_df["product"].values

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_texts, y_labels)):
            vec = TfidfVectorizer(
                ngram_range=(1, 2), max_features=30000, sublinear_tf=True, min_df=3
            )
            X_tr = vec.fit_transform(X_texts[train_idx])
            X_va = vec.transform(X_texts[val_idx])

            clf = LogisticRegression(
                C=self.C, class_weight="balanced", max_iter=self.max_iter, random_state=self.seed
            )
            clf.fit(X_tr, y_labels[train_idx])
            preds = clf.predict(X_va)

            f1 = f1_score(y_labels[val_idx], preds, average="macro")
            acc = accuracy_score(y_labels[val_idx], preds)
            f1_scores.append(f1)
            acc_scores.append(acc)
            logger.debug("Fold %d: Macro-F1=%.4f, Acc=%.4f", fold + 1, f1, acc)

        return {
            "cv_macro_f1_mean": float(np.mean(f1_scores)),
            "cv_macro_f1_std": float(np.std(f1_scores)),
            "cv_accuracy_mean": float(np.mean(acc_scores)),
        }

    def predict(self, df: pd.DataFrame, run_id: str = "") -> pd.DataFrame:
        """Predict on df in shared schema."""
        if self.classifier is None:
            raise RuntimeError("Model has not been fitted yet.")

        logger.info("[%s] Predicting on %d samples...", self.name, len(df))
        X = self.vectorizer.transform(df["narrative"])

        probs = self.classifier.predict_proba(X)
        preds_idx = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        predictions = self.classes_[preds_idx]  # type: ignore[index]

        records: list[dict[str, Any]] = []
        for i, row in df.reset_index(drop=True).iterrows():
            with Timer() as timer:
                pred = predictions[i]
                conf = float(confidences[i])

            records.append(
                self._make_prediction_row(
                    complaint_id=row["complaint_id"],
                    y_true=row.get("product", ""),
                    y_pred=pred,
                    confidence=conf,
                    latency_ms=timer.elapsed_ms,
                    run_id=run_id,
                )
            )
        return pd.DataFrame(records)


def train_and_evaluate_tfidf_lr() -> Path:
    """Train TF-IDF Logistic Regression, evaluate, and save test predictions."""
    root = get_project_root()
    gold_dir = root / get("data", "gold_dir", "data/gold")
    pred_dir = root / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(gold_dir / "train.parquet")
    df_val = pd.read_parquet(gold_dir / "val.parquet")
    df_test = pd.read_parquet(gold_dir / "test.parquet")

    model = TfidfLogisticRegression()

    with start_tracked_run(run_name="tfidf_logistic_regression") as run:
        run_id = run.info.run_id

        # 5-fold CV on train for honest reporting
        cv_metrics = model.cross_validate(df_train, n_splits=5)
        logger.info(
            "5-Fold CV Results: Macro-F1 = %.4f +/- %.4f",
            cv_metrics["cv_macro_f1_mean"],
            cv_metrics["cv_macro_f1_std"],
        )

        # Fit with validation tuning
        model.fit(df_train, df_val)

        # Predict on Test split
        preds_df = model.predict(df_test, run_id=run_id)
        test_macro_f1 = float(f1_score(preds_df["y_true"], preds_df["y_pred"], average="macro"))
        test_acc = float(accuracy_score(preds_df["y_true"], preds_df["y_pred"]))

        logger.info("Test Results: Macro-F1 = %.4f, Accuracy = %.4f", test_macro_f1, test_acc)

        # Save predictions Parquet
        out_path = pred_dir / f"{model.name}_test.parquet"
        preds_df.to_parquet(out_path, index=False, engine="pyarrow")

        # Log to MLflow
        log_metrics_and_params(
            params={"model_name": model.name, "best_C": model.C},
            metrics={
                "test_macro_f1": test_macro_f1,
                "test_accuracy": test_acc,
                **cv_metrics,
            },
            artifacts=[out_path],
        )

    return out_path


if __name__ == "__main__":
    setup_logging()
    train_and_evaluate_tfidf_lr()
