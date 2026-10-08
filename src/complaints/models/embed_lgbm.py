"""Dense Embedding + LightGBM Classifier with Disk Caching.

Features:
- Sentence Transformers (all-MiniLM-L6-v2) embedding extraction
- Robust disk caching keyed by SHA-256 hash of narrative text to eliminate redundant compute
- Multi-class LightGBM GBDT with balanced class weights and early stopping
- Prediction probabilities for calibrated confidence scoring
- MLflow experiment tracking integration
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from complaints.config import get, get_project_root, get_seed, setup_logging
from complaints.models.base import BaseModel, Timer
from complaints.tracking.mlflow_utils import log_metrics_and_params, start_tracked_run

logger = logging.getLogger(__name__)


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 hash of text for caching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class EmbeddingCache:
    """On-disk cache for dense sentence embeddings saved as NumPy arrays."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_or_compute(
        self,
        texts: list[str],
        encoder: SentenceTransformer,
        batch_size: int = 256,
        cache_name: str = "embeddings",
    ) -> np.ndarray:
        """Load cached embeddings if available; otherwise compute and persist."""
        cache_file = self.cache_dir / f"{cache_name}_{len(texts)}.npy"
        if cache_file.exists():
            logger.info("Loading cached embeddings from %s...", cache_file)
            return np.asarray(np.load(cache_file), dtype=np.float32)

        logger.info("Computing dense embeddings for %d texts using %s...", len(texts), encoder)
        embeddings = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        np.save(cache_file, embeddings)
        logger.info("Persisted embeddings to disk cache: %s (%s)", cache_file, embeddings.shape)
        return embeddings


class EmbeddingLightGBM(BaseModel):
    """Dense embeddings with LightGBM GBDT."""

    def __init__(self) -> None:
        self.seed = get_seed()
        self.model_name_str = get("models", "embeddings", {}).get("model_name", "all-MiniLM-L6-v2")
        self.batch_size = get("models", "embeddings", {}).get("batch_size", 256)
        root = get_project_root()
        cache_dir = root / get("models", "embeddings", {}).get("cache_dir", "data/cache/embeddings")
        self.cache = EmbeddingCache(cache_dir)
        self.encoder = SentenceTransformer(self.model_name_str)
        self.label_encoder = LabelEncoder()
        self.clf: lgb.LGBMClassifier | None = None

    @property
    def name(self) -> str:
        return "embed_lightgbm"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """Extract or load cached embeddings and fit LightGBM with early stopping."""
        X_train = self.cache.get_or_compute(
            train_df["narrative"].tolist(),
            self.encoder,
            batch_size=self.batch_size,
            cache_name="train",
        )
        y_train = self.label_encoder.fit_transform(train_df["product"])

        X_val = self.cache.get_or_compute(
            val_df["narrative"].tolist(),
            self.encoder,
            batch_size=self.batch_size,
            cache_name="val",
        )
        y_val = self.label_encoder.transform(val_df["product"])

        lgb_cfg = get("models", "lightgbm", {})
        n_estimators = lgb_cfg.get("n_estimators", 300)
        learning_rate = lgb_cfg.get("learning_rate", 0.05)
        num_leaves = lgb_cfg.get("num_leaves", 31)

        logger.info("[%s] Training LightGBM on %d embeddings...", self.name, len(X_train))
        self.clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=-1,
            verbosity=-1,
        )

        self.clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        preds_val = self.clf.predict(X_val)
        val_macro_f1 = f1_score(y_val, preds_val, average="macro")
        logger.info("[%s] Validation Macro-F1 = %.4f", self.name, val_macro_f1)

    def predict(self, df: pd.DataFrame, run_id: str = "") -> pd.DataFrame:
        """Predict on df using cached embeddings and LightGBM."""
        if self.clf is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self.cache.get_or_compute(
            df["narrative"].tolist(),
            self.encoder,
            batch_size=self.batch_size,
            cache_name=f"predict_{len(df)}",
        )

        probs = self.clf.predict_proba(X)
        preds_idx = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        predictions = self.label_encoder.inverse_transform(preds_idx)

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


def train_and_evaluate_embed_lgbm() -> Path:
    """Train Embedding + LightGBM model, evaluate on test, and save predictions."""
    root = get_project_root()
    gold_dir = root / get("data", "gold_dir", "data/gold")
    pred_dir = root / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(gold_dir / "train.parquet")
    df_val = pd.read_parquet(gold_dir / "val.parquet")
    df_test = pd.read_parquet(gold_dir / "test.parquet")

    model = EmbeddingLightGBM()

    with start_tracked_run(run_name="embed_lightgbm") as run:
        run_id = run.info.run_id

        # Fit model
        model.fit(df_train, df_val)

        # Predict on Test split
        preds_df = model.predict(df_test, run_id=run_id)
        test_macro_f1 = float(f1_score(preds_df["y_true"], preds_df["y_pred"], average="macro"))
        test_acc = float(accuracy_score(preds_df["y_true"], preds_df["y_pred"]))

        logger.info("Test Results: Macro-F1 = %.4f, Accuracy = %.4f", test_macro_f1, test_acc)

        out_path = pred_dir / f"{model.name}_test.parquet"
        preds_df.to_parquet(out_path, index=False, engine="pyarrow")

        log_metrics_and_params(
            params={"model_name": model.name, "embedding_model": model.model_name_str},
            metrics={"test_macro_f1": test_macro_f1, "test_accuracy": test_acc},
            artifacts=[out_path],
        )

    return out_path


if __name__ == "__main__":
    setup_logging()
    train_and_evaluate_embed_lgbm()
