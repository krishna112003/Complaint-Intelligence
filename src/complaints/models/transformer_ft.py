"""Transformer Fine-tuning for complaint classification (CPU smoke test).

Fine-tunes a pre-trained DistilBERT-base-uncased on the complaint product
classification task. Designed for a CPU smoke-test with a tiny subset;
set ``smoke_test=True`` (default) to limit to 50 train + 1 epoch.

Saves predictions in the standard PREDICTION_COLUMNS schema.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from complaints.config import get, get_project_root, get_seed, setup_logging
from complaints.models.base import BaseModel
from complaints.tracking.mlflow_utils import log_metrics_and_params, start_tracked_run

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "distilbert-base-uncased"
SMOKE_TRAIN_SAMPLES = 50
SMOKE_EVAL_SAMPLES = 20
SMOKE_EPOCHS = 1
MAX_LENGTH = 128


class TransformerFineTuner(BaseModel):
    """DistilBERT fine-tune for complaint product classification."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        smoke_test: bool = True,
        output_dir: Path | None = None,
    ) -> None:
        self._model_name = model_name
        self.smoke_test = smoke_test
        self.seed = get_seed()
        root = get_project_root()
        self.output_dir = output_dir or root / "models" / "transformer_ft"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.le: LabelEncoder | None = None
        self.trainer: Any = None

    @property
    def name(self) -> str:
        suffix = "_smoke" if self.smoke_test else ""
        return f"transformer_ft_{self._model_name.split('/')[-1]}{suffix}"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """Fine-tune transformer on train_df with val_df for evaluation."""
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        # Subsample for smoke test
        if self.smoke_test:
            train_df = train_df.head(SMOKE_TRAIN_SAMPLES)
            val_df = val_df.head(SMOKE_EVAL_SAMPLES)

        # Encode labels
        self.le = LabelEncoder()
        self.le.fit(train_df["product"])
        num_labels = len(self.le.classes_)

        train_labels = self.le.transform(train_df["product"]).tolist()
        val_labels = self.le.transform(val_df["product"]).tolist()

        # Prepare HuggingFace datasets
        train_ds = Dataset.from_dict(
            {"text": train_df["narrative"].tolist(), "label": train_labels}
        )
        val_ds = Dataset.from_dict({"text": val_df["narrative"].tolist(), "label": val_labels})

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        def tokenize_fn(batch: dict[str, list[str]]) -> dict[str, Any]:
            res = tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            return cast(dict[str, Any], res)

        train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name,
            num_labels=num_labels,
        )

        epochs = SMOKE_EPOCHS if self.smoke_test else 3
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=10,
            seed=self.seed,
            no_cuda=True,  # CPU-only
            report_to="none",
            disable_tqdm=False,
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
        )

        logger.info(
            "Starting transformer fine-tune: model=%s, train=%d, val=%d, epochs=%d",
            self._model_name,
            len(train_ds),
            len(val_ds),
            epochs,
        )
        self.trainer.train()
        logger.info("Transformer fine-tuning complete.")

    def predict(self, df: pd.DataFrame, run_id: str = "") -> pd.DataFrame:
        """Generate predictions using the fine-tuned transformer."""
        from datasets import Dataset

        if self.trainer is None or self.le is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if self.smoke_test:
            df = df.head(SMOKE_EVAL_SAMPLES)

        ds = Dataset.from_dict({"text": df["narrative"].tolist()})

        tokenizer = self.trainer.tokenizer

        def tokenize_fn(batch: dict[str, list[str]]) -> dict[str, Any]:
            res = tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            return cast(dict[str, Any], res)

        ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

        rows: list[dict[str, Any]] = []
        output = self.trainer.predict(ds)
        logits = output.predictions
        probs = _softmax(logits)
        pred_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        pred_labels = self.le.inverse_transform(pred_indices)

        for i, (_, row) in enumerate(df.head(len(pred_labels)).iterrows()):
            rows.append(
                self._make_prediction_row(
                    complaint_id=row["complaint_id"],
                    y_true=str(row["product"]),
                    y_pred=str(pred_labels[i]),
                    confidence=float(confidences[i]),
                    latency_ms=0.0,
                    run_id=run_id,
                )
            )

        return pd.DataFrame(rows)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    res = e_x / e_x.sum(axis=1, keepdims=True)
    return cast(np.ndarray, res)


def train_and_evaluate_transformer_ft(smoke_test: bool = True) -> Path:
    """End-to-end transformer fine-tune, predict, and persist."""
    from sklearn.metrics import accuracy_score, f1_score

    root = get_project_root()
    gold_dir = root / get("data", "gold_dir", "data/gold")
    pred_dir = root / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached predictions first
    model = TransformerFineTuner(smoke_test=smoke_test)
    out_path = pred_dir / f"{model.name}_test.parquet"
    if out_path.exists():
        logger.info("Cached predictions found at %s — skipping retrain.", out_path)
        return out_path

    df_train = pd.read_parquet(gold_dir / "train.parquet")
    df_val = pd.read_parquet(gold_dir / "val.parquet")
    df_test = pd.read_parquet(gold_dir / "test.parquet")

    with start_tracked_run(run_name=model.name) as run:
        t0 = time.perf_counter()
        model.fit(df_train, df_val)
        train_time = time.perf_counter() - t0

        preds_df = model.predict(df_test, run_id=run.info.run_id)
        macro_f1 = float(
            f1_score(preds_df["y_true"], preds_df["y_pred"], average="macro", zero_division=0)
        )
        acc = float(accuracy_score(preds_df["y_true"], preds_df["y_pred"]))

        logger.info(
            "[%s] Test Macro-F1 = %.4f | Accuracy = %.4f | Train Time = %.1fs",
            model.name,
            macro_f1,
            acc,
            train_time,
        )

        preds_df.to_parquet(out_path, index=False, engine="pyarrow")

        log_metrics_and_params(
            params={
                "model_name": model.name,
                "family": "transformer",
                "smoke_test": str(smoke_test),
                "epochs": SMOKE_EPOCHS if smoke_test else 3,
            },
            metrics={
                "test_macro_f1": macro_f1,
                "test_accuracy": acc,
                "train_time_s": train_time,
            },
            artifacts=[out_path],
        )

    return out_path


if __name__ == "__main__":
    setup_logging()
    train_and_evaluate_transformer_ft(smoke_test=True)
