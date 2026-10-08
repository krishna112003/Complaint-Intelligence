"""Train and evaluate all baseline models.

Trains:
1. Majority Class Floor Baseline
2. Domain Keyword Rules Baseline
3. TF-IDF + Logistic Regression
4. TF-IDF + Calibrated Linear SVM
5. Sentence Embeddings + LightGBM

Saves standardized prediction Parquet files for each model in data/predictions/
and logs all experiments, parameters, and metrics to MLflow.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from complaints.config import get, get_project_root, setup_logging
from complaints.models.baselines import KeywordBaseline, MajorityBaseline
from complaints.models.embed_lgbm import train_and_evaluate_embed_lgbm
from complaints.models.tfidf_lr import train_and_evaluate_tfidf_lr
from complaints.models.tfidf_svm import train_and_evaluate_tfidf_svm
from complaints.tracking.mlflow_utils import log_metrics_and_params, start_tracked_run

logger = logging.getLogger(__name__)


def train_baselines() -> list[Path]:
    """Train floor baselines (Majority & Keyword) and save predictions."""
    root = get_project_root()
    gold_dir = root / get("data", "gold_dir", "data/gold")
    pred_dir = root / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(gold_dir / "train.parquet")
    df_val = pd.read_parquet(gold_dir / "val.parquet")
    df_test = pd.read_parquet(gold_dir / "test.parquet")

    results: list[Path] = []
    baselines = [MajorityBaseline(), KeywordBaseline()]

    for model in baselines:
        logger.info("--- Training & Evaluating %s ---", model.name)
        with start_tracked_run(run_name=model.name) as run:
            run_id = run.info.run_id
            model.fit(df_train, df_val)

            preds_df = model.predict(df_test, run_id=run_id)
            macro_f1 = float(f1_score(preds_df["y_true"], preds_df["y_pred"], average="macro"))
            acc = float(accuracy_score(preds_df["y_true"], preds_df["y_pred"]))

            logger.info("[%s] Test Macro-F1 = %.4f | Accuracy = %.4f", model.name, macro_f1, acc)

            out_path = pred_dir / f"{model.name}_test.parquet"
            preds_df.to_parquet(out_path, index=False, engine="pyarrow")
            results.append(out_path)

            log_metrics_and_params(
                params={"model_name": model.name, "family": "floor_baseline"},
                metrics={"test_macro_f1": macro_f1, "test_accuracy": acc},
                artifacts=[out_path],
            )

    return results


def train_all() -> list[Path]:
    """Train all models in sequence and log to MLflow."""
    logger.info("=== Starting Training of All Baseline Models ===")
    all_paths: list[Path] = []

    # 1. Floor baselines
    all_paths.extend(train_baselines())

    # 2. TF-IDF + Logistic Regression
    logger.info("--- Training & Evaluating TF-IDF + Logistic Regression ---")
    all_paths.append(train_and_evaluate_tfidf_lr())

    # 3. TF-IDF + Calibrated Linear SVM
    logger.info("--- Training & Evaluating TF-IDF + Calibrated Linear SVM ---")
    all_paths.append(train_and_evaluate_tfidf_svm())

    # 4. Dense Embeddings + LightGBM
    logger.info("--- Training & Evaluating Embeddings + LightGBM ---")
    all_paths.append(train_and_evaluate_embed_lgbm())

    logger.info("=== All Models Completed Successfully (%d files generated) ===", len(all_paths))
    return all_paths


if __name__ == "__main__":
    setup_logging()
    train_all()
